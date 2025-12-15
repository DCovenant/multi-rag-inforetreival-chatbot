#!/usr/bin/env python3
"""
Embedding & Indexing Script (Step 2 of 2)

This script reads OCR output from JSON, computes embeddings using PyTorch/sentence-transformers,
and indexes the results to Elasticsearch.

Run this OUTSIDE the PaddlePaddle container (in your local venv with PyTorch):
  source .venv/bin/activate
  python rag-indexer-ppocrv5-embed-separated.py pdfs_ocr_output.json

Prerequisites:
  - Run the OCR script first to generate the JSON file
  - Elasticsearch running at localhost:9200
  - PyTorch and sentence-transformers installed
"""
from __future__ import annotations

import sys
import os
import argparse
import json
import logging
import time
from datetime import datetime
from typing import List, Dict

# Fast JSON (optional - 3-5x faster loading for large files)
try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    orjson = None
    HAS_ORJSON = False

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from elasticsearch import Elasticsearch, exceptions as es_exceptions
    from elasticsearch.helpers import bulk
    ES_AVAILABLE = True
except ImportError:
    ES_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_indexer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIG DEFAULTS ---
ES_URL = 'http://localhost:9200'

# Embedding model (SOTA performance)
EMBED_MODEL = "BAAI/bge-large-en-v1.5"  # 1024 dims
EMBED_DIMS = 1024

# Indexing params
BULK_SIZE = 500
BATCH_EMBED = 64


def get_index_name(folder_path, explicit_name=None):
    if explicit_name:
        return explicit_name
    folder_name = os.path.basename(os.path.normpath(folder_path))
    cleaned = folder_name.lower().replace(' ', '_').replace('-', '_')
    if cleaned.startswith(('_', '-', '+')):
        cleaned = 'idx_' + cleaned
    return cleaned


def create_rag_index(es: Elasticsearch, index_name: str, dims: int = EMBED_DIMS):
    """Create optimized ES index with tuned BM25, HNSW, and compression settings"""
    if es.indices.exists(index=index_name):
        logger.info(f"Index {index_name} already exists.")
        return

    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "30s",
            "codec": "best_compression",  # ~30% smaller index on disk
            "index": {
                "similarity": {
                    "custom_bm25": {
                        "type": "BM25",
                        "k1": 1.2,
                        "b": 0.75
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "file_name": {"type": "keyword"},
                "page_number": {"type": "short"},  # Optimized: short instead of integer
                "chunk_index": {"type": "short"},  # Optimized: short instead of integer
                "chunk_id": {"type": "keyword"},
                "chunk_text": {
                    "type": "text",
                    "similarity": "custom_bm25",
                    "analyzer": "standard"
                },
                "word_count": {"type": "short"},  # Optimized: short instead of integer
                "source_folder": {"type": "keyword"},
                "doc_title": {"type": "text"},
                "text_source": {"type": "keyword"},
                "created_date": {"type": "date"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                    "index_options": {
                        "type": "hnsw",
                        "m": 16,
                        "ef_construction": 100
                    }
                }
            }
        }
    }
    es.indices.create(index=index_name, body=mapping)
    logger.info(f"✓ Created index: {index_name} with {dims} dimensions (compressed)")


# -------------------------
# GPU-Optimized Embedding
# -------------------------
_embedding_model_cache = {}

def get_embedding_model(model_name: str, use_fp16: bool = True):
    """Load embedding model with GPU optimization and caching"""
    global _embedding_model_cache
    
    if model_name in _embedding_model_cache:
        return _embedding_model_cache[model_name]
    
    model = SentenceTransformer(model_name)
    
    if torch.cuda.is_available():
        model = model.to('cuda')
        if use_fp16:
            model.half()
            logger.info(f"✓ Using GPU with FP16 for embeddings")
        else:
            logger.info(f"✓ Using GPU with FP32 for embeddings")
    else:
        logger.info("Using CPU for embeddings (slower)")
    
    _embedding_model_cache[model_name] = model
    return model


def index_chunks(es: Elasticsearch, index_name: str, chunks: List[Dict],
                 embed_model_name: str, batch_embed: int = BATCH_EMBED,
                 bulk_size: int = BULK_SIZE):
    """Compute embeddings with GPU optimization and bulk index"""
    model = get_embedding_model(embed_model_name)
    
    if torch.cuda.is_available():
        batch_embed = min(batch_embed * 2, 128)
        logger.info(f"✓ GPU detected: using batch size {batch_embed}")
    
    docs_to_index = []
    texts = [c["chunk_text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    
    logger.info(f"Embedding {len(texts)} chunks...")
    
    for i in tqdm(range(0, len(texts), batch_embed), desc="Embedding chunks"):
        batch_texts = texts[i:i+batch_embed]
        batch_ids = ids[i:i+batch_embed]
        
        embeddings = model.encode(
            batch_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=batch_embed,
            normalize_embeddings=True
        )
        
        for j, emb in enumerate(embeddings):
            idx_global = i + j
            chunk_data = chunks[idx_global].copy()
            chunk_data["embedding"] = emb.tolist()
            
            doc = {
                "_index": index_name,
                "_id": batch_ids[j],
                "_source": chunk_data
            }
            docs_to_index.append(doc)
        
        if len(docs_to_index) >= bulk_size:
            success, failed = bulk(
                es.options(request_timeout=120),
                docs_to_index,
                raise_on_error=False
            )
            logger.info(f"  Bulk indexed: success={success}, failed={len(failed) if isinstance(failed, list) else failed}")
            docs_to_index = []
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    if docs_to_index:
        success, failed = bulk(
            es.options(request_timeout=120),
            docs_to_index,
            raise_on_error=False
        )
        logger.info(f"  Final bulk: success={success}, failed={len(failed) if isinstance(failed, list) else failed}")


def run_indexing(es: Elasticsearch, index_name: str, chunks: List[Dict], args):
    """Index chunks with incremental support"""
    
    # Check which chunks already exist (incremental indexing)
    ids = [c["chunk_id"] for c in chunks]
    to_add = []
    
    logger.info("Checking for existing chunks...")
    for i in range(0, len(ids), 200):
        batch_ids = ids[i:i+200]
        batch_chunks = chunks[i:i+200]
        try:
            res = es.mget(index=index_name, body={"ids": batch_ids})
            for k, docres in enumerate(res.get("docs", [])):
                if not docres.get("found", False):
                    to_add.append(batch_chunks[k])
        except es_exceptions.NotFoundError:
            to_add.extend(batch_chunks)
    
    logger.info(f"New chunks to index: {len(to_add)} (skipping {len(chunks) - len(to_add)} existing)")
    
    if to_add:
        index_chunks(es, index_name, to_add, args.model,
                    batch_embed=args.batch_embed, bulk_size=args.bulk_size)
        es.indices.refresh(index=index_name)
        logger.info("✓ Indexing complete!")
    else:
        logger.info("✓ No new chunks to add. Index is up to date.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Embedding & Indexing (Step 2) - Computes embeddings and indexes to ES")
    
    parser.add_argument("json_file", help="OCR output JSON file from step 1")
    parser.add_argument("--reset", action="store_true", help="Delete all user indices before indexing")
    parser.add_argument("--es-url", type=str, default=ES_URL, help="Elasticsearch URL")
    parser.add_argument("--model", type=str, default=EMBED_MODEL, help="Embedding model")
    parser.add_argument("--batch-embed", type=int, default=BATCH_EMBED, help="Embedding batch size")
    parser.add_argument("--bulk-size", type=int, default=BULK_SIZE, help="ES bulk indexing size")
    parser.add_argument("--index-name", type=str, default=None, help="Explicit ES index name")
    
    args = parser.parse_args()
    
    if not ES_AVAILABLE:
        parser.error("Elasticsearch package not installed. Run: pip install elasticsearch")
    
    return args


def main():
    args = parse_arguments()
    start_time = time.time()
    
    # Load OCR output
    if not os.path.isfile(args.json_file):
        logger.error(f"JSON file not found: {args.json_file}")
        sys.exit(1)
    
    # Use orjson if available (3-5x faster for large files)
    logger.info(f"Loading OCR output from: {args.json_file} (orjson={'ON' if HAS_ORJSON else 'OFF'})")
    load_start = time.time()
    
    if HAS_ORJSON:
        with open(args.json_file, 'rb') as f:
            data = orjson.loads(f.read())
    else:
        with open(args.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    load_time = time.time() - load_start
    logger.info(f"✓ JSON loaded in {load_time:.2f}s")
    
    folder_path = data.get("folder_path", "unknown")
    chunks = data.get("chunks", [])
    
    if not chunks:
        logger.error("No chunks found in JSON file.")
        sys.exit(1)
    
    logger.info(f"Loaded {len(chunks)} chunks from {folder_path}")
    
    # Determine index name
    index_name = get_index_name(folder_path, args.index_name)
    logger.info(f"Using index: {index_name}")
    
    # Connect to ES
    es = Elasticsearch(args.es_url, request_timeout=60)
    try:
        info = es.info()
        logger.info(f"✓ Connected to Elasticsearch: {info['version']['number']}")
    except Exception as e:
        logger.error(f"Failed to connect to Elasticsearch: {e}")
        sys.exit(1)
    
    # Reset if requested
    if args.reset:
        logger.warning("Deleting user indices...")
        try:
            all_indices = es.indices.get_alias(index="*")
            user_indices = [idx for idx in all_indices.keys() if not idx.startswith(".")]
            for idx in user_indices:
                es.indices.delete(index=idx)
                logger.info(f"  Deleted index: {idx}")
        except Exception as e:
            logger.error(f"Failed to delete indices: {e}")
    
    # Determine dimensions based on model
    dims = EMBED_DIMS
    if args.model == "BAAI/bge-small-en-v1.5":
        dims = 384
    elif args.model == "sentence-transformers/all-mpnet-base-v2":
        dims = 768
    
    # Create index
    create_rag_index(es, index_name, dims=dims)
    
    # Index chunks
    logger.info("Starting embedding and indexing pipeline...")
    embed_start = time.time()
    run_indexing(es, index_name, chunks, args)
    embed_time = time.time() - embed_start
    
    total_time = time.time() - start_time
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Indexing complete for: {index_name}")
    logger.info(f"✓ Total chunks indexed: {len(chunks)}")
    logger.info(f"✓ Embedding + indexing time: {embed_time:.1f}s")
    logger.info(f"✓ Total pipeline time: {total_time:.1f}s")
    logger.info(f"✓ You can now query using: rag_queries.py")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
