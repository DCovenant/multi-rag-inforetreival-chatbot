#!/usr/bin/env python3
"""
Advanced RAG PDF Indexer - Indexing Pipeline Only

Features:
- Semantic chunking (respects sentence boundaries)
- BGE embeddings (SOTA retrieval performance)
- GPU optimization with FP16
- Incremental indexing (only new documents)
- Comprehensive logging and metadata extraction

Usage:
  # Index PDFs
  python rag_indexer.py /path/to/pdfs

  # Reset and reindex
  python rag_indexer.py /path/to/pdfs --reset

  # Use different embedding model
  python rag_indexer.py /path/to/pdfs --model BAAI/bge-small-en-v1.5
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm

import fitz  # PyMuPDF
import torch
import numpy as np
from elasticsearch import Elasticsearch, exceptions as es_exceptions
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_indexer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIG DEFAULTS ---
ES_URL = "http://localhost:9200"

# Embedding model (SOTA performance)
EMBED_MODEL = "BAAI/bge-large-en-v1.5"  # 1024 dims
# Alternative: "BAAI/bge-small-en-v1.5" for speed (384 dims)
EMBED_DIMS = 1024

# Chunking params
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_WORDS_FOR_PAGE = 5

# Indexing params
BULK_SIZE = 500
BATCH_EMBED = 64

# -------------------------
# Argument Parsing
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="RAG PDF Indexer")
    p.add_argument("folder_path", help="Path to folder containing PDF files")
    p.add_argument("--reset", action="store_true", help="Delete all user indices before indexing")
    p.add_argument("--es-url", type=str, default=ES_URL)
    p.add_argument("--model", type=str, default=EMBED_MODEL, help="Embedding model")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    p.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    p.add_argument("--batch-embed", type=int, default=BATCH_EMBED)
    p.add_argument("--bulk-size", type=int, default=BULK_SIZE)
    p.add_argument("--index-name", type=str, default=None, help="Explicit ES index name")
    return p.parse_args()

# -------------------------
# ES Index Management
# -------------------------
def get_index_name(folder_path, explicit_name=None):
    if explicit_name:
        return explicit_name
    base = os.path.basename(os.path.normpath(folder_path))
    cleaned = base.lower().replace(" ", "_").replace("-", "_")
    if cleaned.startswith(("_", "-", "+")):
        cleaned = "idx_" + cleaned
    return cleaned

def create_rag_index(es: Elasticsearch, index_name: str, dims=EMBED_DIMS):
    """Create optimized ES index with tuned BM25 and HNSW settings"""
    if es.indices.exists(index=index_name):
        logger.info(f"Index {index_name} already exists.")
        return

    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "30s",
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
                "page_number": {"type": "integer"},
                "chunk_index": {"type": "integer"},
                "chunk_id": {"type": "keyword"},
                "chunk_text": {
                    "type": "text",
                    "similarity": "custom_bm25",
                    "analyzer": "standard"
                },
                "word_count": {"type": "integer"},
                "source_folder": {"type": "keyword"},
                "doc_title": {"type": "text"},
                "doc_author": {"type": "text"},
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
    logger.info(f"✓ Created index: {index_name} with {dims} dimensions")

# -------------------------
# Semantic Text Chunking
# -------------------------
def chunk_text_semantic(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Chunk text by sentences while respecting chunk_size.
    Maintains sentence integrity and context with overlap.
    """
    if not text:
        return []
    
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_len = len(sentence)
        
        if current_length + sentence_len > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            
            current_chunk = overlap_sentences
            current_length = overlap_len
        
        current_chunk.append(sentence)
        current_length += sentence_len
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# -------------------------
# PDF Processing with Metadata
# -------------------------
def extract_page_texts(pdf_path: str, min_words: int = MIN_WORDS_FOR_PAGE) -> List[Dict]:
    """Extract text and metadata from PDF pages"""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open {pdf_path}: {e}")
        return []
    
    metadata = doc.metadata or {}
    doc_title = metadata.get('title', os.path.basename(pdf_path))
    doc_author = metadata.get('author', '')
    
    results = []
    for pi in range(doc.page_count):
        page = doc.load_page(pi)
        text = page.get_text("text")
        
        page_text = "\n".join([ln.strip() for ln in text.splitlines() if ln.strip()])
        wc = len(page_text.split())
        
        if wc < min_words:
            del page
            continue
        
        results.append({
            'page_num': pi + 1,
            'text': page_text,
            'word_count': wc,
            'doc_title': doc_title,
            'doc_author': doc_author
        })
        
        del page
    
    doc.close()
    return results

# -------------------------
# GPU-Optimized Embedding
# -------------------------
def get_embedding_model(model_name: str, use_fp16: bool = True):
    """Load embedding model with GPU optimization"""
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
    
    return model

# -------------------------
# Indexing Pipeline
# -------------------------
def build_chunks_for_file(pdf_path: str, source_folder: str, chunk_size: int, overlap: int) -> List[Dict]:
    """Build chunks from PDF with metadata"""
    file_base = os.path.splitext(os.path.basename(pdf_path))[0].lower()
    pages = extract_page_texts(pdf_path)
    
    if not pages:
        return []
    
    out = []
    for page_info in pages:
        page_number = page_info['page_num']
        page_text = page_info['text']
        doc_title = page_info['doc_title']
        doc_author = page_info['doc_author']
        
        chunks = chunk_text_semantic(page_text, chunk_size, overlap)
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{file_base}__p{page_number}__c{idx}"
            out.append({
                "_id": chunk_id,
                "_source": {
                    "file_name": file_base,
                    "page_number": page_number,
                    "chunk_index": idx,
                    "chunk_id": chunk_id,
                    "chunk_text": chunk,
                    "word_count": len(chunk.split()),
                    "source_folder": source_folder,
                    "doc_title": doc_title,
                    "doc_author": doc_author,
                    "created_date": datetime.now().isoformat()
                }
            })
    
    return out

def index_chunks(es: Elasticsearch, index_name: str, chunks: List[Dict], 
                 embed_model_name: str, batch_embed: int = BATCH_EMBED, 
                 bulk_size: int = BULK_SIZE):
    """Compute embeddings with GPU optimization and bulk index"""
    model = get_embedding_model(embed_model_name)
    
    # Increase batch size on GPU
    if torch.cuda.is_available():
        batch_embed = min(batch_embed * 2, 128)
        logger.info(f"✓ GPU detected: using batch size {batch_embed}")
    
    docs_to_index = []
    texts = [c["_source"]["chunk_text"] for c in chunks]
    ids = [c["_id"] for c in chunks]
    
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
            doc = {
                "_index": index_name,
                "_id": batch_ids[j],
                "_source": {**chunks[idx_global]["_source"], "embedding": emb.tolist()}
            }
            docs_to_index.append(doc)
        
        if len(docs_to_index) >= bulk_size:
            success, failed = bulk(
                es.options(request_timeout=120),
                docs_to_index,
                chunk_size=bulk_size,
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
            chunk_size=bulk_size,
            raise_on_error=False
        )
        logger.info(f"  Final bulk: success={success}, failed={len(failed) if isinstance(failed, list) else failed}")

# -------------------------
# Incremental Indexing
# -------------------------
def run_indexing(folder_path: str, es: Elasticsearch, index_name: str, args):
    """Scan PDFs and incrementally index new chunks"""
    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        source = os.path.basename(root) if root != folder_path else os.path.basename(folder_path)
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_files.append((os.path.join(root, f), source))
    
    if not pdf_files:
        logger.warning("No PDFs found.")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    all_chunks = []
    for pdf_path, source in tqdm(pdf_files, desc="Processing PDFs"):
        chunks = build_chunks_for_file(pdf_path, source, args.chunk_size, args.overlap)
        if not chunks:
            continue
        
        # Check which chunks already exist
        ids = [c["_id"] for c in chunks]
        to_add = []
        
        for i in range(0, len(ids), 200):
            batch_ids = ids[i:i+200]
            try:
                res = es.mget(index=index_name, body={"ids": batch_ids})
                for k, docres in enumerate(res.get("docs", [])):
                    if not docres.get("found", False):
                        to_add.append(chunks[i + k])
            except es_exceptions.NotFoundError:
                to_add.extend(chunks[i:i+200])
        
        if to_add:
            all_chunks.extend(to_add)
    
    logger.info(f"New chunks to index: {len(all_chunks)}")
    
    if all_chunks:
        index_chunks(es, index_name, all_chunks, args.model, 
                    batch_embed=args.batch_embed, bulk_size=args.bulk_size)
        es.indices.refresh(index=index_name)
        logger.info("✓ Indexing complete!")
    else:
        logger.info("✓ No new chunks to add. Index is up to date.")

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    folder = args.folder_path
    
    if not os.path.isdir(folder):
        logger.error("Invalid folder path.")
        sys.exit(1)
    
    index_name = get_index_name(folder, args.index_name)
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
    
    # Index PDFs
    logger.info("Starting indexing pipeline...")
    run_indexing(folder, es, index_name, args)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Indexing complete for: {index_name}")
    logger.info(f"✓ You can now query using: rag_query.py")
    logger.info(f"{'='*60}\n")

if __name__ == "__main__":
    main()