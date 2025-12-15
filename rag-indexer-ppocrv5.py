#!/usr/bin/env python3
"""
Advanced RAG OCR Indexer using PaddleOCR v5

Features:
- PaddleOCR v5 for scanned PDF text extraction
- Semantic chunking (respects sentence boundaries)
- BGE embeddings (SOTA retrieval performance)
- GPU optimization with FP16
- Incremental indexing (only new documents)
- Multi-page PDF support

Usage:
  # Index PDFs with OCR
  python rag-indexer-ppocrv5.py /path/to/pdfs

  # Reset and reindex
  python rag-indexer-ppocrv5.py /path/to/pdfs --reset

  # Use different embedding model
  python rag-indexer-ppocrv5.py /path/to/pdfs --model BAAI/bge-small-en-v1.5

  # Custom
  python rag-indexer-ppocrv5.py /path/to/pdfs --tiles 4 --dpi 300 --chunk-size 1500 --overlap 300
"""
from __future__ import annotations

from paddleocr import PaddleOCR
import numpy as np
import sys
import os
from pdf2image import convert_from_path
import argparse
from tqdm import tqdm
import gc
import cv2
from PIL import Image
import re
import logging
from datetime import datetime
from typing import List, Dict

import torch
from sentence_transformers import SentenceTransformer

try:
    from elasticsearch import Elasticsearch, exceptions as es_exceptions
    from elasticsearch.helpers import bulk
    ES_AVAILABLE = True
except ImportError:
    ES_AVAILABLE = False

Image.MAX_IMAGE_PIXELS = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_ocr_indexer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIG DEFAULTS ---
ES_URL = 'http://localhost:9200'

# Embedding model (SOTA performance)
EMBED_MODEL = "BAAI/bge-large-en-v1.5"  # 1024 dims
EMBED_DIMS = 1024

# Chunking params
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_WORDS_FOR_PAGE = 5

# Indexing params
BULK_SIZE = 500
BATCH_EMBED = 64


# -------------------------
# Semantic Text Chunking
# -------------------------
def chunk_text_semantic(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Chunk text by sentences while respecting chunk_size.
    Maintains sentence integrity and context with overlap.
    """
    if not text:
        return []
    
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
# OCR Text Detection Class
# -------------------------
class TextDetection():
    def __init__(self) -> None:
        self.ocr = None
        self.stats = {'files': 0, 'pages': 0, 'words': 0}

    def initPaddle(self, det_limit=1920) -> None:
        """Initialize PaddleOCR"""
        self.ocr = PaddleOCR(
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            text_det_limit_type='max',
            text_det_limit_side_len=det_limit,
            text_det_thresh=0.3,
            text_det_box_thresh=0.6,
            text_recognition_batch_size=4,
        )

    def split_into_tiles(self, img, num_tiles=2, overlap=50):
        """Split image into tiles"""
        h, w = img.shape[:2]
        
        if num_tiles == 1:
            return [(img, 0, 0)]
        elif num_tiles == 2:
            mid_y = h // 2
            return [
                (img[0:mid_y + overlap, :], 0, 0),
                (img[mid_y - overlap:h, :], 0, mid_y - overlap)
            ]
        elif num_tiles == 4:
            mid_y, mid_x = h // 2, w // 2
            return [
                (img[0:mid_y + overlap, 0:mid_x + overlap], 0, 0),
                (img[0:mid_y + overlap, mid_x - overlap:w], mid_x - overlap, 0),
                (img[mid_y - overlap:h, 0:mid_x + overlap], 0, mid_y - overlap),
                (img[mid_y - overlap:h, mid_x - overlap:w], mid_x - overlap, mid_y - overlap)
            ]
        else:
            raise ValueError(f"num_tiles must be 1, 2 or 4")

    def process_tile(self, tile_data):
        """Process single tile"""
        tile, offset_x, offset_y, tile_idx = tile_data
        result = self.ocr.predict(tile)
        return result, offset_x, offset_y, tile_idx

    def adjust_coordinates(self, poly, offset_x, offset_y):
        """Adjust coordinates to full image space"""
        return [[float(pt[0]) + offset_x, float(pt[1]) + offset_y] for pt in poly]

    def merge_results(self, tile_results, overlap, num_tiles):
        """Merge results from tiles"""
        merged = []
        
        for result, offset_x, offset_y, tile_idx in tile_results:
            if not result or not result[0]:
                continue
            
            for res in result:
                texts = res.get('rec_texts', []) if isinstance(res, dict) else getattr(res, 'rec_texts', [])
                scores = res.get('rec_scores', []) if isinstance(res, dict) else getattr(res, 'rec_scores', [])
                polys = res.get('rec_polys', []) if isinstance(res, dict) else getattr(res, 'rec_polys', [])
                
                if not texts:
                    continue
                
                for j, text in enumerate(texts):
                    if not text or not str(text).strip():
                        continue
                    
                    score = scores[j] if j < len(scores) else 0.0
                    poly = polys[j] if j < len(polys) else None
                    
                    if poly is None:
                        continue
                    
                    adjusted_box = self.adjust_coordinates(poly, offset_x, offset_y)
                    
                    # Skip overlap regions
                    if num_tiles > 1 and tile_idx > 0:
                        center_y = sum(pt[1] for pt in adjusted_box) / 4
                        if num_tiles == 2 and tile_idx == 1 and center_y < offset_y + overlap:
                            continue
                        if num_tiles == 4:
                            center_x = sum(pt[0] for pt in adjusted_box) / 4
                            if tile_idx == 1 and center_x < offset_x + overlap:
                                continue
                            if tile_idx == 2 and center_y < offset_y + overlap:
                                continue
                            if tile_idx == 3 and (center_x < offset_x + overlap or center_y < offset_y + overlap):
                                continue
                    
                    merged.append({
                        "text": str(text),
                        "confidence": float(score),
                        "box": adjusted_box,
                        "tile_idx": tile_idx
                    })
        
        return merged

    def paddleOCR(self, file_path, num_tiles=2, dpi=200):
        """Main OCR processing"""
        images = convert_from_path(file_path, dpi=dpi)
        img = np.array(images[0])
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        tiles = self.split_into_tiles(img_bgr, num_tiles, overlap=50)
        tile_data = [(tile, ox, oy, i) for i, (tile, ox, oy) in enumerate(tiles)]
        
        tile_results = [self.process_tile(td) for td in tile_data]
        merged_results = self.merge_results(tile_results, 50, num_tiles)

        self.stats['pages'] += 1
        self.stats['words'] += len(merged_results)

        img_height, img_width = img.shape[:2]

        del images, img, img_bgr, tiles, tile_data, tile_results
        gc.collect()

        return merged_results, img_width, img_height

    def process_pdf_pages(self, file_path: str, num_tiles: int = 2, dpi: int = 200) -> List[Dict]:
        """Process all pages of a PDF and return text for each page"""
        try:
            images = convert_from_path(file_path, dpi=dpi)
        except Exception as e:
            logger.error(f"Failed to convert PDF {file_path}: {e}")
            return []
        
        pages_data = []
        
        for page_num, pil_image in enumerate(images, 1):
            img = np.array(pil_image)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_height, img_width = img.shape[:2]
            
            tiles = self.split_into_tiles(img_bgr, num_tiles, overlap=50)
            tile_data = [(tile, ox, oy, i) for i, (tile, ox, oy) in enumerate(tiles)]
            
            tile_results = [self.process_tile(td) for td in tile_data]
            merged_results = self.merge_results(tile_results, 50, num_tiles)
            
            # Extract text from OCR results
            text_parts = []
            for detection in merged_results:
                text = detection.get("text", "")
                if not text:
                    continue
                text_stripped = text.strip()
                if len(text_stripped) == 0:
                    continue
                # Skip single letters and digits
                if re.fullmatch(r'[a-zA-Z]{1,2}', text_stripped):
                    continue
                if re.fullmatch(r'\d', text_stripped):
                    continue
                text_parts.append(text_stripped)
            
            page_text = ' '.join(text_parts)
            word_count = len(page_text.split())
            
            if word_count >= MIN_WORDS_FOR_PAGE:
                pages_data.append({
                    'page_num': page_num,
                    'text': page_text,
                    'word_count': word_count
                })
            
            self.stats['pages'] += 1
            self.stats['words'] += word_count
            
            # Cleanup per page
            del img, img_bgr, tiles, tile_data, tile_results, merged_results
        
        del images
        gc.collect()
        
        return pages_data


# --- ELASTICSEARCH ---

def get_index_name(folder_path, explicit_name=None):
    if explicit_name:
        return explicit_name
    folder_name = os.path.basename(os.path.normpath(folder_path))
    cleaned = folder_name.lower().replace(' ', '_').replace('-', '_')
    if cleaned.startswith(('_', '-', '+')):
        cleaned = 'idx_' + cleaned
    return cleaned


def create_rag_index(es: Elasticsearch, index_name: str, dims: int = EMBED_DIMS):
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
    logger.info(f"✓ Created index: {index_name} with {dims} dimensions")


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
# Chunking Pipeline
# -------------------------
def build_chunks_for_file(pdf_path: str, source_folder: str, ocr_engine: TextDetection,
                          chunk_size: int, overlap: int, num_tiles: int, dpi: int) -> List[Dict]:
    """Build chunks from PDF with OCR extraction and metadata"""
    file_base = os.path.splitext(os.path.basename(pdf_path))[0].lower()
    
    # Extract text from all pages using OCR
    pages = ocr_engine.process_pdf_pages(pdf_path, num_tiles=num_tiles, dpi=dpi)
    
    if not pages:
        return []
    
    out = []
    for page_info in pages:
        page_number = page_info['page_num']
        page_text = page_info['text']
        
        # Semantic chunking
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
                    "doc_title": file_base,  # Use filename as title for OCR docs
                    "text_source": "paddleocr_v5",
                    "created_date": datetime.now().isoformat()
                }
            })
    
    return out


# -------------------------
# Indexing Pipeline
# -------------------------
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
def run_indexing(folder_path: str, es: Elasticsearch, index_name: str, args, ocr_engine: TextDetection):
    """Scan PDFs and incrementally index new chunks"""
    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        source = os.path.basename(root) if root != folder_path else os.path.basename(folder_path)
        for f in files:
            if f.lower().endswith(".pdf") and not f.endswith('.Zone.Identifier'):
                pdf_files.append((os.path.join(root, f), source))
    
    if not pdf_files:
        logger.warning("No PDFs found.")
        return
    
    pdf_files.sort()
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    all_chunks = []
    for pdf_path, source in tqdm(pdf_files, desc="Processing PDFs with OCR"):
        chunks = build_chunks_for_file(
            pdf_path, source, ocr_engine,
            args.chunk_size, args.overlap, args.tiles, args.dpi
        )
        if not chunks:
            continue
        
        # Check which chunks already exist (incremental indexing)
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
        
        # Periodic garbage collection
        if len(all_chunks) % 100 == 0:
            gc.collect()
    
    logger.info(f"New chunks to index: {len(all_chunks)}")
    
    if all_chunks:
        index_chunks(es, index_name, all_chunks, args.model,
                    batch_embed=args.batch_embed, bulk_size=args.bulk_size)
        es.indices.refresh(index=index_name)
        logger.info("✓ Indexing complete!")
    else:
        logger.info("✓ No new chunks to add. Index is up to date.")


# --- CLI ---

def parse_arguments():
    parser = argparse.ArgumentParser(description="RAG OCR Indexer using PaddleOCR v5")
    
    parser.add_argument("folder_path", help="Path to PDF folder")
    parser.add_argument("--reset", action="store_true", help="Delete all user indices before indexing")
    parser.add_argument("--tiles", type=int, choices=[1, 2, 4], default=2,
                        help="Number of tiles for OCR (default: 2)")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF rendering (default: 200)")
    parser.add_argument("--det-limit", type=int, default=1920, help="OCR detection limit")
    parser.add_argument("--es-url", type=str, default=ES_URL, help="Elasticsearch URL")
    parser.add_argument("--model", type=str, default=EMBED_MODEL, help="Embedding model")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help="Chunk overlap in characters")
    parser.add_argument("--batch-embed", type=int, default=BATCH_EMBED, help="Embedding batch size")
    parser.add_argument("--bulk-size", type=int, default=BULK_SIZE, help="ES bulk indexing size")
    parser.add_argument("--index-name", type=str, default=None, help="Explicit ES index name")
    
    args = parser.parse_args()
    
    if not ES_AVAILABLE:
        parser.error("Elasticsearch package not installed. Run: pip install elasticsearch")
    
    return args


# -------------------------
# Main
# -------------------------
def main():
    args = parse_arguments()
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
    
    # Initialize OCR engine
    logger.info("Initializing PaddleOCR v5...")
    ocr_engine = TextDetection()
    ocr_engine.initPaddle(det_limit=args.det_limit)
    logger.info("✓ PaddleOCR ready")
    
    # Index PDFs
    logger.info("Starting OCR indexing pipeline...")
    run_indexing(folder, es, index_name, args, ocr_engine)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Indexing complete for: {index_name}")
    logger.info(f"✓ You can now query using: rag_queries.py")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()