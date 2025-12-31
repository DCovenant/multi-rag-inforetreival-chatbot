#!/usr/bin/env python3
"""
RAG Document Embeddings Generator
=================================

Generates embeddings for OCR-extracted chunks and indexes to Elasticsearch.

Usage:
  python rag-indexer-embed.py ocr_output.json --es-url http://localhost:9200 --index documents

Requirements:
  pip install elasticsearch sentence-transformers torch
"""
import argparse
import json
import logging
import os
from typing import List, Dict, Optional
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logger.info(f"✓ Model loaded (dims: {self.model.get_sentence_embedding_dimension()})")
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.tolist()
    
    @property
    def dims(self) -> int:
        return self.model.get_sentence_embedding_dimension()


class ElasticsearchIndexer:
    """Indexes chunks with embeddings to Elasticsearch."""
    
    def __init__(self, es_url: str, index_name: str):
        from elasticsearch import Elasticsearch
        self.es = Elasticsearch([es_url])
        self.index_name = index_name
        
        # Check connection
        if not self.es.ping():
            raise ConnectionError(f"Cannot connect to Elasticsearch at {es_url}")
        logger.info(f"✓ Connected to Elasticsearch")
    
    def create_index(self, embedding_dims: int):
        """Create index with proper mapping."""
        mapping = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "text_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "snowball", "asciifolding"]
                        }
                    }
                },
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "file_name": {"type": "keyword"},
                    "page_number": {"type": "integer"},
                    "chunk_index": {"type": "integer"},
                    "chunk_text": {
                        "type": "text",
                        "analyzer": "text_analyzer"
                    },
                    "word_count": {"type": "integer"},
                    "content_type": {"type": "keyword"},
                    "parent_sections": {"type": "keyword"},  # Changed: array of parent section numbers
                    "has_table": {"type": "boolean"},
                    "table_data": {"type": "object", "enabled": False},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": embedding_dims,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
        
        if self.es.indices.exists(index=self.index_name):
            logger.warning(f"Index {self.index_name} already exists, skipping creation")
            return
        
        self.es.indices.create(index=self.index_name, body=mapping)
        logger.info(f"✓ Created index: {self.index_name}")
    
    def bulk_index(self, chunks: List[Dict], batch_size: int = 100):
        """Bulk index chunks with embeddings."""
        from elasticsearch.helpers import bulk
        
        def generate_actions():
            for chunk in chunks:
                yield {
                    "_index": self.index_name,
                    "_id": chunk["chunk_id"],
                    "_source": chunk
                }
        
        success, failed = bulk(
            self.es, 
            generate_actions(),
            chunk_size=batch_size,
            raise_on_error=False
        )
        
        logger.info(f"Indexed {success} chunks, {len(failed) if failed else 0} failed")
        return success, failed


def prepare_embedding_text(chunk: Dict) -> str:
    """Prepare text for embedding with context enrichment."""
    parts = []
    
    # Add section context
    if chunk.get('parent_sections'):
        sections = ', '.join(chunk['parent_sections'])
        parts.append(f"[Sections: {sections}]")
    
    # Add content type marker
    if chunk.get('content_type') == 'table':
        parts.append("[Table]")
    elif chunk.get('content_type') in ('title', 'section_header', 'subsection_header'):
        parts.append(f"[{chunk['content_type'].replace('_', ' ').title()}]")
    
    # Add main text
    parts.append(chunk['chunk_text'])
    
    return ' '.join(parts)


def process_and_index(
    input_file: str,
    es_url: str = "http://localhost:9200",
    index_name: str = "documents",
    output_file: Optional[str] = None,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    reset_index: bool = False
):
    """Process OCR output, generate embeddings, and index to Elasticsearch."""
    
    # Load OCR output
    logger.info(f"Loading: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data['chunks']
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Initialize embedding generator
    embedder = EmbeddingGenerator(model_name)
    
    # Prepare texts for embedding
    texts = [prepare_embedding_text(chunk) for chunk in chunks]
    
    # Generate embeddings in batches
    logger.info("Generating embeddings...")
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        embeddings = embedder.embed_batch(batch, batch_size)
        all_embeddings.extend(embeddings)
    
    # Add embeddings to chunks
    for chunk, embedding in zip(chunks, all_embeddings):
        chunk['embedding'] = embedding
    
    logger.info(f"✓ Generated {len(all_embeddings)} embeddings")
    
    # Save to output file if specified
    if output_file:
        data['embedding_model'] = model_name
        data['embedding_dims'] = embedder.dims
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved to: {output_file}")
    
    # Index to Elasticsearch
    indexer = ElasticsearchIndexer(es_url, index_name)

    if reset_index:
        # Get all indices and delete them one by one
        indices = indexer.es.cat.indices(format="json")
        for index_info in indices:
            idx_name = index_info['index']
            indexer.es.indices.delete(index=idx_name)
            logger.info(f"✓ Deleted index: {idx_name}")

    indexer.create_index(embedder.dims)
    indexer.bulk_index(chunks, batch_size=100)
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Generate embeddings and index to Elasticsearch'
    )
    parser.add_argument('input', help='OCR output JSON file')
    parser.add_argument('-o', '--output', help='Output JSON file with embeddings')
    parser.add_argument('--reset', action='store_true', help='Delete index before indexing')
    
    args = parser.parse_args()
    
    # Index name from input filename: "docling_output.json" -> "docling_output"
    index_name = os.path.splitext(os.path.basename(args.input))[0].lower().replace('-', '_')
    
    process_and_index(
        args.input,
        es_url="http://localhost:9200",
        index_name=index_name,
        output_file=args.output,
        model_name="all-MiniLM-L6-v2",
        batch_size=32,
        reset_index=args.reset
    )

if __name__ == '__main__':
    main()