#!/usr/bin/env python3
"""
Advanced RAG Query System - Query Pipeline Only

Features:
- Hybrid search (BM25 + vector)
- Query expansion using LLM
- Cross-encoder reranking
- Result caching for performance
- Source citation and tracking

Usage:
  # Single query
  python rag_query.py /path/to/pdfs "How do I restart the device?"

  # Interactive mode
  python rag_query.py /path/to/pdfs --interactive

  # Fast mode (no reranking/expansion)
  python rag_query.py /path/to/pdfs "question" --fast

  # Custom top-k results
  python rag_query.py /path/to/pdfs "question" --top-k 5
"""

import os
import sys
import argparse
import pickle
import logging
import time
from typing import List, Dict
from functools import lru_cache

import torch
import numpy as np
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, CrossEncoder
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_query.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIG DEFAULTS ---
ES_URL = "http://localhost:9200"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

# Models
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval params
BM25_K = 20
KNN_K = 30
RERANK_TOP_N = 20
FINAL_TOP_K = 3

# Cache
CACHE_FILE = "query_cache.pkl"

# Global model caches (singleton pattern)
_embedding_model_cache = {}
_reranker_model_cache = {}

# -------------------------
# Query Cache
# -------------------------
class QueryCache:
    def __init__(self, cache_file=CACHE_FILE):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self):
        try:
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            return {}
    
    def get(self, query):
        return self.cache.get(query)
    
    def set(self, query, result):
        self.cache[query] = result
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

query_cache = QueryCache()

# -------------------------
# Argument Parsing
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="RAG Query System")
    p.add_argument("folder_path", help="Path to folder (used for index name)")
    p.add_argument("query", nargs='?', help="Question to ask")
    p.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    p.add_argument("--es-url", type=str, default=ES_URL)
    p.add_argument("--model", type=str, default=EMBED_MODEL, help="Embedding model")
    p.add_argument("--rerank-model", type=str, default=RERANK_MODEL)
    p.add_argument("--ollama-model", type=str, default=OLLAMA_MODEL)
    p.add_argument("--top-k", type=int, default=FINAL_TOP_K)
    p.add_argument("--fast", action="store_true", help="Fast mode (no rerank/expansion)")
    p.add_argument("--smart", "-s", action="store_true", 
                   help="Smart mode: Use HyDE + keyword extraction for natural language queries")
    p.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    p.add_argument("--no-expand", action="store_true", help="Disable query expansion")
    p.add_argument("--no-cache", action="store_true", help="Disable caching")
    p.add_argument("--index-name", type=str, default=None)
    return p.parse_args()

# -------------------------
# Index Name Helper
# -------------------------
def get_index_name(folder_path, explicit_name=None):
    if explicit_name:
        return explicit_name
    base = os.path.basename(os.path.normpath(folder_path))
    cleaned = base.lower().replace(" ", "_").replace("-", "_")
    if cleaned.startswith(("_", "-", "+")):
        cleaned = "idx_" + cleaned
    return cleaned

# -------------------------
# Model Loading
# -------------------------
def get_embedding_model(model_name: str):
    """Load embedding model with GPU support (cached)"""
    global _embedding_model_cache
    
    if model_name in _embedding_model_cache:
        return _embedding_model_cache[model_name]
    
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    if torch.cuda.is_available():
        model = model.to('cuda')
        model.half()
        logger.info(f"✓ Using GPU for embeddings")
    
    _embedding_model_cache[model_name] = model
    return model

@lru_cache(maxsize=1000)
def get_cached_embedding(query_text: str, model_name: str) -> tuple:
    """Cache query embeddings (returns tuple for hashability)"""
    model = get_embedding_model(model_name)
    emb = model.encode([query_text], normalize_embeddings=True)[0]
    return tuple(emb.tolist())

# -------------------------
# Ollama Integration
# -------------------------
def ask_ollama(prompt: str, model_name: str = OLLAMA_MODEL) -> str:
    """Query Ollama LLM"""
    body = {"model": model_name, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=body, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        logger.error(f"Ollama request failed: {e}")
        return ""

# -------------------------
# Query Expansion
# -------------------------
def expand_query(query: str, model_name: str = OLLAMA_MODEL) -> List[str]:
    """Generate query variations using LLM"""
    prompt = f"""Generate 2 alternative phrasings of this question that preserve the same meaning but use different words:

Question: {query}

Alternative 1:
Alternative 2:"""
    
    try:
        response = ask_ollama(prompt, model_name)
        alternatives = [query]
        
        for line in response.split('\n'):
            line = line.strip()
            if line and not line.startswith(('Question:', 'Alternative')):
                cleaned = line.lstrip('12.:-) ').strip()
                if cleaned and len(cleaned) > 10:
                    alternatives.append(cleaned)
        
        return alternatives[:3]
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return [query]

# -------------------------
# HyDE (Hypothetical Document Embeddings)
# -------------------------
def generate_hypothetical_answer(query: str, model_name: str = OLLAMA_MODEL) -> str:
    """Generate a hypothetical document/answer that might contain the information.
    This helps bridge the semantic gap between natural questions and technical docs."""
    
    prompt = f"""You are a technical documentation expert for electrical engineering specifications. Given this question, write a short paragraph (2-3 sentences) that might appear in a technical specification document answering this question. 

Include:
- Specific technical terms (e.g., terminal blocks, enclosures, cable glands)
- Type classifications (Type 1, Type 2, Class A, Category B, etc.)
- Standards references (BS EN, IEC, IEEE)
- Specification language (shall be used, is required, must comply)

Question: {query}

Technical specification excerpt:"""
    
    try:
        response = ask_ollama(prompt, model_name)
        if response and len(response) > 20:
            logger.info(f"HyDE generated: {response[:100]}...")
            return response.strip()
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}")
    return ""

def extract_technical_keywords(query: str, model_name: str = OLLAMA_MODEL) -> List[str]:
    """Extract likely technical keywords/terms that might appear in specifications."""
    
    prompt = f"""Given this question about electrical/engineering specifications, list the most likely technical terms that would appear in specification documents.

Include variations like:
- Component names (terminal blocks, cable glands, enclosures, splice boxes)
- Type classifications (Type 1, Type 2, Type "1", Class A)
- Industry terms (screw clamp, spring retention, DIN rail)
- Related equipment and standards

Question: {query}

Return only keywords/phrases, one per line, no explanations or numbering:"""
    
    try:
        response = ask_ollama(prompt, model_name)
        keywords = []
        for line in response.split('\n'):
            line = line.strip().lstrip('•-*123456789.) ')
            if line and len(line) > 2 and len(line) < 50:
                keywords.append(line)
        if keywords:
            logger.info(f"Extracted keywords: {keywords[:8]}")
        return keywords[:10]
    except Exception as e:
        logger.warning(f"Keyword extraction failed: {e}")
        return []

# -------------------------
# Hybrid Search
# -------------------------
def hybrid_search(es: Elasticsearch, index_name: str, query: str, 
                  embed_model_name: str, bm25_k: int = BM25_K, 
                  knn_k: int = KNN_K) -> List[Dict]:
    """Hybrid BM25 + vector search"""
    
    # BM25 search
    bm25_body = {
        "query": {"match": {"chunk_text": query}},
        "size": bm25_k,
        "_source": ["chunk_text", "file_name", "page_number", "chunk_index", "doc_title"]
    }
    bm25_response = es.search(index=index_name, body=bm25_body)
    bm25_hits = bm25_response.get("hits", {}).get("hits", [])
    
    # Vector search (kNN)
    q_emb = list(get_cached_embedding(query, embed_model_name))
    knn_body = {
        "knn": {
            "field": "embedding",
            "query_vector": q_emb,
            "k": knn_k,
            "num_candidates": max(100, knn_k * 10)
        },
        "_source": ["chunk_text", "file_name", "page_number", "chunk_index", "doc_title"]
    }
    
    try:
        knn_response = es.search(index=index_name, body=knn_body)
        knn_hits = knn_response.get("hits", {}).get("hits", [])
    except Exception as e:
        logger.warning(f"KNN search failed: {e}")
        knn_hits = []
    
    # Combine results
    combined = {}
    for h in bm25_hits:
        combined[h["_id"]] = {
            "id": h["_id"],
            "chunk_text": h["_source"]["chunk_text"],
            "file_name": h["_source"].get("file_name"),
            "page_number": h["_source"].get("page_number"),
            "doc_title": h["_source"].get("doc_title"),
            "score": float(h.get("_score", 0.0))
        }
    
    for h in knn_hits:
        if h["_id"] in combined:
            combined[h["_id"]]["score"] += float(h.get("_score", 0.0))
        else:
            combined[h["_id"]] = {
                "id": h["_id"],
                "chunk_text": h["_source"]["chunk_text"],
                "file_name": h["_source"].get("file_name"),
                "page_number": h["_source"].get("page_number"),
                "doc_title": h["_source"].get("doc_title"),
                "score": float(h.get("_score", 0.0))
            }
    
    items = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return items

# -------------------------
# Reranking
# -------------------------
def get_reranker_model(model_name: str):
    """Load reranker model (cached)"""
    global _reranker_model_cache
    
    if model_name in _reranker_model_cache:
        return _reranker_model_cache[model_name]
    
    logger.info(f"Loading reranker model: {model_name}")
    model = CrossEncoder(model_name)
    _reranker_model_cache[model_name] = model
    return model


def rerank_results(query: str, chunks: List[Dict], rerank_model_name: str, 
                   top_k: int = FINAL_TOP_K) -> List[Dict]:
    """Rerank retrieved chunks using cross-encoder"""
    if not chunks:
        return []
    
    try:
        reranker = get_reranker_model(rerank_model_name)
        pairs = [(query, c['chunk_text']) for c in chunks]
        scores = reranker.predict(pairs)
        
        for i, chunk in enumerate(chunks):
            chunk['rerank_score'] = float(scores[i])
            chunk['final_score'] = chunk['score'] + scores[i] * 2
        
        return sorted(chunks, key=lambda x: x['final_score'], reverse=True)[:top_k]
    except Exception as e:
        logger.warning(f"Reranking failed: {e}")
        return chunks[:top_k]

# -------------------------
# Full Retrieval Pipeline
# -------------------------
def retrieve_documents(es: Elasticsearch, index_name: str, query: str,
                      embed_model_name: str, rerank_model_name: str,
                      top_k: int = FINAL_TOP_K, use_rerank: bool = True,
                      use_expansion: bool = True, use_smart: bool = False,
                      ollama_model: str = OLLAMA_MODEL) -> List[Dict]:
    """Full retrieval pipeline with optional query expansion, HyDE, and reranking.
    
    Args:
        use_smart: Enable smart mode (HyDE + keyword extraction) for natural language queries
    """
    
    all_results = {}
    
    # Smart mode: Use HyDE and keyword extraction for better semantic bridging
    if use_smart:
        logger.info("Smart mode: Generating hypothetical answer and extracting keywords...")
        
        # 1. HyDE - Generate hypothetical document and search with its embedding
        hyde_text = generate_hypothetical_answer(query, ollama_model)
        if hyde_text:
            logger.info("Searching with HyDE embedding...")
            hyde_results = hybrid_search(es, index_name, hyde_text, embed_model_name, 
                                        bm25_k=BM25_K, knn_k=KNN_K)
            for r in hyde_results:
                chunk_id = r['id']
                if chunk_id in all_results:
                    all_results[chunk_id]['score'] += r['score'] * 1.5  # Boost HyDE results
                else:
                    all_results[chunk_id] = r
                    all_results[chunk_id]['score'] *= 1.5
        
        # 2. Extract technical keywords and search with them
        keywords = extract_technical_keywords(query, ollama_model)
        if keywords:
            keyword_query = " ".join(keywords[:5])
            logger.info(f"Searching with extracted keywords: {keyword_query}")
            kw_results = hybrid_search(es, index_name, keyword_query, embed_model_name,
                                      bm25_k=BM25_K, knn_k=KNN_K)
            for r in kw_results:
                chunk_id = r['id']
                if chunk_id in all_results:
                    all_results[chunk_id]['score'] += r['score']
                else:
                    all_results[chunk_id] = r
    
    # Standard query expansion
    if use_expansion:
        queries = expand_query(query, ollama_model)
        logger.info(f"Expanded to {len(queries)} query variations")
    else:
        queries = [query]
    
    # Search with all query variations
    for q in queries:
        results = hybrid_search(es, index_name, q, embed_model_name, bm25_k=BM25_K, knn_k=KNN_K)
        for r in results:
            chunk_id = r['id']
            if chunk_id in all_results:
                all_results[chunk_id]['score'] += r['score']
            else:
                all_results[chunk_id] = r
    
    # Sort by aggregated score
    candidates = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)[:RERANK_TOP_N]
    
    # Rerank
    if use_rerank and len(candidates) > 1:
        logger.info(f"Reranking top {len(candidates)} candidates")
        final_results = rerank_results(query, candidates, rerank_model_name, top_k=top_k)
    else:
        final_results = candidates[:top_k]
    
    return final_results

# -------------------------
# Prompt Building
# -------------------------
def build_prompt(question: str, retrieved_chunks: List[Dict]) -> str:
    """Build enhanced prompt with deduplication"""
    
    # Deduplicate by page
    seen_pages = set()
    unique_chunks = []
    for c in retrieved_chunks:
        page_key = f"{c.get('file_name')}_{c.get('page_number')}"
        if page_key not in seen_pages:
            unique_chunks.append(c)
            seen_pages.add(page_key)
    
    # Build context
    context_parts = []
    for i, c in enumerate(unique_chunks, 1):
        doc_title = c.get('doc_title', c.get('file_name', 'Unknown'))
        page = c.get('page_number', '?')
        text = c.get('chunk_text', '')
        context_parts.append(f"[Source {i}: {doc_title} - Page {page}]\n{text}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""You are a knowledgeable assistant. Answer the user's question using ONLY the information provided in the context below.

IMPORTANT INSTRUCTIONS:
- If the answer is not in the context, say "I don't have enough information to answer that question based on the provided documents."
- Always cite the source document and page number (e.g., "According to [Source 1: Document Name - Page 5]...")
- Be concise but complete
- If multiple sources provide information, mention all relevant sources
- Do not make up information or add details not present in the context

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    
    return prompt

# -------------------------
# Query Pipeline
# -------------------------
def query_and_answer(es: Elasticsearch, index_name: str, question: str,
                     embed_model_name: str, rerank_model_name: str, 
                     ollama_model: str, top_k: int = FINAL_TOP_K, 
                     use_rerank: bool = True, use_expansion: bool = True, 
                     use_cache: bool = True, use_smart: bool = False) -> Dict:
    """Complete query pipeline with caching.
    
    Args:
        use_smart: Enable smart mode for natural language queries (HyDE + keyword extraction)
    """
    
    start_time = time.time()
    
    # Check cache
    cache_key = f"{index_name}:{question}:{top_k}:{use_rerank}:{use_expansion}:{use_smart}"
    if use_cache:
        cached = query_cache.get(cache_key)
        if cached:
            logger.info("✓ Retrieved answer from cache")
            cached['from_cache'] = True
            return cached
    
    # Retrieve
    retrieved = retrieve_documents(
        es, index_name, question, embed_model_name, rerank_model_name,
        top_k=top_k, use_rerank=use_rerank, use_expansion=use_expansion,
        use_smart=use_smart, ollama_model=ollama_model
    )
    
    if not retrieved:
        result = {
            "answer": "No relevant documents found for your question.",
            "sources": [],
            "query_time": time.time() - start_time,
            "from_cache": False
        }
        return result
    
    # Generate answer
    prompt = build_prompt(question, retrieved)
    answer = ask_ollama(prompt, ollama_model)
    
    query_time = time.time() - start_time
    
    result = {
        "answer": answer,
        "sources": [
            {
                "file": c.get('file_name'),
                "page": c.get('page_number'),
                "title": c.get('doc_title'),
                "score": round(c.get('final_score', c.get('score', 0)), 3),
                "preview": c.get('chunk_text', '')[:150] + "..."
            }
            for c in retrieved
        ],
        "query_time": query_time,
        "from_cache": False
    }
    
    # Cache result
    if use_cache:
        query_cache.set(cache_key, result)
    
    logger.info(f"✓ Query completed in {query_time:.2f}s with {len(retrieved)} sources")
    
    return result

# -------------------------
# Display Results
# -------------------------
def display_result(result: Dict):
    """Pretty print query results"""
    print("\n" + "="*70)
    print("ANSWER:")
    print("="*70)
    print(result['answer'])
    print("\n" + "="*70)
    print("SOURCES:")
    print("="*70)
    for i, src in enumerate(result['sources'], 1):
        print(f"\n[{i}] {src['title'] or src['file']} - Page {src['page']}")
        print(f"    Score: {src['score']}")
        print(f"    Preview: {src['preview']}")
    print("\n" + "="*70)
    cache_status = " (from cache)" if result.get('from_cache') else ""
    print(f"Query time: {result['query_time']:.2f}s{cache_status}")
    print("="*70 + "\n")

# -------------------------
# Interactive Mode
# -------------------------
def interactive_mode(es: Elasticsearch, index_name: str, args):
    """Interactive query mode"""
    print("\n" + "="*70)
    print("RAG INTERACTIVE QUERY MODE")
    print("="*70)
    print("Type your questions below. Commands:")
    print("  'quit' or 'exit' - Exit interactive mode")
    print("  'fast' - Toggle fast mode (no rerank/expansion)")
    print("  'cache' - Toggle caching")
    print("="*70 + "\n")
    
    fast_mode = args.fast
    use_cache = not args.no_cache
    
    while True:
        try:
            question = input("\nQuestion: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if question.lower() == 'fast':
                fast_mode = not fast_mode
                print(f"Fast mode: {'ON' if fast_mode else 'OFF'}")
                continue
            
            if question.lower() == 'cache':
                use_cache = not use_cache
                print(f"Caching: {'ON' if use_cache else 'OFF'}")
                continue
            
            # Query
            result = query_and_answer(
                es, index_name, question,
                args.model, args.rerank_model, args.ollama_model,
                top_k=args.top_k,
                use_rerank=not fast_mode and not args.no_rerank,
                use_expansion=not fast_mode and not args.no_expand,
                use_cache=use_cache,
                use_smart=args.smart
            )
            
            display_result(result)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"Error: {e}\n")

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    
    if not args.query and not args.interactive:
        print("Error: Provide a query or use --interactive mode")
        sys.exit(1)
    
    index_name = get_index_name(args.folder_path, args.index_name)
    logger.info(f"Using index: {index_name}")
    
    # Connect to ES
    es = Elasticsearch(args.es_url, request_timeout=60)
    try:
        info = es.info()
        logger.info(f"✓ Connected to Elasticsearch: {info['version']['number']}")
    except Exception as e:
        logger.error(f"Failed to connect to Elasticsearch: {e}")
        sys.exit(1)
    
    # Check index exists
    if not es.indices.exists(index=index_name):
        logger.error(f"Index '{index_name}' not found. Run rag_indexer.py first.")
        sys.exit(1)
    
    # Preload models at startup (one-time cost)
    logger.info("Preloading models...")
    get_embedding_model(args.model)
    if not args.fast and not args.no_rerank:
        get_reranker_model(args.rerank_model)
    logger.info("✓ Models ready")
    
    # Interactive or single query
    if args.interactive:
        interactive_mode(es, index_name, args)
    else:
        result = query_and_answer(
            es, index_name, args.query,
            args.model, args.rerank_model, args.ollama_model,
            top_k=args.top_k,
            use_rerank=not args.fast and not args.no_rerank,
            use_expansion=not args.fast and not args.no_expand,
            use_cache=not args.no_cache,
            use_smart=args.smart
        )
        display_result(result)

if __name__ == "__main__":
    main()