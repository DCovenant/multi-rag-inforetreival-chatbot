"""
Optimized RAG Retrieval for Elasticsearch
Hybrid search: BM25 + kNN with RRF fusion + reranking
"""

import logging
from typing import List, Dict, Optional
from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)

# --- CONFIG ---
BM25_K = 50
VECTOR_K = 30
RERANK_TOP_N = 30
FINAL_TOP_K = 10
RRF_K = 60

SOURCE_FIELDS = [
    "chunk_text", "file_name", "page_number", "chunk_index",
    "section_header", "chunk_role", "content_type", "has_table", 
    "table_headers", "table_caption", "table_data", "document_code",
    "parent_sections"
]


def search_documents(
    es: Elasticsearch,
    index_name: str,
    query: str,
    embedding_fn,
    reranker=None,
    filters: Optional[Dict] = None,
    final_k: int = FINAL_TOP_K
) -> List[Dict]:
    """
    Main search: hybrid (BM25 + kNN) with RRF fusion + optional reranking.
    """
    # 1. Compute query embedding
    query_embedding = embedding_fn(query)
    
    # 2. Hybrid search
    results = _hybrid_search(es, index_name, query, query_embedding, filters)
    
    if not results:
        logger.warning("No results from hybrid search")
        return []
    
    # 3. Rerank if available
    if reranker:
        results = _rerank(query, results, reranker)
    else:
        for r in results:
            r["final_score"] = r.get("rrf_score", 0.0)
    
    logger.info(f"Search: {len(results)} results, returning top {final_k}")
    return sorted(results, key=lambda x: x["final_score"], reverse=True)[:final_k]


def _hybrid_search(
    es: Elasticsearch,
    index_name: str,
    query: str,
    query_embedding: List[float],
    filters: Optional[Dict] = None
) -> List[Dict]:
    """BM25 + kNN with RRF fusion."""
    
    # BM25 query
    bm25_query = {
        "bool": {
            "should": [
                {"match": {"chunk_text": {"query": query, "boost": 2.0}}},
                {"match_phrase": {"chunk_text": {"query": query, "boost": 3.0, "slop": 2}}},
                {"match": {"section_header": {"query": query, "boost": 2.5}}},
                {"match": {"table_caption": {"query": query, "boost": 2.0}}},
            ],
            "minimum_should_match": 1
        }
    }
    q_lower = query.lower()
    if any(x in q_lower for x in ['table', 'levels', 'values', 'ratings', 'specifications']):
        bm25_query["bool"]["should"].append(
            {"term": {"has_table": {"value": True, "boost": 3.0}}}
        )
    
    if filters:
        bm25_query["bool"]["filter"] = _build_filters(filters)
    
    # kNN query
    knn_query = {
        "field": "embedding",
        "query_vector": query_embedding,
        "k": VECTOR_K,
        "num_candidates": VECTOR_K * 2
    }
    if filters:
        knn_query["filter"] = {"bool": {"filter": _build_filters(filters)}}
    
    try:
        # Execute both searches
        bm25_resp = es.search(index=index_name, query=bm25_query, size=BM25_K, _source=SOURCE_FIELDS)
        vector_resp = es.search(index=index_name, knn=knn_query, size=VECTOR_K, _source=SOURCE_FIELDS)
        
        bm25_hits = _parse_hits(bm25_resp, "bm25")
        vector_hits = _parse_hits(vector_resp, "vector")
        
        # RRF fusion
        return _rrf_fusion(bm25_hits, vector_hits)
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        return []


def _build_filters(filters: Dict) -> List[Dict]:
    """Convert filter dict to ES clauses."""
    clauses = []
    if "file_name" in filters:
        clauses.append({"term": {"file_name": filters["file_name"]}})
    if "content_type" in filters:
        val = filters["content_type"]
        clauses.append({"terms": {"content_type": val}} if isinstance(val, list) else {"term": {"content_type": val}})
    if "has_table" in filters:
        clauses.append({"term": {"has_table": filters["has_table"]}})
    return clauses


def _parse_hits(response: Dict, source: str) -> List[Dict]:
    """Parse ES hits."""
    results = []
    for rank, hit in enumerate(response.get("hits", {}).get("hits", []), 1):
        src = hit["_source"]
        results.append({
            "id": hit["_id"],
            "chunk_text": src.get("chunk_text", ""),
            "file_name": src.get("file_name"),
            "page_number": src.get("page_number"),
            "chunk_index": src.get("chunk_index", 0),
            "section_header": src.get("section_header"),
            "chunk_role": src.get("chunk_role"),
            "content_type": src.get("content_type"),
            "has_table": src.get("has_table", False),
            "table_headers": src.get("table_headers"),
            "table_caption": src.get("table_caption"),
            "table_data": src.get("table_data"),
            "document_code": src.get("document_code"),
            "parent_sections": src.get("parent_sections", []),
            "raw_score": float(hit.get("_score", 0.0)),
            f"{source}_rank": rank
        })
    return results


def _rrf_fusion(bm25_results: List[Dict], vector_results: List[Dict]) -> List[Dict]:
    """Reciprocal Rank Fusion."""
    docs = {}
    
    for doc in bm25_results:
        doc_id = doc["id"]
        docs[doc_id] = doc.copy()
        docs[doc_id]["rrf_bm25"] = 1.0 / (RRF_K + doc["bm25_rank"])
        docs[doc_id]["rrf_vector"] = 0.0
    
    for doc in vector_results:
        doc_id = doc["id"]
        rrf_v = 1.0 / (RRF_K + doc["vector_rank"])
        if doc_id in docs:
            docs[doc_id]["rrf_vector"] = rrf_v
            docs[doc_id]["vector_rank"] = doc["vector_rank"]
        else:
            docs[doc_id] = doc.copy()
            docs[doc_id]["rrf_bm25"] = 0.0
            docs[doc_id]["rrf_vector"] = rrf_v
    
    for d in docs.values():
        d["rrf_score"] = d["rrf_bm25"] + d["rrf_vector"]
    
    return list(docs.values())


def _rerank(query: str, results: List[Dict], reranker, top_n: int = RERANK_TOP_N) -> List[Dict]:
    """Rerank with cross-encoder."""
    if not results:
        return results
    
    candidates = results[:top_n]
    try:
        pairs = [(query, c["chunk_text"]) for c in candidates]
        scores = reranker.predict(pairs, show_progress_bar=False)
        
        for i, chunk in enumerate(candidates):
            chunk["rerank_score"] = float(scores[i])
            chunk["final_score"] = 0.3 * chunk.get("rrf_score", 0) + 0.7 * chunk["rerank_score"]
        
        # Non-reranked get RRF score as final
        for chunk in results[top_n:]:
            chunk["rerank_score"] = 0.0
            chunk["final_score"] = chunk.get("rrf_score", 0.0)
        
        return results
        
    except Exception as e:
        logger.warning(f"Reranking failed: {e}")
        for r in results:
            r["final_score"] = r.get("rrf_score", 0.0)
        return results