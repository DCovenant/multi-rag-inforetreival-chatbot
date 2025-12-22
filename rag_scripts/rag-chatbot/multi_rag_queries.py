#!/usr/bin/env python3
"""
Advanced Multi-Agent RAG (Retrieval-Augmented Generation) System

This system handles ambiguous natural language queries by using multiple AI agents
that work together to understand intent, retrieve relevant documents, and generate
accurate answers from technical documentation.

ARCHITECTURE:
1. Intent Classification Agent - Determines what type of question is being asked
2. Ambiguity Detector - Identifies unclear queries and asks for clarification
3. Query Expansion Agent - Rewrites queries to be more searchable
4. HyDE (Hypothetical Document Embeddings) - Generates fake answers to find similar real docs
5. Keyword Extractor - Pulls out technical terms from questions
6. Retrieval Pipeline - Combines BM25 (keyword) and semantic search
7. Iterative Refiner - Re-searches with refined queries if needed
8. Answer Generator - Creates context-aware responses with citations

KEY FEATURES:
- Conversation Memory: Tracks context across multiple questions
- Intent Detection: Understands if query is factual, procedural, comparative, etc.
- Smart Retrieval: Uses HyDE + keywords + semantic search for better results
- Iterative Refinement: Re-searches automatically if initial results are poor
- Ambiguity Handling: Asks clarifying questions when query is unclear

USAGE:
  # Single query with all smart features enabled
  python multi_rag_queries.py /path/to/pdfs "ambiguous question"
  
  # Interactive mode with conversation memory (recommended for users)
  python multi_rag_queries.py /path/to/pdfs --interactive
"""

import os
import sys
import argparse
import logging
import time
from typing import List, Dict, Optional, Tuple
from elasticsearch import Elasticsearch
import requests
import numpy as np

from utils.conversation_history import ConversationHistory
from utils.model_loading import get_embedding, get_model
from utils.centralized_prompts import (
    INTENT_CLASSIFICATION_PROMPT,
    HYDE_PROMPT,
    KEYWORD_EXTRACTION_PROMPT,
    AMBIGUITY_DETECTION_PROMPT,
    QUERY_EXPANSION_PROMPT,
    INTERATIVE_REFINEMENT_PROMPT,
    GET_ANSWER_GENERATION_PROMPT,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_rag_query.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DEBUG_PRINTS = True # More debugging

# Caching
_hyde_cache = {} 
_keywords_cache = {}

# --- CONFIG DEFAULTS ---
ES_URL = "http://localhost:9200"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"
#OLLAMA_MODEL = "qwen2.5:14b"

# Models
EMBED_MODEL = "intfloat/multilingual-e5-large"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval params
BM25_K = 2000
KNN_K = 2000
RERANK_TOP_N = 200
FINAL_TOP_K = 20

# Model temeprature control
TOP_P = 0.1 # Nucleus sampling (lower for more focus)
TOP_K = 10  # Consider top tokens (lower for less variety)
### Method temperature control 
TEMP_INTENT_CLASSIFICATION = 0.0
TEMP_AMBIGUITY_CHECK = 0.0
TEMP_HYDE = 0.3
TEMP_TECH_KEYWORDS = 0.0
TEMP_EXPAND_QUERY = 0.2
TEMP_INTERATIVE_RETRIEVAL = 0.1
TEMP_GENERATE_CONTEXTUAL_ANSWER = 0.2

# =========================================================
# OLLAMA INTEGRATION
# =========================================================
# Ollama is a local LLM server (like running ChatGPT on your own machine).
# We use it for query understanding and answer generation.

def ask_ollama(prompt: str, temperature: float) -> str:
    """
    Enhanced Ollama call with temperature control.
    
    Temperature guide:
    - 0.0-0.2: Highly deterministic (classification, extraction, factual)
    - 0.3-0.5: Mostly consistent (query expansion, technical writing)
    - 0.6-0.8: Balanced (general responses, explanations)
    - 0.9-1.0: Creative (brainstorming, varied outputs)
    
    Args:
        prompt: The prompt to send
        temperature: Randomness control (0.0 = deterministic, 1.0 = creative)
        model_name: Which Ollama model to use
    
    Returns:
        LLM response text
    """
    body = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": TOP_P,
            "top_k": TOP_K,
        }
    }
    try:
        # POST request to Ollama's generate endpoint
        resp = requests.post(OLLAMA_URL, json=body, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        logger.error(f"Ollama request failed: {e}")
        return ""

# =========================================================
# QUERY INTENT CLASSIFICATION
# =========================================================
# Intent classification helps the system understand WHAT KIND of answer the user wants.
# This allows us to customize:
# - Which search strategy to use
# - How to format the answer (list vs paragraph vs step-by-step)
# - Whether to search broadly or narrowly

class QueryIntent:
    """
    Different types of questions users ask.
    
    Why this matters:
    - FACTUAL: Search for definitions, focus on precise information
    - PROCEDURAL: Search for instructions, format answer as steps
    - COMPARATIVE: Search for multiple items, format as comparison table
    - CLARIFICATION: Look at previous answer, expand on it
    - EXPLORATORY: Search broadly, provide overview
    - FOLLOWUP: Use conversation context to understand what "it" or "that" refers to
    - LIST: Search for multiple items, format as bullet list
    """
    FACTUAL = "factual"             # "What is Type 1 terminal?" - wants definition
    PROCEDURAL = "procedural"       # "How do I install terminals?" - wants steps
    COMPARATIVE = "comparative"     # "Type 1 vs Type 2?" - wants comparison
    CLARIFICATION = "clarification" # "What do you mean?" - expand previous answer
    EXPLORATORY = "exploratory"     # "Tell me about terminals" - wants overview
    FOLLOWUP = "followup"           # "What about the other one?" - references previous topic
    LIST = "list"                   # "List all types of..." - wants enumerated list


def classify_query_intent(query: str, conversation: ConversationHistory) -> Tuple[str, float]:
    """
    Use an LLM to determine what TYPE of question the user is asking.
    
    This is the first step in handling a query. Understanding intent allows us to:
    1. Choose the right search strategy (broad vs narrow)
    2. Format the answer appropriately (list vs paragraph vs steps)
    3. Decide if we need more context from the user
    
    Example:
    Input: "What are Type 1 terminals?"
    Output: ("factual", 0.95) - high confidence it's asking for a definition
    
    Input: "what about the other one?" (after discussing Type 1)
    Output: ("followup", 0.85) - needs conversation context
    
    Args:
        query: The user's question
        conversation: Conversation history for context
        ollama_model: Which LLM to use for classification
        
    Returns:
        (intent_type, confidence_score) - e.g., ("factual", 0.9)
    """
    
    # Get recent conversation to help with intent detection
    # Example: "And Type 2?" only makes sense with previous context
    recent_context = conversation.get_recent_context(2) if conversation.messages else ""
    
    # Build prompt for the LLM to classify intent
    prompt = INTENT_CLASSIFICATION_PROMPT.format(
        recent_context=recent_context if recent_context else "No previous conversation",
        query=query
    )

    try:
        response = ask_ollama(prompt,TEMP_INTENT_CLASSIFICATION)
        if DEBUG_PRINTS:
            print("-"*40)
            print(f"\nclassify_query_intent: \n{response}\n")
            print("-"*40)
    
        intent = QueryIntent.FACTUAL
        confidence = 0.5
        
        for line in response.lower().split('\n'):
            line = line.strip()
            if line.startswith('intent:'):
                parsed_intent = line.split(':', 1)[1].strip()
                # Validate intent
                valid_intents = [QueryIntent.FACTUAL, QueryIntent.PROCEDURAL, 
                               QueryIntent.COMPARATIVE, QueryIntent.CLARIFICATION,
                               QueryIntent.EXPLORATORY, QueryIntent.FOLLOWUP, QueryIntent.LIST]
                if parsed_intent in valid_intents:
                    intent = parsed_intent
            elif line.startswith('confidence:'):
                try:
                    confidence = float(line.split(':', 1)[1].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except:
                    pass
        
        logger.info(f"Query intent: {intent} (confidence: {confidence:.2f})")
        return intent, confidence
        
    except Exception as e:
        logger.warning(f"Intent classification failed: {e}")
        return QueryIntent.FACTUAL, 0.5

# =========================================================
# AMBIGUITY DETECTION
# =========================================================
def detect_ambiguity(query: str, conversation: ConversationHistory) -> Tuple[bool, Optional[str]]:
    """Detect if query is ambiguous and generate clarification question"""
    
    recent_context = conversation.get_recent_context(2)
    entities = conversation.get_entities_string()
    prompt = AMBIGUITY_DETECTION_PROMPT.format(
        recent_context=recent_context if recent_context else "None",
        entities=entities,
        query=query
    )

    try:
        response = ask_ollama(prompt, TEMP_AMBIGUITY_CHECK)
        if DEBUG_PRINTS:
            print("-"*40)
            print(f"\ndetect_ambiguity: \n{response}\n")
            print("-"*40)

        is_ambiguous = False
        clarification = None
        
        for line in response.split('\n'):
            line = line.strip()
            if line.lower().startswith('ambiguous:'):
                is_ambiguous = 'yes' in line.lower()
            elif line.lower().startswith('clarification:'):
                clarification = line.split(':', 1)[1].strip()
                if clarification.lower() in ['none', 'n/a', '']:
                    clarification = None
        
        if is_ambiguous and clarification:
            logger.info(f"Ambiguity detected: {clarification}")
            return True, clarification
        
    except Exception as e:
        logger.warning(f"Ambiguity detection failed: {e}")
    
    return False, None

# =========================================================
# QUERY EXPANSION AND TRANSFORMATION
# =========================================================
# These functions transform user queries into better search queries.
# Problem: Users ask questions in natural language ("what terminals can I use?")
# but documents use technical language ("Type 1 terminal blocks shall...")
# Solution: Generate technical-sounding text and search for THAT.

def generate_hypothetical_answer(query: str) -> str:
    """
    HyDE (Hypothetical Document Embeddings) - A clever trick to improve search.
    
    THE PROBLEM:
    User asks: "what terminals can I use?"
    Documents say: "Type 1 terminal blocks incorporating screw clamp devices..."
    
    These have different words, so keyword search fails.
    Even semantic search struggles because questions and answers have different structure.
    
    THE SOLUTION:
    1. Generate a FAKE answer that sounds like a specification document
    2. Convert that fake answer to an embedding
    3. Search for documents with similar embeddings
    
    This works because spec documents are more similar to other spec documents
    than they are to questions!
    
    Example:
    Query: "what terminals should I use?"
    HyDE generates: "Terminal blocks shall be Type 1 incorporating screw clamp..."
    Search finds: Real documents that actually say this!
    
    Args:
        query: User's question
        ollama_model: LLM to generate fake answer
        
    Returns:
        Hypothetical document excerpt (fake but realistic-sounding)
    """
    if query in _hyde_cache:
        return _hyde_cache[query]

    # Prompt the LLM to write like a technical specification
    # Centralized HYDE_PROMPT expects only the query parameter
    prompt = HYDE_PROMPT.format(query=query)
    
    try:
        response = ask_ollama(prompt, TEMP_HYDE)
        if DEBUG_PRINTS:
            print("-"*40)
            print(f"\ngenerate_hypothetical_answer: \n{response}\n")
            print("-"*40)

        if response and len(response) > 20:
            logger.info(f"HyDE generated: {response[:80]}...")
            _hyde_cache[query] = response.strip()
            return response.strip()
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}")
    return ""


def extract_technical_keywords(query: str) -> List[str]:
    """
    Extract specific technical terms from a natural language question.
    
    THE PROBLEM:
    User asks: "what can I use for connecting wires in control panels?"
    Key terms in specs: "terminal blocks", "Type 1", "screw clamp"
    
    THE SOLUTION:
    Use LLM to translate natural language into technical jargon.
    
    This is different from HyDE because it extracts KEYWORDS rather than
    generating full sentences. Keywords are used for BM25 (keyword) search.
    
    Example:
    Input: "how do I connect cables in panels?"
    Output: ["terminal blocks", "cable glands", "DIN rail", "screw terminals"]
    
    Then we search for docs containing these exact terms.
    
    Args:
        query: User's question
        ollama_model: LLM for extraction
        
    Returns:
        List of technical keywords/phrases (up to 10)
    """
    
    # Check cache first
    if query in _keywords_cache:
        return _keywords_cache[query]

    # Prompt LLM to think like a technical writer and extract relevant terms
    prompt = KEYWORD_EXTRACTION_PROMPT.format(query=query)
    
    try:
        response = ask_ollama(prompt, TEMP_TECH_KEYWORDS)
        if DEBUG_PRINTS:
            print("-"*40)
            print(f"\nextract_technical_keywords: \n{response}\n")
            print("-"*40)
        
        # Patterns to filter out (preamble/explanatory text)
        skip_patterns = [
            'here are', 'extracted keywords', 'keywords:', 'searchable',
            'the following', 'i found', 'i extracted', 'based on'
        ]
        
        keywords = []
        for line in response.split('\n'):
            line = line.strip().lstrip('•-*123456789.) ')
            # Skip empty, too short, too long
            if not line or len(line) <= 2 or len(line) >= 50:
                continue
            # Skip preamble/explanatory lines
            if any(pattern in line.lower() for pattern in skip_patterns):
                continue
            keywords.append(line)
        
        result = keywords[:10]
        if result:
            logger.info(f"Extracted keywords: {result[:8]}")
            _keywords_cache[query] = result  # Cache result
        return result
    except Exception as e:
        logger.warning(f"Keyword extraction failed: {e}")
        return []


def expand_query_with_context(query: str, conversation: ConversationHistory,
                             intent: str) -> str:
    """
    Rewrite ambiguous queries to be more specific using conversation history.
    
    THE PROBLEM:
    Users often ask followup questions that are ambiguous:
    - "What about Type 2?" (Type 2 what? Terminals?)
    - "Tell me more" (More about what?)
    - "Where can I use it?" (Use what where?)
    
    THE SOLUTION:
    Look at recent conversation to understand context:
    - Previous topics mentioned: "terminal blocks", "screw clamps"
    - Recent discussion about Type 1 terminals
    - Rewrite: "What about Type 2?" → "What are Type 2 terminal blocks?"
    
    EXAMPLE CONVERSATION:
    User: "What are Type 1 terminals?"
    Assistant: "Type 1 terminals are screw clamp devices..."
    User: "What about Type 2?"  ← Ambiguous!
    Expanded: "What are Type 2 terminal blocks specifications and requirements?"
    
    Args:
        query: Original user query (might be ambiguous)
        conversation: History with previous questions and topics
        intent: Query type (factual, procedural, etc.)
        ollama_model: LLM to use for expansion
        
    Returns:
        Expanded, more specific query (or original if already specific)
    """
    
    # Get recent conversation to understand context
    recent_context = conversation.get_recent_context(2)  # Last 2 exchanges
    # Get technical terms mentioned previously
    entities = conversation.get_entities_string()
    
    # Build prompt for LLM to expand the query
    prompt = QUERY_EXPANSION_PROMPT.format(
        recent_context=recent_context if recent_context else "No previous conversation",
        entities=entities,
        query=query,
        intent =intent
    )
    
    try:
        # Ask LLM to rewrite the query
        response = ask_ollama(prompt,TEMP_EXPAND_QUERY)
        if DEBUG_PRINTS:
            print("-"*40)
            print(f"\nexpand_query_with_context: \n{response}\n")
            print("-"*40)

        expanded = response.strip().split('\n')[0]  # Take first line only
        # Only use expanded query if it's reasonable (not empty or too short)
        if expanded and len(expanded) > 10:
            logger.info(f"Expanded query: {query} → {expanded[:80]}...")
            return expanded
    except Exception as e:
        logger.warning(f"Context expansion failed: {e}")
    
    # If expansion fails or produces bad output, use original query
    return query


# =========================================================
# HYBRID SEARCH
# =========================================================
# Hybrid search combines two different search methods:
# 1. BM25 (keyword matching) - finds exact word matches
# 2. Vector/Semantic search - finds similar meanings
# Combining both gives better results than either alone!

def hybrid_search(es: Elasticsearch, index_name: str, query: str) -> List[Dict]:
    """
    Search documents using both keyword matching AND semantic similarity.
    
    WHY HYBRID SEARCH?
    
    BM25 (keyword) is good at:
    - Finding exact terms: "Type 1" will match documents with "Type 1"
    - Technical codes: "BS EN 60529" matches exactly
    
    Vector (semantic) is good at:
    - Finding similar concepts: "terminals" matches "terminal blocks"
    - Understanding context: "wire connectors" finds "terminal blocks"
    
    But each has weaknesses:
    - BM25 misses synonyms: "terminals" won't match "connectors"
    - Vector misses exact codes: "Type 1" might match "Type 2" or "Class 1"
    
    SOLUTION: Use both, combine scores!
    
    Args:
        es: Elasticsearch client
        index_name: Which index to search
        query: Search query text
        embed_model_name: Embedding model for vector search
        bm25_k: How many results from keyword search (default 20)
        knn_k: How many results from vector search (default 30)
        
    Returns:
        List of documents sorted by combined score
    """
    
    # 1. BM25 KEYWORD SEARCH - finds exact word matches
    # This uses Elasticsearch's standard full-text search (tf-idf based)
    try:
        bm25_response = es.search(
            index=index_name,
            query={"match": {"chunk_text": query}},
            size=BM25_K,
            source=["chunk_text", "file_name", "page_number", "chunk_index", 
                    "doc_title", "section_header", "continues_from_previous", 
                    "continues_to_next"]
        )
        bm25_hits = bm25_response.get("hits", {}).get("hits", [])
    except Exception as e:
        logger.warning(f"BM25 search failed: {e}")
        bm25_hits = []
    
    # 2. VECTOR/SEMANTIC SEARCH - finds similar meanings
    # Convert query to embedding (list of numbers representing meaning)
    q_emb = get_embedding(query, EMBED_MODEL)
    
    # kNN = k-Nearest Neighbors - find documents with closest embeddings
    try:
        import requests
        
        knn_body = {
            "size": KNN_K,
            "knn": {
                "field": "embedding",
                "query_vector": q_emb,
                "k": KNN_K,
                "num_candidates": min(10000, max(3000, KNN_K * 20))  # ES only accepts 10000 max!!
            },
            "_source": ["chunk_text", "file_name", "page_number", "chunk_index", 
                    "doc_title", "section_header", "continues_from_previous", 
                    "continues_to_next"]
        }
        
        # Direct HTTP request to Elasticsearch
        response = requests.post(
            f"{ES_URL}/{index_name}/_search",
            json=knn_body,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        knn_response = response.json()
        knn_hits = knn_response.get("hits", {}).get("hits", [])
        logger.info(f"KNN search returned {len(knn_hits)} results")
        
    except Exception as e:
        logger.error(f"KNN search failed: {e}")
        if hasattr(e, 'response'):
            logger.error(f"Response body: {e.response.text if hasattr(e.response, 'text') else 'N/A'}")
        knn_hits = []
    
    # 3. COMBINE RESULTS FROM BOTH SEARCHES
    # Key insight: Documents that appear in BOTH searches are likely the best matches!
    combined = {}
    
    # Add all BM25 results
    for h in bm25_hits:
        combined[h["_id"]] = {
            "id": h["_id"],
            "chunk_text": h["_source"]["chunk_text"],
            "file_name": h["_source"].get("file_name"),
            "page_number": h["_source"].get("page_number"),
            "chunk_index": h["_source"].get("chunk_index", 0),
            "doc_title": h["_source"].get("doc_title"),
            "section_header": h["_source"].get("section_header"),
            "continues_from_previous": h["_source"].get("continues_from_previous", False),
            "continues_to_next": h["_source"].get("continues_to_next", False),
            "score": float(h.get("_score", 0.0))
        }
    
    # Add vector search results
    for h in knn_hits:
        if h["_id"] in combined:
            combined[h["_id"]]["score"] += float(h.get("_score", 0.0))
        else:
            combined[h["_id"]] = {
                "id": h["_id"],
                "chunk_text": h["_source"]["chunk_text"],
                "file_name": h["_source"].get("file_name"),
                "page_number": h["_source"].get("page_number"),
                "chunk_index": h["_source"].get("chunk_index", 0),
                "doc_title": h["_source"].get("doc_title"),
                "section_header": h["_source"].get("section_header"),
                "continues_from_previous": h["_source"].get("continues_from_previous", False),
                "continues_to_next": h["_source"].get("continues_to_next", False),
                "score": float(h.get("_score", 0.0))
            }
    
    # Sort by combined score (highest first) and return
    return sorted(combined.values(), key=lambda x: x["score"], reverse=True)

# =========================================================
# RERANKING
# =========================================================
# After initial retrieval gets ~20 candidate documents, we need to pick the best 5.
# Reranking uses a more sophisticated (but slower) model to score each candidate.

def rerank_results(query: str, chunks: List[Dict],
                   rerank_model_name: str) -> List[Dict]:
    """
    Re-score candidates using a cross-encoder for more accurate ranking.
    
    WHY RERANK?
    
    Initial retrieval (BM25 + vector) is FAST but APPROXIMATE:
    - Vector search: Compares query embedding to ALL document embeddings (~1ms each)
    - Must be fast to search 35,000 documents
    - Uses approximate similarity (cosine distance)
    
    Reranking is SLOW but ACCURATE:
    - Cross-encoder: Directly compares query to each document (~100ms each)
    - Only used on top 20 candidates (manageable)
    - Considers interaction between query and document words
    
    Example:
    Query: "Type 1 terminal blocks for control panels"
    
    Doc A: "Type 1 terminal blocks shall be used for control circuits"
    Doc B: "Type 1 splice enclosures for fiber optic cables"
    
    Vector search might score these similarly (both have "Type 1").
    Cross-encoder understands Doc A is about terminals AND control, Doc B is not.
    
    Args:
        query: Original user question
        chunks: Candidate documents (typically 20) to rerank
        rerank_model_name: Cross-encoder model to use
        top_k: How many final results to return (typically 5)
        
    Returns:
        Top K documents sorted by reranking score
    """
    if not chunks:
        return []
    
    # Limit chunks to avoid memory issues with large batches
    max_rerank = min(len(chunks), RERANK_TOP_N)
    chunks = chunks[:max_rerank]

    try:
        # Load the cross-encoder model
        _, reranker = get_model("", rerank_model_name)
        
        # Create (query, document) pairs for the model
        # Cross-encoder takes BOTH texts and computes relevance directly
        pairs = [(query, c['chunk_text']) for c in chunks]
        
        # Get relevance scores for all pairs
        # Returns array of scores, higher = more relevant
        scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
        
        # Normalize scores before combining
        search_scores = np.array([c['score'] for c in chunks])
        rerank_scores = np.array(scores)

        # Min-max normalize both to 0-1 range
        search_norm = (search_scores - search_scores.min()) / (search_scores.max() - search_scores.min() + 1e-6)
        rerank_norm = (rerank_scores - rerank_scores.min()) / (rerank_scores.max() - rerank_scores.min() + 1e-6)

        # Combine original search score with reranking score
        for i, chunk in enumerate(chunks):
            chunk['rerank_score'] = float(scores[i])  # Store reranking score
            # Final score = original score + (rerank_score × 2)
            # We multiply by 2 because rerank scores are typically -10 to +10
            # while search scores are 0-50, so we need to scale them to similar ranges
            chunk['final_score'] = search_norm[i] * 0.4 + rerank_norm[i] * 0.6
        
        # Sort by final combined score and return top K
        return sorted(chunks, key=lambda x: x['final_score'], reverse=True)[:FINAL_TOP_K]
    except Exception as e:
        logger.warning(f"Reranking failed: {e}")
        # If reranking fails, just return top K from original search
        return chunks[:FINAL_TOP_K]


# =========================================================
# SMART RETRIEVAL PIPELINE
# =========================================================
# This combines multiple search strategies for maximum recall:
# 1. Direct query search (user's exact words)
# 2. HyDE search (search with fake technical answer)
# 3. Keyword search (extracted technical terms)
# Documents appearing in multiple searches get boosted!

def smart_retrieve(es: Elasticsearch, index_name: str, query: str,
                    use_hyde: bool = False, use_keywords: bool = True) -> List[Dict]:
    """
    "Smart" retrieval that uses multiple complementary search strategies.
    
    THE STRATEGY:
    Instead of just searching with the user's exact query, we:
    1. Generate a technical-sounding fake answer (HyDE) and search for docs like it
    2. Extract technical keywords and search for those
    3. Search with the original query
    4. Combine all results, boosting docs that appear multiple times
    
    WHY THIS WORKS:
    Different search strategies find different relevant docs:
    - Original query: Finds docs matching the user's vocabulary
    - HyDE: Finds docs that "answer" in technical language
    - Keywords: Finds docs with specific technical terms
    
    Documents that appear in 2+ searches are likely highly relevant!
    
    Example:
    Query: "what terminals can I use?"
    - Original query finds: General terminal docs
    - HyDE finds: Spec docs about terminal requirements
    - Keywords find: Docs with "terminal blocks", "Type 1", etc.
    - Docs appearing in all 3 get highest scores!
    
    Args:
        es: Elasticsearch client
        index_name: Index to search
        query: User's question
        embed_model_name: Embedding model for vector search
        rerank_model_name: Cross-encoder for reranking
        ollama_model: LLM for HyDE and keyword extraction
        top_k: Number of final results
        use_hyde: Enable HyDE search (slower but better)
        use_keywords: Enable keyword extraction search
        
    Returns:
        Top K documents after reranking
    """
    
    # Dictionary to accumulate all results (key = document ID)
    all_results = {}
    
    # STRATEGY 1: HyDE - Generate fake answer and search for similar docs
    if use_hyde:
        hyde_text = generate_hypothetical_answer(query)
        if hyde_text:
            logger.info("Searching with HyDE embedding...")
            # Search with the fake technical answer instead of the question
            hyde_results = hybrid_search(es, index_name, hyde_text)
            for r in hyde_results:
                chunk_id = r['id']
                if chunk_id in all_results:
                    # Already found by another search - boost it!
                    all_results[chunk_id]['score'] += r['score'] * 1.5
                else:
                    # New document from HyDE search
                    all_results[chunk_id] = r
                    # Boost HyDE results by 1.5x (they're usually high quality)
                    all_results[chunk_id]['score'] *= 1.5
    
    # STRATEGY 2: Extract technical keywords and search for those
    if use_keywords:
        keywords = extract_technical_keywords(query)
        if keywords:
            # Combine top 5 keywords into a search query
            keyword_query = " ".join(keywords[:5])
            logger.info(f"Searching with keywords: {keyword_query}")
            # Search with extracted keywords (e.g., "terminal blocks Type 1 screw clamp")
            kw_results = hybrid_search(es, index_name, keyword_query)
            for r in kw_results:
                chunk_id = r['id']
                if chunk_id in all_results:
                    # Doc already found by HyDE - add to score
                    all_results[chunk_id]['score'] += r['score']
                else:
                    # New document from keyword search
                    all_results[chunk_id] = r
    
    # STRATEGY 3: Direct search with original user query
    # This ensures we don't miss obvious matches to the user's exact words
    direct_results = hybrid_search(es, index_name, query)
    for r in direct_results:
        chunk_id = r['id']
        if chunk_id in all_results:
            # Doc found by multiple strategies - accumulate scores
            # High accumulated score = likely very relevant!
            all_results[chunk_id]['score'] += r['score']
        else:
            # New document from direct search
            all_results[chunk_id] = r
    
    # FINAL STEP: Sort by accumulated score and rerank top candidates
    candidates = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)[:RERANK_TOP_N]
    
    # NEW: Fetch adjacent pages and continuations BEFORE reranking
    if candidates:
        # Add chunks from adjacent pages (catches Type 3 when Type 1/2 found)
        candidates = fetch_adjacent_pages(es, index_name, candidates)
        
        # Add chunks that continue from/to found chunks
        candidates = fetch_continuing_chunks(es, index_name, candidates)
        
        # Re-sort after adding new candidates
        candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:RERANK_TOP_N]
        
        # Rerank ALL candidates (including adjacent) together
        final_results = rerank_results(query, candidates, RERANK_MODEL)
    else:
        final_results = []
    
    return final_results


# =========================================================
# ITERATIVE RETRIEVAL
# =========================================================
# Sometimes the first search doesn't find enough good results.
# Solution: Look at what we DID find, generate a refined query, and search again!
# This is like how humans research - if first search fails, try different terms.

def iterative_retrieval(es: Elasticsearch, index_name: str, query: str,
                        max_iterations: int) -> Tuple[List[Dict], Dict, str]:
    """
    Search iteratively, refining the query based on what we find.
    
    THE IDEA:
    1. Do initial search with user's query
    2. Look at what documents we found
    3. Generate a refined query based on those documents
    4. Search again with refined query
    5. Combine all results
    
    WHY THIS HELPS:
    
    Example scenario:
    User asks: "what connectors should I use?"
    
    Iteration 1:
    - Search for "connectors"
    - Find docs about "terminal blocks" and "cable glands"
    
    Iteration 2:
    - Refined query: "terminal blocks cable glands specifications"
    - Find MORE specific docs about terminal block requirements
    
    The key insight: Documents we find contain the RIGHT TERMINOLOGY!
    We extract that terminology and search again.
    
    Args:
        es: Elasticsearch client
        index_name: Index to search
        query: Original user query
        conversation: Conversation history
        embed_model_name: Embedding model
        rerank_model_name: Reranker model
        ollama_model: LLM for query refinement
        max_iterations: Maximum search iterations (default 2)
        top_k: Final number of results to return
        
    Returns:
        (results, metadata) - Results from all iterations + iteration details
    """
    
    # Accumulate results from all iterations
    all_results = {}
    collected_keywords = []  
    metadata = {"iterations": []}  # Track what happened in each iteration
    
    # Start with original query
    current_query = query
    
    # ITERATION LOOP - search, analyze results, refine, repeat
    for iteration in range(max_iterations):
        logger.info(f"Retrieval iteration {iteration + 1}/{max_iterations}")
        
        # Search with current query (either original or refined)
        results = smart_retrieve(
            es, index_name, current_query,
            use_hyde=False,
            use_keywords=True
        )

        iter_keywords = extract_technical_keywords(current_query)
        collected_keywords.extend(iter_keywords)

        # Merge results from this iteration with previous results
        for r in results:
            chunk_id = r['id']
            if chunk_id not in all_results:
                # New document - add it
                all_results[chunk_id] = r
                # Track which iteration found this doc
                all_results[chunk_id]['found_iteration'] = iteration + 1
        
        # Record metadata about this iteration
        metadata["iterations"].append({
            "iteration": iteration + 1,
            "query": current_query,
            "results_found": len(results),
            "keywords": iter_keywords  # Track per-iteration
        })
        
        # Early stopping: If we have enough results (2x what we need), stop iterating
        if len(all_results) >= FINAL_TOP_K * 2:
            break
        
        # QUERY REFINEMENT - generate better query based on what we found
        # Only refine if this isn't the last iteration and we found something
        if iteration < max_iterations - 1 and results:
            # Extract text from top 3 documents we found
            # These contain the RIGHT terminology!
            found_texts = [r['chunk_text'][:200] for r in results[:3]]
            
            # Ask LLM to generate refined query based on found documents
            refine_prompt = INTERATIVE_REFINEMENT_PROMPT.format(
                original_query=query,
                current_query=current_query,
                found_texts=chr(10).join(f"- {text}" for text in found_texts)
            )

            try:
                # Get refined query from LLM
                refined = ask_ollama(refine_prompt, TEMP_INTERATIVE_RETRIEVAL).strip().split('\n')[0]
                if DEBUG_PRINTS:
                    print("-"*40)
                    print(f"\ninterative_retrival: \n{refined}\n")
                    print("-"*40)
                #  Only use refined query if it's different and reasonable
                if refined and refined != current_query and len(refined) > 10:
                    current_query = refined
                    logger.info(f"Refined query: {refined[:60]}...")
                else:
                    # Refinement didn't produce a new query, stop iterating
                    break
            except:
                # Error in refinement, stop iterating
                break
    
    # Store deduplicated keywords in metadata
    metadata["keywords"] = list(set(collected_keywords))
    
    final_results = sorted(all_results.values(), 
                          key=lambda x: x.get('final_score', x['score']), 
                          reverse=True)[:FINAL_TOP_K]
    
    return final_results, metadata, current_query


# =========================================================
# ANSWER GENERATION
# =========================================================
# This is where we take the retrieved documents and generate a natural
# language answer using an LLM. This is the "Generation" part of RAG.

def generate_contextual_answer(question: str, retrieved_chunks: List[Dict],
                              conversation: ConversationHistory,
                              intent: str) -> str:
    """
    Generate a natural language answer using LLM + retrieved documents.
    
    THE PROCESS:
    1. Build context from retrieved documents (max ~3000 tokens)
    2. Get recent conversation history
    3. Customize prompt based on question intent
    4. Send everything to LLM to generate answer
    
    KEY FEATURES:
    - Grounded in documents (must cite sources)
    - Conversation-aware (references previous discussion)
    - Intent-aware (formats answer appropriately)
    - Honest (admits when info is missing)
    
    Example:
    Question: "What are Type 1 terminals?"
    Retrieved docs: [spec about Type 1 terminal blocks]
    Intent: factual
    Output: "According to [Source 1: sp-net-pac-520 - Page 28], Type 1 terminal 
             blocks are screw clamp devices used for 110/230/400V AC supplies..."
    
    Args:
        question: User's question
        retrieved_chunks: Top documents from search
        conversation: Conversation history
        intent: Question type (factual, procedural, etc.)
        ollama_model: LLM to use for generation
        
    Returns:
        Generated answer with citations
    """
    
    # BUILD CONTEXT STRING FROM RETRIEVED DOCUMENTS
    context_parts = []
    seen_pages = set()  # Avoid duplicate pages
    
    for i, c in enumerate(retrieved_chunks, 1):
        # Check if we already included this page
        page_key = f"{c.get('file_name')}_{c.get('page_number')}"
        if page_key in seen_pages:
            continue  # Skip duplicates
        seen_pages.add(page_key)
        
        # Format document for LLM context
        doc_title = c.get('doc_title', c.get('file_name', 'Unknown'))
        page = c.get('page_number', '?')
        text = c.get('chunk_text', '')
        context_parts.append(f"[Source {i}: {doc_title} - Page {page}]\n{text}")
    
    # Join all documents with separator
    context = "\n\n---\n\n".join(context_parts)
    
    # GET CONVERSATION CONTEXT
    # Include recent messages so LLM can reference previous discussion
    recent_context = conversation.get_recent_context(2)
    
    # CUSTOMIZE INSTRUCTIONS BASED ON INTENT
    # Different question types need different answer formats:
    intent_instructions = {
        # Factual: Just give the facts, cite sources
        QueryIntent.FACTUAL: "Provide a clear, factual answer with specific details from the documents.",
        # Procedural: Format as numbered steps if possible
        QueryIntent.PROCEDURAL: "Provide step-by-step instructions if available, or describe the procedure.",
        # Comparative: Use tables or side-by-side format
        QueryIntent.COMPARATIVE: "Compare the options clearly in a structured way, highlighting key differences.",
        # Clarification: Expand on what was just said
        QueryIntent.CLARIFICATION: "Clarify and expand on the previous answer with more detail or examples.",
        # Exploratory: Give broad overview, multiple aspects
        QueryIntent.EXPLORATORY: "Provide a comprehensive overview covering multiple aspects of the topic.",
        # Followup: Build on previous answer, add new info
        QueryIntent.FOLLOWUP: "Build on the previous discussion and provide additional relevant details.",
        # List: Format as bullet points or numbered list
        QueryIntent.LIST: "Present the information as a clear, organized list."
    }
    
    # Choose instruction based on intent (using local mapping)
    instruction = intent_instructions.get(intent, "Provide a clear and complete answer.")
    
    # BUILD THE FINAL PROMPT FOR THE LLM
    # This prompt contains:
    # 1. System message (you are a technical assistant...)
    # 2. Conversation history (what we discussed before)
    # 3. Current question
    # 4. Instructions (how to format the answer)
    # 5. Rules (must cite sources, be honest about gaps, etc.)
    # 6. Documentation context (the retrieved chunks)
    # Centralized GET_ANSWER_GENERATION_PROMPT expects conversation history, question, instruction, context
    prompt = GET_ANSWER_GENERATION_PROMPT.format(
        conversation_history=recent_context if recent_context else "No previous conversation",
        question=question,
        instruction=instruction,
        context=context
    )
    
    try:
        # Send prompt to LLM and get generated answer
        answer = ask_ollama(prompt, TEMP_GENERATE_CONTEXTUAL_ANSWER)
        if DEBUG_PRINTS:
            print("-"*40)
            print(f"\ngenerate_contextual_answer:\nprompt:\n{prompt}\nanswer:\n{answer}\n")
            print("-"*40)
        return answer.strip()
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return "I encountered an error generating the answer. Please try again."


# =========================================================
# MAIN QUERY PIPELINE
# =========================================================
# This is the main orchestration function that ties everything together.
# It coordinates all the agents (intent classifier, retriever, generator, etc.)

def advanced_query_and_answer(es: Elasticsearch, index_name: str, question: str,
                             conversation: ConversationHistory,
                             check_ambiguity: bool = True) -> Dict:
    """
    Main pipeline: Process question through all stages to generate final answer.
    
    THE FULL PIPELINE:
    
    FULL INTELLIGENT MODE:
       Step 1: Classify intent (factual? procedural? followup?)
       Step 2: Check ambiguity (if intent unclear, ask for clarification)
       Step 3: Expand query with context (rewrite using conversation history)
       Step 4: Retrieve documents:
               Option A: Iterative retrieval (search, refine, search again)
               Option B: Smart retrieval (HyDE + keywords + direct)
       Step 5: Generate answer (LLM + documents + conversation context)
       Step 6: Update conversation history
    
    Args:
        es: Elasticsearch client
        index_name: Index to search
        question: User's question
        conversation: Conversation history
        embed_model_name: Embedding model
        rerank_model_name: Reranker model
        ollama_model: LLM for expansion/generation
        top_k: Number of sources to return
        check_ambiguity: Enable ambiguity detection
        use_iterative: Enable iterative retrieval
        
    Returns:
        Dict with answer, sources, query_time, and metadata
    """
    
    start_time = time.time()  # Track total query time
    
    # ==================== FULL INTELLIGENT MODE ====================
    
    # STEP 1: CLASSIFY INTENT
    # Understand what type of question this is (factual, procedural, etc.)
    intent, confidence = classify_query_intent(question, conversation)
    
    # STEP 2: CHECK FOR AMBIGUITY
    # Only check if: (1) ambiguity detection enabled, (2) low confidence, (3) no conversation context
    # If query is ambiguous, ask user for clarification instead of guessing
    if check_ambiguity and confidence < 0.6 and not conversation.messages:
        is_ambiguous, clarification = detect_ambiguity(question, conversation)
        if is_ambiguous and clarification:
            # Return clarification question instead of answer
            return {
                "answer": None,
                "clarification_needed": clarification,
                "sources": [],
                "query_time": time.time() - start_time,
                "metadata": {"intent": intent, "confidence": confidence, "status": "needs_clarification"}
            }
    
    # STEP 3: EXPAND QUERY WITH CONTEXT
    # Rewrite query to be more specific using conversation history and intent
    expanded_query = expand_query_with_context(question, conversation, intent)
    
    # STEP 4: RETRIEVE DOCUMENTS
    # Iterative: Search, analyze results, refine query, search again
    retrieved, retrieval_metadata, final_query = iterative_retrieval(
        es, index_name, expanded_query,
        max_iterations=1
    )

    # STEP 5: GENERATE ANSWER
    if not retrieved:
        # No documents found - return helpful message
        answer = "I couldn't find relevant documents to answer your question. Could you rephrase or provide more specific details about what you're looking for?"
    else:
        # Generate answer using LLM + retrieved documents + conversation context
        answer = generate_contextual_answer(
            final_query, retrieved, conversation, intent
        )
    
    # Calculate total time
    query_time = time.time() - start_time
    
    # PREPARE SOURCES FOR DISPLAY
    # Format document sources with previews for user
    sources = [
        {
            "file": c.get('file_name'),
            "page": c.get('page_number'),
            "title": c.get('doc_title'),
            "score": round(c.get('final_score', c.get('score', 0)), 3),
            "preview": c.get('chunk_text', '')[:150] + "..."  # First 150 chars
        }
        for c in retrieved
    ]
    
    # BUILD RESULT DICTIONARY
    result = {
        "answer": answer,
        "sources": sources,
        "query_time": query_time,
        "metadata": {
            "intent": intent,
            "confidence": confidence,
            "expanded_query": expanded_query,
            "retrieval": retrieval_metadata  # Details about search iterations
        }
    }
    
    # STEP 6: UPDATE CONVERSATION HISTORY
    # Extract and store keywords from this question for future reference
    #keywords = extract_technical_keywords(question) if retrieved else []
    keywords = retrieval_metadata.get('keywords', [])
    if not keywords and retrieved:
        keywords = extract_technical_keywords(question)

    conversation.add_entities(keywords)  # Track technical terms mentioned
    # Add user question to history
    conversation.add_message("user", question, metadata={"keywords": keywords, "intent": intent})
    # Add assistant answer to history
    conversation.add_message("assistant", answer, sources=sources)
    
    logger.info(f"✓ Query completed in {query_time:.2f}s with {len(retrieved)} sources")
    
    return result

def fetch_adjacent_pages(es: Elasticsearch, index_name: str, 
                         chunks: List[Dict], max_adjacent: int = 3) -> List[Dict]:
    """
    Fetch chunks from adjacent pages (page-1, page+1) of found documents.
    
    This catches content that spans pages (e.g., Type 3 on page 29 when Type 1/2 on page 28).
    
    Args:
        es: Elasticsearch client
        index_name: Index to search
        chunks: Already retrieved chunks
        max_adjacent: Max adjacent chunks to add per found chunk
        
    Returns:
        Original chunks + adjacent page chunks (deduplicated)
    """
    if not chunks:
        return chunks
    
    # Collect unique (file_name, page_number) pairs from results
    found_pages = set()
    for c in chunks:
        file_name = c.get('file_name')
        page = c.get('page_number')
        if file_name and page is not None:
            found_pages.add((file_name, page))
    
    if not found_pages:
        return chunks
    
    # Build query for adjacent pages (page-1 and page+1)
    should_clauses = []
    for file_name, page in found_pages:
        # Page before
        if page > 1:
            should_clauses.append({
                "bool": {
                    "must": [
                        {"term": {"file_name": file_name}},
                        {"term": {"page_number": page - 1}}
                    ]
                }
            })
        # Page after
        should_clauses.append({
            "bool": {
                "must": [
                    {"term": {"file_name": file_name}},
                    {"term": {"page_number": page + 1}}
                ]
            }
        })
    
    if not should_clauses:
        return chunks
    
    try:
        # Fetch adjacent pages
        response = es.search(
            index=index_name,
            query={"bool": {"should": should_clauses, "minimum_should_match": 1}},
            size=len(found_pages) * max_adjacent * 2,  # Estimate max results
            source=["chunk_text", "file_name", "page_number", "chunk_index", 
                    "doc_title", "section_header", "continues_from_previous", 
                    "continues_to_next"]
        )
        
        adjacent_hits = response.get("hits", {}).get("hits", [])
        logger.info(f"Found {len(adjacent_hits)} adjacent page chunks")
        
        # Merge with original chunks (deduplicate by ID)
        existing_ids = {c['id'] for c in chunks}
        
        for hit in adjacent_hits:
            if hit["_id"] not in existing_ids:
                chunks.append({
                    "id": hit["_id"],
                    "chunk_text": hit["_source"]["chunk_text"],
                    "file_name": hit["_source"].get("file_name"),
                    "page_number": hit["_source"].get("page_number"),
                    "chunk_index": hit["_source"].get("chunk_index", 0),
                    "doc_title": hit["_source"].get("doc_title"),
                    "section_header": hit["_source"].get("section_header"),
                    "continues_from_previous": hit["_source"].get("continues_from_previous", False),
                    "continues_to_next": hit["_source"].get("continues_to_next", False),
                    "score": 0.1,
                    "source": "adjacent_page"
                })
                existing_ids.add(hit["_id"])
        
        return chunks
        
    except Exception as e:
        logger.warning(f"Adjacent page fetch failed: {e}")
        return chunks


def fetch_continuing_chunks(es: Elasticsearch, index_name: str,
                            chunks: List[Dict]) -> List[Dict]:
    """
    Fetch chunks that continue from/to found chunks using the 
    continues_from_previous and continues_to_next flags.
    
    This is more precise than page-based adjacency.
    """
    if not chunks:
        return chunks
    
    # Find chunks that have continuation flags
    continuing_queries = []
    for c in chunks:
        file_name = c.get('file_name')
        page = c.get('page_number')
        chunk_idx = c.get('chunk_index', 0)
        
        if not file_name:
            continue
        
        # If this chunk continues to next, fetch next chunk
        if c.get('continues_to_next'):
            continuing_queries.append({
                "bool": {
                    "must": [
                        {"term": {"file_name": file_name}},
                        {"term": {"page_number": page}},
                        {"term": {"chunk_index": chunk_idx + 1}}
                    ]
                }
            })
        
        # If this chunk continues from previous, fetch previous chunk
        if c.get('continues_from_previous'):
            if chunk_idx > 0:
                continuing_queries.append({
                    "bool": {
                        "must": [
                            {"term": {"file_name": file_name}},
                            {"term": {"page_number": page}},
                            {"term": {"chunk_index": chunk_idx - 1}}
                        ]
                    }
                })
    
    if not continuing_queries:
        return chunks
    
    try:
        response = es.search(
            index=index_name,
            query={"bool": {"should": continuing_queries, "minimum_should_match": 1}},
            size=len(continuing_queries),
            source=["chunk_text", "file_name", "page_number", "chunk_index",  # Added chunk_index
                    "doc_title", "section_header", "continues_from_previous", 
                    "continues_to_next"]
        )
        
        existing_ids = {c['id'] for c in chunks}
        
        for hit in response.get("hits", {}).get("hits", []):
            if hit["_id"] not in existing_ids:
                chunks.append({
                    "id": hit["_id"],
                    "chunk_text": hit["_source"]["chunk_text"],
                    "file_name": hit["_source"].get("file_name"),
                    "page_number": hit["_source"].get("page_number"),
                    "chunk_index": hit["_source"].get("chunk_index", 0),
                    "doc_title": hit["_source"].get("doc_title"),
                    "section_header": hit["_source"].get("section_header"),
                    "continues_from_previous": hit["_source"].get("continues_from_previous", False),
                    "continues_to_next": hit["_source"].get("continues_to_next", False),
                    "score": 0.15,
                    "source": "continuation"
                })
                existing_ids.add(hit["_id"])
        
        return chunks
        
    except Exception as e:
        logger.warning(f"Continuation fetch failed: {e}")
        return chunks

# =========================================================
# DISPLAY UTILITIES
# =========================================================
def display_result(result: Dict, verbose: bool = False):
    """Pretty print query results"""
    
    print("\n" + "="*70)
    
    if result.get('clarification_needed'):
        print("🤔 CLARIFICATION NEEDED:")
        print("="*70)
        print(result['clarification_needed'])
        print("="*70)
        return
    
    print("ANSWER:")
    print("="*70)
    print(result['answer'])
    
    print("\n" + "="*70)
    print("SOURCES:")
    print("="*70)
    
    for i, src in enumerate(result['sources'], 1):
        title = src.get('title') or src.get('file', 'Unknown')
        page = src.get('page', '?')
        score = src.get('score', 0)
        print(f"\n[{i}] {title} - Page {page}")
        print(f"    Score: {score}")
        print(f"    Preview: {src.get('preview', 'N/A')}")
    
    print("\n" + "="*70)
    metadata = result.get('metadata', {})
    print(f"Query time: {result.get('query_time', 0):.2f}s")
    if 'intent' in metadata:
        print(f"Intent: {metadata['intent']} (confidence: {metadata.get('confidence', 0):.2f})")
    if verbose and 'expanded_query' in metadata:
        print(f"Expanded query: {metadata['expanded_query']}")
    print("="*70)


# =========================================================
# INTERACTIVE MODE
# =========================================================
def interactive_mode(es: Elasticsearch, index_name: str, args) -> None:
    """Interactive mode with conversation memory"""
    
    conversation = ConversationHistory()
    
    print("\n" + "="*70)
    print("🤖 MULTI-AGENT RAG INTERACTIVE MODE")
    print("="*70)
    print("\nFeatures:")
    print("  • Conversation memory - Remembers context across questions")
    print("  • Intent detection - Understands question type")
    print("  • Smart retrieval - HyDE + keywords + iterative refinement")
    print("  • Ambiguity handling - Asks for clarification when needed")
    print("\nCommands:")
    print("  'quit' or 'exit'  - Exit the program")
    print("  'history'         - Show conversation history")
    print("  'clear'           - Clear conversation history")
    print("="*70)
    
    while True:
        try:
            print()
            question = input("💬 You: ").strip()
            
            if not question:
                continue
            
            # Commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if question.lower() == 'history':
                print("\n📜 Conversation History:")
                print("-"*40)
                if not conversation.messages:
                    print("No messages yet.")
                else:
                    for msg in conversation.messages:
                        role = "You" if msg.role == "user" else "Assistant"
                        content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                        print(f"{role}: {content}\n")
                if conversation.extracted_entities:
                    print(f"Tracked topics: {', '.join(conversation.extracted_entities[-10:])}")
                continue
            
            if question.lower() in ['clear', 'reset']:
                conversation.clear()
                print("🗑️ Conversation history cleared.")
                continue
            
            # EXECUTE QUERY THROUGH FULL PIPELINE
            # This runs all agents: intent detection, query expansion, retrieval, generation
            result = advanced_query_and_answer(
                es, index_name, question, conversation,
                check_ambiguity= True
            )
            
            # HANDLE CLARIFICATION REQUEST
            # If system needs clarification, ask user and continue loop
            if result.get('clarification_needed'):
                print(f"\n🤔 {result['clarification_needed']}")
                continue  # Wait for clarifying question
            
            # DISPLAY ANSWER
            print(f"\n🤖 Assistant: {result['answer']}")
            
            # DISPLAY SOURCES (compact format for interactive mode)
            if result['sources']:
                print(f"\n📚 Sources ({len(result['sources'])}):")
                # Show top 3 sources
                for i, src in enumerate(result['sources'][:3], 1):
                    title = src.get('title') or src.get('file', 'Unknown')
                    print(f"  [{i}] {title} - Page {src.get('page', '?')} (score: {src['score']})")
            
            # DISPLAY TIMING AND INTENT
            metadata = result.get('metadata', {})
            intent_str = f" | Intent: {metadata.get('intent', '?')}" if 'intent' in metadata else ""
            print(f"\n⏱️ {result.get('query_time', 0):.2f}s{intent_str}")
            
        except KeyboardInterrupt:
            # User pressed Ctrl+C
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            # Handle any errors gracefully
            logger.error(f"Error: {e}")
            print(f"\n❌ Error: {e}")


# =========================================================
# ARGUMENT PARSING
# =========================================================
# Parse command line arguments for configuring the system

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Usage examples:
    # Single question:
    python multi_rag_queries.py /path/to/pdfs "what are Type 1 terminals?"
    
    # Interactive mode:
    python multi_rag_queries.py /path/to/pdfs --interactive
    
    # Verbose output (show metadata):
    python multi_rag_queries.py /path/to/pdfs "question" --verbose
    """
    p = argparse.ArgumentParser(description="Multi-Agent RAG Query System")
    
    # Required arguments
    p.add_argument("folder_path", help="Path to folder (used for index name)")
    p.add_argument("query", nargs='?', help="Question to ask (optional if using --interactive)")
    
    # Mode options
    p.add_argument("--interactive", "-i", action="store_true", help="Interactive mode with conversation memory")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose output (show metadata)")
    
    return p.parse_args()


def get_index_name(folder_path: str) -> str:
    """
    Generate Elasticsearch index name from folder path.
    
    Elasticsearch index names must:
    - Be lowercase
    - Not start with special characters
    - Use underscores instead of spaces
    
    Example:
    /home/user/My PDFs/ → my_pdfs
    /data/Engineering-Docs/ → engineering_docs
    
    Args:
        folder_path: Path to PDF folder
        explicit_name: Optional explicit index name (overrides auto-generation)
        
    Returns:
        Valid Elasticsearch index name
    """    
    # Get just the folder name (not full path)
    base = os.path.basename(os.path.normpath(folder_path))
    
    # Clean up: lowercase, replace spaces/hyphens with underscores
    cleaned = base.lower().replace(" ", "_").replace("-", "_")
    
    # ES index names can't start with special chars - add prefix if needed
    if cleaned.startswith(("_", "-", "+")):
        cleaned = "idx_" + cleaned
    
    return cleaned


# =========================================================
# MAIN
# =========================================================
# Entry point - handles setup, validation, and mode selection

def main() -> None:
    """
    Main entry point for the multi-agent RAG query system.
    
    STARTUP SEQUENCE:
    1. Parse command line arguments
    2. Validate inputs (query or interactive mode required)
    3. Determine index name from folder path
    4. Connect to Elasticsearch
    5. Verify index exists
    6. Preload ML models (embedding and reranker)
    7. Run in either interactive or single-query mode
    """
    args = parse_args()
    
    # VALIDATE ARGUMENTS
    # Must provide either a query or use interactive mode
    if not args.query and not args.interactive:
        print("Error: Provide a query or use --interactive mode")
        print("Usage: python multi_rag_queries.py /path/to/folder \"your question\"")
        print("       python multi_rag_queries.py /path/to/folder --interactive")
        sys.exit(1)
    
    # DETERMINE INDEX NAME
    # Generate from folder path or use explicit override
    index_name = get_index_name(args.folder_path)
    logger.info(f"Using index: {index_name}")
    
    # CONNECT TO ELASTICSEARCH
    es = Elasticsearch(ES_URL, request_timeout=60)
    try:
        info = es.info()
        logger.info(f"✓ Connected to Elasticsearch: {info['version']['number']}")
    except Exception as e:
        logger.error(f"Failed to connect to Elasticsearch: {e}")
        logger.error(f"Make sure Elasticsearch is running at {ES_URL}")
        sys.exit(1)
    
    # VERIFY INDEX EXISTS
    # User must run the indexer first to create the index
    if not es.indices.exists(index=index_name):
        logger.error(f"Index '{index_name}' not found. Run the indexer first.")
        logger.error(f"Run: python rag-indexer-ppocrv5-embed-separated.py {args.folder_path}")
        sys.exit(1)
    
    # PRELOAD MODELS INTO MEMORY
    # This takes a few seconds but makes first query faster
    logger.info("Preloading models...")
    get_model(EMBED_MODEL, RERANK_MODEL)
    logger.info("✓ Models ready")
    
    # RUN IN SELECTED MODE
    if args.interactive:
        # Interactive mode - chat interface with conversation memory
        interactive_mode(es, index_name, args)
    else:
        # Single query mode - ask one question and exit
        conversation = ConversationHistory()  # Empty conversation (no history)
        
        # Execute query through full pipeline
        result = advanced_query_and_answer(
            es, index_name, args.query, conversation,
            check_ambiguity=True
        )
        
        # Display result and exit
        display_result(result, verbose=args.verbose)


if __name__ == "__main__":
    main()
