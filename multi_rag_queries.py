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
  
  # Fast mode - skip intent detection and iterative search (for simple queries)
  python multi_rag_queries.py /path/to/pdfs "question" --fast
"""

import os
import sys
import argparse
import logging
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import torch
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, CrossEncoder
import requests

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

# --- CONFIG DEFAULTS ---
ES_URL = "http://localhost:9200"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

# Models
#EMBED_MODEL = "BAAI/bge-large-en-v1.5"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval params
BM25_K = 20
KNN_K = 30
RERANK_TOP_N = 20
FINAL_TOP_K = 5

# Global model caches (singleton pattern)
_embedding_model_cache = {}
_reranker_model_cache = {}


# =========================================================
# MODEL LOADING
# =========================================================
# These functions load and cache AI models to avoid reloading them on every query.
# Models are expensive to load (takes 2-3 seconds) so we keep them in memory.

def get_embedding_model(model_name: str):
    """
    Load the embedding model that converts text into vectors (numbers).
    
    Embeddings allow semantic search - finding documents with similar *meaning*
    even if they don't use the exact same words. For example:
    "terminal blocks" and "electrical connectors" would have similar embeddings.
    
    Args:
        model_name: Name of the Sentence-BERT model (e.g., "BAAI/bge-large-en-v1.5")
    
    Returns:
        SentenceTransformer model ready to encode text
    
    Note: Uses GPU (CUDA) if available for 10x faster encoding
    """
    global _embedding_model_cache
    
    # Check if we already loaded this model (singleton pattern)
    if model_name in _embedding_model_cache:
        return _embedding_model_cache[model_name]
    
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Move to GPU and use FP16 (half precision) for faster inference
    if torch.cuda.is_available():
        model = model.to('cuda')  # Move model to GPU
        model.half()  # Use 16-bit floats instead of 32-bit (2x faster, minimal accuracy loss)
        logger.info(f"✓ Using GPU for embeddings")
    
    # Cache the model so we don't reload it
    _embedding_model_cache[model_name] = model
    return model


def get_reranker_model(model_name: str):
    """
    Load the reranker model (cross-encoder) for final result scoring.
    
    After initial retrieval finds ~20 candidates, the reranker scores each one
    more accurately by directly comparing the query to each document.
    This is slower than embedding search but more accurate.
    
    Example: Initial search might return 20 docs about "terminals",
    reranker picks the 5 that actually answer the specific question.
    
    Args:
        model_name: Name of cross-encoder model (e.g., "ms-marco-MiniLM-L-6-v2")
    
    Returns:
        CrossEncoder model for reranking
    """
    global _reranker_model_cache
    
    # Check cache first
    if model_name in _reranker_model_cache:
        return _reranker_model_cache[model_name]
    
    logger.info(f"Loading reranker model: {model_name}")
    model = CrossEncoder(model_name)
    _reranker_model_cache[model_name] = model
    return model


def get_embedding(query_text: str, model_name: str) -> List[float]:
    """
    Convert text into a vector (list of numbers) representing its meaning.
    
    This is called "embedding" - it maps text to a point in high-dimensional space.
    Similar meanings = nearby points in space.
    
    Example:
    "terminal blocks" → [0.23, -0.45, 0.67, ...] (1024 numbers)
    "electrical connectors" → [0.25, -0.43, 0.69, ...] (similar to above!)
    "coffee machine" → [-0.89, 0.12, -0.34, ...] (far from above)
    
    Args:
        query_text: Text to convert to embedding
        model_name: Which embedding model to use
        
    Returns:
        List of 1024 floats representing the text's meaning
    """
    model = get_embedding_model(model_name)
    # encode() returns numpy array, we normalize to unit length and convert to list
    emb = model.encode([query_text], normalize_embeddings=True)[0]
    return emb.tolist()


# =========================================================
# OLLAMA INTEGRATION
# =========================================================
# Ollama is a local LLM server (like running ChatGPT on your own machine).
# We use it for query understanding and answer generation.

def ask_ollama(prompt: str, model_name: str = OLLAMA_MODEL) -> str:
    """
    Send a prompt to Ollama LLM and get response.
    
    Ollama runs locally (http://localhost:11434) and provides LLM capabilities:
    - Intent classification
    - Query expansion
    - HyDE generation
    - Keyword extraction
    - Answer generation
    
    Args:
        prompt: The prompt to send to the LLM
        model_name: Which model to use (default: llama3.1:8b)
        
    Returns:
        LLM's response text (or empty string if error)
    """
    body = {"model": model_name, "prompt": prompt, "stream": False}
    try:
        # POST request to Ollama's generate endpoint
        resp = requests.post(OLLAMA_URL, json=body, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        logger.error(f"Ollama request failed: {e}")
        return ""


# =========================================================
# CONVERSATION MEMORY
# =========================================================
# These classes track conversation history so the system can handle followup
# questions like "what about Type 2?" after asking about "Type 1 terminals".
# This is what makes the system feel like ChatGPT.

@dataclass
class Message:
    """
    Represents a single message in the conversation (either user or assistant).
    
    Stores:
    - What was said
    - Who said it (user or AI)
    - When it was said
    - What documents were cited
    - Metadata (keywords, intent, etc.)
    """
    role: str  # 'user' or 'assistant'
    content: str  # The actual message text
    timestamp: datetime = field(default_factory=datetime.now)  # When this was said
    sources: Optional[List[Dict]] = None  # Documents cited in the answer
    metadata: Optional[Dict] = None  # Extra info (keywords, intent type, etc.)


class ConversationHistory:
    """
    Manages the entire conversation memory - tracks all messages and extracts context.
    
    This allows the system to:
    1. Remember what was discussed earlier
    2. Track technical terms mentioned ("terminal blocks", "Type 1", etc.)
    3. Resolve ambiguous followup questions ("what about the other type?")
    4. Build context for query expansion
    
    Example conversation:
    User: "What are Type 1 terminals?"
    [System remembers: topic=terminals, entity=Type 1]
    User: "And Type 2?" 
    [System knows to search for Type 2 terminals]
    """
    
    def __init__(self, max_messages: int = 10):
        self.messages: List[Message] = []  # All messages in conversation
        self.max_messages = max_messages  # Limit memory to prevent context overflow
        self.extracted_topics: List[str] = []  # Main topics discussed
        self.extracted_entities: List[str] = []  # Technical terms mentioned
    
    def add_message(self, role: str, content: str, sources=None, metadata=None):
        """
        Add a new message to the conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: The message text
            sources: List of documents cited (for assistant messages)
            metadata: Extra info like keywords, intent type, etc.
        """
        msg = Message(
            role=role,
            content=content,
            sources=sources,
            metadata=metadata
        )
        self.messages.append(msg)
        
        # Keep only recent messages to prevent context from getting too long
        # LLMs have token limits, so we can't send entire conversation history
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]  # Keep last N messages
    
    def add_entities(self, entities: List[str]):
        """
        Track technical terms/entities mentioned in conversation.
        
        Example: If user asks about "Type 1 terminal blocks" and "cable glands",
        we store those terms to help with followup questions.
        
        Args:
            entities: List of technical terms to remember
        """
        for entity in entities:
            if entity not in self.extracted_entities:
                self.extracted_entities.append(entity)
        # Keep last 20 entities only (prevent list from growing forever)
        self.extracted_entities = self.extracted_entities[-20:]
    
    def get_recent_context(self, n: int = 3) -> str:
        """
        Get the last N messages formatted as a string for LLM context.
        
        This is sent to the LLM when classifying intent or expanding queries,
        so it can understand what's been discussed.
        
        Args:
            n: Number of recent messages to include
            
        Returns:
            Formatted string like:
            "User: What are Type 1 terminals?
             Assistant: Type 1 terminals are..."
        """
        recent = self.messages[-n:] if self.messages else []
        context_parts = []
        for msg in recent:
            role = "User" if msg.role == "user" else "Assistant"
            # Truncate long messages to first 300 chars to save tokens
            context_parts.append(f"{role}: {msg.content[:300]}")
        return "\n".join(context_parts)
    
    def get_entities_string(self) -> str:
        """Get mentioned entities as string"""
        return ", ".join(self.extracted_entities[-10:]) if self.extracted_entities else "None"
    
    def clear(self):
        """Clear conversation history"""
        self.messages = []
        self.extracted_topics = []
        self.extracted_entities = []


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


def classify_query_intent(query: str, conversation: ConversationHistory,
                         ollama_model: str = OLLAMA_MODEL) -> Tuple[str, float]:
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
    prompt = f"""Classify the intent of this query for a technical documentation search system.

Conversation History:
{recent_context if recent_context else "No previous conversation"}

Current Query: "{query}"

Intent Types:
- factual: Asking for specific facts/definitions (e.g., "What is X?", "Define Y")
- procedural: Asking how to do something (e.g., "How do I...?", "Steps to...")
- comparative: Comparing options (e.g., "X vs Y?", "difference between...", "which is better")
- clarification: Asking to clarify previous answer (e.g., "what do you mean?", "explain more", "can you elaborate")
- exploratory: Open-ended exploration (e.g., "show me anything about...", "what can you tell me about", "tell me about")
- followup: Following up on previous topic (e.g., "what about...", "and the other one?", "more details on that")
- list: Asking for a list of items (e.g., "list all...", "what types of...", "enumerate")

Respond in EXACTLY this format (one line each):
intent: <type>
confidence: <0.0-1.0>"""

    try:
        response = ask_ollama(prompt, ollama_model)
        
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
def detect_ambiguity(query: str, conversation: ConversationHistory,
                    ollama_model: str = OLLAMA_MODEL) -> Tuple[bool, Optional[str]]:
    """Detect if query is ambiguous and generate clarification question"""
    
    recent_context = conversation.get_recent_context(2)
    entities = conversation.get_entities_string()
    
    prompt = f"""Analyze if this query is too ambiguous to search technical documentation effectively.

Previous Conversation:
{recent_context if recent_context else "None"}

Previously Mentioned Topics: {entities}

Current Query: "{query}"

A query is ambiguous if:
1. It uses vague words without context (e.g., "that thing", "the other one")
2. It could refer to multiple completely different topics
3. Critical information is missing to understand what's being asked
4. It uses pronouns without clear reference AND there's no conversation history

A query is NOT ambiguous if:
1. It's a followup and the topic is clear from conversation history
2. It uses general terms but the domain is clear (e.g., "terminals" in electrical context)
3. It's exploratory but the subject area is defined

Respond in EXACTLY this format:
ambiguous: yes/no
clarification: <question to ask user if ambiguous, otherwise "none">"""

    try:
        response = ask_ollama(prompt, ollama_model)
        
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

def generate_hypothetical_answer(query: str, ollama_model: str = OLLAMA_MODEL) -> str:
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
    
    # Prompt the LLM to write like a technical specification
    prompt = f"""You are a technical documentation expert for electrical engineering specifications.
Given this question, write a short paragraph (2-3 sentences) that might appear in a technical specification document answering this question.

Include:
- Specific technical terms (e.g., terminal blocks, enclosures, cable glands)
- Type classifications (Type 1, Type 2, Class A, Category B, etc.)
- Standards references (BS EN, IEC, IEEE)
- Specification language (shall be used, is required, must comply)

Question: {query}

Technical specification excerpt:"""
    
    try:
        response = ask_ollama(prompt, ollama_model)
        if response and len(response) > 20:
            logger.info(f"HyDE generated: {response[:80]}...")
            return response.strip()
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}")
    return ""


def extract_technical_keywords(query: str, ollama_model: str = OLLAMA_MODEL) -> List[str]:
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
    
    # Prompt LLM to think like a technical writer and extract relevant terms
    prompt = f"""Given this question about electrical/engineering specifications, list the most likely technical terms that would appear in specification documents.

Include variations like:
- Component names (terminal blocks, cable glands, enclosures, splice boxes)
- Type classifications (Type 1, Type 2, Type "1", Class A)
- Industry terms (screw clamp, spring retention, DIN rail)
- Related equipment and standards

Question: {query}

Return only keywords/phrases, one per line, no explanations or numbering:"""
    
    try:
        response = ask_ollama(prompt, ollama_model)
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


def expand_query_with_context(query: str, conversation: ConversationHistory,
                             intent: str, ollama_model: str = OLLAMA_MODEL) -> str:
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
    prompt = f"""Rewrite this query to be more specific and searchable for technical documentation.

Conversation Context:
{recent_context if recent_context else "No previous conversation"}

Previously Mentioned Topics: {entities}

Current Query: "{query}"
Query Intent: {intent}

Instructions:
- If this is a followup question, incorporate what was discussed before
- If terms are ambiguous, expand based on conversation context
- Add relevant technical synonyms
- Keep the query focused and natural
- If already specific, just return it slightly improved

Expanded Query (single line):"""
    
    try:
        # Ask LLM to rewrite the query
        response = ask_ollama(prompt, ollama_model)
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

def hybrid_search(es: Elasticsearch, index_name: str, query: str, 
                  embed_model_name: str, bm25_k: int = BM25_K, 
                  knn_k: int = KNN_K) -> List[Dict]:
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
    bm25_body = {
        "query": {"match": {"chunk_text": query}},  # Search in the text field
        "size": bm25_k,  # Return top 20 results
        "_source": ["chunk_text", "file_name", "page_number", "chunk_index", "doc_title"]  # Fields to return
    }
    bm25_response = es.search(index=index_name, body=bm25_body)
    bm25_hits = bm25_response.get("hits", {}).get("hits", [])
    
    # 2. VECTOR/SEMANTIC SEARCH - finds similar meanings
    # Convert query to embedding (list of numbers representing meaning)
    q_emb = get_embedding(query, embed_model_name)
    
    # kNN = k-Nearest Neighbors - find documents with closest embeddings
    knn_body = {
        "knn": {
            "field": "embedding",  # Search in the embedding field (created during indexing)
            "query_vector": q_emb,  # Our query as a vector of numbers
            "k": knn_k,  # Return top 30 results
            "num_candidates": max(100, knn_k * 10)  # Consider 300 docs before picking best 30 (improves quality)
        },
        "_source": ["chunk_text", "file_name", "page_number", "chunk_index", "doc_title"]
    }
    
    try:
        knn_response = es.search(index=index_name, body=knn_body)
        knn_hits = knn_response.get("hits", {}).get("hits", [])
    except Exception as e:
        logger.warning(f"KNN search failed: {e}")
        knn_hits = []
    
    # 3. COMBINE RESULTS FROM BOTH SEARCHES
    # Key insight: Documents that appear in BOTH searches are likely the best matches!
    combined = {}
    
    # Add all BM25 results
    for h in bm25_hits:
        combined[h["_id"]] = {  # Use document ID as key to prevent duplicates
            "id": h["_id"],
            "chunk_text": h["_source"]["chunk_text"],
            "file_name": h["_source"].get("file_name"),
            "page_number": h["_source"].get("page_number"),
            "doc_title": h["_source"].get("doc_title"),
            "score": float(h.get("_score", 0.0))  # Initial score from BM25
        }
    
    # Add vector search results
    for h in knn_hits:
        if h["_id"] in combined:
            # Document was in BOTH searches - ADD the scores together!
            # This boosts documents that match both keywords AND meaning
            combined[h["_id"]]["score"] += float(h.get("_score", 0.0))
        else:
            # Document only in vector search - add it
            combined[h["_id"]] = {
                "id": h["_id"],
                "chunk_text": h["_source"]["chunk_text"],
                "file_name": h["_source"].get("file_name"),
                "page_number": h["_source"].get("page_number"),
                "doc_title": h["_source"].get("doc_title"),
                "score": float(h.get("_score", 0.0))
            }
    
    # Sort by combined score (highest first) and return
    return sorted(combined.values(), key=lambda x: x["score"], reverse=True)


# =========================================================
# RERANKING
# =========================================================
# After initial retrieval gets ~20 candidate documents, we need to pick the best 5.
# Reranking uses a more sophisticated (but slower) model to score each candidate.

def rerank_results(query: str, chunks: List[Dict], rerank_model_name: str, 
                   top_k: int = FINAL_TOP_K) -> List[Dict]:
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
    
    try:
        # Load the cross-encoder model
        reranker = get_reranker_model(rerank_model_name)
        
        # Create (query, document) pairs for the model
        # Cross-encoder takes BOTH texts and computes relevance directly
        pairs = [(query, c['chunk_text']) for c in chunks]
        
        # Get relevance scores for all pairs
        # Returns array of scores, higher = more relevant
        scores = reranker.predict(pairs)
        
        # Combine original search score with reranking score
        for i, chunk in enumerate(chunks):
            chunk['rerank_score'] = float(scores[i])  # Store reranking score
            # Final score = original score + (rerank_score × 2)
            # We multiply by 2 because rerank scores are typically -10 to +10
            # while search scores are 0-50, so we need to scale them to similar ranges
            chunk['final_score'] = chunk['score'] + scores[i] * 2
        
        # Sort by final combined score and return top K
        return sorted(chunks, key=lambda x: x['final_score'], reverse=True)[:top_k]
    except Exception as e:
        logger.warning(f"Reranking failed: {e}")
        # If reranking fails, just return top K from original search
        return chunks[:top_k]


# =========================================================
# SMART RETRIEVAL PIPELINE
# =========================================================
# This combines multiple search strategies for maximum recall:
# 1. Direct query search (user's exact words)
# 2. HyDE search (search with fake technical answer)
# 3. Keyword search (extracted technical terms)
# Documents appearing in multiple searches get boosted!

def smart_retrieve(es: Elasticsearch, index_name: str, query: str,
                  embed_model_name: str, rerank_model_name: str,
                  ollama_model: str, top_k: int = FINAL_TOP_K,
                  use_hyde: bool = True, use_keywords: bool = True) -> List[Dict]:
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
        hyde_text = generate_hypothetical_answer(query, ollama_model)
        if hyde_text:
            logger.info("Searching with HyDE embedding...")
            # Search with the fake technical answer instead of the question
            hyde_results = hybrid_search(es, index_name, hyde_text, embed_model_name)
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
        keywords = extract_technical_keywords(query, ollama_model)
        if keywords:
            # Combine top 5 keywords into a search query
            keyword_query = " ".join(keywords[:5])
            logger.info(f"Searching with keywords: {keyword_query}")
            # Search with extracted keywords (e.g., "terminal blocks Type 1 screw clamp")
            kw_results = hybrid_search(es, index_name, keyword_query, embed_model_name)
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
    direct_results = hybrid_search(es, index_name, query, embed_model_name)
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
    # Sort all documents by their accumulated scores (sum from all searches)
    candidates = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)[:RERANK_TOP_N]
    
    if candidates:
        # Use cross-encoder to precisely rerank the top 20 candidates
        # This gives us the final top K (usually 5) best matches
        final_results = rerank_results(query, candidates, rerank_model_name, top_k=top_k)
    else:
        # No results found
        final_results = []
    
    return final_results


# =========================================================
# ITERATIVE RETRIEVAL
# =========================================================
# Sometimes the first search doesn't find enough good results.
# Solution: Look at what we DID find, generate a refined query, and search again!
# This is like how humans research - if first search fails, try different terms.

def iterative_retrieval(es: Elasticsearch, index_name: str, query: str,
                       conversation: ConversationHistory,
                       embed_model_name: str, rerank_model_name: str,
                       ollama_model: str, max_iterations: int = 2,
                       top_k: int = FINAL_TOP_K) -> Tuple[List[Dict], Dict]:
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
    metadata = {"iterations": []}  # Track what happened in each iteration
    
    # Start with original query
    current_query = query
    
    # ITERATION LOOP - search, analyze results, refine, repeat
    for iteration in range(max_iterations):
        logger.info(f"Retrieval iteration {iteration + 1}/{max_iterations}")
        
        # Search with current query (either original or refined)
        results = smart_retrieve(
            es, index_name, current_query,
            embed_model_name, rerank_model_name, ollama_model,
            top_k=top_k * 2,  # Get more candidates (10 instead of 5)
            use_hyde=(iteration == 0),  # Only use HyDE on first iteration (it's slow)
            use_keywords=(iteration == 0)  # Only extract keywords on first iteration
        )
        
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
            "results_found": len(results)
        })
        
        # Early stopping: If we have enough results (2x what we need), stop iterating
        if len(all_results) >= top_k * 2:
            break
        
        # QUERY REFINEMENT - generate better query based on what we found
        # Only refine if this isn't the last iteration and we found something
        if iteration < max_iterations - 1 and results:
            # Extract text from top 3 documents we found
            # These contain the RIGHT terminology!
            found_texts = [r['chunk_text'][:200] for r in results[:3]]
            
            # Ask LLM to generate refined query based on found documents
            refine_prompt = f"""Based on these document excerpts, suggest a refined search query that might find more relevant information.

Original Query: {query}
Current Query: {current_query}

Found Document Excerpts:
{chr(10).join(f"- {text}" for text in found_texts)}

Generate a single refined search query (one line only):"""
            
            try:
                # Get refined query from LLM
                refined = ask_ollama(refine_prompt, ollama_model).strip().split('\n')[0]
                # Only use refined query if it's different and reasonable
                if refined and refined != current_query and len(refined) > 10:
                    current_query = refined
                    logger.info(f"Refined query: {refined[:60]}...")
                else:
                    # Refinement didn't produce a new query, stop iterating
                    break
            except:
                # Error in refinement, stop iterating
                break
    
    # Final rerank of all collected results
    final_results = sorted(all_results.values(), 
                          key=lambda x: x.get('final_score', x['score']), 
                          reverse=True)[:top_k]
    
    return final_results, metadata


# =========================================================
# ANSWER GENERATION
# =========================================================
# This is where we take the retrieved documents and generate a natural
# language answer using an LLM. This is the "Generation" part of RAG.

def generate_contextual_answer(question: str, retrieved_chunks: List[Dict],
                              conversation: ConversationHistory,
                              intent: str, ollama_model: str = OLLAMA_MODEL) -> str:
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
    
    # Get the right instruction for this intent type
    instruction = intent_instructions.get(intent, "Provide a clear and complete answer.")
    
    # BUILD THE FINAL PROMPT FOR THE LLM
    # This prompt contains:
    # 1. System message (you are a technical assistant...)
    # 2. Conversation history (what we discussed before)
    # 3. Current question
    # 4. Instructions (how to format the answer)
    # 5. Rules (must cite sources, be honest about gaps, etc.)
    # 6. Documentation context (the retrieved chunks)
    prompt = f"""You are a helpful technical assistant answering questions about engineering specifications and documentation.

CONVERSATION HISTORY:
{recent_context if recent_context else "No previous conversation"}

CURRENT QUESTION: {question}

INSTRUCTIONS:
{instruction}

IMPORTANT:
- Base your answer ONLY on the provided documentation context
- Always cite sources using [Source X] format
- If information is incomplete or not found, acknowledge it honestly
- If this is a followup question, connect to previous discussion naturally
- Be conversational but accurate
- Use bullet points or numbered lists for clarity when appropriate

DOCUMENTATION CONTEXT:
{context}

ANSWER:"""
    
    try:
        # Send prompt to LLM and get generated answer
        answer = ask_ollama(prompt, ollama_model)
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
                             embed_model_name: str, rerank_model_name: str,
                             ollama_model: str, top_k: int = FINAL_TOP_K,
                             check_ambiguity: bool = True,
                             use_iterative: bool = True,
                             fast_mode: bool = False) -> Dict:
    """
    Main pipeline: Process question through all stages to generate final answer.
    
    THE FULL PIPELINE:
    
    1. FAST MODE PATH (if enabled):
       - Skip intent detection and query expansion
       - Do basic hybrid search + rerank
       - Generate simple answer
       - Return (saves ~5 seconds)
    
    2. FULL INTELLIGENT MODE:
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
        fast_mode: Skip advanced features for speed
        
    Returns:
        Dict with answer, sources, query_time, and metadata
    """
    
    start_time = time.time()  # Track total query time
    
    # Fast mode - minimal LLM calls
    if fast_mode:
        results = hybrid_search(es, index_name, question, embed_model_name)
        results = rerank_results(question, results[:RERANK_TOP_N], rerank_model_name, top_k)
        
        # Simple answer generation
        if results:
            context = "\n\n".join([f"[{r['doc_title']} - Page {r['page_number']}]\n{r['chunk_text']}" 
                                  for r in results[:3]])
            prompt = f"Based on this context, answer: {question}\n\nContext:\n{context}\n\nAnswer:"
            answer = ask_ollama(prompt, ollama_model)
        else:
            answer = "No relevant documents found."
        
        return {
            "answer": answer,
            "sources": [{"file": r.get('file_name'), "page": r.get('page_number'),
                        "title": r.get('doc_title'), "score": round(r.get('score', 0), 3),
                        "preview": r.get('chunk_text', '')[:150] + "..."}
                       for r in results],
            "query_time": time.time() - start_time,
            "metadata": {"mode": "fast"}
        }
    
    # ==================== FULL INTELLIGENT MODE ====================
    
    # STEP 1: CLASSIFY INTENT
    # Understand what type of question this is (factual, procedural, etc.)
    intent, confidence = classify_query_intent(question, conversation, ollama_model)
    
    # STEP 2: CHECK FOR AMBIGUITY
    # Only check if: (1) ambiguity detection enabled, (2) low confidence, (3) no conversation context
    # If query is ambiguous, ask user for clarification instead of guessing
    if check_ambiguity and confidence < 0.6 and not conversation.messages:
        is_ambiguous, clarification = detect_ambiguity(question, conversation, ollama_model)
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
    # Example: "what about Type 2?" becomes "what are Type 2 terminal blocks specifications?"
    expanded_query = expand_query_with_context(question, conversation, intent, ollama_model)
    
    # STEP 4: RETRIEVE DOCUMENTS
    # Choose between iterative (search→refine→search) or single smart retrieval
    if use_iterative:
        # Iterative: Search, analyze results, refine query, search again
        retrieved, retrieval_metadata = iterative_retrieval(
            es, index_name, expanded_query, conversation,
            embed_model_name, rerank_model_name, ollama_model,
            max_iterations=2, top_k=top_k
        )
    else:
        # Smart: HyDE + keywords + direct search in one pass
        retrieved = smart_retrieve(
            es, index_name, expanded_query,
            embed_model_name, rerank_model_name, ollama_model,
            top_k=top_k
        )
        retrieval_metadata = {}
    
    # STEP 5: GENERATE ANSWER
    if not retrieved:
        # No documents found - return helpful message
        answer = "I couldn't find relevant documents to answer your question. Could you rephrase or provide more specific details about what you're looking for?"
    else:
        # Generate answer using LLM + retrieved documents + conversation context
        answer = generate_contextual_answer(
            question, retrieved, conversation, intent, ollama_model
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
    keywords = extract_technical_keywords(question, ollama_model) if retrieved else []
    conversation.add_entities(keywords)  # Track technical terms mentioned
    # Add user question to history
    conversation.add_message("user", question, metadata={"keywords": keywords, "intent": intent})
    # Add assistant answer to history
    conversation.add_message("assistant", answer, sources=sources)
    
    logger.info(f"✓ Query completed in {query_time:.2f}s with {len(retrieved)} sources")
    
    return result


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
def interactive_mode(es: Elasticsearch, index_name: str, args):
    """Interactive mode with conversation memory"""
    
    conversation = ConversationHistory(max_messages=20)
    
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
    print("  'fast: <query>'   - Fast mode query (minimal LLM calls)")
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
            
            # Fast mode
            fast_mode = False
            if question.lower().startswith('fast:'):
                fast_mode = True
                question = question[5:].strip()
            
            # EXECUTE QUERY THROUGH FULL PIPELINE
            # This runs all agents: intent detection, query expansion, retrieval, generation
            result = advanced_query_and_answer(
                es, index_name, question, conversation,
                args.model, args.rerank_model, args.ollama_model,
                top_k=args.top_k,
                check_ambiguity=not fast_mode,  # Disable ambiguity detection in fast mode
                use_iterative=not fast_mode,  # Disable iterative retrieval in fast mode
                fast_mode=fast_mode
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

def parse_args():
    """
    Parse command line arguments.
    
    Usage examples:
    # Single question:
    python multi_rag_queries.py /path/to/pdfs "what are Type 1 terminals?"
    
    # Interactive mode:
    python multi_rag_queries.py /path/to/pdfs --interactive
    
    # Fast mode (skip advanced features):
    python multi_rag_queries.py /path/to/pdfs "quick question" --fast
    
    # Verbose output (show metadata):
    python multi_rag_queries.py /path/to/pdfs "question" --verbose
    """
    p = argparse.ArgumentParser(description="Multi-Agent RAG Query System")
    
    # Required arguments
    p.add_argument("folder_path", help="Path to folder (used for index name)")
    p.add_argument("query", nargs='?', help="Question to ask (optional if using --interactive)")
    
    # Mode options
    p.add_argument("--interactive", "-i", action="store_true", help="Interactive mode with conversation memory")
    p.add_argument("--fast", action="store_true", help="Fast mode (minimal LLM calls, ~5s faster)")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose output (show metadata)")
    
    # Model configuration
    p.add_argument("--es-url", type=str, default=ES_URL, help=f"Elasticsearch URL (default: {ES_URL})")
    p.add_argument("--model", type=str, default=EMBED_MODEL, help=f"Embedding model (default: {EMBED_MODEL})")
    p.add_argument("--rerank-model", type=str, default=RERANK_MODEL, help=f"Reranker model (default: {RERANK_MODEL})")
    p.add_argument("--ollama-model", type=str, default=OLLAMA_MODEL, help=f"Ollama LLM model (default: {OLLAMA_MODEL})")
    
    # Feature toggles
    p.add_argument("--no-ambiguity", action="store_true", help="Disable ambiguity detection")
    p.add_argument("--no-iterative", action="store_true", help="Disable iterative retrieval")
    
    # Other options
    p.add_argument("--top-k", type=int, default=FINAL_TOP_K, help=f"Number of results to return (default: {FINAL_TOP_K})")
    p.add_argument("--index-name", type=str, default=None, help="Override index name (auto-generated from folder_path by default)")
    
    return p.parse_args()


def get_index_name(folder_path, explicit_name=None):
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
    if explicit_name:
        return explicit_name
    
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

def main():
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
    index_name = get_index_name(args.folder_path, args.index_name)
    logger.info(f"Using index: {index_name}")
    
    # CONNECT TO ELASTICSEARCH
    es = Elasticsearch(args.es_url, request_timeout=60)
    try:
        info = es.info()
        logger.info(f"✓ Connected to Elasticsearch: {info['version']['number']}")
    except Exception as e:
        logger.error(f"Failed to connect to Elasticsearch: {e}")
        logger.error(f"Make sure Elasticsearch is running at {args.es_url}")
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
    get_embedding_model(args.model)  # Always load embedding model
    if not args.fast:
        get_reranker_model(args.rerank_model)  # Only load reranker if not in fast mode
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
            args.model, args.rerank_model, args.ollama_model,
            top_k=args.top_k,
            check_ambiguity=not args.no_ambiguity,
            use_iterative=not args.no_iterative,
            fast_mode=args.fast
        )
        
        # Display result and exit
        display_result(result, verbose=args.verbose)


if __name__ == "__main__":
    main()
