#!/usr/bin/env python3
"""
Production RAG System with Hybrid Search + Two-Stage Answer Generation

USAGE:
  python integrated_rag_queries.py /path/to/pdfs "technical question"
  python integrated_rag_queries.py /path/to/pdfs --interactive
"""

import os
import sys
import argparse
import logging
import time
from typing import List, Dict, Optional
from elasticsearch import Elasticsearch

from utils.optimized_retrieval import search_documents
from utils.table_context import build_context

DEBUG = False

try:
    from utils.conversation_history import ConversationHistory
    from utils.model_loading import get_model
    from utils.centralized_prompts import ANSWER_GENERATION_DIRECT, build_table_instruction
    from utils.structured_answer_generator import generate_structured_answer, ask_ollama_safe
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIG ---
ES_URL = "http://localhost:9200"
OLLAMA_URL = "http://localhost:11434/api/generate"
#OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_MODEL = "qwen2.5:14b"

# Models - MUST match index embedding dimensions (384)
EMBED_MODEL = "all-MiniLM-L6-v2"  # 384 dims - matches your index
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

FINAL_TOP_K = 10
LLM_TIMEOUT = 60
MAX_RETRIES = 2


def query_and_answer(
    es: Elasticsearch,
    index_name: str,
    question: str,
    conversation: ConversationHistory,
    use_structured: bool = True
) -> Dict:
    """Main RAG pipeline."""
    start = time.time()
    
    embed_model, reranker = get_model(EMBED_MODEL, RERANK_MODEL)
    
    def embedding_fn(text: str) -> List[float]:
        return embed_model.encode([text], normalize_embeddings=True)[0].tolist()
    
    def ask_ollama(prompt, temperature=0.15, timeout=LLM_TIMEOUT):
        return ask_ollama_safe(prompt, OLLAMA_MODEL, temperature, timeout, OLLAMA_URL)
    
    # Search
    logger.info(f"Query: {question}")
    results = search_documents(es, index_name, question, embedding_fn, reranker, final_k=FINAL_TOP_K)
    
    if not results:
        return {
            "answer": "No relevant documents found.",
            "sources": [],
            "query_time": time.time() - start,
            "metadata": {"num_results": 0, "pipeline": "none"}
        }
    
    logger.info(f"Retrieved {len(results)} documents")
    
    # Generate answer
    if use_structured:
        try:
            res = generate_structured_answer(question, results, ask_ollama, MAX_RETRIES)
            answer, qtype, conf, valid = res["answer"], res["query_type"], res["confidence"], res["validation_passed"]
            
            # If extraction failed, fall back to direct
            if res.get("error") or answer.startswith("Extraction failed"):
                logger.warning("Structured extraction failed, falling back to direct")
                answer = _direct_answer(question, results, ask_ollama)
                qtype, conf, valid = "FALLBACK", "MEDIUM", True
            elif not valid:
                answer += "\n\n⚠️ Answer validation flagged potential issues."
        except Exception as e:
            logger.error(f"Structured pipeline failed: {e}")
            answer, qtype, conf, valid = _direct_answer(question, results, ask_ollama), "FALLBACK", "UNKNOWN", False
    else:
        answer, qtype, conf, valid = _direct_answer(question, results, ask_ollama), "DIRECT", "UNKNOWN", True
    
    # Format sources
    sources = [{
        "file": c.get('file_name'),
        "page": c.get('page_number'),
        "score": round(c.get('final_score', 0), 3),
        "content_type": c.get('content_type'),
        "has_table": c.get('has_table', False),
        "preview": c.get('chunk_text', '')[:150] + "..."
    } for c in results]
    
    conversation.add_message("user", question)
    conversation.add_message("assistant", answer, sources=sources)
    
    return {
        "answer": answer,
        "sources": sources,
        "query_time": time.time() - start,
        "metadata": {
            "num_results": len(results),
            "pipeline": "structured" if use_structured else "direct",
            "query_type": qtype,
            "confidence": conf,
            "validation_passed": valid
        }
    }


def _direct_answer(question: str, chunks: List[Dict], ask_ollama) -> str:
    """Fallback direct answer generation."""
    context = build_context(chunks)
    instructions = build_table_instruction(chunks)
    
    prompt = ANSWER_GENERATION_DIRECT.format(
        special_instructions=instructions,
        question=question,
        keywords="",
        context=context
    )
    
    try:
        return ask_ollama(prompt, 0.15, LLM_TIMEOUT).strip() or "Failed to generate answer."
    except Exception as e:
        logger.error(f"Direct answer failed: {e}")
        return f"Error: {e}"


# --- CLI ---

def interactive_mode(es: Elasticsearch, index_name: str, use_structured: bool = True):
    """Interactive chat."""
    conversation = ConversationHistory()
    print(f"\n{'='*60}\n🔧 RAG Interactive Mode\nCommands: quit, history, clear, toggle\n{'='*60}")
    
    while True:
        try:
            q = input("\n💬 You: ").strip()
            if not q:
                continue
            if q.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            if q.lower() == 'toggle':
                use_structured = not use_structured
                print(f"✓ Pipeline: {'Structured' if use_structured else 'Direct'}")
                continue
            if q.lower() == 'history':
                for m in conversation.messages[-10:]:
                    print(f"{'You' if m.role == 'user' else 'Bot'}: {m.content[:200]}...")
                continue
            if q.lower() == 'clear':
                conversation.clear()
                print("🗑️ Cleared.")
                continue
            
            result = query_and_answer(es, index_name, q, conversation, use_structured)
            print(f"\n🤖 {result['answer']}")
            
            if result['sources']:
                print(f"\n📚 Sources ({len(result['sources'])}):")
                for i, s in enumerate(result['sources'][:5], 1):
                    print(f"  [{i}] {s['file']} p.{s['page']} ({s['content_type']})")
            
            meta = result['metadata']
            print(f"\n⏱️ {result['query_time']:.2f}s | {meta['query_type']} | {meta['confidence']}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path")
    parser.add_argument("query", nargs='?')
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    
    if not args.query and not args.interactive:
        print("Error: Provide query or use --interactive")
        sys.exit(1)
    
    # Index name from folder
    index_name = os.path.basename(os.path.normpath(args.folder_path)).lower().replace(" ", "_").replace("-", "_")
    if index_name[0] in "_-+":
        index_name = "idx_" + index_name
    
    logger.info(f"Index: {index_name}")
    
    es = Elasticsearch(ES_URL, request_timeout=60)
    try:
        es.info()
        logger.info("✓ Connected to ES")
    except Exception as e:
        logger.error(f"ES connection failed: {e}")
        sys.exit(1)
    
    if not es.indices.exists(index=index_name):
        logger.error(f"Index '{index_name}' not found")
        sys.exit(1)
    
    logger.info("Loading models...")
    get_model(EMBED_MODEL, RERANK_MODEL)
    logger.info("✓ Models ready")
    
    if args.interactive:
        interactive_mode(es, index_name, not args.direct)
    else:
        result = query_and_answer(es, index_name, args.query, ConversationHistory(), not args.direct)
        print(f"\n{'='*60}\nANSWER:\n{'='*60}\n{result['answer']}")
        print(f"\n{'='*60}\nSOURCES ({len(result['sources'])}):\n{'='*60}")
        for i, s in enumerate(result['sources'], 1):
            print(f"[{i}] {s['file']} - Page {s['page']} | Score: {s['score']:.3f}")
        print(f"\n⏱️ {result['query_time']:.2f}s | {result['metadata']['query_type']} | {result['metadata']['confidence']}")


if __name__ == "__main__":
    main()