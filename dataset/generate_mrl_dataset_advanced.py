#!/usr/bin/env python3
"""
Advanced MRL Dataset Generator with LLM-based Synthetic Query Generation

This script generates higher-quality training data by using an LLM to create
realistic user queries for each chunk, then pairing queries with relevant/irrelevant chunks.

IMPROVEMENTS IN THIS VERSION:
1. Semantic validation using embedding similarity (not just keywords)
2. Query quality validation - ensures query matches source chunk
3. Better label assignment based on actual semantic similarity
4. Hard negative mining with semantic similarity thresholds
5. Removes bad pairs that would hurt training

PAIR TYPES:
1. Query-to-answer (label 1.0): Generated query → source chunk (verified semantic match)
2. Query-to-related (label 0.6-0.8): Query → semantically related chunk
3. Query-to-hard-negative (label 0.0): Query → semantically similar but wrong answer
4. Query-to-negative (label 0.0): Query → unrelated chunk

USAGE:
    # Basic usage
    python generate_mrl_dataset_advanced.py input.json output.json
    
    # Generate fewer pairs but higher quality
    python generate_mrl_dataset_advanced.py input.json output.json --pairs 5000 --queries-per-chunk 3
    
    # Use different Ollama model
    python generate_mrl_dataset_advanced.py input.json output.json --model mistral
"""

import json
import random
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import re
import numpy as np

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

# Embedding model for semantic validation
EMBED_MODEL = None
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Semantic similarity thresholds
MIN_POSITIVE_SIMILARITY = 0.65  # Minimum similarity for positive pairs
MAX_NEGATIVE_SIMILARITY = 0.35  # Maximum similarity for negative pairs
HARD_NEGATIVE_RANGE = (0.35, 0.55)  # Range for hard negatives


def load_embedding_model():
    """Load embedding model for semantic validation."""
    global EMBED_MODEL
    if EMBED_MODEL is None:
        print(f"Loading embedding model: {EMBED_MODEL_NAME}...")
        from sentence_transformers import SentenceTransformer
        import torch
        EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
        if torch.cuda.is_available():
            EMBED_MODEL = EMBED_MODEL.to('cuda')
            print("  ✓ Using GPU for embeddings")
    return EMBED_MODEL


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity between two texts using embeddings.
    Returns cosine similarity in range [0, 1].
    """
    model = load_embedding_model()
    
    # Encode both texts
    embeddings = model.encode([text1, text2], normalize_embeddings=True)
    
    # Compute cosine similarity
    similarity = np.dot(embeddings[0], embeddings[1])
    return float(similarity)


def compute_batch_similarities(query: str, chunks: List[str]) -> List[float]:
    """
    Efficiently compute similarity between one query and multiple chunks.
    """
    model = load_embedding_model()
    
    # Encode query and all chunks together
    all_texts = [query] + chunks
    embeddings = model.encode(all_texts, normalize_embeddings=True)
    
    query_emb = embeddings[0]
    chunk_embs = embeddings[1:]
    
    # Compute similarities
    similarities = [float(np.dot(query_emb, chunk_emb)) for chunk_emb in chunk_embs]
    return similarities


def ask_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Query Ollama LLM for text generation."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            print(f"ERROR: Ollama returned status {response.status_code}")
            return ""
    except Exception as e:
        print(f"ERROR calling Ollama: {e}")
        return ""


def generate_queries_for_chunk(chunk_text: str, file_name: str, 
                               n_queries: int = 3, model: str = OLLAMA_MODEL) -> List[str]:
    """
    Generate realistic user queries that this chunk would answer.
    
    IMPROVED: Better prompt to generate queries that are actually answerable.
    """
    # Truncate chunk to key content
    chunk_preview = chunk_text[:1200] if len(chunk_text) > 1200 else chunk_text
    
    prompt = f"""You are creating training data for a technical document search system.

Given this document excerpt, generate {n_queries} questions that someone would ask and THIS TEXT would directly answer.

IMPORTANT RULES:
1. Each question MUST be answerable by the text below
2. Questions should use different words than the text (paraphrasing)
3. Mix question types: what, how, where, when, why, which
4. Be specific - avoid vague questions like "tell me about..."
5. Questions should be 10-30 words long

Document: {file_name}

Text excerpt:
\"\"\"
{chunk_preview}
\"\"\"

Generate exactly {n_queries} questions (one per line, no numbering):"""

    response = ask_ollama(prompt, model)
    
    if not response:
        return []
    
    # Parse questions from response
    lines = response.strip().split('\n')
    questions = []
    for line in lines:
        # Remove numbering, bullets, etc.
        line = re.sub(r'^[\d\.\-\*\)\]]+\s*', '', line.strip())
        line = line.strip('"\'')
        
        # Quality checks
        if not line or len(line) < 15:
            continue
        if '?' not in line:
            continue
        if len(line) > 200:  # Too long
            continue
        if line.lower().startswith(('here', 'sure', 'of course', 'certainly')):
            continue  # Skip LLM preambles
            
        questions.append(line)
    
    return questions[:n_queries]


def validate_query_chunk_pair(query: str, chunk_text: str, 
                              min_similarity: float = MIN_POSITIVE_SIMILARITY) -> Tuple[bool, float]:
    """
    Validate that a query-chunk pair has sufficient semantic similarity.
    
    Returns (is_valid, similarity_score)
    """
    similarity = compute_semantic_similarity(query, chunk_text)
    is_valid = similarity >= min_similarity
    return is_valid, similarity


def load_chunks(json_path: str) -> List[Dict]:
    """Load chunks from the OCR JSON file."""
    print(f"Loading chunks from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    chunks = data.get('chunks', [])
    print(f"Loaded {len(chunks)} chunks from {data.get('total_pdfs', 'unknown')} documents")
    return chunks


def filter_chunks(chunks: List[Dict], min_words: int = 30) -> List[Dict]:
    """Filter out chunks that are too short or low quality."""
    filtered = []
    for c in chunks:
        word_count = c.get('word_count', 0)
        text = c.get('chunk_text', '')
        
        # Basic length check
        if word_count < min_words:
            continue
        
        # Skip chunks that are mostly numbers/tables
        alpha_ratio = sum(1 for char in text if char.isalpha()) / max(len(text), 1)
        if alpha_ratio < 0.5:
            continue
        
        # Skip chunks with too many special characters (likely OCR noise)
        special_ratio = sum(1 for char in text if not char.isalnum() and not char.isspace()) / max(len(text), 1)
        if special_ratio > 0.2:
            continue
            
        filtered.append(c)
    
    print(f"Filtered to {len(filtered)} quality chunks (min {min_words} words)")
    return filtered


def find_semantic_neighbors(query: str, source_chunk_id: str, 
                           all_chunks: List[Dict], n_chunks: int = 20) -> List[Tuple[Dict, float]]:
    """
    Find chunks semantically similar to a query using embeddings.
    
    Returns list of (chunk, similarity_score) tuples, excluding source chunk.
    """
    # Get chunk texts (excluding source)
    candidate_chunks = [c for c in all_chunks if c.get('chunk_id') != source_chunk_id]
    
    if not candidate_chunks:
        return []
    
    # Sample if too many chunks (for efficiency)
    if len(candidate_chunks) > 500:
        candidate_chunks = random.sample(candidate_chunks, 500)
    
    chunk_texts = [c['chunk_text'] for c in candidate_chunks]
    
    # Compute similarities
    similarities = compute_batch_similarities(query, chunk_texts)
    
    # Pair chunks with similarities and sort
    scored_chunks = list(zip(candidate_chunks, similarities))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    return scored_chunks[:n_chunks]


def generate_query_pairs(chunks: List[Dict], n_pairs: int, 
                        queries_per_chunk: int, model: str) -> List[Dict]:
    """
    Generate high-quality query-to-chunk pairs with semantic validation.
    
    KEY IMPROVEMENTS:
    1. Validates query-chunk semantic similarity before accepting
    2. Uses embedding similarity for finding related/similar chunks
    3. Labels based on actual semantic similarity, not just pair type
    4. Creates proper hard negatives (similar topic, wrong answer)
    """
    print(f"\nGenerating semantically-validated query pairs...")
    print(f"  Using model: {model}")
    print(f"  Queries per chunk: {queries_per_chunk}")
    print(f"  Min positive similarity: {MIN_POSITIVE_SIMILARITY}")
    print(f"  Max negative similarity: {MAX_NEGATIVE_SIMILARITY}")
    
    # Pre-load embedding model
    load_embedding_model()
    
    pairs = []
    stats = {
        'chunks_processed': 0,
        'queries_generated': 0,
        'queries_validated': 0,
        'queries_rejected': 0,
        'positives': 0,
        'related': 0,
        'hard_negatives': 0,
        'negatives': 0
    }
    
    chunks_to_process = min(n_pairs // (queries_per_chunk * 2), len(chunks))
    
    # Group chunks by document
    doc_groups = defaultdict(list)
    for chunk in chunks:
        doc_groups[chunk.get('file_name', 'unknown')].append(chunk)
    
    # Shuffle chunks for variety
    shuffled_chunks = random.sample(chunks, len(chunks))
    
    for chunk in shuffled_chunks:
        if len(pairs) >= n_pairs:
            break
        
        stats['chunks_processed'] += 1
        if stats['chunks_processed'] % 20 == 0:
            print(f"  Processed {stats['chunks_processed']}/{chunks_to_process} chunks, "
                  f"generated {len(pairs)} pairs (validated: {stats['queries_validated']}, "
                  f"rejected: {stats['queries_rejected']})...")
        
        file_name = chunk.get('file_name', chunk.get('file_name', 'Unknown'))
        chunk_text = chunk['chunk_text']
        chunk_id = chunk.get('chunk_id', str(random.randint(0, 1000000)))
        
        # Generate queries for this chunk
        queries = generate_queries_for_chunk(chunk_text, file_name, queries_per_chunk, model)
        stats['queries_generated'] += len(queries)
        
        if not queries:
            continue
        
        for query in queries:
            # CRITICAL: Validate query-chunk semantic match
            is_valid, similarity = validate_query_chunk_pair(query, chunk_text)
            
            if not is_valid:
                stats['queries_rejected'] += 1
                continue  # Skip low-quality pairs!
            
            stats['queries_validated'] += 1
            
            # 1. POSITIVE PAIR: Query → Source chunk
            # Label based on actual similarity (0.8-1.0 range for positives)
            positive_label = min(1.0, 0.8 + (similarity - MIN_POSITIVE_SIMILARITY) * 0.5)
            pairs.append({
                'anchor': query,
                'positive': chunk_text,
                'label': round(positive_label, 2),
                'pair_type': 'query_to_answer',
                'semantic_similarity': round(similarity, 3),
                'doc': chunk.get('file_name', 'unknown'),
                'page': chunk.get('page_number', 0)
            })
            stats['positives'] += 1
            
            # Find semantically similar chunks for related/hard-negative pairs
            neighbors = find_semantic_neighbors(query, chunk_id, chunks, n_chunks=15)
            
            # 2. RELATED PAIRS: High similarity chunks from SAME document
            same_doc_neighbors = [
                (c, sim) for c, sim in neighbors 
                if c.get('file_name') == chunk.get('file_name') and sim > 0.5
            ]
            
            if same_doc_neighbors:
                related_chunk, related_sim = same_doc_neighbors[0]
                # Label based on actual similarity
                related_label = round(0.5 + related_sim * 0.3, 2)  # 0.5-0.8 range
                pairs.append({
                    'anchor': query,
                    'positive': related_chunk['chunk_text'],
                    'label': related_label,
                    'pair_type': 'query_to_related',
                    'semantic_similarity': round(related_sim, 3),
                    'doc': related_chunk.get('file_name', 'unknown'),
                    'page': related_chunk.get('page_number', 0)
                })
                stats['related'] += 1
            
            # 3. HARD NEGATIVES: Similar topic but DIFFERENT document (wrong answer)
            # These are crucial for training - model learns to distinguish similar content
            hard_neg_candidates = [
                (c, sim) for c, sim in neighbors 
                if c.get('file_name') != chunk.get('file_name') 
                and HARD_NEGATIVE_RANGE[0] <= sim <= HARD_NEGATIVE_RANGE[1]
            ]
            
            if hard_neg_candidates:
                hard_neg_chunk, hard_neg_sim = random.choice(hard_neg_candidates[:5])
                pairs.append({
                    'anchor': query,
                    'positive': hard_neg_chunk['chunk_text'],
                    'label': 0.0,  # It's a NEGATIVE despite being similar
                    'pair_type': 'query_to_hard_negative',
                    'semantic_similarity': round(hard_neg_sim, 3),
                    'doc': hard_neg_chunk.get('file_name', 'unknown'),
                    'page': hard_neg_chunk.get('page_number', 0)
                })
                stats['hard_negatives'] += 1
            
            # 4. EASY NEGATIVES: Low similarity, clearly wrong
            easy_neg_candidates = [
                (c, sim) for c, sim in neighbors[-10:]  # Take lowest similarity
                if sim < MAX_NEGATIVE_SIMILARITY
            ]
            
            if not easy_neg_candidates:
                # Fallback: random chunk
                random_chunk = random.choice([c for c in chunks if c.get('chunk_id') != chunk_id])
                easy_neg_candidates = [(random_chunk, 0.1)]
            
            if easy_neg_candidates:
                neg_chunk, neg_sim = random.choice(easy_neg_candidates)
                pairs.append({
                    'anchor': query,
                    'positive': neg_chunk['chunk_text'],
                    'label': 0.0,
                    'pair_type': 'query_to_negative',
                    'semantic_similarity': round(neg_sim, 3),
                    'doc': neg_chunk.get('file_name', 'unknown'),
                    'page': neg_chunk.get('page_number', 0)
                })
                stats['negatives'] += 1
    
    # Print generation statistics
    print(f"\n  Generation Statistics:")
    print(f"    Chunks processed: {stats['chunks_processed']}")
    print(f"    Queries generated: {stats['queries_generated']}")
    print(f"    Queries validated: {stats['queries_validated']} ({stats['queries_validated']/max(stats['queries_generated'],1)*100:.1f}%)")
    print(f"    Queries rejected: {stats['queries_rejected']} ({stats['queries_rejected']/max(stats['queries_generated'],1)*100:.1f}%)")
    print(f"    Positives: {stats['positives']}")
    print(f"    Related: {stats['related']}")
    print(f"    Hard negatives: {stats['hard_negatives']}")
    print(f"    Easy negatives: {stats['negatives']}")
    
    print(f"\nGenerated {len(pairs)} semantically-validated pairs")
    return pairs


def save_dataset(pairs: List[Dict], output_path: str, include_metadata: bool = True):
    """Save the dataset to a JSON file."""
    print(f"\nSaving dataset to {output_path}...")
    
    if not include_metadata:
        clean_pairs = []
        for pair in pairs:
            clean_pairs.append({
                'anchor': pair['anchor'],
                'positive': pair['positive'],
                'label': pair['label']
            })
        pairs = clean_pairs
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(pairs)} pairs to {output_path}")


def print_statistics(pairs: List[Dict]):
    """Print statistics about the generated dataset."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    total = len(pairs)
    print(f"Total pairs: {total}")
    
    by_type = defaultdict(int)
    by_label = defaultdict(list)
    similarities_by_type = defaultdict(list)
    
    for pair in pairs:
        pair_type = pair.get('pair_type', 'unknown')
        by_type[pair_type] += 1
        label = pair['label']
        by_label[label].append(pair)
        
        # Track semantic similarities by type
        if 'semantic_similarity' in pair:
            similarities_by_type[pair_type].append(pair['semantic_similarity'])
    
    print("\nPairs by type:")
    for pair_type, count in sorted(by_type.items()):
        avg_sim = np.mean(similarities_by_type[pair_type]) if similarities_by_type[pair_type] else 0
        print(f"  {pair_type}: {count} ({count/total*100:.1f}%) - avg similarity: {avg_sim:.3f}")
    
    print("\nLabel distribution:")
    unique_labels = sorted(set(p['label'] for p in pairs), reverse=True)
    for label in unique_labels:
        count = len([p for p in pairs if p['label'] == label])
        print(f"  {label}: {count} ({count/total*100:.1f}%)")
    
    # Show example pairs
    print("\nExample pairs:")
    for pair_type in ['query_to_answer', 'query_to_related', 'query_to_hard_negative', 'query_to_negative']:
        examples = [p for p in pairs if p.get('pair_type') == pair_type]
        if examples:
            example = examples[0]
            sim = example.get('semantic_similarity', 'N/A')
            print(f"\n{pair_type} (label: {example['label']}, similarity: {sim}):")
            print(f"  Query: {example['anchor'][:100]}...")
            print(f"  Chunk: {example['positive'][:100]}...")
    
    print("="*60)


def main():
    global OLLAMA_URL, OLLAMA_MODEL
    parser = argparse.ArgumentParser(
        description='Generate advanced MRL dataset with LLM-generated queries',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('input_json', help='Input JSON file with OCR chunks')
    parser.add_argument('output_json', help='Output JSON file for training data')
    parser.add_argument('--pairs', type=int, default=10000,
                       help='Total number of pairs to generate (default: 10000)')
    parser.add_argument('--queries-per-chunk', type=int, default=3,
                       help='Queries to generate per chunk (default: 3)')
    parser.add_argument('--min-words', type=int, default=30,
                       help='Minimum words per chunk (default: 30)')
    parser.add_argument('--model', type=str, default=OLLAMA_MODEL,
                       help=f'Ollama model to use (default: {OLLAMA_MODEL})')
    parser.add_argument('--ollama-url', type=str, default=OLLAMA_URL,
                       help=f'Ollama API URL (default: {OLLAMA_URL})')
    parser.add_argument('--no-metadata', action='store_true',
                       help='Remove metadata fields from output')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Update globals
    OLLAMA_URL = args.ollama_url
    OLLAMA_MODEL = args.model
    
    random.seed(args.seed)
    
    print("="*60)
    print("ADVANCED MRL DATASET GENERATOR")
    print("="*60)
    print(f"Input: {args.input_json}")
    print(f"Output: {args.output_json}")
    print(f"Target pairs: {args.pairs}")
    print(f"LLM Model: {args.model}")
    print(f"Ollama URL: {args.ollama_url}")
    
    # Test Ollama connection
    print("\nTesting Ollama connection...")
    test_response = ask_ollama("Say 'OK' if you can hear me.", args.model)
    if not test_response:
        print("ERROR: Could not connect to Ollama. Make sure it's running.")
        print(f"  URL: {args.ollama_url}")
        print(f"  Model: {args.model}")
        return
    print(f"✓ Ollama connected: {test_response[:50]}")
    
    # Load chunks
    chunks = load_chunks(args.input_json)
    chunks = filter_chunks(chunks, min_words=args.min_words)
    
    if len(chunks) < 50:
        print(f"ERROR: Not enough chunks ({len(chunks)}) to generate meaningful pairs")
        return
    
    # Generate dataset
    pairs = generate_query_pairs(chunks, args.pairs, args.queries_per_chunk, args.model)
    
    # Shuffle
    random.shuffle(pairs)
    
    # Print statistics
    print_statistics(pairs)
    
    # Save
    save_dataset(pairs, args.output_json, include_metadata=not args.no_metadata)
    
    print(f"\n✓ Dataset generation complete!")
    print(f"  Generated: {len(pairs)} pairs")
    print(f"  Output: {args.output_json}")

if __name__ == '__main__':
    main()
