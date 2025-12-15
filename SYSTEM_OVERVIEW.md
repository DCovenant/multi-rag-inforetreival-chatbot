# Multi-Agent RAG System - Complete Documentation

## System Overview

This is an advanced conversational search system that lets you ask natural language questions about your technical PDFs. It handles ambiguous questions like ChatGPT does, using multiple AI agents working together.

## What Problems Does This Solve?

### Problem 1: Ambiguous Questions
**User asks:** "What terminals should I use?"
**Challenge:** Too vague - which type? For what voltage? In what context?
**Our Solution:** System detects ambiguity and asks clarifying questions

### Problem 2: Followup Questions
**User asks:** "What about Type 2?"
**Challenge:** Missing context - Type 2 what?
**Our Solution:** Conversation memory tracks what was discussed, expands query with context

### Problem 3: Technical vs Natural Language
**User asks:** "What connectors do I need?"
**Challenge:** Documents use "terminal blocks" not "connectors"
**Our Solution:** Multi-strategy search (keywords + semantic + HyDE) finds matches regardless

## Architecture: 8 AI Agents Working Together

```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUESTION                            │
│              "what terminals can I use?"                    │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  AGENT 1: INTENT CLASSIFIER                                 │
│  Determines question type:                                  │
│  • Factual (asking for information)                         │
│  • Procedural (asking for steps)                            │
│  • Comparative (comparing options)                          │
│  • Followup (referencing previous discussion)               │
│  → Result: "FACTUAL question about terminal blocks"         │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  AGENT 2: AMBIGUITY DETECTOR                                │
│  Checks if question is too vague                            │
│  • Confidence < 60%? Might be ambiguous                     │
│  • Missing key details? Ask for clarification               │
│  → Result: "Clear enough to proceed"                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  AGENT 3: QUERY EXPANDER                                    │
│  Rewrites query with conversation context                   │
│  • Looks at recent conversation history                     │
│  • Adds missing context from previous discussion            │
│  • Expands "it", "that", "those" with real entities         │
│  → Result: "what terminal blocks types and specifications   │
│             are available for electrical installations?"    │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  RETRIEVAL PIPELINE (3 parallel strategies)                 │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  AGENT 4:       │  │  AGENT 5:       │  │  DIRECT SEARCH  │
│  HYDE           │  │  KEYWORD        │  │                 │
│                 │  │  EXTRACTOR      │  │  Search with    │
│  Generate fake  │  │                 │  │  user's exact   │
│  technical ans  │  │  Extract terms: │  │  query text     │
│  then search    │  │  • "terminal"   │  │                 │
│  for similar    │  │  • "Type 1"     │  │  Uses hybrid:   │
│  docs           │  │  • "screw"      │  │  • BM25         │
│                 │  │                 │  │  • Vector       │
│  Finds docs     │  │  Search for     │  │                 │
│  that "answer"  │  │  those keywords │  │  Finds direct   │
│  in tech terms  │  │                 │  │  matches        │
└─────────┬───────┘  └─────────┬───────┘  └─────────┬───────┘
          │                    │                    │
          └────────────────────┴────────────────────┘
                               │
                               ▼
         ┌─────────────────────────────────────────┐
         │  COMBINE & SCORE                        │
         │  • Docs in multiple searches get bonus  │
         │  • HyDE results weighted 1.5x           │
         │  • Sorted by total score                │
         └────────────────┬────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  AGENT 6: RERANKER                                          │
│  Cross-encoder scores each candidate precisely              │
│  • Top 20 candidates → Cross-encoder → Top 5 best           │
│  • More accurate than initial retrieval                     │
│  → Result: 5 best matching document chunks                  │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  AGENT 7: ITERATIVE REFINER (Optional)                      │
│  If results seem incomplete, refine and search again        │
│  • Look at what we found                                    │
│  • Generate refined query using found terminology           │
│  • Search again                                             │
│  • Combine results from both iterations                     │
│  → Result: Enhanced document set                            │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  AGENT 8: ANSWER GENERATOR                                  │
│  Create natural language answer using LLM + documents       │
│  • Loads document text into context                         │
│  • Considers conversation history                           │
│  • Formats based on intent (list, steps, comparison, etc.)  │
│  • Cites sources with [Source 1], [Source 2]                │
│  → Result: "According to [Source 1: sp-net-pac-520 Page     │
│             28], Type 1 terminal blocks are screw clamp..."  │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  UPDATE CONVERSATION HISTORY                                │
│  • Store question + answer                                  │
│  • Extract and track technical entities                     │
│  • Use for next question's context                          │
└─────────────────────────────────────────────────────────────┘
```

## Key Technologies & Techniques

### 1. **HyDE (Hypothetical Document Embeddings)**
**What:** Generate a fake technical answer, then search for docs similar to the fake answer
**Why:** Documents answer in technical language. Questions ask in casual language.
**Example:**
- Query: "What terminals should I use?"
- HyDE generates: "Type 1 screw clamp terminal blocks are recommended for 110/230/400V AC supplies..."
- Search for docs similar to the fake answer
- Finds real spec documents about terminal blocks!

### 2. **Hybrid Search (BM25 + Vector)**
**What:** Combine keyword search (BM25) with semantic search (vector embeddings)
**Why:** Each has strengths and weaknesses:
- BM25: Finds exact keywords like "Type 1" but misses synonyms
- Vector: Finds similar meanings but might confuse "Type 1" with "Type 2"
**Solution:** Use BOTH, add scores together. Docs appearing in both are likely best!

### 3. **Cross-Encoder Reranking**
**What:** Precisely score each candidate document using a slower but more accurate model
**Why:** Initial retrieval is fast but approximate. Cross-encoder compares query-document pairs accurately.
**Process:**
1. Initial search returns ~50 candidates (fast but approximate)
2. Keep top 20 for reranking
3. Cross-encoder scores each one precisely (slow but accurate)
4. Return top 5 best matches

### 4. **Conversation Memory**
**What:** Track all questions, answers, and mentioned topics
**Why:** Enables followup questions:
- "What are Type 1 terminals?" → Remember we're discussing terminals
- "What about Type 2?" → System knows you mean "Type 2 terminals"
**Implementation:**
- Store messages with metadata (intent, keywords, sources)
- Extract entities (technical terms) from each question
- Build context string from recent history for query expansion

### 5. **Iterative Retrieval**
**What:** Search, look at results, refine query, search again
**Why:** Sometimes first search uses wrong terminology
**Example:**
- Search 1: "connectors" → Finds some docs about "terminal blocks"
- Extract terminology from found docs: "terminal blocks", "screw clamp"
- Search 2: "terminal blocks screw clamp" → Finds MORE specific docs!

### 6. **Intent Classification**
**What:** Determine what TYPE of question is being asked
**Types:**
- FACTUAL: "What is X?"
- PROCEDURAL: "How do I do X?"
- COMPARATIVE: "X vs Y?"
- LIST: "What are all the X?"
- EXPLORATORY: "Tell me about X"
- CLARIFICATION: "Explain more"
- FOLLOWUP: "What about that?"

**Why:** Different intents need different answer formats:
- LIST intent → Bullet points
- PROCEDURAL → Numbered steps
- COMPARATIVE → Side-by-side comparison

## File-by-File Breakdown

### `multi_rag_queries.py` (Main System)
**Purpose:** Complete conversational RAG system with all agents

**Key Functions:**
- `classify_query_intent()` - Agent 1: Detect question type
- `detect_ambiguity()` - Agent 2: Check if query is too vague
- `expand_query_with_context()` - Agent 3: Rewrite with conversation context
- `generate_hypothetical_answer()` - Agent 4: HyDE generation
- `extract_technical_keywords()` - Agent 5: Keyword extraction
- `hybrid_search()` - Combine BM25 + vector search
- `rerank_results()` - Agent 6: Cross-encoder reranking
- `smart_retrieve()` - Orchestrate HyDE + keywords + direct search
- `iterative_retrieval()` - Agent 7: Multi-pass refinement
- `generate_contextual_answer()` - Agent 8: LLM answer generation
- `advanced_query_and_answer()` - Main pipeline orchestrator
- `interactive_mode()` - Chat interface with conversation memory

**Usage:**
```bash
# Interactive mode (recommended)
python multi_rag_queries.py /path/to/pdfs --interactive

# Single question
python multi_rag_queries.py /path/to/pdfs "what terminals should I use?"

# Fast mode (skip advanced features, ~5s faster)
python multi_rag_queries.py /path/to/pdfs "question" --fast
```

### `rag_queries.py` (Simpler System)
**Purpose:** Basic smart retrieval without conversation memory

**When to use:** Simple one-off queries that don't need conversation context

**Features:**
- HyDE generation (--smart flag)
- Keyword extraction (--smart flag)
- Query expansion (--smart flag)
- NO conversation memory
- NO iterative refinement
- Faster than multi_rag_queries.py

**Usage:**
```bash
# Basic query (just hybrid search + rerank)
python rag_queries.py /path/to/pdfs "terminal block specifications"

# Smart mode (add HyDE + keywords)
python rag_queries.py /path/to/pdfs "terminal block specifications" --smart
```

### `rag-indexer-ppocrv5-embed-separated.py` (Indexer)
**Purpose:** Process PDFs with OCR and create searchable Elasticsearch index

**What it does:**
1. Runs PaddleOCR (via Docker) to extract text from PDFs
2. Chunks text into ~400 word segments with 50 word overlap
3. Generates 1024-dim embeddings using BAAI/bge-large-en-v1.5
4. Indexes to Elasticsearch with both text and embeddings
5. Uses best_compression codec to save disk space

**Usage:**
```bash
# Index all PDFs in folder
python rag-indexer-ppocrv5-embed-separated.py /path/to/pdfs

# Custom index name
python rag-indexer-ppocrv5-embed-separated.py /path/to/pdfs --index-name my_specs

# Different chunk size
python rag-indexer-ppocrv5-embed-separated.py /path/to/pdfs --chunk-size 500
```

### `docker-compose-full.yaml` (Infrastructure)
**Purpose:** Run PaddleOCR and Elasticsearch in containers

**Services:**
- `paddleocr`: PaddlePaddle OCR server (GPU-enabled)
- `elasticsearch`: Elasticsearch 8.11.0 (document storage + search)

**Usage:**
```bash
# Start services
docker-compose -f docker-compose-full.yaml up -d

# Check status
docker-compose -f docker-compose-full.yaml ps

# Stop services
docker-compose -f docker-compose-full.yaml down
```

## Workflow: From PDFs to Answers

### Step 1: Setup Infrastructure
```bash
# Start Docker services
docker-compose -f docker-compose-full.yaml up -d

# Verify Elasticsearch is running
curl http://localhost:9200

# Verify Ollama is running (for LLM)
curl http://localhost:11434/api/tags
```

### Step 2: Index Your PDFs
```bash
# Run the indexer
python rag-indexer-ppocrv5-embed-separated.py /path/to/pdfs

# This will:
# 1. OCR all PDFs (via PaddleOCR Docker container)
# 2. Chunk text into segments
# 3. Generate embeddings (via local .venv with GPU)
# 4. Index to Elasticsearch

# Expected output:
# "Indexed 35,969 chunks (332.7MB) in 45 minutes"
```

### Step 3: Query Your Documents

#### Option A: Interactive Mode (Best for exploration)
```bash
python multi_rag_queries.py /path/to/pdfs --interactive

# Then ask questions:
You: What are Type 1 terminals?
Bot: [Detailed answer with sources]

You: What about Type 2?
Bot: [Understands you mean "Type 2 terminals" from context]

# Commands:
# - 'history' to see conversation
# - 'clear' to reset conversation
# - 'fast: question' for quick answers
# - 'exit' to quit
```

#### Option B: Single Query Mode
```bash
# Full intelligence (all agents enabled)
python multi_rag_queries.py /path/to/pdfs "what terminals should I use?"

# Fast mode (skip intent detection and iterative search)
python multi_rag_queries.py /path/to/pdfs "simple question" --fast

# With verbose output (show metadata)
python multi_rag_queries.py /path/to/pdfs "question" --verbose
```

## Performance Characteristics

### Speed (on typical hardware)

**Fast Mode:**
- ~3-5 seconds per query
- Skips: Intent detection, query expansion, iterative refinement
- Uses: Basic hybrid search + rerank

**Full Mode:**
- ~8-12 seconds per query
- Includes: All 8 agents
- First query slower (~15s) due to model loading

**What Takes Time:**
1. Ollama LLM calls (~1-2s each)
   - Intent classification: 1 LLM call
   - Query expansion: 1 LLM call
   - HyDE generation: 1 LLM call
   - Keyword extraction: 1 LLM call
   - Answer generation: 1 LLM call
2. Elasticsearch searches (~0.5s each)
3. Cross-encoder reranking (~1s)
4. Embedding generation (~0.3s)

### Memory Usage

**Embedding Model (BAAI/bge-large-en-v1.5):**
- FP16: ~1.5 GB GPU memory
- FP32: ~3 GB GPU memory

**Reranker Model (cross-encoder MiniLM):**
- ~0.5 GB GPU/CPU memory

**Ollama (llama3.1:8b):**
- ~8 GB RAM (CPU mode)
- ~6 GB GPU memory (GPU mode)

**Total (all models):**
- With GPU: ~10 GB GPU + 4 GB RAM
- CPU only: ~12 GB RAM

### Index Size

**Example (your current index):**
- Documents: 35,969 chunks
- Size: 332.7 MB (with best_compression)
- Without compression: ~500-600 MB

**Scaling:**
- ~10 MB per PDF (compressed)
- 1000 PDFs → ~10 GB index
- Search speed: O(log n), stays fast even at 100K+ docs

## Configuration Options

### Environment Variables
```bash
# Elasticsearch
export ES_URL="http://localhost:9200"

# Ollama LLM
export OLLAMA_URL="http://localhost:11434/api/generate"
export OLLAMA_MODEL="llama3.1:8b"

# Models
export EMBED_MODEL="BAAI/bge-large-en-v1.5"
export RERANK_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
```

### Command Line Arguments

**multi_rag_queries.py:**
```bash
--interactive, -i       # Enable conversation mode
--fast                  # Fast mode (skip advanced features)
--verbose, -v           # Show detailed metadata
--top-k N               # Return N results (default: 5)
--es-url URL            # Elasticsearch URL
--model NAME            # Embedding model
--rerank-model NAME     # Reranker model
--ollama-model NAME     # LLM model for query understanding
--no-ambiguity          # Disable ambiguity detection
--no-iterative          # Disable iterative retrieval
--index-name NAME       # Override index name
```

**rag-indexer:**
```bash
--index-name NAME       # Custom index name
--chunk-size N          # Chunk size in words (default: 400)
--chunk-overlap N       # Overlap in words (default: 50)
--batch-size N          # Embedding batch size (default: 128)
--es-url URL            # Elasticsearch URL
```

## Troubleshooting

### Problem: "No results found" for queries you know should match

**Diagnosis:**
```bash
# Check if query is too generic
python multi_rag_queries.py /path/to/pdfs "terminals" --verbose

# Look at expanded_query in metadata
# If expansion failed, query might be too vague
```

**Solutions:**
1. Use more specific terminology: "Type 1 terminal blocks" not "connectors"
2. Enable smart mode: `--smart` or use multi_rag_queries.py
3. Check conversation history: Maybe context is missing
4. Try fast mode if advanced features are confusing it: `--fast`

### Problem: Slow queries (>20 seconds)

**Diagnosis:**
```bash
# Enable verbose mode to see where time is spent
python multi_rag_queries.py /path/to/pdfs "question" --verbose
```

**Solutions:**
1. Use fast mode: `--fast` (saves ~5-7 seconds)
2. Reduce top-k: `--top-k 3` (saves ~1 second in reranking)
3. Disable iterative: `--no-iterative` (saves ~5 seconds)
4. Check Ollama performance: Slow LLM responses?
5. Move Ollama to GPU if on CPU

### Problem: Wrong or irrelevant answers

**Diagnosis:**
1. Check intent classification: `--verbose` shows intent detection
2. Look at sources: Are retrieved docs relevant?
3. Check if ambiguity detection triggered
4. Review conversation history: Wrong context?

**Solutions:**
1. Clear conversation: `clear` command in interactive mode
2. Use more specific questions: Add technical terms
3. Enable ambiguity detection: Remove `--no-ambiguity`
4. Check index quality: Re-run OCR if documents are unclear
5. Try different query phrasing

### Problem: "Connection refused" errors

**Check Services:**
```bash
# Elasticsearch
curl http://localhost:9200
# Should return JSON with version info

# Ollama
curl http://localhost:11434/api/tags
# Should return list of models

# If either fails, start the service
docker-compose -f docker-compose-full.yaml up -d  # Elasticsearch
# or
ollama serve  # Ollama (if not running)
```

## Advanced Topics

### Customizing the Prompt Templates

All LLM prompts are in multi_rag_queries.py. To customize:

1. **Intent Classification:** Search for "classify_query_intent" function
2. **HyDE Generation:** Search for "generate_hypothetical_answer" function
3. **Query Expansion:** Search for "expand_query_with_context" function
4. **Answer Generation:** Search for "generate_contextual_answer" function

### Using Different Models

**Embedding Models:**
```python
# In code or via --model argument
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Smaller, faster
EMBED_MODEL = "BAAI/bge-large-en-v1.5"  # Current (good balance)
EMBED_MODEL = "BAAI/bge-m3"  # Multilingual
```

**Reranker Models:**
```python
# Via --rerank-model argument
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Current
RERANK_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"  # Faster
RERANK_MODEL = "BAAI/bge-reranker-large"  # More accurate
```

**LLM Models (Ollama):**
```bash
# List available models
ollama list

# Use different model
python multi_rag_queries.py /path/to/pdfs "question" --ollama-model llama3.2:8b
```

### Adjusting Retrieval Parameters

In code (multi_rag_queries.py):
```python
# Number of results from each search type
BM25_K = 20          # Keyword search results
KNN_K = 30           # Vector search results
RERANK_TOP_N = 20    # Candidates for reranking
FINAL_TOP_K = 5      # Final results to return

# Score combination (in hybrid_search)
hyde_results_weight = 1.5  # HyDE results boosted 1.5x
keyword_results_weight = 1.0
direct_results_weight = 1.0
```

## Future Enhancements

Potential improvements to consider:

1. **Query Caching:** Cache results for repeated queries
2. **Source Deduplication:** Merge overlapping document chunks
3. **Confidence Scores:** Show confidence in answer quality
4. **Multi-language Support:** Use multilingual embedding models
5. **Graph Relations:** Track relationships between concepts
6. **Auto-Summarization:** Summarize long documents before chunking
7. **Image Understanding:** OCR with layout analysis for tables/diagrams
8. **Citation Links:** Direct links to source PDF pages

## Summary

You now have a complete understanding of:
- ✅ The 8-agent architecture and how they work together
- ✅ Key techniques (HyDE, hybrid search, reranking, conversation memory)
- ✅ How to use the system (interactive and single-query modes)
- ✅ Configuration options and performance characteristics
- ✅ Troubleshooting common issues
- ✅ How to customize and extend the system

The system handles ambiguous questions by:
1. Understanding intent
2. Detecting ambiguity and asking for clarification
3. Using conversation memory for context
4. Employing multiple complementary search strategies
5. Generating natural, cited answers

This gives you ChatGPT-like flexibility while staying grounded in your actual technical documents!
