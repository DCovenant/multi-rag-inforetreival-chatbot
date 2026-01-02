# RAG Chatbot for Technical Documentation

A Retrieval-Augmented Generation (RAG) chatbot system designed for querying technical documentation using AI-powered semantic search and natural language understanding.

## Overview

This system allows users to ask natural language questions about technical specifications and receive accurate, source-cited answers. It combines modern vector search with classical keyword matching, enhanced by Large Language Model reasoning.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Browser                            │
│                        (localhost:5173)                         │
└─────────────────────────────┬───────────────────────────────────┘
                              │ HTTP POST /query
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Vue 3 Frontend                             │
│              Chat interface with source display                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │ REST API
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                             │
│           Query processing, embedding, RAG pipeline             │
└──────────┬──────────────────────────────────┬───────────────────┘
           │                                  │
           ▼                                  ▼
┌─────────────────────┐            ┌─────────────────────┐
│    Elasticsearch    │            │       Ollama        │
│  Vector + BM25 DB   │            │    LLM Inference    │
│   (localhost:9200)  │            │  (localhost:11434)  │
└─────────────────────┘            └─────────────────────┘
```

All services run as Docker containers on a shared network with GPU acceleration enabled.

## Project Structure

```
chatbot/
├── webapp/                     # Web application (containerized)
│   ├── frontend/               # Vue 3 single-page application
│   │   ├── src/
│   │   │   └── App.vue         # Main chat interface
│   │   ├── package.json        # Node dependencies
│   │   ├── vite.config.js      # Vite build configuration
│   │   └── Dockerfile.frontend
│   │
│   ├── backend/                # FastAPI REST server
│   │   ├── api.py              # Query endpoint
│   │   └── Dockerfile.backend
│   │
│   └── docker-compose.yaml     # Service orchestration
│
├── rag_scripts/                # RAG pipeline and data processing
│   ├── rag_chatbot/            # Core query engine
│   │   ├── integrated_rag_queries.py    # Main RAG pipeline
│   │   └── utils/
│   │       ├── optimized_retrieval.py   # Hybrid search logic
│   │       ├── structured_answer_generator.py
│   │       ├── model_loading.py         # ML model caching
│   │       ├── conversation_history.py  # Context management
│   │       ├── centralized_prompts.py   # LLM prompts
│   │       └── table_context.py         # Table rendering
│   │
│   ├── dataprocessing/         # Document indexing pipeline
│   │   ├── rag-indexer-ppocrv5-ocr-separated.py  # OCR extraction
│   │   ├── text_parser_docling.py       # Document parsing
│   │   └── embeddings_indexer.py        # Vector indexing
│   │
│   └── docker-compose.yaml     # Data processing infrastructure
│
├── dataset/                    # ML dataset utilities
│   ├── generate_mrl_dataset_advanced.py
│   └── train_mrl_embeddings.py
│
└── esbackups/                  # Elasticsearch backups
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Frontend | Vue 3, Vite | Interactive chat interface |
| Backend | FastAPI, Python 3.12 | REST API, RAG orchestration |
| Search | Elasticsearch 8.11 | Vector and keyword search |
| Embeddings | Sentence Transformers (all-mpnet-base-v2) | Semantic vectorization |
| Reranking | Cross-Encoder (ms-marco-MiniLM) | Result refinement |
| LLM | Ollama (llama3.1:8b) | Answer generation |
| OCR | PaddleOCR v5 | Document text extraction |
| Parsing | IBM Docling | Structured document conversion |

## How It Works

### Theoretical Foundation

The system implements a RAG (Retrieval-Augmented Generation) architecture, which addresses the limitation of LLMs not having access to private or recent data. Instead of relying solely on the model's training data, RAG retrieves relevant documents and provides them as context for answer generation.

**Key Concepts:**

1. **Hybrid Search**: Combines lexical matching (BM25) with semantic similarity (vector search) to find relevant documents. BM25 excels at exact term matching while vectors capture meaning and synonyms.

2. **Reciprocal Rank Fusion (RRF)**: Merges results from multiple retrieval methods by combining their rankings rather than raw scores, preventing any single method from dominating.

3. **Cross-Encoder Reranking**: A specialized model that jointly encodes the query and each document together, providing more accurate relevance scores than bi-encoder embeddings alone.

4. **Structured Generation**: The LLM first extracts facts as JSON, then generates a natural language answer. This two-stage approach reduces hallucination by constraining the model to only use retrieved information.

### Practical Workflow

#### Document Ingestion Pipeline

```
PDF Documents
      │
      ▼
┌─────────────┐
│  OCR/Parse  │  Extract text while preserving structure
└─────┬───────┘
      │
      ▼
┌─────────────┐
│   Chunking  │  Split into ~1000 token chunks with overlap
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  Embedding  │  Convert chunks to 384-dimensional vectors
└─────┬───────┘
      │
      ▼
┌─────────────┐
│   Index     │  Store in Elasticsearch with metadata
└─────────────┘
```

**What gets stored:**
- Chunk text and vector embedding
- Source file and page number
- Section headers and hierarchy
- Table data (in coordinate format)
- Content type classification

#### Query Pipeline

```
User Question: "What are the voltage requirements?"
                    │
                    ▼
         ┌─────────────────────┐
         │  Embed Question     │  Convert to vector
         └──────────┬──────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│  BM25 Search  │       │  kNN Search   │
│  (keywords)   │       │  (semantic)   │
└───────┬───────┘       └───────┬───────┘
        │     Top 50            │    Top 30
        └───────────┬───────────┘
                    ▼
         ┌─────────────────────┐
         │   RRF Fusion        │  Merge rankings
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Cross-Encoder      │  Rerank top 30
         │  Reranking          │
         └──────────┬──────────┘
                    │    Top 10
                    ▼
         ┌─────────────────────┐
         │  Query Type         │  Classify intent
         │  Detection          │  (enumeration, definition, etc.)
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Fact Extraction    │  Extract as JSON
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Answer Generation  │  LLM creates response
         └──────────┬──────────┘
                    │
                    ▼
        Answer + Source Citations
```

### Query Types

The system classifies queries to apply specialized handling:

| Type | Example | Handling |
|------|---------|----------|
| ENUMERATION | "List all terminal types" | Structured list extraction |
| TABLE_DATA | "What are the voltage levels?" | Table-aware retrieval |
| DEFINITION | "What is a Type 1 terminal?" | Definition-focused extraction |
| REQUIREMENT | "What specifications must X meet?" | Requirement parsing |
| PROCEDURE | "How do I configure X?" | Step-by-step extraction |
| COMPARISON | "Difference between A and B?" | Multi-entity comparison |

### Conversation Memory

The system maintains conversation context for follow-up questions:
- Stores last 10 messages
- Extracts technical entities
- Provides context for ambiguous references ("it", "this", "the same")

## Running the System

### Prerequisites
- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- 16GB+ RAM

### Start Services

```bash
cd webapp
docker-compose up -d
```

This starts:
- **Elasticsearch** on port 9200
- **Backend API** on port 8000
- **Frontend UI** on port 5173
- **Ollama LLM** on port 11434

### Access the Application

Open `http://localhost:5173` in your browser.

### Index Documents

```bash
cd rag_scripts
docker-compose up -d  # Start processing infrastructure

# Run OCR extraction
python dataprocessing/rag-indexer-ppocrv5-ocr-separated.py

# Generate embeddings and index
python dataprocessing/embeddings_indexer.py
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ES_URL` | `http://localhost:9200` | Elasticsearch endpoint |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | LLM inference endpoint |

### Tunable Parameters

| Parameter | Location | Default | Purpose |
|-----------|----------|---------|---------|
| Chunk size | `embeddings_indexer.py` | 1000 tokens | Document granularity |
| Chunk overlap | `embeddings_indexer.py` | 200 tokens | Context preservation |
| BM25 results | `optimized_retrieval.py` | 50 | Initial keyword matches |
| kNN results | `optimized_retrieval.py` | 30 | Initial vector matches |
| Final results | `optimized_retrieval.py` | 10 | Returned to LLM |
| LLM temperature | `structured_answer_generator.py` | 0.15 | Response creativity |

## API Reference

### POST /query

Send a question and receive an answer with sources.

**Request:**
```json
{
  "question": "What are the voltage requirements for Type 1 terminals?"
}
```

**Response:**
```json
{
  "answer": "Type 1 terminals require...",
  "sources": [
    {
      "file_name": "spec_v2.pdf",
      "page_number": 45,
      "score": 0.892,
      "section_header": "Terminal Specifications",
      "content_type": "body"
    }
  ]
}
```

## Design Decisions

### Why Hybrid Search?

Neither keyword nor semantic search alone is sufficient:
- **BM25** finds exact terminology but misses synonyms
- **Vector search** captures meaning but may miss specific terms
- **Combined** provides robust retrieval across query types

### Why Two-Stage Answer Generation?

Direct LLM generation often hallucinates or adds information not in the sources. The two-stage approach:
1. First extracts facts as validated JSON
2. Then generates answers only from those facts

This constrains the model to retrieved information.

### Why Cross-Encoder Reranking?

Bi-encoders (used for initial retrieval) encode query and document separately. Cross-encoders see both together, enabling:
- Better understanding of query-document relevance
- More accurate final ranking
- Worth the computational cost for small candidate sets

## Performance Considerations

- **Model Caching**: Embedding and reranking models load once and persist in memory
- **GPU Acceleration**: FP16 inference for faster processing
- **Bounded Memory**: Conversation history limited to 10 messages
- **Batch Processing**: Document indexing processes in batches with progress tracking

## Acknowledgments

Built with:
- [Sentence Transformers]
- [Elasticsearch]
- [Ollama]
- [PaddleOCR]
- [IBM Docling]
- [Vue.js]
- [FastAPI]
