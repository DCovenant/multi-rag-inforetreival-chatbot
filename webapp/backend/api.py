from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from elasticsearch import Elasticsearch

from integrated_rag_queries import query_and_answer, ES_URL, EMBED_MODEL, RERANK_MODEL
from utils.conversation_history import ConversationHistory
from utils.model_loading import get_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
INDEX_NAME = "sse_specs"  # Adjust to your index

# Initialize once
es = Elasticsearch(ES_URL, request_timeout=60)
get_model(EMBED_MODEL, RERANK_MODEL)  # Preload models
conversation = ConversationHistory()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    result = query_and_answer(es, INDEX_NAME, req.question, conversation)
    return QueryResponse(answer=result["answer"], sources=result["sources"])