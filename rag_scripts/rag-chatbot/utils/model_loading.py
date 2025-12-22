"""
MODEL LOADING

These functions load and cache AI models to avoid reloading them on every query.
Models are expensive to load (takes 2-3 seconds) so we keep them in memory.
"""
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
import logging
import torch

_embedding_model_cache = {}
_reranker_model_cache = {}

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

def get_model(emb_model_name: Optional[str] = None,
               rerank_model_name: Optional[str] = None
               ) -> Tuple[Optional[SentenceTransformer], Optional[CrossEncoder]]:
    """
    Load and cache embedding and/or reranker models.
    Use None to skip loading a model.
    Returns: (embedding_model_or_None, reranker_model_or_None)
    """
    global _reranker_model_cache, _embedding_model_cache
    emb_model = None
    rerank_model = None

    # ------ RERANKER LOGIC ------
    # Check reranker cache
    if rerank_model_name:
        rerank_model = _reranker_model_cache.get(rerank_model_name) # Checks if cache has the model already loaded
        if rerank_model is None:
            logger.info(f"Loading reranker model: {rerank_model_name}")    
            rerank_model = CrossEncoder(rerank_model_name)

        _reranker_model_cache[rerank_model_name] = rerank_model # Cache the model so we don't reload it
    
    # ------ EMBBEDER LOGIC ------
    # Check if we already loaded this model (singleton pattern)
    if emb_model_name:
        emb_model = _embedding_model_cache.get(emb_model_name) # Checks if cache has the model already loaded
        if emb_model is None:
            logger.info(f"Loading embedding model: {emb_model_name}")
            emb_model = SentenceTransformer(emb_model_name)
        
            # Move to GPU and use FP16 (half precision) for faster inference
            if torch.cuda.is_available():
                try:
                    emb_model = emb_model.to('cuda')  # Move model to GPU
                    emb_model.half()  # Use 16-bit floats instead of 32-bit (2x faster, minimal accuracy loss)
                    logger.info(f"✓ Using GPU for embeddings")
                except Exception as e:
                    print(f"Error -> get_model() <- moving embedder to GPU: {e}")
        
        _embedding_model_cache[emb_model_name] = emb_model  # Cache the model so we don't reload it
    
    return emb_model, rerank_model

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
    model, _= get_model(model_name,"")
    # encode() returns numpy array, we normalize to unit length and convert to list
    emb = model.encode([query_text], normalize_embeddings=True)[0]
    return emb.tolist()