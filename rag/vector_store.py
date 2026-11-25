import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from backend.config import settings

_chroma_client = None
_collection = None

COLLECTION_NAME = "resolveai_company_policies"

def get_chroma_client():
    """Initializes or returns a persistent ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIRECTORY,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
    return _chroma_client

def get_policy_collection():
    """Returns or creates the Chroma collection for enterprise policy documents."""
    global _collection
    if _collection is None:
        client = get_chroma_client()
        try:
            from chromadb.utils import embedding_functions
            embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.EMBEDDING_MODEL_NAME
            )
            _collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=embed_fn,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            # Fallback to default Chroma embedding function if sentence-transformers is loading
            _collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
    return _collection

