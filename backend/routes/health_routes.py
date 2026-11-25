from fastapi import APIRouter
from backend.config import settings

router = APIRouter(tags=["System Health"])

@router.get("/health")
def health_check():
    """System health check and configuration verification for local Qwen3.5:4B (Ollama), DB, checkpoints, and RAG."""
    qwen_health = settings.check_qwen_health()
    return {
        "status": "healthy" if qwen_health.get("status") == "healthy" else "degraded",
        "service": "ResolveAI Multi-Agent System",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "active_provider": settings.LLM_PROVIDER,
        "execution_mode": settings.get_execution_mode().value,
        "database": settings.DATABASE_URL.split(":///")[0],
        "checkpointer": {
            "type": "SQLite",
            "database_file": settings.CHECKPOINT_DATABASE_URL
        },
        "qwen_llm": qwen_health,
        "rag": {
            "vector_store": "ChromaDB",
            "persist_dir": settings.CHROMA_PERSIST_DIRECTORY,
            "embedding_model": settings.EMBEDDING_MODEL_NAME
        }
    }
