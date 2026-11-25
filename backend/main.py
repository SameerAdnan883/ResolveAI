import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from database.connection import engine, Base
from database.seed_data import seed_database
from rag.knowledge_base import index_policy_documents
from backend.routes.auth_routes import router as auth_router
from backend.routes.ticket_routes import router as ticket_router
from backend.routes.analytics_routes import router as analytics_router
from backend.routes.health_routes import router as health_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ResolveAI.Main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Database & RAG Vector Knowledge Base
    logger.info("Initializing ResolveAI database and verifying tables...")
    Base.metadata.create_all(bind=engine)
    seed_database()

    logger.info("Verifying RAG ChromaDB knowledge base indexing...")
    try:
        index_policy_documents()
    except Exception as e:
        logger.warning(f"Chroma indexing deferred: {e}")

    logger.info(f"ResolveAI initialized. Primary LLM: {settings.QWEN_MODEL} via Ollama ({settings.OLLAMA_BASE_URL})")
    yield
    logger.info("Shutting down ResolveAI application.")

app = FastAPI(
    title="ResolveAI – Multi-Agent Customer Support Resolution System",
    description="Autonomous multi-agent enterprise customer support platform with local Qwen3.5:4B LLM, LangGraph, RAG, and human approval controls.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(ticket_router)
app.include_router(analytics_router)

@app.get("/")
def root():
    return {
        "message": "Welcome to ResolveAI Multi-Agent Resolution System API",
        "documentation": "/docs",
        "health": "/health"
    }
