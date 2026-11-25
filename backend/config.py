import os
from enum import Enum
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field
import httpx


class ExecutionMode(str, Enum):
    LIVE_QWEN = "LIVE_QWEN"
    DETERMINISTIC_FALLBACK = "DETERMINISTIC_FALLBACK"


class Settings(BaseSettings):
    # LLM Provider Selection
    LLM_PROVIDER: str = Field(default="qwen", description="Active LLM provider: 'qwen'")

    # Local Qwen3.5:4B LLM Configuration (via Ollama)
    QWEN_MODEL: str = Field(default="qwen3.5:4b", description="Local Qwen model name: qwen3.5:4b")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    QWEN_LOW_REASONING: bool = Field(default=True, description="Disable internal reasoning/thinking tokens for fast inference")

    # Security & JWT
    SECRET_KEY: str = Field(default="resolveai-dev-secret-key-super-secure-at-least-32-chars-long")
    ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=1440)

    # Database
    DATABASE_URL: str = Field(default="sqlite:///./resolveai.db")
    CHECKPOINT_DATABASE_URL: str = Field(default="./resolveai_checkpoints.db")

    # Vector DB / Chroma
    CHROMA_PERSIST_DIRECTORY: str = Field(default="./chroma_db")
    EMBEDDING_MODEL_NAME: str = Field(default="all-MiniLM-L6-v2")

    # App Environment
    ENVIRONMENT: str = Field(default="development")
    BACKEND_HOST: str = Field(default="127.0.0.1")
    BACKEND_PORT: int = Field(default=8000)
    BACKEND_URL: str = Field(default="http://127.0.0.1:8000")
    FRONTEND_PORT: int = Field(default=8501)

    # Business Rules
    AUTO_REFUND_MAX_AMOUNT: float = Field(default=0.00, description="Strict rule: no LLM can auto-refund without human check")
    HIGH_VALUE_THRESHOLD: float = Field(default=150.00, description="Orders over $150 strictly require human review for refund")
    MAX_AGENT_RETRIES: int = Field(default=2)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def check_qwen_health(self) -> Dict[str, Any]:
        """Validates Ollama connectivity and verifies qwen3.5:4b availability."""
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.OLLAMA_BASE_URL.rstrip('/')}/api/tags")
                if res.status_code == 200:
                    models = [m.get("name") for m in res.json().get("models", [])]
                    model_found = any(self.QWEN_MODEL in m for m in models)
                    return {
                        "configured": True,
                        "status": "healthy" if model_found else "model_missing",
                        "model": self.QWEN_MODEL,
                        "low_reasoning_enabled": self.QWEN_LOW_REASONING,
                        "available_models": models,
                        "message": f"Ollama online. Model '{self.QWEN_MODEL}' ready." if model_found else f"Model '{self.QWEN_MODEL}' not found in Ollama."
                    }
                else:
                    return {
                        "configured": False,
                        "status": "error",
                        "model": self.QWEN_MODEL,
                        "message": f"Ollama returned HTTP {res.status_code}"
                    }
        except Exception as e:
            return {
                "configured": False,
                "status": "unavailable",
                "model": self.QWEN_MODEL,
                "message": f"Could not connect to Ollama at {self.OLLAMA_BASE_URL}: {str(e)}"
            }

    def get_execution_mode(self) -> ExecutionMode:
        """Returns whether the system is set to run live Qwen or deterministic fallback."""
        if self.LLM_PROVIDER == "qwen":
            qwen_health = self.check_qwen_health()
            if qwen_health.get("status") == "healthy":
                return ExecutionMode.LIVE_QWEN
        return ExecutionMode.DETERMINISTIC_FALLBACK


settings = Settings()
