# ResolveAI Production Deployment Guide

This guide details steps to deploy ResolveAI in production environments.

---

## 1. System Requirements
- OS: Linux (Ubuntu 22.04+ LTS recommended), macOS, or Windows WSL2
- Python: 3.10, 3.11, or 3.12
- Memory: 8 GB RAM minimum (16 GB recommended for Ollama Qwen3.5:4B + ChromaDB embeddings)
- Disk: 15 GB free storage (for Ollama model weights & ChromaDB index)
- Ollama: Installed and running (`ollama serve`)

---

## 2. Environment Configuration
Create a production `.env` file from `.env.example`:
```bash
cp .env.example .env
```
Populate production configuration:
```env
LLM_PROVIDER=qwen
QWEN_MODEL=qwen3.5:4b
OLLAMA_BASE_URL=http://localhost:11434
QWEN_LOW_REASONING=true

SECRET_KEY=generate-a-strong-random-hex-string-at-least-32-bytes
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

DATABASE_URL=sqlite:///./resolveai.db
CHECKPOINT_DATABASE_URL=./resolveai_checkpoints.db

CHROMA_PERSIST_DIRECTORY=./chroma_db
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2

BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
BACKEND_URL=http://localhost:8000
FRONTEND_PORT=8501

AUTO_REFUND_MAX_AMOUNT=0.00
HIGH_VALUE_THRESHOLD=150.00
MAX_AGENT_RETRIES=2
```

---

## 3. Pulling the Ollama Model
Ensure Ollama is running and download the model:
```bash
ollama serve &
ollama pull qwen3.5:4b
```

---

## 4. Python Virtual Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Database Initialization & Seeding
```bash
python -c "from database.seed_data import seed_database; seed_database()"
python -c "from rag.knowledge_base import index_policy_documents; index_policy_documents()"
```

---

## 6. Running Services with Supervisord or Systemd
Use systemd or Docker Compose to run:
1. `uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 2`
2. `streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0`
