import sys
import subprocess
import uvicorn
from backend.config import settings

def run_backend():
    print(f"Starting ResolveAI FastAPI Backend on http://{settings.BACKEND_HOST}:{settings.BACKEND_PORT}...")
    uvicorn.run("backend.main:app", host=settings.BACKEND_HOST, port=settings.BACKEND_PORT, reload=False)

def run_frontend():
    print(f"Starting ResolveAI Streamlit Frontend on port {settings.FRONTEND_PORT}...")
    subprocess.run(["streamlit", "run", "frontend/app.py", "--server.port", str(settings.FRONTEND_PORT)])

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "frontend":
        run_frontend()
    else:
        run_backend()

