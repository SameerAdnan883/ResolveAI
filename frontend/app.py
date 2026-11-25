import streamlit as st
from backend.config import settings
from frontend.api_client import ResolveAIClient
from frontend.customer_portal import render_customer_portal
from frontend.admin_dashboard import render_admin_dashboard

st.set_page_config(
    page_title="ResolveAI – Multi-Agent Customer Support",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SaaS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

client = ResolveAIClient(base_url=settings.BACKEND_URL)

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.markdown("## **ResolveAI**")
    st.caption("Multi-Agent Customer Support Resolution Platform")
    st.divider()

    portal_mode = st.radio(
        "Select Portal Mode:",
        ["Customer Portal", "Admin & Supervisor Dashboard"],
        index=0
    )

    st.divider()
    st.markdown("### System Architecture")
    st.caption("""
    - **LLM:** Local Qwen3.5:4B (Ollama)
    - **Orchestration:** LangGraph (StateGraph)
    - **Human-in-the-Loop:** LangGraph Native Interrupt
    - **Checkpointer:** SQLite persistent state
    - **RAG Knowledge:** ChromaDB (Sentence-Transformers)
    - **Backend:** FastAPI (Async REST)
    - **Database:** SQLite / SQLAlchemy ORM
    """)

    # Live backend health indicator
    health = client.health()
    if health.get("status") == "healthy":
        st.success("🟢 Backend API: Online")
    else:
        st.warning("🟡 Backend API: Standalone / Launching")

if portal_mode == "Customer Portal":
    render_customer_portal(client)
else:
    render_admin_dashboard(client)
