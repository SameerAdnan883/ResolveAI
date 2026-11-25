from typing import Dict, Any, List, Optional
from rag.knowledge_base import retrieve_policy_knowledge

def search_company_policy(query: str, category: Optional[str] = None, top_k: int = 3) -> Dict[str, Any]:
    """
    RAG-powered tool to search official company policies in ChromaDB.
    Guarantees evidence-based grounding with source citations to prevent hallucinations.
    """
    search_query = f"{category} policy: {query}" if category else query
    evidence_chunks = retrieve_policy_knowledge(search_query, top_k=top_k)

    if not evidence_chunks:
        return {
            "found": False,
            "query": query,
            "evidence": [],
            "summary": "No matching official company policy clauses found."
        }

    formatted_citations = []
    for chunk in evidence_chunks:
        formatted_citations.append({
            "source": chunk["source"],
            "section": chunk["section"],
            "clause_text": chunk["content"],
            "relevance": chunk.get("relevance_score", 0.0)
        })

    return {
        "found": True,
        "query": query,
        "category": category,
        "total_results": len(formatted_citations),
        "citations": formatted_citations
    }

