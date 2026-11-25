import time
from typing import Dict, Any, List, Optional
from tools.policy_tools import search_company_policy

def run_policy_agent(category: str, complaint_text: str, order_status: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes Policy Research Agent.
    Retrieves grounded evidence from ChromaDB for company policies.
    Guarantees no fabricated clauses.
    """
    start_time = time.time()

    query_map = {
        "refund": "refund policy eligibility 30 days approval threshold",
        "replacement": "replacement defective item 14 days expedited shipping",
        "cancellation": "cancellation in transit processing rules",
        "shipping_delay": "delivery delay carrier tracking compensation policy",
        "damaged_item": "damaged product reporting 7 days replacement refund photo",
        "general_inquiry": "customer service support policy return window"
    }

    base_query = query_map.get(category, f"{category} policy rules")
    if order_status:
        base_query += f" order status {order_status}"

    search_result = search_company_policy(query=base_query, category=category, top_k=3)
    citations = search_result.get("citations", [])

    evidence_summary = ""
    if citations:
        sources = list(set([c["source"] for c in citations]))
        evidence_summary = f"Retrieved {len(citations)} official policy clauses from {', '.join(sources)}."
    else:
        evidence_summary = "No exact policy clause matched."

    return {
        "category": category,
        "citations": citations,
        "total_citations": len(citations),
        "evidence_summary": evidence_summary,
        "execution_mode": "LOCAL_CHROMA_RAG",
        "model": "all-MiniLM-L6-v2_chromadb",
        "execution_time_ms": (time.time() - start_time) * 1000
    }

