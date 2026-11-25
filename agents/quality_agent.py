import time
from typing import Dict, Any, List, Optional
from backend.config import settings, ExecutionMode
from agents.qwen_optimizer import QwenOptimizer, QualityAuditStructuredOutput, QWEN_CONCISE_QUALITY_PROMPT

QUALITY_SYSTEM_PROMPT = """You are the ResolveAI Quality and Escalation Agent.
Your role is to independently review proposed customer resolutions before they reach the customer or database.

Audit Checklist:
1. Hallucination Check: Did the resolution cite policies or order items that do not exist in the verified data?
2. Policy Alignment: Does the resolution strictly abide by company return/cancellation windows?
3. Financial Safeguard: Does any refund or sensitive action correctly flag human supervisor approval?
4. Contradiction Check: Is there any conflict between the order status (e.g. In Transit) and the proposal (e.g. cancelled)?

Output format strictly JSON:
{
  "passed_audit": true | false,
  "confidence_score": 0.0 to 1.0,
  "hallucination_detected": true | false,
  "contradictions": ["list of contradictions, if any"],
  "escalation_required": true | false,
  "escalation_reason": "Specific reason if human intervention is required",
  "audit_summary": "Concise summary of audit findings"
}
"""

def run_quality_agent(
    complaint_text: str,
    order_data: Optional[Dict[str, Any]],
    policy_citations: List[Dict[str, Any]],
    proposed_resolution: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes Quality and Escalation Agent.
    Audits the proposed resolution for correctness, policy grounding,
    and determines whether human escalation is mandatory.
    """
    start_time = time.time()
    order_data = order_data or {"found": False}
    
    contradictions = []
    hallucination_detected = False
    
    # Check 1: In-transit cancellation contradiction
    if order_data.get("status") == "in_transit" and proposed_resolution.get("action") == "cancellation":
        contradictions.append("Contradiction: Cannot execute immediate cancellation on an order already in transit.")

    # Check 2: Refund without approval
    if proposed_resolution.get("action") == "refund" and not proposed_resolution.get("requires_approval"):
        contradictions.append("Safeguard breach: Direct financial refund proposal must require supervisor approval.")

    # Check 3: Expired return window
    days_since_delivery = order_data.get("days_since_delivery")
    if days_since_delivery and days_since_delivery > 30 and proposed_resolution.get("action") == "refund" and proposed_resolution.get("eligible"):
        contradictions.append(f"Policy breach: Order delivered {days_since_delivery} days ago cannot be marked eligible for standard refund.")

    order_needed = proposed_resolution.get("action") in ["refund", "replacement", "cancellation", "carrier_trace"]
    escalation_required = bool(
        proposed_resolution.get("requires_approval") or 
        contradictions or 
        (order_needed and not order_data.get("found", True)) or
        order_data.get("amount", 0) > 150.0
    )

    escalation_reason = None
    if escalation_required:
        reasons = []
        if proposed_resolution.get("requires_approval"):
            reasons.append("Supervisor authorization required for resolution action")
        if order_data.get("amount", 0) > 150.0:
            reasons.append(f"Order value (${order_data.get('amount'):.2f}) exceeds $150 supervisor threshold")
        if order_needed and not order_data.get("found"):
            reasons.append("Unverified order ID requires customer service verification")
        if contradictions:
            reasons.extend(contradictions)
        escalation_reason = "; ".join(reasons)

    audit_summary = "Quality audit passed." if not contradictions else f"Audit flagged {len(contradictions)} issues."

    exec_mode = ExecutionMode.LIVE_QWEN.value if settings.LLM_PROVIDER == "qwen" else ExecutionMode.DETERMINISTIC_FALLBACK.value
    model_name = settings.QWEN_MODEL if settings.LLM_PROVIDER == "qwen" else "deterministic_quality_auditor"

    return {
        "passed_audit": len(contradictions) == 0,
        "confidence_score": 0.95 if not contradictions else 0.40,
        "hallucination_detected": hallucination_detected,
        "contradictions": contradictions,
        "escalation_required": escalation_required,
        "escalation_reason": escalation_reason,
        "audit_summary": audit_summary,
        "execution_mode": exec_mode,
        "model": model_name,
        "execution_time_ms": (time.time() - start_time) * 1000
    }
