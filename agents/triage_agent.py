import re
import time
import logging
from typing import Dict, Any, Optional
from backend.config import settings, ExecutionMode
from agents.qwen_optimizer import QwenOptimizer, TriageStructuredOutput, QWEN_CONCISE_TRIAGE_PROMPT

logger = logging.getLogger("ResolveAI.TriageAgent")

TRIAGE_SYSTEM_PROMPT = """You are the ResolveAI Triage Agent, an expert customer support classifier.
Your objective is to carefully analyze customer complaints, classify them into standard categories, determine customer intent, assign appropriate priority, and extract any order IDs mentioned.

Categories to choose from:
- 'refund' : Customer requesting money back, return of goods for reimbursement.
- 'replacement' : Customer requesting replacement for malfunctioning, broken, or mismatched items.
- 'cancellation' : Customer requesting to cancel an order before or during delivery.
- 'shipping_delay' : Package is late, tracking is stuck, delivery inquiry.
- 'damaged_item' : Product arrived physically broken, shattered, or crushed.
- 'general_inquiry' : Questions about store hours, general policies, product specs without specific order dispute.

Priority levels:
- 'urgent' : High monetary impact ($150+), severe physical damage, repeated failed promises.
- 'high' : Active product defect, refund requests, urgent delivery needs.
- 'medium' : Standard delays, minor questions regarding existing shipments.
- 'low' : General inquiries, feedback.

You MUST respond strictly with a valid JSON object matching this schema:
{
  "category": "refund | replacement | cancellation | shipping_delay | damaged_item | general_inquiry",
  "intent": "Brief 1-sentence description of what the customer is trying to achieve",
  "priority": "urgent | high | medium | low",
  "order_id": "Extracted order ID string (e.g. ORD-1001) or null if none found",
  "reasoning": "Short justification for the classification"
}
"""

def extract_order_id_regex(text: str) -> Optional[str]:
    """Fallback regex to extract standard order patterns like ORD-1001 or ORD1001."""
    match = re.search(r"\b(ORD[-\s]?\d{3,6})\b", text, re.IGNORECASE)
    if match:
        clean = match.group(1).upper().replace(" ", "-")
        if "-" not in clean and clean.startswith("ORD"):
            clean = f"ORD-{clean[3:]}"
        return clean
    return None

def heuristic_triage(complaint_text: str) -> Dict[str, Any]:
    """Deterministic fallback classifier for offline/mock test environments."""
    text_lower = complaint_text.lower()
    order_id = extract_order_id_regex(complaint_text)

    category = "general_inquiry"
    priority = "medium"
    intent = "General inquiry regarding customer order or service"

    # Check for general informational inquiries without specific order dispute first
    is_informational = (
        not order_id and any(q in text_lower for q in [
            "what is your", "what are your", "how long does standard delivery",
            "how do replacements work", "tell me about your", "policy window",
            "working hours", "support hours", "headquarters", "payment methods",
            "fee for returning", "reach a human", "internationally"
        ])
    )

    if is_informational:
        category = "general_inquiry"
        priority = "low"
        intent = "Customer inquiring about company information, standard policies, or service terms"
    elif any(k in text_lower for k in ["damage", "broken", "cracked", "shattered", "smashed", "dent"]):
        category = "damaged_item"
        priority = "urgent" if "urgent" in text_lower or "$" in text_lower else "high"
        intent = "Customer reported physical damage to package or product and requests remedy"
    elif any(k in text_lower for k in ["refund", "money back", "reimburse", "return for credit", "return payment", "returning", "return"]):
        category = "refund"
        priority = "high"
        intent = "Customer is requesting a financial refund or return"
    elif any(k in text_lower for k in ["replace", "replacement", "defective", "stopped working", "doesn't work", "broken mic", "dead on arrival"]):
        category = "replacement"
        priority = "high"
        intent = "Customer requests an exchange or replacement for a defective product"
    elif any(k in text_lower for k in ["cancel", "cancellation", "cancel my order", "stop and cancel"]):
        category = "cancellation"
        priority = "high"
        intent = "Customer is attempting to cancel an order"
    elif any(k in text_lower for k in ["where is my", "tracking", "delayed", "late", "has not arrived", "hasn't arrived", "stuck in transit", "shipped yet"]):
        category = "shipping_delay"
        priority = "medium"
        intent = "Customer is asking about delayed shipping or delivery tracking"

    return {
        "category": category,
        "intent": intent,
        "priority": priority,
        "order_id": order_id,
        "reasoning": "Heuristic classification based on keyword matching and regex order identification."
    }

def run_triage_agent(complaint_text: str, allow_fallback: bool = True) -> Dict[str, Any]:
    """
    Executes Triage Agent logic.
    - When LLM_PROVIDER == 'qwen', executes real local Ollama inference using Qwen3.5:4B.
    - If allow_fallback is True, gracefully falls back to deterministic classifier if the LLM is unavailable.
    - If allow_fallback is False, raises RuntimeError on API failure (for benchmark fidelity).
    """
    start_time = time.time()

    # Route A: Real Local Qwen3.5:4B Inference (Ollama)
    if settings.LLM_PROVIDER == "qwen":
        try:
            optimizer = QwenOptimizer()
            triage_obj, telemetry = optimizer.generate_structured_output(
                prompt=f"Customer Complaint:\n{complaint_text}",
                schema=TriageStructuredOutput,
                system_prompt=QWEN_CONCISE_TRIAGE_PROMPT
            )
            res = triage_obj.model_dump()
            if not res.get("order_id"):
                res["order_id"] = extract_order_id_regex(complaint_text)

            res["execution_mode"] = ExecutionMode.LIVE_QWEN.value
            res["model"] = telemetry.get("model", settings.QWEN_MODEL)
            res["execution_time_ms"] = telemetry.get("latency_ms", (time.time() - start_time) * 1000)
            res["token_usage"] = {
                "prompt_tokens": telemetry.get("prompt_eval_count", 0),
                "completion_tokens": telemetry.get("eval_count", 0),
                "total_tokens": telemetry.get("prompt_eval_count", 0) + telemetry.get("eval_count", 0)
            }
            res["reasoning"] = f"Classified by local {settings.QWEN_MODEL} via Ollama."
            return res
        except Exception as e:
            if not allow_fallback:
                raise RuntimeError(f"Qwen triage invocation failed: {e}") from e
            logger.warning(f"Qwen triage failed, falling back to heuristic: {e}")

    # Route B: Deterministic Fallback Path
    res = heuristic_triage(complaint_text)
    res["execution_mode"] = ExecutionMode.DETERMINISTIC_FALLBACK.value
    res["model"] = "heuristic_fallback"
    res["execution_time_ms"] = (time.time() - start_time) * 1000
    res["token_usage"] = None
    res["fallback_reason"] = "Executed via deterministic fallback classifier."

    return res
