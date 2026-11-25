import time
import json
import logging
from typing import Dict, Any, List, Optional
from backend.config import settings, ExecutionMode
from agents.qwen_optimizer import QwenOptimizer, ResolutionStructuredOutput, QWEN_CONCISE_RESOLUTION_PROMPT
from tools.validation_tools import validate_resolution

logger = logging.getLogger("ResolveAI.ResolutionAgent")

RESOLUTION_SYSTEM_PROMPT = """You are the ResolveAI Resolution Agent.
Your objective is to combine the customer complaint, verified order details, and official policy evidence to propose a compliant, fair, and actionable resolution.

CRITICAL RULES:
1. You MUST apply deterministic business rules.
2. NEVER directly authorize an uncontrolled refund. Any refund or high-value action ($150+) requires human supervisor sign-off.
3. If an order is in transit, it CANNOT be cancelled; the customer must receive it and initiate a return.
4. If an order was delivered > 30 days ago, it is NOT eligible for a standard refund under POL-REF-2024-V2.
5. If an order was delivered > 14 days ago, it is NOT eligible for a free replacement under POL-REP-2024-V2.
6. Format your output strictly as a JSON object:
{
  "action": "refund | replacement | cancellation | carrier_trace | reject_ineligible | explanation",
  "proposed_amount": 0.0,
  "eligible": true | false,
  "requires_approval": true | false,
  "reason": "Detailed justification citing the specific policy and order facts",
  "policy_cited": "e.g. POL-REF-2024-V2 Clause 1.1",
  "customer_message": "Courteous explanation to be sent to the customer detailing next steps"
}
"""

def run_resolution_agent(
    complaint_text: str,
    category: str,
    order_data: Optional[Dict[str, Any]],
    policy_citations: List[Dict[str, Any]],
    allow_fallback: bool = True,
    use_llm: bool = True
) -> Dict[str, Any]:
    """
    Executes Resolution Agent.
    Uses LLM (Qwen3.5:4B via Ollama) when available and use_llm=True, and strictly
    validates the proposal through deterministic business rules (validate_resolution).
    """
    start_time = time.time()
    order_data = order_data or {"found": False}
    
    # Run deterministic business validator first
    validation = validate_resolution(
        order_data=order_data,
        policy_evidence=policy_citations,
        proposed_action=category if category in ["refund", "replacement", "cancellation"] else "explanation",
        complaint_category=category
    )

    proposal = None
    execution_mode = ExecutionMode.DETERMINISTIC_FALLBACK.value

    # Route A: Real Local Qwen3.5:4B Inference
    if use_llm and settings.LLM_PROVIDER == "qwen":
        try:
            optimizer = QwenOptimizer()
            context = f"""
Complaint: {complaint_text}
Category: {category}
Order Details: {json.dumps(order_data, indent=2)}
Policy Citations: {json.dumps(policy_citations[:2], indent=2)}
Deterministic Business Constraints: {json.dumps(validation, indent=2)}
"""
            res_out, telemetry = optimizer.generate_structured_output(
                prompt=context,
                schema=ResolutionStructuredOutput,
                system_prompt=QWEN_CONCISE_RESOLUTION_PROMPT
            )
            proposal = res_out.model_dump()
            execution_mode = ExecutionMode.LIVE_QWEN.value
        except Exception as e:
            if not allow_fallback:
                raise RuntimeError(f"Qwen resolution failed: {e}") from e
            logger.warning(f"Qwen resolution failed, falling back to deterministic synthesis: {e}")

    # Route B: Deterministic fallback proposal synthesis
    if not proposal:
        action = validation.get("approved_action", "explanation")
        amount = float(order_data.get("amount", 0.0)) if order_data.get("found") else 0.0
        eligible = validation.get("is_valid", False)
        requires_approval = validation.get("requires_human_approval", False)

        if not order_data.get("found") and category in ["refund", "replacement", "cancellation", "damaged_item"]:
            reason = "Order record could not be verified in our active database."
            customer_msg = "We were unable to locate an active order matching your request. Please check the order number or contact our support team."
            policy_cited = "POL-REF-2024-V2"
        elif category == "cancellation" and order_data.get("status") == "in_transit":
            reason = "Order is currently in transit with carrier. Cancellation in transit is prohibited by POL-CAN-2024-V2."
            customer_msg = "Your order has already shipped and is in transit. It cannot be cancelled now, but you may return it for a refund once delivered."
            policy_cited = "POL-CAN-2024-V2 Clause 1.2"
        elif category == "refund" and not eligible:
            reason = f"Order delivered {order_data.get('days_since_delivery')} days ago, exceeding 30-day window."
            customer_msg = "Your purchase is outside our standard 30-day return window (POL-REF-2024-V2). A supervisor will review if an exception can be granted."
            policy_cited = "POL-REF-2024-V2 Clause 1.1"
        elif category == "refund" and eligible:
            reason = f"Valid refund request within 30-day window for order {order_data.get('order_id')}."
            customer_msg = f"Your refund request for ${amount:.2f} has been prepared. As required by our financial security policies, it is awaiting final supervisor authorization."
            policy_cited = "POL-REF-2024-V2 Clause 1.1 & 2.1"
            requires_approval = True
        elif category == "replacement" and eligible:
            reason = f"Valid replacement request for {order_data.get('product_name')} within 14-day window."
            customer_msg = "We have initiated a complimentary expedited replacement for your item under our 14-day replacement guarantee (POL-REP-2024-V2)."
            policy_cited = "POL-REP-2024-V2 Clause 1.1"
        elif category == "damaged_item":
            reason = "Damaged product claim accepted for replacement or refund."
            customer_msg = "We apologize for the damaged package. We have filed a claim under POL-DAM-2024-V2 for priority replacement."
            policy_cited = "POL-DAM-2024-V2 Clause 2.1"
        elif category == "shipping_delay":
            reason = "Carrier trace and status inquiry dispatched."
            customer_msg = f"We have contacted carrier {order_data.get('carrier', 'ExpressShip')} for tracking number {order_data.get('tracking_number', 'N/A')} to resolve the delivery delay."
            policy_cited = "POL-DEL-2024-V2 Clause 2.1"
        else:
            reason = "General inquiry addressed with policy guidelines."
            customer_msg = "Thank you for reaching out to ResolveAI. Our customer care team is glad to assist you."
            policy_cited = "POL-REF-2024-V2"

        proposal = {
            "action": action,
            "proposed_amount": amount if action == "refund" else 0.0,
            "eligible": eligible,
            "requires_approval": requires_approval,
            "reason": reason,
            "policy_cited": policy_cited,
            "customer_message": customer_msg
        }

    # Strict enforcement: override requires_approval if deterministic rules dictate it!
    if validation.get("requires_human_approval"):
        proposal["requires_approval"] = True

    proposal["execution_mode"] = execution_mode
    proposal["model"] = settings.QWEN_MODEL if execution_mode == ExecutionMode.LIVE_QWEN.value else "deterministic_rules"
    proposal["execution_time_ms"] = (time.time() - start_time) * 1000
    proposal["validation_flags"] = validation.get("flags", [])
    return proposal
