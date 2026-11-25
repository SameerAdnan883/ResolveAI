from typing import Dict, Any, List
from backend.config import settings

def validate_resolution(
    order_data: Dict[str, Any],
    policy_evidence: List[Dict[str, Any]],
    proposed_action: str,
    complaint_category: str
) -> Dict[str, Any]:
    """
    Deterministic Business Rules Validator.
    Applies strict, non-negotiable enterprise constraints that LLMs cannot bypass.
    
    Guarantees:
    1. NEVER auto-execute a refund without human approval if threshold exceeded.
    2. Enforces calendar return window (30 days for return, 14 days for replacement, 7 days for damage).
    3. Blocks in-transit cancellations.
    """
    proposed_action = proposed_action.lower().strip()
    amount = float(order_data.get("amount", 0.0)) if order_data.get("found") else 0.0
    status = order_data.get("status", "").lower()
    days_since_delivery = order_data.get("days_since_delivery")
    
    flags = []
    requires_human_approval = False
    is_valid = True
    reason = "Valid resolution proposal under policy constraints."

    # Rule 1: Zero direct LLM refund authorization safeguard
    if proposed_action == "refund":
        # Always require human supervisor sign-off if amount > AUTO_REFUND_MAX_AMOUNT (default 0.0)
        # or if amount > HIGH_VALUE_THRESHOLD ($150)
        requires_human_approval = True
        if amount > settings.HIGH_VALUE_THRESHOLD:
            flags.append(f"High-value refund (${amount:.2f} > ${settings.HIGH_VALUE_THRESHOLD:.2f}) requires human supervisor authorization.")
        else:
            flags.append("Financial disbursement requires supervisor confirmation.")

    # Rule 2: Order existence check
    if not order_data.get("found", False) and complaint_category in ["refund", "replacement", "cancellation", "damaged_item"]:
        is_valid = False
        requires_human_approval = True
        flags.append("Order ID not found in database; cannot authorize resolution without manual verification.")
        return {
            "is_valid": False,
            "requires_human_approval": True,
            "approved_action": "manual_review",
            "flags": flags,
            "reason": "Order verification failed."
        }

    # Rule 3: Cancellation policy constraints (POL-CAN-2024-V2)
    if proposed_action == "cancel" or complaint_category == "cancellation":
        if status in ["in_transit", "shipped"]:
            is_valid = False
            requires_human_approval = False
            flags.append("Order has already shipped and is in transit. In-transit cancellation prohibited by POL-CAN-2024-V2.")
            return {
                "is_valid": False,
                "requires_human_approval": False,
                "approved_action": "await_delivery_and_return",
                "flags": flags,
                "reason": "Shipment in transit cannot be intercepted. Customer must receive and initiate standard return."
            }
        elif status == "delivered":
            is_valid = False
            flags.append("Order is already delivered. Must be processed as a Return/Refund, not a Cancellation.")
            return {
                "is_valid": False,
                "requires_human_approval": False,
                "approved_action": "standard_return",
                "flags": flags,
                "reason": "Delivered order cannot be cancelled; initiate return."
            }

    # Rule 4: Return / Refund window constraint (POL-REF-2024-V2, 30 days)
    if proposed_action == "refund" or complaint_category == "refund":
        if days_since_delivery is not None and days_since_delivery > 30:
            is_valid = False
            flags.append(f"Delivered {days_since_delivery} days ago, which exceeds the 30-day return window.")
            return {
                "is_valid": False,
                "requires_human_approval": True,  # Escalated if customer requests exception
                "approved_action": "reject_or_manager_exception",
                "flags": flags,
                "reason": "Past 30-day return window (POL-REF-2024-V2 Clause 1.1). Requires manager exception to proceed."
            }

    # Rule 5: Replacement window constraint (POL-REP-2024-V2, 14 days)
    if proposed_action == "replacement" or complaint_category == "replacement":
        if days_since_delivery is not None and days_since_delivery > 14:
            is_valid = False
            flags.append(f"Delivered {days_since_delivery} days ago, exceeding the 14-day replacement window.")
            return {
                "is_valid": False,
                "requires_human_approval": True,
                "approved_action": "manual_review",
                "flags": flags,
                "reason": "Past 14-day replacement window (POL-REP-2024-V2 Clause 1.2). Supervisor review needed."
            }
        if amount > settings.HIGH_VALUE_THRESHOLD:
            requires_human_approval = True
            flags.append(f"High-value item replacement (${amount:.2f} > ${settings.HIGH_VALUE_THRESHOLD:.2f}) requires supervisor approval.")

    # Rule 6: Damaged Item Reporting Window (POL-DAM-2024-V2, 7 days)
    if complaint_category == "damaged_item":
        if days_since_delivery is not None and days_since_delivery > 7:
            flags.append(f"Damage reported {days_since_delivery} days after delivery (standard window is 7 days).")
            requires_human_approval = True
        if amount > settings.HIGH_VALUE_THRESHOLD:
            requires_human_approval = True
            flags.append(f"High-value damaged item (${amount:.2f}) requires proof verification and supervisor sign-off.")

    return {
        "is_valid": is_valid,
        "requires_human_approval": requires_human_approval,
        "approved_action": proposed_action,
        "flags": flags,
        "reason": reason if not flags else "; ".join(flags)
    }

