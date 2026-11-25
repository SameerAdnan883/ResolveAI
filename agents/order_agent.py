import time
from typing import Dict, Any, Optional
from tools.order_tools import get_order_details, get_delivery_status, get_customer_history

def run_order_agent(
    order_id: Optional[str],
    customer_id: int,
    authenticated_customer_id: Optional[int] = None,
    is_admin: bool = False
) -> Dict[str, Any]:
    """
    Executes the Order Investigation Agent.
    Retrieves verifiable database information about the order, delivery status,
    and customer history while strictly enforcing order ownership security.
    """
    start_time = time.time()
    
    # Never trust customer_id from unverified sources; bind to authenticated identity
    effective_customer_id = authenticated_customer_id if (authenticated_customer_id is not None and not is_admin) else customer_id

    order_data = None
    delivery_status = None
    customer_history = None
    
    if order_id:
        order_data = get_order_details(
            order_id,
            authenticated_customer_id=authenticated_customer_id,
            is_admin=is_admin
        )
        if order_data.get("found"):
            delivery_status = get_delivery_status(
                order_id,
                authenticated_customer_id=authenticated_customer_id,
                is_admin=is_admin
            )

    if effective_customer_id:
        customer_history = get_customer_history(
            effective_customer_id,
            authenticated_customer_id=authenticated_customer_id,
            is_admin=is_admin
        )

    order_found = bool(order_data and order_data.get("found"))
    is_unauthorized = bool(order_data and order_data.get("unauthorized"))
    
    summary = ""
    if is_unauthorized:
        summary = f"Security Violation: Authenticated customer is not authorized to access Order {order_id}."
    elif order_found:
        summary = (
            f"Verified Order {order_id}: {order_data['product_name']} "
            f"(${order_data['amount']:.2f}, status: {order_data['status']}). "
            f"Delivered {order_data.get('days_since_delivery', 'N/A')} days ago."
        )
    elif order_id:
        summary = f"Order {order_id} could not be found in active order database."
    else:
        summary = "No order ID specified in customer complaint."

    return {
        "order_found": order_found,
        "is_unauthorized": is_unauthorized,
        "order_data": order_data,
        "delivery_status": delivery_status,
        "customer_history": customer_history,
        "investigation_summary": summary,
        "execution_mode": "DETERMINISTIC_TOOL",
        "model": "sqlite_database_tool",
        "execution_time_ms": (time.time() - start_time) * 1000
    }

