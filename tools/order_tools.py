import datetime
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from database.connection import SessionLocal
from database.models import Order, Customer

def mask_address(address: Optional[str]) -> Optional[str]:
    """Masks street-level address to prevent unnecessary PII exposure."""
    if not address:
        return None
    parts = [p.strip() for p in address.split(",")]
    if len(parts) > 1:
        return "***, " + ", ".join(parts[1:])
    return "***"


def get_order_details(
    order_id: str,
    authenticated_customer_id: Optional[int] = None,
    is_admin: bool = False,
    db: Optional[Session] = None
) -> Dict[str, Any]:
    """
    Retrieves full order information, strictly enforcing customer ownership security.
    Rejects unauthorized access if a non-admin customer attempts to inspect another customer's order.
    """
    local_db = db is None
    session = SessionLocal() if local_db else db

    try:
        clean_id = order_id.strip().upper()
        order = session.query(Order).filter(Order.id == clean_id).first()
        if not order:
            return {
                "found": False,
                "order_id": order_id,
                "error": f"Order {order_id} not found in database."
            }

        # Order Ownership Security Check
        if authenticated_customer_id is not None and not is_admin:
            if order.customer_id != authenticated_customer_id:
                return {
                    "found": False,
                    "order_id": order.id,
                    "unauthorized": True,
                    "error": f"Security Error: Order {order.id} belongs to a different customer account. Access denied."
                }

        now = datetime.datetime.utcnow()
        days_since_delivery = None
        if order.delivery_date:
            days_since_delivery = (now - order.delivery_date).days

        days_since_order = (now - order.order_date).days

        return {
            "found": True,
            "order_id": order.id,
            "customer_id": order.customer_id,
            "product_name": order.product_name,
            "item_category": order.item_category,
            "amount": float(order.amount),
            "currency": order.currency,
            "status": order.status,
            "order_date": order.order_date.isoformat() if order.order_date else None,
            "delivery_date": order.delivery_date.isoformat() if order.delivery_date else None,
            "days_since_delivery": days_since_delivery,
            "days_since_order": days_since_order,
            "tracking_number": order.tracking_number,
            "carrier": order.carrier,
            "delivery_address": order.delivery_address if is_admin else mask_address(order.delivery_address),
            "return_window_days": order.return_window_days
        }
    finally:
        if local_db:
            session.close()

def get_delivery_status(
    order_id: str,
    authenticated_customer_id: Optional[int] = None,
    is_admin: bool = False,
    db: Optional[Session] = None
) -> Dict[str, Any]:
    """Checks the delivery and shipping status for a specific order with ownership verification."""
    details = get_order_details(
        order_id,
        authenticated_customer_id=authenticated_customer_id,
        is_admin=is_admin,
        db=db
    )
    if not details.get("found"):
        return details

    status = details["status"]
    tracking = details.get("tracking_number")
    carrier = details.get("carrier")

    is_delayed = False
    if status == "in_transit" and details.get("days_since_order", 0) > 5:
        is_delayed = True

    return {
        "order_id": order_id,
        "status": status,
        "carrier": carrier,
        "tracking_number": tracking,
        "is_delayed": is_delayed,
        "delivery_date": details.get("delivery_date"),
        "delivery_address": details.get("delivery_address")
    }

def get_customer_history(
    customer_id: int,
    authenticated_customer_id: Optional[int] = None,
    is_admin: bool = False,
    db: Optional[Session] = None
) -> Dict[str, Any]:
    """Retrieves customer purchase history with identity verification."""
    # Enforce identity authorization
    effective_id = customer_id
    if authenticated_customer_id is not None and not is_admin:
        if customer_id != authenticated_customer_id:
            return {
                "found": False,
                "customer_id": customer_id,
                "unauthorized": True,
                "error": "Security Error: Unauthorized attempt to view customer history."
            }
        effective_id = authenticated_customer_id

    local_db = db is None
    session = SessionLocal() if local_db else db

    try:
        customer = session.query(Customer).filter(Customer.id == effective_id).first()
        if not customer:
            return {"found": False, "customer_id": effective_id, "error": "Customer not found"}

        orders = session.query(Order).filter(Order.customer_id == effective_id).all()
        orders_list = [
            {
                "order_id": o.id,
                "product_name": o.product_name,
                "amount": float(o.amount),
                "status": o.status,
                "order_date": o.order_date.isoformat() if o.order_date else None
            }
            for o in orders
        ]

        return {
            "found": True,
            "customer_id": customer.id,
            "full_name": customer.full_name,
            "email": customer.email,
            "total_orders": len(orders),
            "orders": orders_list
        }
    finally:
        if local_db:
            session.close()

