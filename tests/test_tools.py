import pytest
from tools.order_tools import get_order_details, get_delivery_status, get_customer_history
from tools.ticket_tools import create_support_ticket, update_ticket_status, escalate_ticket
from tools.validation_tools import validate_resolution

def test_order_tools():
    details = get_order_details("ORD-1001")
    assert details["found"] is True
    assert details["amount"] == 189.99
    assert details["status"] == "delivered"

    status = get_delivery_status("ORD-1001")
    assert status["status"] == "delivered"

    history = get_customer_history(2)
    assert history["found"] is True
    assert history["total_orders"] >= 2

def test_missing_order_tool():
    details = get_order_details("ORD-999999")
    assert details["found"] is False

def test_ticket_tools():
    ticket = create_support_ticket(
        customer_id=2,
        complaint_text="Test complaint",
        category="shipping_delay",
        priority="low",
        order_id="ORD-1001"
    )
    assert ticket["ticket_id"].startswith("TCK-")
    assert ticket["status"] == "open"

    update_res = update_ticket_status(
        ticket_id=ticket["ticket_id"],
        status="resolved",
        resolution_summary="Resolved successfully"
    )
    assert update_res["success"] is True

    escalate_res = escalate_ticket(
        ticket_id=ticket["ticket_id"],
        reason="Test high value escalation",
        proposed_action="refund",
        amount=200.00
    )
    assert escalate_res["success"] is True
    assert escalate_res["status"] == "pending_approval"

def test_validation_deterministic_rules():
    # Rule 1: High value refund requires supervisor approval
    order_data = {"found": True, "amount": 250.0, "status": "delivered", "days_since_delivery": 5}
    val = validate_resolution(order_data, [], "refund", "refund")
    assert val["requires_human_approval"] is True

    # Rule 2: In-transit cancellation prohibited
    order_in_transit = {"found": True, "amount": 80.0, "status": "in_transit"}
    val_cancel = validate_resolution(order_in_transit, [], "cancellation", "cancellation")
    assert val_cancel["is_valid"] is False
    assert val_cancel["approved_action"] == "await_delivery_and_return"

    # Rule 3: Expired return window (>30 days)
    order_expired = {"found": True, "amount": 50.0, "status": "delivered", "days_since_delivery": 40}
    val_expired = validate_resolution(order_expired, [], "refund", "refund")
    assert val_expired["is_valid"] is False

