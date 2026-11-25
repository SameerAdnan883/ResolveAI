import os
import uuid
import pytest
from database.connection import SessionLocal
from database.models import Customer, Order, Ticket, Approval
from tools.order_tools import get_order_details, get_delivery_status, get_customer_history
from agents.order_agent import run_order_agent

def test_authorized_order_access():
    """Customer Alice (id=2) accessing her own order ORD-1001 must succeed."""
    details = get_order_details(
        order_id="ORD-1001",
        authenticated_customer_id=2,
        is_admin=False
    )
    assert details["found"] is True
    assert details["order_id"] == "ORD-1001"
    assert details["customer_id"] == 2
    assert "unauthorized" not in details

def test_unauthorized_order_access_rejected():
    """Customer Bob (id=3) attempting to access Alice's order ORD-1001 must be strictly rejected."""
    details = get_order_details(
        order_id="ORD-1001",
        authenticated_customer_id=3,
        is_admin=False
    )
    assert details["found"] is False
    assert details.get("unauthorized") is True
    assert "Security Error" in details.get("error", "")

def test_admin_cross_customer_order_access_permitted():
    """Administrator (is_admin=True) is authorized to inspect any customer's order for support oversight."""
    details = get_order_details(
        order_id="ORD-1001",
        authenticated_customer_id=1,  # Admin user ID
        is_admin=True
    )
    assert details["found"] is True
    assert details["order_id"] == "ORD-1001"
    assert details.get("unauthorized") is not True

def test_unauthorized_customer_history_access():
    """Customer Bob (id=3) trying to view Alice's (id=2) order history must be blocked."""
    history = get_customer_history(
        customer_id=2,
        authenticated_customer_id=3,
        is_admin=False
    )
    assert history["found"] is False
    assert history.get("unauthorized") is True

def test_order_agent_security_enforcement():
    """OrderAgent must report security violation when customer does not own order."""
    res = run_order_agent(
        order_id="ORD-1001",
        customer_id=3,
        authenticated_customer_id=3,
        is_admin=False
    )
    assert res["order_found"] is False
    assert res["is_unauthorized"] is True
    assert "Security Violation" in res["investigation_summary"]

def test_delivery_address_privacy_masking():
    """Customer lookup must mask street-level address to prevent unnecessary PII exposure."""
    # Alice looking up her own order (is_admin=False)
    cust_view = get_order_details("ORD-1001", authenticated_customer_id=2, is_admin=False)
    assert cust_view["found"] is True
    assert "***" in cust_view["delivery_address"]
    assert "742 Evergreen Terrace" not in cust_view["delivery_address"]

    # Admin inspecting order (is_admin=True) gets full operational address
    admin_view = get_order_details("ORD-1001", authenticated_customer_id=1, is_admin=True)
    assert admin_view["found"] is True
    assert "742 Evergreen Terrace" in admin_view["delivery_address"]

