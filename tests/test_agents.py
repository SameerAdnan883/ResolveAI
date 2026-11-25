import pytest
from agents.triage_agent import run_triage_agent, heuristic_triage
from agents.order_agent import run_order_agent
from agents.policy_agent import run_policy_agent
from agents.resolution_agent import run_resolution_agent
from agents.quality_agent import run_quality_agent

def test_triage_agent():
    complaint = "My package for ORD-1001 arrived smashed and the glass is cracked."
    res = run_triage_agent(complaint)
    assert res["category"] == "damaged_item"
    assert res["order_id"] == "ORD-1001"
    assert "intent" in res
    assert res["priority"] in ["urgent", "high", "medium", "low"]

def test_order_agent():
    res = run_order_agent("ORD-1001", customer_id=2)
    assert res["order_found"] is True
    assert res["order_data"]["amount"] == 189.99
    assert "investigation_summary" in res

def test_policy_agent():
    res = run_policy_agent("refund", "Customer requesting money back after 10 days")
    assert res["category"] == "refund"
    assert "citations" in res

def test_resolution_agent():
    order_data = {"found": True, "amount": 189.99, "status": "delivered", "days_since_delivery": 6, "product_name": "Headphones"}
    citations = [{"source": "refund_policy.md", "section": "Eligibility", "content": "30 days"}]
    res = run_resolution_agent("My headphones are defective", "replacement", order_data, citations)
    assert "action" in res
    assert "requires_approval" in res
    assert res["requires_approval"] is True  # $189.99 exceeds threshold!

def test_quality_agent():
    order_data = {"found": True, "amount": 200.0, "status": "in_transit"}
    proposal = {"action": "cancellation", "requires_approval": False}
    quality = run_quality_agent("Cancel order", order_data, [], proposal)
    assert quality["escalation_required"] is True
    assert len(quality["contradictions"]) > 0  # In-transit cancellation contradiction

