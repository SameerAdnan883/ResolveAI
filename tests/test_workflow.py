import uuid
import pytest
from langgraph.types import Command
from graph.workflow import resolveai_app

def test_langgraph_full_workflow_resolution():
    initial_state = {
        "customer_id": 2,
        "authenticated_customer_id": 2,
        "complaint_text": "I bought headphones ORD-1001, but the microphone is broken. Can I get a replacement?",
        "agent_logs": []
    }
    config = {"configurable": {"thread_id": f"test-thread-wf-{uuid.uuid4().hex[:8]}"}}
    output = resolveai_app.invoke(initial_state, config=config)

    # If the workflow paused on native LangGraph interrupt (high-value $189.99 replacement):
    if "__interrupt__" in output:
        output = resolveai_app.invoke(
            Command(resume={"approved": True, "admin_notes": "Supervisor approved under warranty policy."}),
            config=config
        )

    assert output["category"] in ["replacement", "damaged_item"]
    assert output["extracted_order_id"] == "ORD-1001"
    assert output["order_data"] is not None
    assert output["order_data"]["found"] is True
    assert output["proposed_resolution"] is not None
    assert output["quality_assessment"] is not None
    assert len(output["agent_logs"]) >= 5  # All 5 agents logged!

def test_langgraph_dynamic_routing_general_inquiry():
    # General inquiry has no order ID; graph routes dynamically directly to policy
    initial_state = {
        "customer_id": 3,
        "authenticated_customer_id": 3,
        "complaint_text": "What are your standard business hours and return policies?",
        "agent_logs": []
    }
    config = {"configurable": {"thread_id": f"test-thread-wf-inq-{uuid.uuid4().hex[:8]}"}}
    output = resolveai_app.invoke(initial_state, config=config)

    assert output["category"] == "general_inquiry"
    assert output["proposed_resolution"] is not None

