import os
import uuid
import sqlite3
import pytest
from langgraph.types import Command
from database.connection import SessionLocal
from database.models import Ticket, Approval
from graph.workflow import build_resolveai_graph, get_checkpointer
from services.approval_service import review_pending_approval

def test_native_langgraph_interrupt_and_resume():
    """
    Tests the complete native LangGraph interrupt and persistent checkpoint resume lifecycle:
    1. A sensitive high-value complaint triggers native interrupt().
    2. Workflow state pauses in SQLite checkpointer with status pending_approval.
    3. Workflow resumes using Command(resume={'approved': True, 'admin_notes': '...'}).
    4. Graph completes to END with status resolved and resolution_applied=True.
    """
    test_db_path = f"./test_chk_{uuid.uuid4().hex[:6]}.db"
    checkpointer = get_checkpointer(test_db_path)
    app = build_resolveai_graph(checkpointer=checkpointer)

    ticket_id = f"TCK-INT-{uuid.uuid4().hex[:6].upper()}"
    thread_id = ticket_id
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "customer_id": 2,
        "authenticated_customer_id": 2,
        "is_admin": False,
        "ticket_id": ticket_id,
        "complaint_text": "I demand a full refund for my $189.99 headphones order ORD-1001.",
        "agent_logs": []
    }

    # 1. First invocation: Should hit interrupt() at quality_node
    result1 = app.invoke(initial_state, config=config)
    assert "__interrupt__" in result1 or result1.get("final_status") == "pending_approval"

    # Verify graph is paused at quality_node
    paused_state = app.get_state(config)
    assert len(paused_state.next) > 0  # Still has pending next step!
    assert "quality_node" in paused_state.next

    # 2. Resume graph using Command(resume=...)
    decision_payload = {
        "approved": True,
        "admin_notes": "Supervisor verified order and authorized refund under policy POL-REF-2024-V2.",
        "admin_id": 1
    }
    result2 = app.invoke(Command(resume=decision_payload), config=config)

    # 3. Post-resume verification: Graph completed to END!
    completed_state = app.get_state(config)
    assert len(completed_state.next) == 0  # Reached END!
    assert result2["final_status"] == "resolved"
    assert result2["resolution_applied"] is True
    assert result2["approval_decision"]["approved"] is True

    # Clean up test checkpoint db
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

def test_native_langgraph_rejection_prevents_action():
    """
    Tests that a supervisor rejection:
    1. Resumes the paused graph.
    2. Leaves resolution_applied=False.
    3. Sets final_status to 'closed'.
    """
    test_db_path = f"./test_chk_{uuid.uuid4().hex[:6]}.db"
    checkpointer = get_checkpointer(test_db_path)
    app = build_resolveai_graph(checkpointer=checkpointer)

    ticket_id = f"TCK-REJ-{uuid.uuid4().hex[:6].upper()}"
    thread_id = ticket_id
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "customer_id": 2,
        "authenticated_customer_id": 2,
        "is_admin": False,
        "ticket_id": ticket_id,
        "complaint_text": "I demand a refund for ORD-1001.",
        "agent_logs": []
    }

    app.invoke(initial_state, config=config)

    # Resume with REJECTION
    rejection_payload = {
        "approved": False,
        "admin_notes": "Item delivered > 30 days ago. Not eligible.",
        "admin_id": 1
    }
    result = app.invoke(Command(resume=rejection_payload), config=config)

    completed_state = app.get_state(config)
    assert len(completed_state.next) == 0
    assert result["final_status"] == "closed"
    assert result["resolution_applied"] is False

    if os.path.exists(test_db_path):
        os.remove(test_db_path)

def test_approval_service_prevents_duplicate_review():
    """Ensures an approval record cannot be reviewed or executed twice."""
    db = SessionLocal()
    ticket_id = f"TCK-DUP-{uuid.uuid4().hex[:6].upper()}"
    try:
        ticket = Ticket(
            id=ticket_id,
            customer_id=2,
            order_id="ORD-1001",
            title="Duplicate review safeguard test",
            category="refund",
            priority="high",
            status="pending_approval",
            complaint_text="Refund test"
        )
        db.add(ticket)
        db.commit()

        approval = Approval(
            ticket_id=ticket_id,
            action_type="refund",
            proposed_amount=150.00,
            status="pending"
        )
        db.add(approval)
        db.commit()

        # First review: must succeed
        res1 = review_pending_approval(approval.id, admin_id=1, approved=True, admin_notes="First approval", db=db)
        assert res1["success"] is True
        assert res1["status"] == "approved"

        # Second review attempt: must be rejected by idempotency safeguard!
        res2 = review_pending_approval(approval.id, admin_id=1, approved=True, admin_notes="Second approval attempt", db=db)
        assert res2["success"] is False
        assert "already finalized" in res2["error"]
    finally:
        db.close()

def test_native_langgraph_resume_after_server_restart():
    """
    Verifies that when a server process restarts after a workflow paused on interrupt():
    1. Workflow state remains safely persisted on disk in SQLite.
    2. A newly booted server instance can load the paused thread and resume execution.
    3. The resumed execution successfully completes to END.
    """
    test_db_path = f"./test_restart_{uuid.uuid4().hex[:6]}.db"
    ticket_id = f"TCK-RST-{uuid.uuid4().hex[:6].upper()}"
    thread_id = ticket_id
    config = {"configurable": {"thread_id": thread_id}}

    # Server Instance 1: Starts execution and hits interrupt()
    chk1 = get_checkpointer(test_db_path)
    app1 = build_resolveai_graph(checkpointer=chk1)

    initial_state = {
        "customer_id": 2,
        "authenticated_customer_id": 2,
        "is_admin": False,
        "ticket_id": ticket_id,
        "complaint_text": "I demand a refund for my $189.99 headphones order ORD-1001.",
        "agent_logs": []
    }
    app1.invoke(initial_state, config=config)

    # Simulate Server Shutdown: Destroy in-memory app and checkpointer instances
    del app1
    del chk1

    # Server Instance 2: Newly booted server loads from SQLite file on disk
    chk2 = get_checkpointer(test_db_path)
    app2 = build_resolveai_graph(checkpointer=chk2)

    # Verify state was restored from persistent SQLite storage
    persisted_state = app2.get_state(config)
    assert len(persisted_state.next) > 0
    assert "quality_node" in persisted_state.next

    # Resume execution on newly booted server
    resume_payload = {
        "approved": True,
        "admin_notes": "Supervisor approved after server restart.",
        "admin_id": 1
    }
    final_output = app2.invoke(Command(resume=resume_payload), config=config)

    assert final_output["final_status"] == "resolved"
    assert final_output["resolution_applied"] is True
    assert len(app2.get_state(config).next) == 0

    if os.path.exists(test_db_path):
        os.remove(test_db_path)


