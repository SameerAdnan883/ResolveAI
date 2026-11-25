import uuid
import pytest
from database.connection import SessionLocal
from database.models import Customer, Ticket, Approval
from services.approval_service import review_pending_approval

def test_human_approval_workflow():
    db = SessionLocal()
    test_ticket_id = f"TCK-APP-{uuid.uuid4().hex[:6].upper()}"
    try:
        # Create a pending ticket and approval
        ticket = Ticket(
            id=test_ticket_id,
            customer_id=2,
            order_id="ORD-1001",
            title="High value refund test",
            category="refund",
            priority="urgent",
            status="pending_approval",
            complaint_text="Refund request for high value order ORD-1001",
            human_needed=True,
            escalation_reason="Refund over $150 threshold"
        )
        db.add(ticket)
        db.commit()

        approval = Approval(
            ticket_id=test_ticket_id,
            action_type="refund",
            proposed_amount=189.99,
            status="pending"
        )
        db.add(approval)
        db.commit()

        # Admin approves
        res = review_pending_approval(
            approval_id=approval.id,
            admin_id=1,
            approved=True,
            admin_notes="Approved after receipt verification.",
            db=db
        )

        assert res["success"] is True
        assert res["status"] == "approved"
        assert res["resolution_applied"] is True

        db.refresh(ticket)
        assert ticket.status == "resolved"
        assert ticket.human_needed is False
    finally:
        db.close()
