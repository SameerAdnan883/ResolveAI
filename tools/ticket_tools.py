import uuid
import datetime
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from database.connection import SessionLocal
from database.models import Ticket, Approval, AgentLog

def create_support_ticket(
    customer_id: int,
    complaint_text: str,
    category: str,
    priority: str = "medium",
    order_id: Optional[str] = None,
    db: Optional[Session] = None
) -> Dict[str, Any]:
    """Creates a new support ticket in the database with unique ticket ID."""
    local_db = db is None
    session = SessionLocal() if local_db else db

    try:
        ticket_id = f"TCK-{uuid.uuid4().hex[:6].upper()}"
        title = f"{category.replace('_', ' ').capitalize()} - {order_id if order_id else 'General'}"

        ticket = Ticket(
            id=ticket_id,
            customer_id=customer_id,
            order_id=order_id.strip().upper() if order_id else None,
            title=title,
            category=category,
            priority=priority,
            status="open",
            complaint_text=complaint_text
        )
        session.add(ticket)
        session.commit()
        session.refresh(ticket)

        return {
            "ticket_id": ticket.id,
            "status": ticket.status,
            "title": ticket.title,
            "category": ticket.category,
            "priority": ticket.priority,
            "created_at": ticket.created_at.isoformat()
        }
    except Exception as e:
        session.rollback()
        raise e
    finally:
        if local_db:
            session.close()

def update_ticket_status(
    ticket_id: str,
    status: str,
    resolution_summary: Optional[str] = None,
    proposal: Optional[dict] = None,
    human_needed: bool = False,
    escalation_reason: Optional[str] = None,
    db: Optional[Session] = None
) -> Dict[str, Any]:
    """Updates a support ticket status, proposed resolution, and human escalation flags."""
    local_db = db is None
    session = SessionLocal() if local_db else db

    try:
        ticket = session.query(Ticket).filter(Ticket.id == ticket_id).first()
        if not ticket:
            return {"success": False, "error": f"Ticket {ticket_id} not found"}

        ticket.status = status
        if resolution_summary:
            ticket.resolution_summary = resolution_summary
        if proposal:
            ticket.resolution_proposal = proposal
        ticket.human_needed = human_needed
        if escalation_reason:
            ticket.escalation_reason = escalation_reason
        ticket.updated_at = datetime.datetime.utcnow()

        session.commit()
        return {
            "success": True,
            "ticket_id": ticket_id,
            "status": ticket.status,
            "human_needed": ticket.human_needed
        }
    except Exception as e:
        session.rollback()
        return {"success": False, "error": str(e)}
    finally:
        if local_db:
            session.close()

def escalate_ticket(
    ticket_id: str,
    reason: str,
    proposed_action: str,
    amount: float = 0.0,
    db: Optional[Session] = None
) -> Dict[str, Any]:
    """Escalates a ticket for human administrator approval and records an Approval entry."""
    local_db = db is None
    session = SessionLocal() if local_db else db

    try:
        ticket = session.query(Ticket).filter(Ticket.id == ticket_id).first()
        if not ticket:
            return {"success": False, "error": f"Ticket {ticket_id} not found"}

        ticket.status = "pending_approval"
        ticket.human_needed = True
        ticket.escalation_reason = reason
        ticket.updated_at = datetime.datetime.utcnow()

        approval = Approval(
            ticket_id=ticket_id,
            action_type=proposed_action,
            proposed_amount=amount,
            status="pending",
            resolution_applied=False,
            created_at=datetime.datetime.utcnow()
        )
        session.add(approval)
        session.commit()

        return {
            "success": True,
            "ticket_id": ticket_id,
            "approval_id": approval.id,
            "action_type": proposed_action,
            "status": "pending_approval",
            "reason": reason
        }
    except Exception as e:
        session.rollback()
        return {"success": False, "error": str(e)}
    finally:
        if local_db:
            session.close()

