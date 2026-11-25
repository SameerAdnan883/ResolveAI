from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database.connection import get_db
from database.models import Customer, Ticket, Approval, ConversationMessage, AgentLog
from backend.dependencies import get_current_user, get_current_admin
from services.resolution_service import process_customer_complaint
from services.approval_service import review_pending_approval

router = APIRouter(prefix="/tickets", tags=["Tickets & Resolutions"])

class ComplaintSubmission(BaseModel):
    complaint_text: str
    order_id: Optional[str] = None

class ApprovalDecisionRequest(BaseModel):
    approved: bool
    admin_notes: Optional[str] = None

@router.post("/submit")
def submit_complaint(
    req: ComplaintSubmission,
    current_user: Customer = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submits a customer complaint and triggers the autonomous multi-agent resolution workflow."""
    if not req.complaint_text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Complaint text cannot be empty.")

    full_complaint = req.complaint_text
    if req.order_id and req.order_id not in full_complaint:
        full_complaint = f"Order: {req.order_id}. {full_complaint}"

    result = process_customer_complaint(
        customer_id=current_user.id,
        complaint_text=full_complaint,
        db=db,
        is_admin=(current_user.role == "admin")
    )

    ticket_id = result.get("ticket_id")
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first() if ticket_id else None

    return {
        "ticket_id": ticket_id,
        "category": result.get("category"),
        "priority": result.get("priority"),
        "status": result.get("final_status"),
        "extracted_order_id": result.get("extracted_order_id"),
        "order_found": bool(result.get("order_data") and result["order_data"].get("found")),
        "human_approval_required": result.get("human_approval_required", False),
        "proposed_resolution": result.get("proposed_resolution"),
        "quality_assessment": result.get("quality_assessment"),
        "agent_logs_count": len(result.get("agent_logs", [])),
        "ticket": {
            "id": ticket.id if ticket else ticket_id,
            "title": ticket.title if ticket else "",
            "status": ticket.status if ticket else result.get("final_status"),
            "resolution_summary": ticket.resolution_summary if ticket else ""
        }
    }

@router.get("/my")
def get_my_tickets(
    current_user: Customer = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves tickets created by the logged-in customer."""
    tickets = db.query(Ticket).filter(Ticket.customer_id == current_user.id).order_by(Ticket.created_at.desc()).all()
    return [
        {
            "id": t.id,
            "order_id": t.order_id,
            "title": t.title,
            "category": t.category,
            "priority": t.priority,
            "status": t.status,
            "complaint_text": t.complaint_text,
            "resolution_summary": t.resolution_summary,
            "human_needed": t.human_needed,
            "created_at": t.created_at.isoformat(),
            "updated_at": t.updated_at.isoformat() if t.updated_at else None
        }
        for t in tickets
    ]

@router.get("/all")
def get_all_tickets(
    status_filter: Optional[str] = None,
    current_admin: Customer = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Admin-only endpoint to list all customer tickets with optional status filtering."""
    query = db.query(Ticket)
    if status_filter:
        query = query.filter(Ticket.status == status_filter)
    tickets = query.order_by(Ticket.created_at.desc()).all()
    return [
        {
            "id": t.id,
            "customer_id": t.customer_id,
            "customer_email": t.customer.email if t.customer else "Unknown",
            "order_id": t.order_id,
            "title": t.title,
            "category": t.category,
            "priority": t.priority,
            "status": t.status,
            "complaint_text": t.complaint_text,
            "resolution_proposal": t.resolution_proposal,
            "resolution_summary": t.resolution_summary,
            "human_needed": t.human_needed,
            "escalation_reason": t.escalation_reason,
            "created_at": t.created_at.isoformat()
        }
        for t in tickets
    ]

@router.get("/{ticket_id}")
def get_ticket_details(
    ticket_id: str,
    current_user: Customer = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Fetches details, messages, agent logs, and approvals for a ticket."""
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found")

    if current_user.role != "admin" and ticket.customer_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    messages = [
        {
            "id": m.id,
            "sender": m.sender,
            "agent_name": m.agent_name,
            "content": m.content,
            "created_at": m.created_at.isoformat()
        }
        for m in ticket.messages
    ]

    agent_logs = [
        {
            "id": l.id,
            "agent_name": l.agent_name,
            "step_name": l.step_name,
            "input_data": l.input_data,
            "output_data": l.output_data,
            "execution_time_ms": l.execution_time_ms,
            "status": l.status,
            "created_at": l.created_at.isoformat()
        }
        for l in ticket.agent_logs
    ]

    approvals = [
        {
            "id": a.id,
            "action_type": a.action_type,
            "proposed_amount": a.proposed_amount,
            "status": a.status,
            "admin_notes": a.admin_notes,
            "resolution_applied": a.resolution_applied,
            "created_at": a.created_at.isoformat()
        }
        for a in ticket.approvals
    ]

    return {
        "id": ticket.id,
        "customer_id": ticket.customer_id,
        "order_id": ticket.order_id,
        "title": ticket.title,
        "category": ticket.category,
        "priority": ticket.priority,
        "status": ticket.status,
        "complaint_text": ticket.complaint_text,
        "resolution_proposal": ticket.resolution_proposal,
        "resolution_summary": ticket.resolution_summary,
        "human_needed": ticket.human_needed,
        "escalation_reason": ticket.escalation_reason,
        "created_at": ticket.created_at.isoformat(),
        "messages": messages,
        "agent_logs": agent_logs,
        "approvals": approvals
    }

@router.get("/approvals/pending")
def list_pending_approvals(
    current_admin: Customer = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Admin-only list of all pending human approval requests."""
    approvals = db.query(Approval).filter(Approval.status == "pending").all()
    results = []
    for a in approvals:
        t = a.ticket
        results.append({
            "approval_id": a.id,
            "ticket_id": a.ticket_id,
            "action_type": a.action_type,
            "proposed_amount": a.proposed_amount,
            "created_at": a.created_at.isoformat(),
            "ticket_title": t.title if t else "N/A",
            "category": t.category if t else "N/A",
            "order_id": t.order_id if t else "N/A",
            "complaint_text": t.complaint_text if t else "N/A",
            "escalation_reason": t.escalation_reason if t else "N/A"
        })
    return results

@router.post("/approvals/{approval_id}/review")
def review_approval(
    approval_id: int,
    req: ApprovalDecisionRequest,
    current_admin: Customer = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Enables admin to approve or reject a pending resolution."""
    res = review_pending_approval(
        approval_id=approval_id,
        admin_id=current_admin.id,
        approved=req.approved,
        admin_notes=req.admin_notes,
        db=db
    )
    if not res.get("success"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=res.get("error"))
    return res

