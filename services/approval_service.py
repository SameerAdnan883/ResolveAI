import datetime
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from database.models import Ticket, Approval, ConversationMessage
from graph.workflow import resolveai_app
from langgraph.types import Command

def review_pending_approval(
    approval_id: int,
    admin_id: int,
    approved: bool,
    admin_notes: Optional[str],
    db: Session
) -> Dict[str, Any]:
    """
    Enables human administrators to review, approve, or reject escalated resolutions.
    Resumes native LangGraph workflow via Command(resume=...) and enforces single-execution safeguards.
    """
    approval = db.query(Approval).filter(Approval.id == approval_id).first()
    if not approval:
        return {"success": False, "error": f"Approval {approval_id} not found"}

    # Safeguard: Prevent duplicate execution of an already reviewed approval
    if approval.status != "pending":
        return {
            "success": False,
            "error": f"Approval {approval_id} was already finalized as '{approval.status}'. Cannot re-execute action."
        }

    ticket = db.query(Ticket).filter(Ticket.id == approval.ticket_id).first()
    if not ticket:
        return {"success": False, "error": f"Associated ticket {approval.ticket_id} not found"}

    now = datetime.datetime.utcnow()
    approval.admin_id = admin_id
    approval.admin_notes = admin_notes
    approval.reviewed_at = now

    # Resume the native LangGraph execution via Command(resume=...) on the thread checkpointer
    thread_config = {"configurable": {"thread_id": approval.ticket_id}}
    graph_resumed = False
    try:
        # Check if graph is currently paused on interrupt for this thread
        graph_state = resolveai_app.get_state(thread_config)
        if graph_state and graph_state.next:
            resolveai_app.invoke(
                Command(resume={
                    "approved": approved,
                    "admin_notes": admin_notes,
                    "admin_id": admin_id
                }),
                config=thread_config
            )
            graph_resumed = True
    except Exception as e:
        # If thread was not interrupted in graph, proceed with DB update safely
        pass

    if approved:
        approval.status = "approved"
        approval.resolution_applied = True
        ticket.status = "resolved"
        ticket.human_needed = False
        summary = f"Approved by Supervisor: {approval.action_type.capitalize()} of ${approval.proposed_amount:.2f} executed."
        if admin_notes:
            summary += f" Notes: {admin_notes}"
        ticket.resolution_summary = summary

        msg = ConversationMessage(
            ticket_id=ticket.id,
            sender="admin",
            agent_name="Support Supervisor",
            content=f"Your resolution ({approval.action_type}) has been officially authorized and completed. Notes: {admin_notes or 'Authorized under standard policy'}",
            created_at=now
        )
        db.add(msg)
    else:
        approval.status = "rejected"
        approval.resolution_applied = False
        ticket.status = "closed"
        ticket.human_needed = False
        summary = f"Supervisor rejected proposal: {approval.action_type.capitalize()}."
        if admin_notes:
            summary += f" Reason: {admin_notes}"
        ticket.resolution_summary = summary

        msg = ConversationMessage(
            ticket_id=ticket.id,
            sender="admin",
            agent_name="Support Supervisor",
            content=f"Your request was reviewed by a supervisor: {admin_notes or 'Proposal could not be approved under current terms.'}",
            created_at=now
        )
        db.add(msg)

    ticket.updated_at = now
    db.commit()

    return {
        "success": True,
        "approval_id": approval.id,
        "ticket_id": ticket.id,
        "status": approval.status,
        "resolution_applied": approval.resolution_applied,
        "ticket_status": ticket.status,
        "graph_resumed": graph_resumed
    }

