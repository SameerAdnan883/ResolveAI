from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from database.connection import get_db
from database.models import Ticket, Approval, AgentLog, Customer
from backend.dependencies import get_current_admin

router = APIRouter(prefix="/analytics", tags=["Admin Analytics"])

@router.get("/summary")
def get_analytics_summary(
    current_admin: Customer = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Computes real-time SaaS operations KPIs."""
    total_tickets = db.query(Ticket).count()
    pending_tickets = db.query(Ticket).filter(Ticket.status.in_(["open", "investigating"])).count()
    pending_approvals = db.query(Approval).filter(Approval.status == "pending").count()
    resolved_tickets = db.query(Ticket).filter(Ticket.status == "resolved").count()
    escalated_tickets = db.query(Ticket).filter(Ticket.status == "pending_approval").count()
    closed_tickets = db.query(Ticket).filter(Ticket.status == "closed").count()

    # Category breakdown
    category_counts = (
        db.query(Ticket.category, func.count(Ticket.id))
        .group_by(Ticket.category)
        .all()
    )
    by_category = {cat: count for cat, count in category_counts}

    # Priority breakdown
    priority_counts = (
        db.query(Ticket.priority, func.count(Ticket.id))
        .group_by(Ticket.priority)
        .all()
    )
    by_priority = {prio: count for prio, count in priority_counts}

    # Agent latency statistics
    avg_latency = db.query(func.avg(AgentLog.execution_time_ms)).scalar() or 0.0

    return {
        "total_tickets": total_tickets,
        "pending_tickets": pending_tickets,
        "pending_approvals": pending_approvals,
        "resolved_tickets": resolved_tickets,
        "escalated_tickets": escalated_tickets,
        "closed_tickets": closed_tickets,
        "by_category": by_category,
        "by_priority": by_priority,
        "average_agent_latency_ms": round(float(avg_latency), 2)
    }

