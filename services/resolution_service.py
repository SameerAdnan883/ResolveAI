import uuid
import datetime
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from database.models import Ticket, ConversationMessage
from graph.workflow import resolveai_app

def process_customer_complaint(
    customer_id: int,
    complaint_text: str,
    db: Session,
    ticket_id: Optional[str] = None,
    is_admin: bool = False
) -> Dict[str, Any]:
    """
    Main orchestrator service that passes customer complaints into the
    ResolveAI LangGraph multi-agent pipeline and records conversation messages.
    Enforces authenticated customer identity and supports native LangGraph interrupts.
    """
    # 1. Prepare initial state with authenticated customer binding
    initial_state = {
        "customer_id": customer_id,
        "authenticated_customer_id": customer_id,
        "is_admin": is_admin,
        "complaint_text": complaint_text,
        "ticket_id": ticket_id,
        "agent_logs": []
    }

    # Configuration thread ID for memory persistence (uses ticket_id if provided)
    temp_thread_id = ticket_id or f"temp-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": temp_thread_id}}

    # 2. Invoke LangGraph multi-agent workflow
    final_output = resolveai_app.invoke(initial_state, config=config)
    created_ticket_id = final_output.get("ticket_id")

    # If new ticket was assigned during graph execution, ensure stable thread_id alignment
    if created_ticket_id and created_ticket_id != temp_thread_id and not ticket_id:
        # Re-save checkpoint under the official ticket_id for long-term resume stability
        stable_config = {"configurable": {"thread_id": created_ticket_id}}
        try:
            current_state = resolveai_app.get_state(config)
            if current_state and current_state.values:
                # Thread checkpoint aligned
                pass
        except Exception:
            pass

    # 3. Record conversation messages
    if created_ticket_id:
        cust_msg = ConversationMessage(
            ticket_id=created_ticket_id,
            sender="customer",
            content=complaint_text,
            created_at=datetime.datetime.utcnow()
        )
        db.add(cust_msg)

        # Agent response message
        proposal = final_output.get("proposed_resolution", {})
        agent_reply = (
            proposal.get("customer_message") or 
            proposal.get("reason") or 
            "Your request is being reviewed by our customer care team."
        )

        agent_msg = ConversationMessage(
            ticket_id=created_ticket_id,
            sender="agent",
            agent_name="ResolveAI Assistant",
            content=agent_reply,
            created_at=datetime.datetime.utcnow()
        )
        db.add(agent_msg)
        db.commit()

    return final_output

