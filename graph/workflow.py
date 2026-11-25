import os
import time
import sqlite3
from typing import Dict, Any, Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

from backend.config import settings
from graph.state import SupportGraphState
from agents.triage_agent import run_triage_agent
from agents.order_agent import run_order_agent
from agents.policy_agent import run_policy_agent
from agents.resolution_agent import run_resolution_agent
from agents.quality_agent import run_quality_agent
from tools.ticket_tools import create_support_ticket, update_ticket_status, escalate_ticket
from database.connection import SessionLocal
from database.models import AgentLog, ConversationMessage, Ticket, Approval


def get_checkpointer(db_path: Optional[str] = None) -> SqliteSaver:
    """Returns a persistent SQLite checkpointer instance for durable graph state across restarts."""
    checkpoint_file = db_path or settings.CHECKPOINT_DATABASE_URL
    conn = sqlite3.connect(checkpoint_file, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def log_agent_execution(ticket_id: str, agent_name: str, step_name: str, input_data: Any, output_data: Any, duration_ms: float):
    """Utility to persist agent execution telemetry in the database."""
    if not ticket_id:
        return
    session = SessionLocal()
    try:
        log_entry = AgentLog(
            ticket_id=ticket_id,
            agent_name=agent_name,
            step_name=step_name,
            input_data=input_data,
            output_data=output_data,
            execution_time_ms=duration_ms,
            status="success"
        )
        session.add(log_entry)
        session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()

# --- Node Implementations ---

def triage_node(state: SupportGraphState) -> Dict[str, Any]:
    """Agent 1: Understands complaint, extracts intent, category, priority, and order ID."""
    complaint = state.get("complaint_text", "")
    customer_id = state.get("customer_id", 1)
    auth_customer_id = state.get("authenticated_customer_id")
    
    triage_result = run_triage_agent(complaint)
    category = triage_result.get("category", "general_inquiry")
    priority = triage_result.get("priority", "medium")
    extracted_order_id = triage_result.get("order_id")
    
    # Create database ticket if not already initialized
    ticket_id = state.get("ticket_id")
    if not ticket_id:
        ticket_info = create_support_ticket(
            customer_id=auth_customer_id or customer_id,
            complaint_text=complaint,
            category=category,
            priority=priority,
            order_id=extracted_order_id
        )
        ticket_id = ticket_info["ticket_id"]

    log_agent_execution(
        ticket_id=ticket_id,
        agent_name="TriageAgent",
        step_name="classify_complaint",
        input_data={"complaint": complaint},
        output_data=triage_result,
        duration_ms=triage_result.get("execution_time_ms", 0.0)
    )

    logs = state.get("agent_logs", [])
    logs.append({"agent": "TriageAgent", "output": triage_result})

    return {
        "ticket_id": ticket_id,
        "category": category,
        "intent": triage_result.get("intent", ""),
        "priority": priority,
        "extracted_order_id": extracted_order_id,
        "agent_logs": logs
    }

def order_node(state: SupportGraphState) -> Dict[str, Any]:
    """Agent 2: Retrieves verifiable order details, delivery status, and customer history with ownership checks."""
    order_id = state.get("extracted_order_id")
    customer_id = state.get("customer_id", 1)
    auth_customer_id = state.get("authenticated_customer_id")
    is_admin = state.get("is_admin", False)
    ticket_id = state.get("ticket_id")

    order_result = run_order_agent(
        order_id=order_id,
        customer_id=customer_id,
        authenticated_customer_id=auth_customer_id,
        is_admin=is_admin
    )
    
    log_agent_execution(
        ticket_id=ticket_id,
        agent_name="OrderAgent",
        step_name="investigate_order",
        input_data={"order_id": order_id, "authenticated_customer_id": auth_customer_id, "is_admin": is_admin},
        output_data=order_result,
        duration_ms=order_result.get("execution_time_ms", 0.0)
    )

    logs = state.get("agent_logs", [])
    logs.append({"agent": "OrderAgent", "output": order_result})

    return {
        "order_data": order_result.get("order_data"),
        "customer_history": order_result.get("customer_history"),
        "agent_logs": logs
    }

def policy_node(state: SupportGraphState) -> Dict[str, Any]:
    """Agent 3: Performs semantic search against ChromaDB policy store for grounded citations."""
    category = state.get("category", "general_inquiry")
    complaint = state.get("complaint_text", "")
    ticket_id = state.get("ticket_id")
    order_status = None
    if state.get("order_data"):
        order_status = state["order_data"].get("status")

    policy_result = run_policy_agent(category=category, complaint_text=complaint, order_status=order_status)
    citations = policy_result.get("citations", [])

    log_agent_execution(
        ticket_id=ticket_id,
        agent_name="PolicyAgent",
        step_name="research_policy",
        input_data={"category": category, "complaint": complaint},
        output_data=policy_result,
        duration_ms=policy_result.get("execution_time_ms", 0.0)
    )

    logs = state.get("agent_logs", [])
    logs.append({"agent": "PolicyAgent", "output": policy_result})

    return {
        "policy_evidence": citations,
        "policy_summary": policy_result.get("evidence_summary"),
        "agent_logs": logs
    }

def resolution_node(state: SupportGraphState) -> Dict[str, Any]:
    """Agent 4: Combines complaint, order verification, and policy to formulate resolution proposal."""
    complaint = state.get("complaint_text", "")
    category = state.get("category", "general_inquiry")
    order_data = state.get("order_data")
    citations = state.get("policy_evidence", [])
    ticket_id = state.get("ticket_id")

    resolution_result = run_resolution_agent(
        complaint_text=complaint,
        category=category,
        order_data=order_data,
        policy_citations=citations
    )

    log_agent_execution(
        ticket_id=ticket_id,
        agent_name="ResolutionAgent",
        step_name="propose_resolution",
        input_data={"category": category, "order_found": bool(order_data and order_data.get("found"))},
        output_data=resolution_result,
        duration_ms=resolution_result.get("execution_time_ms", 0.0)
    )

    logs = state.get("agent_logs", [])
    logs.append({"agent": "ResolutionAgent", "output": resolution_result})

    return {
        "proposed_resolution": resolution_result,
        "agent_logs": logs
    }

def quality_node(state: SupportGraphState) -> Dict[str, Any]:
    """
    Agent 5: Audits resolution for hallucinations, policy compliance, security,
    and pauses via native LangGraph interrupt() when supervisor approval is needed.
    """
    complaint = state.get("complaint_text", "")
    order_data = state.get("order_data")
    citations = state.get("policy_evidence", [])
    proposal = state.get("proposed_resolution", {})
    ticket_id = state.get("ticket_id")
    logs = list(state.get("agent_logs", []))

    quality_result = run_quality_agent(
        complaint_text=complaint,
        order_data=order_data,
        policy_citations=citations,
        proposed_resolution=proposal
    )

    # Security check: if unauthorized order access was attempted, enforce escalation
    if order_data and order_data.get("unauthorized"):
        quality_result["escalation_required"] = True
        quality_result["escalation_reason"] = "Security Warning: Customer is not authorized to access the requested Order ID."

    escalate = quality_result.get("escalation_required", False)
    final_status = "pending_approval" if escalate else "resolved"
    approval_decision = state.get("approval_decision")
    resolution_applied = state.get("resolution_applied", False)

    # If already reviewed, do not pause again
    if escalate and not resolution_applied and approval_decision is None:
        if ticket_id:
            update_ticket_status(
                ticket_id=ticket_id,
                status="pending_approval",
                resolution_summary=proposal.get("customer_message") or proposal.get("reason"),
                proposal=proposal,
                human_needed=True,
                escalation_reason=quality_result.get("escalation_reason")
            )

            escalate_ticket(
                ticket_id=ticket_id,
                reason=quality_result.get("escalation_reason", "Human supervisor approval required"),
                proposed_action=proposal.get("action", "manual_review"),
                amount=float(proposal.get("proposed_amount", 0.0))
            )

        log_agent_execution(
            ticket_id=ticket_id,
            agent_name="QualityAgent",
            step_name="audit_and_pause_for_approval",
            input_data={"proposal": proposal.get("action")},
            output_data=quality_result,
            duration_ms=quality_result.get("execution_time_ms", 0.0)
        )

        logs = state.get("agent_logs", [])
        logs.append({"agent": "QualityAgent", "output": quality_result})

        # NATIVE LANGGRAPH INTERRUPT:
        # Halts graph execution right here. Durable checkpoint saved to SQLite.
        # Resumes when Command(resume={"approved": bool, "admin_notes": str, "admin_id": int}) is provided.
        decision = interrupt({
            "ticket_id": ticket_id,
            "action": proposal.get("action", "manual_review"),
            "proposed_amount": float(proposal.get("proposed_amount", 0.0)),
            "reason": quality_result.get("escalation_reason", "Supervisor authorization required"),
            "order_id": state.get("extracted_order_id")
        })

        # --- EXECUTION RESUMES HERE AFTER HUMAN APPROVAL/REJECTION ---
        if isinstance(decision, dict):
            approval_decision = decision
            is_approved = decision.get("approved", False)
            admin_notes = decision.get("admin_notes", "")
            action_type = proposal.get("action", "action").capitalize()

            if is_approved:
                final_status = "resolved"
                resolution_applied = True
                summary = f"Approved by Supervisor: {action_type} authorized."
                if admin_notes:
                    summary += f" Notes: {admin_notes}"
                if ticket_id:
                    update_ticket_status(
                        ticket_id=ticket_id,
                        status="resolved",
                        resolution_summary=summary,
                        human_needed=False
                    )
            else:
                final_status = "closed"
                resolution_applied = False
                summary = f"Supervisor rejected proposal: {action_type}."
                if admin_notes:
                    summary += f" Reason: {admin_notes}"
                if ticket_id:
                    update_ticket_status(
                        ticket_id=ticket_id,
                        status="closed",
                        resolution_summary=summary,
                        human_needed=False
                    )

    elif not escalate:
        if ticket_id:
            update_ticket_status(
                ticket_id=ticket_id,
                status="resolved",
                resolution_summary=proposal.get("customer_message") or proposal.get("reason"),
                proposal=proposal,
                human_needed=False
            )

        log_agent_execution(
            ticket_id=ticket_id,
            agent_name="QualityAgent",
            step_name="audit_and_escalate",
            input_data={"proposal": proposal.get("action")},
            output_data=quality_result,
            duration_ms=quality_result.get("execution_time_ms", 0.0)
        )

        logs = state.get("agent_logs", [])
        logs.append({"agent": "QualityAgent", "output": quality_result})

    return {
        "quality_assessment": quality_result,
        "human_approval_required": escalate,
        "approval_decision": approval_decision,
        "resolution_applied": resolution_applied,
        "final_status": final_status,
        "agent_logs": logs
    }

# --- Conditional Edge Routing ---

def route_triage(state: SupportGraphState) -> Literal["order_node", "policy_node"]:
    """
    Dynamic routing condition:
    If the complaint has an order ID or is an order-specific issue, route to OrderAgent.
    Otherwise (e.g. pure general inquiry), skip OrderAgent and route directly to PolicyAgent.
    """
    category = state.get("category")
    order_id = state.get("extracted_order_id")
    
    if order_id or category in ["refund", "replacement", "cancellation", "shipping_delay", "damaged_item"]:
        return "order_node"
    return "policy_node"

def build_resolveai_graph(checkpointer: Optional[SqliteSaver] = None):
    """Builds and compiles the complete multi-agent LangGraph workflow with persistent SQLite checkpointer."""
    workflow = StateGraph(SupportGraphState)

    # Add agent nodes
    workflow.add_node("triage_node", triage_node)
    workflow.add_node("order_node", order_node)
    workflow.add_node("policy_node", policy_node)
    workflow.add_node("resolution_node", resolution_node)
    workflow.add_node("quality_node", quality_node)

    # Set entry point
    workflow.set_entry_point("triage_node")

    # Dynamic conditional routing after Triage
    workflow.add_conditional_edges(
        "triage_node",
        route_triage,
        {
            "order_node": "order_node",
            "policy_node": "policy_node"
        }
    )

    # Standard sequential edges
    workflow.add_edge("order_node", "policy_node")
    workflow.add_edge("policy_node", "resolution_node")
    workflow.add_edge("resolution_node", "quality_node")
    workflow.add_edge("quality_node", END)

    # Checkpointer for conversation state and native human-in-the-loop interruption/resumption
    cp = checkpointer or get_checkpointer()
    app = workflow.compile(checkpointer=cp)
    return app

# Singleton compiled app using durable SQLite checkpointer
resolveai_app = build_resolveai_graph()

