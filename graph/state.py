from typing import TypedDict, List, Dict, Any, Optional

class ResolutionProposal(TypedDict, total=False):
    action: str  # refund, replacement, cancellation, explanation, store_credit, manual_review
    amount: float
    eligible: bool
    summary: str
    reasons: List[str]
    policy_reference: str
    requires_human_approval: bool
    next_steps: str

class QualityAssessment(TypedDict, total=False):
    passed_audit: bool
    confidence_score: float
    hallucination_detected: bool
    contradictions: List[str]
    escalation_required: bool
    escalation_reason: Optional[str]
    review_notes: str

class SupportGraphState(TypedDict, total=False):
    ticket_id: Optional[str]
    customer_id: int
    authenticated_customer_id: Optional[int]
    is_admin: bool
    complaint_text: str
    
    # Triage Agent outputs
    category: str  # refund, replacement, cancellation, shipping_delay, damaged_item, general_inquiry
    intent: str
    priority: str  # low, medium, high, urgent
    extracted_order_id: Optional[str]
    
    # Order Agent outputs
    order_data: Optional[Dict[str, Any]]
    customer_history: Optional[Dict[str, Any]]
    
    # Policy Agent outputs
    policy_evidence: List[Dict[str, Any]]
    policy_summary: Optional[str]
    
    # Resolution Agent outputs
    proposed_resolution: Optional[Dict[str, Any]]
    
    # Quality & Escalation Agent outputs
    quality_assessment: Optional[Dict[str, Any]]
    human_approval_required: bool
    approval_decision: Optional[Dict[str, Any]]
    resolution_applied: bool
    final_status: str  # resolved, pending_approval, escalated, closed
    
    # Observability & Conversation
    agent_logs: List[Dict[str, Any]]
    error: Optional[str]
    retry_count: int

