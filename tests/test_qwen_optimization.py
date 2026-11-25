import pytest
from agents.qwen_optimizer import (
    QwenOptimizer,
    TriageStructuredOutput,
    ResolutionStructuredOutput,
    QualityAuditStructuredOutput,
    QWEN_CONCISE_TRIAGE_PROMPT,
    QWEN_CONCISE_RESOLUTION_PROMPT,
    QWEN_CONCISE_QUALITY_PROMPT
)


@pytest.fixture
def optimizer():
    return QwenOptimizer(low_reasoning=True)


def test_qwen_low_reasoning_and_no_internal_reasoning_exposed(optimizer):
    """
    Verifies that low reasoning effort (think: False):
    1. Bypasses internal deliberation tokens (<think>).
    2. Does not expose reasoning in output.
    3. Runs in low latency (< 3.0s).
    """
    prompt = 'Classify: "I want a refund for my order ORD-1001 right away."'
    output, telemetry = optimizer.generate_structured_output(
        prompt=prompt,
        schema=TriageStructuredOutput,
        system_prompt=QWEN_CONCISE_TRIAGE_PROMPT
    )

    # Verify no <think> tokens leaked
    assert "<think>" not in output.intent
    assert "</think>" not in output.intent
    assert telemetry["internal_reasoning_exposed"] is False
    assert telemetry["thinking_suppressed"] is True
    assert telemetry["latency_ms"] < 3000.0  # Low reasoning runs in < 3s


def test_qwen_structured_json_triage_validation(optimizer):
    """
    Tests validated structured JSON output for customer support triage.
    Enforces Pydantic model validation.
    """
    complaint = "My UltraSound headphones ORD-1001 have a broken microphone. Can I get a replacement?"
    triage, telemetry = optimizer.generate_structured_output(
        prompt=f'Complaint: "{complaint}"',
        schema=TriageStructuredOutput,
        system_prompt=QWEN_CONCISE_TRIAGE_PROMPT
    )

    # Validated Pydantic assertions
    assert isinstance(triage, TriageStructuredOutput)
    assert triage.category in ["replacement", "damaged_item"]
    assert triage.order_id == "ORD-1001"
    assert triage.priority in ["urgent", "high", "medium"]
    assert len(triage.intent) > 5

    # Telemetry assertions
    assert telemetry["model"] == "qwen3.5:4b"
    assert telemetry["latency_ms"] > 0
    assert telemetry["eval_count"] > 0


def test_qwen_structured_json_resolution_validation(optimizer):
    """
    Tests validated structured JSON output for resolution synthesis.
    """
    context = """
Complaint: "My order ORD-1001 arrived with a broken microphone. Replace it please."
Category: replacement
Order: {"order_id": "ORD-1001", "product_name": "Headphones", "amount": 189.99, "days_since_delivery": 6}
Policy: POL-REP-2024-V2 Clause 1.1 allows replacements within 14 days of delivery.
"""
    resolution, telemetry = optimizer.generate_structured_output(
        prompt=context,
        schema=ResolutionStructuredOutput,
        system_prompt=QWEN_CONCISE_RESOLUTION_PROMPT
    )

    assert isinstance(resolution, ResolutionStructuredOutput)
    assert resolution.action in ["replacement", "explanation"]
    assert resolution.eligible is True
    assert resolution.policy_cited != ""
    assert len(resolution.customer_message) > 10
    assert "<think>" not in resolution.customer_message


def test_qwen_structured_json_quality_validation(optimizer):
    """
    Tests validated structured JSON output for quality audit agent.
    """
    audit_context = """
Review this resolution proposal:
Action: refund
Amount: $189.99
Requires Human Approval: true
Order Status: delivered 6 days ago
Policy: POL-REF-2024-V2 (30-day return window)
"""
    audit, telemetry = optimizer.generate_structured_output(
        prompt=audit_context,
        schema=QualityAuditStructuredOutput,
        system_prompt=QWEN_CONCISE_QUALITY_PROMPT
    )

    assert isinstance(audit, QualityAuditStructuredOutput)
    assert isinstance(audit.passed_audit, bool)
    assert 0.0 <= audit.confidence_score <= 1.0
    assert audit.hallucination_detected is False
    assert "<think>" not in audit.audit_summary


def test_qwen_tool_calling_order_lookup(optimizer):
    """
    Tests native tool calling with Qwen3.5:4B:
    1. Model receives user request and tool specifications.
    2. Model selects get_order_details with order_id="ORD-1001".
    3. Tool is executed deterministically in Python against the database.
    """
    res = optimizer.call_tools(
        user_message="Can you look up the details for order ORD-1001?",
        authenticated_customer_id=2
    )

    assert res["tool_calls_detected"] >= 1
    first_tool = res["tool_results"][0]
    assert first_tool["tool_name"] == "get_order_details"
    assert first_tool["arguments"]["order_id"] == "ORD-1001"
    
    # Verify deterministic Python tool execution result
    tool_output = first_tool["output"]
    assert tool_output is not None
    assert tool_output["found"] is True
    assert tool_output["order_id"] == "ORD-1001"
    assert "UltraSound" in tool_output["product_name"]
    assert tool_output["amount"] == 189.99

    # Independent latency checks
    assert res["model_latency_ms"] > 0
    assert res["tool_exec_latency_ms"] >= 0
    assert res["total_latency_ms"] >= res["model_latency_ms"]


def test_qwen_tool_calling_delivery_status(optimizer):
    """
    Tests tool calling for shipping/delivery tracking.
    """
    res = optimizer.call_tools(
        user_message="Where is my shipment for order ORD-1001?",
        authenticated_customer_id=2
    )

    assert res["tool_calls_detected"] >= 1
    first_tool = res["tool_results"][0]
    assert first_tool["tool_name"] in ["get_delivery_status", "get_order_details"]
    assert first_tool["arguments"]["order_id"] == "ORD-1001"
    
    tool_output = first_tool["output"]
    assert tool_output is not None
    assert tool_output.get("found") is True or tool_output.get("status") == "delivered"


def test_qwen_deterministic_business_rules_prevents_repeated_llm_calls(optimizer):
    """
    Verifies that simple business rules:
    - 30-day refund window check
    - In-transit cancellation block
    - High-value ($150+) supervisor check
    are executed deterministically in Python in sub-millisecond time
    without wasting repeated LLM calls.
    """
    order_data = {
        "found": True,
        "order_id": "ORD-1002",
        "amount": 299.99,
        "status": "delivered",
        "days_since_delivery": 45  # > 30 days
    }
    policy_evidence = [{"source": "refund_policy.md", "clause_text": "30-day window"}]

    # Execute deterministic business rule validator
    rule_res = optimizer.verify_business_rules_deterministically(
        category="refund",
        order_data=order_data,
        policy_evidence=policy_evidence
    )

    validation = rule_res["deterministic_validation"]
    # 45 days is past 30 days -> ineligible
    assert validation["is_valid"] is False
    assert validation["requires_human_approval"] is True
    assert rule_res["executed_by"] == "deterministic_python"
    # Runs in sub-millisecond time (< 5ms) with zero LLM API calls!
    assert rule_res["latency_ms"] < 5.0


def test_independent_latency_measurement_per_agent(optimizer):
    """
    Verifies that each agent call records independent latency and token metrics.
    """
    # Agent 1: Triage
    _, t1 = optimizer.generate_structured_output(
        prompt='Complaint: "Late package ORD-1001"',
        schema=TriageStructuredOutput,
        system_prompt=QWEN_CONCISE_TRIAGE_PROMPT
    )

    # Agent 2: Resolution
    _, t2 = optimizer.generate_structured_output(
        prompt='Complaint: "Where is ORD-1001?" Category: shipping_delay',
        schema=ResolutionStructuredOutput,
        system_prompt=QWEN_CONCISE_RESOLUTION_PROMPT
    )

    # Verify both latencies are independently captured
    assert t1["latency_ms"] > 0
    assert t2["latency_ms"] > 0
    assert "eval_count" in t1
    assert "eval_count" in t2
    assert t1["latency_ms"] != t2["latency_ms"]

