import re
import time
import json
import logging
from typing import Optional, Dict, Any, List, Literal, Type, TypeVar, Tuple
import httpx
from pydantic import BaseModel, Field, ValidationError

from backend.config import settings, ExecutionMode
from tools.order_tools import get_order_details, get_delivery_status, get_customer_history
from tools.validation_tools import validate_resolution

logger = logging.getLogger("ResolveAI.QwenOptimizer")

T = TypeVar("T", bound=BaseModel)


# =====================================================================
# 1. Validated Pydantic Schemas for Structured Agent Outputs
# =====================================================================

class TriageStructuredOutput(BaseModel):
    category: Literal["refund", "replacement", "cancellation", "shipping_delay", "damaged_item", "general_inquiry"] = Field(
        default="general_inquiry",
        description="Standard customer complaint classification"
    )
    intent: str = Field(
        default="Customer inquiry",
        description="Concise 1-sentence description of customer intent"
    )
    priority: Literal["urgent", "high", "medium", "low"] = Field(
        default="medium",
        description="Ticket urgency level"
    )
    order_id: Optional[str] = Field(
        default=None,
        description="Order identifier (e.g. ORD-1001) or null"
    )


class ResolutionStructuredOutput(BaseModel):
    action: Literal["refund", "replacement", "cancellation", "carrier_trace", "reject_ineligible", "reject", "explanation", "await_delivery_and_return", "manual_review"] = Field(
        default="explanation",
        description="Proposed customer remedy"
    )
    proposed_amount: float = Field(
        default=0.0,
        description="Financial refund or credit amount in USD"
    )
    eligible: bool = Field(
        default=True,
        description="Whether proposal is eligible under policy windows"
    )
    requires_approval: bool = Field(
        default=False,
        description="Whether proposal requires human supervisor review"
    )
    reason: str = Field(
        default="",
        description="Justification citing policy clauses and verified order facts"
    )
    policy_cited: str = Field(
        default="POL-REF-2024-V2",
        description="Policy code reference (e.g. POL-REF-2024-V2 Clause 1.1)"
    )
    customer_message: str = Field(
        default="",
        description="Short, task-focused response for the customer"
    )


class QualityAuditStructuredOutput(BaseModel):
    passed_audit: bool = Field(
        default=True,
        description="True if proposal is compliant and grounded"
    )
    confidence_score: float = Field(
        default=0.9,
        description="Confidence score from 0.0 to 1.0"
    )
    hallucination_detected: bool = Field(
        default=False,
        description="True if proposal invented nonexistent orders or terms"
    )
    contradictions: List[str] = Field(
        default_factory=list,
        description="List of detected policy or state contradictions"
    )
    escalation_required: bool = Field(
        default=False,
        description="True if human supervisor sign-off is needed"
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Reason for human supervisor intervention"
    )
    audit_summary: str = Field(
        default="Audit complete.",
        description="Concise summary of audit findings"
    )


# =====================================================================
# 2. Tool Definitions for Qwen Tool Calling
# =====================================================================

RESOLVEAI_ORDER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": "Retrieve order facts (product, price, order status, purchase date) by order ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID, e.g. ORD-1001"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_delivery_status",
            "description": "Retrieve carrier tracking and shipping status by order ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID, e.g. ORD-1001"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer_history",
            "description": "Retrieve past order count and total spend for the customer",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer", "description": "Customer ID integer"}
                },
                "required": ["customer_id"]
            }
        }
    }
]


# =====================================================================
# 3. Concise System Prompts (Optimized for 4B Parameter Model)
# =====================================================================

QWEN_CONCISE_TRIAGE_PROMPT = """You are a customer support triage classifier.
Output only valid JSON.
Allowed categories: refund, replacement, cancellation, shipping_delay, damaged_item, general_inquiry.
Allowed priority: urgent, high, medium, low.
Format:
{"category": "refund", "intent": "1-sentence summary", "priority": "high", "order_id": "ORD-1001"}
If no order ID is found in the text, use null."""


QWEN_CONCISE_RESOLUTION_PROMPT = """You are a customer support resolution agent.
Output only valid JSON. Apply deterministic constraints provided.
Never auto-refund without supervisor approval.
Format:
{"action": "refund", "proposed_amount": 0.0, "eligible": true, "requires_approval": true, "reason": "...", "policy_cited": "...", "customer_message": "..."}"""


QWEN_CONCISE_QUALITY_PROMPT = """You are a quality audit auditor.
Output only valid JSON. Keep audit_summary under 15 words.
Format:
{"passed_audit": true, "confidence_score": 0.95, "hallucination_detected": false, "contradictions": [], "escalation_required": false, "escalation_reason": null, "audit_summary": "Proposal verified"}"""


# =====================================================================
# 4. QwenOptimizer Engine
# =====================================================================

class QwenOptimizer:
    """
    High-efficiency adapter for Qwen3.5:4B via Ollama.
    Enforces:
    - Low reasoning effort (think: False) for sub-second responses.
    - Strict suppression of internal reasoning (<think> tags).
    - Validated Pydantic structured output.
    - Native tool calling with deterministic Python execution.
    - Independent per-agent latency measurement.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        low_reasoning: bool = True,
        timeout: float = 20.0
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model_name = model_name or settings.QWEN_MODEL
        self.low_reasoning = low_reasoning
        self.timeout = timeout

    @staticmethod
    def strip_internal_reasoning(content: str) -> str:
        """
        Removes any internal reasoning tags (<think>...</think>) or
        deliberation markers from output text so internal thoughts are never exposed.
        """
        if not content:
            return ""
        # Strip <think>...</think> blocks if any slipped through
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE)
        # Strip markdown json backticks if wrapping the content
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
        return cleaned.strip()

    def generate_structured_output(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 350
    ) -> Tuple[T, Dict[str, Any]]:
        """
        Invokes Qwen3.5:4B with low reasoning effort, enforces JSON output,
        strips internal reasoning, validates against a Pydantic schema, and
        records independent latency and token telemetry.
        """
        start_time = time.time()
        sys_msg = system_prompt or "You are a customer support AI assistant. Output only valid JSON."

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        # Low reasoning effort configuration: bypass internal thinking tokens
        if self.low_reasoning:
            payload["think"] = False

        endpoint = f"{self.base_url}/api/chat"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()

        latency_ms = (time.time() - start_time) * 1000
        msg = data.get("message", {})
        raw_content = msg.get("content", "")

        # Crucial security & UX requirement: Suppress internal reasoning
        cleaned_content = self.strip_internal_reasoning(raw_content)

        # Validate with Pydantic model
        validated_output = schema.model_validate_json(cleaned_content)

        telemetry = {
            "execution_mode": ExecutionMode.LIVE_QWEN.value,
            "model": self.model_name,
            "latency_ms": round(latency_ms, 2),
            "eval_count": data.get("eval_count", 0),
            "prompt_eval_count": data.get("prompt_eval_count", 0),
            "thinking_suppressed": self.low_reasoning,
            "internal_reasoning_exposed": False
        }

        return validated_output, telemetry

    def call_tools(
        self,
        user_message: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        authenticated_customer_id: Optional[int] = None,
        is_admin: bool = False
    ) -> Dict[str, Any]:
        """
        Sends user message and tool definitions to Qwen3.5:4B.
        Dispatches any selected tool calls to deterministic Python functions,
        and returns the tool results and telemetry.
        """
        start_time = time.time()
        tool_defs = tools or RESOLVEAI_ORDER_TOOLS
        sys_msg = system_prompt or "You are a customer support agent. Use provided tools to look up customer and order details."

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_message}
            ],
            "tools": tool_defs,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 150
            }
        }

        if self.low_reasoning:
            payload["think"] = False

        endpoint = f"{self.base_url}/api/chat"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()

        model_latency_ms = (time.time() - start_time) * 1000
        msg = data.get("message", {})
        tool_calls = msg.get("tool_calls", [])

        # Execute returned tool calls deterministically via Python
        tool_results = []
        tool_exec_start = time.time()

        for call in tool_calls:
            fn = call.get("function", {})
            fn_name = fn.get("name")
            args = fn.get("arguments", {})

            result = None
            if fn_name == "get_order_details":
                result = get_order_details(
                    order_id=args.get("order_id"),
                    authenticated_customer_id=authenticated_customer_id,
                    is_admin=is_admin
                )
            elif fn_name == "get_delivery_status":
                result = get_delivery_status(
                    order_id=args.get("order_id"),
                    authenticated_customer_id=authenticated_customer_id,
                    is_admin=is_admin
                )
            elif fn_name == "get_customer_history":
                result = get_customer_history(
                    customer_id=int(args.get("customer_id", 0)),
                    authenticated_customer_id=authenticated_customer_id,
                    is_admin=is_admin
                )

            tool_results.append({
                "tool_name": fn_name,
                "arguments": args,
                "output": result
            })

        tool_exec_latency_ms = (time.time() - tool_exec_start) * 1000
        total_latency_ms = (time.time() - start_time) * 1000

        return {
            "model": self.model_name,
            "tool_calls_detected": len(tool_calls),
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "model_latency_ms": round(model_latency_ms, 2),
            "tool_exec_latency_ms": round(tool_exec_latency_ms, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "eval_count": data.get("eval_count", 0),
            "thinking_suppressed": self.low_reasoning
        }

    # =================================================================
    # 5. Deterministic Business Rules Integration (Zero LLM Waste)
    # =================================================================

    @staticmethod
    def verify_business_rules_deterministically(
        category: str,
        order_data: Optional[Dict[str, Any]],
        policy_evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Executes strict business logic via deterministic Python functions.
        Prevents repeating LLM calls for arithmetic, calendar math, or policy flags.
        """
        start_time = time.time()
        order_data = order_data or {"found": False}

        validation = validate_resolution(
            order_data=order_data,
            policy_evidence=policy_evidence,
            proposed_action=category if category in ["refund", "replacement", "cancellation"] else "explanation",
            complaint_category=category
        )

        latency_ms = (time.time() - start_time) * 1000
        return {
            "deterministic_validation": validation,
            "latency_ms": round(latency_ms, 3),
            "executed_by": "deterministic_python"
        }
