# ResolveAI Detailed Module Guide & Learning Reference

This document serves as an architectural textbook for engineers learning **Agentic AI** through the ResolveAI codebase.

---

## Module 1: Triage Agent (`agents/triage_agent.py`)

- **What it does:** Classifies incoming customer text into standardized categories (`refund`, `replacement`, `cancellation`, `shipping_delay`, `damaged_item`, `general_inquiry`), extracts order IDs (e.g. `ORD-1001`), infers customer intent, and sets ticket priority (`low`, `medium`, `high`, `urgent`).
- **Why it exists:** Unstructured human complaints cannot be processed by deterministic systems without reliable normalization, entity extraction, and routing.
- **How it works:**
  1. Primary path: Prompts local `qwen3.5:4b` via Ollama with strict Pydantic structured output (`TriageStructuredOutput`) and low reasoning mode (`think: False`).
  2. Backup path: If Ollama is offline or in mock test environments, uses regex pattern matching (`\b(ORD-\d+)\b`) and semantic keyword classification (`heuristic_triage`).
  3. Measures latency in milliseconds and returns structured telemetry.
- **Which agent uses it:** Entry-point agent for the LangGraph workflow (`triage_node`).
- **Data In:** Plaintext customer complaint string.
- **Data Out:** Dict with `category`, `intent`, `priority`, `order_id`, `execution_mode`, `latency_ms`.
- **Agentic AI Concepts:** *Intent Classification*, *Entity Extraction (NER)*, *Structured Output Schema*.

---

## Module 2: Order Investigation Agent (`agents/order_agent.py`)

- **What it does:** Accesses the relational database to verify whether the order exists, inspects items, monetary values, carrier details, tracking numbers, and calculates elapsed time since delivery.
- **Why it exists:** LLMs cannot know internal company inventory or order status without tool augmentation. Grounding against database state prevents false claims.
- **How it works:** Invokes `tools/order_tools.py` (`get_order_details`, `get_delivery_status`, `get_customer_history`) to query SQLite via SQLAlchemy. Enforces customer ownership and masks private delivery addresses for customer queries.
- **Which agent uses it:** `order_node` in the LangGraph graph.
- **Data In:** `order_id: str`, `customer_id: int`.
- **Data Out:** Dict containing `order_found`, `order_data`, `delivery_status`, `customer_history`.
- **Agentic AI Concepts:** *Tool Calling*, *Database Grounding*, *Context Enrichment*, *PII Data Privacy*.

---

## Module 3: Policy Research Agent (`agents/policy_agent.py`)

- **What it does:** Queries the ChromaDB vector database to retrieve official company policy clauses and citations for the specific category and issue.
- **Why it exists:** Guarantees that resolutions are grounded in legal enterprise terms and prevents hallucinated policies.
- **How it works:** Invokes `tools/policy_tools.py` (`search_company_policy`), which runs cosine similarity search over chunked markdown policies using local `all-MiniLM-L6-v2` embeddings.
- **Which agent uses it:** `policy_node` in LangGraph.
- **Data In:** `category: str`, `complaint_text: str`, `order_status: Optional[str]`.
- **Data Out:** Dict with list of `citations`, source policy IDs, and evidence summary.
- **Agentic AI Concepts:** *Retrieval-Augmented Generation (RAG)*, *Source Grounding*, *Local Zero-Cost Embeddings*.

---

## Module 4: Resolution Agent (`agents/resolution_agent.py`)

- **What it does:** Synthesizes the customer's request, verified order details, and policy evidence to formulate an actionable resolution proposal (e.g. approve replacement, flag refund for supervisor, reject expired return).
- **Why it exists:** Translates findings into a business resolution and draft customer response.
- **How it works:** Prompts local `qwen3.5:4b` using `ResolutionStructuredOutput` while strictly applying `validate_resolution` deterministic business rules. Strict rules override the LLM whenever financial or return windows are at risk.
- **Which agent uses it:** `resolution_node` in LangGraph.
- **Data In:** Complaint text, category, order data dict, policy citations list.
- **Data Out:** Proposed action, proposed amount, eligibility flag, approval requirements, customer message.
- **Agentic AI Concepts:** *Multi-Source Synthesis*, *Constrained Generation*, *Deterministic Guardrails*.

---

## Module 5: Quality & Escalation Agent (`agents/quality_agent.py`)

- **What it does:** Independently audits the proposed resolution before anything reaches the user or database. Checks for hallucinations, logic contradictions (e.g. cancelling an in-transit shipment), and triggers native LangGraph interrupt for supervisor approval.
- **Why it exists:** Provides an independent safety boundary against LLM hallucinations, high financial transactions, and policy breaches.
- **How it works:** Executes validation rules checking for policy compliance, threshold compliance ($150 limit), and order status conflicts.
- **Which agent uses it:** `quality_node` in LangGraph.
- **Data In:** Complaint text, order data, policy citations, proposed resolution.
- **Data Out:** `passed_audit`, `confidence_score`, `escalation_required`, `escalation_reason`, `audit_summary`.
- **Agentic AI Concepts:** *Autonomous Verification*, *Safety Guardrails*, *Human-in-the-Loop*.
