# ResolveAI Interview Preparation & Technical Deep-Dive

### Q1: What is the core problem ResolveAI solves?
Customer support tickets are either handled by rigid keyword bots that cannot handle nuanced queries or by human agents overwhelmed by repetitive lookups. ResolveAI introduces a multi-agent autonomous system where specialized agents collaborate to investigate orders, retrieve company policies, validate business rules, and propose compliant resolutions—with guaranteed human-in-the-loop safeguards for sensitive or high-value transactions.

---

### Q2: Why use a Multi-Agent architecture instead of a single prompt?
A single LLM prompt trying to classify intent, query SQL, search RAG, apply policies, check fraud, and format customer output suffers from context pollution, hallucinations, and unconstrained decision-making. Decomposing the problem into 5 discrete agents (Triage, Order, Policy, Resolution, Quality) creates modularity, clear separation of concerns, and verifiable state transitions.

---

### Q3: How do you prevent LLMs from hallucinating refund amounts or policy clauses?
1. **RAG Grounding**: The Policy Agent retrieves actual policy clauses from ChromaDB using semantic similarity.
2. **Deterministic Guardrails**: The Resolution Agent passes all outputs through `validate_resolution()`, a deterministic Python validator that hard-codes non-negotiable business rules (e.g. max auto-refund = $0.00, high value threshold = $150.00).
3. **Independent Audit**: The Quality Agent verifies that cited policies exist and that status constraints (such as in-transit orders) are strictly respected.

---

### Q4: How is Human-in-the-Loop implemented?
Using LangGraph native `interrupt()`. When the Quality Agent determines that an action (such as a refund or order >$150) requires human supervisor authorization, the graph triggers `interrupt()`. LangGraph pauses execution and persists the workflow state into an SQLite database (`resolveai_checkpoints.db`). When a supervisor approves or rejects via the REST API or Admin Dashboard, execution resumes from the exact paused node using `Command(resume=...)`.

---

### Q5: How is Qwen3.5:4B optimized for customer support?
1. **Low Reasoning Effort**: `think: False` in Ollama options disables the internal 500+ token thinking loop, reducing latency from ~8s to ~1.2s per request.
2. **Pydantic Structured Outputs**: Responses are constrained to strict JSON schemas (`TriageStructuredOutput`, `ResolutionStructuredOutput`, `QualityAuditStructuredOutput`).
3. **Zero-Cost Local Inference**: Runs entirely on local hardware via Ollama, eliminating external API costs and preventing customer data leakage.

---

### Q6: How do you handle order security and privacy?
1. **Customer Ownership**: Endpoints verify that the authenticated customer ID matches the order record.
2. **Address Masking**: Street addresses are masked (e.g., `***, Springfield, IL`) in customer-facing responses, preventing PII exposure. Full addresses are restricted to administrators.
