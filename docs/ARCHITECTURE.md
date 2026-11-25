# ResolveAI System Architecture

## 1. High-Level System Architecture

ResolveAI is an enterprise-grade, autonomous multi-agent customer support resolution platform powered by **Local Qwen3.5:4B LLM (via Ollama)**, **LangGraph (with native interrupt)**, **ChromaDB (Local Embeddings)**, **FastAPI**, and **Streamlit**.

The system replaces rigid rule-based bots with five collaborative, specialized AI agents governed by deterministic business rules and human-in-the-loop safeguards.

```mermaid
flowchart TD
    subgraph ClientLayer ["Client & Interface Layer"]
        CP["Customer Portal\n(Streamlit / Web)"]
        AD["Admin & Supervisor Dashboard\n(Streamlit / Oversight)"]
    end

    subgraph APILayer ["FastAPI Backend Services"]
        API["FastAPI Gateway (/api)"]
        AUTH["JWT & RBAC Service"]
        ANALYTICS["Analytics & Telemetry Engine"]
    end

    subgraph LangGraphOrchestrator ["LangGraph Multi-Agent Orchestrator"]
        Triage["Agent 1: Triage Agent\n(Intent, Category, Priority, Order Extraction)"]
        OrderAgent["Agent 2: Order Investigation Agent\n(Order, Delivery & Customer History)"]
        PolicyAgent["Agent 3: Policy Research Agent\n(ChromaDB Semantic Search & Grounding)"]
        ResolutionAgent["Agent 4: Resolution Agent\n(Deterministic Rules & Action Synthesis)"]
        QualityAgent["Agent 5: Quality & Escalation Agent\n(Hallucination Audit & Human Escalation)"]
    end

    subgraph Checkpointing ["LangGraph Checkpointer"]
        Checkpointer[("SQLite Persistent Checkpoints\n(resolveai_checkpoints.db)\nThread State & Human Interrupt")]
    end

    subgraph KnowledgeAndData ["Data & Knowledge Tier"]
        DB[("SQLite / PostgreSQL\n(Customers, Orders, Tickets,\nLogs, Approvals)")]
        RAG[("ChromaDB Vector Store\n(Company Policies)")]
        LocalEmbed["Local SentenceTransformers\n(all-MiniLM-L6-v2)"]
    end

    subgraph LLMProvider ["Local LLM Tier (Zero Cost, Low Latency)"]
        Qwen["Ollama Local Daemon\n(qwen3.5:4b - Low Reasoning Mode)"]
    end

    CP -->|HTTP / JSON| API
    AD -->|HTTP / JSON| API
    API --> AUTH
    API --> ANALYTICS
    API --> LangGraphOrchestrator

    LangGraphOrchestrator <--> Checkpointer

    Triage -->|Extracts Order ID| OrderAgent
    Triage -.->|General Inquiry Skip| PolicyAgent
    OrderAgent --> PolicyAgent
    PolicyAgent --> ResolutionAgent
    ResolutionAgent --> QualityAgent

    Triage -.-> Qwen
    ResolutionAgent -.-> Qwen
    QualityAgent -.-> Qwen

    OrderAgent --> DB
    PolicyAgent --> RAG
    RAG --> LocalEmbed
    QualityAgent -->|Flags Approval / Interrupt| DB
```

## 2. Agent Workflow & Decision Pipeline

```mermaid
sequenceDiagram
    autonumber
    actor Customer
    participant Gateway as FastAPI Gateway
    participant Graph as LangGraph Orchestrator
    participant Triage as Agent 1: Triage
    participant Order as Agent 2: Order Investigator
    participant Policy as Agent 3: Policy Researcher
    participant ResAgent as Agent 4: Resolution Agent
    participant Quality as Agent 5: Quality Auditor
    participant Checkpoint as SQLite Checkpointer
    actor Supervisor as Human Supervisor

    Customer->>Gateway: POST /tickets/submit (Complaint)
    Gateway->>Graph: invoke(initial_state, thread_id)
    Graph->>Triage: Classify intent, category & extract Order ID (Qwen3.5:4B)
    alt General Inquiry (No Order ID)
        Graph->>Policy: Search ChromaDB directly
    else Order-Specific Complaint
        Graph->>Order: Query SQL Database (Order, Carrier, Elapsed Time)
        Graph->>Policy: Search ChromaDB for relevant clause
    end
    Graph->>ResAgent: Synthesize resolution & apply deterministic rules
    Graph->>Quality: Audit proposal for hallucinations & contradictions
    alt Human Approval Required (Refund / >$150 / Discrepancy)
        Quality->>Checkpoint: Trigger native interrupt() & persist state
        Graph-->>Gateway: Execution Paused (Interrupt: Pending Approval)
        Gateway-->>Customer: Ticket created (Under Review)
        Supervisor->>Gateway: POST /tickets/{id}/resume (Approved/Rejected)
        Gateway->>Graph: invoke(Command(resume=decision), thread_id)
        Graph->>ResAgent: Resume & finalize ticket
    else Fully Autonomous
        Quality-->>Graph: Audit passed
    end
    Graph-->>Gateway: Final Resolution & Ticket Record
    Gateway-->>Customer: Return Courteous Customer Resolution
```

## 3. Core Architectural Highlights
1. **Local Privacy-First Inference**: All reasoning is executed locally by `qwen3.5:4b` hosted in Ollama. Customer PII and order data never leave the perimeter.
2. **Deterministic Business Rules**: Financial thresholds ($150+ approval), in-transit cancellation locks, and 30-day return windows are strictly enforced in Python code.
3. **LangGraph Native Interrupt & Resumption**: Checkpoints persisted in SQLite (`resolveai_checkpoints.db`) allow seamless workflow pausing and resumption across server restarts.
4. **Retrieval-Augmented Generation (RAG)**: ChromaDB with SentenceTransformers (`all-MiniLM-L6-v2`) provides vector similarity search across official policy clauses.
5. **Role-Based Access Control (RBAC)**: JWT authentication partitions Customer (`customer`) operations from Supervisor/Admin (`admin`) approval and audit dashboards.
