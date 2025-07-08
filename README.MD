# 🧠 `aura_agent`: Agentic AI Layer for AURA Risk Platform

`aura_agent` is the GenAI/agentic orchestration server for **AURA**, an enterprise-grade platform for Credit Risk, IFRS 9, Basel, ECL, RWA, and regulatory compliance.

This repo powers intelligent, modular, explainable **agentic workflows** — combining LangGraph + FastAPI + MCP + RAG — for model validation, monitoring, and future risk automation modules.

---

## 🎯 Vision: Human-AI Collaboration for Risk Intelligence

Traditional financial risk systems are rule-based, brittle, and isolated. `aura_agent` enhances them with:

✅ Intelligent agents that assist and explain
✅ Human-in-the-loop (HITL) at every decision point
✅ Document-aware RAG agents
✅ Modular design with clean audit trails
✅ Open-source, composable architecture
✅ Enterprise-ready observability + security

> **Goal**: AURA is a modular, future-proof credit risk and regulatory platform.
> This repo adds explainable, human-in-the-loop **agentic AI** for model validation, monitoring, and document-aware decision support — with auditability and governance built-in.

---

## 🧱 Design Principles

- **Modular-first** – Each business module (Validation, Monitoring, ECL, etc.) is independent and reusable.
- **Explainable-first** – All agent actions are logged, traced, and reviewable.
- **Agentic only where useful** – We only add GenAI where it enhances decision-making.
- **Composable** – Shared logic lives in `core/`; modules remain clean and testable.
- **Governed by HITL** – Every flow supports human review, override, and feedback.
- **Non-vendor lock-in** – All components are open-source and portable.

## ✅ Core Principles for Enterprise-Ready Codebase
| Principle           | Implementation                                                          |
| ------------------- | ----------------------------------------------------------------------- |
| **Modularity**      | Strict separation via `core/`, `modules/`, `tools/`, `agents/`, etc.    |
| **Security**        | `.env` for secrets, CORS policies, no hardcoded creds, RBAC integration |
| **Observability**   | LangGraph tracing, audit trail hooks, logging pipeline                  |
| **Testability**     | Unit + integration tests (`pytest` from day one)                        |
| **Extensibility**   | Plug-and-play modules and flows                                         |
| **HITL Governance** | All flows pausable/injectable via API hooks                             |
| **Config-driven**   | YAML/JSON configs to drive module load/dispatch                         |


---

## 🧩 AURA Ecosystem Overview

| Component        | Repo           | Description                                      |
|------------------|----------------|--------------------------------------------------|
| 🧠 `aura_agent`   | ✅ This repo    | GenAI agents, LangGraph workflows, RAG           |
| 🛠 `aura_backend` | `Django`       | Auth, user data, model configs, records          |
| 📊 `aura_frontend`| `Next.js`      | UI: Dashboards, agent chat, HITL approval views  |

---

## ⚙️ Tech Stack

| Layer                | Tool                        |
|----------------------|-----------------------------|
| Agent Orchestration  | [LangGraph](https://www.langgraph.dev) |
| Packaging/Modularity | [MCP](https://github.com/langchain-ai/langchain/tree/master/libs/langchain-core#multi-component-packages) |
| API + Tools Layer    | FastAPI + Pydantic          |
| Memory / State       | Redis or PostgreSQL         |
| RAG + Docs Search    | LlamaIndex + Qdrant/Chroma  |
| Observability        | LangGraph trace + audit DB  |
| RBAC/Auth            | Via Django backend          |
| Report Generation    | Markdown → HTML → PDF       |

---

## 📂 Folder Structure

```bash
aura_agent/
│
├── agent_service/                  # Main application code
│   ├── core/                       # Core shared logic, reusable across modules
│   │   ├── agents/                 # Base and common agents (e.g. HumanAgent)
│   │   ├── tools/                  # Tool interfaces (e.g. AUC, PSI, drift)
│   │   ├── flows/                  # Shared LangGraph components/utilities
│   │   ├── memory/                 # State management (Redis, PostgreSQL, etc.)
│   │   ├── schema/                 # Pydantic models for inputs/outputs
│   │   └── config.py               # App-level config (env, logging)
│   │
│   ├── modules/                    # Modular, plug-and-play domain flows
│   │   ├── model_validation/       # Model validation flows, agents, tools
│   │   │   ├── agents/
│   │   │   ├── tools/
│   │   │   ├── flows/
│   │   │   └── __init__.py
│   │   │
│   │   ├── model_monitoring/       # Monitoring workflows (drift, stability)
│   │   ├── ecl_engine/             # Placeholder for future ECL module
│   │   └── __init__.py
│   │
│   └── main.py                     # FastAPI app entrypoint (serves flows)
│
├── scripts/                        # CLI utilities (e.g. test run flows, ingestion)
│   └── run_validation_flow.py
│
├── tests/                          # Unit and integration tests
│   ├── unit/                       # Tests for tools, agents
│   └── flows/                      # Tests for end-to-end LangGraph flows
│
├── docs/                           # Documentation folder (in Markdown)
│   ├── architecture.md
│   ├── agent-design.md
│   ├── module-guide.md
│   ├── hitl.md
│   └── reporting.md
│
├── .env                            # Environment variables
├── requirements.txt / pyproject.toml
├── Dockerfile
└── README.md

````

---

## 🧠 Agents Overview

| Agent                | Role                                      |
| -------------------- | ----------------------------------------- |
| `AnalystAgent`       | Parses model and metadata                 |
| `ValidatorAgent`     | Runs metrics (AUC, KS, PSI, SHAP, etc.)   |
| `DocumentationAgent` | Verifies documentation via RAG            |
| `ReviewerAgent`      | Highlights issues and generates feedback  |
| `AuditorAgent`       | Final step: signs off or escalates        |
| `HumanAgent`         | Injected during pause for review/feedback |

All agents are composable and extendable per module.

---

## 🔁 Example Validation Flow

```text
[AnalystAgent]
     ↓
[ValidatorAgent]
     ↓
[DocumentationAgent]
     ↓
[Pause for Human Review]
     ↓
[ReviewerAgent]
     ↓
[AuditorAgent]
     ↓
[ReportGenerator] (future)
```

---

## 🧪 Testing Philosophy

```text
tests/
├── unit/               # Tool and agent logic
│   └── test_auc_tool.py
├── flows/              # Integration-level flow tests
│   └── test_validation_flow.py
```

---

## 🚀 Quickstart (Dev Setup)

```bash
# Clone and setup
git clone https://github.com/your-org/aura_agent.git
cd aura_agent
python -m venv .venv && source .venv/bin/activate

# Install dependencies
poetry install  # or: pip install -r requirements.txt

# Start dev server
uvicorn agent_service.main:app --reload
```

---

## 🧪 Run Example Flow

```bash
# View example LangGraph flow:
agent_service/modules/model_validation/flows/sample_validation_flow.py

# Launch via FastAPI:
POST /run-flow?module=model_validation

# Or via CLI script:
python scripts/run_validation_flow.py
```

---

## 🧭 Development Roadmap

### ✅ Phase 1: Bootstrap

* [x] LangGraph + FastAPI setup
* [x] Modular structure with `core/` and `modules/`
* [x] Add metrics tools: AUC, SHAP, KS
* [x] Redis/Postgres memory

### 🔄 Phase 2: Human-in-the-Loop (HITL)

* [ ] Pause/resume LangGraph workflows
* [ ] UI feedback endpoint
* [ ] LangGraph ↔ UI bridge for HITL inserts
* [ ] Store user override decisions

### 📚 Phase 3: RAG + Document Agent

* [ ] Ingest PDFs, Excels with LlamaIndex
* [ ] Embed and index documents
* [ ] Create DocumentationAgent with compliance Q\&A

### 🧩 Phase 4: Expand Modules

* [ ] Full `model_monitoring` flow
* [ ] ECL agent stubs (stage classification, PD/LGD)
* [ ] Configurable module loader (YAML/JSON)
* [ ] Generalize tools in `core/`

### 🔐 Phase 5: Audit & Governance

* [ ] RBAC integration with Django sessions
* [ ] Store LangGraph trace in audit DB
* [ ] Per-client audit logs

### 📄 Phase 6: Reporting & Packaging

* [ ] Markdown → HTML → PDF reporting agent
* [ ] Add Docker + `.env` support
* [ ] CLI tools for dev/ops validation

---

## 📄 Docs Structure (Coming Soon)

```bash
docs/
├── architecture.md        # System diagrams, flowcharts
├── module-guide.md        # Create new modules (e.g., RWA, ECL)
├── agent-design.md        # How to create a new agent or tool
├── hitl.md                # HITL lifecycle and API hooks
├── reporting.md           # Report pipeline and structure
```

## 📄 Here’s a short guide based on purpose:
```bash
| Purpose                    | Folder                                  | Example File Name                        |
| -------------------------- | --------------------------------------- | ---------------------------------------- |
| Shared core logic          | `agent_service/core/`                   | `agents/validator_agent.py`              |
| LangGraph flows per module | `agent_service/modules/<module>/flows/` | `sample_validation_flow.py`              |
| Tools (AUC, SHAP, etc.)    | `agent_service/core/tools/`             | `metrics_tools.py`, `doc_checker.py`     |
| Agent classes              | `agent_service/core/agents/`            | `validator_agent.py`, `auditor_agent.py` |
| Memory config/loaders      | `agent_service/core/memory/`            | `redis_memory.py`, `pg_state.py`         |
| Data schemas               | `agent_service/core/schema/`            | `input_schema.py`, `result_schema.py`    |
| Module-specific logic      | `agent_service/modules/<module>/`       | `tools.py`, `agents.py`, `flows.py`      |
| CLI/testing/debug scripts  | `scripts/`                              | `run_validation_flow.py`                 |
| Tests                      | `tests/`                                | `test_auc_tool.py`, `test_flow.py`       |

```
