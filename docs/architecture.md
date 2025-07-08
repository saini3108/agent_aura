# 🏗️ AURA Architecture

## 🎯 Goal

Design an extensible, explainable, agentic platform for credit risk and compliance workflows (e.g., IFRS 9, Basel, IRB).

## 🧱 Components

- `aura_backend` – Django API server
- `aura_frontend` – Next.js dashboard
- `aura_agent` – GenAI orchestration server (this repo)

## 🧩 High-Level Diagram

```text
[Frontend UI]
    ↕ WebSocket/API
[Backend API (Django)]
    ↕ REST/DB
[Agent Server (LangGraph + FastAPI)]
```

## 🗂 Module Structure

Each module (e.g., `model_validation`, `ecl_engine`) is self-contained with:

- Agents
- Tools
- LangGraph workflows
- Configs
