# 🙋 Human-in-the-Loop (HITL) Design

## 🔧 Features

- Pause agent flow
- Inject human feedback
- Resume with overrides

## 🧱 Implementation

- FastAPI endpoints for UI triggers
- State updated in Redis/Postgres
- HITL points defined in LangGraph nodes

## 👁️ Visibility

- All feedback is logged
- Every HITL path is traceable and auditable
