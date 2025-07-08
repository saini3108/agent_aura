# 🔁 Workflow Design

## 🔧 Engine: LangGraph

- Pause/resume
- Shared state
- Conditional branches
- HITL control

## 📂 Location

```
agent_service/
  core/
    flows/
```

## 🧪 Example: Model Validation

```text
[Analyst] → [Validator] → [Doc Agent] → [Pause] → [Reviewer] → [Auditor]
```

## ✅ Tips

- Add logging at each node
- Use state injection for traceability
- Pause with metadata (e.g., reason, role)
