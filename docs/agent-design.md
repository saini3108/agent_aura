# 🤖 Agent Design Guide

## 🧠 Agent Types

| Agent | Role |
|-------|------|
| AnalystAgent | Reads model input, prepares analysis |
| ValidatorAgent | Calculates metrics |
| DocumentationAgent | Performs doc compliance checks |
| ReviewerAgent | Summarizes findings |
| AuditorAgent | Final audit |
| HumanAgent | UI-triggered checkpoint |

## 🧩 Best Practices

- Stateless logic, rely on shared context
- Keep agent config modular
- Name clearly by role, not model

## 📦 Agent Implementation

Agents live in:

```
agent_service/
  core/
    agents/
```
