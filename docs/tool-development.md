# 🛠 Tool Development Guide

## 🧩 Tool Purpose

Encapsulate logic for metrics (e.g., AUC, KS, SHAP, PSI).

## 📁 Location

```
agent_service/
  core/
    tools/
```

## 📦 Best Practices

- Use FastAPI + Pydantic
- Keep tools pure + testable
- Group by category (metrics, validation, compliance)

## 🌐 API Exposure

Each tool is available as:

- LangGraph function
- FastAPI endpoint
