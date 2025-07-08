# 📦 Module Development Guide

## 📁 Folder Structure

```
modules/
  model_validation/
    agents/
    tools/
    flows/
    config/
```

## 🧪 Steps to Add New Module

1. Copy `model_validation/` as a template
2. Define tools and agents needed
3. Build a LangGraph workflow
4. Add config YAML (for loader)
5. Add test cases in `tests/flows/`

## ✅ Naming Convention

- Use snake_case for modules
- Use PascalCase for agents
