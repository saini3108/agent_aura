# 🧠 Memory & State

## 📦 Storage Options

- Short-term → Redis
- Long-term → PostgreSQL

## 🧱 Shared State Fields

```json
{
  "chat_history": [],
  "agent_outputs": {},
  "human_feedback": [],
  "flow_state": "paused|running|complete"
}
```

## 📍 Best Practices

- Never mutate state in place
- Keep audit trail for each update
