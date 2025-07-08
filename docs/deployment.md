# 🚀 Deployment Guide

## 📦 Requirements

- Python 3.11+
- Redis/PostgreSQL
- Docker/Docker Compose

## 🐳 Docker Example

```dockerfile
FROM python:3.11
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "agent_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🧪 Env Config

- `.env` file with:
  - `REDIS_URL=...`
  - `PG_URL=...`
  - `VECTOR_DB_URL=...`
