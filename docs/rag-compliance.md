# 📚 RAG + Compliance Agents

## 🔧 Components

- LlamaIndex
- Vector DB (Qdrant, Chroma)
- Parsing: PDF, Excel, Word

## 🧱 Use Cases

- Check policy coverage
- Trace model approval status
- Match against regulation docs

## 🔍 Architecture

```text
[DocParser] → [Vectorizer] → [ComplianceAgent] → [LangGraph State]
```

## 📄 Audit

- Document name
- Match quality
- Section used
