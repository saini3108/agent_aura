# Agentic AI Banking Platform

A production-ready, multi-agent workflow system designed for banking operations using LangGraph for orchestration. The platform provides specialized agents for complex banking tasks such as Expected Credit Loss (ECL) calculations, Risk-Weighted Asset (RWA) modeling, and model validation.

## 🏗️ Architecture

### Core Components
- **FastAPI Backend**: RESTful API for workflow management and agent orchestration
- **LangGraph Workflow Engine**: Multi-agent workflow execution with state management
- **Memory Service**: Redis-based persistence with in-memory fallback
- **Multi-Model LLM Support**: OpenAI, Anthropic, and Google API integration
- **Banking-Specific Tools**: ECL, RWA, and model validation calculators

### Agent System
The platform implements four specialized agents:
1. **Planner Agent**: Analyzes requirements and creates execution plans
2. **Executor Agent**: Executes workflow steps using banking tools
3. **Validator Agent**: Validates results against banking regulations
4. **Summarizer Agent**: Generates comprehensive reports and summaries

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Redis (optional - falls back to in-memory storage)

### Installation

1. **Clone and install dependencies:**
```bash
# Dependencies are already configured in pyproject.toml
# The system will automatically install required packages
```

2. **Environment Configuration:**
Create a `.env` file (optional - defaults work for testing):
```env
# API Keys (optional for testing)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379
REDIS_MAX_CONNECTIONS=10
MEMORY_TTL=3600

# Application Settings
DEBUG=true
LOG_LEVEL=info
```

### Running the Application

**Option 1: Direct Python execution**
```bash
python main.py
```

**Option 2: Using the configured workflow**
The application is pre-configured to run automatically on Replit.

The server will start on `http://0.0.0.0:5000` with the following features:
- ✅ Memory service initialized (Redis or in-memory fallback)
- ✅ Multi-agent workflow system active
- ✅ API endpoints available
- ✅ Interactive documentation at `/docs`

## 📚 API Documentation

### Interactive Documentation
Once running, visit:
- **Swagger UI**: `http://localhost:5000/docs`
- **ReDoc**: `http://localhost:5000/redoc`

### Core Endpoints

#### Start Workflow
```http
POST /api/v1/workflows/start
Content-Type: application/json

{
  "workflow_type": "model_validation",
  "inputs": {
    "model_name": "credit_risk_model_v1",
    "model_configuration": {
      "algorithm": "logistic_regression",
      "features": ["credit_score", "income", "debt_ratio"]
    }
  },
  "llm_config": {
    "provider": "openai",
    "model": "gpt-4"
  }
}
```

#### Get Workflow Status
```http
GET /api/v1/workflows/{workflow_id}/status
```

#### Submit Human Feedback
```http
POST /api/v1/workflows/{workflow_id}/human-feedback
Content-Type: application/json

{
  "user_id": "analyst_1",
  "action": "approve",
  "comments": "Model validation passed all checks"
}
```

## 🔧 Workflow Types

### 1. Model Validation
Validates banking models for regulatory compliance:
```json
{
  "workflow_type": "model_validation",
  "inputs": {
    "model_name": "required",
    "model_configuration": "required",
    "validation_rules": "optional",
    "test_data": "optional"
  }
}
```

### 2. ECL Calculation
Performs IFRS 9 Expected Credit Loss calculations:
```json
{
  "workflow_type": "ecl_calculation",
  "inputs": {
    "portfolio_data": "required",
    "pd_curves": "optional",
    "lgd_estimates": "optional",
    "scenarios": "optional"
  }
}
```

### 3. RWA Calculation
Calculates Basel III Risk-Weighted Assets:
```json
{
  "workflow_type": "rwa_calculation",
  "inputs": {
    "exposure_data": "required",
    "risk_weights": "optional",
    "capital_data": "optional"
  }
}
```

### 4. Reporting
Generates regulatory and management reports:
```json
{
  "workflow_type": "reporting",
  "inputs": {
    "report_type": "required",
    "data_sources": "required",
    "template": "optional",
    "filters": "optional"
  }
}
```

## 🧠 Human-in-the-Loop

The platform supports human intervention for critical decisions:

1. **Automatic Pause**: Workflows pause when validation fails or critical thresholds are exceeded
2. **Review Interface**: Human reviewers can approve, reject, or modify workflow execution
3. **Audit Trail**: All human interactions are logged for regulatory compliance

## 🔍 Monitoring & Logging

### Health Check
```http
GET /api/v1/health
```

### Workflow Metrics
```http
GET /api/v1/workflows/{workflow_id}/metrics
```

### Execution Logs
```http
GET /api/v1/workflows/{workflow_id}/logs
```

## 🛠️ Development

### Project Structure
```
app/
├── agents/          # Multi-agent system
│   ├── planner.py   # Planning agent
│   ├── executor.py  # Execution agent
│   ├── validator.py # Validation agent
│   └── summarizer.py # Summary agent
├── api/             # FastAPI routes and middleware
├── core/            # Configuration and logging
├── models/          # Pydantic data models
├── services/        # Business logic services
├── tools/           # Banking calculation tools
└── workflows/       # LangGraph workflow definitions
```

### Adding New Workflows
1. Define context model in `app/models/context.py`
2. Add workflow logic in agents
3. Implement tools in `app/tools/`
4. Update API routes for new endpoints

### Testing
The platform includes comprehensive logging and can be tested via:
- Interactive API documentation at `/docs`
- Direct API calls using curl or Postman
- Health check endpoint for system status

## 🔐 Security Features

- **Rate Limiting**: Configurable request limits per client
- **Security Middleware**: Protection against common attacks
- **Audit Logging**: Comprehensive logging for compliance
- **CORS Protection**: Configurable cross-origin policies

## 📈 Production Considerations

- **Scalability**: Async/await patterns for high concurrency
- **Reliability**: Redis connection pooling and fallback mechanisms
- **Monitoring**: Structured JSON logging with banking context
- **Compliance**: Full audit trail for regulatory requirements

## 🤝 Integration

This banking workflow engine integrates with:
- External banking systems via REST APIs
- Django backend systems (if present)
- Next.js frontend applications (if present)
- Regulatory reporting systems

## 📝 Example Usage

```python
import requests

# Start a model validation workflow
response = requests.post('http://localhost:5000/api/v1/workflows/start', json={
    "workflow_type": "model_validation",
    "inputs": {
        "model_name": "retail_credit_model",
        "model_configuration": {
            "algorithm": "random_forest",
            "target": "default_probability"
        }
    }
})

workflow_id = response.json()['workflow_id']

# Check status
status = requests.get(f'http://localhost:5000/api/v1/workflows/{workflow_id}/status')
print(status.json())
```

## 📞 Support

For questions or issues:
1. Check the interactive documentation at `/docs`
2. Review the health check endpoint at `/health`
3. Examine logs for detailed error information

The system is designed to be self-contained and ready for immediate testing and development.