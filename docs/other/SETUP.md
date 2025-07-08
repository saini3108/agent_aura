# Quick Setup Guide

## 🚀 Get Started in 2 Minutes

### Step 1: Run the Application
The banking AI platform is ready to run immediately:

```bash
python main.py
```

**That's it!** The server starts on `http://localhost:5000`

### Step 2: Test the API

**Health Check:**
```bash
curl http://localhost:5000/api/v1/health
```

**Interactive Documentation:**
Open `http://localhost:5000/docs` in your browser

### Step 3: Start Your First Workflow

**Model Validation Example:**
```bash
curl -X POST http://localhost:5000/api/v1/workflows/start \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_type": "model_validation",
    "inputs": {
      "model_name": "credit_risk_model",
      "model_configuration": {
        "algorithm": "logistic_regression",
        "features": ["credit_score", "income"]
      }
    }
  }'
```

**ECL Calculation Example:**
```bash
curl -X POST http://localhost:5000/api/v1/workflows/start \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_type": "ecl_calculation",
    "inputs": {
      "portfolio_data": [
        {
          "account_id": "ACC001",
          "balance": 10000,
          "product_type": "personal_loan",
          "origination_date": "2023-01-15"
        }
      ]
    }
  }'
```

## 🔧 Configuration (Optional)

### Environment Variables
Create `.env` file for API keys:
```env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

### Redis (Optional)
The system works without Redis using in-memory storage. To use Redis:
```bash
# Install and start Redis
redis-server

# Set Redis URL
export REDIS_URL=redis://localhost:6379
```

## 📖 Key Features Ready to Use

- **Multi-Agent Workflows**: Planner → Executor → Validator → Summarizer
- **Banking Tools**: ECL calculations, RWA modeling, model validation
- **Human-in-the-Loop**: Pause workflows for human review
- **API Monitoring**: Request logging and metrics
- **Memory Management**: Context persistence across workflow steps

## 🧪 Test All Workflow Types

| Workflow Type | Required Inputs | Description |
|---------------|-----------------|-------------|
| `model_validation` | `model_name`, `model_configuration` | Validate banking models |
| `ecl_calculation` | `portfolio_data` | Calculate expected credit losses |
| `rwa_calculation` | `exposure_data` | Calculate risk-weighted assets |
| `reporting` | `report_type`, `data_sources` | Generate reports |

## 🔍 Monitoring Your Workflows

**Get Status:**
```bash
curl http://localhost:5000/api/v1/workflows/{workflow_id}/status
```

**View Logs:**
```bash
curl http://localhost:5000/api/v1/workflows/{workflow_id}/logs
```

**Check Metrics:**
```bash
curl http://localhost:5000/api/v1/workflows/{workflow_id}/metrics
```

## ✅ Success Indicators

When running correctly, you'll see:
- ✅ Memory service initialized
- ✅ Application started successfully  
- ✅ Server running on port 5000
- ✅ API requests logged and processed

## 🆘 Troubleshooting

**Port Already in Use:**
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9
```

**API Keys Not Working:**
- API keys are optional for basic testing
- System will log warnings but continue working
- Add keys to `.env` file for full functionality

**Redis Connection Errors:**
- Normal - system automatically uses in-memory storage
- No action needed for basic operation

The platform is designed to work out-of-the-box with intelligent fallbacks for all external dependencies.