# API Examples

## Health Check
```bash
curl http://localhost:5000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "services": {
    "memory": "healthy",
    "llm_providers": [],
    "api": "healthy"
  },
  "version": "1.0.0"
}
```

## Model Validation Workflow

### Start Workflow
```bash
curl -X POST http://localhost:5000/api/v1/workflows/start \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_type": "model_validation",
    "inputs": {
      "model_name": "credit_risk_model_v1",
      "model_configuration": {
        "algorithm": "logistic_regression",
        "features": ["credit_score", "income", "debt_ratio"],
        "target": "default_probability"
      }
    },
    "llm_config": {
      "provider": "openai",
      "model": "gpt-4"
    }
  }'
```

### Get Workflow Status
```bash
curl http://localhost:5000/api/v1/workflows/{workflow_id}/status
```

## ECL Calculation Workflow

```bash
curl -X POST http://localhost:5000/api/v1/workflows/start \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_type": "ecl_calculation",
    "inputs": {
      "portfolio_data": [
        {
          "account_id": "ACC001",
          "balance": 50000,
          "product_type": "personal_loan",
          "origination_date": "2023-01-15",
          "maturity_date": "2028-01-15",
          "interest_rate": 5.5,
          "credit_score": 720
        }
      ]
    }
  }'
```

## RWA Calculation Workflow

```bash
curl -X POST http://localhost:5000/api/v1/workflows/start \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_type": "rwa_calculation",
    "inputs": {
      "exposure_data": [
        {
          "exposure_id": "EXP001",
          "counterparty": "Corporate_A",
          "exposure_amount": 1000000,
          "asset_class": "corporate",
          "rating": "BBB",
          "maturity": 3.5
        }
      ]
    }
  }'
```

## Reporting Workflow

```bash
curl -X POST http://localhost:5000/api/v1/workflows/start \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_type": "reporting",
    "inputs": {
      "report_type": "risk_summary",
      "data_sources": ["portfolio_data", "market_data"],
      "template": "monthly_risk_report"
    }
  }'
```

## Human-in-the-Loop

### Submit Human Feedback
```bash
curl -X POST http://localhost:5000/api/v1/workflows/{workflow_id}/human-feedback \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "risk_analyst_1",
    "action": "approve",
    "comments": "Model validation passed all regulatory checks"
  }'
```

## Monitoring

### Get Workflow Logs
```bash
curl http://localhost:5000/api/v1/workflows/{workflow_id}/logs
```

### Get Workflow Metrics
```bash
curl http://localhost:5000/api/v1/workflows/{workflow_id}/metrics
```

### List All Workflows
```bash
curl http://localhost:5000/api/v1/workflows
```

### Cancel Workflow
```bash
curl -X POST http://localhost:5000/api/v1/workflows/{workflow_id}/cancel
```

## Validation

### Validate Workflow Request
```bash
curl -X POST http://localhost:5000/api/v1/workflows/validate \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_type": "model_validation",
    "inputs": {
      "model_name": "test_model",
      "model_configuration": {"algorithm": "test"}
    }
  }'
```