# Agentic AI Banking Platform

## Overview

This is a production-ready, multi-agent workflow system designed for banking operations using LangGraph for orchestration. The platform provides specialized agents for complex banking tasks such as Expected Credit Loss (ECL) calculations, Risk-Weighted Asset (RWA) modeling, and model validation. It follows a modular architecture with human-in-the-loop capabilities and comprehensive audit logging for regulatory compliance.

## System Architecture

### Core Components
- **FastAPI Backend**: RESTful API for workflow management and agent orchestration
- **LangGraph Workflow Engine**: Multi-agent workflow execution with state management
- **Redis Memory Service**: Short-term state management and caching
- **Multi-Model LLM Support**: OpenAI, Anthropic, and Google API integration
- **Banking-Specific Tools**: ECL, RWA, and model validation calculators

### Agent Architecture
The system implements a specialized agent pattern with four core agents:
1. **Planner Agent**: Analyzes requirements and creates execution plans
2. **Executor Agent**: Executes workflow steps using banking tools
3. **Validator Agent**: Validates results against banking regulations
4. **Summarizer Agent**: Generates comprehensive reports and summaries

## Key Components

### Workflow Management
- **Graph-based Execution**: LangGraph orchestrates agent interactions
- **State Management**: Persistent workflow state with Redis backend
- **Human-in-the-Loop**: Built-in approval workflows for regulatory compliance
- **Audit Logging**: Comprehensive logging for banking regulations

### Banking Tools
- **ECL Calculations**: IFRS 9 compliant Expected Credit Loss calculations
- **RWA Modeling**: Basel III Risk-Weighted Asset calculations
- **Model Validation**: Statistical validation and back-testing tools
- **Data Quality Checks**: Comprehensive validation toolkit

### Memory and Context
- **Model Context Protocol (MCP)**: Structured context sharing between agents
- **Agent-to-Agent Communication**: Standardized communication patterns
- **Persistent State**: Redis-based state management with TTL controls
- **Vector Storage**: In-memory vector store for context similarity

## Data Flow

1. **Workflow Initiation**: API receives workflow request with type and inputs
2. **Planning Phase**: Planner agent analyzes requirements and creates execution plan
3. **Execution Phase**: Executor agent runs planned steps using banking tools
4. **Validation Phase**: Validator agent checks results against compliance rules
5. **Human Review**: Optional human approval step for critical decisions
6. **Summarization**: Final report generation and workflow completion

### Context Propagation
- Each agent receives and updates a shared BaseContext object
- Context includes workflow metadata, execution state, and validation results
- Memory service maintains context persistence across agent transitions

## External Dependencies

### LLM Providers
- **OpenAI**: GPT-4 for general reasoning and analysis
- **Anthropic**: Claude for complex regulatory compliance checks
- **Google**: Gemini for mathematical calculations and validations

### Infrastructure
- **Redis**: In-memory data store for workflow state
- **FastAPI**: Web framework for API endpoints
- **LangGraph**: Agent workflow orchestration
- **Pydantic**: Data validation and serialization

### Banking Libraries
- **NumPy/Pandas**: Mathematical calculations and data processing
- **Custom Banking Tools**: ECL, RWA, and validation implementations

## Deployment Strategy

### Configuration Management
- Environment-based configuration using Pydantic Settings
- Support for multiple deployment environments (dev, staging, production)
- Secure API key management for LLM providers

### Scalability Considerations
- Async/await patterns throughout for non-blocking operations
- Redis connection pooling for high-concurrency scenarios
- Configurable timeouts and retry logic for external services

### Monitoring and Logging
- Structured JSON logging with banking-specific context
- Request/response middleware for API monitoring
- Audit trail for regulatory compliance requirements

## Setup and Usage

### Quick Start
1. **Run the application**: The system automatically starts on port 5000
2. **Access API documentation**: Visit `/docs` for interactive Swagger UI
3. **Test health check**: GET `/api/v1/health` to verify system status
4. **Memory fallback**: Redis is optional - system uses in-memory storage if unavailable

### API Testing Examples
```bash
# Health check
curl http://localhost:5000/api/v1/health

# Start model validation workflow
curl -X POST http://localhost:5000/api/v1/workflows/start \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_type": "model_validation",
    "inputs": {
      "model_name": "credit_model_v1",
      "model_configuration": {"algorithm": "logistic_regression"}
    }
  }'
```

### Current Status
- ✅ FastAPI server running on port 5000
- ✅ Multi-agent workflow system operational
- ✅ Memory service with Redis fallback working
- ✅ API endpoints for all workflow operations
- ✅ Comprehensive documentation and examples

## Known Issues & Solutions

### Workflow Validation Errors
**Issue**: Model validation workflows require specific input fields
**Solution**: Ensure `model_name` and `model_configuration` are provided in workflow inputs

### Redis Connection
**Issue**: Redis service not always available
**Solution**: System automatically falls back to in-memory storage

## Changelog

- July 07, 2025: Added comprehensive README, fixed Pydantic v2 compatibility, implemented Redis fallback
- July 06, 2025: Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.