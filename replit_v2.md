# ValiCred-AI: Enterprise Credit Risk Model Validation System

## Overview

ValiCred-AI is an enterprise-grade credit risk model validation platform designed for financial institutions. The system leverages multiple LLM providers (Groq, OpenAI, Anthropic, Gemini) with sophisticated AI agent orchestration to provide intelligent analysis of credit risk models while ensuring regulatory compliance.

The application features a clean enterprise architecture built with Streamlit, providing comprehensive data integration, real-time AI analysis, and advanced configuration management through an intuitive web interface.

## System Architecture

### Clean Enterprise Architecture
- **Modular Design**: Clean separation of concerns with src/ folder structure
- **Configuration Management**: Centralized settings with secure API key management
- **Multi-Provider LLM Support**: Dynamic provider switching with automatic failover
- **Real Data Integration**: Enterprise-grade data loading from multiple sources

### Core Components
- **main.py**: Application entry point with clean navigation and workflow management
- **src/config/**: Centralized configuration management with API key security
- **src/core/**: LLM manager with multi-provider support and health monitoring
- **src/data/**: Real data loading system with quality validation and compliance checks
- **src/ui/**: Advanced configuration panels with secure API key management

### AI Agent System
The system employs specialized AI agents with real LLM capabilities:
1. **Analyst Agent**: Comprehensive data analysis and risk factor identification
2. **Validator Agent**: Statistical validation metrics (AUC, KS, PSI, Gini)
3. **Documentation Agent**: Regulatory compliance review (Basel III, IFRS 9)
4. **Human Review**: Strategic checkpoints for manual review and approval
5. **Reviewer Agent**: Executive summary and actionable recommendations
6. **Auditor Agent**: Final independent validation and compliance assessment

## Key Components

### Agent System (`/agents/`)
- **Analyst Agent**: Data structure analysis, parameter evaluation, and initial assessment
- **Validator Agent**: Statistical validation using scikit-learn, scipy, and custom metrics
- **Documentation Agent**: Compliance checking against Basel III, IFRS 9, and model risk standards
- **Reviewer Agent**: Risk assessment using configurable thresholds and comprehensive findings generation
- **Auditor Agent**: Final approval workflow with governance criteria validation

### Utility Layer (`/utils/`)
- **Workflow Manager**: Step sequencing, prerequisite checking, and execution control
- **Validation Metrics**: Statistical calculations for model performance assessment
- **Audit Logger**: Comprehensive audit trail with session management and entry limiting
- **Report Generator**: Multi-format report generation (validation summary, detailed analysis, audit reports)

### Data Processing
- **Pandas Integration**: Data manipulation and analysis with DataFrame operations
- **NumPy Support**: Numerical computations and array operations
- **Scikit-learn Integration**: Machine learning metrics and model evaluation tools

## Data Flow

1. **Data Ingestion**: Users upload CSV files or provide datasets through the Streamlit interface
2. **Workflow Initialization**: System initializes agent workflow state and audit logging
3. **Sequential Agent Execution**: Each agent processes inputs from previous steps and generates structured outputs
4. **Human Review Integration**: Workflow pauses at designated checkpoints for manual review and feedback
5. **Validation Metrics Calculation**: Statistical validation using ROC analysis, KS tests, and drift detection
6. **Report Generation**: Comprehensive reports generated in multiple formats with executive summaries
7. **Audit Trail Maintenance**: All actions, decisions, and results logged for compliance tracking

## External Dependencies

### Core Dependencies
- **Streamlit 1.46.0+**: Web application framework for interactive dashboards
- **Pandas 2.3.0+**: Data manipulation and analysis library
- **NumPy 2.3.1+**: Numerical computing foundation
- **Plotly 6.1.2+**: Interactive visualization and charting
- **Scikit-learn 1.7.0+**: Machine learning metrics and validation tools

### Supporting Libraries
- **SciPy**: Statistical functions for KS tests and distribution analysis
- **Altair**: Additional visualization capabilities
- **Jinja2**: Templating for report generation
- **JSONSchema**: Data validation and schema enforcement

## Deployment Strategy

### Replit Configuration
- **Python 3.11 Runtime**: Modern Python version with enhanced performance
- **Autoscale Deployment**: Automatic scaling based on demand
- **Port 5000 Configuration**: Streamlit server running on port 5000
- **Nix Package Management**: Stable channel with glibc locales support

### Workflow Execution
- **Parallel Workflow Support**: Multiple workflow tasks can run concurrently
- **Shell Execution**: Direct command execution for Streamlit server startup
- **Port Waiting**: Automatic wait for port availability before marking deployment ready

### Session Management
- **Stateful Operations**: Workflow state persisted across user sessions
- **Memory Management**: Audit trail limited to 1000 entries to prevent memory issues
- **Cache Resource Management**: Streamlit caching for component initialization

## Changelog

```
Changelog:
- June 26, 2025. Initial setup
- June 26, 2025. Added comprehensive sample data system with realistic credit data (50 records)
- June 26, 2025. Created sample documentation files for compliance testing
- June 26, 2025. Implemented FastAPI backend architecture and LangGraph workflow framework
- June 26, 2025. Enhanced dashboard with quick-start functionality and progress tracking
- June 26, 2025. Fixed documentation loading error with actual sample documents
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```