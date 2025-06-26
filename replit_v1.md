# ValiCred-AI: Credit Risk Model Validation System

## Overview

ValiCred-AI is an intelligent credit risk model validation system powered by agentic AI with human-in-the-loop collaboration. The system provides a modular approach to validating credit risk models through specialized AI agents that handle different aspects of the validation process, from initial analysis to final audit approval.

The application is built using Streamlit as the frontend framework with Python 3.11, providing an interactive web interface for uploading data, monitoring agent workflows, and reviewing validation results.

## System Architecture

### Frontend Architecture
- **Streamlit Web Application**: Interactive dashboard built with Streamlit providing real-time workflow monitoring, data upload capabilities, and results visualization
- **Plotly Integration**: Advanced charting and visualization for validation metrics, ROC curves, and performance analytics
- **Session State Management**: Persistent workflow state across user sessions with audit trail preservation

### Backend Architecture
- **Multi-Agent System**: Six specialized AI agents working in sequence with conditional logic and human review checkpoints
- **Workflow Orchestration**: Centralized workflow management with state tracking, step validation, and execution control
- **Validation Engine**: Comprehensive metrics calculation including AUC, KS statistics, PSI, and model performance indicators

### Agent Architecture
The system employs six specialized agents:
1. **Analyst Agent**: Analyzes model structure, parameters, and data characteristics
2. **Validator Agent**: Calculates validation metrics (AUC, KS test, drift detection)
3. **Documentation Agent**: Reviews compliance documentation for regulatory requirements
4. **Human Review**: Workflow pause point for manual review and feedback injection
5. **Reviewer Agent**: Generates findings and recommendations based on validation results
6. **Auditor Agent**: Performs final independent validation and approval assessment

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