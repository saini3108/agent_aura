# ValiCred-AI: Credit Risk Model Validation System

A comprehensive AI-powered credit risk model validation system featuring multi-agent workflows, human-in-the-loop capabilities, and regulatory compliance assessment.

## Features

### 🤖 Multi-Agent System
- **Analyst Agent**: Analyzes model structure, data quality, and characteristics
- **Validator Agent**: Calculates performance metrics (AUC, KS, PSI, Gini)
- **Documentation Agent**: Reviews compliance documentation for Basel III, IFRS 9
- **Reviewer Agent**: Generates findings and risk assessments
- **Auditor Agent**: Performs final independent validation and approval
- **Human Review**: Checkpoint for manual oversight and feedback

### 📊 Validation Metrics
- Area Under Curve (AUC) discrimination analysis
- Kolmogorov-Smirnov (KS) statistic for separation
- Population Stability Index (PSI) for drift detection
- Gini coefficient for model power assessment
- ROC curve analysis and lift calculations

### 🎯 Key Capabilities
- Real-time workflow monitoring with progress tracking
- Comprehensive audit trail for compliance
- Interactive dashboard with data visualization
- Sample data loading for testing and demonstration
- Multi-format report generation (summary, detailed, executive)
- Configurable risk thresholds and validation parameters

## Quick Start

### 1. Launch the Application
The ValiCred-AI server is running on port 5000. Access the dashboard to begin validation.

### 2. Load Sample Data
1. Navigate to the Dashboard
2. Click "Load Sample Credit Data" to load 50 sample credit records
3. Click "Load Sample Documents" to add compliance documentation

### 3. Start Validation Workflow
1. Go to "Agent Workflow" page
2. Execute each agent step in sequence:
   - Run Analyst Agent (analyzes data structure)
   - Run Validator Agent (calculates metrics)
   - Run Documentation Agent (reviews compliance)
   - Complete Human Review (provide feedback)
   - Run Reviewer Agent (generates findings)
   - Run Auditor Agent (final approval)

### 4. Review Results
- **Validation Results**: View performance metrics and charts
- **Audit Trail**: Review complete activity log
- **Reports**: Generate comprehensive validation reports

## Sample Data

The system includes realistic sample credit data with:
- 50 customer records with demographics and loan information
- 18% default rate for realistic model testing
- Features: age, income, credit_score, loan_amount, employment_years, debt_to_income
- Multiple loan purposes and home ownership types

## Architecture

### Frontend (Streamlit)
- Interactive multi-page dashboard
- Real-time progress monitoring
- Data visualization with Plotly
- Human review interface

### Backend (Planned FastAPI)
- RESTful API for agent orchestration
- Workflow state management
- MCP (Model Context Protocol) agent architecture
- LangGraph integration for complex workflows

### Agent System
- Modular agent design with specialized responsibilities
- Configurable validation parameters and thresholds
- Comprehensive error handling and logging
- Human-in-the-loop integration points

## Files Structure

```
vali-cred-ai/
├── agents/                 # AI agent implementations
│   ├── analyst_agent.py
│   ├── validator_agent.py
│   ├── documentation_agent.py
│   ├── reviewer_agent.py
│   └── auditor_agent.py
├── utils/                  # Utility modules
│   ├── workflow_manager.py
│   ├── validation_metrics.py
│   ├── audit_logger.py
│   ├── report_generator.py
│   └── sample_data_loader.py
├── sample_data/           # Sample datasets and configuration
│   ├── credit_data.csv
│   ├── validation_parameters.csv
│   └── risk_thresholds.csv
├── config/                # Configuration files
│   └── mcp_agents.json
├── backend/               # FastAPI backend (planned)
│   ├── fastapi_server.py
│   └── langgraph_workflow.py
├── app.py                 # Main Streamlit application
└── start_backend.py       # Backend server starter
```

## Validation Process

### Step 1: Data Analysis
- Data quality assessment
- Feature analysis and target identification
- Missing value and duplicate detection
- Data completeness scoring

### Step 2: Model Performance Validation
- Train simple logistic regression model
- Calculate discrimination metrics (AUC, Gini)
- Assess separation power (KS statistic)
- Monitor population stability (PSI)

### Step 3: Documentation Review
- Parse uploaded compliance documents
- Check for regulatory framework coverage
- Assess documentation completeness
- Generate compliance gaps analysis

### Step 4: Human Review Checkpoint
- Present agent findings for review
- Collect human feedback and assessment
- Allow workflow modifications if needed
- Document human intervention decisions

### Step 5: Risk Assessment
- Synthesize findings from all agents
- Generate comprehensive risk scoring
- Provide actionable recommendations
- Categorize findings by severity

### Step 6: Final Audit
- Independent validation of all components
- Governance and compliance assessment
- Generate final approval recommendation
- Create comprehensive audit report

## Regulatory Compliance

The system supports multiple regulatory frameworks:
- **Basel III**: Capital adequacy and risk management
- **IFRS 9**: Expected credit loss modeling
- **SR 11-7**: Model Risk Management guidance
- **Model Documentation Standards**: Comprehensive documentation requirements

## Performance Thresholds

Default validation thresholds:
- AUC: Minimum 0.7 (Excellent: 0.8+)
- KS Statistic: Minimum 0.2 (Excellent: 0.3+)
- PSI: Maximum 0.25 (Stable: <0.1)
- Data Quality: Minimum 80% completeness

## Future Enhancements

### Planned Features
- Full LangGraph workflow integration
- Advanced agent orchestration with conditional logic
- Real-time model monitoring capabilities
- Integration with external model repositories
- Advanced visualization and reporting
- Multi-model comparison capabilities

### Technical Roadmap
- FastAPI backend deployment
- Database integration for state persistence
- Advanced authentication and authorization
- Kubernetes deployment configuration
- CI/CD pipeline integration

## Usage Notes

- Load sample data first to explore system capabilities
- Each agent step builds on previous results
- Human review is required at step 4 to continue workflow
- All actions are logged in the audit trail
- Reports can be generated at any stage of validation

The system demonstrates enterprise-ready model validation capabilities while maintaining simplicity for testing and evaluation.