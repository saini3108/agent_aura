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

## Setup and Installation

### New User Setup
1. **Fork or Clone**: Fork this project to your Replit account
2. **Install Dependencies**: Dependencies are automatically installed when you run the project
3. **Start Application**: Click the "Run" button or use the configured workflow
4. **Access Dashboard**: The app will be available at the provided URL on port 5000

### System Requirements
- Python 3.11+
- Required packages are managed automatically via `pyproject.toml`
- Streamlit configuration is pre-configured in `.streamlit/config.toml`

### Local Setup Instructions

To run ValiCred-AI on your laptop:

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd vali-cred-ai
   ```

2. **Set up Python Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   # Install required packages
   pip install streamlit pandas numpy plotly scikit-learn fastapi uvicorn pydantic scipy
   
   # Or use the project file
   pip install -e .
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py --server.port 5000
   ```

5. **Access the Dashboard**
   - Open your browser to `http://localhost:5000`
   - The ValiCred-AI dashboard will be available

## Quick Start

### 1. Launch the Application
The ValiCred-AI server runs automatically on port 5000. Access the dashboard to begin validation.

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

## Implementation Status

### ✅ Currently Implemented (Agent Aura Architecture)
- **MCP + LangGraph Engine**: Complete workflow orchestration with async execution
- **Human-in-the-Loop**: Interactive review checkpoints with approval workflow
- **Agent Aura Structure**: Clean separation of frontend, agents, flows, and shared utilities
- **Multi-Agent System**: All 5 specialized agents with configurable parameters
- **Dynamic Configuration**: Interactive UI for thresholds, parameters, and workflow settings
- **Real-time Monitoring**: Live workflow status, progress tracking, and error handling
- **Comprehensive Audit Trail**: Full activity logging with session management
- **Sample Data Integration**: Realistic credit dataset and compliance documents

### 🔄 Enhanced Features
- **Workflow Engine**: MCP-compliant engine with retry logic and error recovery
- **Configuration Management**: Dynamic parameter adjustment through UI
- **Status Dashboard**: Real-time system monitoring and workflow visualization
- **Interactive Review**: Streamlined human feedback collection and processing
- **Robust Error Handling**: Comprehensive error tracking and recovery mechanisms

### 📋 Architecture Implementation

| Component | Implementation Status | Technology Stack |
|-----------|----------------------|------------------|
| Frontend | ✅ Complete | Streamlit with agent_aura structure |
| Workflow Engine | ✅ MCP + LangGraph | Custom async orchestration engine |
| Agents | ✅ Specialized | MCP-compliant agent classes |
| Configuration | ✅ Dynamic | Interactive UI management |
| Human-in-Loop | ✅ Interactive | Checkpoint-based review system |
| Audit System | ✅ Comprehensive | Real-time logging and tracking |
| Data Management | ✅ Integrated | Sample data with file handling |

## Architecture

### Frontend (Streamlit)
- Interactive multi-page dashboard
- Real-time progress monitoring
- Data visualization with Plotly
- Human review interface

### Backend (FastAPI Foundation)
- RESTful API endpoints for agent orchestration
- Workflow state management framework
- MCP (Model Context Protocol) agent configuration
- LangGraph simulation for complex workflows

### Agent System
- Modular agent design with specialized responsibilities
- Configurable validation parameters and thresholds
- Comprehensive error handling and logging
- Human-in-the-loop integration points

## Project Structure (Agent Aura Architecture)

```
agent_aura/
├── frontend/              # Streamlit application
│   ├── app.py            # Main application with MCP integration
│   └── configuration_manager.py  # Interactive configuration UI
├── agent-service/
│   ├── agents/           # MCP agent implementations
│   │   ├── analyst_agent.py      # Data analysis agent
│   │   ├── validator_agent.py    # Metrics validation agent
│   │   ├── documentation_agent.py # Compliance review agent
│   │   ├── reviewer_agent.py     # Findings generation agent
│   │   └── auditor_agent.py      # Final audit agent
│   ├── flows/            # LangGraph workflow orchestration
│   │   └── mcp_workflow_engine.py # MCP + LangGraph engine
│   ├── tools/            # FastAPI service tools
│   │   ├── fastapi_server.py     # API endpoints
│   │   └── langgraph_workflow.py # Workflow definitions
│   └── memory/           # Context and state management
├── shared/               # Shared utilities and configuration
│   ├── system_config.py  # Dynamic configuration management
│   ├── audit_logger.py   # Comprehensive audit trail
│   ├── validation_metrics.py # Statistical calculations
│   ├── workflow_manager.py    # Workflow coordination
│   ├── report_generator.py    # Multi-format reporting
│   ├── sample_data_loader.py  # Data loading utilities
│   └── sample_data/      # Sample datasets
│       ├── credit_data.csv
│       ├── validation_parameters.csv
│       └── compliance_documents/
├── docker/               # Docker configuration
└── main.py              # Application entry point
app.py                   # Root entry point for Replit
README.md               # This documentation
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

## Enhanced Agent Aura Features

### MCP + LangGraph Integration
- **Async Workflow Engine**: Complete orchestration with pause/resume capabilities
- **Human-in-the-Loop**: Interactive review checkpoints with approval workflow
- **Dynamic Configuration**: Real-time parameter adjustment through interactive UI
- **Comprehensive Monitoring**: Live status tracking, progress visualization, and error handling

### Configuration Management
- **Risk Thresholds**: Dynamic AUC, KS, PSI, and data quality threshold configuration
- **Validation Parameters**: Cross-validation, model training, and statistical test settings
- **Workflow Settings**: Timeout, retry policies, and human review configurations
- **Agent Configuration**: Individual agent timeout and retry attempt settings

### Workflow Capabilities
- **Workflow Creation**: Automated workflow initialization with data and document inputs
- **Step-by-Step Execution**: Individual agent execution with real-time status updates
- **Error Recovery**: Automatic retry logic with configurable parameters
- **Audit Integration**: Complete activity logging with session management

### Interactive Dashboard
- **Real-time Metrics**: Active workflows, agent status, and system health monitoring
- **Progress Tracking**: Visual progress bars and step-by-step execution status
- **Data Integration**: Sample data loading with immediate preview and analysis
- **Status Visualization**: Agent execution times, retry counts, and error tracking

## Usage Guide

### Quick Start Workflow
1. **Load Data**: Use "Load Sample Credit Data" to initialize the system
2. **Load Documents**: Add compliance documents with "Load Sample Documents"
3. **Create Workflow**: Navigate to MCP Workflow and create a new validation workflow
4. **Execute Steps**: Run each agent individually (Analyst → Validator → Documentation)
5. **Human Review**: Complete the human review checkpoint with feedback
6. **Final Steps**: Execute Reviewer and Auditor agents for completion
7. **Monitor Progress**: Track execution through the real-time dashboard

### Configuration Management
- Access the Configuration tab to adjust risk thresholds and validation parameters
- Modify workflow settings including timeouts and retry policies
- Update agent-specific configurations for optimal performance
- Changes take effect immediately without requiring system restart

### Audit and Monitoring
- Review complete audit trail in the Audit Trail section
- Monitor system status and active workflows in System Status
- Track workflow execution history and performance metrics
- Export audit logs for compliance and reporting purposes

The agent_aura architecture provides enterprise-ready model validation with modern MCP + LangGraph orchestration while maintaining intuitive user interaction through Streamlit.