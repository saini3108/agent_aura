# ValiCred-AI: Advanced Credit Risk Model Validation

A comprehensive AI-powered system for credit risk model validation, monitoring, and regulatory compliance.

## 🏗️ Clean Architecture

The system follows a modular architecture with clear separation of concerns:

```
ValiCred-AI/
├── src/
│   ├── core/                     # Core system components
│   │   ├── app_factory.py        # Application initialization
│   │   ├── mcp_protocol.py       # MCP implementation
│   │   ├── langgraph_workflow.py # LangGraph workflows
│   │   ├── human_in_loop.py      # Human review system
│   │   └── memory_manager.py     # Memory management
│   ├── agents/                   # AI Validation Agents
│   │   ├── analyst_agent.py      # Data analysis agent
│   │   ├── validator_agent.py    # Model validation agent
│   │   ├── documentation_agent.py# Document review agent
│   │   ├── reviewer_agent.py     # Final review agent
│   │   └── auditor_agent.py      # Audit compliance agent
│   ├── ui/                       # User Interface Components
│   │   ├── dashboard.py          # Main dashboard
│   │   ├── workflow_interface.py # Workflow management
│   │   ├── live_agent_display.py # Real-time agent status
│   │   ├── reports_interface.py  # Reports and analytics
│   │   └── configuration_panel.py# System configuration
│   ├── utils/                    # Utility Functions
│   │   ├── workflow_engine.py    # Workflow orchestration
│   │   ├── audit_logger.py       # Audit logging
│   │   ├── validation_metrics.py # Statistical calculations
│   │   ├── sample_data_loader.py # Data loading utilities
│   │   └── enhanced_report_generator.py # Report generation
│   └── config/                   # Configuration Files
│       ├── settings.py           # Application settings
│       ├── mcp_agents.json       # Agent configurations
│       ├── workflow_config.json  # Workflow definitions
│       ├── compliance_frameworks.json # Regulatory frameworks
│       ├── risk_thresholds.json  # Risk threshold settings
│       ├── validation_parameters.json # Validation parameters
│       └── ui_settings.json      # UI configurations
├── sample_data/                  # Sample Datasets
│   ├── credit_data.csv           # Sample credit portfolio
│   ├── governance_policy.txt     # Sample governance docs
│   ├── model_methodology_document.txt
│   ├── model_validation_report.txt
│   ├── risk_thresholds.csv
│   └── validation_parameters.csv
├── docs/                         # Documentation
│   └── ONBOARDING.md            # Developer onboarding guide
└── app.py                       # Main application entry point (64 lines)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Streamlit
- Required dependencies (automatically installed)

### Installation & Setup

1. **Clone and Navigate**
   ```bash
   git clone <repository-url>
   cd ValiCred-AI
   ```

2. **Run the Application**
   ```bash
   streamlit run app.py --server.port 5000
   ```

3. **Access the Application**
   - Open your browser to `http://localhost:5000`
   - Navigate through the dashboard tabs

## 🎯 Core Features

### Multi-Agent Validation System
- **Analyst Agent**: Performs comprehensive data analysis and quality assessment
- **Validator Agent**: Calculates key validation metrics (AUC, KS, PSI, Gini)
- **Documentation Agent**: Reviews model documentation for completeness
- **Reviewer Agent**: Conducts final review and generates recommendations
- **Auditor Agent**: Ensures regulatory compliance

### Interactive Dashboard
- **Real-time Workflow Monitoring**: Live agent execution status
- **MCP Workflow Interface**: Step-by-step validation process
- **Comprehensive Reports**: Detailed validation findings and recommendations
- **Human-in-the-Loop**: Manual review checkpoints for critical decisions

### Advanced Capabilities
- **LangGraph Integration**: Sophisticated workflow orchestration
- **Memory Management**: Persistent session state
- **Audit Logging**: Comprehensive activity tracking
- **Regulatory Compliance**: Basel III, IFRS 9, SR 11-7 alignment

## 📊 Validation Metrics

The system calculates and monitors key validation metrics:

- **AUC (Area Under Curve)**: Model discrimination power
- **KS Statistic**: Maximum separation between distributions
- **PSI (Population Stability Index)**: Data drift detection
- **Gini Coefficient**: Alternative discrimination measure
- **Data Quality Scores**: Completeness, consistency, accuracy

## 🔧 Configuration

### Environment Setup
```python
# Key configuration files:
- src/config/settings.py          # Main application settings
- src/config/workflow_config.json # Workflow execution order
- src/config/risk_thresholds.json # Risk assessment thresholds
```

### Customization
- **Agent Behavior**: Modify agent configurations in `src/config/mcp_agents.json`
- **UI Settings**: Customize interface in `src/config/ui_settings.json`
- **Validation Parameters**: Adjust metrics in `src/config/validation_parameters.json`

## 🎯 Usage Examples

### 1. Standard Model Validation
```python
# Upload your credit model data
# Run through MCP Workflow
# Review agent findings
# Generate compliance reports
```

### 2. Custom Risk Thresholds
```python
# Modify risk_thresholds.json
# Restart workflow with new parameters
# Monitor updated validation results
```

### 3. Regulatory Reporting
```python
# Access Reports tab
# Generate compliance-specific reports
# Export findings for regulatory submission
```

## 🏛️ Regulatory Compliance

### Supported Frameworks
- **Basel III**: Capital adequacy requirements
- **IFRS 9**: Expected credit loss modeling
- **SR 11-7**: Federal Reserve supervisory guidance
- **CECL**: Current expected credit losses

### Compliance Features
- Automated regulatory checklist validation
- Missing documentation identification
- Risk threshold breach detection
- Audit trail generation

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py --server.port 5000
```

### Production Deployment
- Configured for Replit deployment
- Scalable architecture for enterprise use
- Secure audit logging and compliance tracking

## 📝 Contributing

1. Follow the modular architecture patterns
2. Add comprehensive docstrings
3. Include unit tests for new features
4. Update configuration files as needed
5. Maintain backwards compatibility

## 🔍 Troubleshooting

### Common Issues
- **Import Errors**: Ensure all dependencies are installed
- **Data Loading**: Check sample data format compatibility
- **Agent Failures**: Review audit logs for detailed error messages
- **UI Issues**: Clear browser cache and restart application

### Support
- Check `docs/ONBOARDING.md` for detailed setup instructions
- Review audit logs for debugging information
- Verify configuration file formats

## 📊 Performance

- **Scalable**: Handles large credit portfolios
- **Efficient**: Optimized agent execution
- **Reliable**: Comprehensive error handling
- **Maintainable**: Clean, modular codebase

## 🔐 Security

- Secure file handling
- Audit trail logging
- Configuration-based access control
- Data privacy compliance

---

**ValiCred-AI** - Transforming credit risk validation through intelligent automation and regulatory excellence.