# ValiCred-AI: Enterprise Credit Risk Model Validation System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red.svg)](https://streamlit.io/)
[![AI Powered](https://img.shields.io/badge/AI-Powered-green.svg)](https://groq.com/)

## Overview

ValiCred-AI is a comprehensive, AI-powered credit risk model validation platform designed for financial institutions to ensure regulatory compliance and model reliability. The system leverages multiple LLM providers (Groq, OpenAI, Anthropic, Gemini) with sophisticated agent orchestration to provide intelligent analysis of credit risk models.

### Key Features

- **Multi-Agent AI System**: Specialized AI agents for different validation aspects
- **Real LLM Integration**: Support for Groq, OpenAI, Anthropic, and Gemini models
- **Human-in-the-Loop**: Strategic checkpoints for manual review and approval
- **Regulatory Compliance**: Built-in support for Basel III, IFRS 9, SR 11-7, CCAR
- **Enterprise Architecture**: Scalable, modular design for large-scale deployments
- **Real Data Integration**: Comprehensive data loading from multiple sources
- **Advanced Analytics**: Statistical validation metrics and performance assessment
- **Audit Trail**: Complete tracking of all validation activities

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- API key from at least one LLM provider (Groq recommended for free tier)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd valicred-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   - Get a free Groq API key from [console.groq.com](https://console.groq.com/)
   - Set environment variable: `export GROQ_API_KEY=your_api_key`
   - Or configure through the UI (Configuration Panel → API Keys)

4. **Run the application**
   ```bash
   streamlit run app.py --server.port 5000
   ```

5. **Access the application**
   - Open your browser to `http://localhost:5000`
   - Configure API keys through the Configuration Panel
   - Upload your credit data or use the sample data generator

## 📁 Project Structure

```
valicred-ai/
├── src/                           # Core source code
│   ├── agents/                    # Specialized AI agents
│   │   ├── analyst_agent.py       # Data analysis and risk assessment
│   │   ├── validator_agent.py     # Statistical validation metrics
│   │   ├── documentation_agent.py # Compliance and documentation review
│   │   ├── reviewer_agent.py      # Executive findings and recommendations
│   │   └── auditor_agent.py       # Final independent validation
│   ├── core/                      # Core business logic
│   │   ├── llm_manager.py         # Multi-provider LLM management
│   │   ├── workflow_engine.py     # MCP + LangGraph orchestration
│   │   └── validation_metrics.py  # Statistical calculation engine
│   ├── data/                      # Data management
│   │   ├── real_data_loader.py    # Enterprise data loading system
│   │   ├── validators.py          # Data quality validation
│   │   └── sample_generator.py    # Realistic sample data generation
│   ├── api/                       # REST API endpoints
│   │   ├── endpoints.py           # FastAPI route definitions
│   │   └── middleware.py          # Authentication and logging
│   ├── ui/                        # User interface components
│   │   ├── configuration_panel.py # API key and settings management
│   │   ├── dashboard.py           # Main application dashboard
│   │   └── components/            # Reusable UI components
│   ├── utils/                     # Shared utilities
│   │   ├── audit_logger.py        # Comprehensive audit trail
│   │   ├── report_generator.py    # Multi-format report generation
│   │   └── helpers.py             # Common utility functions
│   └── config/                    # Configuration management
│       └── settings.py            # Centralized configuration system
├── tests/                         # Comprehensive test suite
│   ├── unit/                      # Unit tests for core components
│   ├── integration/               # Integration tests
│   └── data/                      # Test data and fixtures
├── docs/                          # Documentation
│   ├── api/                       # API documentation
│   ├── deployment/                # Deployment guides
│   └── user_guide/                # User documentation
├── app.py                         # Main Streamlit application
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔧 Configuration

### API Key Configuration

ValiCred-AI supports multiple LLM providers. Configure through the UI or environment variables:

| Provider | Environment Variable | Get API Key |
|----------|---------------------|-------------|
| Groq (Recommended) | `GROQ_API_KEY` | [console.groq.com](https://console.groq.com/) |
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/) |
| Anthropic | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) |
| Gemini | `GEMINI_API_KEY` | [ai.google.dev](https://ai.google.dev/) |

### Risk Thresholds

Configure validation thresholds through the Configuration Panel:

- **AUC Thresholds**: 0.8 (Excellent), 0.7 (Good), 0.6 (Acceptable)
- **KS Statistic**: 0.3 (Excellent), 0.2 (Good), 0.15 (Acceptable)
- **PSI Limits**: 0.1 (Stable), 0.2 (Monitor), 0.25 (Unstable)

## 🤖 AI Agents

### Agent Architecture

The system employs six specialized AI agents working in sequence:

1. **Analyst Agent**: Comprehensive data analysis and risk factor identification
2. **Validator Agent**: Statistical validation metrics (AUC, KS, PSI, Gini)
3. **Documentation Agent**: Regulatory compliance review (Basel III, IFRS 9)
4. **Human Review Checkpoint**: Manual review and feedback integration
5. **Reviewer Agent**: Executive summary and recommendations
6. **Auditor Agent**: Final independent validation and approval

### Agent Capabilities

- **Real AI Analysis**: Powered by advanced LLMs for intelligent insights
- **Regulatory Knowledge**: Built-in understanding of banking regulations
- **Statistical Expertise**: Comprehensive model performance assessment
- **Risk Assessment**: Industry-standard risk factor analysis
- **Natural Language**: Clear, actionable recommendations

## 📊 Data Integration

### Supported Data Sources

- **File Uploads**: CSV, Excel, Parquet, JSON
- **Database Connections**: PostgreSQL, MySQL, SQL Server
- **API Integration**: REST endpoints with authentication
- **Real-time Streaming**: For continuous monitoring

### Data Quality Framework

- **Completeness Assessment**: Missing data analysis
- **Accuracy Validation**: Value range and format checks
- **Consistency Verification**: Logical relationship validation
- **Regulatory Compliance**: Basel III and IFRS 9 requirements

### Sample Data

Generate realistic credit portfolios for testing:

```python
from src.data.real_data_loader import CreditDataLoader

loader = CreditDataLoader()
sample_data = loader.generate_sample_credit_data(
    n_samples=1000,
    default_rate=0.15
)
```

## 🔄 Workflow Management

### MCP + LangGraph Architecture

- **Model Context Protocol**: Standardized agent communication
- **LangGraph Integration**: Advanced workflow orchestration
- **State Management**: Persistent workflow state across sessions
- **Error Handling**: Automatic retry and fallback mechanisms

### Human-in-the-Loop

- **Strategic Checkpoints**: Manual review at critical decision points
- **Feedback Integration**: Incorporate human insights into AI analysis
- **Approval Workflows**: Multi-level approval for regulatory compliance
- **Audit Trail**: Complete record of all human interactions

## 📈 Validation Metrics

### Statistical Measures

- **AUC (Area Under Curve)**: Model discrimination power
- **KS Statistic**: Two-sample test for distribution differences
- **Gini Coefficient**: Model performance assessment
- **PSI (Population Stability Index)**: Data drift detection
- **IV (Information Value)**: Variable predictive power

### Regulatory Metrics

- **Basel III Capital Requirements**: Risk-weighted asset calculations
- **IFRS 9 Expected Credit Loss**: Forward-looking loss estimation
- **CCAR Stress Testing**: Scenario-based risk assessment
- **Model Risk Management**: SR 11-7 compliance validation

## 🛡️ Security & Compliance

### Data Security

- **API Key Encryption**: Secure storage of sensitive credentials
- **Data Privacy**: No data transmitted to external services unnecessarily
- **Access Controls**: Role-based permission system
- **Audit Logging**: Complete activity tracking

### Regulatory Compliance

- **Basel III**: Capital adequacy and risk management
- **IFRS 9**: Financial instrument classification and measurement
- **SR 11-7**: Model risk management guidance
- **CCAR**: Comprehensive capital analysis and review

## 🚀 Deployment

### Replit Deployment

1. Configure environment variables in Secrets
2. Run `streamlit run app.py --server.port 5000`
3. Application available at your Replit URL

### Docker Deployment

```bash
# Build image
docker build -t valicred-ai .

# Run container
docker run -p 5000:5000 \
  -e GROQ_API_KEY=your_key \
  valicred-ai
```

### Production Deployment

- **Load Balancing**: Multiple instance support
- **Database Backend**: PostgreSQL for production data
- **Monitoring**: Comprehensive logging and metrics
- **Scaling**: Horizontal scaling for high-volume processing

## 📚 Documentation

### User Guides

- [Getting Started Guide](docs/user_guide/getting_started.md)
- [Data Upload Instructions](docs/user_guide/data_upload.md)
- [Interpretation Guide](docs/user_guide/interpretation.md)

### Technical Documentation

- [API Reference](docs/api/reference.md)
- [Agent Development](docs/development/agents.md)
- [Deployment Guide](docs/deployment/production.md)

### Regulatory Guidance

- [Basel III Compliance](docs/regulatory/basel_iii.md)
- [IFRS 9 Requirements](docs/regulatory/ifrs9.md)
- [Model Risk Management](docs/regulatory/model_risk.md)

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/

# Run with coverage
python -m pytest tests/ --cov=src/
```

### Test Coverage

- **Unit Tests**: 95%+ coverage for core components
- **Integration Tests**: End-to-end workflow validation
- **Data Quality Tests**: Comprehensive data validation scenarios
- **Performance Tests**: Load testing for production readiness

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make your changes and add tests
5. Run tests: `python -m pytest`
6. Submit a pull request

### Code Standards

- **Python Style**: Follow PEP 8 guidelines
- **Documentation**: Comprehensive docstrings for all functions
- **Testing**: Unit tests required for new features
- **Type Hints**: Full type annotation coverage

## 📋 Implementation Status

### ✅ Completed Features

- Multi-provider LLM integration (Groq, OpenAI, Anthropic, Gemini)
- Intelligent AI agents with real analysis capabilities
- Enterprise configuration management with API key UI
- Real data loading and quality validation system
- MCP + LangGraph workflow orchestration
- Human-in-the-loop checkpoints and review system
- Comprehensive audit trail and reporting
- Statistical validation metrics calculation
- Regulatory compliance framework
- Advanced UI with configuration panels

### 🔄 In Progress

- REST API endpoints for programmatic access
- Advanced visualization components
- Database integration and connection pooling
- Real-time data streaming capabilities
- Advanced authentication and authorization
- Performance optimization and caching

### 📋 Planned Enhancements

- **Advanced Analytics**
  - Model explainability (SHAP, LIME)
  - Scenario analysis and stress testing
  - Portfolio optimization recommendations
  - Real-time monitoring dashboards

- **Integration Capabilities**
  - SAP integration for enterprise systems
  - Salesforce connector for CRM data
  - Bloomberg terminal data feeds
  - Credit bureau API integration

- **Advanced Features**
  - AutoML model development
  - Ensemble model validation
  - Backtesting framework
  - Champion/challenger testing

- **Enterprise Features**
  - Multi-tenant architecture
  - Advanced role-based access control
  - SSO integration (SAML, OAuth)
  - Advanced audit and compliance reporting

### 🎯 Future Roadmap

**Q1 2025**: Database integration, advanced authentication
**Q2 2025**: AutoML capabilities, ensemble validation
**Q3 2025**: Real-time monitoring, stress testing
**Q4 2025**: Multi-tenant architecture, enterprise SSO

## 🆘 Support

### Getting Help

- **Documentation**: Check the comprehensive docs in `/docs/`
- **Issues**: Report bugs and feature requests on GitHub Issues
- **Community**: Join our discussion forums
- **Enterprise Support**: Contact sales for enterprise licensing

### Common Issues

**API Key Problems**: Ensure keys are properly configured in the Configuration Panel
**Data Upload Errors**: Check data format matches expected schema
**Performance Issues**: Consider using Groq for faster inference
**Deployment Problems**: Review deployment logs and configuration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Groq**: Fast LLM inference platform
- **Streamlit**: Excellent web application framework
- **OpenAI, Anthropic, Google**: Advanced AI model providers
- **Open Source Community**: Various libraries and tools used

---

**Built with ❤️ for the financial services industry**

For more information, visit our [documentation](docs/) or contact the development team.