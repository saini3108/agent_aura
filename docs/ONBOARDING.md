
# ValiCred-AI Developer Onboarding Guide

Welcome to the ValiCred-AI development team! This guide will help you understand the system architecture, set up your development environment, and contribute effectively.

## 🏗️ System Architecture Overview

ValiCred-AI follows a clean, modular architecture that separates concerns into logical components:

### Core Architecture Principles
1. **Modular Design**: Each component has a single responsibility
2. **Clean Interfaces**: Well-defined APIs between modules
3. **Scalable Structure**: Easy to extend and maintain
4. **Configuration-Driven**: Behavior controlled through config files

### Directory Structure
```
src/
├── core/                     # System Core Components
│   ├── app_factory.py        # Application initialization & dependency injection
│   ├── mcp_protocol.py       # Model Control Protocol implementation
│   ├── langgraph_workflow.py # Advanced workflow orchestration
│   ├── human_in_loop.py      # Human review integration
│   └── memory_manager.py     # Session and state management
├── agents/                   # AI Validation Agents
│   ├── analyst_agent.py      # Data analysis & quality assessment
│   ├── validator_agent.py    # Statistical validation metrics
│   ├── documentation_agent.py# Document completeness review
│   ├── reviewer_agent.py     # Final review & recommendations
│   └── auditor_agent.py      # Regulatory compliance audit
├── ui/                       # User Interface Components
│   ├── dashboard.py          # Main dashboard interface
│   ├── workflow_interface.py # MCP workflow management
│   ├── live_agent_display.py # Real-time agent monitoring
│   ├── reports_interface.py  # Report generation & viewing
│   └── configuration_panel.py# System configuration UI
├── utils/                    # Utility Functions
│   ├── workflow_engine.py    # Core workflow orchestration
│   ├── audit_logger.py       # Comprehensive audit logging
│   ├── validation_metrics.py # Statistical calculation utilities
│   ├── sample_data_loader.py # Data loading & preprocessing
│   └── enhanced_report_generator.py # Advanced report generation
└── config/                   # Configuration Management
    ├── settings.py           # Application-wide settings
    ├── mcp_agents.json       # Agent behavior configuration
    ├── workflow_config.json  # Workflow execution definitions
    ├── compliance_frameworks.json # Regulatory framework specs
    ├── risk_thresholds.json  # Risk assessment thresholds
    ├── validation_parameters.json # Validation metric parameters
    └── ui_settings.json      # User interface configuration
```

## 🚀 Development Environment Setup

### Prerequisites
- Python 3.11+
- Basic understanding of Streamlit framework
- Familiarity with pandas, numpy for data processing
- Knowledge of financial risk modeling (helpful but not required)

### Getting Started
1. **Environment Setup**
   ```bash
   # The system automatically manages dependencies
   # No virtual environment needed in Replit
   ```

2. **Run the Application**
   ```bash
   streamlit run app.py --server.port 5000
   ```

3. **Access Development Interface**
   - Main application: `http://0.0.0.0:5000`
   - All features available in development mode

## 🔧 Key Components Deep Dive

### 1. Core Components (`src/core/`)

#### `app_factory.py`
- **Purpose**: Central application initialization
- **Key Functions**: Component creation, dependency injection
- **When to Modify**: Adding new system-wide components

#### `workflow_engine.py`
- **Purpose**: Orchestrates multi-agent workflows
- **Key Functions**: Step execution, state management, error handling
- **When to Modify**: Adding new workflow steps or execution logic

#### `mcp_protocol.py`
- **Purpose**: Implements Model Control Protocol for agent communication
- **Key Functions**: Agent messaging, status tracking, result aggregation
- **When to Modify**: Enhancing agent communication protocols

### 2. Agent System (`src/agents/`)

#### Agent Development Pattern
```python
class NewAgent:
    def __init__(self, config):
        self.config = config
    
    def run(self, context):
        # Agent logic here
        return {
            'status': 'completed',
            'analysis': {},
            'recommendations': []
        }
```

#### Adding New Agents
1. Create agent file in `src/agents/`
2. Update `src/config/mcp_agents.json`
3. Add to workflow execution order in `src/config/workflow_config.json`
4. Update UI components to display results

### 3. User Interface (`src/ui/`)

#### UI Development Guidelines
- Use Streamlit components consistently
- Implement proper error handling
- Add loading states for long operations
- Follow existing styling patterns

#### Adding New UI Components
1. Create component in appropriate UI module
2. Import and integrate in `app.py`
3. Update navigation if needed
4. Test responsive behavior

## 📊 Configuration Management

### Configuration Files Overview

#### `settings.py` - Global Settings
```python
class Settings:
    AUDIT_LEVEL = "INFO"
    MAX_WORKFLOW_STEPS = 10
    DEFAULT_THRESHOLDS = {...}
```

#### `workflow_config.json` - Workflow Definitions
```json
{
  "execution_order": ["analyst", "validator", "documentation", "human_review", "reviewer", "auditor"],
  "parallel_execution": false,
  "timeout_seconds": 300
}
```

#### `risk_thresholds.json` - Risk Assessment Parameters
```json
{
  "auc_threshold": 0.65,
  "ks_threshold": 0.3,
  "psi_threshold": 0.25
}
```

### Modifying Configurations
1. **Development**: Edit JSON files directly
2. **Production**: Use configuration panel in UI
3. **Validation**: System validates config on startup

## 🔄 Development Workflow

### 1. Feature Development Process
```bash
# 1. Understand the requirement
# 2. Identify affected components
# 3. Plan the implementation
# 4. Code the feature
# 5. Test thoroughly
# 6. Update documentation
```

### 2. Testing Strategy
- **Unit Testing**: Test individual components
- **Integration Testing**: Test component interactions
- **End-to-End Testing**: Test complete workflows
- **User Testing**: Validate UI/UX improvements

### 3. Code Quality Standards
- Follow PEP 8 for Python code style
- Add comprehensive docstrings
- Include type hints where appropriate
- Handle errors gracefully

## 🐛 Debugging Guide

### Common Issues and Solutions

#### 1. Agent Execution Failures
```python
# Check audit logs
audit_logger.get_recent_logs()

# Verify agent configuration
with open('src/config/mcp_agents.json') as f:
    config = json.load(f)
```

#### 2. UI Component Issues
- Check browser console for JavaScript errors
- Verify Streamlit component integration
- Test with different data inputs

#### 3. Configuration Problems
- Validate JSON syntax
- Check file permissions
- Verify configuration schema compliance

### Debugging Tools
1. **Audit Logger**: Comprehensive system activity logging
2. **Streamlit Debugging**: Built-in error reporting
3. **Python Debugging**: Standard debugging techniques

## 📈 Performance Optimization

### Best Practices
1. **Efficient Data Processing**: Use vectorized operations
2. **Memory Management**: Monitor memory usage in long workflows
3. **Caching**: Implement appropriate caching strategies
4. **Async Operations**: Use async/await for I/O operations

### Monitoring
- Track agent execution times
- Monitor memory usage
- Log performance metrics

## 🔐 Security Considerations

### Data Security
- Sanitize all user inputs
- Validate file uploads
- Implement proper access controls

### Audit Compliance
- Log all user actions
- Maintain data lineage
- Ensure regulatory compliance

## 🚀 Deployment Considerations

### Environment Preparation
- Verify all dependencies
- Test configuration files
- Validate data connections

### Production Checklist
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Configuration validated
- [ ] Performance tested
- [ ] Security reviewed

## 📚 Additional Resources

### Documentation
- [Streamlit Documentation](https://docs.streamlit.io)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Financial Risk Modeling Best Practices]

### Code Examples
- Check `sample_data/` for example data formats
- Review existing agents for implementation patterns
- Study configuration files for customization options

## 🤝 Contributing Guidelines

### Code Contributions
1. Follow the established architecture patterns
2. Add comprehensive tests
3. Update relevant documentation
4. Ensure backwards compatibility

### Bug Reports
1. Provide detailed reproduction steps
2. Include system configuration
3. Attach relevant log files
4. Describe expected vs actual behavior

### Feature Requests
1. Describe the business need
2. Provide implementation suggestions
3. Consider impact on existing features
4. Include acceptance criteria

---

Welcome to the team! If you have questions, refer to this guide or check the audit logs for system behavior insights.
