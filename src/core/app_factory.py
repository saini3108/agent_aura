"""
Application Factory
==================

Main application initialization and configuration
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_application() -> Tuple[Any, Any, Any, bool]:
    """Initialize the ValiCred-AI application components"""
    
    # Always use fallback components for now to avoid complex dependencies
    logger.info("Initializing with fallback components for stability")
    return _initialize_fallback_components()

def _initialize_fallback_components() -> Tuple[Any, Any, Any, bool]:
    """Initialize fallback components when advanced ones are unavailable"""
    
    from src.utils.workflow_engine import MCPWorkflowEngine, AgentExecution
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    # Simple fallback implementations
    class SimpleConfig:
        def __init__(self):
            self.mcp_config = {
                "workflow": {"execution_order": ["analyst", "validator", "documentation", "human_review", "reviewer", "auditor"]},
                "agents": {
                    "analyst": {"timeout": 300, "retry_attempts": 3},
                    "validator": {"timeout": 600, "retry_attempts": 3},
                    "documentation": {"timeout": 300, "retry_attempts": 2},
                    "reviewer": {"timeout": 300, "retry_attempts": 2},
                    "auditor": {"timeout": 300, "retry_attempts": 2}
                }
            }
            self.workflow_config = {"retry_policy": {"max_retries": 3, "retry_delay": 5}}
            self.risk_thresholds = {
                "auc": {"excellent": 0.8, "good": 0.7, "acceptable": 0.6},
                "ks": {"excellent": 0.3, "good": 0.2, "acceptable": 0.15}
            }

        def get_agent_config(self, agent_name: str):
            return self.mcp_config.get("agents", {}).get(agent_name, {})

    class SimpleAuditLogger:
        def __init__(self):
            self.entries = []

        def log_workflow_event(self, event_type, step_name, step_index, additional_info=None):
            self.entries.append({
                "timestamp": datetime.now(),
                "event_type": event_type,
                "step_name": step_name,
                "step_index": step_index,
                "additional_info": additional_info or {}
            })

        def log_data_operation(self, operation, data_info):
            self.entries.append({
                "timestamp": datetime.now(),
                "operation": operation,
                "data_info": data_info
            })

        def log_agent_execution(self, agent_name, status, execution_time, result):
            self.entries.append({
                "timestamp": datetime.now(),
                "agent_name": agent_name,
                "status": status,
                "execution_time": execution_time
            })

        def log_human_interaction(self, interaction_type, details):
            self.entries.append({
                "timestamp": datetime.now(),
                "type": "human_interaction",
                "interaction_type": interaction_type,
                "details": details
            })

        def get_audit_trail(self, limit=None):
            return self.entries[-limit:] if limit else self.entries

    class SimpleSampleLoader:
        def get_sample_data(self):
            np.random.seed(42)
            n_samples = 50
            data = {
                'customer_id': range(1, n_samples + 1),
                'age': np.random.randint(18, 80, n_samples),
                'income': np.random.normal(50000, 20000, n_samples),
                'credit_score': np.random.randint(300, 850, n_samples),
                'loan_amount': np.random.uniform(5000, 100000, n_samples),
                'employment_years': np.random.randint(0, 40, n_samples),
                'debt_to_income': np.random.uniform(0.1, 0.8, n_samples),
                'default': np.random.choice([0, 1], n_samples, p=[0.82, 0.18])
            }
            return pd.DataFrame(data)

        def get_sample_documents(self):
            return {
                "model_methodology.pdf": {"type": "methodology", "size": "2.3MB"},
                "validation_report.pdf": {"type": "validation", "size": "1.8MB"},
                "governance_policy.pdf": {"type": "governance", "size": "956KB"}
            }

    class SimpleAgent:
        def __init__(self, name):
            self.name = name

        def run(self, context):
            return {
                "agent": self.name,
                "status": "completed",
                "analysis": f"{self.name} analysis completed (fallback mode)",
                "execution_time": np.random.uniform(1, 5),
                "llm_enhanced": False
            }

    # Initialize fallback components
    config = SimpleConfig()
    audit_logger = SimpleAuditLogger()
    sample_loader = SimpleSampleLoader()
    
    agents = {
        'analyst': SimpleAgent("analyst"),
        'validator': SimpleAgent("validator"),
        'documentation': SimpleAgent("documentation"),
        'reviewer': SimpleAgent("reviewer"),
        'auditor': SimpleAgent("auditor")
    }
    
    workflow_engine = MCPWorkflowEngine(agents, audit_logger, {
        'mcp_config': {"enabled": True, "workflow": config.mcp_config},
        'workflow_config': config.workflow_config,
        'risk_thresholds': config.risk_thresholds.__dict__ if hasattr(config.risk_thresholds, '__dict__') else config.risk_thresholds
    })
    
    logger.info("Fallback components initialized")
    return workflow_engine, audit_logger, sample_loader, False

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    
    if 'validation_data' not in st.session_state:
        st.session_state.validation_data = None
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    
    if 'current_workflow_id' not in st.session_state:
        st.session_state.current_workflow_id = None
    
    if 'human_review_submitted' not in st.session_state:
        st.session_state.human_review_submitted = False
    
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = False

def configure_page():
    """Configure Streamlit page settings"""
    
    st.set_page_config(
        page_title="ValiCred-AI",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e1e5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)