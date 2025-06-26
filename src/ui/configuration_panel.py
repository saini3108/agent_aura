"""
Configuration Panel
==================

System configuration and settings management
"""

import streamlit as st
import json
from typing import Dict, Any

def show_configuration():
    """Configuration management interface"""
    st.title("⚙️ Configuration")
    
    # System Configuration
    st.subheader("System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Risk Thresholds**")
        
        # AUC Thresholds
        auc_excellent = st.number_input("AUC Excellent", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
        auc_good = st.number_input("AUC Good", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
        auc_acceptable = st.number_input("AUC Acceptable", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
        
        # KS Thresholds
        ks_excellent = st.number_input("KS Excellent", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        ks_good = st.number_input("KS Good", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        ks_acceptable = st.number_input("KS Acceptable", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
    
    with col2:
        st.markdown("**Agent Settings**")
        
        # Timeout Settings
        agent_timeout = st.number_input("Agent Timeout (seconds)", min_value=60, max_value=1800, value=300, step=30)
        retry_attempts = st.number_input("Retry Attempts", min_value=1, max_value=10, value=3, step=1)
        
        # Workflow Settings
        enable_human_review = st.checkbox("Enable Human Review Checkpoints", value=True)
        auto_approve_low_risk = st.checkbox("Auto-approve Low Risk Models", value=False)
        
        # Compliance Frameworks
        st.markdown("**Compliance Frameworks**")
        basel_iii = st.checkbox("Basel III", value=True)
        ifrs_9 = st.checkbox("IFRS 9", value=True)
        model_risk_mgmt = st.checkbox("Model Risk Management", value=True)
    
    # Advanced Configuration
    st.subheader("Advanced Settings")
    
    with st.expander("LLM Configuration"):
        st.markdown("**Large Language Model Settings**")
        
        # LLM Provider Selection
        llm_provider = st.selectbox(
            "LLM Provider",
            ["None", "Groq", "OpenAI", "Anthropic"],
            index=1,
            help="Select the LLM provider for enhanced report generation"
        )
        
        # Provider-specific settings
        if llm_provider == "Groq":
            groq_model = st.selectbox(
                "Groq Model",
                ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
                index=0
            )
            groq_api_key = st.text_input("Groq API Key", type="password", placeholder="Enter your Groq API key")
            if groq_api_key:
                st.success("Groq API key configured")
            else:
                st.warning("Groq API key required for LLM-enhanced reports")
        
        elif llm_provider == "OpenAI":
            openai_model = st.selectbox(
                "OpenAI Model",
                ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                index=0
            )
            openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="Enter your OpenAI API key")
            if openai_api_key:
                st.success("OpenAI API key configured")
            else:
                st.warning("OpenAI API key required for LLM-enhanced reports")
        
        elif llm_provider == "Anthropic":
            anthropic_model = st.selectbox(
                "Anthropic Model",
                ["claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-3-opus-20240229"],
                index=0
            )
            anthropic_api_key = st.text_input("Anthropic API Key", type="password", placeholder="Enter your Anthropic API key")
            if anthropic_api_key:
                st.success("Anthropic API key configured")
            else:
                st.warning("Anthropic API key required for LLM-enhanced reports")
        
        # LLM Enhancement Settings
        st.markdown("**Enhancement Options**")
        enable_llm_reports = st.checkbox("Enable LLM-Enhanced Reports", value=True)
        llm_temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        max_tokens = st.number_input("Max Tokens", min_value=1000, max_value=8000, value=4000, step=500)
        
    with st.expander("MCP Configuration"):
        st.markdown("**Model Context Protocol Settings**")
        mcp_enabled = st.checkbox("Enable MCP Protocol", value=True)
        mcp_timeout = st.number_input("MCP Timeout (seconds)", min_value=30, max_value=600, value=120)
        
        st.markdown("**LangGraph Settings**")
        langgraph_enabled = st.checkbox("Enable LangGraph Workflows", value=True)
        parallel_execution = st.checkbox("Enable Parallel Agent Execution", value=False)
    
    with st.expander("Data Processing"):
        st.markdown("**Data Quality Thresholds**")
        missing_value_threshold = st.number_input("Missing Value Threshold (%)", min_value=0, max_value=100, value=20)
        outlier_threshold = st.number_input("Outlier Detection Threshold (Z-score)", min_value=1.0, max_value=5.0, value=3.0)
        min_sample_size = st.number_input("Minimum Sample Size", min_value=100, max_value=10000, value=1000)
    
    with st.expander("Reporting Configuration"):
        st.markdown("**Report Generation Settings**")
        include_charts = st.checkbox("Include Performance Charts", value=True)
        detailed_metrics = st.checkbox("Include Detailed Metrics", value=True)
        executive_summary = st.checkbox("Generate Executive Summary", value=True)
        
        report_format = st.selectbox("Default Report Format", ["Markdown", "HTML", "PDF"])
        auto_timestamp = st.checkbox("Auto-timestamp Reports", value=True)
    
    # Save Configuration
    if st.button("Save Configuration", type="primary"):
        config = {
            "risk_thresholds": {
                "auc": {"excellent": auc_excellent, "good": auc_good, "acceptable": auc_acceptable},
                "ks": {"excellent": ks_excellent, "good": ks_good, "acceptable": ks_acceptable}
            },
            "agent_settings": {
                "timeout": agent_timeout,
                "retry_attempts": retry_attempts,
                "enable_human_review": enable_human_review,
                "auto_approve_low_risk": auto_approve_low_risk
            },
            "compliance_frameworks": {
                "basel_iii": basel_iii,
                "ifrs_9": ifrs_9,
                "model_risk_management": model_risk_mgmt
            },
            "llm_config": {
                "provider": llm_provider,
                "model": locals().get(f"{llm_provider.lower()}_model", ""),
                "api_key": locals().get(f"{llm_provider.lower()}_api_key", ""),
                "enable_enhanced_reports": enable_llm_reports,
                "temperature": llm_temperature,
                "max_tokens": max_tokens
            },
            "mcp_config": {
                "enabled": mcp_enabled,
                "timeout": mcp_timeout,
                "langgraph_enabled": langgraph_enabled,
                "parallel_execution": parallel_execution
            },
            "data_processing": {
                "missing_value_threshold": missing_value_threshold,
                "outlier_threshold": outlier_threshold,
                "min_sample_size": min_sample_size
            },
            "reporting": {
                "include_charts": include_charts,
                "detailed_metrics": detailed_metrics,
                "executive_summary": executive_summary,
                "format": report_format,
                "auto_timestamp": auto_timestamp
            }
        }
        
        # Store in session state
        st.session_state.system_config = config
        st.success("Configuration saved successfully!")
        
        # Show saved configuration
        with st.expander("View Saved Configuration"):
            st.json(config)
    
    # Load Configuration
    if st.button("Load Default Configuration"):
        default_config = get_default_configuration()
        st.session_state.system_config = default_config
        st.success("Default configuration loaded!")
        st.rerun()
    
    # Current Configuration Display
    if 'system_config' in st.session_state:
        st.subheader("Current Configuration")
        with st.expander("View Current Settings"):
            st.json(st.session_state.system_config)

def get_default_configuration() -> Dict[str, Any]:
    """Get default system configuration"""
    return {
        "risk_thresholds": {
            "auc": {"excellent": 0.8, "good": 0.7, "acceptable": 0.6},
            "ks": {"excellent": 0.3, "good": 0.2, "acceptable": 0.15}
        },
        "agent_settings": {
            "timeout": 300,
            "retry_attempts": 3,
            "enable_human_review": True,
            "auto_approve_low_risk": False
        },
        "compliance_frameworks": {
            "basel_iii": True,
            "ifrs_9": True,
            "model_risk_management": True
        },
        "mcp_config": {
            "enabled": True,
            "timeout": 120,
            "langgraph_enabled": True,
            "parallel_execution": False
        },
        "data_processing": {
            "missing_value_threshold": 20,
            "outlier_threshold": 3.0,
            "min_sample_size": 1000
        },
        "reporting": {
            "include_charts": True,
            "detailed_metrics": True,
            "executive_summary": True,
            "format": "Markdown",
            "auto_timestamp": True
        }
    }