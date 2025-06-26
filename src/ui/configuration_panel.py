"""
Enhanced Configuration Panel
===========================

Dynamic configuration management with real-time updates
"""

import streamlit as st
import json
import os
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path

def show_configuration():
    """Display enhanced configuration management interface"""
    st.title("⚙️ System Configuration")
    st.markdown("Manage all system settings dynamically - changes take effect immediately")
    
    # Try to load clean config system
    try:
        from src.core.clean_config_system import get_config, reload_config
        config = get_config()
        use_dynamic_config = True
    except Exception as e:
        st.warning(f"Using fallback configuration system: {e}")
        config = None
        use_dynamic_config = False
    
    # Configuration sections
    if use_dynamic_config:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Agent Settings", "Risk Thresholds", "Workflow", "Data Upload", "UI Settings", "Import/Export"
        ])
        
        with tab1:
            show_agent_configuration(config)
        
        with tab2:
            show_risk_thresholds(config)
        
        with tab3:
            show_workflow_configuration(config)
        
        with tab4:
            show_data_configuration(config)
        
        with tab5:
            show_ui_configuration(config)
        
        with tab6:
            show_import_export(config)
    else:
        # Fallback configuration interface
        show_fallback_configuration()

def show_agent_configuration(config):
    """Enhanced agent configuration"""
    st.subheader("🤖 Agent Configuration")
    
    # API Key Management
    st.markdown("### API Key Management")
    
    providers = ["groq", "openai", "anthropic"]
    
    for provider in providers:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            api_key_var = f"{provider.upper()}_API_KEY"
            current_key = os.getenv(api_key_var, "")
            
            if current_key:
                st.success(f"✅ {provider.title()} API key configured")
            else:
                st.warning(f"⚠️ {provider.title()} API key not found")
                
                new_api_key = st.text_input(
                    f"{provider.title()} API Key", 
                    type="password",
                    key=f"{provider}_key_input",
                    placeholder="Enter your API key..."
                )
                
                if st.button(f"Save {provider.title()} Key", key=f"save_{provider}"):
                    if new_api_key.strip():
                        os.environ[api_key_var] = new_api_key.strip()
                        st.success("API key saved!")
                        st.rerun()
                    else:
                        st.error("Please enter a valid API key")
        
        with col2:
            if current_key and st.button(f"Update {provider.title()}", key=f"update_{provider}"):
                if f"update_{provider}_key" not in st.session_state:
                    st.session_state[f"update_{provider}_key"] = True
                st.rerun()

def show_risk_thresholds(config):
    """Enhanced risk threshold configuration"""
    st.subheader("📊 Risk Assessment Thresholds")
    
    if config:
        current_thresholds = config.get_validation_thresholds()
    else:
        current_thresholds = {
            "auc_threshold": 0.65,
            "ks_threshold": 0.3,
            "psi_threshold": 0.25,
            "gini_threshold": 0.3
        }
    
    st.markdown("### Model Performance Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        auc_threshold = st.slider(
            "AUC Threshold", 
            0.5, 1.0, 
            current_thresholds.get("auc_threshold", 0.65), 
            0.01,
            help="Minimum Area Under Curve for model acceptance"
        )
        
        ks_threshold = st.slider(
            "KS Threshold", 
            0.1, 0.8, 
            current_thresholds.get("ks_threshold", 0.3), 
            0.01,
            help="Minimum Kolmogorov-Smirnov statistic"
        )
    
    with col2:
        psi_threshold = st.slider(
            "PSI Threshold", 
            0.1, 0.5, 
            current_thresholds.get("psi_threshold", 0.25), 
            0.01,
            help="Maximum Population Stability Index for data drift"
        )
        
        gini_threshold = st.slider(
            "Gini Threshold", 
            0.1, 0.8, 
            current_thresholds.get("gini_threshold", 0.3), 
            0.01,
            help="Minimum Gini coefficient"
        )
    
    if st.button("💾 Save Risk Thresholds", type="primary"):
        if config:
            # Update through config system
            success = True
            thresholds = {
                "auc": auc_threshold,
                "ks": ks_threshold, 
                "psi": psi_threshold,
                "gini": gini_threshold
            }
            
            for metric, value in thresholds.items():
                if not config.update_validation_threshold(metric, value):
                    success = False
            
            if success:
                st.success("Risk thresholds updated successfully!")
            else:
                st.warning("Some thresholds could not be updated")
        else:
            # Store in session state as fallback
            st.session_state.risk_thresholds = {
                "auc_threshold": auc_threshold,
                "ks_threshold": ks_threshold,
                "psi_threshold": psi_threshold,
                "gini_threshold": gini_threshold
            }
            st.success("Risk thresholds saved to session!")

def show_workflow_configuration(config):
    """Enhanced workflow configuration"""
    st.subheader("🔄 Workflow Configuration")
    
    # Get current workflow settings
    if config:
        current_workflow = config.workflow
        available_agents = list(config.agents.keys())
        # Filter out human_review from available agents as it's automatically inserted
        available_agents = [agent for agent in available_agents if agent != "human_review"]
        current_order = [agent for agent in current_workflow.execution_order if agent != "human_review"]
    else:
        available_agents = ["analyst", "validator", "documentation", "reviewer", "auditor"]
        current_order = available_agents.copy()
    
    # Execution order
    st.markdown("### Agent Execution Order")
    
    new_order = st.multiselect(
        "Select and order agents",
        available_agents,
        default=current_order,
        help="Choose agents and their execution order"
    )
    
    # Workflow settings
    col1, col2 = st.columns(2)
    
    with col1:
        parallel_execution = st.checkbox(
            "Enable Parallel Execution",
            value=getattr(config.workflow, 'parallel_execution', False) if config else False,
            help="Allow agents to run in parallel where possible"
        )
        
        enable_human_review = st.checkbox(
            "Enable Human Review",
            value=getattr(config.workflow, 'enable_human_review', True) if config else True,
            help="Include human review checkpoints"
        )
    
    with col2:
        timeout_seconds = st.number_input(
            "Workflow Timeout (seconds)",
            60, 3600,
            getattr(config.workflow, 'timeout_seconds', 300) if config else 300,
            help="Maximum time for entire workflow"
        )
        
        max_retries = st.number_input(
            "Max Retries",
            1, 10,
            getattr(config.workflow, 'max_retries', 3) if config else 3,
            help="Maximum retry attempts for failed steps"
        )
    
    if st.button("💾 Save Workflow Configuration", type="primary"):
        if config:
            # Update workflow configuration
            workflow_updates = {
                "execution_order": new_order,
                "parallel_execution": parallel_execution,
                "enable_human_review": enable_human_review,
                "timeout_seconds": timeout_seconds,
                "max_retries": max_retries
            }
            
            for key, value in workflow_updates.items():
                setattr(config.workflow, key, value)
            
            config._save_workflow_config()
            st.success("Workflow configuration updated!")
        else:
            st.session_state.workflow_config = {
                "execution_order": new_order,
                "parallel_execution": parallel_execution,
                "enable_human_review": enable_human_review,
                "timeout_seconds": timeout_seconds,
                "max_retries": max_retries
            }
            st.success("Workflow configuration saved to session!")

def show_data_configuration(config):
    """Data upload and processing configuration"""
    st.subheader("📁 Data Configuration")
    
    # Get current data config
    if config:
        current_data_config = config.get_data_config()
    else:
        current_data_config = {
            "max_file_size_mb": 200,
            "supported_formats": ["csv", "xlsx", "json"],
            "required_columns": [],
            "validation_rules": {"min_rows": 10, "max_rows": 100000}
        }
    
    # File upload settings
    st.markdown("### File Upload Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_file_size = st.number_input(
            "Max File Size (MB)",
            1, 1000,
            current_data_config.get("max_file_size_mb", 200),
            help="Maximum allowed file size"
        )
        
        supported_formats = st.multiselect(
            "Supported File Formats",
            ["csv", "xlsx", "json", "txt", "parquet"],
            default=current_data_config.get("supported_formats", ["csv", "xlsx", "json"]),
            help="Accepted file formats"
        )
    
    with col2:
        min_rows = st.number_input(
            "Minimum Rows",
            1, 10000,
            current_data_config.get("validation_rules", {}).get("min_rows", 10),
            help="Minimum required rows"
        )
        
        max_rows = st.number_input(
            "Maximum Rows",
            1000, 1000000,
            current_data_config.get("validation_rules", {}).get("max_rows", 100000),
            help="Maximum rows to process"
        )
    
    if st.button("💾 Save Data Configuration", type="primary"):
        data_updates = {
            "max_file_size_mb": max_file_size,
            "supported_formats": supported_formats,
            "validation_rules": {
                "min_rows": min_rows,
                "max_rows": max_rows
            }
        }
        
        if config:
            if config.update_data_config(data_updates):
                st.success("Data configuration updated!")
            else:
                st.error("Failed to update data configuration")
        else:
            st.session_state.data_config = data_updates
            st.success("Data configuration saved to session!")

def show_ui_configuration(config):
    """UI configuration settings"""
    st.subheader("🎨 UI Configuration")
    
    if config:
        current_ui = config.get_ui_settings()
    else:
        current_ui = {
            "theme": "light",
            "show_debug_info": False,
            "max_display_rows": 1000,
            "chart_height": 400
        }
    
    col1, col2 = st.columns(2)
    
    with col1:
        theme = st.selectbox(
            "Theme",
            ["light", "dark", "auto"],
            index=["light", "dark", "auto"].index(current_ui.get("theme", "light"))
        )
        
        show_debug = st.checkbox(
            "Show Debug Information",
            value=current_ui.get("show_debug_info", False)
        )
    
    with col2:
        max_display_rows = st.number_input(
            "Max Display Rows",
            100, 10000,
            current_ui.get("max_display_rows", 1000)
        )
        
        chart_height = st.number_input(
            "Chart Height (px)",
            200, 800,
            current_ui.get("chart_height", 400)
        )
    
    if st.button("💾 Save UI Configuration", type="primary"):
        ui_updates = {
            "theme": theme,
            "show_debug_info": show_debug,
            "max_display_rows": max_display_rows,
            "chart_height": chart_height
        }
        
        if config:
            success = True
            for setting, value in ui_updates.items():
                if not config.update_ui_setting(setting, value):
                    success = False
            
            if success:
                st.success("UI configuration updated!")
            else:
                st.error("Failed to update some UI settings")
        else:
            st.session_state.ui_config = ui_updates
            st.success("UI configuration saved to session!")

def show_import_export(config):
    """Configuration import/export functionality"""
    st.subheader("📦 Import/Export Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Export Configuration")
        
        if st.button("📥 Export All Settings", type="primary"):
            if config:
                config_json = config.export_config()
                
                st.download_button(
                    "💾 Download Configuration",
                    data=config_json,
                    file_name=f"valicred_config_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                # Export session state configuration
                session_config = {
                    "risk_thresholds": st.session_state.get("risk_thresholds", {}),
                    "workflow_config": st.session_state.get("workflow_config", {}),
                    "data_config": st.session_state.get("data_config", {}),
                    "ui_config": st.session_state.get("ui_config", {})
                }
                
                config_json = json.dumps(session_config, indent=2)
                
                st.download_button(
                    "💾 Download Configuration",
                    data=config_json,
                    file_name=f"valicred_config_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            st.success("Configuration ready for download!")
    
    with col2:
        st.markdown("### Import Configuration")
        
        uploaded_config = st.file_uploader(
            "Choose configuration file",
            type=['json'],
            help="Upload a previously exported configuration file"
        )
        
        if uploaded_config is not None:
            try:
                config_data = json.load(uploaded_config)
                
                if st.button("📤 Import Configuration"):
                    if config:
                        if config.import_config(config_data):
                            st.success("Configuration imported successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to import configuration")
                    else:
                        # Import to session state
                        for key, value in config_data.items():
                            st.session_state[key] = value
                        st.success("Configuration imported to session!")
                        st.rerun()
                
            except json.JSONDecodeError:
                st.error("Invalid JSON file")
    
    # Reset to defaults
    st.markdown("### Reset Configuration")
    st.warning("⚠️ This will reset all settings to default values")
    
    if st.button("🔄 Reset to Defaults", type="secondary"):
        if config:
            if config.reset_to_defaults():
                st.success("Configuration reset to defaults!")
                st.rerun()
            else:
                st.error("Failed to reset configuration")
        else:
            # Clear session state
            for key in ["risk_thresholds", "workflow_config", "data_config", "ui_config"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Session configuration cleared!")
            st.rerun()

def show_fallback_configuration():
    """Fallback configuration interface when dynamic config is not available"""
    st.warning("Using basic configuration interface. Dynamic configuration system not available.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Risk Thresholds**")
        auc_threshold = st.slider("AUC Threshold", 0.5, 1.0, 0.65, 0.01)
        ks_threshold = st.slider("KS Threshold", 0.1, 0.8, 0.3, 0.01)
        psi_threshold = st.slider("PSI Threshold", 0.1, 0.5, 0.25, 0.01)
    
    with col2:
        st.markdown("**Workflow Settings**")
        enable_human_review = st.checkbox("Enable Human Review", value=True)
        timeout_seconds = st.number_input("Timeout (seconds)", 60, 1800, 300)
        max_retries = st.number_input("Max Retries", 1, 10, 3)
    
    if st.button("Save Configuration"):
        # Save to session state
        st.session_state.basic_config = {
            "auc_threshold": auc_threshold,
            "ks_threshold": ks_threshold,
            "psi_threshold": psi_threshold,
            "enable_human_review": enable_human_review,
            "timeout_seconds": timeout_seconds,
            "max_retries": max_retries
        }
        st.success("Configuration saved to session!")
    
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