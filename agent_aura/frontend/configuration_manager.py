"""
Interactive UI Configuration Manager
Provides dynamic configuration management through Streamlit interface
"""
import streamlit as st
import json
from typing import Dict, Any, List
import plotly.express as px
import pandas as pd

from config.system_config import config

class UIConfigurationManager:
    """Manages UI configuration through interactive interface"""
    
    def __init__(self):
        self.config = config
    
    def render_configuration_panel(self):
        """Render the main configuration panel"""
        st.title("System Configuration")
        st.write("Configure system parameters, thresholds, and workflow settings")
        
        # Create tabs for different configuration sections
        tabs = st.tabs([
            "Risk Thresholds", 
            "Validation Parameters", 
            "Workflow Settings", 
            "Agent Configuration"
        ])
        
        with tabs[0]:
            self._render_risk_thresholds_config()
        
        with tabs[1]:
            self._render_validation_parameters_config()
        
        with tabs[2]:
            self._render_workflow_config()
        
        with tabs[3]:
            self._render_agent_config()
    
    def _render_risk_thresholds_config(self):
        """Render risk thresholds configuration"""
        st.subheader("Risk Assessment Thresholds")
        st.write("Configure thresholds for model validation metrics")
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**AUC (Area Under Curve)**")
            auc_config = self.config.risk_thresholds.get("auc", {})
            
            auc_excellent = st.slider(
                "Excellent", 0.5, 1.0, 
                float(auc_config.get("excellent", 0.8)), 0.01,
                key="auc_excellent"
            )
            auc_good = st.slider(
                "Good", 0.5, 1.0, 
                float(auc_config.get("good", 0.7)), 0.01,
                key="auc_good"
            )
            auc_acceptable = st.slider(
                "Acceptable", 0.5, 1.0, 
                float(auc_config.get("acceptable", 0.6)), 0.01,
                key="auc_acceptable"
            )
        
        with col2:
            st.write("**KS Statistic**")
            ks_config = self.config.risk_thresholds.get("ks", {})
            
            ks_excellent = st.slider(
                "Excellent", 0.0, 0.5, 
                float(ks_config.get("excellent", 0.3)), 0.01,
                key="ks_excellent"
            )
            ks_good = st.slider(
                "Good", 0.0, 0.5, 
                float(ks_config.get("good", 0.2)), 0.01,
                key="ks_good"
            )
            ks_acceptable = st.slider(
                "Acceptable", 0.0, 0.5, 
                float(ks_config.get("acceptable", 0.15)), 0.01,
                key="ks_acceptable"
            )
        
        # Update button
        if st.button("Update Risk Thresholds", key="update_risk_thresholds"):
            updated_thresholds = {
                "auc": {
                    "excellent": auc_excellent,
                    "good": auc_good,
                    "acceptable": auc_acceptable,
                    "minimum": 0.5
                },
                "ks": {
                    "excellent": ks_excellent,
                    "good": ks_good,
                    "acceptable": ks_acceptable,
                    "minimum": 0.1
                }
            }
            
            if self.config.update_config("risk_thresholds", updated_thresholds):
                st.success("Risk thresholds updated successfully!")
            else:
                st.error("Failed to update risk thresholds")
    
    def _render_validation_parameters_config(self):
        """Render validation parameters configuration"""
        st.subheader("Validation Parameters")
        
        val_params = self.config.validation_parameters
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Cross Validation Settings**")
            cv_folds = st.number_input(
                "Number of Folds", 3, 20, 
                val_params.get("cross_validation", {}).get("folds", 5),
                key="cv_folds"
            )
            cv_stratify = st.checkbox(
                "Stratify", 
                val_params.get("cross_validation", {}).get("stratify", True),
                key="cv_stratify"
            )
        
        with col2:
            st.write("**Model Training Settings**")
            test_size = st.slider(
                "Test Size", 0.1, 0.5, 
                float(val_params.get("model_training", {}).get("test_size", 0.2)), 0.01,
                key="test_size"
            )
            validation_size = st.slider(
                "Validation Size", 0.05, 0.3, 
                float(val_params.get("model_training", {}).get("validation_size", 0.1)), 0.01,
                key="validation_size"
            )
        
        if st.button("Update Validation Parameters", key="update_val_params"):
            updated_params = {
                "cross_validation": {
                    "folds": cv_folds,
                    "stratify": cv_stratify,
                    "random_state": 42
                },
                "model_training": {
                    "test_size": test_size,
                    "validation_size": validation_size,
                    "random_state": 42
                }
            }
            
            if self.config.update_config("validation_parameters", updated_params):
                st.success("Validation parameters updated successfully!")
            else:
                st.error("Failed to update validation parameters")
    
    def _render_workflow_config(self):
        """Render workflow configuration"""
        st.subheader("Workflow Configuration")
        
        workflow_config = self.config.workflow_config
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Execution Settings**")
            step_timeout = st.number_input(
                "Step Timeout (seconds)", 60, 3600, 
                workflow_config.get("execution_settings", {}).get("step_timeout", 600),
                key="step_timeout"
            )
            max_retries = st.number_input(
                "Max Retries", 0, 10, 
                workflow_config.get("retry_policy", {}).get("max_retries", 3),
                key="max_retries"
            )
        
        with col2:
            st.write("**Human Review Settings**")
            review_timeout = st.number_input(
                "Review Timeout (seconds)", 300, 7200, 
                workflow_config.get("human_review", {}).get("review_timeout", 3600),
                key="review_timeout"
            )
            auto_approve = st.checkbox(
                "Auto Approve (Development)", 
                workflow_config.get("human_review", {}).get("auto_approve", False),
                key="auto_approve"
            )
        
        if st.button("Update Workflow Configuration", key="update_workflow_config"):
            updated_config = {
                "execution_settings": {
                    "max_concurrent_agents": 1,
                    "step_timeout": step_timeout,
                    "total_timeout": 3600,
                    "enable_checkpoints": True
                },
                "retry_policy": {
                    "max_retries": max_retries,
                    "retry_delay": 5,
                    "exponential_backoff": True
                },
                "human_review": {
                    "mandatory_steps": ["human_review"],
                    "optional_reviews": ["validator", "reviewer"],
                    "review_timeout": review_timeout,
                    "auto_approve": auto_approve
                }
            }
            
            if self.config.update_config("workflow_config", updated_config):
                st.success("Workflow configuration updated successfully!")
            else:
                st.error("Failed to update workflow configuration")
    
    def _render_agent_config(self):
        """Render agent configuration"""
        st.subheader("Agent Configuration")
        
        mcp_config = self.config.mcp_config
        agents_config = mcp_config.get("agents", {})
        
        for agent_name, agent_config in agents_config.items():
            with st.expander(f"{agent_name.title()} Agent"):
                col1, col2 = st.columns(2)
                
                with col1:
                    timeout = st.number_input(
                        "Timeout (seconds)", 60, 1800, 
                        agent_config.get("timeout", 300),
                        key=f"{agent_name}_timeout"
                    )
                    retry_attempts = st.number_input(
                        "Retry Attempts", 0, 10, 
                        agent_config.get("retry_attempts", 3),
                        key=f"{agent_name}_retry"
                    )
                
                with col2:
                    st.write("**Description**")
                    st.write(agent_config.get("description", "No description available"))
                    
                    st.write("**Available Tools**")
                    tools = agent_config.get("tools", [])
                    for tool in tools:
                        st.write(f"• {tool}")

# Global UI configuration manager instance
ui_config_manager = UIConfigurationManager()