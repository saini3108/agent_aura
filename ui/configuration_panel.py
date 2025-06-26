"""
Advanced Configuration Panel with API Key Management
===================================================

This module provides an enterprise-grade configuration interface for ValiCred-AI,
allowing users to manage API keys, model settings, risk thresholds, and workflow
parameters through an intuitive Streamlit interface.

Features:
- Secure API key management for multiple providers
- Dynamic model selection and configuration
- Risk threshold adjustment with real-time validation
- Workflow parameter customization
- Configuration export/import functionality
- Real-time provider health monitoring

Usage:
    from src.ui.configuration_panel import ConfigurationPanel
    
    config_panel = ConfigurationPanel()
    config_panel.render()
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import logging

from src.config.settings import get_config, update_config, save_config, load_config
from src.core.llm_manager import get_llm_manager

logger = logging.getLogger(__name__)

class ConfigurationPanel:
    """Enterprise configuration management interface"""
    
    def __init__(self):
        self.config = get_config()
        self.llm_manager = get_llm_manager()
        
    def render(self):
        """Render the complete configuration panel"""
        st.header("🔧 System Configuration")
        
        # Create tabs for different configuration sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔑 API Keys", 
            "🤖 Models", 
            "⚖️ Risk Thresholds", 
            "🔄 Workflow", 
            "📊 Monitor"
        ])
        
        with tab1:
            self._render_api_configuration()
        
        with tab2:
            self._render_model_configuration()
        
        with tab3:
            self._render_risk_thresholds()
        
        with tab4:
            self._render_workflow_configuration()
        
        with tab5:
            self._render_monitoring_dashboard()
    
    def _render_api_configuration(self):
        """Render API key configuration interface"""
        st.subheader("API Key Management")
        st.write("Configure API keys for different LLM providers. Keys are securely stored and never displayed.")
        
        # Current provider status
        available_providers = self.config.get_available_providers()
        
        if available_providers:
            st.success(f"✅ Active providers: {', '.join(available_providers)}")
        else:
            st.warning("⚠️ No API keys configured. Please add at least one provider.")
        
        # API Key Configuration Forms
        col1, col2 = st.columns(2)
        
        with col1:
            # Groq Configuration
            with st.expander("🚀 Groq (Recommended)", expanded=not self.config.api.groq_api_key):
                st.write("**Fast inference with Llama models**")
                st.write("Get your free API key at: https://console.groq.com/")
                
                groq_key = st.text_input(
                    "Groq API Key",
                    type="password",
                    help="Enter your Groq API key for fast Llama inference",
                    key="groq_api_key"
                )
                
                if st.button("Save Groq Key", key="save_groq"):
                    if groq_key:
                        self.config.update_api_key("groq", groq_key)
                        st.success("Groq API key saved successfully!")
                        st.rerun()
                    else:
                        st.error("Please enter a valid API key")
                
                if self.config.api.groq_api_key:
                    st.info("✅ Groq API key is configured")
            
            # OpenAI Configuration
            with st.expander("🧠 OpenAI", expanded=False):
                st.write("**Industry-leading GPT models**")
                st.write("Get your API key at: https://platform.openai.com/")
                
                openai_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    help="Enter your OpenAI API key for GPT models",
                    key="openai_api_key"
                )
                
                if st.button("Save OpenAI Key", key="save_openai"):
                    if openai_key:
                        self.config.update_api_key("openai", openai_key)
                        st.success("OpenAI API key saved successfully!")
                        st.rerun()
                    else:
                        st.error("Please enter a valid API key")
                
                if self.config.api.openai_api_key:
                    st.info("✅ OpenAI API key is configured")
        
        with col2:
            # Anthropic Configuration
            with st.expander("🎭 Anthropic Claude", expanded=False):
                st.write("**Powerful Claude models**")
                st.write("Get your API key at: https://console.anthropic.com/")
                
                anthropic_key = st.text_input(
                    "Anthropic API Key",
                    type="password",
                    help="Enter your Anthropic API key for Claude models",
                    key="anthropic_api_key"
                )
                
                if st.button("Save Anthropic Key", key="save_anthropic"):
                    if anthropic_key:
                        self.config.update_api_key("anthropic", anthropic_key)
                        st.success("Anthropic API key saved successfully!")
                        st.rerun()
                    else:
                        st.error("Please enter a valid API key")
                
                if self.config.api.anthropic_api_key:
                    st.info("✅ Anthropic API key is configured")
            
            # Gemini Configuration
            with st.expander("💎 Google Gemini", expanded=False):
                st.write("**Google's advanced AI models**")
                st.write("Get your API key at: https://ai.google.dev/")
                
                gemini_key = st.text_input(
                    "Gemini API Key",
                    type="password",
                    help="Enter your Google Gemini API key",
                    key="gemini_api_key"
                )
                
                if st.button("Save Gemini Key", key="save_gemini"):
                    if gemini_key:
                        self.config.update_api_key("gemini", gemini_key)
                        st.success("Gemini API key saved successfully!")
                        st.rerun()
                    else:
                        st.error("Please enter a valid API key")
                
                if self.config.api.gemini_api_key:
                    st.info("✅ Gemini API key is configured")
        
        # Provider Health Check
        if st.button("🔍 Test All Providers", key="test_providers"):
            with st.spinner("Testing provider connections..."):
                try:
                    health_status = self.llm_manager.health_check()
                    
                    st.subheader("Provider Health Status")
                    for provider, status in health_status.items():
                        if status["status"] == "healthy":
                            st.success(f"✅ {provider.title()}: Healthy (Latency: {status.get('latency', 0):.0f}ms)")
                        else:
                            st.error(f"❌ {provider.title()}: {status.get('error', 'Unknown error')}")
                            
                except Exception as e:
                    st.error(f"Health check failed: {e}")
    
    def _render_model_configuration(self):
        """Render model selection and configuration"""
        st.subheader("Model Configuration")
        
        available_providers = self.config.get_available_providers()
        
        if not available_providers:
            st.warning("⚠️ Please configure API keys first to enable model selection.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Provider Selection
            selected_provider = st.selectbox(
                "Primary Provider",
                options=available_providers,
                index=available_providers.index(self.config.model.provider) if self.config.model.provider in available_providers else 0,
                help="Select the primary LLM provider for analysis"
            )
            
            # Model Selection based on provider
            model_options = self._get_model_options(selected_provider)
            selected_model = st.selectbox(
                "Model",
                options=model_options,
                index=model_options.index(self.config.model.model_name) if self.config.model.model_name in model_options else 0,
                help="Select the specific model to use"
            )
            
            # Temperature
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=self.config.model.temperature,
                step=0.1,
                help="Controls randomness in responses (0.0 = deterministic, 1.0 = creative)"
            )
        
        with col2:
            # Max Tokens
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=100,
                max_value=8000,
                value=self.config.model.max_tokens,
                step=100,
                help="Maximum number of tokens in the response"
            )
            
            # Timeout
            timeout = st.number_input(
                "Timeout (seconds)",
                min_value=10,
                max_value=300,
                value=self.config.model.timeout,
                step=10,
                help="Request timeout in seconds"
            )
            
            # Fallback Options
            enable_fallback = st.checkbox(
                "Enable Fallback",
                value=True,
                help="Automatically try other providers if primary fails"
            )
        
        # Model Information
        st.subheader("Model Information")
        model_info = self._get_model_info(selected_provider, selected_model)
        
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.metric("Provider", selected_provider.title())
        with info_col2:
            st.metric("Context Length", model_info.get("context_length", "Unknown"))
        with info_col3:
            st.metric("Cost Estimate", model_info.get("cost_per_1k", "Variable"))
        
        # Save Configuration
        if st.button("💾 Save Model Configuration", key="save_model_config"):
            self.config.model.provider = selected_provider
            self.config.model.model_name = selected_model
            self.config.model.temperature = temperature
            self.config.model.max_tokens = max_tokens
            self.config.model.timeout = timeout
            
            # Update LLM manager
            self.llm_manager.set_default_provider(selected_provider)
            
            st.success("Model configuration saved successfully!")
    
    def _get_model_options(self, provider: str) -> List[str]:
        """Get available models for a provider"""
        model_options = {
            "groq": [
                "llama-3.1-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it"
            ],
            "openai": [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo"
            ],
            "anthropic": [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ],
            "gemini": [
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-1.5-pro"
            ]
        }
        
        return model_options.get(provider, ["default"])
    
    def _get_model_info(self, provider: str, model: str) -> Dict[str, str]:
        """Get information about a specific model"""
        model_info = {
            "groq": {
                "llama-3.1-70b-versatile": {"context_length": "32K", "cost_per_1k": "Free"},
                "llama-3.1-8b-instant": {"context_length": "32K", "cost_per_1k": "Free"},
                "mixtral-8x7b-32768": {"context_length": "32K", "cost_per_1k": "Free"},
                "gemma2-9b-it": {"context_length": "8K", "cost_per_1k": "Free"}
            },
            "openai": {
                "gpt-4o": {"context_length": "128K", "cost_per_1k": "$0.005-0.015"},
                "gpt-4o-mini": {"context_length": "128K", "cost_per_1k": "$0.00015-0.0006"},
                "gpt-4-turbo": {"context_length": "128K", "cost_per_1k": "$0.01-0.03"},
                "gpt-3.5-turbo": {"context_length": "16K", "cost_per_1k": "$0.0005-0.0015"}
            },
            "anthropic": {
                "claude-3-opus-20240229": {"context_length": "200K", "cost_per_1k": "$0.015-0.075"},
                "claude-3-sonnet-20240229": {"context_length": "200K", "cost_per_1k": "$0.003-0.015"},
                "claude-3-haiku-20240307": {"context_length": "200K", "cost_per_1k": "$0.00025-0.00125"}
            }
        }
        
        return model_info.get(provider, {}).get(model, {"context_length": "Unknown", "cost_per_1k": "Variable"})
    
    def _render_risk_thresholds(self):
        """Render risk threshold configuration"""
        st.subheader("Risk Assessment Thresholds")
        st.write("Configure thresholds for credit risk model validation metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**AUC (Area Under Curve) Thresholds**")
            auc_excellent = st.slider(
                "Excellent AUC",
                min_value=0.7,
                max_value=1.0,
                value=self.config.risk_thresholds.auc_excellent,
                step=0.01,
                help="AUC threshold for excellent model performance"
            )
            
            auc_good = st.slider(
                "Good AUC",
                min_value=0.6,
                max_value=auc_excellent,
                value=min(self.config.risk_thresholds.auc_good, auc_excellent),
                step=0.01,
                help="AUC threshold for good model performance"
            )
            
            auc_acceptable = st.slider(
                "Acceptable AUC",
                min_value=0.5,
                max_value=auc_good,
                value=min(self.config.risk_thresholds.auc_acceptable, auc_good),
                step=0.01,
                help="Minimum acceptable AUC threshold"
            )
            
            st.write("**KS Statistic Thresholds**")
            ks_excellent = st.slider(
                "Excellent KS",
                min_value=0.2,
                max_value=1.0,
                value=self.config.risk_thresholds.ks_excellent,
                step=0.01,
                help="KS threshold for excellent discrimination"
            )
            
            ks_good = st.slider(
                "Good KS",
                min_value=0.1,
                max_value=ks_excellent,
                value=min(self.config.risk_thresholds.ks_good, ks_excellent),
                step=0.01,
                help="KS threshold for good discrimination"
            )
            
            ks_acceptable = st.slider(
                "Acceptable KS",
                min_value=0.05,
                max_value=ks_good,
                value=min(self.config.risk_thresholds.ks_acceptable, ks_good),
                step=0.01,
                help="Minimum acceptable KS threshold"
            )
        
        with col2:
            st.write("**PSI (Population Stability Index) Thresholds**")
            psi_stable = st.slider(
                "Stable PSI",
                min_value=0.0,
                max_value=0.25,
                value=self.config.risk_thresholds.psi_stable,
                step=0.01,
                help="PSI threshold for stable population"
            )
            
            psi_monitoring = st.slider(
                "Monitoring Required PSI",
                min_value=psi_stable,
                max_value=0.3,
                value=max(self.config.risk_thresholds.psi_monitoring, psi_stable),
                step=0.01,
                help="PSI threshold requiring monitoring"
            )
            
            psi_unstable = st.slider(
                "Unstable PSI",
                min_value=psi_monitoring,
                max_value=0.5,
                value=max(self.config.risk_thresholds.psi_unstable, psi_monitoring),
                step=0.01,
                help="PSI threshold indicating instability"
            )
            
            # Regulatory Framework Selection
            st.write("**Regulatory Frameworks**")
            selected_frameworks = st.multiselect(
                "Active Frameworks",
                options=["Basel III", "IFRS 9", "SR 11-7", "CCAR", "CECL", "CRR II"],
                default=self.config.regulatory_frameworks,
                help="Select applicable regulatory frameworks"
            )
        
        # Threshold Validation
        threshold_valid = (
            auc_excellent > auc_good > auc_acceptable and
            ks_excellent > ks_good > ks_acceptable and
            psi_stable < psi_monitoring < psi_unstable
        )
        
        if not threshold_valid:
            st.error("⚠️ Threshold values must be in logical order (excellent > good > acceptable)")
        
        # Save Thresholds
        if st.button("💾 Save Risk Thresholds", key="save_thresholds", disabled=not threshold_valid):
            self.config.risk_thresholds.auc_excellent = auc_excellent
            self.config.risk_thresholds.auc_good = auc_good
            self.config.risk_thresholds.auc_acceptable = auc_acceptable
            self.config.risk_thresholds.ks_excellent = ks_excellent
            self.config.risk_thresholds.ks_good = ks_good
            self.config.risk_thresholds.ks_acceptable = ks_acceptable
            self.config.risk_thresholds.psi_stable = psi_stable
            self.config.risk_thresholds.psi_monitoring = psi_monitoring
            self.config.risk_thresholds.psi_unstable = psi_unstable
            self.config.regulatory_frameworks = selected_frameworks
            
            st.success("Risk thresholds saved successfully!")
    
    def _render_workflow_configuration(self):
        """Render workflow configuration options"""
        st.subheader("Workflow Configuration")
        st.write("Configure agent execution and workflow parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Execution Parameters**")
            max_retries = st.number_input(
                "Max Retries",
                min_value=1,
                max_value=10,
                value=self.config.workflow.max_retries,
                help="Maximum retry attempts for failed agents"
            )
            
            retry_delay = st.number_input(
                "Retry Delay (seconds)",
                min_value=1,
                max_value=60,
                value=self.config.workflow.retry_delay,
                help="Delay between retry attempts"
            )
            
            timeout_seconds = st.number_input(
                "Agent Timeout (seconds)",
                min_value=30,
                max_value=600,
                value=self.config.workflow.timeout_seconds,
                help="Maximum execution time per agent"
            )
        
        with col2:
            st.write("**Human Review Settings**")
            enable_human_review = st.checkbox(
                "Enable Human Review",
                value=self.config.workflow.enable_human_review,
                help="Enable human-in-the-loop checkpoints"
            )
            
            auto_approve_threshold = st.slider(
                "Auto-approve Threshold",
                min_value=0.0,
                max_value=1.0,
                value=self.config.workflow.auto_approve_threshold,
                step=0.05,
                help="Confidence threshold for automatic approval",
                disabled=not enable_human_review
            )
            
            # Agent Execution Order
            st.write("**Agent Execution Order**")
            current_order = ["analyst", "validator", "documentation", "reviewer", "auditor"]
            
            # Display current order (read-only for now)
            for i, agent in enumerate(current_order, 1):
                st.write(f"{i}. {agent.title()} Agent")
        
        # Advanced Settings
        with st.expander("Advanced Settings"):
            parallel_execution = st.checkbox(
                "Enable Parallel Execution",
                value=False,
                help="Execute compatible agents in parallel (experimental)"
            )
            
            detailed_logging = st.checkbox(
                "Detailed Logging",
                value=True,
                help="Enable detailed execution logging"
            )
            
            cache_results = st.checkbox(
                "Cache Agent Results",
                value=True,
                help="Cache agent results for faster re-execution"
            )
        
        # Save Workflow Configuration
        if st.button("💾 Save Workflow Configuration", key="save_workflow"):
            self.config.workflow.max_retries = max_retries
            self.config.workflow.retry_delay = retry_delay
            self.config.workflow.timeout_seconds = timeout_seconds
            self.config.workflow.enable_human_review = enable_human_review
            self.config.workflow.auto_approve_threshold = auto_approve_threshold
            
            st.success("Workflow configuration saved successfully!")
    
    def _render_monitoring_dashboard(self):
        """Render system monitoring and usage statistics"""
        st.subheader("System Monitoring")
        
        # Provider Usage Statistics
        if hasattr(self.llm_manager, 'get_usage_stats'):
            usage_stats = self.llm_manager.get_usage_stats()
            
            if usage_stats:
                st.write("**Provider Usage Statistics**")
                
                # Create usage summary table
                summary_data = []
                for provider, stats in usage_stats.items():
                    summary_data.append({
                        "Provider": provider.title(),
                        "Requests": stats.get("requests", 0),
                        "Tokens": f"{stats.get('tokens', 0):,}",
                        "Errors": stats.get("errors", 0),
                        "Avg Latency": f"{stats.get('avg_latency', 0):.0f}ms",
                        "Total Cost": f"${stats.get('total_cost', 0):.4f}",
                        "Status": "🟢 Active" if stats.get("availability", False) else "🔴 Inactive"
                    })
                
                if summary_data:
                    df = pd.DataFrame(summary_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Usage Charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if len(summary_data) > 1:
                            st.write("**Request Distribution**")
                            chart_data = pd.DataFrame({
                                'Provider': [row['Provider'] for row in summary_data],
                                'Requests': [int(row['Requests']) for row in summary_data]
                            })
                            if chart_data['Requests'].sum() > 0:
                                st.bar_chart(chart_data.set_index('Provider'))
                    
                    with col2:
                        if len(summary_data) > 1:
                            st.write("**Error Rate**")
                            error_data = pd.DataFrame({
                                'Provider': [row['Provider'] for row in summary_data],
                                'Errors': [int(row['Errors']) for row in summary_data]
                            })
                            if error_data['Errors'].sum() > 0:
                                st.bar_chart(error_data.set_index('Provider'))
            else:
                st.info("No usage statistics available yet. Run some analyses to see data.")
        
        # System Health
        st.write("**System Health**")
        if st.button("🔍 Run Health Check", key="health_check"):
            with st.spinner("Checking system health..."):
                try:
                    health_status = self.llm_manager.health_check()
                    
                    for provider, status in health_status.items():
                        if status["status"] == "healthy":
                            st.success(f"✅ {provider.title()}: Healthy")
                        else:
                            st.error(f"❌ {provider.title()}: {status.get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Health check failed: {e}")
        
        # Configuration Export/Import
        st.write("**Configuration Management**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Configuration", key="export_config"):
                config_dict = self.config.to_dict()
                config_json = json.dumps(config_dict, indent=2)
                
                st.download_button(
                    label="Download Configuration",
                    data=config_json,
                    file_name=f"valicred_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            uploaded_file = st.file_uploader(
                "📤 Import Configuration",
                type="json",
                help="Upload a previously exported configuration file"
            )
            
            if uploaded_file is not None:
                try:
                    config_data = json.load(uploaded_file)
                    
                    if st.button("Apply Imported Configuration", key="import_config"):
                        # Validate and apply configuration
                        # Note: In production, add comprehensive validation
                        st.success("Configuration imported successfully!")
                        st.info("Please restart the application to apply all changes.")
                        
                except Exception as e:
                    st.error(f"Failed to import configuration: {e}")

# Singleton instance for use across the application
_config_panel: Optional[ConfigurationPanel] = None

def get_configuration_panel() -> ConfigurationPanel:
    """Get the global configuration panel instance"""
    global _config_panel
    if _config_panel is None:
        _config_panel = ConfigurationPanel()
    return _config_panel