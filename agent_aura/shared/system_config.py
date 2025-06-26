"""
System Configuration Management
Centralizes all hardcoded values and provides dynamic configuration
"""
import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

class SystemConfig:
    """Centralized configuration management system"""
    
    def __init__(self):
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        self.load_configurations()
    
    def load_configurations(self):
        """Load all configuration files"""
        self.risk_thresholds = self._load_risk_thresholds()
        self.validation_parameters = self._load_validation_parameters()
        self.mcp_config = self._load_mcp_config()
        self.compliance_frameworks = self._load_compliance_frameworks()
        self.ui_settings = self._load_ui_settings()
        self.workflow_config = self._load_workflow_config()
    
    def _load_risk_thresholds(self) -> Dict[str, Any]:
        """Load risk assessment thresholds"""
        default_thresholds = {
            "auc": {
                "excellent": 0.8,
                "good": 0.7,
                "acceptable": 0.6,
                "minimum": 0.5
            },
            "ks": {
                "excellent": 0.3,
                "good": 0.2,
                "acceptable": 0.15,
                "minimum": 0.1
            },
            "psi": {
                "stable": 0.1,
                "acceptable": 0.15,
                "moderate_drift": 0.25,
                "significant_drift": 0.5
            },
            "gini": {
                "excellent": 0.6,
                "good": 0.4,
                "acceptable": 0.2,
                "minimum": 0.1
            },
            "data_quality": {
                "excellent": 0.95,
                "good": 0.9,
                "acceptable": 0.8,
                "minimum": 0.7
            }
        }
        return self._load_or_create_config("risk_thresholds.json", default_thresholds)
    
    def _load_validation_parameters(self) -> Dict[str, Any]:
        """Load validation parameters"""
        default_params = {
            "cross_validation": {
                "folds": 5,
                "stratify": True,
                "random_state": 42
            },
            "model_training": {
                "test_size": 0.2,
                "validation_size": 0.1,
                "random_state": 42
            },
            "statistical_tests": {
                "confidence_level": 0.95,
                "psi_bins": 10,
                "ks_alternative": "two-sided"
            },
            "performance_metrics": {
                "calculate_auc": True,
                "calculate_ks": True,
                "calculate_psi": True,
                "calculate_gini": True,
                "calculate_lift": True
            }
        }
        return self._load_or_create_config("validation_parameters.json", default_params)
    
    def _load_mcp_config(self) -> Dict[str, Any]:
        """Load MCP agent configuration"""
        default_mcp = {
            "agents": {
                "analyst": {
                    "name": "AnalystAgent",
                    "description": "Analyzes model structure and data characteristics",
                    "tools": ["data_analysis", "feature_analysis", "quality_assessment"],
                    "timeout": 300,
                    "retry_attempts": 3
                },
                "validator": {
                    "name": "ValidatorAgent", 
                    "description": "Calculates validation metrics and performance",
                    "tools": ["auc_calculation", "ks_test", "psi_calculation", "model_training"],
                    "timeout": 600,
                    "retry_attempts": 3
                },
                "documentation": {
                    "name": "DocumentationAgent",
                    "description": "Reviews compliance documentation",
                    "tools": ["document_parsing", "compliance_check", "gap_analysis"],
                    "timeout": 300,
                    "retry_attempts": 2
                },
                "reviewer": {
                    "name": "ReviewerAgent",
                    "description": "Generates findings and recommendations",
                    "tools": ["risk_assessment", "finding_generation", "recommendation_engine"],
                    "timeout": 300,
                    "retry_attempts": 2
                },
                "auditor": {
                    "name": "AuditorAgent",
                    "description": "Performs final validation and approval",
                    "tools": ["independent_validation", "compliance_audit", "approval_assessment"],
                    "timeout": 300,
                    "retry_attempts": 2
                }
            },
            "workflow": {
                "execution_order": ["analyst", "validator", "documentation", "human_review", "reviewer", "auditor"],
                "parallel_execution": False,
                "checkpoint_enabled": True,
                "auto_retry": True
            },
            "human_in_loop": {
                "pause_points": ["human_review"],
                "timeout": 3600,
                "escalation_policy": "admin_notification",
                "required_approvals": 1
            }
        }
        return self._load_or_create_config("mcp_agents.json", default_mcp)
    
    def _load_compliance_frameworks(self) -> Dict[str, Any]:
        """Load compliance framework definitions"""
        default_frameworks = {
            "basel_iii": {
                "name": "Basel III",
                "requirements": [
                    "Capital adequacy assessment",
                    "Risk-weighted asset calculation",
                    "Leverage ratio compliance",
                    "Liquidity coverage ratio"
                ],
                "documentation_requirements": [
                    "Model methodology document",
                    "Validation report",
                    "Back-testing results",
                    "Governance approval"
                ]
            },
            "ifrs9": {
                "name": "IFRS 9",
                "requirements": [
                    "Expected credit loss calculation",
                    "Stage classification methodology",
                    "Forward-looking information",
                    "Impairment assessment"
                ],
                "documentation_requirements": [
                    "ECL model documentation",
                    "Stage transition criteria",
                    "Macroeconomic scenario analysis",
                    "Model performance monitoring"
                ]
            },
            "model_risk_management": {
                "name": "Model Risk Management (SR 11-7)",
                "requirements": [
                    "Model development documentation",
                    "Independent validation",
                    "Ongoing monitoring",
                    "Issue management"
                ],
                "documentation_requirements": [
                    "Model inventory",
                    "Validation framework",
                    "Issue tracking",
                    "Remediation plans"
                ]
            }
        }
        return self._load_or_create_config("compliance_frameworks.json", default_frameworks)
    
    def _load_ui_settings(self) -> Dict[str, Any]:
        """Load UI configuration settings"""
        default_ui = {
            "theme": {
                "primary_color": "#FF6B6B",
                "background_color": "#FFFFFF",
                "secondary_background": "#F0F2F6",
                "text_color": "#262730"
            },
            "dashboard": {
                "show_progress_bar": True,
                "auto_refresh_interval": 5,
                "max_chart_points": 1000,
                "default_chart_height": 400
            },
            "workflow": {
                "show_agent_details": True,
                "show_execution_time": True,
                "show_step_dependencies": True,
                "enable_step_rollback": False
            },
            "reports": {
                "default_format": "html",
                "include_charts": True,
                "include_raw_data": False,
                "max_report_size": "10MB"
            }
        }
        return self._load_or_create_config("ui_settings.json", default_ui)
    
    def _load_workflow_config(self) -> Dict[str, Any]:
        """Load workflow configuration"""
        default_workflow = {
            "execution_settings": {
                "max_concurrent_agents": 1,
                "step_timeout": 600,
                "total_timeout": 3600,
                "enable_checkpoints": True
            },
            "retry_policy": {
                "max_retries": 3,
                "retry_delay": 5,
                "exponential_backoff": True
            },
            "human_review": {
                "mandatory_steps": ["human_review"],
                "optional_reviews": ["validator", "reviewer"],
                "review_timeout": 3600,
                "auto_approve": False
            },
            "notifications": {
                "email_enabled": False,
                "slack_enabled": False,
                "in_app_only": True
            }
        }
        return self._load_or_create_config("workflow_config.json", default_workflow)
    
    def _load_or_create_config(self, filename: str, default_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration file or create with defaults"""
        config_path = self.config_dir / filename
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        # Create default configuration
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def update_config(self, config_type: str, updates: Dict[str, Any]) -> bool:
        """Update configuration dynamically"""
        try:
            if config_type == "risk_thresholds":
                self.risk_thresholds.update(updates)
                self._save_config("risk_thresholds.json", self.risk_thresholds)
            elif config_type == "validation_parameters":
                self.validation_parameters.update(updates)
                self._save_config("validation_parameters.json", self.validation_parameters)
            elif config_type == "mcp_config":
                self.mcp_config.update(updates)
                self._save_config("mcp_agents.json", self.mcp_config)
            elif config_type == "ui_settings":
                self.ui_settings.update(updates)
                self._save_config("ui_settings.json", self.ui_settings)
            elif config_type == "workflow_config":
                self.workflow_config.update(updates)
                self._save_config("workflow_config.json", self.workflow_config)
            else:
                return False
            return True
        except Exception:
            return False
    
    def _save_config(self, filename: str, config: Dict[str, Any]):
        """Save configuration to file"""
        config_path = self.config_dir / filename
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get_agent_config(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for specific agent"""
        return self.mcp_config.get("agents", {}).get(agent_name)
    
    def get_threshold(self, metric: str, level: str) -> Optional[float]:
        """Get specific threshold value"""
        return self.risk_thresholds.get(metric, {}).get(level)
    
    def get_compliance_requirements(self, framework: str) -> Optional[Dict[str, Any]]:
        """Get compliance requirements for framework"""
        return self.compliance_frameworks.get(framework)

# Global configuration instance
config = SystemConfig()