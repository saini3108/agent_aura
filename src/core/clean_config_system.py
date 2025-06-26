"""
Clean Configuration System
=========================

Centralized, dynamic configuration management that eliminates hardcoded values
and provides runtime configuration updates.
"""

import json
import os
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    name: str
    description: str
    timeout: int = 300
    retry_attempts: int = 3
    priority: int = 1
    prompts: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationConfig:
    """Validation parameters and thresholds"""
    auc_threshold: float = 0.65
    ks_threshold: float = 0.3
    psi_threshold: float = 0.25
    gini_threshold: float = 0.3
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    cross_validation_folds: int = 5

@dataclass
class WorkflowConfig:
    """Workflow execution configuration"""
    execution_order: List[str] = field(default_factory=lambda: [
        "analyst", "validator", "documentation", "human_review", "reviewer", "auditor"
    ])
    parallel_execution: bool = False
    timeout_seconds: int = 300
    enable_human_review: bool = True
    auto_approve_threshold: float = 0.9
    max_retries: int = 3

@dataclass
class UIConfig:
    """User interface configuration"""
    theme: str = "light"
    show_debug_info: bool = False
    max_display_rows: int = 1000
    chart_height: int = 400
    enable_real_time_updates: bool = True
    refresh_interval: int = 5

@dataclass
class DataConfig:
    """Data processing configuration"""
    max_file_size_mb: int = 200
    supported_formats: List[str] = field(default_factory=lambda: ["csv", "xlsx", "json"])
    required_columns: List[str] = field(default_factory=list)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=dict)

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_dir: str = "src/config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Configuration instances
        self.agents: Dict[str, AgentConfig] = {}
        self.validation = ValidationConfig()
        self.workflow = WorkflowConfig()
        self.ui = UIConfig()
        self.data = DataConfig()
        
        # Load configurations
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all configuration files"""
        self._load_agent_configs()
        self._load_validation_config()
        self._load_workflow_config()
        self._load_ui_config()
        self._load_data_config()
    
    def _load_agent_configs(self):
        """Load agent configurations"""
        agents_file = self.config_dir / "agents.json"
        if agents_file.exists():
            try:
                with open(agents_file, 'r') as f:
                    agents_data = json.load(f)
                
                for agent_name, config_data in agents_data.items():
                    self.agents[agent_name] = AgentConfig(**config_data)
            except Exception as e:
                logger.error(f"Error loading agent configs: {e}")
                self._create_default_agent_configs()
        else:
            self._create_default_agent_configs()
    
    def _create_default_agent_configs(self):
        """Create default agent configurations"""
        default_agents = {
            "analyst": AgentConfig(
                name="Credit Risk Analyst",
                description="Analyzes model structure, data quality, and risk parameters",
                timeout=300,
                retry_attempts=3,
                priority=1,
                prompts={
                    "system": "You are a senior credit risk analyst specializing in model validation.",
                    "analysis": "Analyze the provided credit data for quality, completeness, and risk indicators."
                },
                parameters={
                    "min_data_quality_score": 0.8,
                    "required_features": ["credit_score", "income", "debt_ratio"],
                    "data_validation_rules": {
                        "min_records": 100,
                        "max_missing_percentage": 0.1
                    }
                }
            ),
            "validator": AgentConfig(
                name="Model Validator",
                description="Calculates validation metrics and statistical measures",
                timeout=600,
                retry_attempts=3,
                priority=2,
                prompts={
                    "system": "You are a model validation specialist focused on statistical accuracy.",
                    "validation": "Calculate and interpret key validation metrics for the credit model."
                },
                parameters={
                    "metrics_to_calculate": ["auc", "ks", "psi", "gini"],
                    "statistical_tests": ["ks_test", "chi_square", "t_test"],
                    "confidence_intervals": True
                }
            ),
            "documentation": AgentConfig(
                name="Compliance Specialist",
                description="Reviews documentation for regulatory compliance",
                timeout=300,
                retry_attempts=2,
                priority=3,
                prompts={
                    "system": "You are a compliance expert specializing in Basel III and IFRS 9.",
                    "review": "Review documentation for completeness and regulatory compliance."
                },
                parameters={
                    "required_documents": [
                        "model_methodology",
                        "validation_report",
                        "governance_policy"
                    ],
                    "compliance_frameworks": ["Basel III", "IFRS 9", "SR 11-7"],
                    "documentation_standards": {
                        "min_sections": 5,
                        "require_approval_signatures": True
                    }
                }
            ),
            "reviewer": AgentConfig(
                name="Risk Reviewer",
                description="Generates findings and recommendations",
                timeout=300,
                retry_attempts=2,
                priority=4,
                prompts={
                    "system": "You are a risk management reviewer providing executive insights.",
                    "review": "Generate comprehensive findings and actionable recommendations."
                },
                parameters={
                    "risk_categories": ["credit", "market", "operational", "compliance"],
                    "severity_levels": ["low", "medium", "high", "critical"],
                    "recommendation_types": ["immediate", "short_term", "long_term"]
                }
            ),
            "auditor": AgentConfig(
                name="Independent Auditor",
                description="Provides final validation and approval recommendation",
                timeout=300,
                retry_attempts=2,
                priority=5,
                prompts={
                    "system": "You are an independent auditor ensuring validation quality.",
                    "audit": "Perform final audit and provide approval recommendation."
                },
                parameters={
                    "audit_criteria": [
                        "data_quality",
                        "model_performance",
                        "documentation_completeness",
                        "compliance_adherence"
                    ],
                    "approval_thresholds": {
                        "minimum_auc": 0.6,
                        "maximum_psi": 0.25,
                        "required_documentation_score": 0.8
                    }
                }
            )
        }
        
        self.agents = default_agents
        self._save_agent_configs()
    
    def _load_validation_config(self):
        """Load validation configuration"""
        validation_file = self.config_dir / "validation.json"
        if validation_file.exists():
            try:
                with open(validation_file, 'r') as f:
                    data = json.load(f)
                self.validation = ValidationConfig(**data)
            except Exception as e:
                logger.error(f"Error loading validation config: {e}")
        else:
            self._save_validation_config()
    
    def _load_workflow_config(self):
        """Load workflow configuration"""
        workflow_file = self.config_dir / "workflow.json"
        if workflow_file.exists():
            try:
                with open(workflow_file, 'r') as f:
                    data = json.load(f)
                self.workflow = WorkflowConfig(**data)
            except Exception as e:
                logger.error(f"Error loading workflow config: {e}")
        else:
            self._save_workflow_config()
    
    def _load_ui_config(self):
        """Load UI configuration"""
        ui_file = self.config_dir / "ui.json"
        if ui_file.exists():
            try:
                with open(ui_file, 'r') as f:
                    data = json.load(f)
                self.ui = UIConfig(**data)
            except Exception as e:
                logger.error(f"Error loading UI config: {e}")
        else:
            self._save_ui_config()
    
    def _load_data_config(self):
        """Load data configuration"""
        data_file = self.config_dir / "data.json"
        if data_file.exists():
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                self.data = DataConfig(**data)
            except Exception as e:
                logger.error(f"Error loading data config: {e}")
        else:
            self._save_data_config()
    
    def _save_agent_configs(self):
        """Save agent configurations"""
        agents_data = {name: asdict(config) for name, config in self.agents.items()}
        with open(self.config_dir / "agents.json", 'w') as f:
            json.dump(agents_data, f, indent=2)
    
    def _save_validation_config(self):
        """Save validation configuration"""
        with open(self.config_dir / "validation.json", 'w') as f:
            json.dump(asdict(self.validation), f, indent=2)
    
    def _save_workflow_config(self):
        """Save workflow configuration"""
        with open(self.config_dir / "workflow.json", 'w') as f:
            json.dump(asdict(self.workflow), f, indent=2)
    
    def _save_ui_config(self):
        """Save UI configuration"""
        with open(self.config_dir / "ui.json", 'w') as f:
            json.dump(asdict(self.ui), f, indent=2)
    
    def _save_data_config(self):
        """Save data configuration"""
        with open(self.config_dir / "data.json", 'w') as f:
            json.dump(asdict(self.data), f, indent=2)
    
    # Public API methods
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Get configuration for specific agent"""
        return self.agents.get(agent_name)
    
    def update_agent_config(self, agent_name: str, updates: Dict[str, Any]) -> bool:
        """Update agent configuration"""
        if agent_name in self.agents:
            current_config = asdict(self.agents[agent_name])
            current_config.update(updates)
            self.agents[agent_name] = AgentConfig(**current_config)
            self._save_agent_configs()
            return True
        return False
    
    def get_validation_thresholds(self) -> Dict[str, float]:
        """Get validation thresholds"""
        return {
            "auc_threshold": self.validation.auc_threshold,
            "ks_threshold": self.validation.ks_threshold,
            "psi_threshold": self.validation.psi_threshold,
            "gini_threshold": self.validation.gini_threshold
        }
    
    def update_validation_threshold(self, metric: str, value: float) -> bool:
        """Update validation threshold"""
        if hasattr(self.validation, f"{metric}_threshold"):
            setattr(self.validation, f"{metric}_threshold", value)
            self._save_validation_config()
            return True
        return False
    
    def get_workflow_order(self) -> List[str]:
        """Get workflow execution order"""
        return self.workflow.execution_order.copy()
    
    def update_workflow_order(self, new_order: List[str]) -> bool:
        """Update workflow execution order"""
        # Validate that all required agents are included
        required_agents = set(self.agents.keys())
        provided_agents = set(new_order)
        
        if required_agents.issubset(provided_agents):
            self.workflow.execution_order = new_order
            self._save_workflow_config()
            return True
        return False
    
    def get_ui_settings(self) -> Dict[str, Any]:
        """Get UI configuration"""
        return asdict(self.ui)
    
    def update_ui_setting(self, setting: str, value: Any) -> bool:
        """Update UI setting"""
        if hasattr(self.ui, setting):
            setattr(self.ui, setting, value)
            self._save_ui_config()
            return True
        return False
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return asdict(self.data)
    
    def update_data_config(self, updates: Dict[str, Any]) -> bool:
        """Update data configuration"""
        try:
            current_config = asdict(self.data)
            current_config.update(updates)
            self.data = DataConfig(**current_config)
            self._save_data_config()
            return True
        except Exception as e:
            logger.error(f"Error updating data config: {e}")
            return False
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations"""
        return {
            "agents": {name: asdict(config) for name, config in self.agents.items()},
            "validation": asdict(self.validation),
            "workflow": asdict(self.workflow),
            "ui": asdict(self.ui),
            "data": asdict(self.data)
        }
    
    def export_config(self, filepath: Optional[str] = None) -> str:
        """Export all configurations to JSON"""
        config_data = self.get_all_configs()
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)
            return filepath
        else:
            return json.dumps(config_data, indent=2)
    
    def import_config(self, config_data: Union[str, Dict[str, Any]]) -> bool:
        """Import configurations from JSON"""
        try:
            if isinstance(config_data, str):
                data = json.loads(config_data)
            else:
                data = config_data
            
            # Update each configuration section
            if "agents" in data:
                for agent_name, agent_data in data["agents"].items():
                    self.agents[agent_name] = AgentConfig(**agent_data)
                self._save_agent_configs()
            
            if "validation" in data:
                self.validation = ValidationConfig(**data["validation"])
                self._save_validation_config()
            
            if "workflow" in data:
                self.workflow = WorkflowConfig(**data["workflow"])
                self._save_workflow_config()
            
            if "ui" in data:
                self.ui = UIConfig(**data["ui"])
                self._save_ui_config()
            
            if "data" in data:
                self.data = DataConfig(**data["data"])
                self._save_data_config()
            
            return True
            
        except Exception as e:
            logger.error(f"Error importing config: {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """Reset all configurations to defaults"""
        try:
            self._create_default_agent_configs()
            self.validation = ValidationConfig()
            self.workflow = WorkflowConfig()
            self.ui = UIConfig()
            self.data = DataConfig()
            
            # Save all defaults
            self._save_validation_config()
            self._save_workflow_config()
            self._save_ui_config()
            self._save_data_config()
            
            return True
        except Exception as e:
            logger.error(f"Error resetting to defaults: {e}")
            return False

# Global configuration instance
_config_manager = None

def get_config() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def reload_config() -> ConfigManager:
    """Reload configuration from files"""
    global _config_manager
    _config_manager = ConfigManager()
    return _config_manager