"""
Configuration Management for ValiCred-AI
========================================

This module provides centralized configuration management for the entire application.
It handles API keys, model settings, risk thresholds, and workflow configurations.

Usage:
    from src.config.settings import get_config, update_config
    
    config = get_config()
    config.update_api_key('groq', 'your-api-key')
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import json
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for AI models and providers"""
    provider: str = "groq"
    model_name: str = "llama-3.1-8b-instant"
    temperature: float = 0.3
    max_tokens: int = 2000
    timeout: int = 30

@dataclass
class RiskThresholds:
    """Credit risk assessment thresholds"""
    auc_excellent: float = 0.8
    auc_good: float = 0.7
    auc_acceptable: float = 0.6
    ks_excellent: float = 0.3
    ks_good: float = 0.2
    ks_acceptable: float = 0.15
    psi_stable: float = 0.1
    psi_monitoring: float = 0.2
    psi_unstable: float = 0.25

@dataclass
class WorkflowConfig:
    """Workflow execution configuration"""
    max_retries: int = 3
    retry_delay: int = 5
    timeout_seconds: int = 300
    enable_human_review: bool = True
    auto_approve_threshold: float = 0.9

@dataclass
class APIConfig:
    """API configuration for external integrations"""
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    
    def __post_init__(self):
        """Load API keys from environment variables"""
        self.groq_api_key = self.groq_api_key or os.getenv("GROQ_API_KEY")
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.gemini_api_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY")

@dataclass
class ValiCredConfig:
    """Main configuration class for ValiCred-AI"""
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk_thresholds: RiskThresholds = field(default_factory=RiskThresholds)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    
    # Agent configuration
    agents: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "analyst": {
            "name": "Credit Risk Analyst",
            "description": "Senior credit risk analyst with expertise in model validation and regulatory compliance",
            "timeout": 300,
            "retry_attempts": 3,
            "priority": 1
        },
        "validator": {
            "name": "Model Validator",
            "description": "Model validation specialist focused on statistical metrics and performance assessment",
            "timeout": 600,
            "retry_attempts": 3,
            "priority": 2
        },
        "documentation": {
            "name": "Compliance Specialist",
            "description": "Compliance expert specializing in Basel III, IFRS 9, and model risk management",
            "timeout": 300,
            "retry_attempts": 2,
            "priority": 3
        },
        "reviewer": {
            "name": "Risk Reviewer",
            "description": "Risk management reviewer providing executive-level findings and recommendations",
            "timeout": 300,
            "retry_attempts": 2,
            "priority": 4
        },
        "auditor": {
            "name": "Independent Auditor",
            "description": "Independent auditor providing final validation and approval recommendations",
            "timeout": 300,
            "retry_attempts": 2,
            "priority": 5
        }
    })
    
    # Regulatory frameworks
    regulatory_frameworks: List[str] = field(default_factory=lambda: [
        "Basel III",
        "IFRS 9",
        "SR 11-7",
        "CCAR",
        "CECL",
        "CRR II"
    ])
    
    def update_api_key(self, provider: str, api_key: str) -> None:
        """Update API key for a specific provider"""
        if provider == "groq":
            self.api.groq_api_key = api_key
        elif provider == "openai":
            self.api.openai_api_key = api_key
        elif provider == "anthropic":
            self.api.anthropic_api_key = api_key
        elif provider == "gemini":
            self.api.gemini_api_key = api_key
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of providers with valid API keys"""
        providers = []
        if self.api.groq_api_key:
            providers.append("groq")
        if self.api.openai_api_key:
            providers.append("openai")
        if self.api.anthropic_api_key:
            providers.append("anthropic")
        if self.api.gemini_api_key:
            providers.append("gemini")
        return providers
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent"""
        return self.agents.get(agent_name, {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "model": {
                "provider": self.model.provider,
                "model_name": self.model.model_name,
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens,
                "timeout": self.model.timeout
            },
            "risk_thresholds": {
                "auc_excellent": self.risk_thresholds.auc_excellent,
                "auc_good": self.risk_thresholds.auc_good,
                "auc_acceptable": self.risk_thresholds.auc_acceptable,
                "ks_excellent": self.risk_thresholds.ks_excellent,
                "ks_good": self.risk_thresholds.ks_good,
                "ks_acceptable": self.risk_thresholds.ks_acceptable,
                "psi_stable": self.risk_thresholds.psi_stable,
                "psi_monitoring": self.risk_thresholds.psi_monitoring,
                "psi_unstable": self.risk_thresholds.psi_unstable
            },
            "workflow": {
                "max_retries": self.workflow.max_retries,
                "retry_delay": self.workflow.retry_delay,
                "timeout_seconds": self.workflow.timeout_seconds,
                "enable_human_review": self.workflow.enable_human_review,
                "auto_approve_threshold": self.workflow.auto_approve_threshold
            },
            "agents": self.agents,
            "regulatory_frameworks": self.regulatory_frameworks
        }

# Global configuration instance
_config: Optional[ValiCredConfig] = None

def get_config() -> ValiCredConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = ValiCredConfig()
    return _config

def update_config(**kwargs) -> ValiCredConfig:
    """Update configuration with new values"""
    config = get_config()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config

def save_config(config: ValiCredConfig, filepath: Optional[str] = None) -> None:
    """Save configuration to file"""
    if filepath is None:
        config_path = Path.home() / ".valicred" / "config.json"
    else:
        config_path = Path(filepath)
    
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

def load_config(filepath: Optional[str] = None) -> ValiCredConfig:
    """Load configuration from file"""
    if filepath is None:
        config_path = Path.home() / ".valicred" / "config.json"
    else:
        config_path = Path(filepath)
    
    if not config_path.exists():
        return ValiCredConfig()
    
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    config = ValiCredConfig()
    
    # Update model config
    if 'model' in data:
        for key, value in data['model'].items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)
    
    # Update risk thresholds
    if 'risk_thresholds' in data:
        for key, value in data['risk_thresholds'].items():
            if hasattr(config.risk_thresholds, key):
                setattr(config.risk_thresholds, key, value)
    
    # Update workflow config
    if 'workflow' in data:
        for key, value in data['workflow'].items():
            if hasattr(config.workflow, key):
                setattr(config.workflow, key, value)
    
    # Update agents config
    if 'agents' in data:
        config.agents.update(data['agents'])
    
    # Update regulatory frameworks
    if 'regulatory_frameworks' in data:
        config.regulatory_frameworks = data['regulatory_frameworks']
    
    return config

# Initialize with environment variables
config = get_config()