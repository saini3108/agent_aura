"""
Utilities Package
================

Core utility functions and classes for ValiCred-AI system
"""

from .workflow_engine import MCPWorkflowEngine, AgentExecution
from .audit_logger import AuditLogger
from .validation_metrics import ValidationMetrics
from .enhanced_report_generator import EnhancedReportGenerator

# Sample data loader is defined inline in app_factory.py
class SampleDataLoader:
    """Simple sample data loader for demo purposes"""

    def get_sample_data(self):
        import pandas as pd
        import numpy as np

        # Generate sample credit data
        np.random.seed(42)
        n_samples = 1000

        data = {
            'customer_id': range(1, n_samples + 1),
            'credit_score': np.random.normal(650, 100, n_samples).astype(int),
            'income': np.random.lognormal(10.5, 0.5, n_samples),
            'debt_to_income': np.random.beta(2, 5, n_samples),
            'default': np.random.binomial(1, 0.15, n_samples)
        }

        df = pd.DataFrame(data)
        df['credit_score'] = df['credit_score'].clip(300, 850)
        return df

    def get_sample_documents(self):
        return {
            'model_documentation.txt': 'Sample model documentation content',
            'validation_report.txt': 'Sample validation report content',
            'governance_policy.txt': 'Sample governance policy content'
        }

__all__ = [
    'MCPWorkflowEngine',
    'AgentExecution', 
    'SampleDataLoader',
    'AuditLogger',
    'ValidationMetrics',
    'EnhancedReportGenerator'
]