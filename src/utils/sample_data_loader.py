"""
Sample Data Loader for ValiCred-AI
=================================

This module provides sample data loading capabilities for the ValiCred-AI system.
It handles loading sample credit data and documentation files for testing and demonstration.
"""

import pandas as pd
import os
from typing import Dict, Any, List

class SimpleSampleLoader:
    """Simple sample data loader for ValiCred-AI"""
    
    def __init__(self):
        self.sample_data_path = "sample_data"
    
    def get_sample_data(self) -> pd.DataFrame:
        """Load sample credit data"""
        try:
            data_path = os.path.join(self.sample_data_path, "credit_data.csv")
            if os.path.exists(data_path):
                return pd.read_csv(data_path)
            else:
                # Return minimal sample data if file doesn't exist
                return pd.DataFrame({
                    'customer_id': range(1, 51),
                    'default_probability': [0.1 + (i * 0.01) for i in range(50)],
                    'credit_score': [600 + (i * 5) for i in range(50)],
                    'loan_amount': [10000 + (i * 1000) for i in range(50)],
                    'actual_default': [1 if i % 10 == 0 else 0 for i in range(50)]
                })
        except Exception as e:
            print(f"Error loading sample data: {e}")
            return pd.DataFrame()
    
    def get_sample_documents(self) -> Dict[str, str]:
        """Load sample documentation files"""
        documents = {}
        
        doc_files = [
            "model_methodology_document.txt",
            "model_validation_report.txt",
            "governance_policy.txt"
        ]
        
        for doc_file in doc_files:
            doc_path = os.path.join(self.sample_data_path, doc_file)
            try:
                if os.path.exists(doc_path):
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        documents[doc_file] = f.read()
                else:
                    documents[doc_file] = f"Sample {doc_file.replace('_', ' ').title()}"
            except Exception as e:
                print(f"Error loading {doc_file}: {e}")
                documents[doc_file] = f"Error loading {doc_file}"
        
        return documents
    
    def get_validation_parameters(self) -> Dict[str, Any]:
        """Load validation parameters"""
        try:
            params_path = os.path.join(self.sample_data_path, "validation_parameters.csv")
            if os.path.exists(params_path):
                df = pd.read_csv(params_path)
                return df.to_dict('records')
            else:
                return [
                    {"parameter": "AUC_threshold", "value": 0.7, "description": "Minimum AUC for model acceptance"},
                    {"parameter": "KS_threshold", "value": 0.2, "description": "Minimum KS statistic for model acceptance"},
                    {"parameter": "PSI_threshold", "value": 0.1, "description": "Maximum PSI for stability"}
                ]
        except Exception as e:
            print(f"Error loading validation parameters: {e}")
            return []
    
    def get_risk_thresholds(self) -> Dict[str, Any]:
        """Load risk thresholds"""
        try:
            thresholds_path = os.path.join(self.sample_data_path, "risk_thresholds.csv")
            if os.path.exists(thresholds_path):
                df = pd.read_csv(thresholds_path)
                return df.to_dict('records')
            else:
                return [
                    {"metric": "AUC", "excellent": 0.8, "good": 0.7, "acceptable": 0.6},
                    {"metric": "KS", "excellent": 0.3, "good": 0.2, "acceptable": 0.15},
                    {"metric": "PSI", "stable": 0.1, "monitoring": 0.2, "unstable": 0.25}
                ]
        except Exception as e:
            print(f"Error loading risk thresholds: {e}")
            return []