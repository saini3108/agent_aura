import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import json
import os
from datetime import datetime

class SampleDataLoader:
    """Utility class for loading and managing sample data for ValiCred-AI"""
    
    def __init__(self, base_path: str = "sample_data"):
        self.base_path = base_path
        self._ensure_sample_data_exists()
    
    def _ensure_sample_data_exists(self):
        """Ensure sample data directory and files exist"""
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
    
    def load_credit_data(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load sample credit risk data"""
        try:
            file_path = os.path.join(self.base_path, "credit_data.csv")
            df = pd.read_csv(file_path)
            
            # Generate data info
            info = {
                "source": "Sample Credit Risk Dataset",
                "description": "Synthetic credit risk data for model validation testing",
                "shape": df.shape,
                "features": df.columns.tolist(),
                "target_variable": "default_flag" if "default_flag" in df.columns else None,
                "created_at": datetime.now().isoformat(),
                "data_quality": {
                    "missing_values": df.isnull().sum().sum(),
                    "duplicate_records": df.duplicated().sum(),
                    "completeness_rate": 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
                }
            }
            
            if "default_flag" in df.columns:
                info["default_rate"] = df["default_flag"].mean()
                info["class_distribution"] = df["default_flag"].value_counts().to_dict()
            
            return df, info
            
        except FileNotFoundError:
            raise Exception(f"Sample credit data not found at {file_path}")
        except Exception as e:
            raise Exception(f"Error loading credit data: {str(e)}")
    
    def load_validation_parameters(self) -> Dict[str, Any]:
        """Load validation parameters from CSV"""
        try:
            file_path = os.path.join(self.base_path, "validation_parameters.csv")
            df = pd.read_csv(file_path)
            
            parameters = {}
            for _, row in df.iterrows():
                param_name = row['parameter_name']
                param_value = row['parameter_value']
                param_type = row['parameter_type']
                
                # Convert to appropriate type
                if param_type == 'float':
                    param_value = float(param_value)
                elif param_type == 'int':
                    param_value = int(param_value)
                elif param_type == 'bool':
                    param_value = bool(param_value)
                
                parameters[param_name] = {
                    "value": param_value,
                    "type": param_type,
                    "description": row['description'],
                    "threshold_type": row['threshold_type']
                }
            
            return parameters
            
        except FileNotFoundError:
            raise Exception(f"Validation parameters not found at {file_path}")
        except Exception as e:
            raise Exception(f"Error loading validation parameters: {str(e)}")
    
    def load_risk_thresholds(self) -> Dict[str, Any]:
        """Load risk assessment thresholds"""
        try:
            file_path = os.path.join(self.base_path, "risk_thresholds.csv")
            df = pd.read_csv(file_path)
            
            thresholds = {}
            for _, row in df.iterrows():
                category = row['risk_category']
                if category not in thresholds:
                    thresholds[category] = {}
                
                thresholds[category][row['metric_name']] = {
                    "excellent": float(row['excellent_threshold']),
                    "good": float(row['good_threshold']),
                    "acceptable": float(row['acceptable_threshold']),
                    "poor": float(row['poor_threshold']),
                    "weight": float(row['weight'])
                }
            
            return thresholds
            
        except FileNotFoundError:
            raise Exception(f"Risk thresholds not found at {file_path}")
        except Exception as e:
            raise Exception(f"Error loading risk thresholds: {str(e)}")
    
    def get_sample_documents(self) -> Dict[str, Dict[str, Any]]:
        """Load actual sample documentation files for testing"""
        sample_docs = {}
        
        # List of actual sample document files
        doc_files = [
            "model_validation_report.txt",
            "model_methodology_document.txt", 
            "governance_policy.txt"
        ]
        
        for filename in doc_files:
            file_path = os.path.join(self.base_path, filename)
            try:
                if os.path.exists(file_path):
                    # Read file content and get size
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    file_size = len(content.encode('utf-8'))
                    
                    # Determine document type and compliance areas
                    doc_type = "documentation"
                    compliance_areas = []
                    
                    if "validation" in filename.lower():
                        doc_type = "validation_report"
                        compliance_areas = ["Basel III", "Model Risk Management"]
                    elif "methodology" in filename.lower():
                        doc_type = "methodology_document"
                        compliance_areas = ["IFRS 9", "Model Documentation"]
                    elif "governance" in filename.lower():
                        doc_type = "policy_document"
                        compliance_areas = ["Basel III", "Model Risk Management", "Governance"]
                    
                    sample_docs[filename] = {
                        "file_content": content,
                        "size": file_size,
                        "uploaded_at": datetime.now().isoformat(),
                        "type": doc_type,
                        "compliance_areas": compliance_areas,
                        "file_path": file_path
                    }
                    
            except Exception as e:
                print(f"Warning: Could not load {filename}: {str(e)}")
        
        return sample_docs
    
    def create_comprehensive_dataset(self, size: int = 1000) -> pd.DataFrame:
        """Create a more comprehensive credit risk dataset for testing"""
        np.random.seed(42)
        
        # Generate base features
        age = np.random.normal(40, 12, size).astype(int)
        age = np.clip(age, 18, 80)
        
        income = np.random.lognormal(10.5, 0.7, size)
        income = np.round(income, 0).astype(int)
        
        credit_score = np.random.normal(700, 80, size).astype(int)
        credit_score = np.clip(credit_score, 300, 850)
        
        employment_years = np.random.exponential(8, size)
        employment_years = np.clip(employment_years, 0, 40).round(1)
        
        # Loan characteristics
        loan_amount = income * np.random.uniform(0.1, 0.8, size)
        loan_amount = np.round(loan_amount, 0).astype(int)
        
        debt_to_income = np.random.beta(2, 5, size)
        debt_to_income = np.round(debt_to_income, 3)
        
        # Categorical features
        home_ownership = np.random.choice(['Own', 'Rent', 'Mortgage'], size, p=[0.4, 0.35, 0.25])
        loan_purpose = np.random.choice(['debt_consolidation', 'home_improvement', 'major_purchase', 'credit_card'], 
                                       size, p=[0.4, 0.25, 0.2, 0.15])
        
        # Create default probability based on features
        default_prob = (
            0.1 +  # Base rate
            0.05 * (credit_score < 650) +  # Poor credit
            0.03 * (debt_to_income > 0.4) +  # High DTI
            0.02 * (employment_years < 2) +  # Short employment
            0.04 * (age < 25) +  # Young age
            0.02 * (home_ownership == 'Rent')  # Renting
        )
        
        # Generate defaults
        default_flag = np.random.binomial(1, default_prob, size)
        
        # Create DataFrame
        df = pd.DataFrame({
            'customer_id': range(1, size + 1),
            'age': age,
            'income': income,
            'credit_score': credit_score,
            'loan_amount': loan_amount,
            'employment_years': employment_years,
            'debt_to_income': debt_to_income,
            'home_ownership': home_ownership,
            'loan_purpose': loan_purpose,
            'default_flag': default_flag
        })
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality for the loaded dataset"""
        quality_report = {
            "timestamp": datetime.now().isoformat(),
            "total_records": len(df),
            "total_features": len(df.columns),
            "missing_data": {
                "total_missing": df.isnull().sum().sum(),
                "missing_by_column": df.isnull().sum().to_dict(),
                "completeness_rate": 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
            },
            "duplicates": {
                "total_duplicates": df.duplicated().sum(),
                "duplicate_rate": df.duplicated().sum() / len(df)
            },
            "data_types": df.dtypes.to_dict(),
            "quality_score": 0.0
        }
        
        # Calculate overall quality score
        completeness_score = quality_report["missing_data"]["completeness_rate"]
        duplicate_penalty = quality_report["duplicates"]["duplicate_rate"]
        
        quality_report["quality_score"] = max(0, completeness_score - duplicate_penalty)
        
        # Add feature-specific analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            quality_report["numeric_features"] = {
                "count": len(numeric_columns),
                "summary_stats": df[numeric_columns].describe().to_dict()
            }
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            quality_report["categorical_features"] = {
                "count": len(categorical_columns),
                "unique_values": {col: df[col].nunique() for col in categorical_columns}
            }
        
        return quality_report

# Global instance for easy access
sample_loader = SampleDataLoader()