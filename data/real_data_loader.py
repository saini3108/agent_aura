"""
Real Credit Risk Data Integration System
=======================================

This module provides enterprise-grade data loading and validation for real credit risk datasets.
It supports multiple data sources, formats, and implements comprehensive data quality checks
following industry standards for credit risk modeling.

Features:
- Multiple data source support (CSV, Excel, Database, API)
- Comprehensive data validation and quality assessment
- Regulatory compliance data checks (Basel III, IFRS 9)
- Real-time data profiling and statistics
- Data lineage and audit trail tracking
- Integration with external credit bureaus and data providers

Usage:
    from src.data.real_data_loader import CreditDataLoader
    
    loader = CreditDataLoader()
    data, metadata = loader.load_credit_portfolio("path/to/data.csv")
    quality_report = loader.validate_data_quality(data)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json
import warnings
from dataclasses import dataclass
from io import StringIO
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """Comprehensive data quality metrics for credit risk data"""
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    validity_score: float
    uniqueness_score: float
    overall_score: float
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    regulatory_compliance: Dict[str, bool]

@dataclass
class CreditDataMetadata:
    """Metadata for credit risk datasets"""
    source: str
    load_timestamp: datetime
    record_count: int
    feature_count: int
    data_quality: DataQualityMetrics
    schema_version: str
    regulatory_period: str
    data_lineage: Dict[str, Any]

class CreditDataValidator:
    """Enterprise-grade credit data validation"""
    
    def __init__(self):
        self.required_fields = [
            'customer_id', 'loan_amount', 'interest_rate', 'loan_term',
            'credit_score', 'income', 'default_flag'
        ]
        
        self.optional_fields = [
            'age', 'employment_length', 'home_ownership', 'purpose',
            'debt_to_income', 'delinq_2yrs', 'open_acc', 'pub_rec',
            'total_acc', 'total_pymnt', 'recoveries'
        ]
        
        self.data_types = {
            'customer_id': 'object',
            'loan_amount': 'float64',
            'interest_rate': 'float64',
            'loan_term': 'int64',
            'credit_score': 'int64',
            'income': 'float64',
            'default_flag': 'int64',
            'age': 'int64',
            'employment_length': 'float64',
            'debt_to_income': 'float64'
        }
        
        self.value_ranges = {
            'loan_amount': (1000, 50000),
            'interest_rate': (0.03, 0.30),
            'loan_term': (12, 84),
            'credit_score': (300, 850),
            'income': (0, 500000),
            'default_flag': (0, 1),
            'age': (18, 100),
            'debt_to_income': (0, 1.0)
        }
    
    def validate_schema(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data schema against credit risk standards"""
        issues = []
        
        # Check required fields
        missing_required = set(self.required_fields) - set(data.columns)
        if missing_required:
            issues.append(f"Missing required fields: {missing_required}")
        
        # Check data types
        for field, expected_type in self.data_types.items():
            if field in data.columns:
                actual_type = str(data[field].dtype)
                if not self._is_compatible_type(actual_type, expected_type):
                    issues.append(f"Field '{field}' has type {actual_type}, expected {expected_type}")
        
        return len(issues) == 0, issues
    
    def _is_compatible_type(self, actual: str, expected: str) -> bool:
        """Check if data types are compatible"""
        if expected == 'float64' and actual in ['int64', 'float64']:
            return True
        if expected == 'int64' and actual in ['int64', 'int32']:
            return True
        if expected == 'object' and 'object' in actual:
            return True
        return actual == expected
    
    def validate_data_quality(self, data: pd.DataFrame) -> DataQualityMetrics:
        """Comprehensive data quality assessment"""
        issues = []
        recommendations = []
        
        # Completeness assessment
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells
        
        if completeness < 0.95:
            issues.append({
                "type": "completeness",
                "severity": "high" if completeness < 0.90 else "medium",
                "description": f"Data completeness is {completeness:.2%}",
                "affected_records": int(missing_cells)
            })
            recommendations.append("Address missing data through imputation or additional data collection")
        
        # Accuracy assessment (value range validation)
        accuracy_issues = 0
        total_validated_cells = 0
        
        for field, (min_val, max_val) in self.value_ranges.items():
            if field in data.columns and data[field].dtype in ['int64', 'float64']:
                out_of_range = ((data[field] < min_val) | (data[field] > max_val)).sum()
                total_validated_cells += len(data[field].dropna())
                accuracy_issues += out_of_range
                
                if out_of_range > 0:
                    issues.append({
                        "type": "accuracy",
                        "severity": "high" if out_of_range > len(data) * 0.05 else "medium",
                        "description": f"Field '{field}' has {out_of_range} values outside valid range [{min_val}, {max_val}]",
                        "affected_records": int(out_of_range)
                    })
        
        accuracy = (total_validated_cells - accuracy_issues) / max(total_validated_cells, 1)
        
        # Consistency assessment
        consistency_issues = []
        
        # Check for logical inconsistencies
        if 'loan_amount' in data.columns and 'income' in data.columns:
            high_dti = data['loan_amount'] > data['income'] * 0.5
            if high_dti.sum() > len(data) * 0.1:
                consistency_issues.append("High proportion of loans exceed 50% of income")
        
        if 'credit_score' in data.columns and 'default_flag' in data.columns:
            # High credit scores with defaults (potential data quality issue)
            high_score_defaults = ((data['credit_score'] > 750) & (data['default_flag'] == 1)).sum()
            if high_score_defaults > len(data) * 0.02:
                consistency_issues.append("Unusually high default rate for high credit scores")
        
        consistency = 1.0 - (len(consistency_issues) / 10)  # Normalize to 0-1
        
        # Validity assessment
        validity_issues = 0
        
        # Check for duplicates
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            validity_issues += duplicate_count
            issues.append({
                "type": "validity",
                "severity": "medium",
                "description": f"Found {duplicate_count} duplicate records",
                "affected_records": int(duplicate_count)
            })
        
        # Check customer ID uniqueness
        if 'customer_id' in data.columns:
            non_unique_ids = len(data) - data['customer_id'].nunique()
            if non_unique_ids > 0:
                validity_issues += non_unique_ids
                issues.append({
                    "type": "validity",
                    "severity": "high",
                    "description": f"Customer IDs are not unique: {non_unique_ids} duplicates",
                    "affected_records": int(non_unique_ids)
                })
        
        validity = max(0, 1.0 - (validity_issues / len(data)))
        
        # Uniqueness assessment
        uniqueness_scores = []
        for col in data.columns:
            if data[col].dtype == 'object':
                unique_ratio = data[col].nunique() / len(data)
                uniqueness_scores.append(unique_ratio)
        
        uniqueness = np.mean(uniqueness_scores) if uniqueness_scores else 1.0
        
        # Overall score (weighted average)
        overall_score = (
            completeness * 0.25 +
            accuracy * 0.30 +
            consistency * 0.25 +
            validity * 0.15 +
            uniqueness * 0.05
        )
        
        # Regulatory compliance checks
        regulatory_compliance = self._check_regulatory_compliance(data)
        
        # Generate recommendations
        if overall_score < 0.8:
            recommendations.append("Data quality score is below acceptable threshold for regulatory modeling")
        if accuracy < 0.95:
            recommendations.append("Implement data validation rules at source systems")
        if len(consistency_issues) > 0:
            recommendations.append("Review business logic and data collection processes")
        
        return DataQualityMetrics(
            completeness_score=completeness,
            accuracy_score=accuracy,
            consistency_score=consistency,
            validity_score=validity,
            uniqueness_score=uniqueness,
            overall_score=overall_score,
            issues=issues,
            recommendations=recommendations,
            regulatory_compliance=regulatory_compliance
        )
    
    def _check_regulatory_compliance(self, data: pd.DataFrame) -> Dict[str, bool]:
        """Check compliance with regulatory requirements"""
        compliance = {}
        
        # Basel III - Minimum data requirements
        basel_required = ['customer_id', 'loan_amount', 'credit_score', 'default_flag']
        compliance['basel_iii_data'] = all(field in data.columns for field in basel_required)
        
        # IFRS 9 - Forward looking information
        ifrs9_fields = ['loan_amount', 'interest_rate', 'loan_term', 'credit_score']
        compliance['ifrs9_forward_looking'] = all(field in data.columns for field in ifrs9_fields)
        
        # Data sufficiency for modeling
        compliance['sufficient_sample_size'] = len(data) >= 1000
        compliance['sufficient_defaults'] = data.get('default_flag', pd.Series()).sum() >= 50
        
        # Data recency (assume we need data within last 3 years)
        if 'origination_date' in data.columns:
            try:
                recent_cutoff = datetime.now() - timedelta(days=3*365)
                recent_data_pct = (pd.to_datetime(data['origination_date']) >= recent_cutoff).mean()
                compliance['data_recency'] = recent_data_pct >= 0.7
            except:
                compliance['data_recency'] = False
        else:
            compliance['data_recency'] = False
        
        return compliance

class CreditDataLoader:
    """Enterprise credit data loading and management system"""
    
    def __init__(self):
        self.validator = CreditDataValidator()
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.parquet', '.json']
        self.data_sources = {}
        
    def load_credit_portfolio(self, file_path: Union[str, Path], 
                            source_type: str = "file",
                            **kwargs) -> Tuple[pd.DataFrame, CreditDataMetadata]:
        """Load credit portfolio data from various sources"""
        start_time = datetime.now()
        
        try:
            # Load data based on source type
            if source_type == "file":
                data = self._load_from_file(file_path, **kwargs)
                source = f"file:{file_path}"
            elif source_type == "database":
                data = self._load_from_database(file_path, **kwargs)
                source = f"database:{kwargs.get('table_name', 'unknown')}"
            elif source_type == "api":
                data = self._load_from_api(file_path, **kwargs)
                source = f"api:{file_path}"
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
            
            # Validate schema
            schema_valid, schema_issues = self.validator.validate_schema(data)
            if not schema_valid:
                logger.warning(f"Schema validation issues: {schema_issues}")
            
            # Perform data quality assessment
            quality_metrics = self.validator.validate_data_quality(data)
            
            # Create metadata
            metadata = CreditDataMetadata(
                source=source,
                load_timestamp=start_time,
                record_count=len(data),
                feature_count=len(data.columns),
                data_quality=quality_metrics,
                schema_version="1.0",
                regulatory_period=self._determine_regulatory_period(data),
                data_lineage={
                    "load_method": source_type,
                    "load_timestamp": start_time.isoformat(),
                    "data_hash": self._calculate_data_hash(data),
                    "transformations": []
                }
            )
            
            logger.info(f"Successfully loaded {len(data)} records from {source}")
            logger.info(f"Data quality score: {quality_metrics.overall_score:.3f}")
            
            return data, metadata
            
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise
    
    def _load_from_file(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load data from file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Load based on file extension
        if file_path.suffix == '.csv':
            data = pd.read_csv(file_path, **kwargs)
        elif file_path.suffix in ['.xlsx', '.xls']:
            data = pd.read_excel(file_path, **kwargs)
        elif file_path.suffix == '.parquet':
            data = pd.read_parquet(file_path, **kwargs)
        elif file_path.suffix == '.json':
            data = pd.read_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return data
    
    def _load_from_database(self, connection_string: str, **kwargs) -> pd.DataFrame:
        """Load data from database"""
        try:
            import sqlalchemy
            
            engine = sqlalchemy.create_engine(connection_string)
            query = kwargs.get('query', f"SELECT * FROM {kwargs.get('table_name', 'credit_data')}")
            
            data = pd.read_sql(query, engine)
            return data
            
        except ImportError:
            raise ImportError("SQLAlchemy is required for database connections")
        except Exception as e:
            raise Exception(f"Database connection failed: {e}")
    
    def _load_from_api(self, api_endpoint: str, **kwargs) -> pd.DataFrame:
        """Load data from API endpoint"""
        try:
            headers = kwargs.get('headers', {})
            params = kwargs.get('params', {})
            auth = kwargs.get('auth', None)
            
            response = requests.get(api_endpoint, headers=headers, params=params, auth=auth)
            response.raise_for_status()
            
            if response.headers.get('content-type', '').startswith('application/json'):
                data = pd.DataFrame(response.json())
            elif response.headers.get('content-type', '').startswith('text/csv'):
                data = pd.read_csv(StringIO(response.text))
            else:
                raise ValueError("Unsupported API response format")
            
            return data
            
        except Exception as e:
            raise Exception(f"API data loading failed: {e}")
    
    def _determine_regulatory_period(self, data: pd.DataFrame) -> str:
        """Determine regulatory reporting period"""
        if 'origination_date' in data.columns:
            try:
                dates = pd.to_datetime(data['origination_date'])
                latest_date = dates.max()
                return f"Q{(latest_date.month-1)//3 + 1}_{latest_date.year}"
            except:
                pass
        
        current_date = datetime.now()
        return f"Q{(current_date.month-1)//3 + 1}_{current_date.year}"
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash for data lineage tracking"""
        import hashlib
        
        # Create a hash based on data shape and basic statistics
        data_str = f"{data.shape}_{data.dtypes.to_string()}_{data.describe().to_string()}"
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def generate_sample_credit_data(self, n_samples: int = 1000, 
                                  default_rate: float = 0.15,
                                  seed: int = 42) -> pd.DataFrame:
        """Generate realistic sample credit data for testing and development"""
        np.random.seed(seed)
        
        # Generate correlated features for realistic credit data
        data = {}
        
        # Customer IDs
        data['customer_id'] = [f"CUST_{i:06d}" for i in range(1, n_samples + 1)]
        
        # Credit scores (normally distributed around 650)
        data['credit_score'] = np.clip(
            np.random.normal(650, 80, n_samples).astype(int), 300, 850
        )
        
        # Age (between 18 and 80)
        data['age'] = np.random.randint(18, 81, n_samples)
        
        # Income (log-normal distribution)
        data['income'] = np.random.lognormal(np.log(45000), 0.6, n_samples)
        
        # Loan amount (correlated with income and credit score)
        income_factor = data['income'] / 50000
        credit_factor = data['credit_score'] / 650
        base_loan = 15000
        data['loan_amount'] = np.clip(
            base_loan * income_factor * credit_factor * np.random.uniform(0.5, 2.0, n_samples),
            1000, 50000
        )
        
        # Interest rate (inversely correlated with credit score)
        base_rate = 0.15
        credit_adjustment = (750 - data['credit_score']) / 1000
        data['interest_rate'] = np.clip(
            base_rate + credit_adjustment + np.random.normal(0, 0.02, n_samples),
            0.03, 0.30
        )
        
        # Loan term
        data['loan_term'] = np.random.choice([36, 48, 60, 72], n_samples, p=[0.3, 0.4, 0.2, 0.1])
        
        # Employment length
        data['employment_length'] = np.random.exponential(5, n_samples)
        
        # Debt-to-income ratio
        data['debt_to_income'] = np.clip(
            np.random.beta(2, 5, n_samples) * 0.6, 0, 1.0
        )
        
        # Home ownership
        data['home_ownership'] = np.random.choice(
            ['RENT', 'OWN', 'MORTGAGE'], n_samples, p=[0.4, 0.3, 0.3]
        )
        
        # Purpose
        data['purpose'] = np.random.choice([
            'debt_consolidation', 'credit_card', 'home_improvement', 
            'major_purchase', 'medical', 'vacation', 'other'
        ], n_samples, p=[0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.1])
        
        # Generate default flag based on risk factors
        risk_score = (
            (750 - data['credit_score']) / 500 * 0.4 +  # Credit score impact
            np.array(data['debt_to_income']) * 0.3 +       # DTI impact
            (data['interest_rate'] - 0.05) / 0.25 * 0.2 +  # Interest rate impact
            np.random.uniform(0, 0.1, n_samples)            # Random factor
        )
        
        # Convert risk score to default probability and generate defaults
        default_threshold = np.percentile(risk_score, (1 - default_rate) * 100)
        data['default_flag'] = (risk_score > default_threshold).astype(int)
        
        # Additional fields for completeness
        data['open_acc'] = np.random.poisson(8, n_samples)
        data['total_acc'] = data['open_acc'] + np.random.poisson(5, n_samples)
        data['delinq_2yrs'] = np.random.poisson(0.5, n_samples)
        data['pub_rec'] = np.random.poisson(0.1, n_samples)
        
        # Origination date (last 3 years)
        start_date = datetime.now() - timedelta(days=3*365)
        data['origination_date'] = [
            start_date + timedelta(days=np.random.randint(0, 3*365))
            for _ in range(n_samples)
        ]
        
        df = pd.DataFrame(data)
        
        # Add some realistic missing data patterns
        missing_patterns = {
            'employment_length': 0.05,  # 5% missing
            'debt_to_income': 0.02,     # 2% missing
            'delinq_2yrs': 0.01         # 1% missing
        }
        
        for col, missing_rate in missing_patterns.items():
            if col in df.columns:
                missing_mask = np.random.random(len(df)) < missing_rate
                df.loc[missing_mask, col] = np.nan
        
        return df
    
    def export_data_profile(self, data: pd.DataFrame, 
                          metadata: CreditDataMetadata,
                          output_path: str = "data_profile.json") -> Dict[str, Any]:
        """Export comprehensive data profile for audit and documentation"""
        profile = {
            "metadata": {
                "source": metadata.source,
                "load_timestamp": metadata.load_timestamp.isoformat(),
                "record_count": metadata.record_count,
                "feature_count": metadata.feature_count,
                "schema_version": metadata.schema_version,
                "regulatory_period": metadata.regulatory_period
            },
            "data_quality": {
                "completeness_score": metadata.data_quality.completeness_score,
                "accuracy_score": metadata.data_quality.accuracy_score,
                "consistency_score": metadata.data_quality.consistency_score,
                "validity_score": metadata.data_quality.validity_score,
                "uniqueness_score": metadata.data_quality.uniqueness_score,
                "overall_score": metadata.data_quality.overall_score,
                "issues": metadata.data_quality.issues,
                "recommendations": metadata.data_quality.recommendations,
                "regulatory_compliance": metadata.data_quality.regulatory_compliance
            },
            "feature_statistics": {},
            "correlation_matrix": {},
            "target_analysis": {}
        }
        
        # Feature statistics
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                profile["feature_statistics"][col] = {
                    "type": "numeric",
                    "count": int(data[col].count()),
                    "mean": float(data[col].mean()),
                    "std": float(data[col].std()),
                    "min": float(data[col].min()),
                    "max": float(data[col].max()),
                    "missing_count": int(data[col].isnull().sum()),
                    "missing_percentage": float(data[col].isnull().sum() / len(data) * 100)
                }
            else:
                profile["feature_statistics"][col] = {
                    "type": "categorical",
                    "count": int(data[col].count()),
                    "unique_values": int(data[col].nunique()),
                    "most_frequent": str(data[col].mode().iloc[0]) if not data[col].empty else None,
                    "missing_count": int(data[col].isnull().sum()),
                    "missing_percentage": float(data[col].isnull().sum() / len(data) * 100)
                }
        
        # Target analysis
        if 'default_flag' in data.columns:
            target_stats = data['default_flag'].value_counts()
            profile["target_analysis"] = {
                "default_rate": float(data['default_flag'].mean()),
                "default_count": int(target_stats.get(1, 0)),
                "non_default_count": int(target_stats.get(0, 0)),
                "class_balance": float(min(target_stats) / max(target_stats)) if len(target_stats) > 1 else 1.0
            }
        
        # Save profile
        with open(output_path, 'w') as f:
            json.dump(profile, f, indent=2, default=str)
        
        logger.info(f"Data profile exported to {output_path}")
        return profile