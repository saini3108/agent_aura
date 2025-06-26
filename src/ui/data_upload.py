"""
Data Upload Interface
====================

Clean, dynamic data upload interface with validation and preprocessing
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import io
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataUploadManager:
    """Manages data upload with dynamic validation and processing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.supported_formats = {
            'csv': self._read_csv,
            'xlsx': self._read_excel,
            'json': self._read_json,
            'txt': self._read_text
        }
        self.validation_rules = self.config.get('validation_rules', {})
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'max_file_size': 200,  # MB
            'required_columns': [],
            'optional_columns': [],
            'validation_rules': {
                'min_rows': 10,
                'max_rows': 100000,
                'required_dtypes': {}
            },
            'preprocessing': {
                'handle_missing': 'auto',
                'normalize_columns': True,
                'validate_credit_data': True
            }
        }
    
    def show_upload_interface(self) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """Display upload interface and return data"""
        st.subheader("📊 Data Upload")
        
        upload_method = st.radio(
            "Choose upload method:",
            ["Upload File", "Use Sample Data", "Connect Database"],
            horizontal=True
        )
        
        if upload_method == "Upload File":
            return self._handle_file_upload()
        elif upload_method == "Use Sample Data":
            return self._handle_sample_data()
        else:
            return self._handle_database_connection()
    
    def _handle_file_upload(self) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """Handle file upload"""
        uploaded_file = st.file_uploader(
            "Upload your credit data file",
            type=['csv', 'xlsx', 'json', 'txt'],
            help=f"Maximum file size: {self.config.get('max_file_size_mb', 200)}MB"
        )
        
        metadata = {"source": "file_upload", "processed_at": pd.Timestamp.now()}
        
        if uploaded_file is not None:
            try:
                # Check file size
                if uploaded_file.size > self.config.get('max_file_size_mb', 200) * 1024 * 1024:
                    st.error(f"File too large. Maximum size: {self.config.get('max_file_size_mb', 200)}MB")
                    return None, metadata
                
                # Determine file type and read
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                if file_extension in self.supported_formats:
                    data = self.supported_formats[file_extension](uploaded_file)
                    metadata.update({
                        "filename": uploaded_file.name,
                        "file_type": file_extension,
                        "file_size": uploaded_file.size
                    })
                    
                    # Validate and process
                    processed_data, validation_results = self._validate_and_process(data)
                    metadata["validation"] = validation_results
                    
                    # Display upload results
                    self._display_upload_results(processed_data, metadata)
                    
                    return processed_data, metadata
                else:
                    st.error(f"Unsupported file format: {file_extension}")
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                logger.error(f"File upload error: {e}")
        
        return None, metadata
    
    def _handle_sample_data(self) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """Handle sample data selection"""
        sample_options = {
            "Credit Portfolio (50 records)": "credit_data.csv",
            "Risk Metrics Dataset": "risk_thresholds.csv", 
            "Validation Parameters": "validation_parameters.csv"
        }
        
        selected_sample = st.selectbox("Choose sample dataset:", list(sample_options.keys()))
        
        if st.button("Load Sample Data"):
            try:
                sample_file = sample_options[selected_sample]
                data_path = Path("sample_data") / sample_file
                
                if data_path.exists():
                    data = pd.read_csv(data_path)
                    metadata = {
                        "source": "sample_data",
                        "filename": sample_file,
                        "processed_at": pd.Timestamp.now()
                    }
                    
                    # Process sample data
                    processed_data, validation_results = self._validate_and_process(data)
                    metadata["validation"] = validation_results
                    
                    self._display_upload_results(processed_data, metadata)
                    return processed_data, metadata
                else:
                    st.error(f"Sample file not found: {sample_file}")
                    
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
                logger.error(f"Sample data error: {e}")
        
        return None, {"source": "sample_data"}
    
    def _handle_database_connection(self) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """Handle database connections"""
        st.info("Database connection feature coming soon")
        
        # Future implementation for database connections
        db_type = st.selectbox("Database Type:", ["PostgreSQL", "MySQL", "SQLite", "SQL Server"])
        
        col1, col2 = st.columns(2)
        with col1:
            host = st.text_input("Host")
            database = st.text_input("Database")
        with col2:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
        
        query = st.text_area("SQL Query", placeholder="SELECT * FROM credit_data WHERE...")
        
        if st.button("Connect & Query"):
            st.info("Database connectivity will be implemented in next version")
        
        return None, {"source": "database"}
    
    def _read_csv(self, file) -> pd.DataFrame:
        """Read CSV file with smart detection"""
        # Try different encodings and separators
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            for sep in [',', ';', '\t']:
                try:
                    file.seek(0)
                    return pd.read_csv(file, encoding=encoding, sep=sep)
                except:
                    continue
        
        # Fallback
        file.seek(0)
        return pd.read_csv(file)
    
    def _read_excel(self, file) -> pd.DataFrame:
        """Read Excel file"""
        return pd.read_excel(file)
    
    def _read_json(self, file) -> pd.DataFrame:
        """Read JSON file"""
        data = json.load(file)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        else:
            raise ValueError("Unsupported JSON format")
    
    def _read_text(self, file) -> pd.DataFrame:
        """Read text file and attempt to parse"""
        content = file.read().decode('utf-8')
        # Try to parse as CSV-like format
        lines = content.strip().split('\n')
        
        # Simple heuristic to detect delimiter
        first_line = lines[0]
        delimiters = [',', '\t', ';', '|']
        delimiter = ','
        
        for delim in delimiters:
            if delim in first_line:
                delimiter = delim
                break
        
        # Create DataFrame
        data = []
        headers = lines[0].split(delimiter)
        
        for line in lines[1:]:
            row = line.split(delimiter)
            if len(row) == len(headers):
                data.append(row)
        
        return pd.DataFrame(data, columns=headers)
    
    def _validate_and_process(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate and process uploaded data"""
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "info": []
        }
        
        # Basic validation
        if len(data) < self.validation_rules.get('min_rows', 10):
            validation_results["errors"].append(f"Insufficient data: {len(data)} rows (minimum: {self.validation_rules['min_rows']})")
            validation_results["is_valid"] = False
        
        if len(data) > self.validation_rules.get('max_rows', 100000):
            validation_results["warnings"].append(f"Large dataset: {len(data)} rows. Processing may be slow.")
        
        # Column validation
        required_cols = self.config.get('required_columns', [])
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            validation_results["warnings"].append(f"Missing recommended columns: {missing_cols}")
        
        # Data quality checks
        missing_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        if missing_percentage > 20:
            validation_results["warnings"].append(f"High missing data: {missing_percentage:.1f}%")
        
        # Credit data specific validation
        if self.config.get('preprocessing', {}).get('validate_credit_data', True):
            credit_validation = self._validate_credit_data(data)
            validation_results.update(credit_validation)
        
        # Process data
        processed_data = self._preprocess_data(data)
        
        validation_results["info"].append(f"Successfully processed {len(processed_data)} records with {len(processed_data.columns)} features")
        
        return processed_data, validation_results
    
    def _validate_credit_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate credit-specific data patterns"""
        validation = {"credit_warnings": [], "credit_info": []}
        
        # Look for typical credit columns
        credit_columns = {
            'target': ['default', 'target', 'bad', 'delinquent'],
            'score': ['credit_score', 'score', 'rating'],
            'amount': ['loan_amount', 'exposure', 'balance'],
            'income': ['income', 'salary', 'annual_income']
        }
        
        found_columns = {}
        for category, possible_names in credit_columns.items():
            found = [col for col in data.columns if any(name.lower() in col.lower() for name in possible_names)]
            if found:
                found_columns[category] = found[0]
                validation["credit_info"].append(f"Identified {category} column: {found[0]}")
        
        # Validate target variable
        if 'target' in found_columns:
            target_col = found_columns['target']
            unique_values = data[target_col].nunique()
            if unique_values == 2:
                validation["credit_info"].append("Binary target variable detected (good for classification)")
            elif unique_values > 10:
                validation["credit_warnings"].append("Target variable has many unique values - may need preprocessing")
        
        # Validate numeric ranges
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].min() < 0 and any(keyword in col.lower() for keyword in ['score', 'amount', 'balance']):
                validation["credit_warnings"].append(f"Negative values in {col} - may need review")
        
        return validation
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data based on configuration"""
        processed = data.copy()
        
        preprocessing_config = self.config.get('preprocessing', {})
        
        # Handle missing values
        if preprocessing_config.get('handle_missing') == 'auto':
            # Numeric columns: fill with median
            numeric_cols = processed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if processed[col].isnull().any():
                    processed[col].fillna(processed[col].median(), inplace=True)
            
            # Categorical columns: fill with mode
            categorical_cols = processed.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if processed[col].isnull().any():
                    mode_val = processed[col].mode().iloc[0] if len(processed[col].mode()) > 0 else 'Unknown'
                    processed[col].fillna(mode_val, inplace=True)
        
        # Normalize column names
        if preprocessing_config.get('normalize_columns', True):
            processed.columns = [col.lower().strip().replace(' ', '_') for col in processed.columns]
        
        return processed
    
    def _display_upload_results(self, data: pd.DataFrame, metadata: Dict[str, Any]):
        """Display upload results and data preview"""
        validation = metadata.get("validation", {})
        
        # Status indicators
        if validation.get("is_valid", True):
            st.success("✅ Data uploaded and validated successfully")
        else:
            st.error("❌ Data validation failed")
        
        # Display validation messages
        if validation.get("errors"):
            st.error("Errors:")
            for error in validation["errors"]:
                st.write(f"• {error}")
        
        if validation.get("warnings"):
            st.warning("Warnings:")
            for warning in validation["warnings"]:
                st.write(f"• {warning}")
        
        if validation.get("info"):
            st.info("Information:")
            for info in validation["info"]:
                st.write(f"• {info}")
        
        # Data summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", len(data))
        with col2:
            st.metric("Columns", len(data.columns))
        with col3:
            missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            st.metric("Missing %", f"{missing_pct:.1f}%")
        with col4:
            numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Cols", numeric_cols)
        
        # Data preview
        with st.expander("📋 Data Preview", expanded=True):
            st.dataframe(data.head(10))
        
        # Column info
        with st.expander("📊 Column Information"):
            col_info = pd.DataFrame({
                'Column': data.columns,
                'Type': data.dtypes,
                'Non-Null': data.count(),
                'Null %': ((len(data) - data.count()) / len(data) * 100).round(1)
            })
            st.dataframe(col_info)

def show_data_upload_interface(config: Dict[str, Any] = None) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Main interface function"""
    upload_manager = DataUploadManager(config)
    return upload_manager.show_upload_interface()