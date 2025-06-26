import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json
from datetime import datetime

class AnalystAgent:
    """Agent responsible for analyzing model structure and parameters"""
    
    def __init__(self):
        self.name = "Analyst Agent"
        self.description = "Analyzes model structure, parameters, and data characteristics"
    
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for the analyst agent
        
        Args:
            context: Dictionary containing data, files, and previous outputs
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            data = context.get('data')
            files = context.get('files', {})
            
            analysis_results = {
                'timestamp': datetime.now().isoformat(),
                'agent': self.name,
                'status': 'completed',
                'analysis': {}
            }
            
            # Analyze data if available
            if data is not None:
                data_analysis = self._analyze_data(data)
                analysis_results['analysis']['data_analysis'] = data_analysis
            
            # Analyze uploaded files
            if files:
                file_analysis = self._analyze_files(files)
                analysis_results['analysis']['file_analysis'] = file_analysis
            
            # Model structure analysis
            model_analysis = self._analyze_model_structure(data)
            analysis_results['analysis']['model_analysis'] = model_analysis
            
            # Generate recommendations
            recommendations = self._generate_recommendations(analysis_results['analysis'])
            analysis_results['recommendations'] = recommendations
            
            return analysis_results
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'agent': self.name,
                'status': 'error',
                'error': str(e)
            }
    
    def _analyze_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the dataset characteristics"""
        analysis = {
            'shape': data.shape,
            'columns': data.columns.tolist(),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'summary_stats': {}
        }
        
        # Calculate summary statistics for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis['summary_stats'] = data[numeric_cols].describe().to_dict()
        
        # Identify potential target variable
        target_candidates = []
        for col in data.columns:
            if any(keyword in col.lower() for keyword in ['target', 'default', 'pd', 'flag', 'label']):
                target_candidates.append(col)
        
        analysis['target_candidates'] = target_candidates
        
        # Check for categorical variables
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        analysis['categorical_columns'] = categorical_cols
        
        return analysis
    
    def _analyze_files(self, files: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze uploaded documentation files"""
        file_analysis = {
            'total_files': len(files),
            'file_types': {},
            'file_sizes': {},
            'upload_timestamps': {}
        }
        
        for filename, file_info in files.items():
            # Extract file extension
            file_ext = filename.split('.')[-1].lower() if '.' in filename else 'unknown'
            
            if file_ext not in file_analysis['file_types']:
                file_analysis['file_types'][file_ext] = 0
            file_analysis['file_types'][file_ext] += 1
            
            file_analysis['file_sizes'][filename] = file_info.get('size', 0)
            file_analysis['upload_timestamps'][filename] = file_info.get('uploaded_at', '')
        
        return file_analysis
    
    def _analyze_model_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze potential model structure and features"""
        if data is None:
            return {'status': 'no_data', 'message': 'No data available for model analysis'}
        
        model_analysis = {
            'feature_count': data.shape[1],
            'sample_size': data.shape[0],
            'data_quality_score': 0.0,
            'feature_types': {},
            'potential_issues': []
        }
        
        # Analyze feature types
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = data.select_dtypes(include=['object']).columns.tolist()
        
        model_analysis['feature_types'] = {
            'numeric': len(numeric_features),
            'categorical': len(categorical_features)
        }
        
        # Calculate data quality score
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        completeness_score = (total_cells - missing_cells) / total_cells
        
        # Check for duplicate records
        duplicate_rate = data.duplicated().sum() / len(data)
        
        model_analysis['data_quality_score'] = completeness_score * (1 - duplicate_rate)
        
        # Identify potential issues
        if completeness_score < 0.95:
            model_analysis['potential_issues'].append(f"High missing data rate: {(1-completeness_score)*100:.1f}%")
        
        if duplicate_rate > 0.05:
            model_analysis['potential_issues'].append(f"High duplicate rate: {duplicate_rate*100:.1f}%")
        
        if len(numeric_features) < 5:
            model_analysis['potential_issues'].append("Limited numeric features for model building")
        
        return model_analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Data quality recommendations
        data_analysis = analysis.get('data_analysis', {})
        if data_analysis:
            missing_values = data_analysis.get('missing_values', {})
            total_missing = sum(missing_values.values())
            
            if total_missing > 0:
                recommendations.append(f"Address {total_missing} missing values across the dataset")
            
            # Feature engineering recommendations
            categorical_cols = data_analysis.get('categorical_columns', [])
            if len(categorical_cols) > 5:
                recommendations.append("Consider feature selection for categorical variables")
        
        # Model analysis recommendations
        model_analysis = analysis.get('model_analysis', {})
        if model_analysis:
            quality_score = model_analysis.get('data_quality_score', 0)
            if quality_score < 0.8:
                recommendations.append("Improve data quality before model validation")
            
            potential_issues = model_analysis.get('potential_issues', [])
            for issue in potential_issues:
                recommendations.append(f"Review: {issue}")
        
        # File analysis recommendations
        file_analysis = analysis.get('file_analysis', {})
        if file_analysis:
            if file_analysis.get('total_files', 0) == 0:
                recommendations.append("Upload model documentation for comprehensive validation")
            
            file_types = file_analysis.get('file_types', {})
            if 'pdf' not in file_types:
                recommendations.append("Consider uploading model documentation in PDF format")
        
        return recommendations
