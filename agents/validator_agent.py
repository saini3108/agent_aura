import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Dict, Any, List, Tuple
import json
from datetime import datetime

class ValidatorAgent:
    """Agent responsible for calculating validation metrics"""
    
    def __init__(self):
        self.name = "Validator Agent"
        self.description = "Calculates validation metrics including AUC, KS test, and drift detection"
    
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for the validator agent
        
        Args:
            context: Dictionary containing data, files, and previous outputs
            
        Returns:
            Dictionary containing validation results
        """
        try:
            data = context.get('data')
            previous_outputs = context.get('previous_outputs', {})
            
            validation_results = {
                'timestamp': datetime.now().isoformat(),
                'agent': self.name,
                'status': 'completed',
                'metrics': {}
            }
            
            if data is None:
                validation_results['status'] = 'error'
                validation_results['error'] = 'No data available for validation'
                return validation_results
            
            # Prepare data for validation
            X, y = self._prepare_data(data)
            
            if X is None or y is None:
                validation_results['status'] = 'error'
                validation_results['error'] = 'Could not prepare data for validation'
                return validation_results
            
            # Calculate validation metrics
            metrics = self._calculate_metrics(X, y)
            validation_results['metrics'] = metrics
            
            # Generate validation summary
            summary = self._generate_validation_summary(metrics)
            validation_results['summary'] = summary
            
            return validation_results
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'agent': self.name,
                'status': 'error',
                'error': str(e)
            }
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for validation"""
        try:
            # Look for target variable
            target_col = None
            for col in data.columns:
                if any(keyword in col.lower() for keyword in ['target', 'default', 'pd', 'flag', 'label']):
                    target_col = col
                    break
            
            # If no target found, create a synthetic one for demonstration
            if target_col is None:
                # Create synthetic target based on first numeric column
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    first_numeric = numeric_cols[0]
                    # Create binary target based on median split
                    median_val = data[first_numeric].median()
                    y = (data[first_numeric] > median_val).astype(int)
                    target_col = 'synthetic_target'
                else:
                    return None, None
            else:
                y = data[target_col]
            
            # Prepare features
            feature_cols = [col for col in data.columns if col != target_col]
            X = data[feature_cols]
            
            # Handle only numeric features for simplicity
            numeric_features = X.select_dtypes(include=[np.number])
            if len(numeric_features.columns) == 0:
                return None, None
            
            # Fill missing values
            X_numeric = numeric_features.fillna(numeric_features.median())
            
            return X_numeric, y
            
        except Exception as e:
            print(f"Error in data preparation: {e}")
            return None, None
    
    def _calculate_metrics(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Calculate validation metrics"""
        metrics = {}
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train a simple logistic regression model
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train_scaled, y_train)
            
            # Get predictions
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = model.predict(X_test_scaled)
            
            # Calculate AUC
            auc = roc_auc_score(y_test, y_pred_proba)
            metrics['auc'] = auc
            
            # Calculate ROC curve data
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            metrics['roc_data'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
            
            # Calculate KS statistic
            ks_stat = self._calculate_ks_statistic(y_test, y_pred_proba)
            metrics['ks_statistic'] = ks_stat
            
            # Calculate Population Stability Index (PSI)
            psi = self._calculate_psi(X_train.iloc[:, 0], X_test.iloc[:, 0])
            metrics['psi'] = psi
            
            # Calculate Gini coefficient
            gini = 2 * auc - 1
            metrics['gini'] = gini
            
            # Feature importance
            feature_importance = dict(zip(X.columns, model.coef_[0]))
            metrics['feature_importance'] = feature_importance
            
            # Score distribution
            metrics['score_distribution'] = {
                'train_scores': model.predict_proba(X_train_scaled)[:, 1].tolist(),
                'test_scores': y_pred_proba.tolist()
            }
            
            # Basic statistics
            metrics['basic_stats'] = {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'positive_rate_train': y_train.mean(),
                'positive_rate_test': y_test.mean(),
                'feature_count': X.shape[1]
            }
            
        except Exception as e:
            metrics['error'] = f"Error calculating metrics: {str(e)}"
        
        return metrics
    
    def _calculate_ks_statistic(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
        """Calculate Kolmogorov-Smirnov statistic"""
        try:
            # Separate scores by class
            scores_pos = y_pred_proba[y_true == 1]
            scores_neg = y_pred_proba[y_true == 0]
            
            # Calculate KS statistic
            ks_stat, _ = stats.ks_2samp(scores_pos, scores_neg)
            return ks_stat
            
        except Exception:
            return 0.0
    
    def _calculate_psi(self, expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        try:
            # Create bins based on expected distribution
            _, bin_edges = np.histogram(expected, bins=bins)
            
            # Calculate expected and actual distributions
            expected_dist = np.histogram(expected, bins=bin_edges)[0]
            actual_dist = np.histogram(actual, bins=bin_edges)[0]
            
            # Convert to percentages
            expected_pct = expected_dist / len(expected)
            actual_pct = actual_dist / len(actual)
            
            # Avoid division by zero
            expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
            actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
            
            # Calculate PSI
            psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
            return psi
            
        except Exception:
            return 0.0
    
    def _generate_validation_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation summary with interpretations"""
        summary = {
            'overall_performance': 'Unknown',
            'key_findings': [],
            'recommendations': [],
            'risk_flags': []
        }
        
        # Evaluate AUC
        auc = metrics.get('auc', 0)
        if auc >= 0.8:
            summary['overall_performance'] = 'Excellent'
            summary['key_findings'].append(f"Excellent discrimination with AUC of {auc:.3f}")
        elif auc >= 0.7:
            summary['overall_performance'] = 'Good'
            summary['key_findings'].append(f"Good discrimination with AUC of {auc:.3f}")
        elif auc >= 0.6:
            summary['overall_performance'] = 'Acceptable'
            summary['key_findings'].append(f"Acceptable discrimination with AUC of {auc:.3f}")
            summary['recommendations'].append("Consider feature engineering to improve model performance")
        else:
            summary['overall_performance'] = 'Poor'
            summary['key_findings'].append(f"Poor discrimination with AUC of {auc:.3f}")
            summary['risk_flags'].append("Model shows poor discriminatory power")
        
        # Evaluate KS statistic
        ks_stat = metrics.get('ks_statistic', 0)
        if ks_stat >= 0.3:
            summary['key_findings'].append(f"Strong separation with KS statistic of {ks_stat:.3f}")
        elif ks_stat >= 0.2:
            summary['key_findings'].append(f"Adequate separation with KS statistic of {ks_stat:.3f}")
        else:
            summary['key_findings'].append(f"Weak separation with KS statistic of {ks_stat:.3f}")
            summary['risk_flags'].append("KS statistic indicates weak model separation")
        
        # Evaluate PSI
        psi = metrics.get('psi', 0)
        if psi <= 0.1:
            summary['key_findings'].append(f"Stable population with PSI of {psi:.3f}")
        elif psi <= 0.25:
            summary['key_findings'].append(f"Moderate population shift with PSI of {psi:.3f}")
            summary['recommendations'].append("Monitor population stability closely")
        else:
            summary['key_findings'].append(f"Significant population shift with PSI of {psi:.3f}")
            summary['risk_flags'].append("High population instability detected")
        
        return summary
