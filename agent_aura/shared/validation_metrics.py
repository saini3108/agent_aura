import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy import stats
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ValidationMetrics:
    """Utility class for calculating validation metrics"""
    
    def __init__(self):
        self.metrics_calculated = {}
    
    def calculate_auc(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Calculate Area Under the ROC Curve"""
        try:
            return roc_auc_score(y_true, y_score)
        except Exception as e:
            print(f"Error calculating AUC: {e}")
            return 0.0
    
    def calculate_roc_curve(self, y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, List[float]]:
        """Calculate ROC curve data"""
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            return {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
        except Exception as e:
            print(f"Error calculating ROC curve: {e}")
            return {'fpr': [], 'tpr': [], 'thresholds': []}
    
    def calculate_ks_statistic(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Calculate Kolmogorov-Smirnov statistic"""
        try:
            # Split scores by class
            scores_pos = y_score[y_true == 1]
            scores_neg = y_score[y_true == 0]
            
            # Calculate KS statistic
            ks_stat, p_value = stats.ks_2samp(scores_pos, scores_neg)
            return ks_stat
        except Exception as e:
            print(f"Error calculating KS statistic: {e}")
            return 0.0
    
    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        try:
            # Handle edge cases
            if len(expected) == 0 or len(actual) == 0:
                return 0.0
            
            # Create bins based on expected distribution
            bin_edges = np.histogram_bin_edges(expected, bins=bins)
            
            # Calculate expected and actual distributions
            expected_counts, _ = np.histogram(expected, bins=bin_edges)
            actual_counts, _ = np.histogram(actual, bins=bin_edges)
            
            # Convert to percentages
            expected_pct = expected_counts / len(expected)
            actual_pct = actual_counts / len(actual)
            
            # Avoid division by zero by adding small epsilon
            epsilon = 1e-10
            expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
            actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)
            
            # Calculate PSI
            psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
            return psi
        except Exception as e:
            print(f"Error calculating PSI: {e}")
            return 0.0
    
    def calculate_gini_coefficient(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Calculate Gini coefficient"""
        try:
            auc = self.calculate_auc(y_true, y_score)
            return 2 * auc - 1
        except Exception as e:
            print(f"Error calculating Gini coefficient: {e}")
            return 0.0
    
    def calculate_iv(self, X: pd.DataFrame, y: pd.Series, bins: int = 10) -> Dict[str, float]:
        """Calculate Information Value for each feature"""
        iv_dict = {}
        
        for column in X.columns:
            try:
                # Skip non-numeric columns
                if not pd.api.types.is_numeric_dtype(X[column]):
                    continue
                
                # Create bins
                X_binned, bin_edges = pd.cut(X[column], bins=bins, retbins=True, duplicates='drop')
                
                # Calculate IV
                iv_table = pd.DataFrame()
                iv_table['Variable'] = X_binned
                iv_table['Target'] = y
                
                # Group by bins
                grouped = iv_table.groupby('Variable')['Target'].agg(['count', 'sum'])
                grouped.columns = ['Total', 'Bad']
                grouped['Good'] = grouped['Total'] - grouped['Bad']
                
                # Calculate rates
                total_good = grouped['Good'].sum()
                total_bad = grouped['Bad'].sum()
                
                if total_good == 0 or total_bad == 0:
                    iv_dict[column] = 0.0
                    continue
                
                grouped['Good_Rate'] = grouped['Good'] / total_good
                grouped['Bad_Rate'] = grouped['Bad'] / total_bad
                
                # Avoid division by zero
                grouped['Good_Rate'] = np.where(grouped['Good_Rate'] == 0, 1e-10, grouped['Good_Rate'])
                grouped['Bad_Rate'] = np.where(grouped['Bad_Rate'] == 0, 1e-10, grouped['Bad_Rate'])
                
                # Calculate WoE and IV
                grouped['WoE'] = np.log(grouped['Good_Rate'] / grouped['Bad_Rate'])
                grouped['IV'] = (grouped['Good_Rate'] - grouped['Bad_Rate']) * grouped['WoE']
                
                iv_dict[column] = grouped['IV'].sum()
                
            except Exception as e:
                print(f"Error calculating IV for {column}: {e}")
                iv_dict[column] = 0.0
        
        return iv_dict
    
    def calculate_lift_curve(self, y_true: np.ndarray, y_score: np.ndarray, 
                           bins: int = 10) -> Dict[str, List[float]]:
        """Calculate lift curve data"""
        try:
            # Create a DataFrame for easier manipulation
            df = pd.DataFrame({
                'score': y_score,
                'target': y_true
            })
            
            # Sort by score descending
            df = df.sort_values('score', ascending=False)
            
            # Create deciles
            df['decile'] = pd.qcut(df['score'], q=bins, labels=False, duplicates='drop')
            
            # Calculate lift
            lift_data = df.groupby('decile').agg({
                'target': ['count', 'sum']
            }).reset_index()
            
            lift_data.columns = ['decile', 'total', 'positive']
            lift_data['positive_rate'] = lift_data['positive'] / lift_data['total']
            
            # Calculate overall positive rate
            overall_positive_rate = df['target'].mean()
            
            # Calculate lift
            lift_data['lift'] = lift_data['positive_rate'] / overall_positive_rate
            
            return {
                'deciles': lift_data['decile'].tolist(),
                'lift': lift_data['lift'].tolist(),
                'positive_rate': lift_data['positive_rate'].tolist()
            }
        except Exception as e:
            print(f"Error calculating lift curve: {e}")
            return {'deciles': [], 'lift': [], 'positive_rate': []}
    
    def calculate_confusion_matrix_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate metrics from confusion matrix"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                
                # Calculate metrics
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                return {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'specificity': specificity,
                    'f1_score': f1_score,
                    'true_positives': int(tp),
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn)
                }
            else:
                return {'error': 'Invalid confusion matrix shape'}
        except Exception as e:
            print(f"Error calculating confusion matrix metrics: {e}")
            return {'error': str(e)}
    
    def calculate_stability_metrics(self, development_data: pd.DataFrame, 
                                   validation_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate stability metrics between development and validation datasets"""
        stability_metrics = {}
        
        try:
            # Calculate PSI for each numeric feature
            for column in development_data.columns:
                if pd.api.types.is_numeric_dtype(development_data[column]):
                    if column in validation_data.columns:
                        dev_values = development_data[column].dropna()
                        val_values = validation_data[column].dropna()
                        
                        if len(dev_values) > 0 and len(val_values) > 0:
                            psi = self.calculate_psi(dev_values, val_values)
                            stability_metrics[f'{column}_psi'] = psi
            
            # Overall stability assessment
            psi_values = [v for k, v in stability_metrics.items() if k.endswith('_psi')]
            if psi_values:
                stability_metrics['average_psi'] = np.mean(psi_values)
                stability_metrics['max_psi'] = np.max(psi_values)
                stability_metrics['min_psi'] = np.min(psi_values)
                
                # Stability categorization
                max_psi = stability_metrics['max_psi']
                if max_psi <= 0.1:
                    stability_metrics['stability_status'] = 'Stable'
                elif max_psi <= 0.25:
                    stability_metrics['stability_status'] = 'Moderate Shift'
                else:
                    stability_metrics['stability_status'] = 'Significant Shift'
            
        except Exception as e:
            print(f"Error calculating stability metrics: {e}")
            stability_metrics['error'] = str(e)
        
        return stability_metrics
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_score: np.ndarray, 
                                      y_pred: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate comprehensive set of validation metrics"""
        metrics = {}
        
        try:
            # Discrimination metrics
            metrics['auc'] = self.calculate_auc(y_true, y_score)
            metrics['gini'] = self.calculate_gini_coefficient(y_true, y_score)
            metrics['ks_statistic'] = self.calculate_ks_statistic(y_true, y_score)
            
            # ROC curve data
            metrics['roc_curve'] = self.calculate_roc_curve(y_true, y_score)
            
            # Lift curve data
            metrics['lift_curve'] = self.calculate_lift_curve(y_true, y_score)
            
            # If predictions are provided, calculate classification metrics
            if y_pred is not None:
                classification_metrics = self.calculate_confusion_matrix_metrics(y_true, y_pred)
                metrics.update(classification_metrics)
            
            # Score distribution statistics
            score_stats = {
                'score_mean': float(np.mean(y_score)),
                'score_std': float(np.std(y_score)),
                'score_min': float(np.min(y_score)),
                'score_max': float(np.max(y_score)),
                'score_median': float(np.median(y_score))
            }
            metrics['score_statistics'] = score_stats
            
            # Rank order statistics
            sorted_indices = np.argsort(y_score)[::-1]  # Descending order
            sorted_targets = y_true[sorted_indices]
            
            # Calculate cumulative capture rate
            cumulative_targets = np.cumsum(sorted_targets)
            total_targets = np.sum(y_true)
            
            if total_targets > 0:
                capture_rates = cumulative_targets / total_targets
                # Get capture rates at specific percentiles
                n_samples = len(y_score)
                percentiles = [0.1, 0.2, 0.3, 0.4, 0.5]
                capture_at_percentiles = {}
                
                for p in percentiles:
                    idx = int(p * n_samples)
                    if idx < len(capture_rates):
                        capture_at_percentiles[f'capture_rate_{int(p*100)}pct'] = float(capture_rates[idx])
                
                metrics['capture_rates'] = capture_at_percentiles
            
        except Exception as e:
            print(f"Error calculating comprehensive metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def interpret_metrics(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Provide interpretations for calculated metrics"""
        interpretations = {}
        
        # AUC interpretation
        auc = metrics.get('auc', 0.0)
        if auc >= 0.9:
            interpretations['auc'] = 'Excellent discrimination'
        elif auc >= 0.8:
            interpretations['auc'] = 'Good discrimination'
        elif auc >= 0.7:
            interpretations['auc'] = 'Acceptable discrimination'
        elif auc >= 0.6:
            interpretations['auc'] = 'Poor discrimination'
        else:
            interpretations['auc'] = 'No discrimination'
        
        # Gini interpretation
        gini = metrics.get('gini', 0.0)
        if gini >= 0.8:
            interpretations['gini'] = 'Excellent model power'
        elif gini >= 0.6:
            interpretations['gini'] = 'Good model power'
        elif gini >= 0.4:
            interpretations['gini'] = 'Acceptable model power'
        elif gini >= 0.2:
            interpretations['gini'] = 'Poor model power'
        else:
            interpretations['gini'] = 'No model power'
        
        # KS interpretation
        ks = metrics.get('ks_statistic', 0.0)
        if ks >= 0.4:
            interpretations['ks'] = 'Excellent separation'
        elif ks >= 0.3:
            interpretations['ks'] = 'Good separation'
        elif ks >= 0.2:
            interpretations['ks'] = 'Acceptable separation'
        elif ks >= 0.1:
            interpretations['ks'] = 'Poor separation'
        else:
            interpretations['ks'] = 'No separation'
        
        return interpretations
