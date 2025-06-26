import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json
from datetime import datetime

class ReviewerAgent:
    """Agent responsible for generating findings and recommendations"""
    
    def __init__(self):
        self.name = "Reviewer Agent"
        self.description = "Generates comprehensive findings and recommendations based on validation results"
        
        # Define risk thresholds
        self.risk_thresholds = {
            'auc': {'excellent': 0.8, 'good': 0.7, 'acceptable': 0.6},
            'ks': {'excellent': 0.3, 'good': 0.2, 'acceptable': 0.1},
            'psi': {'stable': 0.1, 'moderate': 0.25, 'unstable': 0.5},
            'gini': {'excellent': 0.6, 'good': 0.4, 'acceptable': 0.2}
        }
    
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for the reviewer agent
        
        Args:
            context: Dictionary containing data, files, and previous outputs
            
        Returns:
            Dictionary containing review findings and recommendations
        """
        try:
            previous_outputs = context.get('previous_outputs', {})
            
            review_results = {
                'timestamp': datetime.now().isoformat(),
                'agent': self.name,
                'status': 'completed',
                'findings': {},
                'recommendations': {},
                'risk_assessment': {}
            }
            
            # Extract results from previous agents
            analyst_results = previous_outputs.get('step_0', {})
            validator_results = previous_outputs.get('step_1', {})
            documentation_results = previous_outputs.get('step_2', {})
            
            # Generate findings based on each agent's output
            findings = self._generate_findings(analyst_results, validator_results, documentation_results)
            review_results['findings'] = findings
            
            # Generate recommendations
            recommendations = self._generate_recommendations(findings)
            review_results['recommendations'] = recommendations
            
            # Perform risk assessment
            risk_assessment = self._perform_risk_assessment(findings)
            review_results['risk_assessment'] = risk_assessment
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(findings, recommendations, risk_assessment)
            review_results['executive_summary'] = executive_summary
            
            return review_results
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'agent': self.name,
                'status': 'error',
                'error': str(e)
            }
    
    def _generate_findings(self, analyst_results: Dict[str, Any], 
                          validator_results: Dict[str, Any], 
                          documentation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive findings from all agent outputs"""
        findings = {
            'data_quality_findings': [],
            'model_performance_findings': [],
            'documentation_findings': [],
            'compliance_findings': [],
            'overall_findings': []
        }
        
        # Data quality findings from analyst
        if analyst_results.get('status') == 'completed':
            analysis = analyst_results.get('analysis', {})
            data_analysis = analysis.get('data_analysis', {})
            model_analysis = analysis.get('model_analysis', {})
            
            # Data quality issues
            missing_values = data_analysis.get('missing_values', {})
            total_missing = sum(missing_values.values()) if missing_values else 0
            
            if total_missing > 0:
                findings['data_quality_findings'].append({
                    'issue': 'Missing Data',
                    'severity': 'Medium' if total_missing < 1000 else 'High',
                    'description': f"Dataset contains {total_missing} missing values",
                    'recommendation': 'Address missing values before model deployment'
                })
            
            # Data quality score
            quality_score = model_analysis.get('data_quality_score', 0.0)
            if quality_score < 0.8:
                findings['data_quality_findings'].append({
                    'issue': 'Data Quality Score',
                    'severity': 'High' if quality_score < 0.6 else 'Medium',
                    'description': f"Data quality score of {quality_score:.2f} is below acceptable threshold",
                    'recommendation': 'Improve data quality before proceeding'
                })
        
        # Model performance findings from validator
        if validator_results.get('status') == 'completed':
            metrics = validator_results.get('metrics', {})
            
            # AUC assessment
            auc = metrics.get('auc', 0.0)
            auc_finding = self._assess_metric('auc', auc)
            if auc_finding:
                findings['model_performance_findings'].append(auc_finding)
            
            # KS statistic assessment
            ks_stat = metrics.get('ks_statistic', 0.0)
            ks_finding = self._assess_metric('ks', ks_stat)
            if ks_finding:
                findings['model_performance_findings'].append(ks_finding)
            
            # PSI assessment
            psi = metrics.get('psi', 0.0)
            psi_finding = self._assess_metric('psi', psi)
            if psi_finding:
                findings['model_performance_findings'].append(psi_finding)
            
            # Gini coefficient assessment
            gini = metrics.get('gini', 0.0)
            gini_finding = self._assess_metric('gini', gini)
            if gini_finding:
                findings['model_performance_findings'].append(gini_finding)
        
        # Documentation findings
        if documentation_results.get('status') == 'completed':
            review_results = documentation_results.get('review_results', {})
            overall_assessment = review_results.get('overall_assessment', {})
            
            # Documentation completeness
            completeness = overall_assessment.get('documentation_completeness', 0.0)
            if completeness < 0.8:
                findings['documentation_findings'].append({
                    'issue': 'Documentation Completeness',
                    'severity': 'High' if completeness < 0.5 else 'Medium',
                    'description': f"Documentation completeness score of {completeness:.2f} is insufficient",
                    'recommendation': 'Provide comprehensive documentation covering all required sections'
                })
            
            # Compliance coverage
            compliance_coverage = overall_assessment.get('compliance_coverage', [])
            if 'Basel III' not in compliance_coverage:
                findings['compliance_findings'].append({
                    'issue': 'Basel III Compliance',
                    'severity': 'High',
                    'description': 'No Basel III compliance documentation identified',
                    'recommendation': 'Provide Basel III compliance documentation'
                })
            
            if 'IFRS 9' not in compliance_coverage:
                findings['compliance_findings'].append({
                    'issue': 'IFRS 9 Compliance',
                    'severity': 'High',
                    'description': 'No IFRS 9 compliance documentation identified',
                    'recommendation': 'Provide IFRS 9 compliance documentation'
                })
        
        # Generate overall findings
        findings['overall_findings'] = self._generate_overall_findings(findings)
        
        return findings
    
    def _assess_metric(self, metric_name: str, value: float) -> Dict[str, Any]:
        """Assess a specific metric against thresholds"""
        if metric_name not in self.risk_thresholds:
            return None
        
        thresholds = self.risk_thresholds[metric_name]
        
        finding = {
            'metric': metric_name.upper(),
            'value': value,
            'severity': 'Low',
            'description': '',
            'recommendation': ''
        }
        
        if metric_name == 'auc':
            if value >= thresholds['excellent']:
                finding['severity'] = 'Info'
                finding['description'] = f"Excellent model discrimination with AUC of {value:.3f}"
                finding['recommendation'] = 'Model shows strong discriminatory power'
            elif value >= thresholds['good']:
                finding['severity'] = 'Low'
                finding['description'] = f"Good model discrimination with AUC of {value:.3f}"
                finding['recommendation'] = 'Model performance is acceptable'
            elif value >= thresholds['acceptable']:
                finding['severity'] = 'Medium'
                finding['description'] = f"Marginal model discrimination with AUC of {value:.3f}"
                finding['recommendation'] = 'Consider model improvements or additional features'
            else:
                finding['severity'] = 'High'
                finding['description'] = f"Poor model discrimination with AUC of {value:.3f}"
                finding['recommendation'] = 'Model requires significant improvement or replacement'
        
        elif metric_name == 'ks':
            if value >= thresholds['excellent']:
                finding['severity'] = 'Info'
                finding['description'] = f"Excellent separation with KS statistic of {value:.3f}"
                finding['recommendation'] = 'Model shows strong separation capability'
            elif value >= thresholds['good']:
                finding['severity'] = 'Low'
                finding['description'] = f"Good separation with KS statistic of {value:.3f}"
                finding['recommendation'] = 'Model separation is acceptable'
            else:
                finding['severity'] = 'Medium'
                finding['description'] = f"Weak separation with KS statistic of {value:.3f}"
                finding['recommendation'] = 'Improve model separation or review feature selection'
        
        elif metric_name == 'psi':
            if value <= thresholds['stable']:
                finding['severity'] = 'Info'
                finding['description'] = f"Stable population with PSI of {value:.3f}"
                finding['recommendation'] = 'Population remains stable'
            elif value <= thresholds['moderate']:
                finding['severity'] = 'Medium'
                finding['description'] = f"Moderate population shift with PSI of {value:.3f}"
                finding['recommendation'] = 'Monitor population stability and consider model updates'
            else:
                finding['severity'] = 'High'
                finding['description'] = f"Significant population shift with PSI of {value:.3f}"
                finding['recommendation'] = 'Review model applicability and consider recalibration'
        
        elif metric_name == 'gini':
            if value >= thresholds['excellent']:
                finding['severity'] = 'Info'
                finding['description'] = f"Excellent Gini coefficient of {value:.3f}"
                finding['recommendation'] = 'Model shows strong predictive power'
            elif value >= thresholds['good']:
                finding['severity'] = 'Low'
                finding['description'] = f"Good Gini coefficient of {value:.3f}"
                finding['recommendation'] = 'Model performance is acceptable'
            else:
                finding['severity'] = 'Medium'
                finding['description'] = f"Marginal Gini coefficient of {value:.3f}"
                finding['recommendation'] = 'Consider model improvements'
        
        return finding
    
    def _generate_overall_findings(self, findings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate overall findings summary"""
        overall_findings = []
        
        # Count findings by severity
        severity_counts = {'High': 0, 'Medium': 0, 'Low': 0, 'Info': 0}
        
        for category, category_findings in findings.items():
            if category != 'overall_findings':
                for finding in category_findings:
                    severity = finding.get('severity', 'Low')
                    severity_counts[severity] += 1
        
        # Generate overall assessment
        if severity_counts['High'] > 0:
            overall_findings.append({
                'issue': 'High Severity Issues',
                'severity': 'Critical',
                'description': f"Identified {severity_counts['High']} high severity issues requiring immediate attention",
                'recommendation': 'Address all high severity issues before model approval'
            })
        
        if severity_counts['Medium'] > 2:
            overall_findings.append({
                'issue': 'Multiple Medium Severity Issues',
                'severity': 'High',
                'description': f"Identified {severity_counts['Medium']} medium severity issues",
                'recommendation': 'Review and address medium severity issues'
            })
        
        if severity_counts['High'] == 0 and severity_counts['Medium'] <= 2:
            overall_findings.append({
                'issue': 'Overall Assessment',
                'severity': 'Info',
                'description': 'Model validation shows acceptable results with minor issues',
                'recommendation': 'Proceed with model approval subject to addressing identified issues'
            })
        
        return overall_findings
    
    def _generate_recommendations(self, findings: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive recommendations"""
        recommendations = {
            'immediate_actions': [],
            'short_term_actions': [],
            'long_term_actions': [],
            'monitoring_requirements': []
        }
        
        # Process findings to generate recommendations
        for category, category_findings in findings.items():
            if category != 'overall_findings':
                for finding in category_findings:
                    severity = finding.get('severity', 'Low')
                    recommendation = finding.get('recommendation', '')
                    
                    if severity == 'High' or severity == 'Critical':
                        recommendations['immediate_actions'].append({
                            'action': recommendation,
                            'category': category,
                            'priority': 'High'
                        })
                    elif severity == 'Medium':
                        recommendations['short_term_actions'].append({
                            'action': recommendation,
                            'category': category,
                            'priority': 'Medium'
                        })
                    else:
                        recommendations['long_term_actions'].append({
                            'action': recommendation,
                            'category': category,
                            'priority': 'Low'
                        })
        
        # Add monitoring requirements
        recommendations['monitoring_requirements'] = [
            'Implement ongoing performance monitoring',
            'Set up population stability monitoring',
            'Establish regular model review schedule',
            'Monitor key performance indicators monthly'
        ]
        
        return recommendations
    
    def _perform_risk_assessment(self, findings: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk assessment"""
        risk_assessment = {
            'overall_risk_level': 'Medium',
            'risk_factors': [],
            'mitigation_strategies': [],
            'risk_score': 0.0
        }
        
        risk_score = 0.0
        risk_factors = []
        
        # Assess risk based on findings
        for category, category_findings in findings.items():
            if category != 'overall_findings':
                for finding in category_findings:
                    severity = finding.get('severity', 'Low')
                    
                    if severity == 'Critical':
                        risk_score += 0.4
                        risk_factors.append(f"Critical issue: {finding.get('issue', 'Unknown')}")
                    elif severity == 'High':
                        risk_score += 0.3
                        risk_factors.append(f"High severity: {finding.get('issue', 'Unknown')}")
                    elif severity == 'Medium':
                        risk_score += 0.1
                        risk_factors.append(f"Medium severity: {finding.get('issue', 'Unknown')}")
        
        # Normalize risk score
        risk_score = min(1.0, risk_score)
        
        # Determine overall risk level
        if risk_score >= 0.7:
            risk_assessment['overall_risk_level'] = 'High'
        elif risk_score >= 0.4:
            risk_assessment['overall_risk_level'] = 'Medium'
        else:
            risk_assessment['overall_risk_level'] = 'Low'
        
        risk_assessment['risk_score'] = risk_score
        risk_assessment['risk_factors'] = risk_factors
        
        # Generate mitigation strategies
        mitigation_strategies = [
            'Implement enhanced monitoring for identified risk areas',
            'Establish regular review cycles for model performance',
            'Develop contingency plans for model performance degradation',
            'Ensure adequate documentation and governance processes'
        ]
        
        if risk_score >= 0.7:
            mitigation_strategies.extend([
                'Consider model replacement or significant enhancement',
                'Implement additional validation checks',
                'Increase monitoring frequency'
            ])
        
        risk_assessment['mitigation_strategies'] = mitigation_strategies
        
        return risk_assessment
    
    def _generate_executive_summary(self, findings: Dict[str, Any], 
                                   recommendations: Dict[str, Any], 
                                   risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary"""
        executive_summary = {
            'validation_status': 'Pending',
            'key_findings': [],
            'critical_actions': [],
            'approval_recommendation': 'Conditional',
            'summary_text': ''
        }
        
        # Determine validation status
        risk_level = risk_assessment.get('overall_risk_level', 'Medium')
        high_severity_count = sum(1 for category_findings in findings.values() 
                                 if category_findings != findings.get('overall_findings', [])
                                 for finding in category_findings 
                                 if finding.get('severity') == 'High')
        
        if risk_level == 'High' or high_severity_count > 2:
            executive_summary['validation_status'] = 'Requires Action'
            executive_summary['approval_recommendation'] = 'Conditional - Address Critical Issues'
        elif risk_level == 'Medium':
            executive_summary['validation_status'] = 'Acceptable with Conditions'
            executive_summary['approval_recommendation'] = 'Conditional - Address Medium Issues'
        else:
            executive_summary['validation_status'] = 'Satisfactory'
            executive_summary['approval_recommendation'] = 'Approved'
        
        # Extract key findings
        key_findings = []
        for category, category_findings in findings.items():
            if category != 'overall_findings':
                for finding in category_findings:
                    if finding.get('severity') in ['High', 'Critical']:
                        key_findings.append(finding.get('description', ''))
        
        executive_summary['key_findings'] = key_findings[:5]  # Top 5 findings
        
        # Extract critical actions
        critical_actions = [action['action'] for action in recommendations.get('immediate_actions', [])]
        executive_summary['critical_actions'] = critical_actions[:3]  # Top 3 actions
        
        # Generate summary text
        summary_text = f"Model validation completed with overall risk level of {risk_level}. "
        summary_text += f"Identified {len(key_findings)} critical findings requiring attention. "
        summary_text += f"Recommendation: {executive_summary['approval_recommendation']}."
        
        executive_summary['summary_text'] = summary_text
        
        return executive_summary
