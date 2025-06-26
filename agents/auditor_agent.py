import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json
from datetime import datetime

class AuditorAgent:
    """Agent responsible for final validation and approval"""
    
    def __init__(self):
        self.name = "Auditor Agent"
        self.description = "Performs final independent validation and approval assessment"
        
        # Define audit criteria
        self.audit_criteria = {
            'model_performance': {
                'auc_threshold': 0.6,
                'ks_threshold': 0.1,
                'psi_threshold': 0.25
            },
            'documentation': {
                'min_completeness': 0.7,
                'required_compliance': ['Basel III', 'IFRS 9']
            },
            'governance': {
                'required_approvals': True,
                'independent_validation': True
            }
        }
    
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for the auditor agent
        
        Args:
            context: Dictionary containing data, files, and previous outputs
            
        Returns:
            Dictionary containing audit results and final recommendation
        """
        try:
            previous_outputs = context.get('previous_outputs', {})
            
            audit_results = {
                'timestamp': datetime.now().isoformat(),
                'agent': self.name,
                'status': 'completed',
                'audit_findings': {},
                'compliance_assessment': {},
                'final_recommendation': {}
            }
            
            # Extract results from previous agents
            analyst_results = previous_outputs.get('step_0', {})
            validator_results = previous_outputs.get('step_1', {})
            documentation_results = previous_outputs.get('step_2', {})
            reviewer_results = previous_outputs.get('step_4', {})
            
            # Perform independent audit
            audit_findings = self._perform_independent_audit(
                analyst_results, validator_results, documentation_results, reviewer_results
            )
            audit_results['audit_findings'] = audit_findings
            
            # Assess compliance
            compliance_assessment = self._assess_compliance(
                analyst_results, validator_results, documentation_results
            )
            audit_results['compliance_assessment'] = compliance_assessment
            
            # Generate final recommendation
            final_recommendation = self._generate_final_recommendation(
                audit_findings, compliance_assessment, reviewer_results
            )
            audit_results['final_recommendation'] = final_recommendation
            
            # Generate audit report
            audit_report = self._generate_audit_report(audit_results)
            audit_results['audit_report'] = audit_report
            
            return audit_results
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'agent': self.name,
                'status': 'error',
                'error': str(e)
            }
    
    def _perform_independent_audit(self, analyst_results: Dict[str, Any], 
                                  validator_results: Dict[str, Any],
                                  documentation_results: Dict[str, Any],
                                  reviewer_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform independent audit of all validation components"""
        audit_findings = {
            'data_quality_audit': {},
            'model_performance_audit': {},
            'documentation_audit': {},
            'process_audit': {},
            'independent_assessment': {}
        }
        
        # Audit data quality
        if analyst_results.get('status') == 'completed':
            data_quality_audit = self._audit_data_quality(analyst_results)
            audit_findings['data_quality_audit'] = data_quality_audit
        
        # Audit model performance
        if validator_results.get('status') == 'completed':
            model_performance_audit = self._audit_model_performance(validator_results)
            audit_findings['model_performance_audit'] = model_performance_audit
        
        # Audit documentation
        if documentation_results.get('status') == 'completed':
            documentation_audit = self._audit_documentation(documentation_results)
            audit_findings['documentation_audit'] = documentation_audit
        
        # Audit process
        process_audit = self._audit_validation_process(
            analyst_results, validator_results, documentation_results, reviewer_results
        )
        audit_findings['process_audit'] = process_audit
        
        # Independent assessment
        independent_assessment = self._perform_independent_assessment(audit_findings)
        audit_findings['independent_assessment'] = independent_assessment
        
        return audit_findings
    
    def _audit_data_quality(self, analyst_results: Dict[str, Any]) -> Dict[str, Any]:
        """Audit data quality aspects"""
        audit = {
            'data_completeness': 'Unknown',
            'data_accuracy': 'Unknown',
            'data_consistency': 'Unknown',
            'issues_identified': [],
            'audit_score': 0.0
        }
        
        analysis = analyst_results.get('analysis', {})
        data_analysis = analysis.get('data_analysis', {})
        model_analysis = analysis.get('model_analysis', {})
        
        # Assess data completeness
        missing_values = data_analysis.get('missing_values', {})
        total_missing = sum(missing_values.values()) if missing_values else 0
        
        if total_missing == 0:
            audit['data_completeness'] = 'Excellent'
            audit['audit_score'] += 0.3
        elif total_missing < 100:
            audit['data_completeness'] = 'Good'
            audit['audit_score'] += 0.2
        elif total_missing < 1000:
            audit['data_completeness'] = 'Acceptable'
            audit['audit_score'] += 0.1
            audit['issues_identified'].append(f"Moderate missing data: {total_missing} values")
        else:
            audit['data_completeness'] = 'Poor'
            audit['issues_identified'].append(f"High missing data: {total_missing} values")
        
        # Assess data quality score
        quality_score = model_analysis.get('data_quality_score', 0.0)
        if quality_score >= 0.9:
            audit['data_accuracy'] = 'Excellent'
            audit['audit_score'] += 0.3
        elif quality_score >= 0.8:
            audit['data_accuracy'] = 'Good'
            audit['audit_score'] += 0.2
        elif quality_score >= 0.7:
            audit['data_accuracy'] = 'Acceptable'
            audit['audit_score'] += 0.1
            audit['issues_identified'].append("Data quality score below optimal threshold")
        else:
            audit['data_accuracy'] = 'Poor'
            audit['issues_identified'].append("Poor data quality score")
        
        # Assess data consistency
        shape = data_analysis.get('shape', (0, 0))
        if shape[0] > 1000 and shape[1] > 5:
            audit['data_consistency'] = 'Adequate'
            audit['audit_score'] += 0.2
        else:
            audit['data_consistency'] = 'Insufficient'
            audit['issues_identified'].append("Insufficient data volume or features")
        
        return audit
    
    def _audit_model_performance(self, validator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Audit model performance metrics"""
        audit = {
            'discrimination_power': 'Unknown',
            'stability': 'Unknown',
            'calibration': 'Unknown',
            'issues_identified': [],
            'audit_score': 0.0
        }
        
        metrics = validator_results.get('metrics', {})
        
        # Audit discrimination power (AUC)
        auc = metrics.get('auc', 0.0)
        if auc >= self.audit_criteria['model_performance']['auc_threshold']:
            if auc >= 0.8:
                audit['discrimination_power'] = 'Excellent'
                audit['audit_score'] += 0.4
            elif auc >= 0.7:
                audit['discrimination_power'] = 'Good'
                audit['audit_score'] += 0.3
            else:
                audit['discrimination_power'] = 'Acceptable'
                audit['audit_score'] += 0.2
        else:
            audit['discrimination_power'] = 'Insufficient'
            audit['issues_identified'].append(f"AUC of {auc:.3f} below minimum threshold")
        
        # Audit stability (PSI)
        psi = metrics.get('psi', 0.0)
        if psi <= self.audit_criteria['model_performance']['psi_threshold']:
            if psi <= 0.1:
                audit['stability'] = 'Stable'
                audit['audit_score'] += 0.3
            else:
                audit['stability'] = 'Moderate'
                audit['audit_score'] += 0.2
        else:
            audit['stability'] = 'Unstable'
            audit['issues_identified'].append(f"PSI of {psi:.3f} indicates population instability")
        
        # Audit separation (KS)
        ks_stat = metrics.get('ks_statistic', 0.0)
        if ks_stat >= self.audit_criteria['model_performance']['ks_threshold']:
            if ks_stat >= 0.3:
                audit['calibration'] = 'Excellent'
                audit['audit_score'] += 0.3
            elif ks_stat >= 0.2:
                audit['calibration'] = 'Good'
                audit['audit_score'] += 0.2
            else:
                audit['calibration'] = 'Acceptable'
                audit['audit_score'] += 0.1
        else:
            audit['calibration'] = 'Poor'
            audit['issues_identified'].append(f"KS statistic of {ks_stat:.3f} indicates poor separation")
        
        return audit
    
    def _audit_documentation(self, documentation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Audit documentation completeness and compliance"""
        audit = {
            'completeness': 'Unknown',
            'compliance_coverage': 'Unknown',
            'quality': 'Unknown',
            'issues_identified': [],
            'audit_score': 0.0
        }
        
        review_results = documentation_results.get('review_results', {})
        overall_assessment = review_results.get('overall_assessment', {})
        
        # Audit completeness
        completeness = overall_assessment.get('documentation_completeness', 0.0)
        if completeness >= self.audit_criteria['documentation']['min_completeness']:
            if completeness >= 0.9:
                audit['completeness'] = 'Comprehensive'
                audit['audit_score'] += 0.4
            elif completeness >= 0.8:
                audit['completeness'] = 'Adequate'
                audit['audit_score'] += 0.3
            else:
                audit['completeness'] = 'Acceptable'
                audit['audit_score'] += 0.2
        else:
            audit['completeness'] = 'Insufficient'
            audit['issues_identified'].append(f"Documentation completeness of {completeness:.2f} below threshold")
        
        # Audit compliance coverage
        compliance_coverage = overall_assessment.get('compliance_coverage', [])
        required_compliance = self.audit_criteria['documentation']['required_compliance']
        
        covered_requirements = [req for req in required_compliance if req in compliance_coverage]
        coverage_ratio = len(covered_requirements) / len(required_compliance)
        
        if coverage_ratio >= 1.0:
            audit['compliance_coverage'] = 'Complete'
            audit['audit_score'] += 0.3
        elif coverage_ratio >= 0.5:
            audit['compliance_coverage'] = 'Partial'
            audit['audit_score'] += 0.1
            missing_requirements = [req for req in required_compliance if req not in compliance_coverage]
            audit['issues_identified'].append(f"Missing compliance coverage: {', '.join(missing_requirements)}")
        else:
            audit['compliance_coverage'] = 'Inadequate'
            audit['issues_identified'].append("Insufficient regulatory compliance coverage")
        
        # Audit quality
        quality_score = overall_assessment.get('quality_score', 0.0)
        if quality_score >= 0.8:
            audit['quality'] = 'High'
            audit['audit_score'] += 0.3
        elif quality_score >= 0.6:
            audit['quality'] = 'Adequate'
            audit['audit_score'] += 0.2
        elif quality_score >= 0.4:
            audit['quality'] = 'Acceptable'
            audit['audit_score'] += 0.1
            audit['issues_identified'].append("Documentation quality could be improved")
        else:
            audit['quality'] = 'Poor'
            audit['issues_identified'].append("Poor documentation quality")
        
        return audit
    
    def _audit_validation_process(self, analyst_results: Dict[str, Any],
                                 validator_results: Dict[str, Any],
                                 documentation_results: Dict[str, Any],
                                 reviewer_results: Dict[str, Any]) -> Dict[str, Any]:
        """Audit the validation process itself"""
        audit = {
            'process_completeness': 'Unknown',
            'independence': 'Unknown',
            'methodology': 'Unknown',
            'issues_identified': [],
            'audit_score': 0.0
        }
        
        # Check process completeness
        completed_steps = 0
        if analyst_results.get('status') == 'completed':
            completed_steps += 1
        if validator_results.get('status') == 'completed':
            completed_steps += 1
        if documentation_results.get('status') == 'completed':
            completed_steps += 1
        if reviewer_results.get('status') == 'completed':
            completed_steps += 1
        
        if completed_steps >= 4:
            audit['process_completeness'] = 'Complete'
            audit['audit_score'] += 0.3
        elif completed_steps >= 3:
            audit['process_completeness'] = 'Mostly Complete'
            audit['audit_score'] += 0.2
        elif completed_steps >= 2:
            audit['process_completeness'] = 'Partial'
            audit['audit_score'] += 0.1
            audit['issues_identified'].append("Validation process not fully completed")
        else:
            audit['process_completeness'] = 'Incomplete'
            audit['issues_identified'].append("Validation process significantly incomplete")
        
        # Assess independence
        # In a real system, this would check for actual independence
        audit['independence'] = 'Adequate'
        audit['audit_score'] += 0.2
        
        # Assess methodology
        if validator_results.get('status') == 'completed':
            metrics = validator_results.get('metrics', {})
            if 'auc' in metrics and 'ks_statistic' in metrics and 'psi' in metrics:
                audit['methodology'] = 'Appropriate'
                audit['audit_score'] += 0.3
            else:
                audit['methodology'] = 'Limited'
                audit['audit_score'] += 0.1
                audit['issues_identified'].append("Limited validation methodology")
        else:
            audit['methodology'] = 'Unknown'
            audit['issues_identified'].append("Validation methodology not assessed")
        
        return audit
    
    def _perform_independent_assessment(self, audit_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Perform independent assessment of all audit findings"""
        assessment = {
            'overall_audit_score': 0.0,
            'critical_issues': [],
            'recommendations': [],
            'independent_opinion': 'Unknown'
        }
        
        # Calculate overall audit score
        total_score = 0.0
        score_count = 0
        
        for category, findings in audit_findings.items():
            if category != 'independent_assessment' and isinstance(findings, dict):
                category_score = findings.get('audit_score', 0.0)
                total_score += category_score
                score_count += 1
        
        if score_count > 0:
            assessment['overall_audit_score'] = total_score / score_count
        
        # Identify critical issues
        critical_issues = []
        for category, findings in audit_findings.items():
            if category != 'independent_assessment' and isinstance(findings, dict):
                issues = findings.get('issues_identified', [])
                critical_issues.extend(issues)
        
        assessment['critical_issues'] = critical_issues
        
        # Generate independent opinion
        audit_score = assessment['overall_audit_score']
        if audit_score >= 0.8 and len(critical_issues) == 0:
            assessment['independent_opinion'] = 'Satisfactory'
            assessment['recommendations'].append('Model validation meets all requirements')
        elif audit_score >= 0.6 and len(critical_issues) <= 2:
            assessment['independent_opinion'] = 'Acceptable with Conditions'
            assessment['recommendations'].extend([
                'Address identified issues before final approval',
                'Implement enhanced monitoring'
            ])
        elif audit_score >= 0.4:
            assessment['independent_opinion'] = 'Requires Improvement'
            assessment['recommendations'].extend([
                'Significant improvements required',
                'Re-validation recommended after improvements'
            ])
        else:
            assessment['independent_opinion'] = 'Unsatisfactory'
            assessment['recommendations'].extend([
                'Model does not meet validation requirements',
                'Substantial rework required'
            ])
        
        return assessment
    
    def _assess_compliance(self, analyst_results: Dict[str, Any],
                          validator_results: Dict[str, Any],
                          documentation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess regulatory compliance"""
        compliance = {
            'basel_compliance': 'Unknown',
            'ifrs9_compliance': 'Unknown',
            'model_risk_compliance': 'Unknown',
            'overall_compliance': 'Unknown',
            'compliance_issues': []
        }
        
        # Assess Basel compliance
        basel_score = 0.0
        if validator_results.get('status') == 'completed':
            metrics = validator_results.get('metrics', {})
            auc = metrics.get('auc', 0.0)
            if auc >= 0.6:  # Basel minimum threshold
                basel_score += 0.5
        
        if documentation_results.get('status') == 'completed':
            review_results = documentation_results.get('review_results', {})
            overall_assessment = review_results.get('overall_assessment', {})
            compliance_coverage = overall_assessment.get('compliance_coverage', [])
            if 'Basel III' in compliance_coverage:
                basel_score += 0.5
        
        if basel_score >= 1.0:
            compliance['basel_compliance'] = 'Compliant'
        elif basel_score >= 0.5:
            compliance['basel_compliance'] = 'Partially Compliant'
            compliance['compliance_issues'].append('Basel III compliance partially met')
        else:
            compliance['basel_compliance'] = 'Non-Compliant'
            compliance['compliance_issues'].append('Basel III compliance not met')
        
        # Assess IFRS 9 compliance
        ifrs9_score = 0.0
        if validator_results.get('status') == 'completed':
            # IFRS 9 requires specific validation approaches
            ifrs9_score += 0.5
        
        if documentation_results.get('status') == 'completed':
            review_results = documentation_results.get('review_results', {})
            overall_assessment = review_results.get('overall_assessment', {})
            compliance_coverage = overall_assessment.get('compliance_coverage', [])
            if 'IFRS 9' in compliance_coverage:
                ifrs9_score += 0.5
        
        if ifrs9_score >= 1.0:
            compliance['ifrs9_compliance'] = 'Compliant'
        elif ifrs9_score >= 0.5:
            compliance['ifrs9_compliance'] = 'Partially Compliant'
            compliance['compliance_issues'].append('IFRS 9 compliance partially met')
        else:
            compliance['ifrs9_compliance'] = 'Non-Compliant'
            compliance['compliance_issues'].append('IFRS 9 compliance not met')
        
        # Assess model risk compliance
        model_risk_score = 0.0
        if len(compliance['compliance_issues']) == 0:
            model_risk_score = 1.0
        elif len(compliance['compliance_issues']) <= 2:
            model_risk_score = 0.5
        
        if model_risk_score >= 1.0:
            compliance['model_risk_compliance'] = 'Compliant'
        elif model_risk_score >= 0.5:
            compliance['model_risk_compliance'] = 'Partially Compliant'
        else:
            compliance['model_risk_compliance'] = 'Non-Compliant'
        
        # Overall compliance assessment
        compliant_areas = [
            compliance['basel_compliance'] == 'Compliant',
            compliance['ifrs9_compliance'] == 'Compliant',
            compliance['model_risk_compliance'] == 'Compliant'
        ]
        
        if all(compliant_areas):
            compliance['overall_compliance'] = 'Fully Compliant'
        elif sum(compliant_areas) >= 2:
            compliance['overall_compliance'] = 'Mostly Compliant'
        elif sum(compliant_areas) >= 1:
            compliance['overall_compliance'] = 'Partially Compliant'
        else:
            compliance['overall_compliance'] = 'Non-Compliant'
        
        return compliance
    
    def _generate_final_recommendation(self, audit_findings: Dict[str, Any],
                                      compliance_assessment: Dict[str, Any],
                                      reviewer_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final recommendation based on all assessments"""
        recommendation = {
            'approval_status': 'Pending',
            'recommendation_type': 'Unknown',
            'conditions': [],
            'next_steps': [],
            'monitoring_requirements': [],
            'approval_validity': 'Unknown'
        }
        
        # Get independent assessment
        independent_assessment = audit_findings.get('independent_assessment', {})
        independent_opinion = independent_assessment.get('independent_opinion', 'Unknown')
        audit_score = independent_assessment.get('overall_audit_score', 0.0)
        critical_issues = independent_assessment.get('critical_issues', [])
        
        # Get compliance status
        overall_compliance = compliance_assessment.get('overall_compliance', 'Unknown')
        compliance_issues = compliance_assessment.get('compliance_issues', [])
        
        # Generate recommendation based on assessments
        if (independent_opinion == 'Satisfactory' and 
            overall_compliance == 'Fully Compliant' and 
            len(critical_issues) == 0):
            
            recommendation['approval_status'] = 'Approved'
            recommendation['recommendation_type'] = 'Unconditional Approval'
            recommendation['next_steps'] = [
                'Proceed with model implementation',
                'Establish regular monitoring schedule'
            ]
            recommendation['approval_validity'] = '12 months'
            
        elif (independent_opinion in ['Satisfactory', 'Acceptable with Conditions'] and
              overall_compliance in ['Fully Compliant', 'Mostly Compliant'] and
              len(critical_issues) <= 2):
            
            recommendation['approval_status'] = 'Conditionally Approved'
            recommendation['recommendation_type'] = 'Conditional Approval'
            recommendation['conditions'] = critical_issues + compliance_issues
            recommendation['next_steps'] = [
                'Address all conditions within 30 days',
                'Provide evidence of condition resolution',
                'Schedule follow-up review'
            ]
            recommendation['approval_validity'] = '6 months'
            
        elif (independent_opinion == 'Requires Improvement' or
              overall_compliance == 'Partially Compliant' or
              len(critical_issues) > 2):
            
            recommendation['approval_status'] = 'Rejected'
            recommendation['recommendation_type'] = 'Rejection with Resubmission'
            recommendation['conditions'] = [
                'Address all identified issues',
                'Provide comprehensive remediation plan',
                'Conduct additional validation testing'
            ]
            recommendation['next_steps'] = [
                'Develop remediation plan',
                'Implement required improvements',
                'Resubmit for validation'
            ]
            recommendation['approval_validity'] = 'Not Applicable'
            
        else:
            recommendation['approval_status'] = 'Rejected'
            recommendation['recommendation_type'] = 'Rejection'
            recommendation['next_steps'] = [
                'Conduct comprehensive model review',
                'Consider model replacement',
                'Engage with model development team'
            ]
            recommendation['approval_validity'] = 'Not Applicable'
        
        # Add monitoring requirements
        recommendation['monitoring_requirements'] = [
            'Monthly performance monitoring',
            'Quarterly stability assessment',
            'Annual comprehensive review',
            'Immediate escalation for performance degradation'
        ]
        
        return recommendation
    
    def _generate_audit_report(self, audit_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        report = {
            'executive_summary': '',
            'audit_scope': '',
            'key_findings': [],
            'recommendations': [],
            'conclusion': '',
            'next_review_date': ''
        }
        
        # Executive summary
        final_recommendation = audit_results.get('final_recommendation', {})
        approval_status = final_recommendation.get('approval_status', 'Unknown')
        
        report['executive_summary'] = f"""
        Independent audit of the credit risk model validation has been completed. 
        The audit assessed model performance, documentation completeness, and regulatory compliance.
        Final recommendation: {approval_status}.
        """
        
        # Audit scope
        report['audit_scope'] = """
        The audit covered:
        - Data quality assessment
        - Model performance validation
        - Documentation review
        - Regulatory compliance assessment
        - Validation process review
        """
        
        # Key findings
        audit_findings = audit_results.get('audit_findings', {})
        independent_assessment = audit_findings.get('independent_assessment', {})
        critical_issues = independent_assessment.get('critical_issues', [])
        
        report['key_findings'] = critical_issues[:5]  # Top 5 findings
        
        # Recommendations
        recommendations = independent_assessment.get('recommendations', [])
        conditions = final_recommendation.get('conditions', [])
        
        report['recommendations'] = recommendations + conditions
        
        # Conclusion
        recommendation_type = final_recommendation.get('recommendation_type', 'Unknown')
        report['conclusion'] = f"""
        Based on the comprehensive audit, the model validation is assessed as {recommendation_type}.
        All identified issues must be addressed according to the specified timeline.
        """
        
        # Next review date
        from datetime import datetime, timedelta
        next_review = datetime.now() + timedelta(days=365)
        report['next_review_date'] = next_review.strftime('%Y-%m-%d')
        
        return report
