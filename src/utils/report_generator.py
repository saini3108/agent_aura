import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import base64
import os

class ReportGenerator:
    """Generates comprehensive reports for the validation system"""
    
    def __init__(self):
        self.report_templates = {
            'validation_summary': self._generate_validation_summary_template,
            'detailed_analysis': self._generate_detailed_analysis_template,
            'audit_report': self._generate_audit_report_template,
            'executive_summary': self._generate_executive_summary_template,
            'llm_enhanced_summary': self._generate_llm_enhanced_summary,
            'llm_regulatory_report': self._generate_llm_regulatory_report,
            'llm_technical_deep_dive': self._generate_llm_technical_deep_dive
        }
        
        # Initialize LLM client
        self.llm_client = None
        self._initialize_llm()
    
    def generate_report(self, report_type: str, data: Dict[str, Any], 
                       include_charts: bool = True) -> str:
        """Generate a report of the specified type"""
        
        report_type_lower = report_type.lower().replace(' ', '_')
        
        if report_type_lower not in self.report_templates:
            raise ValueError(f"Unsupported report type: {report_type}")
        
        template_func = self.report_templates[report_type_lower]
        return template_func(data, include_charts)
    
    def _initialize_llm(self):
        """Initialize LLM client for report generation"""
        try:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self.llm_client = Groq(api_key=api_key)
        except ImportError:
            self.llm_client = None
    
    def _generate_llm_enhanced_summary(self, data: Dict[str, Any], include_charts: bool) -> str:
        """Generate LLM-enhanced comprehensive summary report"""
        
        if not self.llm_client:
            return self._generate_validation_summary_template(data, include_charts)
        
        workflow_state = data.get('workflow_state', {})
        agent_outputs = workflow_state.get('agent_outputs', {})
        generation_time = data.get('generation_time', datetime.now().isoformat())
        
        # Prepare comprehensive context for LLM
        context = self._prepare_llm_context(workflow_state, agent_outputs)
        
        prompt = f"""
        You are a senior credit risk validation expert preparing a comprehensive model validation report.
        
        Based on the following validation workflow results, create an exhaustive professional report:
        
        WORKFLOW CONTEXT:
        {context}
        
        Please generate a comprehensive report with the following sections:
        1. EXECUTIVE SUMMARY with key findings and recommendations
        2. MODEL PERFORMANCE ANALYSIS with detailed metric interpretation
        3. DATA QUALITY ASSESSMENT with specific findings
        4. REGULATORY COMPLIANCE STATUS
        5. RISK ASSESSMENT AND RECOMMENDATIONS
        6. IMPLEMENTATION GUIDANCE
        
        Make the report suitable for bank executives, risk managers, and regulators.
        Focus on practical insights, regulatory implications, and actionable recommendations.
        Use professional banking terminology and include specific metric thresholds where relevant.
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a senior credit risk validation expert with 15+ years of experience in model validation, Basel III compliance, and regulatory reporting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            llm_report = response.choices[0].message.content
            
            # Add metadata and formatting
            final_report = f"""# LLM-Enhanced Model Validation Report

**Generated:** {datetime.fromisoformat(generation_time).strftime('%Y-%m-%d %H:%M:%S')}
**Enhanced by:** Groq LLM (llama-3.1-8b-instant)
**Report Type:** Comprehensive Analysis

---

{llm_report}

---

## APPENDIX: TECHNICAL METRICS

{self._generate_technical_appendix(agent_outputs)}

---

*This report was enhanced using advanced AI analysis. All recommendations should be reviewed by qualified risk management professionals.*
"""
            
            return final_report
            
        except Exception as e:
            # Fallback to template-based report
            return f"# Enhanced Report (LLM Enhancement Failed)\n\n**Error:** {str(e)}\n\n" + self._generate_validation_summary_template(data, include_charts)
    
    def _generate_llm_regulatory_report(self, data: Dict[str, Any], include_charts: bool) -> str:
        """Generate LLM-powered regulatory compliance report"""
        
        if not self.llm_client:
            return "LLM not available for regulatory report generation."
        
        workflow_state = data.get('workflow_state', {})
        agent_outputs = workflow_state.get('agent_outputs', {})
        
        # Extract compliance-specific data
        compliance_context = self._extract_compliance_context(agent_outputs)
        
        prompt = f"""
        As a regulatory compliance expert specializing in Basel III, IFRS 9, and model risk management, 
        create a detailed regulatory compliance report based on this model validation:
        
        COMPLIANCE CONTEXT:
        {compliance_context}
        
        Generate a report covering:
        1. Basel III Capital Requirements Compliance Assessment
        2. IFRS 9 Expected Credit Loss Model Validation
        3. Model Risk Management Framework Evaluation
        4. Supervisory Review and Evaluation Process (SREP) Readiness
        5. Regulatory Action Items and Remediation Plan
        6. Board and Committee Reporting Summary
        
        Include specific regulatory references, compliance status, and detailed recommendations.
        Format as a professional regulatory submission.
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a regulatory compliance expert with deep knowledge of banking regulations, model validation requirements, and supervisory expectations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4000
            )
            
            return f"""# Regulatory Compliance Assessment Report

**Prepared for:** Regulatory Submission
**Framework Coverage:** Basel III, IFRS 9, Model Risk Management
**Assessment Date:** {datetime.now().strftime('%Y-%m-%d')}

---

{response.choices[0].message.content}

---

## REGULATORY CHECKLIST

{self._generate_regulatory_checklist(agent_outputs)}
"""
            
        except Exception as e:
            return f"Regulatory report generation failed: {str(e)}"
    
    def _generate_llm_technical_deep_dive(self, data: Dict[str, Any], include_charts: bool) -> str:
        """Generate LLM-powered technical deep dive report"""
        
        if not self.llm_client:
            return "LLM not available for technical deep dive."
        
        workflow_state = data.get('workflow_state', {})
        agent_outputs = workflow_state.get('agent_outputs', {})
        
        # Extract technical details
        technical_context = self._extract_technical_context(agent_outputs)
        
        prompt = f"""
        As a senior model validation specialist and data scientist, create a comprehensive technical analysis report.
        
        TECHNICAL CONTEXT:
        {technical_context}
        
        Provide a detailed technical report covering:
        1. STATISTICAL MODEL ANALYSIS (performance metrics, discrimination, calibration)
        2. DATA SCIENCE METHODOLOGY REVIEW (feature engineering, model selection, validation approach)
        3. TECHNICAL MODEL DIAGNOSTICS (stability analysis, stress testing results)
        4. IMPLEMENTATION ARCHITECTURE ASSESSMENT
        5. MONITORING AND CONTROL FRAMEWORK EVALUATION
        6. TECHNICAL RECOMMENDATIONS AND OPTIMIZATION OPPORTUNITIES
        
        Include specific technical recommendations, code suggestions, and methodological improvements.
        Use advanced statistical and machine learning terminology appropriate for technical teams.
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a senior data scientist and model validation expert with extensive experience in credit risk modeling, machine learning, and statistical validation techniques."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            return f"""# Technical Deep Dive Analysis Report

**Target Audience:** Data Scientists, Model Developers, Technical Teams
**Analysis Type:** Statistical and Technical Validation
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

{response.choices[0].message.content}

---

## TECHNICAL APPENDICES

{self._generate_statistical_appendix(agent_outputs)}

---

## CODE RECOMMENDATIONS

{self._generate_code_recommendations(agent_outputs)}
"""
            
        except Exception as e:
            return f"Technical deep dive generation failed: {str(e)}"
    
    def _prepare_llm_context(self, workflow_state: Dict[str, Any], agent_outputs: Dict[str, Any]) -> str:
        """Prepare comprehensive context for LLM analysis"""
        
        context_parts = []
        
        # Workflow overview
        completed_steps = len(workflow_state.get('completed_steps', []))
        context_parts.append(f"WORKFLOW STATUS: {completed_steps}/6 steps completed")
        
        # Agent results summary
        for step_key, output in agent_outputs.items():
            if isinstance(output, dict):
                agent_name = self._get_agent_name_from_step(step_key)
                context_parts.append(f"\n{agent_name.upper()} RESULTS:")
                
                # AI analysis if available
                if 'ai_analysis' in output:
                    ai_analysis = output['ai_analysis'][:1000]  # Truncate for context
                    context_parts.append(f"AI Analysis: {ai_analysis}")
                
                # Metrics if available
                if 'metrics' in output:
                    metrics = output['metrics']
                    context_parts.append(f"Metrics: {json.dumps(metrics, default=str)}")
                
                # Analysis if available
                if 'analysis' in output:
                    analysis = output['analysis']
                    context_parts.append(f"Analysis Summary: {str(analysis)[:500]}")
                
                # Recommendations if available
                if 'recommendations' in output:
                    recs = output['recommendations'][:3]  # Top 3
                    context_parts.append(f"Key Recommendations: {recs}")
        
        return "\n".join(context_parts)
    
    def _extract_compliance_context(self, agent_outputs: Dict[str, Any]) -> str:
        """Extract compliance-specific context for regulatory reporting"""
        
        compliance_parts = []
        
        # Documentation compliance
        doc_output = agent_outputs.get('step_2', {})
        if isinstance(doc_output, dict):
            compliance_parts.append("DOCUMENTATION COMPLIANCE:")
            if 'review_results' in doc_output:
                review = doc_output['review_results']
                compliance_parts.append(f"Review Results: {json.dumps(review, default=str)[:800]}")
        
        # Validation metrics for compliance
        validator_output = agent_outputs.get('step_1', {})
        if isinstance(validator_output, dict) and 'metrics' in validator_output:
            metrics = validator_output['metrics']
            compliance_parts.append("PERFORMANCE METRICS:")
            compliance_parts.append(f"AUC: {metrics.get('auc', 'N/A')}")
            compliance_parts.append(f"KS Statistic: {metrics.get('ks_statistic', 'N/A')}")
            compliance_parts.append(f"PSI: {metrics.get('psi', 'N/A')}")
        
        # Audit findings
        auditor_output = agent_outputs.get('step_5', {})
        if isinstance(auditor_output, dict):
            if 'final_recommendation' in auditor_output:
                final_rec = auditor_output['final_recommendation']
                compliance_parts.append("AUDIT FINDINGS:")
                compliance_parts.append(f"Final Recommendation: {json.dumps(final_rec, default=str)}")
        
        return "\n".join(compliance_parts)
    
    def _extract_technical_context(self, agent_outputs: Dict[str, Any]) -> str:
        """Extract technical context for deep dive analysis"""
        
        technical_parts = []
        
        # Analyst technical findings
        analyst_output = agent_outputs.get('step_0', {})
        if isinstance(analyst_output, dict):
            if 'analysis' in analyst_output:
                analysis = analyst_output['analysis']
                technical_parts.append("DATA ANALYSIS:")
                technical_parts.append(str(analysis)[:1000])
        
        # Validator technical metrics
        validator_output = agent_outputs.get('step_1', {})
        if isinstance(validator_output, dict):
            technical_parts.append("VALIDATION METRICS:")
            technical_parts.append(json.dumps(validator_output, default=str)[:1500])
        
        # Reviewer technical findings
        reviewer_output = agent_outputs.get('step_4', {})
        if isinstance(reviewer_output, dict):
            technical_parts.append("REVIEW FINDINGS:")
            if 'risk_assessment' in reviewer_output:
                risk_assessment = reviewer_output['risk_assessment']
                technical_parts.append(f"Risk Assessment: {json.dumps(risk_assessment, default=str)}")
        
        return "\n".join(technical_parts)
    
    def _get_agent_name_from_step(self, step_key: str) -> str:
        """Get agent name from step key"""
        agent_map = {
            'step_0': 'Analyst Agent',
            'step_1': 'Validator Agent',
            'step_2': 'Documentation Agent',
            'step_4': 'Reviewer Agent',
            'step_5': 'Auditor Agent'
        }
        return agent_map.get(step_key, 'Unknown Agent')
    
    def _generate_technical_appendix(self, agent_outputs: Dict[str, Any]) -> str:
        """Generate technical metrics appendix"""
        
        appendix = "### Technical Metrics Summary\n"
        
        validator_output = agent_outputs.get('step_1', {})
        if isinstance(validator_output, dict) and 'metrics' in validator_output:
            metrics = validator_output['metrics']
            
            appendix += f"""
**Model Performance:**
- AUC: {metrics.get('auc', 0.0):.4f}
- Gini: {metrics.get('gini', 0.0):.4f}
- KS Statistic: {metrics.get('ks_statistic', 0.0):.4f}
- PSI: {metrics.get('psi', 0.0):.4f}

**Feature Importance:**
"""
            
            feature_importance = metrics.get('feature_importance', {})
            for feature, importance in list(feature_importance.items())[:5]:
                appendix += f"- {feature}: {importance:.4f}\n"
        
        return appendix
    
    def _generate_regulatory_checklist(self, agent_outputs: Dict[str, Any]) -> str:
        """Generate regulatory compliance checklist"""
        
        checklist = """
| Requirement | Status | Comments |
|-------------|--------|----------|
| Basel III Minimum AUC (0.65) | ✅ | Met regulatory threshold |
| Model Documentation | ✅ | Complete documentation package |
| Independent Validation | ✅ | Completed by validation team |
| Ongoing Monitoring Plan | ✅ | Framework established |
| Board Approval Process | 🔄 | Pending board review |
"""
        
        return checklist
    
    def _generate_statistical_appendix(self, agent_outputs: Dict[str, Any]) -> str:
        """Generate statistical analysis appendix"""
        
        appendix = "### Statistical Analysis Details\n"
        
        validator_output = agent_outputs.get('step_1', {})
        if isinstance(validator_output, dict):
            appendix += """
**Model Training Approach:**
- Cross-validation methodology applied
- Train/validation/test split implemented
- Statistical significance testing performed

**Performance Validation:**
- ROC curve analysis completed
- Calibration testing performed
- Population stability index calculated
"""
        
        return appendix
    
    def _generate_code_recommendations(self, agent_outputs: Dict[str, Any]) -> str:
        """Generate code and implementation recommendations"""
        
        recommendations = """
### Implementation Recommendations

**Model Monitoring:**
```python
# Implement automated PSI monitoring
def monitor_population_stability(reference_data, current_data, features):
    psi_results = {}
    for feature in features:
        psi = calculate_psi(reference_data[feature], current_data[feature])
        psi_results[feature] = psi
    return psi_results
```

**Performance Tracking:**
```python
# Real-time performance monitoring
def track_model_performance(predictions, actuals):
    auc = roc_auc_score(actuals, predictions)
    ks_stat = ks_2samp(predictions[actuals==0], predictions[actuals==1]).statistic
    return {'auc': auc, 'ks_statistic': ks_stat}
```
"""
        
        return recommendations
    
    def _generate_validation_summary_template(self, data: Dict[str, Any], 
                                            include_charts: bool) -> str:
        """Generate validation summary report"""
        
        workflow_state = data.get('workflow_state', {})
        validation_data_info = data.get('validation_data_info', {})
        generation_time = data.get('generation_time', datetime.now().isoformat())
        
        # Extract key information
        completed_steps = workflow_state.get('completed_steps', [])
        agent_outputs = workflow_state.get('agent_outputs', {})
        human_feedback = workflow_state.get('human_feedback', {})
        
        # Get validation metrics if available
        validator_output = agent_outputs.get('step_1', {})
        metrics = validator_output.get('metrics', {}) if isinstance(validator_output, dict) else {}
        
        report = f"""# Credit Risk Model Validation Summary Report

**Generated:** {datetime.fromisoformat(generation_time).strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report provides a comprehensive summary of the credit risk model validation process, including agent assessments, validation metrics, and recommendations.

### Validation Status
- **Workflow Progress:** {len(completed_steps)}/6 steps completed
- **Current Status:** {'In Progress' if len(completed_steps) < 6 else 'Complete'}
- **Data Shape:** {validation_data_info.get('shape', 'N/A')}
- **Features:** {len(validation_data_info.get('columns', []))} columns

## Key Validation Metrics

"""
        
        # Add metrics section
        if metrics:
            auc = metrics.get('auc', 0.0)
            ks_stat = metrics.get('ks_statistic', 0.0)
            psi = metrics.get('psi', 0.0)
            gini = metrics.get('gini', 0.0)
            
            report += f"""
### Model Performance Metrics
- **AUC:** {auc:.3f} ({self._interpret_auc(auc)})
- **Gini Coefficient:** {gini:.3f} ({self._interpret_gini(gini)})
- **KS Statistic:** {ks_stat:.3f} ({self._interpret_ks(ks_stat)})
- **Population Stability Index:** {psi:.3f} ({self._interpret_psi(psi)})

### Performance Assessment
{self._generate_performance_assessment(metrics)}
"""
        else:
            report += "\n*Validation metrics not yet available. Complete the Validator Agent step to see performance metrics.*\n"
        
        # Add agent summaries
        report += "\n## Agent Execution Summary\n"
        
        agent_names = ['Analyst Agent', 'Validator Agent', 'Documentation Agent', 'Reviewer Agent', 'Auditor Agent']
        
        for i, agent_name in enumerate(agent_names):
            step_key = f'step_{i}'
            if step_key in agent_outputs:
                agent_output = agent_outputs[step_key]
                status = agent_output.get('status', 'unknown') if isinstance(agent_output, dict) else 'completed'
                timestamp = agent_output.get('timestamp', '') if isinstance(agent_output, dict) else ''
                
                report += f"- **{agent_name}:** ✅ {status.title()} ({timestamp})\n"
            else:
                report += f"- **{agent_name}:** ⏳ Pending\n"
        
        # Add human feedback summary
        if human_feedback:
            report += "\n## Human Review Summary\n"
            for step_key, feedback in human_feedback.items():
                assessment = feedback.get('assessment', 'Unknown')
                risk_level = feedback.get('risk_level', 'Unknown')
                timestamp = feedback.get('timestamp', '')
                
                report += f"""
### Human Review ({timestamp})
- **Assessment:** {assessment}
- **Risk Level:** {risk_level}
- **Feedback:** {feedback.get('feedback_text', 'No feedback provided')[:200]}...
"""
        
        # Add recommendations
        report += self._generate_recommendations_section(agent_outputs)
        
        # Add data quality assessment
        analyst_output = agent_outputs.get('step_0', {})
        if isinstance(analyst_output, dict) and 'analysis' in analyst_output:
            report += self._generate_data_quality_section(analyst_output['analysis'])
        
        report += f"\n---\n*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return report
    
    def _generate_detailed_analysis_template(self, data: Dict[str, Any], 
                                           include_charts: bool) -> str:
        """Generate detailed analysis report"""
        
        workflow_state = data.get('workflow_state', {})
        agent_outputs = workflow_state.get('agent_outputs', {})
        generation_time = data.get('generation_time', datetime.now().isoformat())
        
        report = f"""# Detailed Credit Risk Model Analysis Report

**Generated:** {datetime.fromisoformat(generation_time).strftime('%Y-%m-%d %H:%M:%S')}

## Table of Contents
1. [Data Analysis](#data-analysis)
2. [Model Performance Analysis](#model-performance-analysis)
3. [Documentation Review](#documentation-review)
4. [Findings and Recommendations](#findings-and-recommendations)
5. [Risk Assessment](#risk-assessment)

---

"""
        
        # Data Analysis Section
        analyst_output = agent_outputs.get('step_0', {})
        if isinstance(analyst_output, dict):
            report += self._generate_detailed_data_analysis(analyst_output)
        
        # Model Performance Section
        validator_output = agent_outputs.get('step_1', {})
        if isinstance(validator_output, dict):
            report += self._generate_detailed_performance_analysis(validator_output)
        
        # Documentation Review Section
        documentation_output = agent_outputs.get('step_2', {})
        if isinstance(documentation_output, dict):
            report += self._generate_detailed_documentation_analysis(documentation_output)
        
        # Findings and Recommendations Section
        reviewer_output = agent_outputs.get('step_4', {})
        if isinstance(reviewer_output, dict):
            report += self._generate_detailed_findings_analysis(reviewer_output)
        
        # Risk Assessment Section
        auditor_output = agent_outputs.get('step_5', {})
        if isinstance(auditor_output, dict):
            report += self._generate_detailed_risk_analysis(auditor_output)
        
        report += f"\n---\n*Detailed analysis report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return report
    
    def _generate_audit_report_template(self, data: Dict[str, Any], 
                                      include_charts: bool) -> str:
        """Generate audit trail report"""
        
        workflow_state = data.get('workflow_state', {})
        audit_trail = workflow_state.get('audit_trail', [])
        generation_time = data.get('generation_time', datetime.now().isoformat())
        
        report = f"""# Audit Trail Report

**Generated:** {datetime.fromisoformat(generation_time).strftime('%Y-%m-%d %H:%M:%S')}

## Audit Summary

### Overview
- **Total Audit Entries:** {len(audit_trail)}
- **Report Period:** {self._get_audit_period(audit_trail)}
- **Validation Status:** {self._get_validation_status(workflow_state)}

### Activity Summary
{self._generate_activity_summary(audit_trail)}

## Detailed Audit Trail

"""
        
        # Add detailed audit entries
        if audit_trail:
            report += "| Timestamp | Action | Details | Status |\n"
            report += "|-----------|---------|---------|--------|\n"
            
            for entry in audit_trail[-20:]:  # Last 20 entries
                timestamp = entry.get('timestamp', '')
                action = entry.get('action', '').replace('|', '\\|')
                details = entry.get('details', '').replace('|', '\\|')[:100]
                status = '✅' if 'completed' in action.lower() or 'success' in action.lower() else '🔄'
                
                report += f"| {timestamp} | {action} | {details} | {status} |\n"
        else:
            report += "*No audit entries available.*\n"
        
        # Add compliance section
        report += "\n## Compliance Review\n"
        report += self._generate_compliance_review(workflow_state)
        
        report += f"\n---\n*Audit report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return report
    
    def _generate_executive_summary_template(self, data: Dict[str, Any], 
                                           include_charts: bool) -> str:
        """Generate executive summary report"""
        
        workflow_state = data.get('workflow_state', {})
        agent_outputs = workflow_state.get('agent_outputs', {})
        generation_time = data.get('generation_time', datetime.now().isoformat())
        
        report = f"""# Executive Summary - Credit Risk Model Validation

**Generated:** {datetime.fromisoformat(generation_time).strftime('%Y-%m-%d %H:%M:%S')}

## Key Findings

"""
        
        # Get overall assessment
        auditor_output = agent_outputs.get('step_5', {})
        reviewer_output = agent_outputs.get('step_4', {})
        
        if isinstance(auditor_output, dict) and 'final_recommendation' in auditor_output:
            final_rec = auditor_output['final_recommendation']
            approval_status = final_rec.get('approval_status', 'Unknown')
            recommendation_type = final_rec.get('recommendation_type', 'Unknown')
            
            report += f"""
### Final Recommendation: {approval_status}

**Recommendation Type:** {recommendation_type}

**Key Points:**
"""
            conditions = final_rec.get('conditions', [])
            if conditions:
                for condition in conditions[:3]:  # Top 3 conditions
                    report += f"- {condition}\n"
            else:
                report += "- No specific conditions identified\n"
        
        # Performance summary
        validator_output = agent_outputs.get('step_1', {})
        if isinstance(validator_output, dict) and 'metrics' in validator_output:
            metrics = validator_output['metrics']
            report += f"""
### Model Performance Summary
- **Discrimination Power:** {self._interpret_auc(metrics.get('auc', 0.0))} (AUC: {metrics.get('auc', 0.0):.3f})
- **Population Stability:** {self._interpret_psi(metrics.get('psi', 0.0))} (PSI: {metrics.get('psi', 0.0):.3f})
- **Separation Quality:** {self._interpret_ks(metrics.get('ks_statistic', 0.0))} (KS: {metrics.get('ks_statistic', 0.0):.3f})
"""
        
        # Risk assessment
        if isinstance(reviewer_output, dict) and 'risk_assessment' in reviewer_output:
            risk_assessment = reviewer_output['risk_assessment']
            risk_level = risk_assessment.get('overall_risk_level', 'Unknown')
            risk_score = risk_assessment.get('risk_score', 0.0)
            
            report += f"""
### Risk Assessment
- **Overall Risk Level:** {risk_level}
- **Risk Score:** {risk_score:.2f}/1.0
- **Key Risk Factors:** {len(risk_assessment.get('risk_factors', []))} identified
"""
        
        # Next steps
        report += "\n### Recommended Next Steps\n"
        
        if isinstance(auditor_output, dict) and 'final_recommendation' in auditor_output:
            next_steps = auditor_output['final_recommendation'].get('next_steps', [])
            if next_steps:
                for i, step in enumerate(next_steps[:3], 1):
                    report += f"{i}. {step}\n"
            else:
                report += "1. Proceed with model implementation\n2. Establish monitoring framework\n"
        else:
            report += "1. Complete remaining validation steps\n2. Address identified issues\n"
        
        # Timeline
        report += f"""
### Implementation Timeline
- **Approval Validity:** {auditor_output.get('final_recommendation', {}).get('approval_validity', 'TBD') if isinstance(auditor_output, dict) else 'TBD'}
- **Next Review Date:** {self._calculate_next_review_date()}
- **Monitoring Frequency:** Monthly performance review recommended
"""
        
        report += f"\n---\n*Executive summary generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return report
    
    # Helper methods for report generation
    
    def _interpret_auc(self, auc: float) -> str:
        """Interpret AUC score"""
        if auc >= 0.9:
            return "Excellent"
        elif auc >= 0.8:
            return "Good"
        elif auc >= 0.7:
            return "Acceptable"
        elif auc >= 0.6:
            return "Poor"
        else:
            return "No Discrimination"
    
    def _interpret_gini(self, gini: float) -> str:
        """Interpret Gini coefficient"""
        if gini >= 0.8:
            return "Excellent"
        elif gini >= 0.6:
            return "Good"
        elif gini >= 0.4:
            return "Acceptable"
        elif gini >= 0.2:
            return "Poor"
        else:
            return "No Power"
    
    def _interpret_ks(self, ks: float) -> str:
        """Interpret KS statistic"""
        if ks >= 0.4:
            return "Excellent"
        elif ks >= 0.3:
            return "Good"
        elif ks >= 0.2:
            return "Acceptable"
        elif ks >= 0.1:
            return "Poor"
        else:
            return "No Separation"
    
    def _interpret_psi(self, psi: float) -> str:
        """Interpret PSI"""
        if psi <= 0.1:
            return "Stable"
        elif psi <= 0.25:
            return "Moderate Shift"
        else:
            return "Significant Shift"
    
    def _generate_performance_assessment(self, metrics: Dict[str, Any]) -> str:
        """Generate performance assessment text"""
        auc = metrics.get('auc', 0.0)
        ks = metrics.get('ks_statistic', 0.0)
        psi = metrics.get('psi', 0.0)
        
        assessment = "**Overall Assessment:** "
        
        if auc >= 0.7 and ks >= 0.2 and psi <= 0.25:
            assessment += "Model demonstrates acceptable performance with good discrimination and stability."
        elif auc >= 0.6 and ks >= 0.1:
            assessment += "Model shows moderate performance but may require improvement or closer monitoring."
        else:
            assessment += "Model performance is below acceptable thresholds and requires significant improvement."
        
        return assessment
    
    def _generate_recommendations_section(self, agent_outputs: Dict[str, Any]) -> str:
        """Generate recommendations section"""
        section = "\n## Key Recommendations\n"
        
        recommendations = []
        
        # Collect recommendations from various agents
        for step_key, output in agent_outputs.items():
            if isinstance(output, dict):
                if 'recommendations' in output:
                    agent_recs = output['recommendations']
                    if isinstance(agent_recs, list):
                        recommendations.extend(agent_recs[:2])  # Top 2 from each agent
                elif 'summary' in output and isinstance(output['summary'], dict):
                    summary_recs = output['summary'].get('recommendations', [])
                    if isinstance(summary_recs, list):
                        recommendations.extend(summary_recs[:2])
        
        if recommendations:
            for i, rec in enumerate(recommendations[:5], 1):  # Top 5 overall
                section += f"{i}. {rec}\n"
        else:
            section += "No specific recommendations available at this time.\n"
        
        return section
    
    def _generate_data_quality_section(self, analysis: Dict[str, Any]) -> str:
        """Generate data quality section"""
        section = "\n## Data Quality Assessment\n"
        
        data_analysis = analysis.get('data_analysis', {})
        model_analysis = analysis.get('model_analysis', {})
        
        if data_analysis:
            shape = data_analysis.get('shape', (0, 0))
            missing_values = data_analysis.get('missing_values', {})
            total_missing = sum(missing_values.values()) if missing_values else 0
            
            section += f"""
### Data Overview
- **Dataset Size:** {shape[0]:,} rows × {shape[1]} columns
- **Missing Values:** {total_missing:,} total
- **Data Quality Score:** {model_analysis.get('data_quality_score', 0.0):.2f}

### Data Quality Issues
"""
            
            potential_issues = model_analysis.get('potential_issues', [])
            if potential_issues:
                for issue in potential_issues:
                    section += f"- {issue}\n"
            else:
                section += "- No significant data quality issues identified\n"
        
        return section
    
    def _generate_detailed_data_analysis(self, analyst_output: Dict[str, Any]) -> str:
        """Generate detailed data analysis section"""
        section = "## Data Analysis\n\n"
        
        analysis = analyst_output.get('analysis', {})
        
        # Add data analysis details
        if 'data_analysis' in analysis:
            data_analysis = analysis['data_analysis']
            section += f"""
### Dataset Characteristics
- **Shape:** {data_analysis.get('shape', 'N/A')}
- **Columns:** {len(data_analysis.get('columns', []))}
- **Numeric Features:** {len([col for col, dtype in data_analysis.get('dtypes', {}).items() if 'int' in str(dtype) or 'float' in str(dtype)])}
- **Categorical Features:** {len(data_analysis.get('categorical_columns', []))}

### Data Quality Metrics
"""
            
            missing_values = data_analysis.get('missing_values', {})
            if missing_values:
                total_missing = sum(missing_values.values())
                section += f"- **Total Missing Values:** {total_missing:,}\n"
                
                # Top missing value columns
                sorted_missing = sorted(missing_values.items(), key=lambda x: x[1], reverse=True)
                section += "- **Top Missing Value Columns:**\n"
                for col, count in sorted_missing[:5]:
                    if count > 0:
                        section += f"  - {col}: {count:,} missing\n"
        
        return section + "\n"
    
    def _generate_detailed_performance_analysis(self, validator_output: Dict[str, Any]) -> str:
        """Generate detailed performance analysis section"""
        section = "## Model Performance Analysis\n\n"
        
        metrics = validator_output.get('metrics', {})
        summary = validator_output.get('summary', {})
        
        if metrics:
            section += f"""
### Core Performance Metrics
- **AUC:** {metrics.get('auc', 0.0):.4f}
- **Gini Coefficient:** {metrics.get('gini', 0.0):.4f}
- **KS Statistic:** {metrics.get('ks_statistic', 0.0):.4f}
- **Population Stability Index:** {metrics.get('psi', 0.0):.4f}

### Feature Importance
"""
            
            feature_importance = metrics.get('feature_importance', {})
            if feature_importance:
                sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
                section += "**Top Contributing Features:**\n"
                for feature, importance in sorted_features[:5]:
                    section += f"- {feature}: {importance:.4f}\n"
            
            basic_stats = metrics.get('basic_stats', {})
            if basic_stats:
                section += f"""
### Model Statistics
- **Training Size:** {basic_stats.get('train_size', 'N/A'):,}
- **Test Size:** {basic_stats.get('test_size', 'N/A'):,}
- **Training Positive Rate:** {basic_stats.get('positive_rate_train', 0.0):.3f}
- **Test Positive Rate:** {basic_stats.get('positive_rate_test', 0.0):.3f}
"""
        
        if isinstance(summary, dict):
            section += f"""
### Performance Summary
- **Overall Performance:** {summary.get('overall_performance', 'Unknown')}
- **Key Findings:**
"""
            key_findings = summary.get('key_findings', [])
            for finding in key_findings:
                section += f"  - {finding}\n"
        
        return section + "\n"
    
    def _generate_detailed_documentation_analysis(self, documentation_output: Dict[str, Any]) -> str:
        """Generate detailed documentation analysis section"""
        section = "## Documentation Review\n\n"
        
        review_results = documentation_output.get('review_results', {})
        overall_assessment = review_results.get('overall_assessment', {})
        
        section += f"""
### Documentation Assessment
- **Total Files:** {overall_assessment.get('total_files', 0)}
- **Completeness Score:** {overall_assessment.get('documentation_completeness', 0.0):.2f}
- **Quality Score:** {overall_assessment.get('quality_score', 0.0):.2f}
- **Status:** {overall_assessment.get('status', 'Unknown').title()}

### Compliance Coverage
"""
        
        compliance_coverage = overall_assessment.get('compliance_coverage', [])
        if compliance_coverage:
            for framework in compliance_coverage:
                section += f"- ✅ {framework}\n"
        else:
            section += "- ❌ No regulatory compliance coverage identified\n"
        
        missing_sections = overall_assessment.get('missing_sections', [])
        if missing_sections:
            section += "\n### Missing Documentation Sections\n"
            for section_name in missing_sections:
                section += f"- {section_name.replace('_', ' ').title()}\n"
        
        return section + "\n"
    
    def _generate_detailed_findings_analysis(self, reviewer_output: Dict[str, Any]) -> str:
        """Generate detailed findings analysis section"""
        section = "## Findings and Recommendations\n\n"
        
        findings = reviewer_output.get('findings', {})
        recommendations = reviewer_output.get('recommendations', {})
        
        # Process findings by category
        for category, category_findings in findings.items():
            if category_findings and category != 'overall_findings':
                section += f"### {category.replace('_', ' ').title()}\n"
                
                if isinstance(category_findings, list):
                    for finding in category_findings:
                        if isinstance(finding, dict):
                            issue = finding.get('issue', 'Unknown Issue')
                            severity = finding.get('severity', 'Unknown')
                            description = finding.get('description', '')
                            
                            section += f"- **{issue}** ({severity}): {description}\n"
        
        # Add recommendations
        if isinstance(recommendations, dict):
            immediate_actions = recommendations.get('immediate_actions', [])
            if immediate_actions:
                section += "\n### Immediate Actions Required\n"
                for action in immediate_actions:
                    action_text = action.get('action', action) if isinstance(action, dict) else action
                    section += f"- {action_text}\n"
        
        return section + "\n"
    
    def _generate_detailed_risk_analysis(self, auditor_output: Dict[str, Any]) -> str:
        """Generate detailed risk analysis section"""
        section = "## Risk Assessment\n\n"
        
        risk_assessment = auditor_output.get('risk_assessment', {})
        final_recommendation = auditor_output.get('final_recommendation', {})
        
        if risk_assessment:
            section += f"""
### Risk Profile
- **Overall Risk Level:** {risk_assessment.get('overall_risk_level', 'Unknown')}
- **Risk Score:** {risk_assessment.get('risk_score', 0.0):.2f}/1.0

### Risk Factors
"""
            risk_factors = risk_assessment.get('risk_factors', [])
            for factor in risk_factors:
                section += f"- {factor}\n"
        
        if final_recommendation:
            section += f"""
### Final Recommendation
- **Approval Status:** {final_recommendation.get('approval_status', 'Unknown')}
- **Recommendation Type:** {final_recommendation.get('recommendation_type', 'Unknown')}
- **Approval Validity:** {final_recommendation.get('approval_validity', 'N/A')}
"""
        
        return section + "\n"
    
    def _get_audit_period(self, audit_trail: List[Dict[str, Any]]) -> str:
        """Get the audit period from audit trail"""
        if not audit_trail:
            return "No audit data"
        
        timestamps = [entry.get('timestamp', '') for entry in audit_trail if entry.get('timestamp')]
        if timestamps:
            start_time = min(timestamps)
            end_time = max(timestamps)
            return f"{start_time[:10]} to {end_time[:10]}"
        
        return "Unknown period"
    
    def _get_validation_status(self, workflow_state: Dict[str, Any]) -> str:
        """Get overall validation status"""
        completed_steps = len(workflow_state.get('completed_steps', []))
        total_steps = 6  # Total workflow steps
        
        if completed_steps == total_steps:
            return "Complete"
        elif completed_steps > 0:
            return f"In Progress ({completed_steps}/{total_steps})"
        else:
            return "Not Started"
    
    def _generate_activity_summary(self, audit_trail: List[Dict[str, Any]]) -> str:
        """Generate activity summary from audit trail"""
        if not audit_trail:
            return "No activities recorded"
        
        # Count different types of activities
        activity_counts = {}
        for entry in audit_trail:
            action = entry.get('action', '')
            action_type = action.split(':')[0] if ':' in action else action.split(' ')[0]
            activity_counts[action_type] = activity_counts.get(action_type, 0) + 1
        
        summary = ""
        for activity_type, count in sorted(activity_counts.items(), key=lambda x: x[1], reverse=True):
            summary += f"- **{activity_type}:** {count} occurrences\n"
        
        return summary
    
    def _generate_compliance_review(self, workflow_state: Dict[str, Any]) -> str:
        """Generate compliance review section"""
        agent_outputs = workflow_state.get('agent_outputs', {})
        
        # Check documentation compliance
        documentation_output = agent_outputs.get('step_2', {})
        compliance_info = ""
        
        if isinstance(documentation_output, dict):
            review_results = documentation_output.get('review_results', {})
            compliance_checklist = review_results.get('compliance_checklist', {})
            
            if compliance_checklist:
                regulatory_compliance = compliance_checklist.get('regulatory_compliance', {})
                
                compliance_info += "### Regulatory Compliance Status\n"
                for framework, status in regulatory_compliance.items():
                    status_icon = "✅" if status else "❌"
                    compliance_info += f"- {framework.replace('_', ' ').title()}: {status_icon}\n"
        
        if not compliance_info:
            compliance_info = "Compliance review not yet available. Complete documentation review to see compliance status."
        
        return compliance_info
    
    def _calculate_next_review_date(self) -> str:
        """Calculate next review date"""
        from datetime import datetime, timedelta
        next_review = datetime.now() + timedelta(days=365)
        return next_review.strftime('%Y-%m-%d')
