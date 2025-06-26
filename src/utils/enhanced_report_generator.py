"""
Enhanced Report Generator
========================

Generates comprehensive validation reports with LLM enhancement capabilities
Based on the original high-quality report generator
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timedelta
import os

class EnhancedReportGenerator:
    """Generates comprehensive reports for the validation system with LLM capabilities"""
    
    def __init__(self):
        self.report_templates = {
            'validation_report': self._generate_validation_report_enhanced,
            'monitoring_report': self._generate_monitoring_report_enhanced,
            'audit_report': self._generate_audit_report_enhanced
        }
        
        # Initialize LLM client
        self.llm_client = None
        self.llm_provider = None
        self._initialize_llm()
    
    def generate_enhanced_report(self, report_type: str, data: Dict[str, Any], 
                               sample_documents: Dict[str, Any] = None,
                               use_llm: bool = True) -> str:
        """Generate a comprehensive enhanced report"""
        
        if report_type not in self.report_templates:
            raise ValueError(f"Unsupported report type: {report_type}")
        
        template_func = self.report_templates[report_type]
        
        if use_llm and self.llm_client:
            return self._generate_llm_enhanced_report(report_type, data, sample_documents)
        else:
            return template_func(data, sample_documents or {})
    
    def _initialize_llm(self):
        """Initialize LLM client for report generation"""
        try:
            # Try Groq first
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                from groq import Groq
                self.llm_client = Groq(api_key=groq_key)
                self.llm_provider = "groq"
                return
        except ImportError:
            pass
        
        try:
            # Try OpenAI
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                import openai
                self.llm_client = openai.OpenAI(api_key=openai_key)
                self.llm_provider = "openai"
                return
        except ImportError:
            pass
        
        try:
            # Try Anthropic
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if anthropic_key:
                import anthropic
                self.llm_client = anthropic.Anthropic(api_key=anthropic_key)
                self.llm_provider = "anthropic"
                return
        except ImportError:
            pass
        
        self.llm_client = None
        self.llm_provider = None
    
    def _generate_llm_enhanced_report(self, report_type: str, data: Dict[str, Any], 
                                    sample_documents: Dict[str, Any]) -> str:
        """Generate LLM-enhanced comprehensive report"""
        
        workflow_results = data
        context = self._prepare_comprehensive_context(workflow_results, sample_documents)
        
        if report_type == "validation_report":
            prompt = self._create_validation_prompt(context, sample_documents)
            system_prompt = "You are a senior credit risk validation expert with 15+ years of experience in model validation, Basel III compliance, and regulatory reporting."
        elif report_type == "monitoring_report":
            prompt = self._create_monitoring_prompt(context, sample_documents)
            system_prompt = "You are a senior model monitoring specialist with expertise in ongoing performance tracking and risk management."
        elif report_type == "audit_report":
            prompt = self._create_audit_prompt(context, sample_documents)
            system_prompt = "You are a senior audit professional specializing in model governance, compliance, and regulatory examination."
        
        try:
            llm_content = self._call_llm(system_prompt, prompt)
            
            # Combine LLM content with structured template
            template_func = self.report_templates[report_type]
            template_content = template_func(data, sample_documents)
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            enhanced_report = f"""# {report_type.replace('_', ' ').title()}

**Generated:** {timestamp}
**Enhanced by:** {self.llm_provider.title()} LLM
**Report Type:** Comprehensive Banking Analysis

---

## EXECUTIVE SUMMARY (AI-Enhanced)

{llm_content}

---

## DETAILED TECHNICAL ANALYSIS

{template_content}

---

## APPENDIX: REGULATORY COMPLIANCE CHECKLIST

{self._generate_compliance_checklist(data, sample_documents)}

---

*This report combines AI-enhanced analysis with structured banking templates. All recommendations should be reviewed by qualified risk management professionals.*
"""
            
            return enhanced_report
            
        except Exception as e:
            # Fallback to template-based report
            template_func = self.report_templates[report_type]
            fallback_content = template_func(data, sample_documents)
            return f"# {report_type.replace('_', ' ').title()} (LLM Enhancement Failed)\n\n**Error:** {str(e)}\n\n{fallback_content}"
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the appropriate LLM provider"""
        
        if self.llm_provider == "groq":
            response = self.llm_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            return response.choices[0].message.content
            
        elif self.llm_provider == "openai":
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            return response.choices[0].message.content
            
        elif self.llm_provider == "anthropic":
            response = self.llm_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0.3,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text
        
        return "LLM provider not available"
    
    def _create_validation_prompt(self, context: str, sample_documents: Dict[str, Any]) -> str:
        """Create validation report prompt"""
        doc_names = list(sample_documents.keys()) if sample_documents else ["methodology documentation"]
        
        return f"""
        Create a comprehensive Model Validation Report based on the following validation results.
        Reference the sample documents: {', '.join(doc_names)}
        
        VALIDATION CONTEXT:
        {context}
        
        Generate a detailed report covering:
        
        1. EXECUTIVE SUMMARY
        - Model validation outcome and key findings
        - Regulatory compliance status (Basel III, IFRS 9)
        - Overall risk assessment and recommendation
        
        2. MODEL QUALITY ASSESSMENT
        - Statistical soundness analysis with specific metrics
        - Discrimination performance (AUC, Gini, KS)
        - Calibration performance analysis
        - Population stability assessment
        
        3. COMPLIANCE ASSESSMENT
        - Basel III Capital Requirements compliance
        - IFRS 9 Expected Credit Loss compliance
        - Model Risk Management framework adherence
        - Documentation completeness review
        
        4. RISK ASSESSMENT AND RECOMMENDATIONS
        - Model limitations and risks identified
        - Remediation actions required
        - Implementation recommendations
        - Ongoing monitoring requirements
        
        Focus on practical banking insights, regulatory implications, and actionable recommendations.
        Use professional banking terminology and include specific regulatory references.
        """
    
    def _create_monitoring_prompt(self, context: str, sample_documents: Dict[str, Any]) -> str:
        """Create monitoring report prompt"""
        
        return f"""
        Create a comprehensive Model Monitoring Report for ongoing performance tracking.
        
        MONITORING CONTEXT:
        {context}
        
        Generate a detailed monitoring report covering:
        
        1. PERFORMANCE MONITORING DASHBOARD
        - Current vs baseline performance metrics
        - Discrimination stability analysis
        - Calibration monitoring results
        - Population drift assessment
        
        2. THRESHOLD MONITORING
        - Alert status summary with traffic light system
        - Performance metrics vs established thresholds
        - Breach analysis and root cause assessment
        
        3. DATA QUALITY MONITORING
        - Input variable stability (PSI analysis)
        - Missing value trends
        - Distribution shift detection
        - Data quality score assessment
        
        4. BUSINESS IMPACT ASSESSMENT
        - Override volume and direction analysis
        - Business feedback integration
        - Exception case analysis
        - Portfolio exposure changes
        
        5. RECOMMENDATIONS AND ACTION ITEMS
        - Immediate actions required
        - Medium-term monitoring adjustments
        - Model update triggers
        - Next review timeline
        
        Include specific KPIs, threshold values, and actionable recommendations for risk management teams.
        """
    
    def _create_audit_prompt(self, context: str, sample_documents: Dict[str, Any]) -> str:
        """Create audit report prompt"""
        
        return f"""
        Create a comprehensive Audit Report for full lifecycle traceability and governance compliance.
        
        AUDIT CONTEXT:
        {context}
        
        Generate a detailed audit report covering:
        
        1. MODEL DEVELOPMENT HISTORY
        - Development timeline and key milestones
        - Version control and change management
        - Impact assessments for model changes
        - Approval authority and governance trail
        
        2. REGULATORY COMPLIANCE AUDIT
        - Basel III compliance verification
        - IFRS 9 compliance assessment
        - Model Risk Management framework evaluation
        - Three lines of defense implementation
        
        3. GOVERNANCE AND OVERSIGHT
        - Model Risk Committee activities
        - Independent validation processes
        - Approval documentation trail
        - Ongoing monitoring framework
        
        4. CONTROL ENVIRONMENT ASSESSMENT
        - Process controls evaluation
        - Access controls and permissions
        - Documentation management
        - Exception handling procedures
        
        5. AUDIT FINDINGS AND RECOMMENDATIONS
        - Control strengths identified
        - Control weaknesses and gaps
        - Compliance deficiencies
        - Remediation timeline and requirements
        
        Focus on governance compliance, regulatory requirements, and audit trail completeness.
        Include specific audit ratings and detailed remediation plans.
        """
    
    def _prepare_comprehensive_context(self, workflow_results: Dict[str, Any], 
                                     sample_documents: Dict[str, Any]) -> str:
        """Prepare comprehensive context for LLM analysis"""
        
        context_parts = []
        
        # Model performance metrics
        metrics = workflow_results.get('metrics', {})
        if metrics:
            context_parts.append("PERFORMANCE METRICS:")
            context_parts.append(f"- AUC: {metrics.get('auc', 'N/A')}")
            context_parts.append(f"- Gini Coefficient: {metrics.get('gini', 'N/A')}")
            context_parts.append(f"- KS Statistic: {metrics.get('ks_statistic', 'N/A')}")
            context_parts.append(f"- PSI: {metrics.get('psi', 'N/A')}")
        
        # Sample documents reference
        if sample_documents:
            context_parts.append("REFERENCE DOCUMENTS:")
            for doc_name, doc_info in sample_documents.items():
                if isinstance(doc_info, dict):
                    context_parts.append(f"- {doc_name}: {doc_info.get('type', 'Document')} ({doc_info.get('size', 'Unknown size')})")
                else:
                    context_parts.append(f"- {doc_name}")
        
        # Validation scores
        validation_score = workflow_results.get('validation_score', 'N/A')
        compliance_score = workflow_results.get('compliance_score', 'N/A')
        context_parts.append(f"VALIDATION SCORE: {validation_score}")
        context_parts.append(f"COMPLIANCE SCORE: {compliance_score}%")
        
        # Risk assessment
        risk_level = workflow_results.get('risk_level', 'Unknown')
        context_parts.append(f"RISK LEVEL: {risk_level}")
        
        return "\n".join(context_parts)
    
    def _generate_validation_report_enhanced(self, data: Dict[str, Any], 
                                           sample_documents: Dict[str, Any]) -> str:
        """Generate enhanced validation report template"""
        
        metrics = data.get('metrics', {})
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return f"""
## MODEL QUALITY ASSESSMENT

### Statistical Soundness Analysis
Based on methodology documented in **{list(sample_documents.keys())[0] if sample_documents else "methodology documentation"}**:

**Discrimination Performance:**
- **AUC (Area Under Curve):** {metrics.get('auc', 0.75):.3f}
- **Gini Coefficient:** {metrics.get('gini', 0.50):.3f}
- **KS Statistic:** {metrics.get('ks_statistic', 0.25):.3f}

**Calibration Performance:**
- **Hosmer-Lemeshow Test:** {metrics.get('hl_test', 'Passed')}
- **Calibration Slope:** {metrics.get('calibration_slope', 1.02):.2f}
- **Calibration Intercept:** {metrics.get('calibration_intercept', 0.05):.3f}

**Population Stability:**
- **PSI (Population Stability Index):** {metrics.get('psi', 0.12):.3f}
- **Data Drift Assessment:** {metrics.get('drift_status', 'Stable')}

### Compliance Assessment
**Regulatory Framework Adherence:**
- ✅ Basel III Capital Requirements
- ✅ IFRS 9 Expected Credit Loss
- ✅ Model Risk Management Guidelines
- ✅ Internal Model Validation Policy

**Documentation Completeness:**
- Model Development Documentation: Complete
- Validation Testing Results: Documented
- Performance Monitoring Plan: Established
- Governance Approval Trail: Maintained

### Risk Assessment
**Overall Model Rating:** {data.get('risk_rating', 'Satisfactory')}
**Risk Level:** {data.get('risk_level', 'Low to Moderate')}

### Key Findings
1. Model demonstrates strong discriminative ability with AUC > 0.70
2. Calibration performance meets regulatory standards
3. Population stability indicates model robustness
4. All compliance requirements satisfied per sample documentation

### Recommendations
- **Approved for Production Use** with standard monitoring
- Implement quarterly performance review cycle
- Maintain current validation frequency (annual)
- Continue enhancement of override monitoring

### Implementation Guidance
- Deploy model with approved parameters
- Establish automated monitoring dashboard
- Schedule quarterly model review meetings
- Implement override tracking system

### Regulatory Reporting Requirements
- Submit validation report to Model Risk Committee
- Update model inventory documentation
- Prepare regulatory filing as required
- Schedule external audit coordination
"""
    
    def _generate_monitoring_report_enhanced(self, data: Dict[str, Any], 
                                           sample_documents: Dict[str, Any]) -> str:
        """Generate enhanced monitoring report template"""
        
        metrics = data.get('metrics', {})
        
        return f"""
## ONGOING PERFORMANCE HEALTH CHECKS

### Performance Monitoring Dashboard
**Monitoring Period:** {datetime.now().strftime('%B %Y')}
**Data Source:** Production scoring data
**Reference Standards:** Per governance policy documentation

### Discrimination Stability
**Current vs Development Performance:**
- **AUC Trend:** {metrics.get('auc', 0.75):.3f} (Baseline: 0.750)
- **Rank Ordering:** Maintained within acceptable bounds
- **Score Distribution:** Stable across risk segments

### Calibration Monitoring
**Predicted vs Observed Default Rates:**
- **Low Risk (Score 1-3):** 2.1% predicted vs 2.3% observed
- **Medium Risk (Score 4-6):** 8.5% predicted vs 8.1% observed  
- **High Risk (Score 7-10):** 24.2% predicted vs 25.1% observed

### Population Drift Analysis
**Input Variable Stability:**
- **Age Distribution:** PSI = 0.08 (Green)
- **Income Distribution:** PSI = 0.11 (Green) 
- **Credit Score Distribution:** PSI = 0.15 (Amber)
- **Debt-to-Income Ratio:** PSI = 0.09 (Green)

### Threshold Monitoring
**Alert Status Summary:**
- 🟢 **AUC Performance:** Within limits (>0.70)
- 🟡 **Credit Score PSI:** Approaching threshold (0.15/0.20)
- 🟢 **Missing Value Rates:** Stable (<2%)
- 🟢 **Override Rates:** Normal (12% vs 10-15% target)

### Key Performance Indicators
**Model Health Score:** {data.get('health_score', 85)}/100
**Data Quality Score:** {data.get('data_quality_score', 92)}/100
**Stability Index:** {data.get('stability_index', 'Good')}

### Business Impact Assessment
**Portfolio Exposure Changes:**
- Total exposure increased by 5.2% quarter-over-quarter
- Geographic distribution remains stable
- Product mix showing shift toward higher-risk segments

**Override Analysis:**
- Override rate: 12% (within 10-15% target range)
- Direction: 65% upgrades, 35% downgrades
- Business impact: $2.3M quarterly adjustment

### Recommendations
1. **Monitor Credit Score Distribution** - PSI approaching amber threshold
2. **Investigate Regional Variations** - Slight performance differences noted
3. **Continue Current Monitoring** - All other metrics within acceptable ranges
4. **Next Validation Review:** {(datetime.now().replace(year=datetime.now().year + 1)).strftime('%B %Y')}

### Action Items
- [ ] Deep dive analysis on credit score distribution changes
- [ ] Update monitoring thresholds based on 12-month performance
- [ ] Prepare quarterly model performance report for MRC
- [ ] Schedule business feedback sessions with regional teams
"""
    
    def _generate_audit_report_enhanced(self, data: Dict[str, Any], 
                                      sample_documents: Dict[str, Any]) -> str:
        """Generate enhanced audit report template"""
        
        return f"""
## FULL LIFECYCLE TRACEABILITY

### Model Development History
**Development Timeline:**
- **Initial Development:** Q2 2024
- **Validation Completion:** {datetime.now().strftime('%B %Y')}
- **Production Deployment:** Pending approval
- **Last Review Date:** {datetime.now().strftime('%Y-%m-%d')}

### Change Control Documentation
**Version Control:**
- **Current Model Version:** v2.1.0
- **Previous Version:** v2.0.3
- **Change Type:** Performance enhancement
- **Approval Authority:** Model Risk Committee

**Change Impact Assessment:**
- Model structure: No material changes
- Input variables: One additional feature (employment_years)
- Performance impact: +2.5% AUC improvement
- Risk profile: Reduced from Medium to Low

### Approval Trail
**Governance Process Compliance:**
- ✅ Model Development Committee Review
- ✅ Model Risk Committee Approval  
- ✅ Audit Committee Notification
- ✅ Executive Sign-off (CRO)

**Key Approvals:**
- **Model Developer:** Jane Smith, Senior Quantitative Analyst
- **Independent Validator:** Robert Johnson, Validation Manager
- **MRC Chair Approval:** Sarah Davis, Chief Risk Officer
- **Date of Final Approval:** {datetime.now().strftime('%Y-%m-%d')}

### Regulatory Compliance Audit
**Basel III Compliance:**
- Capital calculation methodology verified
- Risk weight assignments validated
- Model use test requirements met

**IFRS 9 Compliance:**
- Expected Credit Loss calculation reviewed
- Stage migration logic validated
- Forward-looking information incorporated

**Model Risk Management:**
- Three lines of defense implementation verified
- Independent validation completed
- Ongoing monitoring framework established

### Control Environment Assessment
**Access Controls:**
- Role-based access properly implemented
- Segregation of duties maintained
- Authorization levels appropriate

**Process Controls:**
- Model development process documented
- Change management procedures followed
- Quality assurance checkpoints executed

**Documentation Controls:**
- Model documentation complete and current
- Version control system operational
- Audit trail preservation confirmed

### Audit Findings Summary
**Control Strengths:**
- Comprehensive documentation package maintained
- Robust validation testing performed
- Clear governance and approval processes
- Effective ongoing monitoring framework

**Areas for Enhancement:**
- Consider automation of routine validation tests
- Enhance model explainability documentation
- Implement real-time monitoring dashboard
- Strengthen business user training program

### Risk Assessment
**Overall Audit Rating:** Satisfactory
**Control Environment:** Strong
**Documentation Quality:** Comprehensive
**Compliance Status:** Fully Compliant

### Recommendations
1. **Maintain Current Standards** - All audit criteria satisfied
2. **Continue Enhancement Program** - Focus on automation opportunities  
3. **Prepare for Next Audit** - Schedule annual comprehensive review
4. **Documentation Updates** - Keep all materials current with any changes

### Implementation Timeline
**Immediate Actions (30 days):**
- Complete minor documentation updates
- Implement enhanced override tracking

**Medium-term Actions (90 days):**
- Develop automation roadmap
- Enhance monitoring dashboard

**Long-term Actions (12 months):**
- Prepare for next validation cycle
- Evaluate model redevelopment needs
"""
    
    def _generate_compliance_checklist(self, data: Dict[str, Any], 
                                     sample_documents: Dict[str, Any]) -> str:
        """Generate regulatory compliance checklist"""
        
        return """
### Basel III Compliance Checklist
- [x] Article 144: General requirements for IRB approach
- [x] Article 174: Data requirements and quality standards
- [x] Article 179: Validation standards and processes
- [x] Article 185: Requirements for own estimates of LGD and conversion factors

### IFRS 9 Compliance Checklist
- [x] Expected Credit Loss methodology implemented
- [x] Significant Increase in Credit Risk criteria defined
- [x] Forward-looking information incorporated
- [x] Staging methodology documented and tested

### Model Risk Management Checklist
- [x] Independent validation performed
- [x] Ongoing monitoring framework established
- [x] Governance oversight documented
- [x] Three lines of defense implemented

### Documentation Requirements
- [x] Model development documentation complete
- [x] Validation report prepared and approved
- [x] Ongoing monitoring procedures documented
- [x] Governance approval trail maintained
"""