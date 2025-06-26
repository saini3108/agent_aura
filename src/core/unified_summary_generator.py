"""
Unified Summary Generator
========================

Single, clean, dynamic summary system that replaces all other summary generators.
Supports both LLM-powered and structured fallback summaries.
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class AgentSummary:
    """Clean agent summary structure"""
    title: str
    description: str
    impact: str
    recommendation: str
    severity: str  # critical, high, medium, low, info
    confidence: float
    key_metrics: Dict[str, str] = field(default_factory=dict)
    risk_flags: List[str] = field(default_factory=list)

@dataclass
class ExecutiveSummary:
    """Executive summary structure"""
    overall_assessment: str
    key_findings: List[str]
    critical_issues: List[str]
    recommendations: List[str]
    risk_level: str
    approval_status: str
    confidence_score: float
    executive_recommendation: str

class UnifiedSummaryGenerator:
    """Unified summary generator with LLM integration and structured fallbacks"""
    
    def __init__(self):
        self.llm_available = self._check_llm_availability()
        self.client = self._initialize_llm_client() if self.llm_available else None
        
    def _check_llm_availability(self) -> bool:
        """Check if LLM is available and configured"""
        groq_key = os.getenv('GROQ_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        return bool(groq_key or openai_key)
    
    def _initialize_llm_client(self):
        """Initialize the best available LLM client"""
        try:
            # Try Groq first (faster and cheaper)
            if os.getenv('GROQ_API_KEY'):
                from groq import Groq
                return Groq(api_key=os.getenv('GROQ_API_KEY'))
        except ImportError:
            pass
        
        try:
            # Fallback to OpenAI
            if os.getenv('OPENAI_API_KEY'):
                from openai import OpenAI
                return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        except ImportError:
            pass
        
        return None
    
    def generate_agent_summary(self, agent_name: str, raw_output: Dict[str, Any], context: Dict[str, Any] = None) -> AgentSummary:
        """Generate clean agent summary using best available method"""
        
        if self.llm_available and self.client:
            try:
                return self._generate_llm_summary(agent_name, raw_output, context)
            except Exception as e:
                logger.warning(f"LLM summary failed for {agent_name}: {e}")
        
        # Fallback to structured summary
        return self._generate_structured_summary(agent_name, raw_output, context)
    
    def _generate_llm_summary(self, agent_name: str, raw_output: Dict[str, Any], context: Dict[str, Any] = None) -> AgentSummary:
        """Generate summary using LLM"""
        
        prompt = self._create_summary_prompt(agent_name, raw_output)
        
        try:
            if hasattr(self.client, 'chat'):  # Groq/OpenAI style
                response = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant" if 'groq' in str(type(self.client)).lower() else "gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a senior credit risk analyst. Provide concise, professional summaries focused on business impact."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                return self._parse_llm_response(response.choices[0].message.content, agent_name, raw_output)
            else:
                return self._generate_structured_summary(agent_name, raw_output, context)
                
        except Exception as e:
            logger.warning(f"LLM API call failed: {e}")
            return self._generate_structured_summary(agent_name, raw_output, context)
    
    def _create_summary_prompt(self, agent_name: str, raw_output: Dict[str, Any]) -> str:
        """Create focused prompt for agent analysis"""
        
        # Format raw output for better LLM processing
        formatted_data = self._format_data_for_llm(raw_output)
        
        agent_contexts = {
            'validator': 'Model validation specialist focusing on AUC, KS statistics, PSI, and performance metrics',
            'analyst': 'Data quality analyst focusing on missing data, outliers, and data completeness',
            'documentation': 'Compliance reviewer focusing on regulatory requirements and documentation gaps',
            'reviewer': 'Risk assessor providing overall risk evaluation and recommendations',
            'auditor': 'Independent auditor providing final validation and approval decisions'
        }
        
        context = agent_contexts.get(agent_name, 'Credit risk specialist')
        
        return f"""As a {context}, analyze this data and provide a summary:

{formatted_data}

Format your response exactly as:
TITLE: [One clear line summary]
DESCRIPTION: [2-3 sentences explaining findings]
IMPACT: [Business impact]
RECOMMENDATION: [Clear next steps]
SEVERITY: [critical/high/medium/low/info]
CONFIDENCE: [0.XX as decimal]
METRICS: [Key metric: value; Key metric: value]
FLAGS: [Risk concern; Another concern]

Be concise and business-focused."""
    
    def _format_data_for_llm(self, raw_output: Dict[str, Any]) -> str:
        """Format raw output for LLM processing"""
        formatted_lines = []
        
        for key, value in raw_output.items():
            if key == 'metrics' and isinstance(value, dict):
                formatted_lines.append("METRICS:")
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        formatted_lines.append(f"  {k}: {v:.4f}")
            elif key == 'summary' and isinstance(value, dict):
                formatted_lines.append("SUMMARY:")
                for k, v in value.items():
                    formatted_lines.append(f"  {k}: {v}")
            elif isinstance(value, (str, int, float)):
                formatted_lines.append(f"{key}: {value}")
        
        return "\n".join(formatted_lines)
    
    def _parse_llm_response(self, response: str, agent_name: str, raw_output: Dict[str, Any]) -> AgentSummary:
        """Parse LLM response into structured format"""
        
        # Initialize defaults
        title = f"{agent_name.title()} Analysis Complete"
        description = "Analysis completed with findings available."
        impact = "Impact assessment completed."
        recommendation = "Review findings and proceed."
        severity = "medium"
        confidence = 0.8
        key_metrics = {}
        risk_flags = []
        
        # Parse response
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('TITLE:'):
                title = line.replace('TITLE:', '').strip()
            elif line.startswith('DESCRIPTION:'):
                description = line.replace('DESCRIPTION:', '').strip()
            elif line.startswith('IMPACT:'):
                impact = line.replace('IMPACT:', '').strip()
            elif line.startswith('RECOMMENDATION:'):
                recommendation = line.replace('RECOMMENDATION:', '').strip()
            elif line.startswith('SEVERITY:'):
                sev = line.replace('SEVERITY:', '').strip().lower()
                if sev in ['critical', 'high', 'medium', 'low', 'info']:
                    severity = sev
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.replace('CONFIDENCE:', '').strip())
                except ValueError:
                    confidence = 0.8
            elif line.startswith('METRICS:'):
                metrics_text = line.replace('METRICS:', '').strip()
                for metric_pair in metrics_text.split(';'):
                    if ':' in metric_pair:
                        k, v = metric_pair.split(':', 1)
                        key_metrics[k.strip()] = v.strip()
            elif line.startswith('FLAGS:'):
                flags_text = line.replace('FLAGS:', '').strip()
                risk_flags = [flag.strip() for flag in flags_text.split(';') if flag.strip()]
        
        return AgentSummary(
            title=title,
            description=description,
            impact=impact,
            recommendation=recommendation,
            severity=severity,
            confidence=confidence,
            key_metrics=key_metrics,
            risk_flags=risk_flags
        )
    
    def _generate_structured_summary(self, agent_name: str, raw_output: Dict[str, Any], context: Dict[str, Any] = None) -> AgentSummary:
        """Generate structured summary as fallback"""
        
        if agent_name == 'validator':
            return self._create_validator_summary(raw_output)
        elif agent_name == 'analyst':
            return self._create_analyst_summary(raw_output)
        elif agent_name == 'documentation':
            return self._create_documentation_summary(raw_output)
        elif agent_name == 'reviewer':
            return self._create_reviewer_summary(raw_output)
        elif agent_name == 'auditor':
            return self._create_auditor_summary(raw_output)
        else:
            return self._create_generic_summary(agent_name, raw_output)
    
    def _create_validator_summary(self, raw_output: Dict[str, Any]) -> AgentSummary:
        """Create validator summary"""
        metrics = raw_output.get('metrics', {})
        auc = metrics.get('auc', 0.0)
        ks_stat = metrics.get('ks_statistic', 0.0)
        psi = metrics.get('psi', 0.0)
        
        if auc >= 0.8:
            performance, severity = "Excellent", "info"
        elif auc >= 0.7:
            performance, severity = "Good", "low"
        elif auc >= 0.6:
            performance, severity = "Acceptable", "medium"
        else:
            performance, severity = "Poor", "high"
        
        risk_flags = []
        if auc < 0.6:
            risk_flags.append("Poor model discrimination below acceptable threshold")
        if psi > 0.25:
            risk_flags.append("High population instability detected")
        
        return AgentSummary(
            title=f"Model Validation Complete - {performance} Performance",
            description=f"Model shows {performance.lower()} discriminatory power with AUC of {auc:.3f}. Population stability analysis indicates {'stable' if psi <= 0.1 else 'moderate drift' if psi <= 0.25 else 'significant drift'} conditions.",
            impact=f"Model performance is {'suitable for production' if auc >= 0.7 else 'below recommended thresholds and requires improvement'}.",
            recommendation="Proceed with deployment" if auc >= 0.7 else "Model requires improvement before deployment",
            severity=severity,
            confidence=0.9,
            key_metrics={
                "AUC Score": f"{auc:.3f}",
                "KS Statistic": f"{ks_stat:.3f}",
                "PSI": f"{psi:.3f}",
                "Performance": performance
            },
            risk_flags=risk_flags
        )
    
    def _create_analyst_summary(self, raw_output: Dict[str, Any]) -> AgentSummary:
        """Create analyst summary"""
        analysis = raw_output.get('analysis', {})
        data_quality = analysis.get('data_quality', {})
        missing_pct = data_quality.get('missing_percentage', 0)
        outlier_pct = data_quality.get('outlier_percentage', 0)
        
        if missing_pct <= 5 and outlier_pct <= 10:
            quality, severity = "Excellent", "info"
        elif missing_pct <= 15 and outlier_pct <= 20:
            quality, severity = "Good", "low"
        else:
            quality, severity = "Acceptable", "medium"
        
        risk_flags = []
        if missing_pct > 20:
            risk_flags.append("High missing data rate may impact model performance")
        if outlier_pct > 25:
            risk_flags.append("Excessive outliers detected")
        
        return AgentSummary(
            title=f"Data Analysis Complete - {quality} Data Quality",
            description=f"Data quality assessment shows {missing_pct:.1f}% missing values and {outlier_pct:.1f}% outliers. Overall quality is {quality.lower()} for modeling.",
            impact=f"Data is {'ready for modeling' if severity == 'info' else 'suitable with preprocessing' if severity == 'low' else 'requires significant cleaning'}.",
            recommendation="Proceed with modeling" if severity in ['info', 'low'] else "Apply data preprocessing",
            severity=severity,
            confidence=0.85,
            key_metrics={
                "Missing Data": f"{missing_pct:.1f}%",
                "Outliers": f"{outlier_pct:.1f}%",
                "Quality": quality
            },
            risk_flags=risk_flags
        )
    
    def _create_documentation_summary(self, raw_output: Dict[str, Any]) -> AgentSummary:
        """Create documentation summary"""
        review = raw_output.get('review', {})
        compliance = review.get('compliance_assessment', {})
        score = compliance.get('overall_score', 75)
        
        if score >= 90:
            status, severity = "Fully Compliant", "info"
        elif score >= 75:
            status, severity = "Largely Compliant", "low"
        else:
            status, severity = "Partially Compliant", "medium"
        
        return AgentSummary(
            title=f"Documentation Review Complete - {status}",
            description=f"Documentation compliance review achieves {score:.0f}% score. Status: {status.lower()}.",
            impact=f"Documentation is {'sufficient for approval' if score >= 80 else 'adequate with minor gaps' if score >= 65 else 'insufficient'}.",
            recommendation="Approve documentation" if score >= 85 else "Address documentation gaps",
            severity=severity,
            confidence=0.8,
            key_metrics={
                "Compliance Score": f"{score:.0f}%",
                "Status": status
            },
            risk_flags=[] if score >= 70 else ["Documentation gaps may delay approval"]
        )
    
    def _create_reviewer_summary(self, raw_output: Dict[str, Any]) -> AgentSummary:
        """Create reviewer summary"""
        findings = raw_output.get('findings', {})
        risk_assessment = findings.get('risk_assessment', {})
        risk_level = risk_assessment.get('overall_risk', 'medium')
        confidence = risk_assessment.get('confidence', 0.8)
        
        severity_map = {'low': 'info', 'medium': 'low', 'high': 'medium', 'critical': 'high'}
        
        return AgentSummary(
            title=f"Risk Review Complete - {risk_level.title()} Risk Profile",
            description=f"Comprehensive risk assessment shows {risk_level} risk profile with {confidence:.0%} confidence.",
            impact=f"Model presents {risk_level} risk for deployment with {'standard' if risk_level == 'low' else 'enhanced'} monitoring required.",
            recommendation="Approve for deployment" if risk_level == 'low' else "Approve with monitoring",
            severity=severity_map.get(risk_level, 'medium'),
            confidence=confidence,
            key_metrics={
                "Risk Level": risk_level.title(),
                "Confidence": f"{confidence:.0%}"
            },
            risk_flags=[] if risk_level in ['low', 'medium'] else ["High risk requires management attention"]
        )
    
    def _create_auditor_summary(self, raw_output: Dict[str, Any]) -> AgentSummary:
        """Create auditor summary"""
        audit = raw_output.get('audit', {})
        final_rec = audit.get('final_recommendation', {})
        status = final_rec.get('status', 'conditional')
        score = final_rec.get('score', 75)
        
        status_map = {
            'approved': ('Approved', 'info'),
            'conditional': ('Conditional Approval', 'low'),
            'rejected': ('Rejected', 'high')
        }
        
        status_text, severity = status_map.get(status, ('Under Review', 'medium'))
        
        return AgentSummary(
            title=f"Final Audit Complete - {status_text}",
            description=f"Independent audit completed with {status_text.lower()} and score of {score:.0f}%.",
            impact=f"Model {'meets requirements for deployment' if status == 'approved' else 'meets requirements with conditions' if status == 'conditional' else 'does not meet requirements'}.",
            recommendation="Deploy to production" if status == 'approved' else "Deploy with conditions",
            severity=severity,
            confidence=0.95,
            key_metrics={
                "Status": status_text,
                "Audit Score": f"{score:.0f}%"
            },
            risk_flags=[] if status == 'approved' else ["Conditional approval requires monitoring"]
        )
    
    def _create_generic_summary(self, agent_name: str, raw_output: Dict[str, Any]) -> AgentSummary:
        """Create generic summary"""
        return AgentSummary(
            title=f"{agent_name.title()} Analysis Complete",
            description="Agent analysis completed with findings available for review.",
            impact="Analysis provides insights for validation process continuation.",
            recommendation="Review findings and proceed with next validation step.",
            severity="info",
            confidence=0.8,
            key_metrics={"Status": "Complete"},
            risk_flags=[]
        )
    
    def generate_executive_summary(self, workflow_results: Dict[str, Any], context: Dict[str, Any] = None) -> ExecutiveSummary:
        """Generate executive summary from workflow results"""
        
        if self.llm_available and self.client:
            try:
                return self._generate_llm_executive_summary(workflow_results)
            except Exception as e:
                logger.warning(f"LLM executive summary failed: {e}")
        
        return self._generate_structured_executive_summary(workflow_results)
    
    def _generate_llm_executive_summary(self, workflow_results: Dict[str, Any]) -> ExecutiveSummary:
        """Generate executive summary using LLM"""
        
        # Prepare summary data
        summary_data = []
        for agent_name, result in workflow_results.items():
            if agent_name == 'human_review':
                continue
            summary = result.get('clean_summary') or result.get('summary')
            if summary and hasattr(summary, 'title'):
                summary_data.append(f"{agent_name.upper()}: {summary.title} - {summary.recommendation}")
            elif isinstance(summary, dict):
                title = summary.get('title', f"{agent_name} completed")
                rec = summary.get('recommendation', 'Review findings')
                summary_data.append(f"{agent_name.upper()}: {title} - {rec}")
        
        prompt = f"""As a Senior Credit Risk Officer, provide an executive summary:

VALIDATION RESULTS:
{chr(10).join(summary_data)}

Format as:
ASSESSMENT: [Executive assessment]
FINDINGS: [Finding 1; Finding 2; Finding 3]
ISSUES: [Critical issue 1; Critical issue 2]
RECOMMENDATIONS: [Recommendation 1; Recommendation 2]
RISK: [Low/Medium/High/Critical]
STATUS: [Approved/Conditional/Rejected]
CONFIDENCE: [0.XX]
DECISION: [One sentence final recommendation]

Be concise and strategic."""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant" if 'groq' in str(type(self.client)).lower() else "gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a Senior Credit Risk Officer providing executive briefings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=600
            )
            
            return self._parse_executive_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.warning(f"Executive LLM call failed: {e}")
            return self._generate_structured_executive_summary(workflow_results)
    
    def _parse_executive_response(self, response: str) -> ExecutiveSummary:
        """Parse LLM executive response"""
        
        # Defaults
        assessment = "Model validation process completed with findings available."
        findings = ["Validation process completed"]
        issues = []
        recommendations = ["Review detailed findings"]
        risk_level = "Medium"
        status = "Conditional"
        confidence = 0.8
        decision = "Proceed with caution."
        
        # Parse response
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('ASSESSMENT:'):
                assessment = line.replace('ASSESSMENT:', '').strip()
            elif line.startswith('FINDINGS:'):
                findings_text = line.replace('FINDINGS:', '').strip()
                findings = [f.strip() for f in findings_text.split(';') if f.strip()]
            elif line.startswith('ISSUES:'):
                issues_text = line.replace('ISSUES:', '').strip()
                issues = [i.strip() for i in issues_text.split(';') if i.strip()]
            elif line.startswith('RECOMMENDATIONS:'):
                rec_text = line.replace('RECOMMENDATIONS:', '').strip()
                recommendations = [r.strip() for r in rec_text.split(';') if r.strip()]
            elif line.startswith('RISK:'):
                risk_level = line.replace('RISK:', '').strip()
            elif line.startswith('STATUS:'):
                status = line.replace('STATUS:', '').strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.replace('CONFIDENCE:', '').strip())
                except ValueError:
                    confidence = 0.8
            elif line.startswith('DECISION:'):
                decision = line.replace('DECISION:', '').strip()
        
        return ExecutiveSummary(
            overall_assessment=assessment,
            key_findings=findings,
            critical_issues=issues,
            recommendations=recommendations,
            risk_level=risk_level,
            approval_status=status,
            confidence_score=confidence,
            executive_recommendation=decision
        )
    
    def _generate_structured_executive_summary(self, workflow_results: Dict[str, Any]) -> ExecutiveSummary:
        """Generate structured executive summary as fallback"""
        
        # Analyze workflow results
        completed = len([r for r in workflow_results.values() if r.get('status') == 'completed'])
        total = len(workflow_results)
        
        # Determine risk and status from results
        risk_indicators = []
        for result in workflow_results.values():
            summary = result.get('clean_summary') or result.get('summary')
            if summary:
                if hasattr(summary, 'severity'):
                    risk_indicators.append(summary.severity)
                elif isinstance(summary, dict):
                    risk_indicators.append(summary.get('severity', 'medium'))
        
        if 'critical' in risk_indicators or 'high' in risk_indicators:
            risk_level = "High"
            status = "Conditional"
        elif 'medium' in risk_indicators:
            risk_level = "Medium"
            status = "Conditional"
        else:
            risk_level = "Low"
            status = "Approved"
        
        return ExecutiveSummary(
            overall_assessment=f"Model validation process {completed}/{total} agents completed. Risk level: {risk_level}. Status: {status}.",
            key_findings=[
                f"Validation process {(completed/total)*100:.0f}% complete",
                f"Overall risk assessed as {risk_level.lower()}",
                f"Current approval status: {status.lower()}"
            ],
            critical_issues=[] if risk_level != "High" else ["High risk factors identified"],
            recommendations=[
                "Review detailed agent findings",
                "Proceed with standard monitoring" if risk_level == "Low" else "Implement enhanced monitoring",
                "Ensure compliance documentation complete"
            ],
            risk_level=risk_level,
            approval_status=status,
            confidence_score=0.8,
            executive_recommendation=f"{'Proceed with deployment' if status == 'Approved' else 'Proceed with enhanced monitoring and conditions'}."
        )