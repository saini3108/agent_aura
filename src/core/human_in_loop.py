"""
Human-in-the-Loop Integration
============================

Implements human review checkpoints and feedback integration
for the ValiCred-AI validation workflow.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid
import asyncio

logger = logging.getLogger(__name__)

class ReviewStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    ESCALATED = "escalated"

class ReviewPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ReviewRequest:
    """Represents a human review request"""
    id: str
    workflow_id: str
    step_name: str
    agent_output: Dict[str, Any]
    review_questions: List[str]
    context: Dict[str, Any]
    priority: ReviewPriority
    created_at: datetime
    status: ReviewStatus = ReviewStatus.PENDING
    reviewer_id: Optional[str] = None
    feedback: Dict[str, Any] = field(default_factory=dict)
    decision_reason: Optional[str] = None
    reviewed_at: Optional[datetime] = None

@dataclass
class ReviewFeedback:
    """Structured feedback from human reviewers"""
    decision: ReviewStatus
    comments: str
    specific_feedback: Dict[str, Any]
    suggested_modifications: List[str]
    confidence_level: int  # 1-5 scale
    reviewer_expertise: str
    additional_data_needed: List[str] = field(default_factory=list)

class HumanInLoopManager:
    """Manages human-in-the-loop interactions for workflow validation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pending_reviews: Dict[str, ReviewRequest] = {}
        self.completed_reviews: Dict[str, ReviewRequest] = {}
        self.review_callbacks: Dict[str, Callable] = {}
        self.auto_escalation_hours = config.get('auto_escalation_hours', 24)
        
    def create_review_checkpoint(self, workflow_id: str, step_name: str, 
                                agent_output: Dict[str, Any], 
                                review_questions: List[str] = None) -> str:
        """Create a human review checkpoint"""
        
        review_id = str(uuid.uuid4())
        
        # Determine priority based on agent output
        priority = self._assess_review_priority(agent_output, step_name)
        
        # Generate context-aware review questions if not provided
        if not review_questions:
            review_questions = self._generate_review_questions(step_name, agent_output)
        
        review_request = ReviewRequest(
            id=review_id,
            workflow_id=workflow_id,
            step_name=step_name,
            agent_output=agent_output,
            review_questions=review_questions,
            context=self._build_review_context(workflow_id, step_name, agent_output),
            priority=priority,
            created_at=datetime.now()
        )
        
        self.pending_reviews[review_id] = review_request
        
        logger.info(f"Created review checkpoint {review_id} for workflow {workflow_id}, step {step_name}")
        return review_id
    
    def submit_human_feedback(self, review_id: str, feedback: ReviewFeedback, 
                             reviewer_id: str) -> bool:
        """Submit human feedback for a review request"""
        
        if review_id not in self.pending_reviews:
            logger.error(f"Review {review_id} not found in pending reviews")
            return False
        
        review_request = self.pending_reviews[review_id]
        review_request.status = feedback.decision
        review_request.reviewer_id = reviewer_id
        review_request.feedback = {
            "decision": feedback.decision.value,
            "comments": feedback.comments,
            "specific_feedback": feedback.specific_feedback,
            "suggested_modifications": feedback.suggested_modifications,
            "confidence_level": feedback.confidence_level,
            "reviewer_expertise": feedback.reviewer_expertise,
            "additional_data_needed": feedback.additional_data_needed
        }
        review_request.decision_reason = feedback.comments
        review_request.reviewed_at = datetime.now()
        
        # Move to completed reviews
        self.completed_reviews[review_id] = review_request
        del self.pending_reviews[review_id]
        
        # Execute callback if registered
        if review_id in self.review_callbacks:
            callback = self.review_callbacks[review_id]
            try:
                callback(review_request)
            except Exception as e:
                logger.error(f"Review callback failed for {review_id}: {e}")
        
        logger.info(f"Review {review_id} completed with decision: {feedback.decision.value}")
        return True
    
    def get_pending_reviews(self, reviewer_id: Optional[str] = None, 
                           priority_filter: Optional[ReviewPriority] = None) -> List[ReviewRequest]:
        """Get pending reviews, optionally filtered"""
        
        reviews = list(self.pending_reviews.values())
        
        if priority_filter:
            reviews = [r for r in reviews if r.priority == priority_filter]
        
        # Sort by priority and creation time
        reviews.sort(key=lambda x: (x.priority.value, x.created_at), reverse=True)
        
        return reviews
    
    def register_review_callback(self, review_id: str, callback: Callable):
        """Register a callback to be executed when review is completed"""
        self.review_callbacks[review_id] = callback
    
    def _assess_review_priority(self, agent_output: Dict[str, Any], step_name: str) -> ReviewPriority:
        """Assess the priority level for a review request"""
        
        # Check for critical issues
        if step_name == "auditor" or "critical" in str(agent_output).lower():
            return ReviewPriority.CRITICAL
        
        # Check for performance issues
        metrics = agent_output.get('metrics', {})
        if isinstance(metrics, dict):
            auc = metrics.get('auc', 1.0)
            if auc < 0.65:  # Below regulatory threshold
                return ReviewPriority.HIGH
            elif auc < 0.70:  # Marginal performance
                return ReviewPriority.MEDIUM
        
        # Check for data quality issues
        analysis = agent_output.get('analysis', {})
        if isinstance(analysis, dict):
            missing_pct = analysis.get('missing_percentage', 0)
            if missing_pct > 10:  # High missing data
                return ReviewPriority.HIGH
            elif missing_pct > 5:  # Moderate missing data
                return ReviewPriority.MEDIUM
        
        return ReviewPriority.LOW
    
    def _generate_review_questions(self, step_name: str, agent_output: Dict[str, Any]) -> List[str]:
        """Generate context-appropriate review questions"""
        
        base_questions = [
            "Do the results appear reasonable and accurate?",
            "Are there any concerning patterns or anomalies?",
            "Should the workflow proceed to the next step?"
        ]
        
        step_specific_questions = {
            "analyst": [
                "Is the data analysis comprehensive and accurate?",
                "Are the identified features appropriate for credit risk modeling?",
                "Do the data quality findings align with your expectations?"
            ],
            "validator": [
                "Are the validation metrics within acceptable ranges?",
                "Do the performance results meet regulatory requirements?",
                "Are there any model stability concerns?"
            ],
            "documentation": [
                "Is the documentation review thorough and accurate?",
                "Are all regulatory requirements properly addressed?",
                "Are there any compliance gaps that need attention?"
            ],
            "reviewer": [
                "Are the findings and recommendations appropriate?",
                "Do the risk assessments align with the evidence?",
                "Are the proposed actions adequate and feasible?"
            ],
            "auditor": [
                "Is the final audit opinion well-supported?",
                "Are all material issues properly disclosed?",
                "Do you agree with the overall risk rating?"
            ]
        }
        
        questions = base_questions.copy()
        if step_name in step_specific_questions:
            questions.extend(step_specific_questions[step_name])
        
        return questions
    
    def _build_review_context(self, workflow_id: str, step_name: str, 
                             agent_output: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive context for human reviewers"""
        
        context = {
            "workflow_id": workflow_id,
            "step_name": step_name,
            "timestamp": datetime.now().isoformat(),
            "agent_summary": self._summarize_agent_output(agent_output),
            "key_metrics": self._extract_key_metrics(agent_output),
            "potential_issues": self._identify_potential_issues(agent_output),
            "regulatory_implications": self._assess_regulatory_implications(step_name, agent_output)
        }
        
        return context
    
    def _summarize_agent_output(self, agent_output: Dict[str, Any]) -> str:
        """Create a human-readable summary of agent output"""
        
        summary_parts = []
        
        if 'metrics' in agent_output:
            metrics = agent_output['metrics']
            if isinstance(metrics, dict):
                summary_parts.append(f"Performance metrics calculated: AUC={metrics.get('auc', 'N/A')}")
        
        if 'analysis' in agent_output:
            analysis = agent_output['analysis']
            if isinstance(analysis, dict):
                summary_parts.append(f"Data analysis completed with {analysis.get('feature_count', 'unknown')} features")
        
        if 'recommendations' in agent_output:
            recs = agent_output['recommendations']
            if isinstance(recs, list):
                summary_parts.append(f"{len(recs)} recommendations generated")
        
        return "; ".join(summary_parts) if summary_parts else "Agent completed processing"
    
    def _extract_key_metrics(self, agent_output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for reviewer attention"""
        
        key_metrics = {}
        
        if 'metrics' in agent_output and isinstance(agent_output['metrics'], dict):
            metrics = agent_output['metrics']
            key_metrics.update({
                'auc': metrics.get('auc'),
                'gini': metrics.get('gini'),
                'ks_statistic': metrics.get('ks_statistic'),
                'psi': metrics.get('psi')
            })
        
        if 'analysis' in agent_output and isinstance(agent_output['analysis'], dict):
            analysis = agent_output['analysis']
            key_metrics.update({
                'missing_percentage': analysis.get('missing_percentage'),
                'outlier_percentage': analysis.get('outlier_percentage'),
                'feature_count': analysis.get('feature_count')
            })
        
        return {k: v for k, v in key_metrics.items() if v is not None}
    
    def _identify_potential_issues(self, agent_output: Dict[str, Any]) -> List[str]:
        """Identify potential issues requiring human attention"""
        
        issues = []
        
        # Check performance metrics
        metrics = agent_output.get('metrics', {})
        if isinstance(metrics, dict):
            auc = metrics.get('auc', 1.0)
            if auc < 0.65:
                issues.append("AUC below regulatory minimum (0.65)")
            elif auc < 0.70:
                issues.append("AUC in marginal range (0.65-0.70)")
            
            psi = metrics.get('psi', 0.0)
            if psi > 0.25:
                issues.append("High population stability index indicating drift")
        
        # Check data quality
        analysis = agent_output.get('analysis', {})
        if isinstance(analysis, dict):
            missing_pct = analysis.get('missing_percentage', 0)
            if missing_pct > 10:
                issues.append(f"High missing data percentage: {missing_pct}%")
        
        return issues
    
    def _assess_regulatory_implications(self, step_name: str, agent_output: Dict[str, Any]) -> List[str]:
        """Assess regulatory implications of the agent output"""
        
        implications = []
        
        if step_name == "validator":
            metrics = agent_output.get('metrics', {})
            if isinstance(metrics, dict):
                auc = metrics.get('auc', 1.0)
                if auc < 0.65:
                    implications.append("Model may not meet Basel III requirements")
        
        if step_name == "documentation":
            review_results = agent_output.get('review_results', {})
            if isinstance(review_results, dict):
                missing_docs = review_results.get('missing_documents', [])
                if missing_docs:
                    implications.append("Missing documentation may impact regulatory compliance")
        
        return implications
    
    def get_review_statistics(self) -> Dict[str, Any]:
        """Get statistics about review activity"""
        
        total_pending = len(self.pending_reviews)
        total_completed = len(self.completed_reviews)
        
        priority_breakdown = {}
        for priority in ReviewPriority:
            priority_breakdown[priority.name] = sum(
                1 for r in self.pending_reviews.values() 
                if r.priority == priority
            )
        
        approval_rate = 0
        if total_completed > 0:
            approved = sum(
                1 for r in self.completed_reviews.values()
                if r.status == ReviewStatus.APPROVED
            )
            approval_rate = approved / total_completed
        
        return {
            "total_pending": total_pending,
            "total_completed": total_completed,
            "priority_breakdown": priority_breakdown,
            "approval_rate": approval_rate,
            "average_review_time_hours": self._calculate_avg_review_time()
        }
    
    def _calculate_avg_review_time(self) -> float:
        """Calculate average review time in hours"""
        
        completed_with_times = [
            r for r in self.completed_reviews.values()
            if r.reviewed_at and r.created_at
        ]
        
        if not completed_with_times:
            return 0.0
        
        total_hours = sum(
            (r.reviewed_at - r.created_at).total_seconds() / 3600
            for r in completed_with_times
        )
        
        return total_hours / len(completed_with_times)