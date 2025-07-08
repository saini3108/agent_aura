"""
Reviewer agent for analyzing validation results and making recommendations.
"""

import logging
from typing import Any

from agent_service.core.agents.base import BaseAgent

logger = logging.getLogger("aura_agent")


class ReviewerAgent(BaseAgent):
    """
    Reviewer agent responsible for analyzing validation results and making recommendations.

    This agent evaluates validation metrics against thresholds and business rules to:
    - Determine if model meets quality standards
    - Generate recommendations for model approval/rejection
    - Provide detailed review comments
    """

    def __init__(self, name: str = "ReviewerAgent", description: str = None):
        """
        Initialize the reviewer agent.

        Args:
            name: Agent name
            description: Agent description
        """
        super().__init__(
            name=name,
            description=description
            or "Reviews validation results and makes approval recommendations",
        )

        # Define validation thresholds
        self.thresholds = {
            "AUC": {"min": 0.65, "good": 0.75, "excellent": 0.85},
            "KS": {"min": 0.20, "good": 0.40, "excellent": 0.60},
            "PSI": {"max": 0.25, "warning": 0.15, "good": 0.10},
            "Gini": {"min": 0.30, "good": 0.50, "excellent": 0.70},
        }

    async def __call__(self, state: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Execute reviewer agent logic.

        Args:
            state: Current workflow state
            **kwargs: Additional arguments

        Returns:
            Updated state with review results
        """
        self.log_execution_start(state)

        try:
            # Validate required state keys
            self.validate_state(state, ["model_id", "validation"])

            model_id = state["model_id"]
            validation_metrics = state["validation"]

            # Perform review analysis
            review_results = await self._analyze_validation_results(
                model_id,
                validation_metrics,
                state,
            )

            # Update state with review
            state["review"] = review_results["summary"]
            state["review_details"] = review_results
            state["review_completed"] = True
            state["reviewer_notes"] = f"Review completed for model {model_id}"

            # Set success based on approval status
            state["success"] = review_results["approved"]

            self.logger.info(
                f"🔍 Review completed for model: {model_id} - Status: {'APPROVED' if review_results['approved'] else 'REJECTED'}",
            )

            self.log_execution_end(state)
            return state

        except Exception as e:
            return self.handle_error(e, state)

    async def _analyze_validation_results(
        self,
        model_id: str,
        validation_metrics: dict[str, float],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Analyze validation results and generate review.

        Args:
            model_id: Model identifier
            validation_metrics: Validation metrics
            state: Current state

        Returns:
            Review analysis results
        """
        # Evaluate each metric against thresholds
        metric_evaluations = {}
        overall_score = 0
        total_weight = 0

        for metric_name, value in validation_metrics.items():
            if metric_name in self.thresholds:
                evaluation = self._evaluate_metric(metric_name, value)
                metric_evaluations[metric_name] = evaluation

                # Weight metrics for overall score
                weight = self._get_metric_weight(metric_name)
                overall_score += evaluation["score"] * weight
                total_weight += weight

        # Calculate overall score
        if total_weight > 0:
            overall_score = overall_score / total_weight
        else:
            overall_score = 0

        # Determine approval status
        approved = self._determine_approval(metric_evaluations, overall_score)

        # Generate review summary
        summary = self._generate_review_summary(
            model_id,
            metric_evaluations,
            overall_score,
            approved,
        )

        # Generate detailed recommendations
        recommendations = self._generate_recommendations(
            metric_evaluations,
            validation_metrics,
            state,
        )

        review_results = {
            "model_id": model_id,
            "overall_score": round(overall_score, 2),
            "approved": approved,
            "summary": summary,
            "metric_evaluations": metric_evaluations,
            "recommendations": recommendations,
            "validation_metrics": validation_metrics,
        }

        self.logger.debug(f"Generated review for {model_id}: {review_results}")
        return review_results

    def _evaluate_metric(self, metric_name: str, value: float) -> dict[str, Any]:
        """
        Evaluate a single metric against thresholds.

        Args:
            metric_name: Name of the metric
            value: Metric value

        Returns:
            Evaluation results
        """
        thresholds = self.thresholds[metric_name]

        if metric_name == "PSI":  # Lower is better for PSI
            if value <= thresholds["good"]:
                status = "excellent"
                score = 1.0
            elif value <= thresholds["warning"]:
                status = "good"
                score = 0.8
            elif value <= thresholds["max"]:
                status = "acceptable"
                score = 0.6
            else:
                status = "poor"
                score = 0.2
        elif value >= thresholds["excellent"]:
            status = "excellent"
            score = 1.0
        elif value >= thresholds["good"]:
            status = "good"
            score = 0.8
        elif value >= thresholds["min"]:
            status = "acceptable"
            score = 0.6
        else:
            status = "poor"
            score = 0.2

        return {
            "value": value,
            "status": status,
            "score": score,
            "thresholds": thresholds,
        }

    def _get_metric_weight(self, metric_name: str) -> float:
        """Get weight for metric in overall score calculation."""
        weights = {
            "AUC": 0.3,
            "KS": 0.25,
            "PSI": 0.2,
            "Gini": 0.25,
        }
        return weights.get(metric_name, 0.1)

    def _determine_approval(
        self,
        metric_evaluations: dict[str, dict[str, Any]],
        overall_score: float,
    ) -> bool:
        """
        Determine if model should be approved based on evaluations.

        Args:
            metric_evaluations: Individual metric evaluations
            overall_score: Overall score

        Returns:
            True if model should be approved
        """
        # Check for any critical failures
        critical_metrics = ["AUC", "KS"]
        for metric in critical_metrics:
            if metric in metric_evaluations:
                if metric_evaluations[metric]["status"] == "poor":
                    return False

        # Check overall score threshold
        if overall_score < 0.6:
            return False

        # Check PSI threshold (stability)
        if "PSI" in metric_evaluations:
            if metric_evaluations["PSI"]["status"] == "poor":
                return False

        return True

    def _generate_review_summary(
        self,
        model_id: str,
        metric_evaluations: dict[str, dict[str, Any]],
        overall_score: float,
        approved: bool,
    ) -> str:
        """
        Generate human-readable review summary.

        Args:
            model_id: Model identifier
            metric_evaluations: Metric evaluations
            overall_score: Overall score
            approved: Approval status

        Returns:
            Review summary text
        """
        status = "APPROVED" if approved else "REJECTED"

        # Count metrics by status
        status_counts = {"excellent": 0, "good": 0, "acceptable": 0, "poor": 0}
        for evaluation in metric_evaluations.values():
            status_counts[evaluation["status"]] += 1

        # Generate summary based on results
        if approved:
            if overall_score >= 0.9:
                summary = f"Model {model_id} {status}: Excellent performance across all metrics. "
            elif overall_score >= 0.8:
                summary = f"Model {model_id} {status}: Good performance with strong validation results. "
            else:
                summary = f"Model {model_id} {status}: Acceptable performance meets minimum requirements. "
        else:
            summary = (
                f"Model {model_id} {status}: Performance below acceptable thresholds. "
            )

        # Add metric highlights
        key_metrics = []
        for metric, evaluation in metric_evaluations.items():
            if metric in ["AUC", "KS"]:
                key_metrics.append(
                    f"{metric}: {evaluation['value']:.3f} ({evaluation['status']})",
                )

        if key_metrics:
            summary += f"Key metrics - {', '.join(key_metrics)}. "

        # Add overall score
        summary += f"Overall score: {overall_score:.2f}/1.0."

        return summary

    def _generate_recommendations(
        self,
        metric_evaluations: dict[str, dict[str, Any]],
        validation_metrics: dict[str, float],
        state: dict[str, Any],
    ) -> list[str]:
        """
        Generate specific recommendations based on results.

        Args:
            metric_evaluations: Metric evaluations
            validation_metrics: Raw validation metrics
            state: Current state

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check for specific issues and suggest improvements
        for metric, evaluation in metric_evaluations.items():
            if evaluation["status"] == "poor":
                if metric == "AUC":
                    recommendations.append(
                        "AUC is below acceptable threshold. Consider feature engineering, "
                        "hyperparameter tuning, or alternative algorithms.",
                    )
                elif metric == "KS":
                    recommendations.append(
                        "KS statistic is low, indicating poor class separation. "
                        "Review feature selection and model architecture.",
                    )
                elif metric == "PSI":
                    recommendations.append(
                        "High PSI indicates population instability. "
                        "Investigate data drift and consider model retraining.",
                    )
            elif evaluation["status"] == "acceptable":
                if metric == "AUC":
                    recommendations.append(
                        "AUC meets minimum requirements but has room for improvement. "
                        "Consider additional feature engineering.",
                    )

        # General recommendations based on overall performance
        # TODO: uncomment
        # overall_score = sum(eval["score"] for eval in metric_evaluations.values()) / len(metric_evaluations)
        overall_score = 0.4  # sum(eval["score"] for eval in metric_evaluations.values()) / len(metric_evaluations)

        if overall_score < 0.7:
            recommendations.append(
                "Overall model performance could be improved. "
                "Consider comprehensive model review and potential retraining.",
            )
        elif overall_score >= 0.9:
            recommendations.append(
                "Excellent model performance. Consider this model for production deployment.",
            )

        # Add monitoring recommendations
        if "PSI" in validation_metrics and validation_metrics["PSI"] > 0.1:
            recommendations.append(
                "Implement ongoing monitoring for population stability and data drift.",
            )

        return recommendations
