"""
AUC calculation tool for model validation.
"""

from typing import Any

from agent_service.modules.model_validation.tools.ml_metrics import AUCCalculator


class AUCTool(AUCCalculator):
    """
    Model validation specific AUC calculation tool.

    Extends the base AUC calculator with model validation specific functionality.
    """

    name = "model_validation_auc"
    description = "Calculate AUC for model validation workflows"

    async def execute(self, **kwargs) -> dict[str, Any]:
        """
        Execute AUC calculation with model validation context.

        Args:
            **kwargs: Tool arguments including y_true, y_pred, and optional model context

        Returns:
            AUC calculation results with validation context
        """
        # Call parent execute method
        result = await super().execute(**kwargs)

        if result.get("success"):
            # Add model validation specific context
            auc_score = result["auc"]

            # Add interpretation for model validation
            result["interpretation"] = self._interpret_auc_for_validation(auc_score)
            result["validation_context"] = {
                "tool": self.name,
                "suitable_for_production": auc_score >= 0.65,
                "performance_tier": self._get_performance_tier(auc_score),
            }

        return result

    def _interpret_auc_for_validation(self, auc_score: float) -> str:
        """
        Interpret AUC score in model validation context.

        Args:
            auc_score: AUC score

        Returns:
            Interpretation string
        """
        if auc_score >= 0.9:
            return "Outstanding discrimination ability"
        if auc_score >= 0.8:
            return "Excellent discrimination ability"
        if auc_score >= 0.7:
            return "Good discrimination ability"
        if auc_score >= 0.6:
            return "Acceptable discrimination ability"
        return "Poor discrimination ability - model may not be suitable"

    def _get_performance_tier(self, auc_score: float) -> str:
        """
        Get performance tier for AUC score.

        Args:
            auc_score: AUC score

        Returns:
            Performance tier
        """
        if auc_score >= 0.85:
            return "Tier 1 - Excellent"
        if auc_score >= 0.75:
            return "Tier 2 - Good"
        if auc_score >= 0.65:
            return "Tier 3 - Acceptable"
        return "Below Standard"
