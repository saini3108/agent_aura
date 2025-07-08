"""
KS statistic calculation tool for model validation.
"""

from typing import Any

from agent_service.modules.model_validation.tools.ml_metrics import KSCalculator


class KSTool(KSCalculator):
    """
    Model validation specific KS calculation tool.

    Extends the base KS calculator with model validation specific functionality.
    """

    name = "model_validation_ks"
    description = "Calculate KS statistic for model validation workflows"

    async def execute(self, **kwargs) -> dict[str, Any]:
        """
        Execute KS calculation with model validation context.

        Args:
            **kwargs: Tool arguments including y_true, y_pred, and optional model context

        Returns:
            KS calculation results with validation context
        """
        # Call parent execute method
        result = await super().execute(**kwargs)

        if result.get("success"):
            # Add model validation specific context
            ks_score = result["ks"]

            # Add interpretation for model validation
            result["interpretation"] = self._interpret_ks_for_validation(ks_score)
            result["validation_context"] = {
                "tool": self.name,
                "suitable_for_production": ks_score >= 0.2,
                "separation_quality": self._get_separation_quality(ks_score),
            }

        return result

    def _interpret_ks_for_validation(self, ks_score: float) -> str:
        """
        Interpret KS score in model validation context.

        Args:
            ks_score: KS score

        Returns:
            Interpretation string
        """
        if ks_score >= 0.6:
            return "Excellent class separation"
        if ks_score >= 0.4:
            return "Good class separation"
        if ks_score >= 0.2:
            return "Acceptable class separation"
        return "Poor class separation - model may lack discriminatory power"

    def _get_separation_quality(self, ks_score: float) -> str:
        """
        Get separation quality description for KS score.

        Args:
            ks_score: KS score

        Returns:
            Separation quality description
        """
        if ks_score >= 0.7:
            return "Exceptional separation"
        if ks_score >= 0.5:
            return "Strong separation"
        if ks_score >= 0.3:
            return "Moderate separation"
        if ks_score >= 0.2:
            return "Weak but acceptable separation"
        return "Insufficient separation"
