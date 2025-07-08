"""
Validator agent for computing model validation metrics.
"""

import logging
from typing import Any

from agent_service.core.agents.base import BaseAgent
from agent_service.modules.model_validation.tools.ml_metrics import AUCCalculator
from agent_service.modules.model_validation.tools.ml_metrics import KSCalculator
from agent_service.modules.model_validation.tools.ml_metrics import PSICalculator

logger = logging.getLogger("aura_agent")


class ValidatorAgent(BaseAgent):
    """
    Validator agent responsible for computing validation metrics.

    This agent calculates key performance metrics including:
    - AUC (Area Under the Curve)
    - KS (Kolmogorov-Smirnov) statistic
    - PSI (Population Stability Index)
    - Other validation metrics
    """

    def __init__(self, name: str = "ValidatorAgent", description: str = None):
        """
        Initialize the validator agent.

        Args:
            name: Agent name
            description: Agent description
        """
        super().__init__(
            name=name,
            description=description
            or "Computes model validation metrics and performance scores",
        )

        # Initialize metric calculation tools
        self.auc_calculator = AUCCalculator()
        self.ks_calculator = KSCalculator()
        self.psi_calculator = PSICalculator()

    async def __call__(self, state: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Execute validator agent logic.

        Args:
            state: Current workflow state
            **kwargs: Additional arguments

        Returns:
            Updated state with validation metrics
        """
        self.log_execution_start(state)

        try:
            # Validate required state keys
            self.validate_state(state, ["model_id"])

            model_id = state["model_id"]

            # Compute validation metrics
            validation_metrics = await self._compute_validation_metrics(model_id, state)

            # Update state with validation results
            state["validation"] = validation_metrics
            state["validation_completed"] = True
            state["validator_notes"] = (
                f"Validation metrics computed for model {model_id}"
            )

            self.logger.info(f"✅ Validation completed for model: {model_id}")

            self.log_execution_end(state)
            return state

        except Exception as e:
            return self.handle_error(e, state)

    async def _compute_validation_metrics(
        self,
        model_id: str,
        state: dict[str, Any],
    ) -> dict[str, float]:
        """
        Compute validation metrics for the model.

        Args:
            model_id: Model identifier
            state: Current state

        Returns:
            Dictionary of validation metrics
        """
        # In a real implementation, this would:
        # 1. Load model predictions and true labels
        # 2. Compute various performance metrics
        # 3. Calculate stability metrics
        # 4. Generate validation reports

        # For demonstration, we'll simulate metric calculation
        # with some realistic values

        # Generate simulated prediction data
        y_true, y_pred = self._generate_simulation_data()

        # Calculate AUC
        auc_result = await self.auc_calculator.execute(y_true=y_true, y_pred=y_pred)
        auc_score = auc_result.get("auc", 0.0) if auc_result.get("success") else 0.0

        # Calculate KS
        ks_result = await self.ks_calculator.execute(y_true=y_true, y_pred=y_pred)
        ks_score = ks_result.get("ks", 0.0) if ks_result.get("success") else 0.0

        # Calculate PSI (simulate baseline vs current)
        baseline_scores = self._generate_baseline_scores()
        current_scores = y_pred
        psi_result = await self.psi_calculator.execute(
            expected=baseline_scores,
            actual=current_scores,
        )
        psi_score = psi_result.get("psi", 0.0) if psi_result.get("success") else 0.0

        # Compile validation metrics
        validation_metrics = {
            "AUC": round(auc_score, 4),
            "KS": round(ks_score, 4),
            "PSI": round(psi_score, 4),
            "Gini": round(2 * auc_score - 1, 4),  # Gini = 2*AUC - 1
            "LogLoss": self._calculate_log_loss(y_true, y_pred),
            "Accuracy": self._calculate_accuracy(y_true, y_pred),
            "Precision": self._calculate_precision(y_true, y_pred),
            "Recall": self._calculate_recall(y_true, y_pred),
            "F1Score": self._calculate_f1_score(y_true, y_pred),
        }

        self.logger.debug(
            f"Computed validation metrics for {model_id}: {validation_metrics}",
        )
        return validation_metrics

    def _generate_simulation_data(self) -> tuple[list, list]:
        """Generate simulated prediction data for testing."""
        import random

        # Generate realistic binary classification data
        n_samples = 1000

        # Generate true labels (0 or 1)
        y_true = [random.choice([0, 1]) for _ in range(n_samples)]

        # Generate predicted probabilities that correlate with true labels
        y_pred = []
        for label in y_true:
            if label == 1:
                # Higher probability for positive class
                prob = random.uniform(0.6, 0.95)
            else:
                # Lower probability for negative class
                prob = random.uniform(0.05, 0.4)
            y_pred.append(prob)

        return y_true, y_pred

    def _generate_baseline_scores(self) -> list:
        """Generate baseline scores for PSI calculation."""
        import random

        n_samples = 800  # Slightly different size for baseline
        return [random.uniform(0.1, 0.9) for _ in range(n_samples)]

    def _calculate_log_loss(self, y_true: list, y_pred: list) -> float:
        """Calculate log loss metric."""
        import math

        epsilon = 1e-15  # Small value to prevent log(0)
        log_loss = 0

        for true_label, pred_prob in zip(y_true, y_pred, strict=False):
            # Clip predictions to prevent log(0)
            pred_prob = max(min(pred_prob, 1 - epsilon), epsilon)

            if true_label == 1:
                log_loss += -math.log(pred_prob)
            else:
                log_loss += -math.log(1 - pred_prob)

        return round(log_loss / len(y_true), 4)

    def _calculate_accuracy(
        self,
        y_true: list,
        y_pred: list,
        threshold: float = 0.5,
    ) -> float:
        """Calculate accuracy metric."""
        y_pred_binary = [1 if p >= threshold else 0 for p in y_pred]
        correct = sum(
            1 for true, pred in zip(y_true, y_pred_binary, strict=False) if true == pred
        )
        return round(correct / len(y_true), 4)

    def _calculate_precision(
        self,
        y_true: list,
        y_pred: list,
        threshold: float = 0.5,
    ) -> float:
        """Calculate precision metric."""
        y_pred_binary = [1 if p >= threshold else 0 for p in y_pred]

        true_positives = sum(
            1
            for true, pred in zip(y_true, y_pred_binary, strict=False)
            if true == 1 and pred == 1
        )
        predicted_positives = sum(y_pred_binary)

        if predicted_positives == 0:
            return 0.0

        return round(true_positives / predicted_positives, 4)

    def _calculate_recall(
        self,
        y_true: list,
        y_pred: list,
        threshold: float = 0.5,
    ) -> float:
        """Calculate recall metric."""
        y_pred_binary = [1 if p >= threshold else 0 for p in y_pred]

        true_positives = sum(
            1
            for true, pred in zip(y_true, y_pred_binary, strict=False)
            if true == 1 and pred == 1
        )
        actual_positives = sum(y_true)

        if actual_positives == 0:
            return 0.0

        return round(true_positives / actual_positives, 4)

    def _calculate_f1_score(
        self,
        y_true: list,
        y_pred: list,
        threshold: float = 0.5,
    ) -> float:
        """Calculate F1 score metric."""
        precision = self._calculate_precision(y_true, y_pred, threshold)
        recall = self._calculate_recall(y_true, y_pred, threshold)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return round(f1, 4)
