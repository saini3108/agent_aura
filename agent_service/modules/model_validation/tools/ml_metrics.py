"""
Machine Learning metrics calculation tools for model validation.
"""

import logging
from typing import Any

import numpy as np

from agent_service.core.tools.base import BaseTool

logger = logging.getLogger("aura_agent")


class AUCCalculator(BaseTool):
    """
    Tool for calculating Area Under the ROC Curve (AUC).
    """

    name = "auc_calculator"
    description = "Calculates AUC (Area Under the ROC Curve) for binary classification"

    async def execute(self, **kwargs) -> dict[str, Any]:
        """
        Calculate AUC from predictions and true labels.

        Args:
            y_true: True binary labels (0 or 1)
            y_pred: Predicted probabilities or scores

        Returns:
            Dictionary containing AUC score and metadata
        """
        self.log_execution_start(**kwargs)

        try:
            # Validate required parameters
            self.validate_inputs(["y_true", "y_pred"], **kwargs)

            y_true = kwargs["y_true"]
            y_pred = kwargs["y_pred"]

            # Convert to numpy arrays for easier processing
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            # Validate input shapes
            if len(y_true) != len(y_pred):
                raise ValueError("y_true and y_pred must have the same length")

            if len(y_true) == 0:
                raise ValueError("Input arrays cannot be empty")

            # Calculate AUC using trapezoidal rule
            auc_score = self._calculate_auc_manual(y_true, y_pred)

            result = {
                "success": True,
                "auc": float(auc_score),
                "n_samples": len(y_true),
                "n_positive": int(np.sum(y_true)),
                "n_negative": int(len(y_true) - np.sum(y_true)),
            }

            self.log_execution_end(result)
            return result

        except Exception as e:
            return self.handle_error(e, **kwargs)

    def _calculate_auc_manual(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Manual AUC calculation using the trapezoidal rule.

        Args:
            y_true: True binary labels
            y_pred: Predicted scores

        Returns:
            AUC score
        """
        # Get unique thresholds
        thresholds = np.unique(y_pred)
        thresholds = np.sort(thresholds)[::-1]  # Sort in descending order

        # Calculate TPR and FPR for each threshold
        tpr_list = []
        fpr_list = []

        for threshold in thresholds:
            y_pred_binary = (y_pred >= threshold).astype(int)

            tp = np.sum((y_true == 1) & (y_pred_binary == 1))
            fp = np.sum((y_true == 0) & (y_pred_binary == 1))
            tn = np.sum((y_true == 0) & (y_pred_binary == 0))
            fn = np.sum((y_true == 1) & (y_pred_binary == 0))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        # Add (0, 0) and (1, 1) points
        fpr_list = [0] + fpr_list + [1]
        tpr_list = [0] + tpr_list + [1]

        # Calculate AUC using trapezoidal rule
        auc = 0
        for i in range(1, len(fpr_list)):
            auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

        return auc


class KSCalculator(BaseTool):
    """
    Tool for calculating Kolmogorov-Smirnov (KS) statistic.
    """

    name = "ks_calculator"
    description = (
        "Calculates KS (Kolmogorov-Smirnov) statistic for binary classification"
    )

    async def execute(self, **kwargs) -> dict[str, Any]:
        """
        Calculate KS statistic from predictions and true labels.

        Args:
            y_true: True binary labels (0 or 1)
            y_pred: Predicted probabilities or scores

        Returns:
            Dictionary containing KS statistic and metadata
        """
        self.log_execution_start(**kwargs)

        try:
            # Validate required parameters
            self.validate_inputs(["y_true", "y_pred"], **kwargs)

            y_true = kwargs["y_true"]
            y_pred = kwargs["y_pred"]

            # Convert to numpy arrays
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            # Validate input shapes
            if len(y_true) != len(y_pred):
                raise ValueError("y_true and y_pred must have the same length")

            if len(y_true) == 0:
                raise ValueError("Input arrays cannot be empty")

            # Calculate KS statistic
            ks_score = self._calculate_ks_statistic(y_true, y_pred)

            result = {
                "success": True,
                "ks": float(ks_score),
                "n_samples": len(y_true),
                "n_positive": int(np.sum(y_true)),
                "n_negative": int(len(y_true) - np.sum(y_true)),
            }

            self.log_execution_end(result)
            return result

        except Exception as e:
            return self.handle_error(e, **kwargs)

    def _calculate_ks_statistic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate KS statistic.

        Args:
            y_true: True binary labels
            y_pred: Predicted scores

        Returns:
            KS statistic
        """
        # Separate scores for positive and negative classes
        pos_scores = y_pred[y_true == 1]
        neg_scores = y_pred[y_true == 0]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return 0.0

        # Sort scores
        pos_scores = np.sort(pos_scores)
        neg_scores = np.sort(neg_scores)

        # Calculate cumulative distributions
        all_scores = np.sort(np.concatenate([pos_scores, neg_scores]))

        max_ks = 0
        for score in all_scores:
            # Calculate cumulative probability for positive class
            cdf_pos = np.sum(pos_scores <= score) / len(pos_scores)

            # Calculate cumulative probability for negative class
            cdf_neg = np.sum(neg_scores <= score) / len(neg_scores)

            # KS is the maximum difference between CDFs
            ks = abs(cdf_pos - cdf_neg)
            max_ks = max(max_ks, ks)

        return max_ks


class PSICalculator(BaseTool):
    """
    Tool for calculating Population Stability Index (PSI).
    """

    name = "psi_calculator"
    description = (
        "Calculates PSI (Population Stability Index) for distribution comparison"
    )

    async def execute(self, **kwargs) -> dict[str, Any]:
        """
        Calculate PSI between expected and actual distributions.

        Args:
            expected: Expected distribution (baseline)
            actual: Actual distribution (current)
            buckets: Number of buckets for discretization (default: 10)

        Returns:
            Dictionary containing PSI score and metadata
        """
        self.log_execution_start(**kwargs)

        try:
            # Validate required parameters
            self.validate_inputs(["expected", "actual"], **kwargs)

            expected = kwargs["expected"]
            actual = kwargs["actual"]
            buckets = kwargs.get("buckets", 10)

            # Convert to numpy arrays
            expected = np.array(expected)
            actual = np.array(actual)

            # Validate inputs
            if len(expected) == 0 or len(actual) == 0:
                raise ValueError("Input arrays cannot be empty")

            if buckets < 2:
                raise ValueError("Number of buckets must be at least 2")

            # Calculate PSI
            psi_score = self._calculate_psi(expected, actual, buckets)

            result = {
                "success": True,
                "psi": float(psi_score),
                "n_expected": len(expected),
                "n_actual": len(actual),
                "buckets": buckets,
                "interpretation": self._interpret_psi(psi_score),
            }

            self.log_execution_end(result)
            return result

        except Exception as e:
            return self.handle_error(e, **kwargs)

    def _calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        buckets: int,
    ) -> float:
        """
        Calculate PSI between two distributions.

        Args:
            expected: Expected distribution
            actual: Actual distribution
            buckets: Number of buckets

        Returns:
            PSI score
        """
        # Define bucket boundaries based on expected distribution
        bucket_boundaries = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        bucket_boundaries[0] = -np.inf
        bucket_boundaries[-1] = np.inf

        # Calculate frequencies for each bucket
        expected_freq = np.histogram(expected, bins=bucket_boundaries)[0]
        actual_freq = np.histogram(actual, bins=bucket_boundaries)[0]

        # Convert to proportions
        expected_prop = expected_freq / len(expected)
        actual_prop = actual_freq / len(actual)

        # Add small constant to avoid division by zero
        epsilon = 1e-10
        expected_prop = np.maximum(expected_prop, epsilon)
        actual_prop = np.maximum(actual_prop, epsilon)

        # Calculate PSI
        psi = np.sum(
            (actual_prop - expected_prop) * np.log(actual_prop / expected_prop),
        )

        return psi

    def _interpret_psi(self, psi_score: float) -> str:
        """
        Interpret PSI score.

        Args:
            psi_score: PSI score

        Returns:
            Interpretation string
        """
        if psi_score < 0.1:
            return "No significant change"
        if psi_score < 0.2:
            return "Minor change"
        return "Major change - investigate required"
