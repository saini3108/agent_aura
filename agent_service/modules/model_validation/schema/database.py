"""
Database schema and models for model validation module.
"""

import json
import logging
from typing import Any

from agent_service.core.memory import PostgresStore

logger = logging.getLogger("aura_agent")


class ModelValidationDB:
    """
    Production-grade database operations for model validation.
    """

    def __init__(self):
        self.postgres_store = PostgresStore()

    async def initialize_schema(self) -> None:
        """Initialize model validation specific database schema."""
        try:
            pool = await self.postgres_store._ensure_connected()
            async with pool.acquire() as conn:
                # Model validation results table
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS model_validation_results (
                        id SERIAL PRIMARY KEY,
                        model_id VARCHAR(255) NOT NULL,
                        workflow_id VARCHAR(255) NOT NULL,
                        validation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        auc_score DECIMAL(6,4),
                        ks_score DECIMAL(6,4),
                        psi_score DECIMAL(6,4),
                        gini_score DECIMAL(6,4),
                        log_loss DECIMAL(10,6),
                        accuracy DECIMAL(6,4),
                        precision_score DECIMAL(6,4),
                        recall_score DECIMAL(6,4),
                        f1_score DECIMAL(6,4),
                        n_samples INTEGER,
                        n_positive INTEGER,
                        n_negative INTEGER,
                        validation_status VARCHAR(50) DEFAULT 'completed',
                        agent_name VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_model_id (model_id),
                        INDEX idx_workflow_id (workflow_id),
                        INDEX idx_validation_timestamp (validation_timestamp)
                    )
                """,
                )

                # Model validation history table
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS model_validation_history (
                        id SERIAL PRIMARY KEY,
                        model_id VARCHAR(255) NOT NULL,
                        validation_run_id INTEGER REFERENCES model_validation_results(id),
                        previous_auc DECIMAL(6,4),
                        current_auc DECIMAL(6,4),
                        auc_drift DECIMAL(6,4),
                        previous_ks DECIMAL(6,4),
                        current_ks DECIMAL(6,4),
                        ks_drift DECIMAL(6,4),
                        drift_detected BOOLEAN DEFAULT FALSE,
                        drift_severity VARCHAR(50),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_model_drift (model_id, drift_detected)
                    )
                """,
                )

                # Model approval decisions table
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS model_approval_decisions (
                        id SERIAL PRIMARY KEY,
                        model_id VARCHAR(255) NOT NULL,
                        validation_result_id INTEGER REFERENCES model_validation_results(id),
                        workflow_id VARCHAR(255) NOT NULL,
                        approval_status VARCHAR(50) NOT NULL,
                        reviewer_agent VARCHAR(255),
                        human_reviewer VARCHAR(255),
                        approval_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        approval_reason TEXT,
                        risk_assessment JSONB,
                        conditions JSONB,
                        expires_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_model_approval (model_id, approval_status),
                        INDEX idx_approval_timestamp (approval_timestamp)
                    )
                """,
                )

                logger.info("✅ Model validation database schema initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize model validation schema: {e}")
            raise

    async def save_validation_result(
        self,
        model_id: str,
        workflow_id: str,
        metrics: dict[str, float],
        agent_name: str = "ValidatorAgent",
    ) -> int:
        """
        Save validation results to database.

        Args:
            model_id: Model identifier
            workflow_id: Workflow identifier
            metrics: Validation metrics
            agent_name: Name of the validating agent

        Returns:
            ID of the saved validation result
        """
        try:
            pool = await self.postgres_store._ensure_connected()
            async with pool.acquire() as conn:
                # Insert validation result
                result_id = await conn.fetchval(
                    """
                    INSERT INTO model_validation_results (
                        model_id, workflow_id, auc_score, ks_score, psi_score,
                        gini_score, log_loss, accuracy, precision_score,
                        recall_score, f1_score, agent_name
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    RETURNING id
                """,
                    model_id,
                    workflow_id,
                    metrics.get("AUC"),
                    metrics.get("KS"),
                    metrics.get("PSI"),
                    metrics.get("Gini"),
                    metrics.get("LogLoss"),
                    metrics.get("Accuracy"),
                    metrics.get("Precision"),
                    metrics.get("Recall"),
                    metrics.get("F1Score"),
                    agent_name,
                )

                # Check for drift if previous results exist
                await self._check_and_record_drift(conn, model_id, result_id, metrics)

                logger.info(
                    f"💾 Saved validation result {result_id} for model {model_id}",
                )
                return result_id

        except Exception as e:
            logger.error(f"❌ Failed to save validation result: {e}")
            raise

    async def save_approval_decision(
        self,
        model_id: str,
        workflow_id: str,
        validation_result_id: int,
        approval_status: str,
        reviewer_agent: str,
        approval_reason: str,
        risk_assessment: dict[str, Any],
        human_reviewer: str | None = None,
    ) -> int:
        """
        Save model approval decision.

        Args:
            model_id: Model identifier
            workflow_id: Workflow identifier
            validation_result_id: Reference to validation result
            approval_status: approved/rejected/conditional
            reviewer_agent: Name of reviewing agent
            approval_reason: Reason for decision
            risk_assessment: Risk assessment details
            human_reviewer: Human reviewer if applicable

        Returns:
            ID of the approval record
        """
        try:
            pool = await self.postgres_store._ensure_connected()
            async with pool.acquire() as conn:
                approval_id = await conn.fetchval(
                    """
                    INSERT INTO model_approval_decisions (
                        model_id, validation_result_id, workflow_id, approval_status,
                        reviewer_agent, human_reviewer, approval_reason, risk_assessment
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING id
                """,
                    model_id,
                    validation_result_id,
                    workflow_id,
                    approval_status,
                    reviewer_agent,
                    human_reviewer,
                    approval_reason,
                    json.dumps(risk_assessment),
                )

                logger.info(
                    f"📋 Saved approval decision {approval_id} for model {model_id}",
                )
                return approval_id

        except Exception as e:
            logger.error(f"❌ Failed to save approval decision: {e}")
            raise

    async def get_model_validation_history(
        self,
        model_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get validation history for a model."""
        try:
            pool = await self.postgres_store._ensure_connected()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT mvr.*, mad.approval_status, mad.approval_reason
                    FROM model_validation_results mvr
                    LEFT JOIN model_approval_decisions mad ON mvr.id = mad.validation_result_id
                    WHERE mvr.model_id = $1
                    ORDER BY mvr.validation_timestamp DESC
                    LIMIT $2
                """,
                    model_id,
                    limit,
                )

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"❌ Failed to get model validation history: {e}")
            return []

    async def _check_and_record_drift(
        self,
        conn,
        model_id: str,
        current_result_id: int,
        current_metrics: dict[str, float],
    ) -> None:
        """Check for model drift and record if detected."""
        try:
            # Get the most recent previous validation
            previous_result = await conn.fetchrow(
                """
                SELECT auc_score, ks_score FROM model_validation_results
                WHERE model_id = $1 AND id != $2
                ORDER BY validation_timestamp DESC
                LIMIT 1
            """,
                model_id,
                current_result_id,
            )

            if previous_result:
                prev_auc = float(previous_result["auc_score"] or 0)
                prev_ks = float(previous_result["ks_score"] or 0)
                curr_auc = current_metrics.get("AUC", 0)
                curr_ks = current_metrics.get("KS", 0)

                auc_drift = abs(curr_auc - prev_auc)
                ks_drift = abs(curr_ks - prev_ks)

                # Define drift thresholds
                drift_detected = auc_drift > 0.05 or ks_drift > 0.05
                drift_severity = (
                    "high" if (auc_drift > 0.1 or ks_drift > 0.1) else "moderate"
                )

                # Record drift analysis
                await conn.execute(
                    """
                    INSERT INTO model_validation_history (
                        model_id, validation_run_id, previous_auc, current_auc, auc_drift,
                        previous_ks, current_ks, ks_drift, drift_detected, drift_severity
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                    model_id,
                    current_result_id,
                    prev_auc,
                    curr_auc,
                    auc_drift,
                    prev_ks,
                    curr_ks,
                    ks_drift,
                    drift_detected,
                    drift_severity,
                )

                if drift_detected:
                    logger.warning(
                        f"🚨 Model drift detected for {model_id}: AUC drift={auc_drift}, KS drift={ks_drift}",
                    )

        except Exception as e:
            logger.error(f"❌ Failed to check model drift: {e}")
