"""
Human agent for human-in-the-loop workflows.
"""

import logging
from datetime import datetime
from typing import Any

from .base import BaseAgent

logger = logging.getLogger("aura_agent")


class HumanAgent(BaseAgent):
    """
    Human agent that requires human intervention in the workflow.

    This agent is used for human-in-the-loop scenarios where manual
    review, approval, or input is required.
    """

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        prompt_message: str | None = None,
    ):
        """
        Initialize the human agent.

        Args:
            name: Optional name for the agent
            description: Optional description
            prompt_message: Custom message to display to humans
        """
        super().__init__(name, description)
        self.prompt_message = prompt_message or "Human review required"

    async def __call__(self, state: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Execute human agent logic.

        This method marks the state as requiring human intervention
        and can integrate with external systems for notifications.

        Args:
            state: Current workflow state
            **kwargs: Additional arguments

        Returns:
            Updated state with human review requirements
        """
        self.log_execution_start(state)

        try:
            # Mark state as requiring human review
            state["human_review_required"] = True
            state["human_review_message"] = self.prompt_message
            state["human_review_agent"] = self.name

            # Add any context needed for human review
            if "model_id" in state:
                state["human_review_context"] = {
                    "model_id": state["model_id"],
                    "timestamp": self._get_current_timestamp(),
                    "workflow_stage": self.name,
                }

            # In a production system, this would trigger:
            # - Notification to human reviewers
            # - Creation of review tasks in external systems
            # - Workflow pause until human input is received

            self.logger.info("👤 Human review requested: %s", self.prompt_message)

            # For now, we'll simulate human approval
            # In production, this would be replaced by actual human input
            state["human_review_status"] = "pending"

            self.log_execution_end(state)
            return state

        except RuntimeError as e:
            return self.handle_error(e, state)

    def _get_current_timestamp(self) -> str:
        """Get current timestamp for review tracking."""
        return datetime.utcnow().isoformat()

    def approve_review(
        self,
        state: dict[str, Any],
        approval_notes: str | None = None,
    ) -> dict[str, Any]:
        """
        Process human approval.

        Args:
            state: Current state
            approval_notes: Optional notes from reviewer

        Returns:
            Updated state with approval
        """
        state["human_review_status"] = "approved"
        state["human_review_notes"] = approval_notes or "Approved"
        state["human_review_completed_at"] = self._get_current_timestamp()

        self.logger.info("✅ Human review approved by %s", self.name)
        return state

    def reject_review(
        self,
        state: dict[str, Any],
        rejection_reason: str,
    ) -> dict[str, Any]:
        """
        Process human rejection.

        Args:
            state: Current state
            rejection_reason: Reason for rejection

        Returns:
            Updated state with rejection
        """
        state["human_review_status"] = "rejected"
        state["human_review_rejection_reason"] = rejection_reason
        state["human_review_completed_at"] = self._get_current_timestamp()

        self.logger.warning(
            "❌ Human review rejected by %s: %s",
            self.name,
            rejection_reason,
        )
        return state
