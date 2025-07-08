# agent_service/modules/model_validation/flows/sample_validation_flow.py
import logging
from typing import Any

from langgraph.graph import END
from langgraph.graph import StateGraph

from agent_service.modules.model_validation.agents import AnalystAgent
from agent_service.modules.model_validation.agents import ReviewerAgent
from agent_service.modules.model_validation.agents import ValidatorAgent

logger = logging.getLogger("aura_agent")


# Alternative flow configurations for different validation scenarios
def get_model_validation_flow():
    """
    Enhanced model validation flow with additional checks and human review.

    Returns:
        Enhanced compiled LangGraph flow
    """
    logger.info("🔧 Building enhanced model validation flow")

    from agent_service.core.agents import HumanAgent

    builder = StateGraph(dict)

    # Add all agents including human reviewer
    analyst = AnalystAgent()
    validator = ValidatorAgent()
    reviewer = ReviewerAgent()
    human_reviewer = HumanAgent(
        name="HumanReviewer",
        prompt_message="Human review required for final model approval",
    )

    builder.add_node("analyst", analyst)
    builder.add_node("validator", validator)
    builder.add_node("reviewer", reviewer)
    builder.add_node("human_reviewer", human_reviewer)

    # Enhanced flow with conditional human review
    builder.set_entry_point("analyst")
    builder.add_edge("analyst", "validator")
    builder.add_edge("validator", "reviewer")

    # Add conditional edge based on review outcome
    def should_require_human_review(state: dict[str, Any]) -> str:
        """Determine if human review is required based on validation results."""
        review_details = state.get("review_details", {})
        overall_score = review_details.get("overall_score", 0)

        # Require human review for borderline cases
        if overall_score < 0.8:
            return "human_reviewer"
        return END

    builder.add_conditional_edges(
        "reviewer",
        should_require_human_review,
        {
            "human_reviewer": "human_reviewer",
            END: END,
        },
    )

    builder.add_edge("human_reviewer", END)

    logger.info("✅ Enhanced model validation flow construction completed")
    return builder.compile()
