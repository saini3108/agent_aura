# agent_service/core/schema/flow.py

from typing import Any

from pydantic import BaseModel
from pydantic import Field


class FlowInput(BaseModel):
    """
    🎯 Schema representing the **initial input state** for a LangGraph flow.
    This is the structured state that agents in the flow will consume and mutate.
    """

    model_id: str = Field(
        ...,
        description="Unique identifier for the model being validated(e.g. 'xgb_24_v3')",
    )
    model_type: str | None = Field(
        default=None,
        description="Optional type of the model (e.g. 'scorecard', 'xgboost')",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata about the model (e.g. training_date, author)",
    )
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Optional precomputed metrics (e.g. {'AUC': 0.89, 'KS': 0.65})",
    )


class FlowOutput(BaseModel):
    """
    ✅ Final state returned after LangGraph flow execution.
    This is typically shaped by final agent in the graph (e.g. reviewer or approver).
    """

    model_id: str = Field(
        ...,
        description="ID of the model that was processed by the flow",
    )
    validation: dict[str, float] = Field(
        ...,
        description="Validation metrics (e.g. AUC, KS, PSI, Gini)",
    )
    review: str | None = Field(
        default=None,
        description=(
            "Optional human or automated review conclusion (e.g. 'Model approved')"
        ),
    )
    success: bool = Field(
        default=True,
        description="Whether flow execution completed successfully",
    )
