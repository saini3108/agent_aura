from typing import Dict, Any, List, Optional, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import uuid

class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PLANNED = "planned"
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_review"
    PAUSED = "paused"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentRole(str, Enum):
    """Agent roles in the workflow"""
    PLANNER = "planner"
    EXECUTOR = "executor"
    VALIDATOR = "validator"
    SUMMARIZER = "summarizer"
    HUMAN = "human"

class ToolResult(BaseModel):
    """Result from tool execution"""
    tool_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ValidationResult(BaseModel):
    """Result from validation step"""
    validation_type: str
    passed: bool
    score: Optional[float] = None
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HumanFeedback(BaseModel):
    """Human feedback in the workflow"""
    user_id: str
    action: Literal["approve", "reject", "modify", "pause"]
    comments: Optional[str] = None
    modifications: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class BaseContext(BaseModel):
    """Base context model for all workflows"""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_type: str
    status: WorkflowStatus = WorkflowStatus.PLANNED

    # Core data
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)

    # Execution tracking
    plan_steps: List[str] = Field(default_factory=list)
    current_step: int = 0
    tool_results: List[ToolResult] = Field(default_factory=list)

    # Agent communication
    agent_messages: List[Dict[str, Any]] = Field(default_factory=list)
    current_agent: Optional[AgentRole] = None

    # Human interaction
    human_feedback: Optional[HumanFeedback] = None
    requires_human_review: bool = False

    # Memory and context
    memory_refs: List[str] = Field(default_factory=list)
    context_history: List[Dict[str, Any]] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Configuration
    llm_config: Dict[str, Any] = Field(default_factory=dict)
    timeout: int = 1800  # 30 minutes default

    def add_tool_result(self, tool_result: ToolResult) -> None:
        """Add a tool execution result"""
        self.tool_results.append(tool_result)
        self.updated_at = datetime.utcnow()

    def add_agent_message(self, agent: AgentRole, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Add an agent message"""
        self.agent_messages.append({
            "agent": agent.value,
            "message": message,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat()
        })
        self.updated_at = datetime.utcnow()

    def update_status(self, status: WorkflowStatus) -> None:
        """Update workflow status"""
        self.status = status
        self.updated_at = datetime.utcnow()

        if status == WorkflowStatus.COMPLETE:
            self.completed_at = datetime.utcnow()

    def set_human_feedback(self, feedback: HumanFeedback) -> None:
        """Set human feedback"""
        self.human_feedback = feedback
        self.updated_at = datetime.utcnow()

class ModelValidationContext(BaseContext):
    """Context for model validation workflow"""
    workflow_type: str = "model_validation"
    model_name: str
    model_configuration: Dict[str, Any]
    validation_rules: List[str] = Field(default_factory=list)
    validation_results: List[ValidationResult] = Field(default_factory=list)

    def add_validation_result(self, result: ValidationResult) -> None:
        """Add validation result"""
        self.validation_results.append(result)
        self.updated_at = datetime.utcnow()

class ECLCalculationContext(BaseContext):
    """Context for ECL calculation workflow"""
    workflow_type: str = "ecl_calculation"
    portfolio_data: Dict[str, Any]
    calculation_parameters: Dict[str, Any]
    ifrs9_scenario: str = "base"
    calculation_results: Dict[str, Any] = Field(default_factory=dict)

    def set_calculation_results(self, results: Dict[str, Any]) -> None:
        """Set calculation results"""
        self.calculation_results = results
        self.updated_at = datetime.utcnow()

class RWACalculationContext(BaseContext):
    """Context for RWA calculation workflow"""
    workflow_type: str = "rwa_calculation"
    portfolio_data: Dict[str, Any]
    basel_framework: str = "basel_iii"
    calculation_parameters: Dict[str, Any]
    rwa_results: Dict[str, Any] = Field(default_factory=dict)

    def set_rwa_results(self, results: Dict[str, Any]) -> None:
        """Set RWA results"""
        self.rwa_results = results
        self.updated_at = datetime.utcnow()

class ReportingContext(BaseContext):
    """Context for reporting workflow"""
    workflow_type: str = "reporting"
    report_type: str
    template_name: str
    data_sources: List[str] = Field(default_factory=list)
    report_config: Dict[str, Any] = Field(default_factory=dict)
    generated_reports: List[Dict[str, Any]] = Field(default_factory=list)

    def add_generated_report(self, report: Dict[str, Any]) -> None:
        """Add generated report"""
        self.generated_reports.append(report)
        self.updated_at = datetime.utcnow()

# Context factory function
def create_context(workflow_type: str, inputs: Dict[str, Any] = None, **kwargs) -> BaseContext:
    """Factory function to create appropriate context based on workflow type"""
    context_map = {
        "model_validation": ModelValidationContext,
        "ecl_calculation": ECLCalculationContext,
        "rwa_calculation": RWACalculationContext,
        "reporting": ReportingContext
    }

    context_class = context_map.get(workflow_type, BaseContext)

    # Prepare context parameters
    context_params = {"workflow_type": workflow_type, **kwargs}

    # For specific workflow types, extract required fields from inputs
    if inputs:
        if workflow_type == "model_validation":
            context_params.update({
                "model_name": inputs.get("model_name"),
                "model_configuration": inputs.get("model_configuration", {})
            })
        elif workflow_type == "ecl_calculation":
            context_params.update({
                "portfolio_data": inputs.get("portfolio_data", [])
            })
        elif workflow_type == "rwa_calculation":
            context_params.update({
                "exposure_data": inputs.get("exposure_data", [])
            })
        elif workflow_type == "reporting":
            context_params.update({
                "report_type": inputs.get("report_type"),
                "template_name": inputs.get("template", "default"),
                "data_sources": inputs.get("data_sources", [])
            })

        # Add any remaining inputs to the base context
        context_params["inputs"] = inputs

    return context_class(**context_params)
