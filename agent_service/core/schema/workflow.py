from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import uuid

class WorkflowRequest(BaseModel):
    """Request to start a workflow"""
    workflow_type: str
    inputs: Dict[str, Any]
    llm_config: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = None
    user_id: Optional[str] = None
    priority: int = Field(default=1, ge=1, le=10)

class WorkflowResponse(BaseModel):
    """Response from workflow operations"""
    workflow_id: str
    task_id: str
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class WorkflowStatus(BaseModel):
    """Current workflow status"""
    workflow_id: str
    task_id: str
    status: str
    current_step: int
    total_steps: int
    progress: float
    current_agent: Optional[str] = None
    estimated_completion: Optional[datetime] = None
    last_update: datetime

    def calculate_progress(self) -> float:
        """Calculate workflow progress percentage"""
        if self.total_steps == 0:
            return 0.0
        return min(100.0, (self.current_step / self.total_steps) * 100)

class HumanReviewRequest(BaseModel):
    """Request for human review"""
    workflow_id: str
    review_type: str
    data: Dict[str, Any]
    questions: List[str] = Field(default_factory=list)
    options: List[str] = Field(default_factory=list)
    deadline: Optional[datetime] = None

class HumanReviewResponse(BaseModel):
    """Response from human review"""
    workflow_id: str
    user_id: str
    action: str  # approve, reject, modify, pause
    comments: Optional[str] = None
    modifications: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class WorkflowMetrics(BaseModel):
    """Workflow execution metrics"""
    workflow_id: str
    total_execution_time: float
    agent_times: Dict[str, float] = Field(default_factory=dict)
    tool_times: Dict[str, float] = Field(default_factory=dict)
    memory_usage: Dict[str, Any] = Field(default_factory=dict)
    api_calls: Dict[str, int] = Field(default_factory=dict)
    success_rate: float = 0.0
    error_count: int = 0

class WorkflowConfig(BaseModel):
    """Configuration for workflow execution"""
    max_steps: int = 50
    timeout: int = 1800  # 30 minutes
    retry_attempts: int = 3
    parallel_execution: bool = False
    require_human_review: bool = True
    auto_approve_threshold: float = 0.95

    # Agent configuration
    agent_config: Dict[str, Any] = Field(default_factory=dict)

    # Model configuration
    model_provider: str = "openai"
    model_name: str = "gpt-4"
    model_params: Dict[str, Any] = Field(default_factory=dict)

    # Tool configuration
    tool_config: Dict[str, Any] = Field(default_factory=dict)

    # Memory configuration
    memory_config: Dict[str, Any] = Field(default_factory=dict)

class WorkflowTemplate(BaseModel):
    """Template for workflow configuration"""
    name: str
    description: str
    workflow_type: str
    config: WorkflowConfig
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    steps: List[Dict[str, Any]]

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"

class WorkflowExecution(BaseModel):
    """Complete workflow execution record"""
    workflow_id: str
    task_id: str
    template_name: Optional[str] = None

    # Execution details
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None

    # Configuration
    config: WorkflowConfig
    inputs: Dict[str, Any]
    outputs: Dict[str, Any] = Field(default_factory=dict)

    # Execution tracking
    steps_executed: List[Dict[str, Any]] = Field(default_factory=list)
    agent_actions: List[Dict[str, Any]] = Field(default_factory=list)
    tool_executions: List[Dict[str, Any]] = Field(default_factory=list)

    # Results
    metrics: Optional[WorkflowMetrics] = None
    error_details: Optional[Dict[str, Any]] = None

    # Audit trail
    audit_log: List[Dict[str, Any]] = Field(default_factory=list)

    def add_step(self, step_name: str, agent: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Add executed step"""
        self.steps_executed.append({
            "step_name": step_name,
            "agent": agent,
            "inputs": inputs,
            "outputs": outputs,
            "timestamp": datetime.utcnow().isoformat()
        })

    def add_agent_action(self, agent: str, action: str, data: Dict[str, Any]) -> None:
        """Add agent action"""
        self.agent_actions.append({
            "agent": agent,
            "action": action,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        })

    def add_tool_execution(self, tool_name: str, inputs: Dict[str, Any], outputs: Dict[str, Any], execution_time: float) -> None:
        """Add tool execution"""
        self.tool_executions.append({
            "tool_name": tool_name,
            "inputs": inputs,
            "outputs": outputs,
            "execution_time": execution_time,
            "timestamp": datetime.utcnow().isoformat()
        })

    def complete_execution(self, outputs: Dict[str, Any], metrics: WorkflowMetrics) -> None:
        """Mark execution as complete"""
        self.end_time = datetime.utcnow()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.outputs = outputs
        self.metrics = metrics
        self.status = "complete"
