"""
Enhanced MCP + LangGraph Integration Engine
Provides robust workflow orchestration with human-in-the-loop capabilities
"""
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, TypedDict
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd

from config.system_config import config

class WorkflowStatus(Enum):
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_FOR_HUMAN = "waiting_for_human"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class AgentExecution:
    agent_name: str
    status: AgentStatus = AgentStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0

@dataclass
class HumanReviewCheckpoint:
    checkpoint_id: str
    agent_outputs: Dict[str, Any]
    review_required: bool = True
    timeout_seconds: int = 3600
    created_at: datetime = field(default_factory=datetime.now)
    reviewed_at: Optional[datetime] = None
    reviewer_feedback: Dict[str, Any] = field(default_factory=dict)
    approval_status: Optional[str] = None  # approved, rejected, modified

class WorkflowState(TypedDict):
    workflow_id: str
    status: WorkflowStatus
    current_step: int
    total_steps: int
    agent_executions: List[AgentExecution]
    human_checkpoints: List[HumanReviewCheckpoint]
    data: Optional[pd.DataFrame]
    documents: Dict[str, Any]
    global_context: Dict[str, Any]
    error_log: List[str]
    created_at: datetime
    updated_at: datetime

class MCPLangGraphEngine:
    """
    Enhanced workflow engine combining MCP agent architecture with LangGraph orchestration
    """
    
    def __init__(self, agents: Dict[str, Any], audit_logger: Any):
        self.agents = agents
        self.audit_logger = audit_logger
        self.active_workflows: Dict[str, WorkflowState] = {}
        self.workflow_history: List[str] = []
        
        # Load configuration
        self.mcp_config = config.mcp_config
        self.workflow_config = config.workflow_config
        self.risk_thresholds = config.risk_thresholds
        
        # Initialize workflow graph
        self._initialize_workflow_graph()
    
    def _initialize_workflow_graph(self):
        """Initialize the workflow execution graph"""
        self.execution_order = self.mcp_config["workflow"]["execution_order"]
        self.parallel_execution = self.mcp_config["workflow"]["parallel_execution"]
        self.checkpoint_enabled = self.mcp_config["workflow"]["checkpoint_enabled"]
        
        # Create agent dependency mapping
        self.agent_dependencies = self._build_dependency_graph()
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build agent dependency graph for execution order"""
        dependencies = {}
        order = self.execution_order
        
        for i, agent in enumerate(order):
            if agent == "human_review":
                # Human review depends on all previous agents
                dependencies[agent] = order[:i]
            elif i == 0:
                # First agent has no dependencies
                dependencies[agent] = []
            else:
                # Each agent depends on the previous agent
                dependencies[agent] = [order[i-1]]
        
        return dependencies
    
    async def create_workflow(self, initial_data: Dict[str, Any]) -> str:
        """Create a new workflow instance"""
        workflow_id = str(uuid.uuid4())
        
        # Initialize agent executions
        agent_executions = []
        for agent_name in self.execution_order:
            if agent_name != "human_review":
                agent_executions.append(AgentExecution(agent_name=agent_name))
        
        # Create workflow state
        workflow_state: WorkflowState = {
            'workflow_id': workflow_id,
            'status': WorkflowStatus.INITIALIZED,
            'current_step': 0,
            'total_steps': len(self.execution_order),
            'agent_executions': agent_executions,
            'human_checkpoints': [],
            'data': initial_data.get('data'),
            'documents': initial_data.get('documents', {}),
            'global_context': initial_data.get('context', {}),
            'error_log': [],
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        
        self.active_workflows[workflow_id] = workflow_state
        self.workflow_history.append(workflow_id)
        
        # Log workflow creation
        self.audit_logger.log_workflow_event(
            "workflow_created",
            "workflow_initialization", 
            0,
            {"workflow_id": workflow_id, "initial_data_keys": list(initial_data.keys())}
        )
        
        return workflow_id
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current workflow status"""
        if workflow_id not in self.active_workflows:
            return None
        
        workflow_state = self.active_workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "status": workflow_state['status'].value,
            "current_step": workflow_state['current_step'],
            "total_steps": workflow_state['total_steps'],
            "progress_percentage": (workflow_state['current_step'] / workflow_state['total_steps']) * 100,
            "agent_statuses": {
                exec_record.agent_name: exec_record.status.value
                for exec_record in workflow_state['agent_executions']
            },
            "pending_human_reviews": [
                cp.checkpoint_id for cp in workflow_state['human_checkpoints']
                if not cp.reviewed_at
            ],
            "created_at": workflow_state['created_at'].isoformat(),
            "updated_at": workflow_state['updated_at'].isoformat()
        }
    
    def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List all active workflows"""
        return [
            self.get_workflow_status(workflow_id)
            for workflow_id in self.active_workflows.keys()
        ]