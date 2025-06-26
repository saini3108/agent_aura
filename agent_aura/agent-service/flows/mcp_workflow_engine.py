"""
MCP + LangGraph Workflow Engine
Robust workflow orchestration with human-in-the-loop capabilities
"""
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, TypedDict, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd

# Import shared utilities
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "shared"))

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
    approval_status: Optional[str] = None

class WorkflowState(TypedDict):
    workflow_id: str
    status: WorkflowStatus
    current_step: int
    total_steps: int
    execution_order: List[str]
    agent_executions: List[AgentExecution]
    human_checkpoints: List[HumanReviewCheckpoint]
    data: Optional[pd.DataFrame]
    documents: Dict[str, Any]
    global_context: Dict[str, Any]
    error_log: List[str]
    created_at: datetime
    updated_at: datetime

class MCPWorkflowEngine:
    """
    Enhanced MCP workflow engine with LangGraph-style orchestration
    """
    
    def __init__(self, agents: Dict[str, Any], audit_logger: Any, config: Dict[str, Any]):
        self.agents = agents
        self.audit_logger = audit_logger
        self.config = config
        self.active_workflows: Dict[str, WorkflowState] = {}
        self.workflow_history: List[str] = []
        
        # Load workflow configuration
        self.mcp_config = config.get("mcp_config", {})
        self.workflow_config = config.get("workflow_config", {})
        self.risk_thresholds = config.get("risk_thresholds", {})
        
        # Initialize workflow execution order
        self.execution_order = self.mcp_config.get("workflow", {}).get(
            "execution_order", 
            ["analyst", "validator", "documentation", "human_review", "reviewer", "auditor"]
        )
    
    async def create_workflow(self, initial_data: Dict[str, Any], workflow_type: str = "standard") -> str:
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
            'execution_order': self.execution_order.copy(),
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
            {
                "workflow_id": workflow_id, 
                "workflow_type": workflow_type,
                "initial_data_keys": list(initial_data.keys())
            }
        )
        
        return workflow_id
    
    async def execute_workflow_step(self, workflow_id: str, step_name: str) -> Dict[str, Any]:
        """Execute a single workflow step"""
        if workflow_id not in self.active_workflows:
            return {"success": False, "error": "Workflow not found"}
        
        workflow_state = self.active_workflows[workflow_id]
        
        if step_name == "human_review":
            return await self._create_human_checkpoint(workflow_id)
        else:
            return await self._execute_agent_step(workflow_id, step_name)
    
    async def _execute_agent_step(self, workflow_id: str, agent_name: str) -> Dict[str, Any]:
        """Execute a single agent step with retry logic"""
        workflow_state = self.active_workflows[workflow_id]
        
        # Find agent execution record
        agent_execution = None
        for exec_record in workflow_state['agent_executions']:
            if exec_record.agent_name == agent_name:
                agent_execution = exec_record
                break
        
        if not agent_execution:
            return {"success": False, "error": f"Agent execution record not found for {agent_name}"}
        
        # Prepare agent context
        context = self._prepare_agent_context(workflow_id, agent_name)
        
        # Get retry configuration
        max_retries = self.workflow_config.get("retry_policy", {}).get("max_retries", 3)
        
        agent_execution.status = AgentStatus.RUNNING
        agent_execution.start_time = datetime.now()
        
        for attempt in range(max_retries + 1):
            try:
                agent_execution.retry_count = attempt
                
                # Execute agent
                if agent_name in self.agents:
                    agent_instance = self.agents[agent_name]
                    result = agent_instance.run(context)
                    
                    # Record successful execution
                    agent_execution.status = AgentStatus.COMPLETED
                    agent_execution.end_time = datetime.now()
                    agent_execution.execution_time = (
                        agent_execution.end_time - agent_execution.start_time
                    ).total_seconds()
                    agent_execution.output_data = result
                    
                    # Update workflow state
                    workflow_state['updated_at'] = datetime.now()
                    
                    # Log successful execution
                    self.audit_logger.log_agent_execution(
                        agent_name,
                        "completed",
                        agent_execution.execution_time,
                        result
                    )
                    
                    return {"success": True, "result": result, "execution_time": agent_execution.execution_time}
                else:
                    raise ValueError(f"Agent {agent_name} not found")
                    
            except Exception as e:
                if attempt < max_retries:
                    # Wait before retry
                    retry_delay = self.workflow_config.get("retry_policy", {}).get("retry_delay", 5)
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    # Final failure
                    agent_execution.status = AgentStatus.FAILED
                    agent_execution.end_time = datetime.now()
                    agent_execution.error_message = str(e)
                    workflow_state['error_log'].append(f"Agent {agent_name} failed: {str(e)}")
                    
                    self.audit_logger.log_error(
                        "agent_execution_error",
                        str(e),
                        {"agent_name": agent_name, "workflow_id": workflow_id, "attempts": attempt + 1}
                    )
                    
                    return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Unknown execution error"}
    
    def _prepare_agent_context(self, workflow_id: str, agent_name: str) -> Dict[str, Any]:
        """Prepare context for agent execution"""
        workflow_state = self.active_workflows[workflow_id]
        
        # Collect outputs from previous agents
        previous_outputs = {}
        for exec_record in workflow_state['agent_executions']:
            if exec_record.status == AgentStatus.COMPLETED:
                previous_outputs[exec_record.agent_name] = exec_record.output_data
        
        # Build context
        context = {
            'workflow_id': workflow_id,
            'agent_name': agent_name,
            'data': workflow_state['data'],
            'files': workflow_state['documents'],
            'previous_outputs': previous_outputs,
            'global_context': workflow_state['global_context'],
            'risk_thresholds': self.risk_thresholds,
            'config': self.mcp_config.get("agents", {}).get(agent_name, {})
        }
        
        return context
    
    async def _create_human_checkpoint(self, workflow_id: str) -> Dict[str, Any]:
        """Create human review checkpoint"""
        workflow_state = self.active_workflows[workflow_id]
        
        # Collect all completed agent outputs
        agent_outputs = {}
        for exec_record in workflow_state['agent_executions']:
            if exec_record.status == AgentStatus.COMPLETED:
                agent_outputs[exec_record.agent_name] = exec_record.output_data
        
        # Create checkpoint
        checkpoint_id = str(uuid.uuid4())
        timeout_seconds = self.mcp_config.get("human_in_loop", {}).get("timeout", 3600)
        
        checkpoint = HumanReviewCheckpoint(
            checkpoint_id=checkpoint_id,
            agent_outputs=agent_outputs,
            timeout_seconds=timeout_seconds
        )
        
        workflow_state['human_checkpoints'].append(checkpoint)
        workflow_state['status'] = WorkflowStatus.WAITING_FOR_HUMAN
        workflow_state['updated_at'] = datetime.now()
        
        # Log checkpoint creation
        self.audit_logger.log_workflow_event(
            "human_checkpoint_created",
            "human_review",
            workflow_state['current_step'],
            {"checkpoint_id": checkpoint_id, "agent_count": len(agent_outputs)}
        )
        
        return {
            "success": True,
            "requires_review": True,
            "checkpoint_id": checkpoint_id,
            "review_data": {
                "agent_outputs": agent_outputs,
                "workflow_summary": self._generate_review_summary(agent_outputs),
                "recommendations": self._generate_review_recommendations(agent_outputs)
            }
        }
    
    async def submit_human_feedback(self, workflow_id: str, checkpoint_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Submit human feedback and update workflow state"""
        if workflow_id not in self.active_workflows:
            return {"success": False, "error": "Workflow not found"}
        
        workflow_state = self.active_workflows[workflow_id]
        
        # Find checkpoint
        checkpoint = None
        for cp in workflow_state['human_checkpoints']:
            if cp.checkpoint_id == checkpoint_id:
                checkpoint = cp
                break
        
        if not checkpoint:
            return {"success": False, "error": "Checkpoint not found"}
        
        # Update checkpoint with feedback
        checkpoint.reviewed_at = datetime.now()
        checkpoint.reviewer_feedback = feedback
        checkpoint.approval_status = feedback.get("approval_status", "approved")
        
        # Update workflow state
        workflow_state['status'] = WorkflowStatus.RUNNING
        workflow_state['updated_at'] = datetime.now()
        
        # Log human feedback
        self.audit_logger.log_human_interaction(
            "checkpoint_feedback",
            {
                "checkpoint_id": checkpoint_id, 
                "approval_status": checkpoint.approval_status,
                "feedback_summary": feedback.get("summary", "No summary provided")
            }
        )
        
        return {
            "success": True,
            "status": "feedback_submitted",
            "approval_status": checkpoint.approval_status,
            "message": "Human feedback submitted successfully"
        }
    
    def _generate_review_summary(self, agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary for human review"""
        summary = {
            "total_agents_completed": len(agent_outputs),
            "key_findings": [],
            "risk_indicators": [],
            "validation_metrics": {},
            "documentation_status": "unknown"
        }
        
        # Extract key metrics from validator agent
        if "validator" in agent_outputs:
            validator_results = agent_outputs["validator"]
            if "validation_results" in validator_results:
                summary["validation_metrics"] = validator_results["validation_results"].get("metrics", {})
        
        # Extract findings from analyst agent
        if "analyst" in agent_outputs:
            analyst_results = agent_outputs["analyst"]
            if "analysis" in analyst_results:
                analysis = analyst_results["analysis"]
                summary["key_findings"].extend([
                    f"Data quality score: {analysis.get('data_analysis', {}).get('quality_score', 'N/A')}",
                    f"Features analyzed: {len(analysis.get('model_analysis', {}).get('features', []))}"
                ])
        
        # Extract documentation status
        if "documentation" in agent_outputs:
            doc_results = agent_outputs["documentation"]
            if "review_results" in doc_results:
                doc_status = doc_results["review_results"].get("overall_assessment", {}).get("status", "unknown")
                summary["documentation_status"] = doc_status
        
        return summary
    
    def _generate_review_recommendations(self, agent_outputs: Dict[str, Any]) -> List[str]:
        """Generate recommendations for human review"""
        recommendations = []
        
        # Check validation metrics
        if "validator" in agent_outputs:
            validator_results = agent_outputs["validator"]
            metrics = validator_results.get("validation_results", {}).get("metrics", {})
            
            for metric, value in metrics.items():
                if metric == "auc" and value < 0.7:
                    recommendations.append(f"AUC of {value:.3f} is below recommended threshold - consider model improvements")
                elif metric == "ks" and value < 0.2:
                    recommendations.append(f"KS statistic of {value:.3f} indicates weak separation - review feature selection")
                elif metric == "psi" and value > 0.25:
                    recommendations.append(f"PSI of {value:.3f} indicates significant drift - investigate data changes")
        
        # Check documentation coverage
        if "documentation" in agent_outputs:
            doc_results = agent_outputs["documentation"]
            if doc_results.get("review_results", {}).get("overall_assessment", {}).get("status") == "incomplete":
                recommendations.append("Documentation appears incomplete - ensure all compliance requirements are met")
        
        if not recommendations:
            recommendations.append("All validation checks passed - ready for final review")
        
        return recommendations
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current workflow status"""
        if workflow_id not in self.active_workflows:
            return None
        
        workflow_state = self.active_workflows[workflow_id]
        
        # Calculate progress
        completed_agents = sum(1 for exec_record in workflow_state['agent_executions'] 
                             if exec_record.status == AgentStatus.COMPLETED)
        progress_percentage = (completed_agents / len(workflow_state['agent_executions'])) * 100
        
        return {
            "workflow_id": workflow_id,
            "status": workflow_state['status'].value,
            "current_step": workflow_state['current_step'],
            "total_steps": workflow_state['total_steps'],
            "progress_percentage": progress_percentage,
            "execution_order": workflow_state['execution_order'],
            "agent_statuses": {
                exec_record.agent_name: {
                    "status": exec_record.status.value,
                    "execution_time": exec_record.execution_time,
                    "retry_count": exec_record.retry_count
                }
                for exec_record in workflow_state['agent_executions']
            },
            "pending_human_reviews": [
                {
                    "checkpoint_id": cp.checkpoint_id,
                    "created_at": cp.created_at.isoformat(),
                    "timeout_seconds": cp.timeout_seconds
                }
                for cp in workflow_state['human_checkpoints']
                if not cp.reviewed_at
            ],
            "error_count": len(workflow_state['error_log']),
            "created_at": workflow_state['created_at'].isoformat(),
            "updated_at": workflow_state['updated_at'].isoformat()
        }
    
    def get_workflow_results(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow execution results"""
        if workflow_id not in self.active_workflows:
            return None
        
        workflow_state = self.active_workflows[workflow_id]
        
        results = {
            "workflow_id": workflow_id,
            "status": workflow_state['status'].value,
            "execution_summary": {
                "total_agents": len(workflow_state['agent_executions']),
                "completed_agents": sum(1 for e in workflow_state['agent_executions'] 
                                      if e.status == AgentStatus.COMPLETED),
                "failed_agents": sum(1 for e in workflow_state['agent_executions'] 
                                   if e.status == AgentStatus.FAILED),
                "total_execution_time": sum(e.execution_time for e in workflow_state['agent_executions']),
                "human_reviews": len([cp for cp in workflow_state['human_checkpoints'] if cp.reviewed_at])
            },
            "agent_outputs": {},
            "human_feedback": [],
            "errors": workflow_state['error_log']
        }
        
        # Collect agent outputs
        for exec_record in workflow_state['agent_executions']:
            if exec_record.status == AgentStatus.COMPLETED:
                results["agent_outputs"][exec_record.agent_name] = exec_record.output_data
        
        # Collect human feedback
        for checkpoint in workflow_state['human_checkpoints']:
            if checkpoint.reviewed_at:
                results["human_feedback"].append({
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "approval_status": checkpoint.approval_status,
                    "feedback": checkpoint.reviewer_feedback,
                    "reviewed_at": checkpoint.reviewed_at.isoformat()
                })
        
        return results
    
    def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List all active workflows"""
        return [
            self.get_workflow_status(workflow_id)
            for workflow_id in self.active_workflows.keys()
        ]
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow"""
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]['status'] = WorkflowStatus.CANCELLED
            self.active_workflows[workflow_id]['updated_at'] = datetime.now()
            
            self.audit_logger.log_workflow_event(
                "workflow_cancelled",
                "manual_cancellation",
                self.active_workflows[workflow_id]['current_step'],
                {"workflow_id": workflow_id}
            )
            
            return True
        return False