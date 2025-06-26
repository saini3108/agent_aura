"""
Workflow Engine Utilities
=========================

Core workflow management and orchestration utilities
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)

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
    execution_time: float = 0.0
    retry_count: int = 0
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

@dataclass
class HumanReviewCheckpoint:
    checkpoint_id: str
    agent_outputs: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    reviewed_at: Optional[datetime] = None
    reviewer_feedback: Dict[str, Any] = field(default_factory=dict)
    approval_status: Optional[str] = None

class MCPWorkflowEngine:
    def __init__(self, agents, audit_logger, config_dict):
        self.agents = agents
        self.audit_logger = audit_logger
        self.config = config_dict
        self.active_workflows = {}
        self.workflow_history = []
        workflow_config = config_dict.get('workflow_config', {})
        self.execution_order = workflow_config.get('execution_order', [
            "analyst", "validator", "documentation", "human_review", "reviewer", "auditor"
        ])

    async def create_workflow(self, initial_data):
        workflow_id = str(uuid.uuid4())

        agent_executions = []
        for agent_name in self.execution_order:
            if agent_name != "human_review":
                agent_executions.append(AgentExecution(agent_name=agent_name))

        workflow_state = {
            "id": workflow_id,
            "status": WorkflowStatus.INITIALIZED,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "initial_data": initial_data,
            "agent_executions": {exec.agent_name: exec for exec in agent_executions},
            "current_step": 0,
            "results": {},
            "human_checkpoint": None
        }

        self.active_workflows[workflow_id] = workflow_state

        self.audit_logger.log_workflow_event(
            "workflow_created", "initialization", 0,
            {"workflow_id": workflow_id, "agents": len(agent_executions)}
        )

        return workflow_id

    async def execute_workflow_step(self, workflow_id: str, step_name: str) -> Dict[str, Any]:
        """Execute a specific workflow step"""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow = self.active_workflows[workflow_id]

        try:
            # Get agent for the step
            agent = self.agents.get(step_name)
            if not agent:
                raise ValueError(f"Agent {step_name} not found")

            # Prepare context with previous outputs
            context = {
                'data': workflow.get('data'),
                'files': workflow.get('files', {}),
                'previous_outputs': workflow.get('agent_outputs', {}),
                'workflow_id': workflow_id
            }

            # Add direct agent outputs to context for easier access
            agent_outputs = workflow.get('agent_outputs', {})
            context.update({
                'analyst_output': agent_outputs.get('step_0', {}),
                'validator_output': agent_outputs.get('step_1', {}),
                'documentation_output': agent_outputs.get('step_2', {}),
                'reviewer_output': agent_outputs.get('step_4', {})
            })

            # Execute agent
            start_time = time.time()
            result = agent.run(context)
            execution_time = time.time() - start_time

            # Store result with step mapping
            if 'agent_outputs' not in workflow:
                workflow['agent_outputs'] = {}

            step_key = f"step_{len(workflow['agent_outputs'])}"
            workflow['agent_outputs'][step_key] = result

            # Store in agent_executions for status tracking
            if 'agent_executions' not in workflow:
                workflow['agent_executions'] = {}

            workflow['agent_executions'][step_name] = {
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'execution_time': execution_time,
                'output': result
            }

            # Update workflow status
            workflow['last_executed_step'] = step_name
            workflow['last_execution_time'] = datetime.now().isoformat()
            workflow['current_step'] = f"{step_name}_completed"

            return {
                'agent_name': step_name,
                'status': 'completed',
                'execution_time': execution_time,
                'result': result
            }

        except Exception as e:
            logger.error(f"Step execution failed: {e}")

            # Store failed execution
            if 'agent_executions' not in workflow:
                workflow['agent_executions'] = {}

            workflow['agent_executions'][step_name] = {
                'status': 'failed',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

            return {
                'agent_name': step_name,
                'status': 'failed',
                'error': str(e),
                'execution_time': 0
            }

    def _prepare_agent_context(self, workflow_id, agent_name):
        workflow = self.active_workflows[workflow_id]

        context = {
            "workflow_id": workflow_id,
            "data": workflow["initial_data"].get("data", {}),
            "files": workflow["initial_data"].get("documents", {}),
            "agent_name": agent_name
        }

        # Add previous agent outputs
        for prev_agent, result in workflow["results"].items():
            context[f"{prev_agent}_output"] = result

        return context

    async def _create_human_checkpoint(self, workflow_id):
        workflow = self.active_workflows[workflow_id]

        checkpoint_id = str(uuid.uuid4())
        agent_outputs = workflow["results"].copy()

        checkpoint = HumanReviewCheckpoint(
            checkpoint_id=checkpoint_id,
            agent_outputs=agent_outputs
        )

        workflow["human_checkpoint"] = checkpoint
        workflow["status"] = WorkflowStatus.WAITING_FOR_HUMAN
        workflow["updated_at"] = datetime.now()

        review_summary = self._generate_review_summary(agent_outputs)

        self.audit_logger.log_workflow_event(
            "human_checkpoint_created", "human_review", 0,
            {"checkpoint_id": checkpoint_id, "summary": review_summary}
        )

        return {
            "checkpoint_id": checkpoint_id,
            "status": "created",
            "review_summary": review_summary,
            "agent_outputs": agent_outputs
        }

    async def submit_human_feedback(self, workflow_id, checkpoint_id, feedback):
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow = self.active_workflows[workflow_id]
        checkpoint = workflow.get("human_checkpoint")

        if not checkpoint or checkpoint.checkpoint_id != checkpoint_id:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        checkpoint.reviewed_at = datetime.now()
        checkpoint.reviewer_feedback = feedback
        checkpoint.approval_status = feedback.get("approval_status")

        workflow["status"] = WorkflowStatus.RUNNING
        workflow["updated_at"] = datetime.now()

        self.audit_logger.log_human_interaction(
            "feedback_submitted",
            {"checkpoint_id": checkpoint_id, "approval_status": feedback.get("approval_status")}
        )

        return {"status": "submitted", "feedback": feedback}

    def _generate_review_summary(self, agent_outputs):
        summary = {
            "agents_completed": len(agent_outputs),
            "key_findings": []
        }

        if "analyst" in agent_outputs:
            analyst_result = agent_outputs["analyst"]
            summary["key_findings"].append(f"Data analysis: {analyst_result.get('summary', 'Completed')}")

        if "validator" in agent_outputs:
            validator_result = agent_outputs["validator"]
            metrics = validator_result.get('metrics', {})
            summary["key_findings"].append(f"Validation metrics calculated: AUC={metrics.get('auc', 'N/A')}")

        if "documentation" in agent_outputs:
            doc_result = agent_outputs["documentation"]
            summary["key_findings"].append(f"Documentation review: {doc_result.get('compliance_summary', 'Completed')}")

        return summary

    def get_workflow_status(self, workflow_id):
        if workflow_id not in self.active_workflows:
            return None

        workflow = self.active_workflows[workflow_id]

        return {
            "id": workflow_id,
            "status": workflow["status"].value if isinstance(workflow["status"], WorkflowStatus) else workflow["status"],
            "current_step": workflow.get("current_step", 0),
            "created_at": workflow["created_at"].isoformat() if isinstance(workflow["created_at"], datetime) else workflow["created_at"],
            "updated_at": workflow["updated_at"].isoformat() if isinstance(workflow["updated_at"], datetime) else workflow["updated_at"],
            "agent_executions": {
                name: {
                    "status": exec.get("status", "pending") if isinstance(exec, dict) else (exec.status.value if isinstance(exec.status, AgentStatus) else exec.status),
                    "execution_time": exec.get("execution_time", 0.0) if isinstance(exec, dict) else exec.execution_time,
                    "retry_count": exec.get("retry_count", 0) if isinstance(exec, dict) else exec.retry_count
                }
                for name, exec in workflow["agent_executions"].items()
            }
        }

    def get_workflow_results(self, workflow_id):
        if workflow_id not in self.active_workflows:
            return None

        workflow = self.active_workflows[workflow_id]
        results = workflow["results"].copy()

        # Add summary metrics
        if "validator" in results:
            validator_results = results["validator"]
            results["validation_score"] = validator_results.get("validation_score", 0)
            results["metrics"] = validator_results.get("metrics", {})

        if "documentation" in results:
            doc_results = results["documentation"]
            results["compliance_score"] = doc_results.get("compliance_score", 0)

        # Add checkpoint info
        if workflow.get("human_checkpoint"):
            checkpoint = workflow["human_checkpoint"]
            results["checkpoint_id"] = checkpoint.checkpoint_id

        return results

    def cancel_workflow(self, workflow_id):
        """Cancel an active workflow"""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow = self.active_workflows[workflow_id]
        workflow["status"] = WorkflowStatus.CANCELLED
        workflow["updated_at"] = datetime.now()

        # Move to history
        self.workflow_history.append(workflow)
        del self.active_workflows[workflow_id]

        self.audit_logger.log_workflow_event(
            "workflow_cancelled", "cancellation", 0,
            {"workflow_id": workflow_id}
        )

        return {"status": "cancelled", "workflow_id": workflow_id}