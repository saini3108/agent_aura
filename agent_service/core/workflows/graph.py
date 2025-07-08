import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from agent_service.core.agents.executor import ExecutorAgent
from agent_service.core.agents.planner import PlannerAgent
from agent_service.core.agents.summarizer import SummarizerAgent
from agent_service.core.agents.validator import ValidatorAgent
from agent_service.core.memory.redis_store import MemoryService
from agent_service.core.schema.context import BaseContext
from agent_service.core.schema.context import WorkflowStatus
from agent_service.core.schema.context import create_context
from agent_service.core.services.llm_client import llm_manager
from agent_service.core.utils.logging_config import AuditLogger
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)

@dataclass
class WorkflowState:
    """State for the workflow graph"""
    context: BaseContext
    messages: Annotated[List[Dict], add_messages]
    current_step: int = 0
    should_continue: bool = True
    human_input_required: bool = False
    error: Optional[str] = None

class BankingWorkflowGraph:
    """LangGraph workflow orchestrator for banking operations"""

    def __init__(self, memory_service: MemoryService):
        self.memory_service = memory_service
        self.audit_logger = AuditLogger("workflow")

        # Initialize agents
        self.planner = PlannerAgent(llm_manager)
        self.executor = ExecutorAgent(llm_manager)
        self.validator = ValidatorAgent(llm_manager)
        self.summarizer = SummarizerAgent(llm_manager)

        # Create workflow graph
        self.graph = self._create_workflow_graph()

        # Compile graph with memory
        self.checkpointer = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.checkpointer)

    def _create_workflow_graph(self) -> StateGraph:
        """Create the workflow graph structure"""

        # Define workflow state schema
        workflow = StateGraph(WorkflowState)

        # Add nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("validator", self._validator_node)
        workflow.add_node("human_review", self._human_review_node)
        workflow.add_node("summarizer", self._summarizer_node)
        workflow.add_node("error_handler", self._error_handler_node)

        # Define edges
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "validator")

        # Conditional edges
        workflow.add_conditional_edges(
            "validator",
            self._should_require_human_review,
            {
                "human_review": "human_review",
                "summarizer": "summarizer",
                "error": "error_handler"
            }
        )

        workflow.add_conditional_edges(
            "human_review",
            self._process_human_feedback,
            {
                "approved": "summarizer",
                "rejected": "planner",
                "modified": "executor",
                "error": "error_handler"
            }
        )

        workflow.add_edge("summarizer", END)
        workflow.add_edge("error_handler", END)

        return workflow

    async def _planner_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute planner agent"""

        try:
            logger.info(f"Executing planner for workflow {state.context.workflow_id}")

            # Save context to memory
            await self.memory_service.save_context(state.context)

            # Execute planner
            updated_context = await self.planner.process(state.context)

            # Update state
            state.context = updated_context
            state.messages.append({
                "role": "planner",
                "content": f"Planning completed with {len(updated_context.plan_steps)} steps",
                "timestamp": datetime.utcnow().isoformat()
            })

            # Log audit event
            self.audit_logger.log_workflow_start(
                workflow_id=state.context.workflow_id,
                workflow_type=state.context.workflow_type,
                inputs=state.context.inputs
            )

            return {
                "context": state.context,
                "messages": state.messages,
                "current_step": state.current_step,
                "should_continue": True
            }

        except Exception as e:
            logger.error(f"Planner node failed: {e}")
            return {
                "context": state.context,
                "messages": state.messages,
                "current_step": state.current_step,
                "should_continue": False,
                "error": str(e)
            }

    async def _executor_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute executor agent"""

        try:
            logger.info(f"Executing executor for workflow {state.context.workflow_id}")

            # Execute executor
            updated_context = await self.executor.process(state.context)

            # Save updated context
            await self.memory_service.save_context(updated_context)

            # Update state
            state.context = updated_context
            state.messages.append({
                "role": "executor",
                "content": f"Execution completed with {len(updated_context.tool_results)} tool results",
                "timestamp": datetime.utcnow().isoformat()
            })

            return {
                "context": state.context,
                "messages": state.messages,
                "current_step": state.current_step + 1,
                "should_continue": True
            }

        except Exception as e:
            logger.error(f"Executor node failed: {e}")
            return {
                "context": state.context,
                "messages": state.messages,
                "current_step": state.current_step,
                "should_continue": False,
                "error": str(e)
            }

    async def _validator_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute validator agent"""

        try:
            logger.info(f"Executing validator for workflow {state.context.workflow_id}")

            # Execute validator
            updated_context = await self.validator.process(state.context)

            # Save updated context
            await self.memory_service.save_context(updated_context)

            # Update state
            state.context = updated_context
            state.messages.append({
                "role": "validator",
                "content": f"Validation completed with status: {updated_context.status.value}",
                "timestamp": datetime.utcnow().isoformat()
            })

            return {
                "context": state.context,
                "messages": state.messages,
                "current_step": state.current_step + 1,
                "should_continue": True
            }

        except Exception as e:
            logger.error(f"Validator node failed: {e}")
            return {
                "context": state.context,
                "messages": state.messages,
                "current_step": state.current_step,
                "should_continue": False,
                "error": str(e)
            }

    async def _human_review_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Handle human review requirement"""

        try:
            logger.info(f"Waiting for human review for workflow {state.context.workflow_id}")

            # Mark as requiring human input
            state.context.requires_human_review = True
            state.context.update_status(WorkflowStatus.AWAITING_REVIEW)

            # Save context
            await self.memory_service.save_context(state.context)

            # Update state
            state.human_input_required = True
            state.messages.append({
                "role": "system",
                "content": "Workflow paused for human review",
                "timestamp": datetime.utcnow().isoformat()
            })

            # Log audit event
            self.audit_logger.log_agent_action(
                workflow_id=state.context.workflow_id,
                agent_name="human_review",
                action="pause_for_review",
                inputs={"status": state.context.status.value},
                outputs={"requires_review": True}
            )

            return {
                "context": state.context,
                "messages": state.messages,
                "current_step": state.current_step,
                "should_continue": False,
                "human_input_required": True
            }

        except Exception as e:
            logger.error(f"Human review node failed: {e}")
            return {
                "context": state.context,
                "messages": state.messages,
                "current_step": state.current_step,
                "should_continue": False,
                "error": str(e)
            }

    async def _summarizer_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute summarizer agent"""

        try:
            logger.info(f"Executing summarizer for workflow {state.context.workflow_id}")

            # Execute summarizer
            updated_context = await self.summarizer.process(state.context)

            # Save final context
            await self.memory_service.save_context(updated_context)

            # Update state
            state.context = updated_context
            state.messages.append({
                "role": "summarizer",
                "content": f"Summarization completed, workflow status: {updated_context.status.value}",
                "timestamp": datetime.utcnow().isoformat()
            })

            # Log audit event
            self.audit_logger.log_workflow_end(
                workflow_id=state.context.workflow_id,
                status=updated_context.status.value,
                outputs=updated_context.outputs
            )

            return {
                "context": state.context,
                "messages": state.messages,
                "current_step": state.current_step + 1,
                "should_continue": False
            }

        except Exception as e:
            logger.error(f"Summarizer node failed: {e}")
            return {
                "context": state.context,
                "messages": state.messages,
                "current_step": state.current_step,
                "should_continue": False,
                "error": str(e)
            }

    async def _error_handler_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Handle workflow errors"""

        try:
            logger.error(f"Handling error for workflow {state.context.workflow_id}: {state.error}")

            # Update context with error
            state.context.update_status(WorkflowStatus.FAILED)
            state.context.outputs["error"] = state.error

            # Save context
            await self.memory_service.save_context(state.context)

            # Update state
            state.messages.append({
                "role": "error_handler",
                "content": f"Workflow failed: {state.error}",
                "timestamp": datetime.utcnow().isoformat()
            })

            # Log audit event
            self.audit_logger.log_error(
                workflow_id=state.context.workflow_id,
                error_type="workflow_error",
                error_message=state.error,
                context={"current_step": state.current_step}
            )

            return {
                "context": state.context,
                "messages": state.messages,
                "current_step": state.current_step,
                "should_continue": False,
                "error": state.error
            }

        except Exception as e:
            logger.error(f"Error handler node failed: {e}")
            return {
                "context": state.context,
                "messages": state.messages,
                "current_step": state.current_step,
                "should_continue": False,
                "error": str(e)
            }

    def _should_require_human_review(self, state: WorkflowState) -> str:
        """Determine if human review is required"""

        # Check if workflow failed
        if state.context.status == WorkflowStatus.FAILED:
            return "error"

        # Check if human review is explicitly required
        if state.context.requires_human_review:
            return "human_review"

        # Check validation results
        if hasattr(state.context, 'validation_results'):
            failed_validations = [r for r in state.context.validation_results if not r.passed]
            if failed_validations:
                return "human_review"

        # Check for tool failures
        failed_tools = [r for r in state.context.tool_results if not r.success]
        if failed_tools:
            return "human_review"

        # Default to summarizer
        return "summarizer"

    def _process_human_feedback(self, state: WorkflowState) -> str:
        """Process human feedback and determine next step"""

        if not state.context.human_feedback:
            return "error"

        feedback = state.context.human_feedback

        # Log human intervention
        self.audit_logger.log_human_intervention(
            workflow_id=state.context.workflow_id,
            action=feedback.action,
            user_id=feedback.user_id,
            decision=feedback.comments or "No comments"
        )

        # Route based on feedback action
        if feedback.action == "approve":
            return "approved"
        elif feedback.action == "reject":
            return "rejected"
        elif feedback.action == "modify":
            return "modified"
        else:
            return "error"

    async def start_workflow(self, workflow_type: str, inputs: Dict[str, Any],
                           config: Optional[Dict[str, Any]] = None) -> str:
        """Start a new workflow"""

        try:
            # Create workflow context
            context = create_context(workflow_type, inputs=inputs)

            # Apply configuration if provided
            if config:
                context.model_config.update(config)

            # Create initial state
            initial_state = WorkflowState(
                context=context,
                messages=[{
                    "role": "system",
                    "content": f"Starting {workflow_type} workflow",
                    "timestamp": datetime.utcnow().isoformat()
                }]
            )

            # Start workflow execution
            thread_id = f"workflow_{context.workflow_id}"

            # Run workflow asynchronously
            asyncio.create_task(self._run_workflow(initial_state, thread_id))

            logger.info(f"Workflow started: {context.workflow_id} (type: {workflow_type})")

            return context.workflow_id

        except Exception as e:
            logger.error(f"Failed to start workflow: {e}")
            raise

    async def _run_workflow(self, initial_state: WorkflowState, thread_id: str) -> None:
        """Run the workflow to completion"""

        try:
            # Execute workflow
            final_state = await self.app.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": thread_id}}
            )

            logger.info(f"Workflow completed: {initial_state.context.workflow_id}")

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")

            # Update context with error
            initial_state.context.update_status(WorkflowStatus.FAILED)
            initial_state.context.outputs["error"] = str(e)

            # Save error state
            await self.memory_service.save_context(initial_state.context)

    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current workflow status"""

        try:
            # Load context from memory
            context = await self.memory_service.load_context(workflow_id)

            if not context:
                return None

            return {
                "workflow_id": workflow_id,
                "workflow_type": context.workflow_type,
                "status": context.status.value,
                "current_step": context.current_step,
                "total_steps": len(context.plan_steps),
                "progress": (context.current_step / len(context.plan_steps) * 100) if context.plan_steps else 0,
                "created_at": context.created_at.isoformat(),
                "updated_at": context.updated_at.isoformat(),
                "requires_human_review": context.requires_human_review,
                "agent_messages": context.agent_messages[-5:],  # Last 5 messages
                "tool_results_count": len(context.tool_results)
            }

        except Exception as e:
            logger.error(f"Failed to get workflow status: {e}")
            return None

    async def submit_human_feedback(self, workflow_id: str, feedback: Dict[str, Any]) -> bool:
        """Submit human feedback to continue workflow"""

        try:
            # Load context
            context = await self.memory_service.load_context(workflow_id)

            if not context:
                logger.error(f"Workflow not found: {workflow_id}")
                return False

            # Create human feedback
            from agent_service.core.schema.context import HumanFeedback
            human_feedback = HumanFeedback(
                user_id=feedback.get("user_id", "unknown"),
                action=feedback.get("action", "approve"),
                comments=feedback.get("comments"),
                modifications=feedback.get("modifications")
            )

            # Update context
            context.set_human_feedback(human_feedback)
            context.requires_human_review = False

            # Save context
            await self.memory_service.save_context(context)

            # Resume workflow
            thread_id = f"workflow_{workflow_id}"

            # Create state for continuation
            continue_state = WorkflowState(
                context=context,
                messages=[{
                    "role": "human",
                    "content": f"Human feedback received: {feedback.get('action', 'approve')}",
                    "timestamp": datetime.utcnow().isoformat()
                }]
            )

            # Continue workflow execution
            asyncio.create_task(self._continue_workflow(continue_state, thread_id))

            logger.info(f"Human feedback submitted for workflow: {workflow_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to submit human feedback: {e}")
            return False

    async def _continue_workflow(self, state: WorkflowState, thread_id: str) -> None:
        """Continue workflow execution after human feedback"""

        try:
            # Continue from human review node
            final_state = await self.app.ainvoke(
                state,
                config={"configurable": {"thread_id": thread_id}}
            )

            logger.info(f"Workflow continued: {state.context.workflow_id}")

        except Exception as e:
            logger.error(f"Workflow continuation failed: {e}")

            # Update context with error
            state.context.update_status(WorkflowStatus.FAILED)
            state.context.outputs["error"] = str(e)

            # Save error state
            await self.memory_service.save_context(state.context)

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow"""

        try:
            # Load context
            context = await self.memory_service.load_context(workflow_id)

            if not context:
                return False

            # Update status to cancelled
            context.update_status(WorkflowStatus.CANCELLED)

            # Save context
            await self.memory_service.save_context(context)

            # Log cancellation
            self.audit_logger.log_workflow_end(
                workflow_id=workflow_id,
                status=WorkflowStatus.CANCELLED.value,
                outputs={"cancelled_by": "user"}
            )

            logger.info(f"Workflow cancelled: {workflow_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to cancel workflow: {e}")
            return False

    async def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List all active workflows"""

        try:
            # Get active workflow IDs
            workflow_ids = await self.memory_service.list_active_workflows()

            # Get status for each workflow
            workflows = []
            for workflow_id in workflow_ids:
                status = await self.get_workflow_status(workflow_id)
                if status:
                    workflows.append(status)

            return workflows

        except Exception as e:
            logger.error(f"Failed to list active workflows: {e}")
            return []

    async def get_workflow_results(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get final workflow results"""

        try:
            # Load context
            context = await self.memory_service.load_context(workflow_id)

            if not context:
                return None

            return {
                "workflow_id": workflow_id,
                "workflow_type": context.workflow_type,
                "status": context.status.value,
                "inputs": context.inputs,
                "outputs": context.outputs,
                "plan_steps": context.plan_steps,
                "tool_results": [
                    {
                        "tool_name": result.tool_name,
                        "success": result.success,
                        "execution_time": result.execution_time,
                        "outputs": result.outputs,
                        "error_message": result.error_message
                    } for result in context.tool_results
                ],
                "agent_messages": context.agent_messages,
                "created_at": context.created_at.isoformat(),
                "updated_at": context.updated_at.isoformat(),
                "completed_at": context.completed_at.isoformat() if context.completed_at else None
            }

        except Exception as e:
            logger.error(f"Failed to get workflow results: {e}")
            return None
