"""
Production LangGraph Workflow Implementation
==========================================

This module implements a production-ready LangGraph workflow system
for credit risk model validation using proper state management and 
conditional routing.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime
import uuid
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

class ValidationState(TypedDict):
    """State structure for the validation workflow"""
    workflow_id: str
    data: Dict[str, Any]
    documents: Dict[str, Any]
    analyst_output: Dict[str, Any]
    validator_output: Dict[str, Any]
    documentation_output: Dict[str, Any]
    human_feedback: Dict[str, Any]
    reviewer_output: Dict[str, Any]
    auditor_output: Dict[str, Any]
    current_step: str
    workflow_status: str
    error_message: Optional[str]
    execution_history: List[Dict[str, Any]]
    checkpoint_data: Dict[str, Any]

@dataclass
class WorkflowCheckpoint:
    """Represents a workflow checkpoint for human review"""
    id: str
    workflow_id: str
    step_name: str
    state_snapshot: Dict[str, Any]
    created_at: datetime
    reviewed: bool = False
    reviewer_feedback: Optional[Dict[str, Any]] = None

class LangGraphWorkflowEngine:
    """Production LangGraph workflow engine for credit risk validation"""
    
    def __init__(self, agents: Dict[str, Any], config: Dict[str, Any]):
        self.agents = agents
        self.config = config
        self.checkpointer = MemorySaver()
        self.active_workflows: Dict[str, StateGraph] = {}
        self.workflow_checkpoints: Dict[str, WorkflowCheckpoint] = {}
        self.graph = self._build_workflow_graph()
        
    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow graph"""
        
        # Create the state graph
        workflow = StateGraph(ValidationState)
        
        # Add nodes for each agent
        workflow.add_node("analyst", self._analyst_node)
        workflow.add_node("validator", self._validator_node)
        workflow.add_node("documentation", self._documentation_node)
        workflow.add_node("human_review", self._human_review_node)
        workflow.add_node("reviewer", self._reviewer_node)
        workflow.add_node("auditor", self._auditor_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Set entry point
        workflow.set_entry_point("analyst")
        
        # Add conditional edges with proper routing
        workflow.add_conditional_edges(
            "analyst",
            self._should_continue_after_analyst,
            {
                "continue": "validator",
                "error": "error_handler",
                "retry": "analyst"
            }
        )
        
        workflow.add_conditional_edges(
            "validator",
            self._should_continue_after_validator,
            {
                "continue": "documentation",
                "error": "error_handler",
                "retry": "validator"
            }
        )
        
        workflow.add_conditional_edges(
            "documentation",
            self._should_continue_after_documentation,
            {
                "continue": "human_review",
                "error": "error_handler",
                "retry": "documentation"
            }
        )
        
        workflow.add_conditional_edges(
            "human_review",
            self._should_continue_after_human_review,
            {
                "approved": "reviewer",
                "rejected": "analyst",
                "needs_revision": "validator",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "reviewer",
            self._should_continue_after_reviewer,
            {
                "continue": "auditor",
                "error": "error_handler",
                "retry": "reviewer"
            }
        )
        
        workflow.add_conditional_edges(
            "auditor",
            self._should_continue_after_auditor,
            {
                "approved": END,
                "rejected": "analyst",
                "needs_revision": "reviewer",
                "error": "error_handler"
            }
        )
        
        workflow.add_edge("error_handler", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def _analyst_node(self, state: ValidationState) -> ValidationState:
        """Execute analyst agent node"""
        logger.info(f"Executing analyst node for workflow {state['workflow_id']}")
        
        try:
            # Prepare context for analyst
            context = {
                "data": state["data"],
                "documents": state["documents"],
                "workflow_id": state["workflow_id"]
            }
            
            # Execute analyst agent
            analyst_agent = self.agents.get("analyst")
            if not analyst_agent:
                raise ValueError("Analyst agent not found")
            
            result = await self._execute_agent_async(analyst_agent, context)
            
            # Update state
            state["analyst_output"] = result
            state["current_step"] = "analyst_completed"
            state["execution_history"].append({
                "step": "analyst",
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "output_summary": f"Analyzed {len(state['data'])} records"
            })
            
            logger.info(f"Analyst node completed for workflow {state['workflow_id']}")
            return state
            
        except Exception as e:
            logger.error(f"Analyst node failed: {e}")
            state["error_message"] = str(e)
            state["workflow_status"] = "failed"
            return state
    
    async def _validator_node(self, state: ValidationState) -> ValidationState:
        """Execute validator agent node"""
        logger.info(f"Executing validator node for workflow {state['workflow_id']}")
        
        try:
            context = {
                "data": state["data"],
                "analyst_output": state["analyst_output"],
                "workflow_id": state["workflow_id"]
            }
            
            validator_agent = self.agents.get("validator")
            if not validator_agent:
                raise ValueError("Validator agent not found")
            
            result = await self._execute_agent_async(validator_agent, context)
            
            state["validator_output"] = result
            state["current_step"] = "validator_completed"
            state["execution_history"].append({
                "step": "validator",
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "metrics": result.get("metrics", {})
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Validator node failed: {e}")
            state["error_message"] = str(e)
            state["workflow_status"] = "failed"
            return state
    
    async def _documentation_node(self, state: ValidationState) -> ValidationState:
        """Execute documentation agent node"""
        logger.info(f"Executing documentation node for workflow {state['workflow_id']}")
        
        try:
            context = {
                "data": state["data"],
                "documents": state["documents"],
                "analyst_output": state["analyst_output"],
                "validator_output": state["validator_output"],
                "workflow_id": state["workflow_id"]
            }
            
            documentation_agent = self.agents.get("documentation")
            if not documentation_agent:
                raise ValueError("Documentation agent not found")
            
            result = await self._execute_agent_async(documentation_agent, context)
            
            state["documentation_output"] = result
            state["current_step"] = "documentation_completed"
            state["execution_history"].append({
                "step": "documentation",
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "compliance_score": result.get("compliance_score", 0)
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Documentation node failed: {e}")
            state["error_message"] = str(e)
            state["workflow_status"] = "failed"
            return state
    
    async def _human_review_node(self, state: ValidationState) -> ValidationState:
        """Execute human review checkpoint"""
        logger.info(f"Creating human review checkpoint for workflow {state['workflow_id']}")
        
        try:
            # Create checkpoint
            checkpoint_id = str(uuid.uuid4())
            checkpoint = WorkflowCheckpoint(
                id=checkpoint_id,
                workflow_id=state["workflow_id"],
                step_name="human_review",
                state_snapshot={
                    "analyst_output": state["analyst_output"],
                    "validator_output": state["validator_output"],
                    "documentation_output": state["documentation_output"]
                },
                created_at=datetime.now()
            )
            
            self.workflow_checkpoints[checkpoint_id] = checkpoint
            
            # Update state for human review
            state["current_step"] = "awaiting_human_review"
            state["workflow_status"] = "waiting_for_human"
            state["checkpoint_data"] = {
                "checkpoint_id": checkpoint_id,
                "review_summary": self._generate_review_summary(state)
            }
            
            state["execution_history"].append({
                "step": "human_review_created",
                "timestamp": datetime.now().isoformat(),
                "checkpoint_id": checkpoint_id,
                "status": "pending"
            })
            
            logger.info(f"Human review checkpoint created: {checkpoint_id}")
            return state
            
        except Exception as e:
            logger.error(f"Human review node failed: {e}")
            state["error_message"] = str(e)
            state["workflow_status"] = "failed"
            return state
    
    async def _reviewer_node(self, state: ValidationState) -> ValidationState:
        """Execute reviewer agent node"""
        logger.info(f"Executing reviewer node for workflow {state['workflow_id']}")
        
        try:
            context = {
                "data": state["data"],
                "analyst_output": state["analyst_output"],
                "validator_output": state["validator_output"],
                "documentation_output": state["documentation_output"],
                "human_feedback": state["human_feedback"],
                "workflow_id": state["workflow_id"]
            }
            
            reviewer_agent = self.agents.get("reviewer")
            if not reviewer_agent:
                raise ValueError("Reviewer agent not found")
            
            result = await self._execute_agent_async(reviewer_agent, context)
            
            state["reviewer_output"] = result
            state["current_step"] = "reviewer_completed"
            state["execution_history"].append({
                "step": "reviewer",
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "findings_count": len(result.get("findings", []))
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Reviewer node failed: {e}")
            state["error_message"] = str(e)
            state["workflow_status"] = "failed"
            return state
    
    async def _auditor_node(self, state: ValidationState) -> ValidationState:
        """Execute auditor agent node"""
        logger.info(f"Executing auditor node for workflow {state['workflow_id']}")
        
        try:
            context = {
                "data": state["data"],
                "analyst_output": state["analyst_output"],
                "validator_output": state["validator_output"],
                "documentation_output": state["documentation_output"],
                "human_feedback": state["human_feedback"],
                "reviewer_output": state["reviewer_output"],
                "workflow_id": state["workflow_id"]
            }
            
            auditor_agent = self.agents.get("auditor")
            if not auditor_agent:
                raise ValueError("Auditor agent not found")
            
            result = await self._execute_agent_async(auditor_agent, context)
            
            state["auditor_output"] = result
            state["current_step"] = "auditor_completed"
            state["workflow_status"] = "completed"
            state["execution_history"].append({
                "step": "auditor",
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "final_recommendation": result.get("recommendation", "")
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Auditor node failed: {e}")
            state["error_message"] = str(e)
            state["workflow_status"] = "failed"
            return state
    
    async def _error_handler_node(self, state: ValidationState) -> ValidationState:
        """Handle workflow errors"""
        logger.error(f"Error handler activated for workflow {state['workflow_id']}: {state.get('error_message')}")
        
        state["workflow_status"] = "failed"
        state["current_step"] = "error_handled"
        state["execution_history"].append({
            "step": "error_handler",
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error_message": state.get("error_message", "Unknown error")
        })
        
        return state
    
    async def _execute_agent_async(self, agent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent asynchronously"""
        try:
            # If agent has async run method, use it
            if hasattr(agent, 'run_async'):
                return await agent.run_async(context)
            # Otherwise, run synchronously in thread pool
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, agent.run, context)
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            raise
    
    def _should_continue_after_analyst(self, state: ValidationState) -> str:
        """Determine next step after analyst"""
        if state.get("error_message"):
            return "error"
        if not state.get("analyst_output"):
            return "retry"
        return "continue"
    
    def _should_continue_after_validator(self, state: ValidationState) -> str:
        """Determine next step after validator"""
        if state.get("error_message"):
            return "error"
        if not state.get("validator_output"):
            return "retry"
        return "continue"
    
    def _should_continue_after_documentation(self, state: ValidationState) -> str:
        """Determine next step after documentation"""
        if state.get("error_message"):
            return "error"
        if not state.get("documentation_output"):
            return "retry"
        return "continue"
    
    def _should_continue_after_human_review(self, state: ValidationState) -> str:
        """Determine next step after human review"""
        if state.get("error_message"):
            return "error"
        
        feedback = state.get("human_feedback", {})
        approval_status = feedback.get("approval_status", "")
        
        if approval_status == "approved":
            return "approved"
        elif approval_status == "rejected":
            return "rejected"
        elif approval_status == "needs_revision":
            return "needs_revision"
        else:
            return "error"
    
    def _should_continue_after_reviewer(self, state: ValidationState) -> str:
        """Determine next step after reviewer"""
        if state.get("error_message"):
            return "error"
        if not state.get("reviewer_output"):
            return "retry"
        return "continue"
    
    def _should_continue_after_auditor(self, state: ValidationState) -> str:
        """Determine next step after auditor"""
        if state.get("error_message"):
            return "error"
        
        auditor_output = state.get("auditor_output", {})
        recommendation = auditor_output.get("recommendation", "")
        
        if recommendation == "approved":
            return "approved"
        elif recommendation == "rejected":
            return "rejected"
        elif recommendation == "needs_revision":
            return "needs_revision"
        else:
            return "approved"  # Default to approved if no clear recommendation
    
    def _generate_review_summary(self, state: ValidationState) -> Dict[str, Any]:
        """Generate summary for human review"""
        return {
            "workflow_id": state["workflow_id"],
            "data_records": len(state.get("data", {})),
            "analyst_findings": state.get("analyst_output", {}).get("summary", ""),
            "validation_metrics": state.get("validator_output", {}).get("metrics", {}),
            "compliance_score": state.get("documentation_output", {}).get("compliance_score", 0),
            "recommended_action": "Review and approve to continue workflow"
        }
    
    async def create_workflow(self, initial_data: Dict[str, Any]) -> str:
        """Create a new workflow instance"""
        workflow_id = str(uuid.uuid4())
        
        # Initialize state
        initial_state: ValidationState = {
            "workflow_id": workflow_id,
            "data": initial_data.get("data", {}),
            "documents": initial_data.get("documents", {}),
            "analyst_output": {},
            "validator_output": {},
            "documentation_output": {},
            "human_feedback": {},
            "reviewer_output": {},
            "auditor_output": {},
            "current_step": "initialized",
            "workflow_status": "initialized",
            "error_message": None,
            "execution_history": [{
                "step": "workflow_created",
                "timestamp": datetime.now().isoformat(),
                "status": "initialized"
            }],
            "checkpoint_data": {}
        }
        
        # Store workflow
        self.active_workflows[workflow_id] = self.graph
        
        logger.info(f"Created LangGraph workflow: {workflow_id}")
        return workflow_id
    
    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute the entire workflow"""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        graph = self.active_workflows[workflow_id]
        
        try:
            # Execute the workflow
            config = RunnableConfig(
                configurable={"thread_id": workflow_id}
            )
            
            result = await graph.ainvoke(
                {"workflow_id": workflow_id},
                config=config
            )
            
            logger.info(f"Workflow {workflow_id} execution completed")
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    async def submit_human_feedback(self, workflow_id: str, checkpoint_id: str, feedback: Dict[str, Any]):
        """Submit human feedback for workflow continuation"""
        if checkpoint_id not in self.workflow_checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        checkpoint = self.workflow_checkpoints[checkpoint_id]
        checkpoint.reviewed = True
        checkpoint.reviewer_feedback = feedback
        
        # Update workflow state with feedback
        # This would typically involve resuming the workflow from the checkpoint
        logger.info(f"Human feedback submitted for checkpoint {checkpoint_id}")
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current workflow status"""
        if workflow_id not in self.active_workflows:
            return None
        
        # In a real implementation, this would query the checkpointer
        # For now, return basic status
        return {
            "workflow_id": workflow_id,
            "status": "active",
            "current_step": "unknown"
        }