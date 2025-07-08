import logging
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from fastapi import APIRouter
from fastapi import BackgroundTasks
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse

from agent_service.core.memory.redis_store import MemoryService
from agent_service.core.schema.workflow import HumanReviewResponse
from agent_service.core.schema.workflow import WorkflowRequest
from agent_service.core.schema.workflow import WorkflowResponse
from agent_service.core.utils.logging_config import AuditLogger
from agent_service.core.workflows.graph import BankingWorkflowGraph

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency to get memory service
def get_memory_service(request: Request) -> MemoryService:
    return request.app.state.memory

# Dependency to get workflow graph
def get_workflow_graph(memory_service: MemoryService = Depends(get_memory_service)) -> BankingWorkflowGraph:
    return BankingWorkflowGraph(memory_service)

# Initialize audit logger
audit_logger = AuditLogger("api")

@router.post("/workflows/start", response_model=WorkflowResponse)
async def start_workflow(
    request: WorkflowRequest,
    background_tasks: BackgroundTasks,
    workflow_graph: BankingWorkflowGraph = Depends(get_workflow_graph),
):
    """Start a new workflow"""

    try:
        # Validate workflow type
        valid_types = ["model_validation", "ecl_calculation", "rwa_calculation", "reporting"]
        if request.workflow_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid workflow type. Must be one of: {valid_types}"
            )

        # Start workflow
        workflow_id = await workflow_graph.start_workflow(
            workflow_type=request.workflow_type,
            inputs=request.inputs,
            config=request.llm_config
        )

        # Log workflow start
        audit_logger.log_workflow_start(
            workflow_id=workflow_id,
            workflow_type=request.workflow_type,
            inputs=request.inputs
        )

        logger.info(f"Workflow started: {workflow_id}")

        return WorkflowResponse(
            workflow_id=workflow_id,
            task_id=workflow_id,  # Using workflow_id as task_id for simplicity
            status="started",
            message=f"Workflow {request.workflow_type} started successfully",
            data={"workflow_type": request.workflow_type, "inputs": request.inputs}
        )

    except Exception as e:
        logger.error(f"Failed to start workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows/{workflow_id}/status", response_model=Dict[str, Any])
async def get_workflow_status(
    workflow_id: str,
    workflow_graph: BankingWorkflowGraph = Depends(get_workflow_graph)
):
    """Get workflow status"""

    try:
        status = await workflow_graph.get_workflow_status(workflow_id)

        if not status:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows/{workflow_id}/results", response_model=Dict[str, Any])
async def get_workflow_results(
    workflow_id: str,
    workflow_graph: BankingWorkflowGraph = Depends(get_workflow_graph)
):
    """Get workflow results"""

    try:
        results = await workflow_graph.get_workflow_results(workflow_id)

        if not results:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflows/{workflow_id}/human-feedback", response_model=Dict[str, Any])
async def submit_human_feedback(
    workflow_id: str,
    feedback: HumanReviewResponse,
    workflow_graph: BankingWorkflowGraph = Depends(get_workflow_graph)
):
    """Submit human feedback to continue workflow"""

    try:
        # Prepare feedback data
        feedback_data = {
            "user_id": feedback.user_id,
            "action": feedback.action,
            "comments": feedback.comments,
            "modifications": feedback.modifications
        }

        # Submit feedback
        success = await workflow_graph.submit_human_feedback(workflow_id, feedback_data)

        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found or feedback submission failed")

        # Log human intervention
        audit_logger.log_human_intervention(
            workflow_id=workflow_id,
            action=feedback.action,
            user_id=feedback.user_id,
            decision=feedback.comments or "No comments"
        )

        return {
            "success": True,
            "message": "Human feedback submitted successfully",
            "workflow_id": workflow_id,
            "feedback_action": feedback.action
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit human feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflows/{workflow_id}/cancel", response_model=Dict[str, Any])
async def cancel_workflow(
    workflow_id: str,
    workflow_graph: BankingWorkflowGraph = Depends(get_workflow_graph)
):
    """Cancel a running workflow"""

    try:
        success = await workflow_graph.cancel_workflow(workflow_id)

        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found or cancellation failed")

        # Log cancellation
        audit_logger.log_workflow_end(
            workflow_id=workflow_id,
            status="cancelled",
            outputs={"cancelled_by": "api_request"}
        )

        return {
            "success": True,
            "message": "Workflow cancelled successfully",
            "workflow_id": workflow_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows", response_model=List[Dict[str, Any]])
async def list_workflows(
    status: Optional[str] = None,
    workflow_type: Optional[str] = None,
    limit: int = 100,
    workflow_graph: BankingWorkflowGraph = Depends(get_workflow_graph)
):
    """List workflows with optional filtering"""

    try:
        # Get all active workflows
        workflows = await workflow_graph.list_active_workflows()

        # Apply filters
        if status:
            workflows = [w for w in workflows if w.get("status") == status]

        if workflow_type:
            workflows = [w for w in workflows if w.get("workflow_type") == workflow_type]

        # Apply limit
        workflows = workflows[:limit]

        return workflows

    except Exception as e:
        logger.error(f"Failed to list workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows/{workflow_id}/logs", response_model=List[Dict[str, Any]])
async def get_workflow_logs(
    workflow_id: str,
    limit: int = 100,
    workflow_graph: BankingWorkflowGraph = Depends(get_workflow_graph)
):
    """Get workflow execution logs"""

    try:
        # Get workflow status (includes recent messages)
        status = await workflow_graph.get_workflow_status(workflow_id)

        if not status:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Return agent messages as logs
        logs = status.get("agent_messages", [])

        # Limit results
        logs = logs[-limit:] if len(logs) > limit else logs

        return logs

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows/{workflow_id}/metrics", response_model=Dict[str, Any])
async def get_workflow_metrics(
    workflow_id: str,
    workflow_graph: BankingWorkflowGraph = Depends(get_workflow_graph)
):
    """Get workflow execution metrics"""

    try:
        # Get workflow results
        results = await workflow_graph.get_workflow_results(workflow_id)

        if not results:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Calculate metrics
        tool_results = results.get("tool_results", [])
        successful_tools = [r for r in tool_results if r.get("success")]
        failed_tools = [r for r in tool_results if not r.get("success")]

        # Calculate execution times
        execution_times = [r.get("execution_time", 0) for r in tool_results]
        total_execution_time = sum(execution_times)
        avg_execution_time = total_execution_time / len(execution_times) if execution_times else 0

        # Calculate workflow duration
        created_at = datetime.fromisoformat(results["created_at"])
        completed_at = datetime.fromisoformat(results["completed_at"]) if results.get("completed_at") else datetime.utcnow()
        workflow_duration = (completed_at - created_at).total_seconds()

        metrics = {
            "workflow_id": workflow_id,
            "workflow_type": results.get("workflow_type"),
            "status": results.get("status"),
            "workflow_duration": workflow_duration,
            "total_steps": len(results.get("plan_steps", [])),
            "tools_executed": len(tool_results),
            "successful_tools": len(successful_tools),
            "failed_tools": len(failed_tools),
            "tool_success_rate": (len(successful_tools) / len(tool_results) * 100) if tool_results else 0,
            "total_execution_time": total_execution_time,
            "average_execution_time": avg_execution_time,
            "agent_messages": len(results.get("agent_messages", [])),
            "created_at": results["created_at"],
            "completed_at": results.get("completed_at"),
            "tool_breakdown": {}
        }

        # Tool breakdown
        for tool_result in tool_results:
            tool_name = tool_result.get("tool_name")
            if tool_name not in metrics["tool_breakdown"]:
                metrics["tool_breakdown"][tool_name] = {
                    "count": 0,
                    "success_count": 0,
                    "total_time": 0
                }

            metrics["tool_breakdown"][tool_name]["count"] += 1
            if tool_result.get("success"):
                metrics["tool_breakdown"][tool_name]["success_count"] += 1
            metrics["tool_breakdown"][tool_name]["total_time"] += tool_result.get("execution_time", 0)

        return metrics

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflows/validate", response_model=Dict[str, Any])
async def validate_workflow_request(request: WorkflowRequest):
    """Validate a workflow request without starting it"""

    try:
        # Validate workflow type
        valid_types = ["model_validation", "ecl_calculation", "rwa_calculation", "reporting"]
        if request.workflow_type not in valid_types:
            return {
                "valid": False,
                "errors": [f"Invalid workflow type. Must be one of: {valid_types}"]
            }

        errors = []
        warnings = []

        # Type-specific validation
        if request.workflow_type == "model_validation":
            if "model_name" not in request.inputs:
                errors.append("Missing required input: model_name")
            if "model_config" not in request.inputs:
                errors.append("Missing required input: model_config")

        elif request.workflow_type == "ecl_calculation":
            if "portfolio_data" not in request.inputs:
                errors.append("Missing required input: portfolio_data")
            if "pd_curves" not in request.inputs:
                warnings.append("Missing PD curves - default values will be used")

        elif request.workflow_type == "rwa_calculation":
            if "exposure_data" not in request.inputs:
                errors.append("Missing required input: exposure_data")
            if "risk_weights" not in request.inputs:
                warnings.append("Missing risk weights - default values will be used")

        elif request.workflow_type == "reporting":
            if "report_type" not in request.inputs:
                errors.append("Missing required input: report_type")
            if "data_sources" not in request.inputs:
                errors.append("Missing required input: data_sources")

        # Check timeout
        if request.timeout and request.timeout > 7200:  # 2 hours max
            warnings.append("Timeout exceeds recommended maximum of 2 hours")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "workflow_type": request.workflow_type,
            "estimated_duration": _estimate_workflow_duration(request.workflow_type),
            "required_inputs": _get_required_inputs(request.workflow_type),
            "optional_inputs": _get_optional_inputs(request.workflow_type)
        }

    except Exception as e:
        logger.error(f"Failed to validate workflow request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _estimate_workflow_duration(workflow_type: str) -> str:
    """Estimate workflow duration based on type"""

    durations = {
        "model_validation": "5-15 minutes",
        "ecl_calculation": "10-30 minutes",
        "rwa_calculation": "15-45 minutes",
        "reporting": "3-10 minutes"
    }

    return durations.get(workflow_type, "5-20 minutes")

def _get_required_inputs(workflow_type: str) -> List[str]:
    """Get required inputs for workflow type"""

    required_inputs = {
        "model_validation": ["model_name", "model_config"],
        "ecl_calculation": ["portfolio_data"],
        "rwa_calculation": ["exposure_data"],
        "reporting": ["report_type", "data_sources"]
    }

    return required_inputs.get(workflow_type, [])

def _get_optional_inputs(workflow_type: str) -> List[str]:
    """Get optional inputs for workflow type"""

    optional_inputs = {
        "model_validation": ["validation_rules", "test_data"],
        "ecl_calculation": ["pd_curves", "lgd_estimates", "ead_estimates", "scenarios"],
        "rwa_calculation": ["risk_weights", "capital_data", "regulatory_adjustments"],
        "reporting": ["template", "report_config", "filters"]
    }

    return optional_inputs.get(workflow_type, [])

@router.get("/health", response_model=Dict[str, Any])
async def health_check(
    memory_service: MemoryService = Depends(get_memory_service)
):
    """Health check for the API"""

    try:
        # Check memory service
        memory_healthy = await memory_service.health_check()

        # Check LLM availability
        from agent_service.core.services.llm_client import llm_manager
        available_providers = llm_manager.get_available_providers()

        health_status = {
            "status": "healthy" if memory_healthy and available_providers else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "memory": "healthy" if memory_healthy else "unhealthy",
                "llm_providers": available_providers,
                "api": "healthy"
            },
            "version": "1.0.0"
        }

        if health_status["status"] == "unhealthy":
            return JSONResponse(
                status_code=503,
                content=health_status
            )

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@router.get("/info", response_model=Dict[str, Any])
async def get_api_info():
    """Get API information"""

    return {
        "name": "Agentic AI Banking Platform",
        "version": "1.0.0",
        "description": "Multi-agent workflow system for banking operations using LangGraph",
        "supported_workflows": [
            "model_validation",
            "ecl_calculation",
            "rwa_calculation",
            "reporting"
        ],
        "features": [
            "Multi-agent workflow orchestration",
            "Human-in-the-loop capabilities",
            "Multi-model GenAI support",
            "Comprehensive audit logging",
            "Real-time workflow monitoring"
        ],
        "endpoints": {
            "workflows": "/api/v1/workflows",
            "health": "/api/v1/health",
            "info": "/api/v1/info"
        }
    }
