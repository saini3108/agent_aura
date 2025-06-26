from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
import pandas as pd
from datetime import datetime
import asyncio
import uuid

# Import agent classes
import sys
sys.path.append('..')
from agents.analyst_agent import AnalystAgent
from agents.validator_agent import ValidatorAgent
from agents.documentation_agent import DocumentationAgent
from agents.reviewer_agent import ReviewerAgent
from agents.auditor_agent import AuditorAgent

app = FastAPI(title="ValiCred-AI Backend", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state management (in production, use Redis or database)
workflow_states = {}
agent_instances = {
    'analyst': AnalystAgent(),
    'validator': ValidatorAgent(),
    'documentation': DocumentationAgent(),
    'reviewer': ReviewerAgent(),
    'auditor': AuditorAgent()
}

# Pydantic models
class WorkflowRequest(BaseModel):
    workflow_id: str
    agent_name: str
    context: Dict[str, Any]

class WorkflowState(BaseModel):
    workflow_id: str
    current_step: int
    completed_steps: List[int]
    agent_outputs: Dict[str, Any]
    human_feedback: Dict[str, Any]
    status: str
    created_at: str
    updated_at: str

class AgentExecutionRequest(BaseModel):
    agent_name: str
    context: Dict[str, Any]
    workflow_id: Optional[str] = None

class HumanFeedbackRequest(BaseModel):
    workflow_id: str
    step_index: int
    feedback: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Initialize the FastAPI server"""
    print("ValiCred-AI Backend Server Starting...")
    
    # Load MCP agent configuration
    try:
        with open('../config/mcp_agents.json', 'r') as f:
            mcp_config = json.load(f)
        print("MCP Agent Configuration Loaded")
    except FileNotFoundError:
        print("Warning: MCP configuration file not found")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ValiCred-AI Backend API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agents": list(agent_instances.keys()),
        "active_workflows": len(workflow_states),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/workflow/create")
async def create_workflow():
    """Create a new workflow"""
    workflow_id = str(uuid.uuid4())
    
    workflow_state = WorkflowState(
        workflow_id=workflow_id,
        current_step=0,
        completed_steps=[],
        agent_outputs={},
        human_feedback={},
        status="created",
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )
    
    workflow_states[workflow_id] = workflow_state.dict()
    
    return {
        "workflow_id": workflow_id,
        "status": "created",
        "message": "Workflow created successfully"
    }

@app.get("/workflow/{workflow_id}")
async def get_workflow(workflow_id: str):
    """Get workflow state"""
    if workflow_id not in workflow_states:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return workflow_states[workflow_id]

@app.post("/workflow/{workflow_id}/execute_step")
async def execute_workflow_step(workflow_id: str, request: WorkflowRequest):
    """Execute a specific workflow step"""
    if workflow_id not in workflow_states:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow_state = workflow_states[workflow_id]
    agent_name = request.agent_name
    
    if agent_name not in agent_instances:
        raise HTTPException(status_code=400, detail=f"Agent {agent_name} not found")
    
    try:
        # Execute agent
        agent = agent_instances[agent_name]
        result = agent.run(request.context)
        
        # Update workflow state
        step_key = f"step_{workflow_state['current_step']}"
        workflow_state['agent_outputs'][step_key] = result
        workflow_state['completed_steps'].append(workflow_state['current_step'])
        workflow_state['current_step'] += 1
        workflow_state['updated_at'] = datetime.now().isoformat()
        workflow_state['status'] = 'in_progress'
        
        workflow_states[workflow_id] = workflow_state
        
        return {
            "workflow_id": workflow_id,
            "agent_name": agent_name,
            "result": result,
            "status": "completed"
        }
        
    except Exception as e:
        workflow_state['status'] = 'error'
        workflow_state['updated_at'] = datetime.now().isoformat()
        workflow_states[workflow_id] = workflow_state
        
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

@app.post("/agent/execute")
async def execute_agent(request: AgentExecutionRequest):
    """Execute an agent independently"""
    agent_name = request.agent_name
    
    if agent_name not in agent_instances:
        raise HTTPException(status_code=400, detail=f"Agent {agent_name} not found")
    
    try:
        agent = agent_instances[agent_name]
        result = agent.run(request.context)
        
        return {
            "agent_name": agent_name,
            "result": result,
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

@app.post("/workflow/{workflow_id}/human_feedback")
async def submit_human_feedback(workflow_id: str, request: HumanFeedbackRequest):
    """Submit human feedback for a workflow step"""
    if workflow_id not in workflow_states:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow_state = workflow_states[workflow_id]
    
    # Store human feedback
    step_key = f"step_{request.step_index}"
    workflow_state['human_feedback'][step_key] = {
        **request.feedback,
        "timestamp": datetime.now().isoformat(),
        "step_index": request.step_index
    }
    
    workflow_state['updated_at'] = datetime.now().isoformat()
    workflow_states[workflow_id] = workflow_state
    
    return {
        "workflow_id": workflow_id,
        "step_index": request.step_index,
        "status": "feedback_received",
        "message": "Human feedback submitted successfully"
    }

@app.get("/agents")
async def list_agents():
    """List available agents"""
    agents_info = {}
    
    for name, agent in agent_instances.items():
        agents_info[name] = {
            "name": agent.name,
            "description": agent.description
        }
    
    return {
        "agents": agents_info,
        "total_count": len(agent_instances)
    }

@app.get("/data/sample")
async def get_sample_data():
    """Get sample credit data"""
    try:
        df = pd.read_csv('../sample_data/credit_data.csv')
        return {
            "data": df.to_dict('records'),
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "info": {
                "total_records": len(df),
                "features": len(df.columns),
                "missing_values": df.isnull().sum().sum(),
                "default_rate": df['default_flag'].mean() if 'default_flag' in df.columns else None
            }
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Sample data file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading sample data: {str(e)}")

@app.get("/parameters")
async def get_validation_parameters():
    """Get validation parameters"""
    try:
        df = pd.read_csv('../sample_data/validation_parameters.csv')
        parameters = {}
        
        for _, row in df.iterrows():
            param_name = row['parameter_name']
            param_value = row['parameter_value']
            param_type = row['parameter_type']
            
            # Convert to appropriate type
            if param_type == 'float':
                param_value = float(param_value)
            elif param_type == 'int':
                param_value = int(param_value)
            
            parameters[param_name] = {
                "value": param_value,
                "type": param_type,
                "description": row['description'],
                "threshold_type": row['threshold_type']
            }
        
        return {"parameters": parameters}
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Validation parameters file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading parameters: {str(e)}")

@app.get("/risk_thresholds")
async def get_risk_thresholds():
    """Get risk assessment thresholds"""
    try:
        df = pd.read_csv('../sample_data/risk_thresholds.csv')
        thresholds = {}
        
        for _, row in df.iterrows():
            category = row['risk_category']
            if category not in thresholds:
                thresholds[category] = {}
            
            thresholds[category][row['metric_name']] = {
                "excellent": row['excellent_threshold'],
                "good": row['good_threshold'],
                "acceptable": row['acceptable_threshold'],
                "poor": row['poor_threshold'],
                "weight": row['weight']
            }
        
        return {"thresholds": thresholds}
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Risk thresholds file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading thresholds: {str(e)}")

@app.delete("/workflow/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """Delete a workflow"""
    if workflow_id not in workflow_states:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    del workflow_states[workflow_id]
    
    return {
        "workflow_id": workflow_id,
        "status": "deleted",
        "message": "Workflow deleted successfully"
    }

@app.get("/workflows")
async def list_workflows():
    """List all workflows"""
    workflows = []
    
    for workflow_id, state in workflow_states.items():
        workflows.append({
            "workflow_id": workflow_id,
            "status": state['status'],
            "current_step": state['current_step'],
            "completed_steps": len(state['completed_steps']),
            "created_at": state['created_at'],
            "updated_at": state['updated_at']
        })
    
    return {
        "workflows": workflows,
        "total_count": len(workflows)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)