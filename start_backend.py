#!/usr/bin/env python3
"""
Simple FastAPI backend starter for ValiCred-AI
This creates a basic API server that can work alongside the Streamlit app
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import pandas as pd
import json
import uvicorn
from datetime import datetime

# Import our agents
from agents.analyst_agent import AnalystAgent
from agents.validator_agent import ValidatorAgent
from agents.documentation_agent import DocumentationAgent
from agents.reviewer_agent import ReviewerAgent
from agents.auditor_agent import AuditorAgent

app = FastAPI(title="ValiCred-AI API", version="1.0.0")

# Enable CORS for Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
agents = {
    'analyst': AnalystAgent(),
    'validator': ValidatorAgent(),
    'documentation': DocumentationAgent(),
    'reviewer': ReviewerAgent(),
    'auditor': AuditorAgent()
}

class AgentRequest(BaseModel):
    agent_name: str
    context: Dict[str, Any]

@app.get("/")
async def root():
    return {
        "message": "ValiCred-AI Backend API",
        "status": "running",
        "agents": list(agents.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "agents_available": len(agents)}

@app.post("/execute_agent")
async def execute_agent(request: AgentRequest):
    """Execute a specific agent with given context"""
    
    if request.agent_name not in agents:
        raise HTTPException(status_code=400, detail=f"Agent {request.agent_name} not found")
    
    try:
        agent = agents[request.agent_name]
        result = agent.run(request.context)
        
        return {
            "agent": request.agent_name,
            "status": "success",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

@app.get("/sample_data")
async def get_sample_data():
    """Get sample credit data"""
    try:
        df = pd.read_csv('sample_data/credit_data.csv')
        return {
            "data": df.to_dict('records')[:10],  # Return first 10 rows
            "total_records": len(df),
            "columns": df.columns.tolist(),
            "shape": df.shape
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading sample data: {str(e)}")

if __name__ == "__main__":
    print("Starting ValiCred-AI Backend Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)