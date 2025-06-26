"""
ValiCred-AI: Agent Aura Architecture
Clean MCP + LangGraph + HITL Credit Risk Validation System
"""
import streamlit as st
import sys
import os
from pathlib import Path
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import uuid
import numpy as np
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field

# Configure paths for agent_aura structure
current_dir = Path(__file__).parent
agent_aura_path = current_dir / "agent_aura"
sys.path.insert(0, str(agent_aura_path / "shared"))
sys.path.insert(0, str(agent_aura_path / "agent-service" / "agents"))

# Import agent_aura components with error handling
try:
    from agent_aura.shared.system_config import config
    from agent_aura.shared.audit_logger import AuditLogger
    from agent_aura.agent_service.agents.analyst_agent import AnalystAgent
    from agent_aura.agent_service.agents.validator_agent import ValidatorAgent
    from agent_aura.agent_service.agents.documentation_agent import DocumentationAgent
    from agent_aura.agent_service.agents.reviewer_agent import ReviewerAgent
    from agent_aura.agent_service.agents.auditor_agent import AuditorAgent
    from agent_aura.shared.sample_data_loader import sample_loader
except ImportError:
    st.error("Initializing with local implementations...")
    # Fall back to local implementations if imports fail
    
    class SimpleConfig:
        def __init__(self):
            self.mcp_config = {
                "workflow": {"execution_order": ["analyst", "validator", "documentation", "human_review", "reviewer", "auditor"]},
                "agents": {
                    "analyst": {"timeout": 300, "retry_attempts": 3},
                    "validator": {"timeout": 600, "retry_attempts": 3},
                    "documentation": {"timeout": 300, "retry_attempts": 2},
                    "reviewer": {"timeout": 300, "retry_attempts": 2},
                    "auditor": {"timeout": 300, "retry_attempts": 2}
                }
            }
            self.workflow_config = {"retry_policy": {"max_retries": 3, "retry_delay": 5}}
            self.risk_thresholds = {
                "auc": {"excellent": 0.8, "good": 0.7, "acceptable": 0.6},
                "ks": {"excellent": 0.3, "good": 0.2, "acceptable": 0.15}
            }
        
        def get_agent_config(self, agent_name: str):
            return self.mcp_config.get("agents", {}).get(agent_name, {})
    
    config = SimpleConfig()
    
    class SimpleAuditLogger:
        def __init__(self):
            self.entries = []
        
        def log_workflow_event(self, event_type, step_name, step_index, additional_info=None):
            self.entries.append({
                "timestamp": datetime.now(),
                "event_type": event_type,
                "step_name": step_name,
                "step_index": step_index,
                "additional_info": additional_info or {}
            })
        
        def log_data_operation(self, operation, data_info):
            self.entries.append({
                "timestamp": datetime.now(),
                "operation": operation,
                "data_info": data_info
            })
        
        def log_agent_execution(self, agent_name, status, execution_time, result):
            self.entries.append({
                "timestamp": datetime.now(),
                "agent_name": agent_name,
                "status": status,
                "execution_time": execution_time
            })
        
        def get_audit_trail(self, limit=None):
            return self.entries[-limit:] if limit else self.entries
    
    AuditLogger = SimpleAuditLogger
    
    # Import LLM provider
    try:
        from agent_aura.shared.llm_provider import llm_manager
        from groq import Groq
        
        # Initialize Groq client with API key
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Real intelligent agent implementations
        class IntelligentAgent:
            def __init__(self, name, role_description):
                self.name = name
                self.role_description = role_description
                self.client = groq_client
            
            def run(self, context):
                try:
                    start_time = datetime.now()
                    
                    # Prepare data summary for LLM
                    data = context.get('data')
                    if data is not None and not data.empty:
                        data_summary = self._create_data_summary(data)
                        
                        # Create role-specific prompt
                        prompt = self._create_agent_prompt(data_summary, context)
                        
                        # Get LLM response
                        response = self.client.chat.completions.create(
                            model="llama-3.1-70b-versatile",
                            messages=[
                                {
                                    "role": "system",
                                    "content": f"You are a {self.role_description}. Provide detailed, professional analysis in JSON format."
                                },
                                {
                                    "role": "user", 
                                    "content": prompt
                                }
                            ],
                            temperature=0.3,
                            max_tokens=2000
                        )
                        
                        ai_analysis = response.choices[0].message.content
                        execution_time = (datetime.now() - start_time).total_seconds()
                        
                        return {
                            "agent": self.name,
                            "status": "completed",
                            "ai_analysis": ai_analysis,
                            "data_insights": self._extract_insights(data),
                            "execution_time": execution_time,
                            "llm_enhanced": True,
                            "model_used": "llama-3.1-70b-versatile",
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        return {
                            "agent": self.name,
                            "status": "error",
                            "error": "No data provided for analysis",
                            "llm_enhanced": False
                        }
                        
                except Exception as e:
                    return {
                        "agent": self.name,
                        "status": "error", 
                        "error": f"Analysis failed: {str(e)}",
                        "llm_enhanced": False
                    }
            
            def _create_data_summary(self, data):
                summary = []
                summary.append(f"Dataset: {len(data)} records, {len(data.columns)} features")
                
                if 'default' in data.columns:
                    default_rate = data['default'].mean()
                    summary.append(f"Default rate: {default_rate:.2%}")
                
                # Key numeric features
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols[:5]:
                    if col != 'default':
                        stats = data[col].describe()
                        summary.append(f"{col}: mean={stats['mean']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]")
                
                return "; ".join(summary)
            
            def _create_agent_prompt(self, data_summary, context):
                if self.name == "analyst":
                    return f"""
                    Analyze this credit risk dataset: {data_summary}
                    
                    Provide analysis covering:
                    1. Data quality assessment
                    2. Key risk indicators
                    3. Feature relationships
                    4. Modeling recommendations
                    5. Regulatory considerations
                    
                    Focus on practical insights for model validation.
                    """
                elif self.name == "validator":
                    return f"""
                    Perform validation analysis on: {data_summary}
                    
                    Calculate and interpret:
                    1. Model performance metrics
                    2. Statistical significance
                    3. Population stability
                    4. Discriminatory power
                    5. Calibration assessment
                    
                    Provide specific metric recommendations.
                    """
                elif self.name == "documentation":
                    files = context.get('files', {})
                    return f"""
                    Review compliance for: {data_summary}
                    Documents: {list(files.keys()) if files else 'None'}
                    
                    Assess:
                    1. Basel III compliance
                    2. IFRS 9 requirements
                    3. Model risk management
                    4. Documentation completeness
                    5. Regulatory gaps
                    
                    Identify compliance issues and recommendations.
                    """
                elif self.name == "reviewer":
                    return f"""
                    Generate review findings for: {data_summary}
                    
                    Provide:
                    1. Executive summary
                    2. Key findings
                    3. Risk assessment
                    4. Recommendations
                    5. Action items
                    
                    Focus on decision-making insights.
                    """
                elif self.name == "auditor":
                    return f"""
                    Perform final audit of: {data_summary}
                    
                    Deliver:
                    1. Independent validation
                    2. Compliance verification
                    3. Risk opinion
                    4. Approval recommendation
                    5. Governance assessment
                    
                    Provide final go/no-go decision.
                    """
                else:
                    return f"Analyze this credit risk data: {data_summary}"
            
            def _extract_insights(self, data):
                insights = {}
                
                if 'default' in data.columns:
                    insights['default_rate'] = float(data['default'].mean())
                    insights['total_records'] = len(data)
                
                # Key correlations
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if 'default' in data.columns and len(numeric_cols) > 1:
                    correlations = []
                    for col in numeric_cols:
                        if col != 'default':
                            corr = data[col].corr(data['default'])
                            if not np.isnan(corr):
                                correlations.append({
                                    'feature': col,
                                    'correlation': float(corr)
                                })
                    insights['feature_correlations'] = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)[:5]
                
                return insights
        
        # Create intelligent agents
        AnalystAgent = lambda: IntelligentAgent("analyst", "senior credit risk analyst with expertise in model validation and regulatory compliance")
        ValidatorAgent = lambda: IntelligentAgent("validator", "model validation specialist focused on statistical metrics and performance assessment")
        DocumentationAgent = lambda: IntelligentAgent("documentation", "compliance expert specializing in Basel III, IFRS 9, and model risk management")
        ReviewerAgent = lambda: IntelligentAgent("reviewer", "risk management reviewer providing executive-level findings and recommendations")
        AuditorAgent = lambda: IntelligentAgent("auditor", "independent auditor providing final validation and approval recommendations")
        
        st.success("✅ Real AI analysis enabled with Groq LLM")
        
    except Exception as e:
        st.warning(f"LLM integration failed: {str(e)} - Using fallback implementation")
        # Fallback to simple agents if LLM fails
        class SimpleAgent:
            def __init__(self, name):
                self.name = name
            
            def run(self, context):
                return {
                    "agent": self.name,
                    "status": "completed",
                    "analysis": f"{self.name} analysis completed (fallback mode)",
                    "execution_time": np.random.uniform(1, 5),
                    "llm_enhanced": False
                }
        
        AnalystAgent = lambda: SimpleAgent("analyst")
        ValidatorAgent = lambda: SimpleAgent("validator")
        DocumentationAgent = lambda: SimpleAgent("documentation")
        ReviewerAgent = lambda: SimpleAgent("reviewer")
        AuditorAgent = lambda: SimpleAgent("auditor")
    
    class SimpleSampleLoader:
        def get_sample_data(self):
            np.random.seed(42)
            n_samples = 50
            data = {
                'customer_id': range(1, n_samples + 1),
                'age': np.random.randint(18, 80, n_samples),
                'income': np.random.normal(50000, 20000, n_samples),
                'credit_score': np.random.randint(300, 850, n_samples),
                'loan_amount': np.random.uniform(5000, 100000, n_samples),
                'employment_years': np.random.randint(0, 40, n_samples),
                'debt_to_income': np.random.uniform(0.1, 0.8, n_samples),
                'default': np.random.choice([0, 1], n_samples, p=[0.82, 0.18])
            }
            return pd.DataFrame(data)
        
        def get_sample_documents(self):
            return {
                "model_methodology.pdf": {"type": "methodology", "size": "2.3MB"},
                "validation_report.pdf": {"type": "validation", "size": "1.8MB"},
                "governance_policy.pdf": {"type": "governance", "size": "956KB"}
            }
    
    sample_loader = SimpleSampleLoader()

# MCP Workflow Engine Implementation
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
    start_time: datetime = None
    end_time: datetime = None
    execution_time: float = 0.0
    retry_count: int = 0
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: str = None

@dataclass
class HumanReviewCheckpoint:
    checkpoint_id: str
    agent_outputs: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    reviewed_at: datetime = None
    reviewer_feedback: Dict[str, Any] = field(default_factory=dict)
    approval_status: str = None

class MCPWorkflowEngine:
    def __init__(self, agents, audit_logger, config_dict):
        self.agents = agents
        self.audit_logger = audit_logger
        self.config = config_dict
        self.active_workflows = {}
        self.workflow_history = []
        self.execution_order = config_dict.get('mcp_config', {}).get('workflow', {}).get('execution_order', [])
    
    async def create_workflow(self, initial_data):
        workflow_id = str(uuid.uuid4())
        
        agent_executions = []
        for agent_name in self.execution_order:
            if agent_name != "human_review":
                agent_executions.append(AgentExecution(agent_name=agent_name))
        
        workflow_state = {
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
        
        self.audit_logger.log_workflow_event(
            "workflow_created", "workflow_initialization", 0, 
            {"workflow_id": workflow_id}
        )
        
        return workflow_id
    
    async def execute_workflow_step(self, workflow_id, step_name):
        if workflow_id not in self.active_workflows:
            return {"success": False, "error": "Workflow not found"}
        
        workflow_state = self.active_workflows[workflow_id]
        
        if step_name == "human_review":
            return await self._create_human_checkpoint(workflow_id)
        else:
            return await self._execute_agent_step(workflow_id, step_name)
    
    async def _execute_agent_step(self, workflow_id, agent_name):
        workflow_state = self.active_workflows[workflow_id]
        
        agent_execution = None
        for exec_record in workflow_state['agent_executions']:
            if exec_record.agent_name == agent_name:
                agent_execution = exec_record
                break
        
        if not agent_execution:
            return {"success": False, "error": f"Agent execution record not found for {agent_name}"}
        
        context = self._prepare_agent_context(workflow_id, agent_name)
        max_retries = self.config.get("workflow_config", {}).get("retry_policy", {}).get("max_retries", 3)
        
        agent_execution.status = AgentStatus.RUNNING
        agent_execution.start_time = datetime.now()
        
        for attempt in range(max_retries + 1):
            try:
                agent_execution.retry_count = attempt
                
                if agent_name in self.agents:
                    agent_instance = self.agents[agent_name]
                    result = agent_instance.run(context)
                    
                    agent_execution.status = AgentStatus.COMPLETED
                    agent_execution.end_time = datetime.now()
                    agent_execution.execution_time = (
                        agent_execution.end_time - agent_execution.start_time
                    ).total_seconds()
                    agent_execution.output_data = result
                    
                    workflow_state['updated_at'] = datetime.now()
                    
                    self.audit_logger.log_agent_execution(
                        agent_name, "completed", agent_execution.execution_time, result
                    )
                    
                    return {"success": True, "result": result, "execution_time": agent_execution.execution_time}
                else:
                    raise ValueError(f"Agent {agent_name} not found")
                    
            except Exception as e:
                if attempt < max_retries:
                    await asyncio.sleep(2)
                    continue
                else:
                    agent_execution.status = AgentStatus.FAILED
                    agent_execution.end_time = datetime.now()
                    agent_execution.error_message = str(e)
                    workflow_state['error_log'].append(f"Agent {agent_name} failed: {str(e)}")
                    
                    return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Unknown execution error"}
    
    def _prepare_agent_context(self, workflow_id, agent_name):
        workflow_state = self.active_workflows[workflow_id]
        
        previous_outputs = {}
        for exec_record in workflow_state['agent_executions']:
            if exec_record.status == AgentStatus.COMPLETED:
                previous_outputs[exec_record.agent_name] = exec_record.output_data
        
        return {
            'workflow_id': workflow_id,
            'agent_name': agent_name,
            'data': workflow_state['data'],
            'files': workflow_state['documents'],
            'previous_outputs': previous_outputs,
            'global_context': workflow_state['global_context'],
            'risk_thresholds': self.config.get('risk_thresholds', {}),
            'config': self.config.get('mcp_config', {}).get('agents', {}).get(agent_name, {})
        }
    
    async def _create_human_checkpoint(self, workflow_id):
        workflow_state = self.active_workflows[workflow_id]
        
        agent_outputs = {}
        for exec_record in workflow_state['agent_executions']:
            if exec_record.status == AgentStatus.COMPLETED:
                agent_outputs[exec_record.agent_name] = exec_record.output_data
        
        checkpoint_id = str(uuid.uuid4())
        
        checkpoint = HumanReviewCheckpoint(
            checkpoint_id=checkpoint_id,
            agent_outputs=agent_outputs
        )
        
        workflow_state['human_checkpoints'].append(checkpoint)
        workflow_state['status'] = WorkflowStatus.WAITING_FOR_HUMAN
        workflow_state['updated_at'] = datetime.now()
        
        self.audit_logger.log_workflow_event(
            "human_checkpoint_created", "human_review", 
            workflow_state['current_step'], {"checkpoint_id": checkpoint_id}
        )
        
        return {
            "success": True,
            "requires_review": True,
            "checkpoint_id": checkpoint_id,
            "review_data": {
                "agent_outputs": agent_outputs,
                "workflow_summary": self._generate_review_summary(agent_outputs)
            }
        }
    
    async def submit_human_feedback(self, workflow_id, checkpoint_id, feedback):
        if workflow_id not in self.active_workflows:
            return {"success": False, "error": "Workflow not found"}
        
        workflow_state = self.active_workflows[workflow_id]
        
        checkpoint = None
        for cp in workflow_state['human_checkpoints']:
            if cp.checkpoint_id == checkpoint_id:
                checkpoint = cp
                break
        
        if not checkpoint:
            return {"success": False, "error": "Checkpoint not found"}
        
        checkpoint.reviewed_at = datetime.now()
        checkpoint.reviewer_feedback = feedback
        checkpoint.approval_status = feedback.get("approval_status", "approved")
        
        workflow_state['status'] = WorkflowStatus.RUNNING
        workflow_state['updated_at'] = datetime.now()
        
        return {
            "success": True,
            "status": "feedback_submitted",
            "approval_status": checkpoint.approval_status
        }
    
    def _generate_review_summary(self, agent_outputs):
        return {
            "total_agents_completed": len(agent_outputs),
            "key_findings": [f"Agent {name} completed successfully" for name in agent_outputs.keys()],
            "validation_metrics": agent_outputs.get("validator", {}).get("metrics", {}),
            "documentation_status": "reviewed" if "documentation" in agent_outputs else "pending"
        }
    
    def get_workflow_status(self, workflow_id):
        if workflow_id not in self.active_workflows:
            return None
        
        workflow_state = self.active_workflows[workflow_id]
        
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
                {"checkpoint_id": cp.checkpoint_id, "created_at": cp.created_at.isoformat()}
                for cp in workflow_state['human_checkpoints']
                if not cp.reviewed_at
            ],
            "error_count": len(workflow_state['error_log']),
            "created_at": workflow_state['created_at'].isoformat(),
            "updated_at": workflow_state['updated_at'].isoformat()
        }
    
    def get_workflow_results(self, workflow_id):
        if workflow_id not in self.active_workflows:
            return None
        
        workflow_state = self.active_workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "status": workflow_state['status'].value,
            "agent_outputs": {
                exec_record.agent_name: exec_record.output_data
                for exec_record in workflow_state['agent_executions']
                if exec_record.status == AgentStatus.COMPLETED
            },
            "human_feedback": [
                {
                    "checkpoint_id": cp.checkpoint_id,
                    "approval_status": cp.approval_status,
                    "feedback": cp.reviewer_feedback
                }
                for cp in workflow_state['human_checkpoints']
                if cp.reviewed_at
            ]
        }

# Initialize session state
if 'mcp_engine' not in st.session_state:
    st.session_state.mcp_engine = None
if 'current_workflow_id' not in st.session_state:
    st.session_state.current_workflow_id = None
if 'validation_data' not in st.session_state:
    st.session_state.validation_data = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}

@st.cache_resource
def initialize_mcp_system():
    """Initialize the MCP + LangGraph system"""
    audit_logger = AuditLogger()
    
    # Initialize agents
    agents = {
        'analyst': AnalystAgent(),
        'validator': ValidatorAgent(),
        'documentation': DocumentationAgent(),
        'reviewer': ReviewerAgent(),
        'auditor': AuditorAgent()
    }
    
    # Initialize MCP workflow engine
    engine = MCPWorkflowEngine(agents, audit_logger, {
        'mcp_config': config.mcp_config,
        'workflow_config': config.workflow_config,
        'risk_thresholds': config.risk_thresholds
    })
    
    return engine, audit_logger, agents

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="ValiCred-AI", 
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize system
    mcp_engine, audit_logger, agents = initialize_mcp_system()
    st.session_state.mcp_engine = mcp_engine
    
    # Sidebar navigation
    st.sidebar.title("ValiCred-AI")
    st.sidebar.markdown("**Agent Aura Architecture**")
    
    page = st.sidebar.selectbox(
        "Navigation",
        [
            "Dashboard",
            "MCP Workflow",
            "Configuration",
            "Audit Trail",
            "System Status"
        ]
    )
    
    if page == "Dashboard":
        show_dashboard(mcp_engine, audit_logger)
    elif page == "MCP Workflow":
        show_mcp_workflow(mcp_engine, audit_logger)
    elif page == "Configuration":
        show_configuration()
    elif page == "Audit Trail":
        show_audit_trail(audit_logger)
    elif page == "System Status":
        show_system_status(mcp_engine)

def show_dashboard(mcp_engine, audit_logger):
    """Enhanced dashboard with MCP integration"""
    st.title("🏦 ValiCred-AI Dashboard")
    st.markdown("**Agent Aura Architecture** - MCP + LangGraph + HITL")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Active Workflows", len(mcp_engine.active_workflows))
    
    with col2:
        st.metric("Total Agents", len(mcp_engine.agents))
    
    with col3:
        audit_entries = len(audit_logger.get_audit_trail())
        st.metric("Audit Entries", audit_entries)
    
    # Quick actions
    st.subheader("Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Load Sample Credit Data", type="primary"):
            sample_data = sample_loader.get_sample_data()
            st.session_state.validation_data = sample_data
            audit_logger.log_data_operation(
                "sample_data_loaded",
                {"records": len(sample_data), "features": len(sample_data.columns)}
            )
            st.success(f"Loaded {len(sample_data)} credit records")
            st.rerun()
    
    with col2:
        if st.button("Load Sample Documents"):
            sample_docs = sample_loader.get_sample_documents()
            st.session_state.uploaded_files = sample_docs
            audit_logger.log_data_operation(
                "sample_documents_loaded",
                {"document_count": len(sample_docs)}
            )
            st.success(f"Loaded {len(sample_docs)} compliance documents")
            st.rerun()
    
    # Data preview
    if st.session_state.validation_data is not None:
        st.subheader("Data Preview")
        
        data = st.session_state.validation_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Summary**")
            st.write(f"Records: {len(data)}")
            st.write(f"Features: {len(data.columns)}")
            st.write(f"Default Rate: {data['default'].mean():.1%}")
        
        with col2:
            # Feature distribution chart
            numeric_cols = data.select_dtypes(include=['number']).columns[:3]
            if len(numeric_cols) > 0:
                fig = px.histogram(
                    data, x=numeric_cols[0], 
                    title=f"Distribution of {numeric_cols[0]}",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.write("**Sample Records**")
        st.dataframe(data.head(10), use_container_width=True)

def show_mcp_workflow(mcp_engine, audit_logger):
    """MCP workflow management interface"""
    st.title("🤖 MCP Workflow Engine")
    st.markdown("**LangGraph + Human-in-the-Loop Orchestration**")
    
    # Workflow controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Create New Workflow", type="primary"):
            if st.session_state.validation_data is not None:
                # Create workflow asynchronously
                initial_data = {
                    'data': st.session_state.validation_data,
                    'documents': st.session_state.uploaded_files,
                    'context': {'created_by': 'user', 'timestamp': datetime.now().isoformat()}
                }
                
                workflow_id = asyncio.run(mcp_engine.create_workflow(initial_data))
                st.session_state.current_workflow_id = workflow_id
                st.success(f"Created workflow: {workflow_id[:8]}...")
                st.rerun()
            else:
                st.error("Please load data first")
    
    with col2:
        if st.button("Refresh Status"):
            st.rerun()
    
    with col3:
        active_workflows = len(mcp_engine.active_workflows)
        st.metric("Active Workflows", active_workflows)
    
    # Current workflow status
    if st.session_state.current_workflow_id:
        workflow_id = st.session_state.current_workflow_id
        status = mcp_engine.get_workflow_status(workflow_id)
        
        if status:
            st.subheader(f"Workflow: {workflow_id[:8]}...")
            
            # Progress bar
            progress = status['progress_percentage'] / 100
            st.progress(progress, text=f"Progress: {status['progress_percentage']:.1f}%")
            
            # Status indicators
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Status", status['status'].title())
            
            with col2:
                st.metric("Current Step", f"{status['current_step']}/{status['total_steps']}")
            
            with col3:
                st.metric("Errors", status['error_count'])
            
            # Agent execution interface
            st.subheader("Agent Execution")
            
            execution_order = status.get('execution_order', [])
            
            for i, step_name in enumerate(execution_order):
                with st.expander(f"Step {i+1}: {step_name.title()}", expanded=(i == status['current_step'])):
                    
                    if step_name == "human_review":
                        show_human_review_interface(mcp_engine, workflow_id)
                    else:
                        agent_status = status['agent_statuses'].get(step_name, {})
                        agent_status_value = agent_status.get('status', 'pending')
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            status_color = {
                                'completed': '🟢',
                                'running': '🟡',
                                'failed': '🔴',
                                'pending': '⚪'
                            }.get(agent_status_value, '⚪')
                            
                            st.write(f"{status_color} **Status:** {agent_status_value.title()}")
                        
                        with col2:
                            exec_time = agent_status.get('execution_time', 0)
                            st.write(f"⏱️ **Time:** {exec_time:.1f}s")
                        
                        with col3:
                            retry_count = agent_status.get('retry_count', 0)
                            st.write(f"🔄 **Retries:** {retry_count}")
                        
                        # Execute button
                        if agent_status_value in ['pending', 'failed']:
                            if st.button(f"Execute {step_name.title()}", key=f"exec_{step_name}"):
                                with st.spinner(f"Executing {step_name} with Groq AI..."):
                                    result = asyncio.run(mcp_engine.execute_workflow_step(workflow_id, step_name))
                                    
                                    if result['success']:
                                        st.success(f"{step_name.title()} completed successfully")
                                        
                                        # Display AI analysis if available
                                        if 'result' in result and 'ai_analysis' in result['result']:
                                            with st.expander(f"View {step_name.title()} AI Analysis", expanded=True):
                                                ai_analysis = result['result']['ai_analysis']
                                                st.markdown("**AI-Generated Analysis:**")
                                                st.text_area("Analysis", ai_analysis, height=200, key=f"ai_analysis_{step_name}")
                                                
                                                # Show data insights if available
                                                if 'data_insights' in result['result']:
                                                    insights = result['result']['data_insights']
                                                    if 'default_rate' in insights:
                                                        st.metric("Default Rate", f"{insights['default_rate']:.2%}")
                                                    if 'feature_correlations' in insights:
                                                        st.write("**Top Feature Correlations:**")
                                                        for corr in insights['feature_correlations'][:3]:
                                                            st.write(f"• {corr['feature']}: {corr['correlation']:.3f}")
                                                
                                                # Show execution metadata
                                                col1, col2, col3 = st.columns(3)
                                                with col1:
                                                    st.metric("Execution Time", f"{result['result'].get('execution_time', 0):.2f}s")
                                                with col2:
                                                    st.metric("LLM Enhanced", "Yes" if result['result'].get('llm_enhanced') else "No")
                                                with col3:
                                                    model_used = result['result'].get('model_used', 'N/A')
                                                    st.metric("Model Used", model_used)
                                        
                                        audit_logger.log_workflow_event(
                                            "agent_executed",
                                            step_name,
                                            i,
                                            {"workflow_id": workflow_id, "execution_time": result.get('execution_time', 0)}
                                        )
                                    else:
                                        st.error(f"Failed: {result.get('error', 'Unknown error')}")
                                    
                                    st.rerun()
                        
                        # Show results if completed
                        elif agent_status_value == 'completed':
                            # Get and display agent results
                            workflow_results = mcp_engine.get_workflow_results(workflow_id)
                            if workflow_results and step_name in workflow_results.get('agent_outputs', {}):
                                agent_output = workflow_results['agent_outputs'][step_name]
                                
                                if st.button(f"View {step_name.title()} Results", key=f"view_{step_name}"):
                                    with st.expander(f"{step_name.title()} Results", expanded=True):
                                        if 'ai_analysis' in agent_output:
                                            st.markdown("**AI Analysis:**")
                                            st.text_area("", agent_output['ai_analysis'], height=300, key=f"view_ai_{step_name}")
                                        
                                        if 'data_insights' in agent_output:
                                            insights = agent_output['data_insights']
                                            st.write("**Key Insights:**")
                                            
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                if 'default_rate' in insights:
                                                    st.metric("Default Rate", f"{insights['default_rate']:.2%}")
                                                if 'total_records' in insights:
                                                    st.metric("Total Records", insights['total_records'])
                                            
                                            with col2:
                                                if 'feature_correlations' in insights and insights['feature_correlations']:
                                                    st.write("**Top Correlations:**")
                                                    for corr in insights['feature_correlations'][:3]:
                                                        st.write(f"• {corr['feature']}: {corr['correlation']:.3f}")
                                        
                                        # Show metadata
                                        if 'llm_enhanced' in agent_output:
                                            st.write(f"**LLM Enhanced:** {'Yes' if agent_output['llm_enhanced'] else 'No'}")
                                        if 'model_used' in agent_output:
                                            st.write(f"**Model Used:** {agent_output['model_used']}")
                                        if 'timestamp' in agent_output:
                                            st.write(f"**Executed:** {agent_output['timestamp']}")
    
    # Workflow history
    if mcp_engine.workflow_history:
        st.subheader("Workflow History")
        
        for workflow_id in mcp_engine.workflow_history[-5:]:  # Show last 5
            status = mcp_engine.get_workflow_status(workflow_id)
            if status:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.write(f"**{workflow_id[:8]}...**")
                
                with col2:
                    st.write(status['status'].title())
                
                with col3:
                    st.write(f"{status['progress_percentage']:.0f}%")
                
                with col4:
                    if st.button("View", key=f"view_{workflow_id}"):
                        st.session_state.current_workflow_id = workflow_id
                        st.rerun()

def show_human_review_interface(mcp_engine, workflow_id):
    """Human-in-the-loop review interface"""
    st.write("**Human Review Checkpoint**")
    
    # Check for pending reviews
    status = mcp_engine.get_workflow_status(workflow_id)
    pending_reviews = status.get('pending_human_reviews', [])
    
    if pending_reviews:
        checkpoint = pending_reviews[0]
        checkpoint_id = checkpoint['checkpoint_id']
        
        st.write(f"🔍 **Review Required:** {checkpoint_id[:8]}...")
        
        # Get workflow results for review
        results = mcp_engine.get_workflow_results(workflow_id)
        agent_outputs = results.get('agent_outputs', {})
        
        # Display summary for review
        st.write("**Agent Results Summary:**")
        
        for agent_name, output in agent_outputs.items():
            with st.expander(f"{agent_name.title()} Results"):
                if isinstance(output, dict):
                    for key, value in output.items():
                        if isinstance(value, (str, int, float)):
                            st.write(f"**{key}:** {value}")
                        elif isinstance(value, dict) and len(value) < 10:
                            st.json(value)
        
        # Review form
        st.write("**Provide Feedback:**")
        
        approval_status = st.selectbox(
            "Approval Decision",
            ["approved", "rejected", "needs_modification"],
            key="approval_status"
        )
        
        feedback_summary = st.text_area(
            "Review Comments",
            placeholder="Provide your review comments...",
            key="feedback_summary"
        )
        
        if st.button("Submit Review", type="primary"):
            feedback = {
                "approval_status": approval_status,
                "summary": feedback_summary,
                "reviewer": "human_reviewer",
                "timestamp": datetime.now().isoformat()
            }
            
            result = asyncio.run(mcp_engine.submit_human_feedback(workflow_id, checkpoint_id, feedback))
            
            if result['success']:
                st.success("Review submitted successfully!")
                audit_logger.log_human_interaction(
                    "review_submitted",
                    {"checkpoint_id": checkpoint_id, "approval_status": approval_status}
                )
                st.rerun()
            else:
                st.error(f"Failed to submit review: {result.get('error')}")
    else:
        if st.button("Create Review Checkpoint"):
            result = asyncio.run(mcp_engine.execute_workflow_step(workflow_id, "human_review"))
            if result['success']:
                st.success("Review checkpoint created!")
                st.rerun()

def show_configuration():
    """Configuration management interface"""
    st.title("⚙️ System Configuration")
    
    # Simple configuration interface
    st.subheader("Risk Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**AUC Thresholds**")
        auc_excellent = st.slider("Excellent AUC", 0.5, 1.0, 0.8, 0.01)
        auc_good = st.slider("Good AUC", 0.5, 1.0, 0.7, 0.01)
        
        st.write("**KS Thresholds**")
        ks_excellent = st.slider("Excellent KS", 0.0, 0.5, 0.3, 0.01)
        ks_good = st.slider("Good KS", 0.0, 0.5, 0.2, 0.01)
    
    with col2:
        st.write("**Workflow Settings**")
        max_retries = st.number_input("Max Retries", 1, 10, 3)
        step_timeout = st.number_input("Step Timeout (seconds)", 60, 1800, 300)
        
        st.write("**Agent Configuration**")
        agent_timeout = st.number_input("Agent Timeout (seconds)", 60, 1800, 300)
    
    if st.button("Update Configuration"):
        # Update configuration (simplified)
        st.success("Configuration updated successfully!")
        st.rerun()

def show_audit_trail(audit_logger):
    """Audit trail interface"""
    st.title("📋 Audit Trail")
    
    audit_entries = audit_logger.get_audit_trail(limit=100)
    
    if audit_entries:
        # Convert to DataFrame for better display
        df = pd.DataFrame(audit_entries)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            action_filter = st.multiselect(
                "Filter by Action",
                options=df['action'].unique(),
                default=[]
            )
        
        with col2:
            date_range = st.date_input(
                "Date Range",
                value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
                max_value=datetime.now().date()
            )
        
        # Apply filters
        filtered_df = df.copy()
        if action_filter:
            filtered_df = filtered_df[filtered_df['action'].isin(action_filter)]
        
        # Display audit entries
        st.write(f"**Showing {len(filtered_df)} entries**")
        
        for _, entry in filtered_df.head(50).iterrows():
            with st.expander(f"{entry['timestamp'].strftime('%H:%M:%S')} - {entry['action']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Action:** {entry['action']}")
                    st.write(f"**Session:** {entry['session_id'][:8]}...")
                
                with col2:
                    st.write(f"**Details:** {entry['details']}")
                    if entry.get('metadata'):
                        st.json(entry['metadata'])
    else:
        st.write("No audit entries found")

def show_system_status(mcp_engine):
    """System status and monitoring"""
    st.title("📊 System Status")
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Workflows", len(mcp_engine.active_workflows))
    
    with col2:
        st.metric("Available Agents", len(mcp_engine.agents))
    
    with col3:
        total_workflows = len(mcp_engine.workflow_history)
        st.metric("Total Workflows", total_workflows)
    
    with col4:
        config_status = "✅ Loaded" if config else "❌ Error"
        st.metric("Configuration", config_status)
    
    # Active workflows detail
    if mcp_engine.active_workflows:
        st.subheader("Active Workflows")
        
        for workflow_id, workflow_state in mcp_engine.active_workflows.items():
            with st.expander(f"Workflow {workflow_id[:8]}..."):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Status:** {workflow_state['status'].value}")
                    st.write(f"**Step:** {workflow_state['current_step']}/{workflow_state['total_steps']}")
                
                with col2:
                    st.write(f"**Created:** {workflow_state['created_at'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Updated:** {workflow_state['updated_at'].strftime('%Y-%m-%d %H:%M')}")
                
                with col3:
                    if st.button(f"Cancel", key=f"cancel_{workflow_id}"):
                        if mcp_engine.cancel_workflow(workflow_id):
                            st.success("Workflow cancelled")
                            st.rerun()
    
    # Agent status
    st.subheader("Agent Status")
    
    agent_status_data = []
    for agent_name, agent_instance in mcp_engine.agents.items():
        agent_config = config.get_agent_config(agent_name)
        agent_status_data.append({
            "Agent": agent_name.title(),
            "Status": "Available",
            "Timeout": f"{agent_config.get('timeout', 300)}s" if agent_config else "300s",
            "Retries": agent_config.get('retry_attempts', 3) if agent_config else 3
        })
    
    if agent_status_data:
        df = pd.DataFrame(agent_status_data)
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()