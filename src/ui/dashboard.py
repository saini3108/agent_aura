"""
Dashboard UI Components
======================

Main dashboard interface for ValiCred-AI system
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any, Optional

def show_dashboard(mcp_engine, audit_logger, sample_loader):
    """Enhanced dashboard with MCP integration"""
    st.title("🏦 ValiCred-AI Dashboard")
    st.markdown("**Agent Aura Architecture** - MCP + LangGraph + HITL")
    
    # Initialize components if not already in session state
    if 'user_feedback' not in st.session_state:
        st.session_state.user_feedback = {}
    
    if 'memory_manager' not in st.session_state:
        from src.core.memory_manager import MemoryManager
        memory_config = st.session_state.get('system_config', {}).get('memory_config', {})
        st.session_state.memory_manager = MemoryManager(memory_config)
    
    if 'human_loop_manager' not in st.session_state:
        from src.core.human_in_loop import HumanInLoopManager
        hitl_config = st.session_state.get('system_config', {}).get('hitl_config', {})
        st.session_state.human_loop_manager = HumanInLoopManager(hitl_config)

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
            if audit_logger:
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
            if audit_logger:
                audit_logger.log_data_operation(
                    "sample_documents_loaded",
                    {"documents": len(sample_docs)}
                )
            st.success(f"Loaded {len(sample_docs)} documents")
            st.rerun()

    # Data Preview
    if st.session_state.get('validation_data') is not None:
        st.subheader("Data Preview")
        data_preview = st.session_state.validation_data.head()
        st.dataframe(data_preview)
        
        # Basic data visualization
        col1, col2 = st.columns(2)
        
        with col1:
            if 'default' in st.session_state.validation_data.columns:
                default_dist = st.session_state.validation_data['default'].value_counts()
                fig = px.pie(
                    values=default_dist.values, 
                    names=['Non-Default', 'Default'],
                    title="Default Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'credit_score' in st.session_state.validation_data.columns:
                fig = px.histogram(
                    st.session_state.validation_data,
                    x='credit_score',
                    title="Credit Score Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

    # Recent Activity
    st.subheader("Recent Activity")
    recent_activities = audit_logger.get_audit_trail(limit=5)
    if recent_activities:
        for activity in recent_activities:
            timestamp = activity.get('timestamp', datetime.now())
            if hasattr(timestamp, 'strftime'):
                time_str = timestamp.strftime('%H:%M:%S')
            else:
                time_str = str(timestamp)
            
            event_type = activity.get('event_type', activity.get('operation', 'activity'))
            st.write(f"**{time_str}** - {event_type}")
    else:
        st.info("No recent activity")

def show_system_status(mcp_engine):
    """System status and monitoring"""
    st.title("🔧 System Status")
    
    # System Health
    st.subheader("System Health")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Server Status", "Running", delta="Healthy")
    
    with col2:
        st.metric("Active Workflows", len(mcp_engine.active_workflows))
    
    with col3:
        st.metric("Available Agents", len(mcp_engine.agents))
    
    with col4:
        st.metric("System Uptime", "Connected")

    # Workflow Status
    st.subheader("Workflow Status")
    if mcp_engine.active_workflows:
        for workflow_id, workflow in mcp_engine.active_workflows.items():
            with st.expander(f"Workflow {workflow_id[:8]}..."):
                status = mcp_engine.get_workflow_status(workflow_id)
                if status:
                    st.json(status)
                else:
                    st.info("Status not available")
    else:
        st.info("No active workflows")

    # Agent Status
    st.subheader("Agent Status")
    for agent_name, agent in mcp_engine.agents.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{agent_name.title()} Agent**")
        with col2:
            st.success("Ready")