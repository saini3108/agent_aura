import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from typing import Dict, Any, List

# Import custom modules with new structure
import sys
from pathlib import Path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir / "agent-service" / "agents"))
sys.path.append(str(current_dir / "shared"))

from analyst_agent import AnalystAgent
from validator_agent import ValidatorAgent
from documentation_agent import DocumentationAgent
from reviewer_agent import ReviewerAgent
from auditor_agent import AuditorAgent
from validation_metrics import ValidationMetrics
from workflow_manager import WorkflowManager
from audit_logger import AuditLogger
from report_generator import ReportGenerator
from sample_data_loader import sample_loader
from system_config import config

# Initialize session state
if 'workflow_state' not in st.session_state:
    st.session_state.workflow_state = {
        'current_step': 0,
        'completed_steps': [],
        'agent_outputs': {},
        'human_feedback': {},
        'validation_results': {},
        'audit_trail': []
    }

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}

if 'validation_data' not in st.session_state:
    st.session_state.validation_data = None

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all system components"""
    audit_logger = AuditLogger()
    workflow_manager = WorkflowManager(audit_logger)
    report_generator = ReportGenerator()
    
    # Initialize agents
    agents = {
        'analyst': AnalystAgent(),
        'validator': ValidatorAgent(),
        'documentation': DocumentationAgent(),
        'reviewer': ReviewerAgent(),
        'auditor': AuditorAgent()
    }
    
    return workflow_manager, audit_logger, report_generator, agents

def main():
    st.set_page_config(
        page_title="ValiCred-AI: Credit Risk Model Validation",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize components
    workflow_manager, audit_logger, report_generator, agents = initialize_components()
    
    # Main title
    st.title("🏦 ValiCred-AI: Credit Risk Model Validation System")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Dashboard", "Data Upload", "Agent Workflow", "Validation Results", "Human Review", "Audit Trail", "Reports"]
    )
    
    # Display workflow status
    st.sidebar.markdown("### Workflow Status")
    progress = len(st.session_state.workflow_state['completed_steps']) / 6
    st.sidebar.progress(progress)
    st.sidebar.write(f"Step {st.session_state.workflow_state['current_step']}/6")
    
    # Page routing
    if page == "Dashboard":
        show_dashboard()
    elif page == "Data Upload":
        show_data_upload()
    elif page == "Agent Workflow":
        show_agent_workflow(workflow_manager, agents, audit_logger)
    elif page == "Validation Results":
        show_validation_results()
    elif page == "Human Review":
        show_human_review()
    elif page == "Audit Trail":
        show_audit_trail(audit_logger)
    elif page == "Reports":
        show_reports(report_generator)

def show_dashboard():
    """Display the main dashboard"""
    st.header("📊 ValiCred-AI Dashboard")
    
    # System status overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        validation_status = "In Progress" if st.session_state.workflow_state['current_step'] > 0 else "Ready"
        if len(st.session_state.workflow_state['completed_steps']) >= 6:
            validation_status = "Complete"
        st.metric(
            "Validation Status",
            validation_status,
            f"{len(st.session_state.workflow_state['completed_steps'])}/6 steps"
        )
    
    with col2:
        data_status = "Loaded" if st.session_state.validation_data is not None else "None"
        records_count = len(st.session_state.validation_data) if st.session_state.validation_data is not None else 0
        st.metric(
            "Dataset",
            data_status,
            f"{records_count:,} records" if records_count > 0 else "No data"
        )
    
    with col3:
        st.metric(
            "Documents",
            len(st.session_state.uploaded_files),
            "files uploaded"
        )
    
    with col4:
        st.metric(
            "Audit Trail",
            len(st.session_state.workflow_state['audit_trail']),
            "entries logged"
        )
    
    st.markdown("---")
    
    # Quick start section
    st.subheader("Quick Start")
    
    if st.session_state.validation_data is None:
        st.info("💡 **Get Started**: Load sample data to begin the validation process")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Load Sample Credit Data", use_container_width=True, type="primary"):
                try:
                    df, data_info = sample_loader.load_credit_data()
                    st.session_state.validation_data = df
                    st.session_state.data_info = data_info
                    st.success(f"Sample data loaded! {len(df):,} records with {data_info.get('default_rate', 0):.1%} default rate")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with col2:
            if st.button("📄 Load Sample Documents", use_container_width=True):
                sample_docs = sample_loader.get_sample_documents()
                st.session_state.uploaded_files.update(sample_docs)
                st.success(f"Loaded {len(sample_docs)} sample documents")
                st.rerun()
    
    else:
        # Show data summary
        df = st.session_state.validation_data
        st.subheader("Current Dataset Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Data Overview**")
            st.write(f"• Records: {len(df):,}")
            st.write(f"• Features: {len(df.columns)}")
            if 'default_flag' in df.columns:
                st.write(f"• Default Rate: {df['default_flag'].mean():.1%}")
        
        with col2:
            st.write("**Data Quality**")
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.write(f"• Missing Values: {missing_pct:.1%}")
            st.write(f"• Duplicates: {df.duplicated().sum()}")
            st.write(f"• Completeness: {100-missing_pct:.1%}")
        
        with col3:
            st.write("**Ready for Validation**")
            if st.button("🚀 Start Validation Workflow", use_container_width=True, type="primary"):
                st.switch_page("Agent Workflow")
    
    # Workflow progress
    if st.session_state.workflow_state['current_step'] > 0:
        st.subheader("Validation Progress")
        
        progress_steps = [
            "Analyst Agent",
            "Validator Agent", 
            "Documentation Agent",
            "Human Review",
            "Reviewer Agent",
            "Auditor Agent"
        ]
        
        current_step = st.session_state.workflow_state['current_step']
        completed_steps = st.session_state.workflow_state['completed_steps']
        
        progress_bar = st.progress(len(completed_steps) / len(progress_steps))
        
        # Show step status
        for i, step_name in enumerate(progress_steps):
            if i in completed_steps:
                st.write(f"✅ {step_name} - Completed")
            elif i == current_step:
                st.write(f"🔄 {step_name} - In Progress")
            else:
                st.write(f"⏳ {step_name} - Pending")
    
    # Recent activity
    st.subheader("Recent Activity")
    if st.session_state.workflow_state['audit_trail']:
        recent_activities = st.session_state.workflow_state['audit_trail'][-5:]
        for activity in reversed(recent_activities):
            timestamp = activity['timestamp'].split('T')[1][:8] if 'T' in activity['timestamp'] else activity['timestamp']
            st.write(f"• `{timestamp}` {activity['action']}")
    else:
        st.info("No activity yet. Load data and start validation to see activity here.")
    
    # System information
    with st.expander("System Information"):
        st.write("**ValiCred-AI Components:**")
        st.write("• Multi-agent validation system with 6 specialized agents")
        st.write("• Human-in-the-loop review capabilities")
        st.write("• Comprehensive audit trail and reporting")
        st.write("• Real-time validation metrics calculation")
        st.write("• Regulatory compliance assessment")
        
        if st.button("Reset System", type="secondary"):
            st.session_state.workflow_state = {
                'current_step': 0,
                'completed_steps': [],
                'agent_outputs': {},
                'human_feedback': {},
                'validation_results': {},
                'audit_trail': []
            }
            st.session_state.validation_data = None
            st.session_state.uploaded_files = {}
            st.success("System reset successfully!")
            st.rerun()

def show_data_upload():
    """Handle data and document uploads"""
    st.header("📁 Data Upload")
    
    # Sample data section
    st.subheader("Sample Data")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Load Sample Credit Data", use_container_width=True):
            try:
                df, data_info = sample_loader.load_credit_data()
                st.session_state.validation_data = df
                st.session_state.data_info = data_info
                st.success(f"Sample data loaded! Shape: {df.shape}, Default rate: {data_info.get('default_rate', 0):.2%}")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
    
    with col2:
        if st.button("Load Sample Documents", use_container_width=True):
            try:
                sample_docs = sample_loader.get_sample_documents()
                st.session_state.uploaded_files.update(sample_docs)
                st.success(f"Loaded {len(sample_docs)} sample documents!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample documents: {str(e)}")
    
    st.markdown("---")
    
    # Model data upload
    st.subheader("Model Data")
    uploaded_data = st.file_uploader(
        "Upload model data (CSV format)",
        type=['csv'],
        help="Upload the dataset used for model training/validation"
    )
    
    if uploaded_data is not None:
        try:
            df = pd.read_csv(uploaded_data)
            st.session_state.validation_data = df
            st.success(f"Data uploaded successfully! Shape: {df.shape}")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Basic data info
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Data Info:**")
                st.write(f"- Rows: {df.shape[0]:,}")
                st.write(f"- Columns: {df.shape[1]}")
                st.write(f"- Missing values: {df.isnull().sum().sum()}")
                if 'default_flag' in df.columns:
                    st.write(f"- Default rate: {df['default_flag'].mean():.2%}")
            
            with col2:
                st.write("**Column Types:**")
                for dtype in df.dtypes.value_counts().items():
                    st.write(f"- {dtype[0]}: {dtype[1]} columns")
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    # Show current data if loaded
    if st.session_state.validation_data is not None:
        st.subheader("Current Dataset")
        df = st.session_state.validation_data
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Records", f"{len(df):,}")
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            if 'default_flag' in df.columns:
                st.metric("Default Rate", f"{df['default_flag'].mean():.2%}")
        
        # Show columns
        with st.expander("View Column Details"):
            for col in df.columns:
                col_info = f"**{col}**: {df[col].dtype}"
                if df[col].dtype in ['int64', 'float64']:
                    col_info += f" (Range: {df[col].min():.2f} - {df[col].max():.2f})"
                elif df[col].dtype == 'object':
                    unique_vals = df[col].nunique()
                    col_info += f" ({unique_vals} unique values)"
                st.write(col_info)
    
    st.markdown("---")
    
    # Document upload
    st.subheader("Model Documentation")
    uploaded_docs = st.file_uploader(
        "Upload model documentation",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload model documentation, validation reports, or compliance documents"
    )
    
    if uploaded_docs:
        for doc in uploaded_docs:
            st.session_state.uploaded_files[doc.name] = {
                'file': doc,
                'uploaded_at': datetime.now().isoformat(),
                'size': doc.size
            }
        
        st.success(f"Uploaded {len(uploaded_docs)} document(s)")
        
        # Display uploaded files
        if st.session_state.uploaded_files:
            st.subheader("Uploaded Files")
            for filename, file_info in st.session_state.uploaded_files.items():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"📄 {filename}")
                with col2:
                    st.write(f"{file_info['size']} bytes")
                with col3:
                    if st.button("Remove", key=f"remove_{filename}"):
                        del st.session_state.uploaded_files[filename]
                        st.rerun()

def show_agent_workflow(workflow_manager, agents, audit_logger):
    """Display and manage the agent workflow"""
    st.header("🤖 Agent Workflow")
    
    # Workflow steps
    steps = [
        ("Analyst Agent", "Analyze model structure and parameters"),
        ("Validator Agent", "Calculate validation metrics"),
        ("Documentation Agent", "Review compliance documentation"),
        ("Human Review", "Human-in-the-loop checkpoint"),
        ("Reviewer Agent", "Generate findings and recommendations"),
        ("Auditor Agent", "Final validation and approval")
    ]
    
    # Display workflow progress
    st.subheader("Workflow Progress")
    for i, (step_name, description) in enumerate(steps):
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            if i in st.session_state.workflow_state['completed_steps']:
                st.success("✅")
            elif i == st.session_state.workflow_state['current_step']:
                st.warning("🔄")
            else:
                st.info("⏳")
        
        with col2:
            st.write(f"**{step_name}**")
            st.write(description)
        
        with col3:
            if i == st.session_state.workflow_state['current_step']:
                if st.button(f"Run Step {i+1}", key=f"run_step_{i}"):
                    run_workflow_step(i, workflow_manager, agents, audit_logger)
    
    st.markdown("---")
    
    # Display current step details
    current_step = st.session_state.workflow_state['current_step']
    if current_step < len(steps):
        st.subheader(f"Current Step: {steps[current_step][0]}")
        
        # Show agent output if available
        step_key = f"step_{current_step}"
        if step_key in st.session_state.workflow_state['agent_outputs']:
            output = st.session_state.workflow_state['agent_outputs'][step_key]
            st.write("**Agent Output:**")
            st.write(output)

def run_workflow_step(step_index, workflow_manager, agents, audit_logger):
    """Execute a specific workflow step"""
    step_names = ['analyst', 'validator', 'documentation', 'human_review', 'reviewer', 'auditor']
    
    if step_index == 3:  # Human review step
        st.info("Human review step - please go to the Human Review page to continue")
        return
    
    # Check prerequisites
    if step_index == 0:  # Analyst step
        if st.session_state.validation_data is None:
            st.error("Please upload validation data first")
            return
    
    if step_index == 2:  # Documentation step
        if not st.session_state.uploaded_files:
            st.error("Please upload documentation first")
            return
    
    # Execute agent step
    with st.spinner(f"Running {step_names[step_index]} agent..."):
        try:
            agent = agents[step_names[step_index]]
            
            # Prepare context
            context = {
                'data': st.session_state.validation_data,
                'files': st.session_state.uploaded_files,
                'previous_outputs': st.session_state.workflow_state['agent_outputs']
            }
            
            # Run agent
            result = agent.run(context)
            
            # Store result
            step_key = f"step_{step_index}"
            st.session_state.workflow_state['agent_outputs'][step_key] = result
            st.session_state.workflow_state['completed_steps'].append(step_index)
            
            # Move to next step
            if step_index + 1 < 6:
                st.session_state.workflow_state['current_step'] = step_index + 1
            
            # Log to audit trail
            audit_logger.log_action(
                f"Agent {step_names[step_index]} completed",
                f"Step {step_index + 1} completed successfully",
                {'result': result}
            )
            
            st.success(f"{step_names[step_index]} agent completed successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error running agent: {str(e)}")
            audit_logger.log_action(
                f"Agent {step_names[step_index]} failed",
                f"Error: {str(e)}",
                {'step': step_index}
            )

def show_validation_results():
    """Display validation results and metrics"""
    st.header("📊 Validation Results")
    
    if 'step_1' not in st.session_state.workflow_state['agent_outputs']:
        st.warning("No validation results available. Please run the Validator Agent first.")
        return
    
    # Get validation results
    validator_output = st.session_state.workflow_state['agent_outputs'].get('step_1', {})
    
    if not validator_output:
        st.info("Validation results will appear here after running the Validator Agent.")
        return
    
    # Display metrics
    st.subheader("Validation Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "AUC Score",
            f"{validator_output.get('auc', 0):.3f}",
            f"{'✅ Good' if validator_output.get('auc', 0) > 0.7 else '⚠️ Review'}"
        )
    
    with col2:
        st.metric(
            "KS Statistic",
            f"{validator_output.get('ks_statistic', 0):.3f}",
            f"{'✅ Good' if validator_output.get('ks_statistic', 0) > 0.2 else '⚠️ Review'}"
        )
    
    with col3:
        st.metric(
            "Population Stability",
            f"{validator_output.get('psi', 0):.3f}",
            f"{'✅ Stable' if validator_output.get('psi', 0) < 0.1 else '⚠️ Drift'}"
        )
    
    # Visualizations
    st.subheader("Validation Charts")
    
    # Create sample charts for demonstration
    if validator_output.get('roc_data'):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ROC Curve")
            roc_data = validator_output['roc_data']
            fig = px.line(
                x=roc_data['fpr'],
                y=roc_data['tpr'],
                title="ROC Curve",
                labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}
            )
            fig.add_shape(
                type="line", line=dict(dash="dash"),
                x0=0, x1=1, y0=0, y1=1
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Score Distribution")
            if validator_output.get('score_distribution'):
                fig = px.histogram(
                    validator_output['score_distribution'],
                    title="Score Distribution",
                    nbins=30
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results
    st.subheader("Detailed Results")
    st.json(validator_output)

def show_human_review():
    """Human-in-the-loop review interface"""
    st.header("👤 Human Review")
    
    # Check if human review is the current step
    if st.session_state.workflow_state['current_step'] != 3:
        st.info("Human review is not the current step in the workflow.")
        return
    
    # Display previous agent outputs for review
    st.subheader("Agent Outputs for Review")
    
    for step_key, output in st.session_state.workflow_state['agent_outputs'].items():
        step_num = int(step_key.split('_')[1])
        agent_names = ['Analyst', 'Validator', 'Documentation']
        
        if step_num < len(agent_names):
            with st.expander(f"{agent_names[step_num]} Agent Output"):
                st.write(output)
    
    st.markdown("---")
    
    # Human feedback form
    st.subheader("Provide Feedback")
    
    with st.form("human_feedback"):
        st.write("**Review the agent outputs above and provide your feedback:**")
        
        # Overall assessment
        assessment = st.radio(
            "Overall Assessment",
            ["Approve", "Approve with Comments", "Request Changes", "Reject"],
            help="Your overall assessment of the validation so far"
        )
        
        # Specific feedback
        feedback_text = st.text_area(
            "Detailed Feedback",
            placeholder="Provide specific comments, concerns, or recommendations...",
            height=150
        )
        
        # Risk assessment
        risk_level = st.select_slider(
            "Risk Level Assessment",
            options=["Low", "Medium", "High", "Critical"],
            value="Medium"
        )
        
        # Additional requirements
        requirements = st.text_area(
            "Additional Requirements",
            placeholder="Any additional validation requirements or tests needed...",
            height=100
        )
        
        submitted = st.form_submit_button("Submit Review")
        
        if submitted:
            # Store human feedback
            human_feedback = {
                'assessment': assessment,
                'feedback_text': feedback_text,
                'risk_level': risk_level,
                'requirements': requirements,
                'timestamp': datetime.now().isoformat(),
                'reviewer': 'Human Reviewer'  # In real system, this would be the logged-in user
            }
            
            st.session_state.workflow_state['human_feedback']['step_3'] = human_feedback
            
            # Log the review
            from utils.audit_logger import AuditLogger
            audit_logger = AuditLogger()
            audit_logger.log_action(
                "Human review completed",
                f"Assessment: {assessment}, Risk Level: {risk_level}",
                human_feedback
            )
            
            # Move to next step if approved
            if assessment in ["Approve", "Approve with Comments"]:
                st.session_state.workflow_state['completed_steps'].append(3)
                st.session_state.workflow_state['current_step'] = 4
                st.success("Review submitted! Workflow can continue to the next step.")
            else:
                st.warning("Review submitted. Workflow paused for required changes.")
            
            st.rerun()

def show_audit_trail(audit_logger):
    """Display the audit trail"""
    st.header("📋 Audit Trail")
    
    # Get audit entries
    audit_entries = st.session_state.workflow_state.get('audit_trail', [])
    
    if not audit_entries:
        st.info("No audit entries yet. The audit trail will populate as you use the system.")
        return
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        show_all = st.checkbox("Show all entries", value=True)
    with col2:
        if not show_all:
            max_entries = st.slider("Max entries to show", 1, len(audit_entries), 10)
            audit_entries = audit_entries[-max_entries:]
    
    # Display entries
    st.subheader(f"Audit Entries ({len(audit_entries)})")
    
    for i, entry in enumerate(reversed(audit_entries)):
        with st.expander(f"{entry['timestamp']} - {entry['action']}"):
            st.write(f"**Action:** {entry['action']}")
            st.write(f"**Details:** {entry['details']}")
            st.write(f"**Timestamp:** {entry['timestamp']}")
            if entry.get('metadata'):
                st.write("**Metadata:**")
                st.json(entry['metadata'])
    
    # Export audit trail
    st.markdown("---")
    if st.button("Export Audit Trail"):
        audit_data = pd.DataFrame(audit_entries)
        csv = audit_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def show_reports(report_generator):
    """Generate and display reports"""
    st.header("📄 Reports")
    
    # Check if workflow is complete enough for reporting
    completed_steps = len(st.session_state.workflow_state['completed_steps'])
    
    if completed_steps < 2:
        st.warning("Complete at least the Analyst and Validator steps to generate reports.")
        return
    
    # Report options
    st.subheader("Report Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Report Type",
            ["Validation Summary", "Detailed Analysis", "Audit Report", "Executive Summary"]
        )
    
    with col2:
        include_charts = st.checkbox("Include Charts", value=True)
    
    # Generate report button
    if st.button("Generate Report", use_container_width=True):
        with st.spinner("Generating report..."):
            try:
                # Prepare report data
                report_data = {
                    'workflow_state': st.session_state.workflow_state,
                    'validation_data_info': {
                        'shape': st.session_state.validation_data.shape if st.session_state.validation_data is not None else None,
                        'columns': st.session_state.validation_data.columns.tolist() if st.session_state.validation_data is not None else []
                    },
                    'uploaded_files': list(st.session_state.uploaded_files.keys()),
                    'generation_time': datetime.now().isoformat()
                }
                
                # Generate report content
                report_content = report_generator.generate_report(
                    report_type, 
                    report_data, 
                    include_charts
                )
                
                # Display report
                st.subheader(f"{report_type} Report")
                st.markdown(report_content)
                
                # Download option
                st.download_button(
                    label="Download Report (Markdown)",
                    data=report_content,
                    file_name=f"{report_type.lower().replace(' ', '_')}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
    
    # Report history
    st.markdown("---")
    st.subheader("Report History")
    st.info("Report history will be implemented in future versions.")

if __name__ == "__main__":
    main()
