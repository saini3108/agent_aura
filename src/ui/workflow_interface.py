
"""
Workflow Interface Components
============================

MCP workflow management and execution interface with integrated agent results
"""

import streamlit as st
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List

def show_mcp_workflow(mcp_engine, audit_logger):
    """MCP workflow management interface with integrated agent results"""
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

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Status", status.get('status', 'Unknown'))
            with col2:
                st.metric("Current Step", status.get('current_step', 'Unknown'))

            # Agent execution with human-readable results
            st.subheader("Agent Execution & Results")

            agents_config = {
                'analyst': {
                    'title': 'Data Analyst',
                    'description': 'Analyzes data quality, features, and structure',
                    'icon': '📊',
                    'relevant_metrics': ['missing_percentage', 'outlier_percentage', 'feature_count']
                },
                'validator': {
                    'title': 'Model Validator', 
                    'description': 'Validates model performance and statistics',
                    'icon': '✅',
                    'relevant_metrics': ['auc', 'gini', 'ks_statistic', 'psi']
                },
                'documentation': {
                    'title': 'Documentation Reviewer',
                    'description': 'Reviews compliance and documentation',
                    'icon': '📋',
                    'relevant_metrics': ['compliance_score', 'missing_documents']
                },
                'reviewer': {
                    'title': 'Risk Reviewer',
                    'description': 'Assesses overall risk and provides recommendations',
                    'icon': '🔍',
                    'relevant_metrics': ['risk_level', 'confidence_score']
                },
                'auditor': {
                    'title': 'Final Auditor',
                    'description': 'Performs final validation and approval',
                    'icon': '🏛️',
                    'relevant_metrics': ['approval_status', 'audit_score']
                }
            }

            # Track which agent is currently expanded
            if 'expanded_agent' not in st.session_state:
                st.session_state.expanded_agent = None

            for agent_name, config in agents_config.items():
                # Check if this agent has been executed
                agent_status = status.get('agent_executions', {}).get(agent_name, {})
                is_completed = agent_status.get('status') == 'completed'
                is_failed = agent_status.get('status') == 'failed'
                is_running = agent_status.get('status') == 'running'

                # Auto-expand logic: expand if completed and no other agent is expanded, or if this agent was just executed
                should_expand = (is_completed and st.session_state.expanded_agent is None) or st.session_state.expanded_agent == agent_name

                # Create the expander with auto-expand behavior
                with st.expander(f"{config['icon']} {config['title']}", expanded=should_expand):

                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        st.write(f"**{config['description']}**")

                    with col2:
                        if st.button(f"Execute", key=f"exec_{agent_name}"):
                            try:
                                result = asyncio.run(mcp_engine.execute_workflow_step(workflow_id, agent_name))
                                
                                # Set this agent as expanded after execution and collapse others
                                st.session_state.expanded_agent = agent_name

                                st.success(f"{config['title']} completed")
                                audit_logger.log_agent_execution(
                                    agent_name, 
                                    "completed", 
                                    result.get('execution_time', 0),
                                    result
                                )
                                st.rerun()
                            except Exception as e:
                                st.error(f"Execution failed: {str(e)}")
                                st.session_state.expanded_agent = agent_name  # Expand to show error

                    with col3:
                        # Show agent status with better visual indicators
                        if is_completed:
                            st.success("✓ Completed")
                        elif is_running:
                            st.info("⏳ Running")
                        elif is_failed:
                            st.error("✗ Failed")
                        else:
                            st.write("⚪ Pending")

                    # Show human-readable results for completed agents
                    if is_completed and 'output' in agent_status:
                        result = agent_status['output']
                        
                        # Executive Summary Section
                        if result.get('executive_summary'):
                            st.markdown("### 📋 Executive Summary")
                            st.info(result['executive_summary'])
                        elif result.get('analysis'):
                            st.markdown("### 📋 Analysis Summary")
                            st.info(result['analysis'])

                        # Key Metrics Section (agent-specific)
                        if result.get('metrics'):
                            st.markdown("### 📊 Key Performance Metrics")
                            metrics = result['metrics']
                            
                            # Display agent-specific metrics in a clean format
                            if agent_name == 'auditor':
                                _display_auditor_metrics(metrics)
                            elif agent_name == 'validator':
                                _display_validator_metrics(metrics)
                            elif agent_name == 'analyst':
                                _display_analyst_metrics(metrics)
                            elif agent_name == 'documentation':
                                _display_documentation_metrics(metrics)
                            elif agent_name == 'reviewer':
                                _display_reviewer_metrics(metrics)

                        # Recommendations Section
                        if result.get('recommendations'):
                            st.markdown("### 💡 Key Recommendations")
                            for i, rec in enumerate(result['recommendations'][:3], 1):
                                st.write(f"**{i}.** {rec}")

                        # Detailed Analysis (Collapsible)
                        with st.expander("📋 Detailed Analysis Results", expanded=False):
                            _display_detailed_results(agent_name, result)

                        # User Feedback Section
                        _display_feedback_section(agent_name)

                    elif is_failed:
                        st.error(f"**Execution Failed:** {agent_status.get('error', 'Unknown error')}")
                    elif is_running:
                        st.info("**Status:** Agent is currently executing...")
                    else:
                        st.info(f"**Ready to Execute:** {config['description']}")

            # Human review checkpoint
            if status.get('current_step') == 'waiting_for_human':
                show_human_review_interface(mcp_engine, workflow_id)

        else:
            st.info("No workflow status available")

    # Workflow history
    st.subheader("📚 Workflow History")
    if hasattr(mcp_engine, 'workflow_history') and mcp_engine.workflow_history:
        for workflow in mcp_engine.workflow_history[-3:]:  # Show last 3
            with st.expander(f"Workflow {workflow.get('id', 'Unknown')[:8]}... - {workflow.get('status', 'Unknown')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Created:** {workflow.get('created_at', 'Unknown')}")
                    st.write(f"**Status:** {workflow.get('status', 'Unknown')}")
                with col2:
                    st.write(f"**Steps:** {workflow.get('current_step', 'Unknown')}")
                    agent_count = len(workflow.get('agent_executions', {}))
                    st.write(f"**Agents:** {agent_count}")
    else:
        st.info("No workflow history available")

def _display_auditor_metrics(metrics):
    """Display auditor-specific metrics in a clean format"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'audit_score' in metrics:
            st.metric("Audit Score", f"{metrics['audit_score']:.2f}")
        if 'approval_status' in metrics:
            st.metric("Approval Status", metrics['approval_status'])
    
    with col2:
        if 'compliance_status' in metrics:
            st.metric("Compliance", metrics['compliance_status'])
        if 'overall_audit_score' in metrics:
            st.metric("Overall Score", f"{metrics['overall_audit_score']:.2f}")
    
    with col3:
        if 'independent_opinion' in metrics:
            st.metric("Independent Opinion", metrics['independent_opinion'])

def _display_validator_metrics(metrics):
    """Display validator-specific metrics in a clean format"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'auc' in metrics:
            st.metric("AUC", f"{metrics['auc']:.3f}")
    
    with col2:
        if 'gini' in metrics:
            st.metric("Gini Coefficient", f"{metrics['gini']:.3f}")
    
    with col3:
        if 'ks_statistic' in metrics:
            st.metric("KS Statistic", f"{metrics['ks_statistic']:.3f}")
    
    with col4:
        if 'psi' in metrics:
            st.metric("PSI", f"{metrics['psi']:.3f}")

def _display_analyst_metrics(metrics):
    """Display analyst-specific metrics in a clean format"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'missing_percentage' in metrics:
            st.metric("Missing Data", f"{metrics['missing_percentage']:.1f}%")
    
    with col2:
        if 'outlier_percentage' in metrics:
            st.metric("Outliers", f"{metrics['outlier_percentage']:.1f}%")
    
    with col3:
        if 'feature_count' in metrics:
            st.metric("Features", metrics['feature_count'])

def _display_documentation_metrics(metrics):
    """Display documentation-specific metrics in a clean format"""
    col1, col2 = st.columns(2)
    
    with col1:
        if 'compliance_score' in metrics:
            st.metric("Compliance Score", f"{metrics['compliance_score']:.1f}%")
    
    with col2:
        if 'missing_documents' in metrics:
            st.metric("Missing Documents", metrics['missing_documents'])

def _display_reviewer_metrics(metrics):
    """Display reviewer-specific metrics in a clean format"""
    col1, col2 = st.columns(2)
    
    with col1:
        if 'risk_level' in metrics:
            st.metric("Risk Level", metrics['risk_level'])
    
    with col2:
        if 'confidence_score' in metrics:
            st.metric("Confidence", f"{metrics['confidence_score']:.2f}")

def _display_detailed_results(agent_name, result):
    """Display detailed results for each agent type"""
    if agent_name == 'auditor':
        # Audit findings
        if result.get('audit_findings'):
            st.markdown("#### 🔍 Audit Findings")
            audit_findings = result['audit_findings']
            
            if audit_findings.get('independent_assessment'):
                assessment = audit_findings['independent_assessment']
                st.write(f"**Independent Opinion:** {assessment.get('independent_opinion', 'Unknown')}")
                st.write(f"**Overall Audit Score:** {assessment.get('overall_audit_score', 0.0):.2f}")
                
                if assessment.get('critical_issues'):
                    st.write("**Critical Issues:**")
                    for issue in assessment['critical_issues']:
                        st.write(f"• {issue}")

        # Compliance assessment
        if result.get('compliance_assessment'):
            st.markdown("#### 📋 Compliance Assessment")
            compliance = result['compliance_assessment']
            st.write(f"**Basel III:** {compliance.get('basel_compliance', 'Unknown')}")
            st.write(f"**IFRS 9:** {compliance.get('ifrs9_compliance', 'Unknown')}")
            st.write(f"**Overall:** {compliance.get('overall_compliance', 'Unknown')}")

        # Final recommendation
        if result.get('final_recommendation'):
            st.markdown("#### 📝 Final Recommendation")
            recommendation = result['final_recommendation']
            st.write(f"**Status:** {recommendation.get('approval_status', 'Unknown')}")
            st.write(f"**Type:** {recommendation.get('recommendation_type', 'Unknown')}")
            if recommendation.get('conditions'):
                st.write("**Conditions:**")
                for condition in recommendation['conditions']:
                    st.write(f"• {condition}")
    
    # For other agent types, show a summary
    else:
        if result.get('summary'):
            st.write(f"**Summary:** {result['summary']}")
        if result.get('findings'):
            st.write("**Key Findings:**")
            for finding in result['findings'][:5]:
                st.write(f"• {finding}")

def _display_feedback_section(agent_name):
    """Display user feedback section for each agent"""
    st.markdown("#### 💬 Provide Feedback (Optional)")
    feedback_key = f"feedback_{agent_name}_{datetime.now().strftime('%H%M%S')}"
    
    col_feedback, col_submit = st.columns([3, 1])
    with col_feedback:
        user_feedback = st.text_input(
            "Your feedback on this analysis:",
            key=feedback_key,
            placeholder="Optional: Share your thoughts on this analysis..."
        )
    
    with col_submit:
        if st.button("Submit", key=f"submit_feedback_{agent_name}_{datetime.now().strftime('%H%M%S')}"):
            if user_feedback.strip():
                # Store feedback
                if 'user_feedback' not in st.session_state:
                    st.session_state.user_feedback = {}
                st.session_state.user_feedback[f"{agent_name}_{datetime.now().isoformat()}"] = {
                    'feedback': user_feedback,
                    'timestamp': datetime.now().isoformat(),
                    'agent': agent_name
                }
                st.success("Feedback submitted!")
                st.rerun()

def show_human_review_interface(mcp_engine, workflow_id):
    """Human-in-the-loop review interface"""
    st.subheader("🧑‍💼 Human Review Required")

    # Get workflow results for review
    workflow_results = mcp_engine.get_workflow_results(workflow_id)

    if workflow_results:
        # Display summary
        st.markdown("### Review Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Data Records", len(workflow_results.get('data', {})))
            st.metric("Validation Score", f"{workflow_results.get('validation_score', 0):.2f}")

        with col2:
            st.metric("Compliance Score", f"{workflow_results.get('compliance_score', 0):.1f}%")
            st.metric("Risk Level", workflow_results.get('risk_level', 'Unknown'))

        # Show key findings
        st.markdown("### Key Findings")

        findings = workflow_results.get('findings', [])
        if findings:
            for i, finding in enumerate(findings[:5]):  # Show top 5
                st.write(f"**{i+1}.** {finding}")
        else:
            st.info("No specific findings to review")

        # Review form
        st.markdown("### Your Review")

        with st.form("human_review_form"):
            approval_status = st.radio(
                "Decision",
                ["approved", "rejected", "needs_revision"],
                format_func=lambda x: {
                    "approved": "✅ Approve - Continue workflow",
                    "rejected": "❌ Reject - Stop workflow", 
                    "needs_revision": "🔄 Needs Revision - Request changes"
                }[x]
            )

            feedback_text = st.text_area(
                "Comments and Feedback",
                placeholder="Please provide your feedback and any specific recommendations..."
            )

            risk_adjustment = st.slider(
                "Risk Level Adjustment",
                min_value=-2, max_value=2, value=0,
                help="Adjust the risk assessment based on your expert judgment"
            )

            submitted = st.form_submit_button("Submit Review")

            if submitted:
                feedback = {
                    "approval_status": approval_status,
                    "feedback_text": feedback_text,
                    "risk_adjustment": risk_adjustment,
                    "reviewer": "Human Reviewer",
                    "timestamp": datetime.now().isoformat()
                }

                try:
                    # Submit feedback to workflow
                    checkpoint_id = workflow_results.get('checkpoint_id', 'unknown')
                    asyncio.run(mcp_engine.submit_human_feedback(workflow_id, checkpoint_id, feedback))

                    st.success("Review submitted successfully!")
                    st.session_state.human_review_submitted = True
                    st.rerun()

                except Exception as e:
                    st.error(f"Failed to submit review: {str(e)}")

    else:
        st.warning("No workflow results available for review")
