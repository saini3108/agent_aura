"""
Reports Interface Components
===========================

Report generation and audit trail interfaces
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional

def show_reports(mcp_engine, audit_logger):
    """Reports and documentation generation"""
    st.title("📊 Reports & Documentation")
    
    # Report Generation Section
    st.subheader("Generate Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Report Type",
            ["Validation Report", "Monitoring Report", "Audit Report"]
        )
        
        # Display report descriptions
        report_descriptions = {
            "Validation Report": "Model quality, statistical soundness, compliance",
            "Monitoring Report": "Ongoing health checks, drift, thresholds", 
            "Audit Report": "Full lifecycle traceability, change control, approvals"
        }
        st.caption(f"**Focus:** {report_descriptions[report_type]}")
    
    with col2:
        workflow_id = st.selectbox(
            "Workflow",
            ["Current Workflow"] + [f"Workflow {wid[:8]}..." for wid in mcp_engine.active_workflows.keys()],
            help="Select workflow to generate report for"
        )
    
    if st.button("Generate Report", type="primary"):
        if workflow_id == "Current Workflow" and st.session_state.get('current_workflow_id'):
            selected_workflow_id = st.session_state.current_workflow_id
        else:
            # Extract actual workflow ID
            selected_workflow_id = list(mcp_engine.active_workflows.keys())[0] if mcp_engine.active_workflows else None
        
        if selected_workflow_id:
            try:
                workflow_status = mcp_engine.get_workflow_status(selected_workflow_id)
                workflow_results = mcp_engine.get_workflow_results(selected_workflow_id)
                
                # Get sample documents for reference
                sample_loader = st.session_state.get('sample_loader')
                sample_documents = sample_loader.get_sample_documents() if sample_loader else {}
                
                # Check if enhanced reports are enabled
                system_config = st.session_state.get('system_config', {})
                llm_config = system_config.get('llm_config', {})
                use_llm = llm_config.get('enable_enhanced_reports', False)
                
                # Use enhanced report generator
                from src.utils.enhanced_report_generator import EnhancedReportGenerator
                enhanced_generator = EnhancedReportGenerator()
                
                report_content = enhanced_generator.generate_enhanced_report(
                    report_type.lower().replace(' ', '_'), 
                    workflow_results, 
                    sample_documents,
                    use_llm=use_llm
                )
                
                st.success("Report generated successfully!")
                
                # Display report
                st.markdown("### Generated Report")
                st.markdown(report_content)
                
                # Download button
                st.download_button(
                    label="Download Report",
                    data=report_content,
                    file_name=f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"Report generation failed: {str(e)}")
        else:
            st.warning("No workflow selected or available")
    
    # System Status Report
    st.subheader("System Reports")
    
    if st.button("Generate System Status Report"):
        system_report = generate_system_status_report(mcp_engine, audit_logger)
        st.markdown("### System Status Report")
        st.markdown(system_report)
        
        st.download_button(
            label="Download System Report",
            data=system_report,
            file_name=f"system_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

def show_audit_trail(audit_logger):
    """Audit trail interface"""
    st.title("📋 Audit Trail")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        limit = st.number_input("Number of entries", min_value=10, max_value=1000, value=50)
    
    with col2:
        event_filter = st.selectbox(
            "Filter by Event Type",
            ["All", "workflow_event", "data_operation", "agent_execution", "human_interaction"]
        )
    
    with col3:
        if st.button("Refresh"):
            st.rerun()
    
    # Get audit entries
    all_entries = audit_logger.get_audit_trail(limit=limit)
    
    if event_filter != "All":
        filtered_entries = [
            entry for entry in all_entries
            if entry.get('event_type') == event_filter or entry.get('type') == event_filter
        ]
    else:
        filtered_entries = all_entries
    
    # Display audit trail
    st.subheader(f"Audit Entries ({len(filtered_entries)} shown)")
    
    if filtered_entries:
        for i, entry in enumerate(filtered_entries):
            with st.expander(f"Entry {i+1}: {entry.get('event_type', entry.get('operation', 'Unknown'))} - {entry.get('timestamp', 'No timestamp')}"):
                st.json(entry)
    else:
        st.info("No audit entries found")
    
    # Export audit trail
    if st.button("Export Audit Trail"):
        audit_report = generate_audit_summary_report(audit_logger)
        st.download_button(
            label="Download Audit Report",
            data=audit_report,
            file_name=f"audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )



def generate_system_status_report(mcp_engine, audit_logger) -> str:
    """Generate system status report"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""# System Status Report

**Generated:** {timestamp}

## System Overview
- **Active Workflows:** {len(mcp_engine.active_workflows)}
- **Available Agents:** {len(mcp_engine.agents)}
- **Total Audit Entries:** {len(audit_logger.get_audit_trail())}

## Agent Status
"""
    
    for agent_name in mcp_engine.agents.keys():
        report += f"- **{agent_name.title()} Agent:** Ready\n"
    
    report += """
## Recent Activity
"""
    
    recent_activities = audit_logger.get_audit_trail(limit=5)
    for activity in recent_activities:
        timestamp_str = activity.get('timestamp', 'Unknown time')
        if hasattr(timestamp_str, 'strftime'):
            timestamp_str = timestamp_str.strftime('%H:%M:%S')
        
        event_type = activity.get('event_type', activity.get('operation', 'activity'))
        report += f"- **{timestamp_str}:** {event_type}\n"
    
    return report

def generate_audit_summary_report(audit_logger) -> str:
    """Generate audit summary report"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    all_entries = audit_logger.get_audit_trail()
    
    report = f"""# Audit Trail Summary

**Generated:** {timestamp}
**Total Entries:** {len(all_entries)}

## Activity Summary
"""
    
    # Count activities by type
    activity_counts = {}
    for entry in all_entries:
        activity_type = entry.get('event_type', entry.get('operation', 'unknown'))
        activity_counts[activity_type] = activity_counts.get(activity_type, 0) + 1
    
    for activity_type, count in activity_counts.items():
        report += f"- **{activity_type}:** {count} occurrences\n"
    
    report += "\n## Recent Entries\n"
    
    recent_entries = all_entries[-10:] if len(all_entries) > 10 else all_entries
    for i, entry in enumerate(recent_entries, 1):
        timestamp_str = entry.get('timestamp', 'Unknown time')
        if hasattr(timestamp_str, 'strftime'):
            timestamp_str = timestamp_str.strftime('%Y-%m-%d %H:%M:%S')
        
        activity_type = entry.get('event_type', entry.get('operation', 'activity'))
        report += f"{i}. **{timestamp_str}** - {activity_type}\n"
    
    return report

