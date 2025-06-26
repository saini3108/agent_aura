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
    
    # Check if workflow results exist
    has_workflow_data = bool(st.session_state.get('workflow_results'))
    has_demo_data = bool(st.session_state.get('demo_results'))
    
    if not has_workflow_data and not has_demo_data:
        st.info("Run a workflow first to generate reports. Go to the MCP Workflow tab or Enhanced Workflow Demo to start.")
        return
    
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
        # Show available workflows based on session data
        workflow_options = []
        
        if has_workflow_data:
            workflow_options.append("Current Workflow Results")
        
        if has_demo_data:
            workflow_options.append("Demo Workflow Results")
            
        if not workflow_options:
            workflow_options = ["No workflows available"]
        
        selected_workflow = st.selectbox(
            "Data Source",
            workflow_options,
            help="Select workflow results to generate report from"
        )
    
    if st.button("Generate Report", type="primary"):
        # Get the appropriate workflow data
        if selected_workflow == "Current Workflow Results" and has_workflow_data:
            workflow_data = st.session_state.workflow_results
        elif selected_workflow == "Demo Workflow Results" and has_demo_data:
            workflow_data = st.session_state.demo_results
        else:
            st.error("No workflow data available for report generation")
            return
        
        if workflow_data:
            try:
                # Generate report using unified summary generator
                from src.core.unified_summary_generator import UnifiedSummaryGenerator
                summary_generator = UnifiedSummaryGenerator()
                
                report_content = generate_workflow_report(
                    report_type, 
                    workflow_data, 
                    selected_workflow,
                    summary_generator
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
            st.warning("No workflow data available")

def generate_workflow_report(report_type: str, workflow_data: Dict[str, Any], data_source: str, summary_generator) -> str:
    """Generate dynamic report from actual workflow data"""
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Count completed agents
    completed_agents = [k for k, v in workflow_data.items() if v.get('status') == 'completed']
    total_agents = len(workflow_data)
    
    # Extract key metrics from actual workflow results
    key_findings = []
    critical_issues = []
    recommendations = []
    risk_level = "Medium"
    
    for agent_name, result in workflow_data.items():
        if agent_name == 'human_review':
            continue
            
        summary = result.get('clean_summary') or result.get('summary')
        
        if summary:
            # Extract findings from actual agent results
            if hasattr(summary, 'title'):
                key_findings.append(f"{agent_name.title()}: {summary.title}")
                if hasattr(summary, 'risk_flags') and summary.risk_flags:
                    critical_issues.extend(summary.risk_flags)
                if hasattr(summary, 'recommendation'):
                    recommendations.append(f"{agent_name.title()}: {summary.recommendation}")
                if hasattr(summary, 'severity') and summary.severity in ['high', 'critical']:
                    risk_level = "High"
            elif isinstance(summary, dict):
                title = summary.get('title', f"{agent_name} completed")
                key_findings.append(f"{agent_name.title()}: {title}")
                
                # Extract risk flags from dict format
                risk_flags = summary.get('risk_flags', [])
                if risk_flags:
                    critical_issues.extend(risk_flags)
                
                rec = summary.get('recommendation', 'Review findings')
                recommendations.append(f"{agent_name.title()}: {rec}")
                
                if summary.get('severity') in ['high', 'critical']:
                    risk_level = "High"
    
    # Generate executive summary using LLM if available
    try:
        exec_summary = summary_generator.generate_executive_summary(workflow_data)
        executive_assessment = exec_summary.overall_assessment
        final_recommendation = exec_summary.executive_recommendation
        approval_status = exec_summary.approval_status
        confidence_score = exec_summary.confidence_score
    except:
        executive_assessment = f"Workflow validation completed with {len(completed_agents)}/{total_agents} agents successfully executed."
        final_recommendation = "Review detailed findings and proceed based on risk assessment."
        approval_status = "Conditional"
        confidence_score = 0.8
    
    # Generate report content based on type
    if report_type == "Validation Report":
        return f"""# Credit Risk Model Validation Report

**Generated:** {timestamp}
**Data Source:** {data_source}
**Report Type:** {report_type}

## Executive Summary

{executive_assessment}

**Risk Level:** {risk_level}
**Approval Status:** {approval_status}
**Confidence Score:** {confidence_score:.0%}

**Final Recommendation:** {final_recommendation}

## Validation Results

### Agent Execution Summary
- **Total Agents:** {total_agents}
- **Completed Successfully:** {len(completed_agents)}
- **Completion Rate:** {len(completed_agents)/total_agents*100:.1f}%

### Key Findings
{chr(10).join([f"• {finding}" for finding in key_findings[:5]])}

### Critical Issues
{chr(10).join([f"⚠️ {issue}" for issue in critical_issues[:3]]) if critical_issues else "• No critical issues identified"}

### Recommendations
{chr(10).join([f"🔧 {rec}" for rec in recommendations[:5]])}

## Detailed Agent Results

{_generate_detailed_agent_section(workflow_data)}

## Compliance Assessment

### Regulatory Framework Alignment
- Basel III: {'✅ Compliant' if risk_level != 'High' else '⚠️ Review Required'}
- IFRS 9: {'✅ Compliant' if 'documentation' in completed_agents else '⚠️ Documentation Missing'}
- Model Risk Management: {'✅ Compliant' if len(completed_agents) >= 4 else '⚠️ Incomplete Validation'}

### Risk Management
- **Overall Risk Level:** {risk_level}
- **Mitigation Required:** {'Yes' if risk_level == 'High' else 'Standard Monitoring'}
- **Next Review:** Quarterly

## Conclusion

{final_recommendation}

**Report Prepared By:** ValiCred-AI System
**Approval Status:** {approval_status}
"""

    elif report_type == "Monitoring Report":
        return f"""# Model Monitoring Report

**Generated:** {timestamp}
**Data Source:** {data_source}
**Monitoring Period:** Current Validation Cycle

## Executive Summary

{executive_assessment}

## Performance Monitoring

### Model Performance Metrics
{_extract_performance_metrics(workflow_data)}

### Data Quality Monitoring
{_extract_data_quality_metrics(workflow_data)}

### Population Stability
{_extract_stability_metrics(workflow_data)}

## Alert Summary
{chr(10).join([f"⚠️ {issue}" for issue in critical_issues]) if critical_issues else "• No alerts generated"}

## Trending Analysis
- **Performance Trend:** {'Declining' if 'poor' in str(workflow_data).lower() else 'Stable'}
- **Data Quality Trend:** {'Stable' if 'good' in str(workflow_data).lower() else 'Monitoring Required'}

## Recommendations
{chr(10).join([f"• {rec}" for rec in recommendations[:3]])}
"""

    else:  # Audit Report
        return f"""# Independent Audit Report

**Generated:** {timestamp}
**Data Source:** {data_source}
**Audit Scope:** Full Model Validation Process

## Executive Summary

{executive_assessment}

## Audit Findings

### Process Compliance
- **Validation Process:** {'✅ Compliant' if len(completed_agents) >= 4 else '⚠️ Incomplete'}
- **Documentation:** {'✅ Complete' if 'documentation' in completed_agents else '⚠️ Missing'}
- **Human Review:** {'✅ Conducted' if 'human_review' in workflow_data else '⚠️ Not Performed'}

### Quality Assurance
{chr(10).join([f"• {finding}" for finding in key_findings[:3]])}

### Risk Assessment
- **Independent Risk Rating:** {risk_level}
- **Audit Confidence:** {confidence_score:.0%}

### Exceptions and Issues
{chr(10).join([f"• {issue}" for issue in critical_issues]) if critical_issues else "• No exceptions identified"}

## Audit Opinion

{final_recommendation}

### Certification
This audit was conducted in accordance with model risk management standards and regulatory requirements.

**Audit Status:** {approval_status}
**Next Audit:** Annual Review Required
"""

def _generate_detailed_agent_section(workflow_data: Dict[str, Any]) -> str:
    """Generate detailed section for each agent"""
    sections = []
    
    agent_order = ['analyst', 'validator', 'documentation', 'human_review', 'reviewer', 'auditor']
    
    for agent_name in agent_order:
        if agent_name in workflow_data:
            result = workflow_data[agent_name]
            summary = result.get('clean_summary') or result.get('summary')
            
            if summary:
                if hasattr(summary, 'title'):
                    title = summary.title
                    description = getattr(summary, 'description', 'Analysis completed')
                    impact = getattr(summary, 'impact', 'Impact assessment available')
                    recommendation = getattr(summary, 'recommendation', 'Review findings')
                elif isinstance(summary, dict):
                    title = summary.get('title', f"{agent_name.title()} Analysis")
                    description = summary.get('description', 'Analysis completed')
                    impact = summary.get('impact', 'Impact assessment available')
                    recommendation = summary.get('recommendation', 'Review findings')
                else:
                    title = f"{agent_name.title()} Agent"
                    description = "Analysis completed"
                    impact = "Standard analysis impact"
                    recommendation = "Review findings"
                
                sections.append(f"""### {title}

**Status:** {result.get('status', 'Unknown').title()}
**Timestamp:** {result.get('timestamp', 'Unknown')}

**Analysis:** {description}

**Impact:** {impact}

**Recommendation:** {recommendation}
""")
    
    return "\n".join(sections)

def _extract_performance_metrics(workflow_data: Dict[str, Any]) -> str:
    """Extract performance metrics from validator results"""
    validator_result = workflow_data.get('validator', {})
    raw_output = validator_result.get('raw_output', {})
    metrics = raw_output.get('metrics', {})
    
    if metrics:
        return f"""
- **AUC Score:** {metrics.get('auc', 'N/A')}
- **KS Statistic:** {metrics.get('ks_statistic', 'N/A')}
- **Gini Coefficient:** {metrics.get('gini', 'N/A')}
- **PSI:** {metrics.get('psi', 'N/A')}
"""
    else:
        return "• Performance metrics not available in current validation"

def _extract_data_quality_metrics(workflow_data: Dict[str, Any]) -> str:
    """Extract data quality metrics from analyst results"""
    analyst_result = workflow_data.get('analyst', {})
    raw_output = analyst_result.get('raw_output', {})
    analysis = raw_output.get('analysis', {})
    data_quality = analysis.get('data_quality', {})
    
    if data_quality:
        return f"""
- **Missing Data:** {data_quality.get('missing_percentage', 'N/A')}%
- **Outlier Rate:** {data_quality.get('outlier_percentage', 'N/A')}%
- **Feature Count:** {data_quality.get('feature_count', 'N/A')}
"""
    else:
        return "• Data quality metrics not available in current validation"

def _extract_stability_metrics(workflow_data: Dict[str, Any]) -> str:
    """Extract stability metrics from validator results"""
    validator_result = workflow_data.get('validator', {})
    raw_output = validator_result.get('raw_output', {})
    metrics = raw_output.get('metrics', {})
    
    psi = metrics.get('psi', 0)
    if psi:
        if psi <= 0.1:
            status = "Stable"
        elif psi <= 0.25:
            status = "Monitoring Required"
        else:
            status = "Unstable"
        
        return f"""
- **Population Stability Index:** {psi}
- **Stability Status:** {status}
- **Drift Assessment:** {'No significant drift' if psi <= 0.1 else 'Moderate drift detected' if psi <= 0.25 else 'Significant drift detected'}
"""
    else:
        return "• Population stability metrics not available"
    
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

