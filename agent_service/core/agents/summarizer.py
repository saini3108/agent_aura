import json
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from agent_service.core.agents.base import BaseAgent, AgentError
from agent_service.core.schema.context import BaseContext, AgentRole, WorkflowStatus
from agent_service.core.services.llm_client import LLMClientManager

logger = logging.getLogger(__name__)

class SummarizerAgent(BaseAgent):
    """Agent responsible for summarizing workflow results and generating reports"""

    def __init__(self, llm_manager: LLMClientManager):
        super().__init__("summarizer", AgentRole.SUMMARIZER, llm_manager)

        # Summarizer-specific configuration
        self.config.update({
            "include_technical_details": True,
            "include_risk_assessment": True,
            "include_recommendations": True,
            "executive_summary": True,
            "detailed_findings": True
        })

    def get_system_prompt(self) -> str:
        """Get system prompt for summarizer agent"""
        return """
        You are a Banking Workflow Summarizer Agent. Your role is to create comprehensive summaries and reports of workflow execution.

        Key Responsibilities:
        - Analyze workflow execution results
        - Create executive summaries for management
        - Generate detailed technical reports
        - Highlight key findings and insights
        - Provide actionable recommendations
        - Assess risk implications
        - Ensure regulatory compliance in reporting

        Report Structure:
        1. Executive Summary
        2. Workflow Overview
        3. Key Findings
        4. Risk Assessment
        5. Validation Results
        6. Recommendations
        7. Technical Details
        8. Compliance Status

        Writing Style:
        - Clear and concise for executives
        - Technically accurate for specialists
        - Risk-focused and conservative
        - Compliant with banking standards
        - Actionable and practical
        """

    async def execute(self, context: BaseContext) -> BaseContext:
        """Execute summarization logic"""
        try:
            # Update status
            context.update_status(WorkflowStatus.RUNNING)

            # Generate comprehensive summary
            summary_report = await self._generate_summary_report(context)

            # Create executive summary
            executive_summary = await self._generate_executive_summary(context)

            # Generate recommendations
            recommendations = await self._generate_recommendations(context)

            # Update context with summaries
            context.outputs.update({
                "summary_report": summary_report,
                "executive_summary": executive_summary,
                "recommendations": recommendations,
                "workflow_metrics": self._calculate_workflow_metrics(context)
            })

            # Add summarizer message
            context.add_agent_message(
                self.role,
                "Workflow summarization completed",
                {
                    "summary_sections": len(summary_report.get("sections", [])),
                    "recommendations_count": len(recommendations),
                    "executive_summary_length": len(executive_summary.get("content", ""))
                }
            )

            # Update status
            context.update_status(WorkflowStatus.COMPLETE)

            return context

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            context.update_status(WorkflowStatus.FAILED)
            raise AgentError(f"Summarization failed: {e}", self.name)

    async def _generate_summary_report(self, context: BaseContext) -> Dict[str, Any]:
        """Generate comprehensive summary report"""

        # Prepare summary prompt
        prompt = f"""
        Generate a comprehensive summary report for the completed workflow:

        Workflow Details:
        - Workflow ID: {context.workflow_id}
        - Workflow Type: {context.workflow_type}
        - Status: {context.status.value}
        - Duration: {self._calculate_duration(context)}
        - Steps Executed: {len(context.plan_steps)}
        - Tools Used: {len(context.tool_results)}

        Execution Summary:
        {self._get_execution_summary(context)}

        Tool Results:
        {self._get_tool_results_summary(context)}

        Validation Results:
        {self._get_validation_summary(context)}

        Human Interactions:
        {self._get_human_interaction_summary(context)}

        Create a detailed report with the following sections:
        1. Workflow Overview
        2. Execution Summary
        3. Key Findings
        4. Risk Assessment
        5. Validation Results
        6. Technical Details
        7. Compliance Status
        8. Next Steps
        """

        # Define report schema
        report_schema = {
            "type": "object",
            "properties": {
                "workflow_overview": {
                    "type": "object",
                    "properties": {
                        "workflow_id": {"type": "string"},
                        "workflow_type": {"type": "string"},
                        "execution_date": {"type": "string"},
                        "duration": {"type": "string"},
                        "status": {"type": "string"},
                        "summary": {"type": "string"}
                    }
                },
                "execution_summary": {
                    "type": "object",
                    "properties": {
                        "steps_completed": {"type": "integer"},
                        "tools_executed": {"type": "integer"},
                        "success_rate": {"type": "number"},
                        "issues_encountered": {"type": "array", "items": {"type": "string"}},
                        "performance_metrics": {"type": "object"}
                    }
                },
                "key_findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "finding": {"type": "string"},
                            "importance": {"type": "string"},
                            "details": {"type": "string"}
                        }
                    }
                },
                "risk_assessment": {
                    "type": "object",
                    "properties": {
                        "overall_risk_level": {"type": "string"},
                        "identified_risks": {"type": "array", "items": {"type": "string"}},
                        "risk_mitigation": {"type": "array", "items": {"type": "string"}},
                        "risk_score": {"type": "number"}
                    }
                },
                "validation_results": {
                    "type": "object",
                    "properties": {
                        "overall_validation_score": {"type": "number"},
                        "validations_passed": {"type": "integer"},
                        "validations_failed": {"type": "integer"},
                        "critical_issues": {"type": "array", "items": {"type": "string"}},
                        "recommendations": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "compliance_status": {
                    "type": "object",
                    "properties": {
                        "regulatory_compliance": {"type": "string"},
                        "compliance_issues": {"type": "array", "items": {"type": "string"}},
                        "compliance_score": {"type": "number"}
                    }
                },
                "next_steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "priority": {"type": "string"},
                            "responsible_party": {"type": "string"},
                            "deadline": {"type": "string"}
                        }
                    }
                }
            },
            "required": ["workflow_overview", "execution_summary", "key_findings", "risk_assessment"]
        }

        # Generate report
        response = await self.generate_structured_response(prompt, report_schema, context)

        try:
            report = json.loads(response.content)
            return report
        except json.JSONDecodeError as e:
            raise AgentError(f"Invalid report format: {e}", self.name)

    async def _generate_executive_summary(self, context: BaseContext) -> Dict[str, Any]:
        """Generate executive summary"""

        prompt = f"""
        Create an executive summary for senior management:

        Workflow: {context.workflow_type}
        Status: {context.status.value}
        Duration: {self._calculate_duration(context)}

        Key Points to Cover:
        1. What was accomplished
        2. Key results and findings
        3. Risk assessment
        4. Compliance status
        5. Action items requiring management attention
        6. Business impact

        Keep the summary concise (2-3 paragraphs) and focused on business impact.
        Use clear, non-technical language suitable for executives.
        """

        # Define executive summary schema
        summary_schema = {
            "type": "object",
            "properties": {
                "headline": {"type": "string"},
                "content": {"type": "string"},
                "key_metrics": {
                    "type": "object",
                    "properties": {
                        "success_rate": {"type": "string"},
                        "completion_time": {"type": "string"},
                        "risk_level": {"type": "string"},
                        "compliance_status": {"type": "string"}
                    }
                },
                "management_actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "urgency": {"type": "string"},
                            "impact": {"type": "string"}
                        }
                    }
                },
                "business_impact": {"type": "string"}
            },
            "required": ["headline", "content", "key_metrics"]
        }

        # Generate summary
        response = await self.generate_structured_response(prompt, summary_schema, context)

        try:
            summary = json.loads(response.content)
            return summary
        except json.JSONDecodeError as e:
            raise AgentError(f"Invalid executive summary format: {e}", self.name)

    async def _generate_recommendations(self, context: BaseContext) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""

        prompt = f"""
        Generate actionable recommendations based on the workflow results:

        Workflow: {context.workflow_type}
        Status: {context.status.value}

        Analysis Points:
        {self._get_analysis_points(context)}

        Provide specific, actionable recommendations in the following categories:
        1. Immediate Actions (next 24-48 hours)
        2. Short-term Improvements (next 1-4 weeks)
        3. Long-term Enhancements (next 1-6 months)
        4. Process Improvements
        5. Risk Mitigation
        6. Compliance Actions

        Each recommendation should include:
        - Clear action description
        - Business justification
        - Priority level
        - Responsible party
        - Expected timeline
        - Success metrics
        """

        # Define recommendations schema
        recommendations_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "action": {"type": "string"},
                    "justification": {"type": "string"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    "responsible_party": {"type": "string"},
                    "timeline": {"type": "string"},
                    "success_metrics": {"type": "array", "items": {"type": "string"}},
                    "risk_impact": {"type": "string"}
                },
                "required": ["category", "action", "justification", "priority", "timeline"]
            }
        }

        # Generate recommendations
        response = await self.generate_structured_response(prompt, recommendations_schema, context)

        try:
            recommendations = json.loads(response.content)
            return recommendations
        except json.JSONDecodeError as e:
            raise AgentError(f"Invalid recommendations format: {e}", self.name)

    def _get_execution_summary(self, context: BaseContext) -> str:
        """Get execution summary text"""

        successful_tools = len([r for r in context.tool_results if r.success])
        total_tools = len(context.tool_results)

        summary = f"""
        - Total Steps Planned: {len(context.plan_steps)}
        - Current Step: {context.current_step + 1}
        - Tools Executed: {total_tools}
        - Successful Tools: {successful_tools}
        - Success Rate: {(successful_tools / total_tools * 100) if total_tools > 0 else 0:.1f}%
        - Agent Messages: {len(context.agent_messages)}
        """

        return summary

    def _get_tool_results_summary(self, context: BaseContext) -> str:
        """Get tool results summary"""

        if not context.tool_results:
            return "No tool results available"

        summary = []
        for result in context.tool_results:
            summary.append({
                "tool": result.tool_name,
                "success": result.success,
                "execution_time": f"{result.execution_time:.2f}s",
                "error": result.error_message if result.error_message else None
            })

        return json.dumps(summary, indent=2)

    def _get_validation_summary(self, context: BaseContext) -> str:
        """Get validation summary"""

        if hasattr(context, 'validation_results') and context.validation_results:
            passed = len([r for r in context.validation_results if r.passed])
            total = len(context.validation_results)

            summary = f"""
            - Total Validations: {total}
            - Passed: {passed}
            - Failed: {total - passed}
            - Success Rate: {(passed / total * 100) if total > 0 else 0:.1f}%
            """

            return summary

        return "No validation results available"

    def _get_human_interaction_summary(self, context: BaseContext) -> str:
        """Get human interaction summary"""

        if context.human_feedback:
            return f"""
            - Human Feedback: {context.human_feedback.action}
            - Comments: {context.human_feedback.comments or 'None'}
            - Timestamp: {context.human_feedback.timestamp}
            """

        return "No human interactions"

    def _get_analysis_points(self, context: BaseContext) -> str:
        """Get analysis points for recommendations"""

        points = []

        # Tool performance analysis
        if context.tool_results:
            avg_execution_time = sum(r.execution_time for r in context.tool_results) / len(context.tool_results)
            points.append(f"Average tool execution time: {avg_execution_time:.2f}s")

            failed_tools = [r for r in context.tool_results if not r.success]
            if failed_tools:
                points.append(f"Failed tools: {[r.tool_name for r in failed_tools]}")

        # Validation analysis
        if hasattr(context, 'validation_results') and context.validation_results:
            failed_validations = [r for r in context.validation_results if not r.passed]
            if failed_validations:
                points.append(f"Failed validations: {[r.validation_type for r in failed_validations]}")

        # Status analysis
        if context.status == WorkflowStatus.FAILED:
            points.append("Workflow failed - requires immediate attention")
        elif context.status == WorkflowStatus.AWAITING_REVIEW:
            points.append("Workflow awaiting human review")

        return "\n".join(f"- {point}" for point in points)

    def _calculate_duration(self, context: BaseContext) -> str:
        """Calculate workflow duration"""

        if context.completed_at:
            duration = (context.completed_at - context.created_at).total_seconds()
        else:
            duration = (datetime.utcnow() - context.created_at).total_seconds()

        if duration < 60:
            return f"{duration:.1f} seconds"
        elif duration < 3600:
            return f"{duration/60:.1f} minutes"
        else:
            return f"{duration/3600:.1f} hours"

    def _calculate_workflow_metrics(self, context: BaseContext) -> Dict[str, Any]:
        """Calculate workflow metrics"""

        metrics = {
            "duration": self._calculate_duration(context),
            "steps_completed": context.current_step + 1,
            "total_steps": len(context.plan_steps),
            "completion_rate": ((context.current_step + 1) / len(context.plan_steps) * 100) if context.plan_steps else 0,
            "tools_executed": len(context.tool_results),
            "successful_tools": len([r for r in context.tool_results if r.success]),
            "tool_success_rate": (len([r for r in context.tool_results if r.success]) / len(context.tool_results) * 100) if context.tool_results else 0,
            "agent_messages": len(context.agent_messages),
            "human_interventions": 1 if context.human_feedback else 0
        }

        # Add validation metrics if available
        if hasattr(context, 'validation_results') and context.validation_results:
            metrics.update({
                "validations_run": len(context.validation_results),
                "validations_passed": len([r for r in context.validation_results if r.passed]),
                "validation_success_rate": (len([r for r in context.validation_results if r.passed]) / len(context.validation_results) * 100)
            })

        return metrics

    async def generate_custom_report(self, context: BaseContext, report_type: str, template: Dict[str, Any]) -> Dict[str, Any]:
        """Generate custom report based on template"""

        try:
            prompt = f"""
            Generate a custom {report_type} report using the provided template:

            Template: {json.dumps(template, indent=2)}

            Workflow Data:
            - Workflow ID: {context.workflow_id}
            - Workflow Type: {context.workflow_type}
            - Status: {context.status.value}
            - Execution Summary: {self._get_execution_summary(context)}
            - Tool Results: {self._get_tool_results_summary(context)}

            Follow the template structure and populate with relevant data.
            """

            # Generate custom report
            response = await self.generate_structured_response(prompt, template, context)

            try:
                report = json.loads(response.content)
                return report
            except json.JSONDecodeError as e:
                raise AgentError(f"Invalid custom report format: {e}", self.name)

        except Exception as e:
            logger.error(f"Custom report generation failed: {e}")
            raise AgentError(f"Custom report generation failed: {e}", self.name)
