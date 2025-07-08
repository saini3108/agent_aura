import json
from typing import Dict, Any, List
import logging

from agent_service.core.agents.base import BaseAgent, AgentError
from agent_service.core.schema.context import BaseContext, AgentRole, WorkflowStatus
from agent_service.core.services.llm_client import LLMClientManager

logger = logging.getLogger(__name__)

class PlannerAgent(BaseAgent):
    """Agent responsible for planning workflow execution"""

    def __init__(self, llm_manager: LLMClientManager):
        super().__init__("planner", AgentRole.PLANNER, llm_manager)

        # Planner-specific configuration
        self.config.update({
            "planning_depth": 3,
            "max_steps": 20,
            "include_contingencies": True,
            "risk_assessment": True
        })

    def get_system_prompt(self) -> str:
        """Get system prompt for planner agent"""
        return """
        You are a Banking Workflow Planner Agent. Your role is to analyze banking tasks and create detailed execution plans.

        Key Responsibilities:
        - Analyze the workflow requirements and input data
        - Create step-by-step execution plans
        - Identify required tools and resources
        - Assess risks and compliance requirements
        - Plan for human review checkpoints
        - Create contingency plans for potential failures

        Banking Domain Expertise:
        - IFRS 9 Expected Credit Loss (ECL) calculations
        - Basel III Risk-Weighted Assets (RWA) modeling
        - Model validation and back-testing
        - Regulatory reporting requirements
        - Credit risk assessment
        - Market risk analysis

        Always ensure plans are:
        - Compliant with banking regulations
        - Auditable and transparent
        - Risk-aware and conservative
        - Detailed enough for execution
        """

    async def execute(self, context: BaseContext) -> BaseContext:
        """Execute planning logic"""
        try:
            # Update status
            context.update_status(WorkflowStatus.RUNNING)

            # Generate execution plan
            plan = await self._generate_execution_plan(context)

            # Validate plan
            validated_plan = await self._validate_plan(plan, context)

            # Update context with plan
            context.plan_steps = validated_plan["steps"]
            context.add_agent_message(
                self.role,
                f"Generated execution plan with {len(validated_plan['steps'])} steps",
                {"plan": validated_plan}
            )

            # Set up next steps
            context.current_step = 0
            context.update_status(WorkflowStatus.PLANNED)

            return context

        except Exception as e:
            logger.error(f"Planning failed: {e}")
            context.update_status(WorkflowStatus.FAILED)
            raise AgentError(f"Planning failed: {e}", self.name)

    async def _generate_execution_plan(self, context: BaseContext) -> Dict[str, Any]:
        """Generate detailed execution plan"""

        # Prepare planning prompt
        prompt = self._create_planning_prompt(context)

        # Define plan schema
        plan_schema = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_id": {"type": "string"},
                            "description": {"type": "string"},
                            "agent": {"type": "string"},
                            "tools": {"type": "array", "items": {"type": "string"}},
                            "inputs": {"type": "object"},
                            "expected_outputs": {"type": "object"},
                            "dependencies": {"type": "array", "items": {"type": "string"}},
                            "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
                            "requires_human_review": {"type": "boolean"}
                        },
                        "required": ["step_id", "description", "agent", "tools"]
                    }
                },
                "overall_risk_assessment": {"type": "string"},
                "compliance_requirements": {"type": "array", "items": {"type": "string"}},
                "contingency_plans": {"type": "array", "items": {"type": "object"}},
                "estimated_duration": {"type": "number"},
                "resource_requirements": {"type": "object"}
            },
            "required": ["steps", "overall_risk_assessment", "compliance_requirements"]
        }

        # Generate plan
        response = await self.generate_structured_response(prompt, plan_schema, context)

        try:
            plan = json.loads(response.content)
            return plan
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            raise AgentError(f"Invalid plan format: {e}", self.name)

    def _create_planning_prompt(self, context: BaseContext) -> str:
        """Create planning prompt based on workflow type"""

        base_prompt = f"""
        Create a detailed execution plan for a {context.workflow_type} workflow.

        Workflow Details:
        - Workflow ID: {context.workflow_id}
        - Workflow Type: {context.workflow_type}
        - Inputs: {json.dumps(context.inputs, indent=2)}
        """

        if context.workflow_type == "model_validation":
            specific_prompt = self._get_model_validation_prompt(context)
        elif context.workflow_type == "ecl_calculation":
            specific_prompt = self._get_ecl_calculation_prompt(context)
        elif context.workflow_type == "rwa_calculation":
            specific_prompt = self._get_rwa_calculation_prompt(context)
        elif context.workflow_type == "reporting":
            specific_prompt = self._get_reporting_prompt(context)
        else:
            specific_prompt = self._get_generic_prompt(context)

        return base_prompt + "\n\n" + specific_prompt

    def _get_model_validation_prompt(self, context: BaseContext) -> str:
        """Get model validation specific prompt"""
        return """
        This is a model validation workflow. The plan should include:

        1. Data quality assessment
        2. Model configuration validation
        3. Statistical tests execution
        4. Back-testing procedures
        5. Model performance evaluation
        6. Compliance checks
        7. Documentation generation
        8. Human expert review

        Required tools: validate_model_config, run_statistical_tests, perform_backtesting,
                       generate_validation_report, assess_model_performance

        Compliance: Ensure adherence to SR 11-7 (Model Risk Management), Basel guidelines
        """

    def _get_ecl_calculation_prompt(self, context: BaseContext) -> str:
        """Get ECL calculation specific prompt"""
        return """
        This is an Expected Credit Loss (ECL) calculation workflow. The plan should include:

        1. Portfolio data validation
        2. IFRS 9 parameter setup
        3. PD/LGD/EAD calculations
        4. Scenario analysis (base, upside, downside)
        5. ECL computation
        6. Results validation
        7. Regulatory reporting
        8. Senior management review

        Required tools: validate_portfolio_data, calculate_pd_lgd_ead, run_ifrs9_calculation,
                       perform_scenario_analysis, validate_ecl_results, generate_ecl_report

        Compliance: Ensure adherence to IFRS 9, local banking regulations
        """

    def _get_rwa_calculation_prompt(self, context: BaseContext) -> str:
        """Get RWA calculation specific prompt"""
        return """
        This is a Risk-Weighted Assets (RWA) calculation workflow. The plan should include:

        1. Exposure data validation
        2. Risk weight assignment
        3. Credit risk calculations
        4. Market risk calculations
        5. Operational risk calculations
        6. Capital adequacy assessment
        7. Regulatory reporting
        8. Risk committee review

        Required tools: validate_exposure_data, assign_risk_weights, calculate_credit_rwa,
                       calculate_market_rwa, calculate_operational_rwa, assess_capital_adequacy

        Compliance: Ensure adherence to Basel III, local capital requirements
        """

    def _get_reporting_prompt(self, context: BaseContext) -> str:
        """Get reporting specific prompt"""
        return """
        This is a regulatory reporting workflow. The plan should include:

        1. Data collection and validation
        2. Report template setup
        3. Data aggregation and calculation
        4. Report generation
        5. Quality assurance checks
        6. Regulatory compliance validation
        7. Management review
        8. Submission preparation

        Required tools: collect_report_data, validate_data_quality, generate_report,
                       perform_qa_checks, validate_regulatory_compliance

        Compliance: Ensure adherence to specific regulatory requirements
        """

    def _get_generic_prompt(self, context: BaseContext) -> str:
        """Get generic workflow prompt"""
        return """
        This is a generic banking workflow. The plan should include:

        1. Input validation
        2. Data processing
        3. Calculation execution
        4. Results validation
        5. Output generation
        6. Quality checks
        7. Human review
        8. Final approval

        Required tools: validate_inputs, process_data, perform_calculations, validate_results

        Compliance: Ensure adherence to general banking regulations
        """

    async def _validate_plan(self, plan: Dict[str, Any], context: BaseContext) -> Dict[str, Any]:
        """Validate generated plan"""

        # Basic validation
        if "steps" not in plan:
            raise AgentError("Plan missing required 'steps' field", self.name)

        if not plan["steps"]:
            raise AgentError("Plan cannot be empty", self.name)

        if len(plan["steps"]) > self.config["max_steps"]:
            raise AgentError(f"Plan has too many steps: {len(plan['steps'])}", self.name)

        # Validate each step
        for i, step in enumerate(plan["steps"]):
            if "step_id" not in step:
                raise AgentError(f"Step {i} missing step_id", self.name)

            if "description" not in step:
                raise AgentError(f"Step {i} missing description", self.name)

            if "agent" not in step:
                raise AgentError(f"Step {i} missing agent", self.name)

            if "tools" not in step:
                raise AgentError(f"Step {i} missing tools", self.name)

        # Validate dependencies
        step_ids = {step["step_id"] for step in plan["steps"]}
        for step in plan["steps"]:
            if "dependencies" in step:
                for dep in step["dependencies"]:
                    if dep not in step_ids:
                        raise AgentError(f"Invalid dependency: {dep}", self.name)

        # Log validation success
        logger.info(f"Plan validated successfully with {len(plan['steps'])} steps")

        return plan

    async def revise_plan(self, context: BaseContext, feedback: str) -> BaseContext:
        """Revise plan based on feedback"""
        try:
            # Create revision prompt
            prompt = f"""
            Revise the current execution plan based on the following feedback:

            Feedback: {feedback}

            Current Plan: {json.dumps(context.plan_steps, indent=2)}

            Provide an updated plan that addresses the feedback while maintaining
            compliance and risk management standards.
            """

            # Generate revised plan
            revised_plan = await self._generate_execution_plan(context)

            # Validate revised plan
            validated_plan = await self._validate_plan(revised_plan, context)

            # Update context
            context.plan_steps = validated_plan["steps"]
            context.add_agent_message(
                self.role,
                f"Revised execution plan based on feedback",
                {"revised_plan": validated_plan, "feedback": feedback}
            )

            return context

        except Exception as e:
            logger.error(f"Plan revision failed: {e}")
            raise AgentError(f"Plan revision failed: {e}", self.name)
