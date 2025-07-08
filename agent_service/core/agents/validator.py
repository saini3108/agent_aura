import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime

from agent_service.core.agents.base import BaseAgent, AgentError
from agent_service.core.schema.context import BaseContext, AgentRole, WorkflowStatus, ValidationResult
from agent_service.core.services.llm_client import LLMClientManager

logger = logging.getLogger(__name__)

class ValidatorAgent(BaseAgent):
    """Agent responsible for validating workflow results and ensuring compliance"""

    def __init__(self, llm_manager: LLMClientManager):
        super().__init__("validator", AgentRole.VALIDATOR, llm_manager)

        # Validator-specific configuration
        self.config.update({
            "validation_threshold": 0.95,
            "require_all_validations": True,
            "detailed_analysis": True,
            "compliance_checks": True
        })

        # Validation rules by workflow type
        self.validation_rules = {
            "model_validation": self._get_model_validation_rules(),
            "ecl_calculation": self._get_ecl_validation_rules(),
            "rwa_calculation": self._get_rwa_validation_rules(),
            "reporting": self._get_reporting_validation_rules()
        }

    def get_system_prompt(self) -> str:
        """Get system prompt for validator agent"""
        return """
        You are a Banking Workflow Validator Agent. Your role is to validate results and ensure compliance with banking regulations.

        Key Responsibilities:
        - Validate all workflow outputs for accuracy and completeness
        - Check compliance with banking regulations and internal policies
        - Assess risk levels and flag potential issues
        - Verify data quality and consistency
        - Validate mathematical calculations and models
        - Check for regulatory compliance violations
        - Provide detailed validation reports

        Validation Standards:
        - Basel III capital requirements
        - IFRS 9 accounting standards
        - SR 11-7 Model Risk Management
        - Local banking regulations
        - Internal risk management policies

        Validation Process:
        1. Data quality assessment
        2. Calculation accuracy verification
        3. Regulatory compliance check
        4. Risk assessment
        5. Documentation review
        6. Final validation score

        Always be conservative in your assessments and flag any potential issues.
        """

    async def execute(self, context: BaseContext) -> BaseContext:
        """Execute validation logic"""
        try:
            # Update status
            context.update_status(WorkflowStatus.RUNNING)

            # Perform comprehensive validation
            validation_results = await self._perform_validation(context)

            # Analyze validation results
            overall_score = self._calculate_overall_score(validation_results)

            # Update context with validation results
            if hasattr(context, 'validation_results'):
                context.validation_results.extend(validation_results)
            else:
                context.outputs["validation_results"] = validation_results

            # Determine if validation passed
            if overall_score >= self.config["validation_threshold"]:
                context.update_status(WorkflowStatus.AWAITING_REVIEW)
                context.requires_human_review = True
            else:
                context.update_status(WorkflowStatus.FAILED)

            # Add validation summary
            context.add_agent_message(
                self.role,
                f"Validation completed with score: {overall_score:.2f}",
                {
                    "overall_score": overall_score,
                    "validation_count": len(validation_results),
                    "passed_validations": len([r for r in validation_results if r.passed])
                }
            )

            return context

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            context.update_status(WorkflowStatus.FAILED)
            raise AgentError(f"Validation failed: {e}", self.name)

    async def _perform_validation(self, context: BaseContext) -> List[ValidationResult]:
        """Perform comprehensive validation"""

        validation_results = []

        # Get validation rules for workflow type
        rules = self.validation_rules.get(context.workflow_type, [])

        # Execute each validation rule
        for rule in rules:
            try:
                result = await self._execute_validation_rule(rule, context)
                validation_results.append(result)

                # Log validation result
                logger.info(f"Validation {rule['name']}: {'PASSED' if result.passed else 'FAILED'}")

            except Exception as e:
                # Create failed validation result
                result = ValidationResult(
                    validation_type=rule['name'],
                    passed=False,
                    score=0.0,
                    issues=[f"Validation execution failed: {str(e)}"],
                    recommendations=[f"Fix validation error: {str(e)}"]
                )
                validation_results.append(result)

                logger.error(f"Validation rule {rule['name']} failed: {e}")

        # Perform additional context-specific validations
        additional_results = await self._perform_additional_validations(context)
        validation_results.extend(additional_results)

        return validation_results

    async def _execute_validation_rule(self, rule: Dict[str, Any], context: BaseContext) -> ValidationResult:
        """Execute a single validation rule"""

        # Prepare validation prompt
        prompt = self._create_validation_prompt(rule, context)

        # Define validation result schema
        result_schema = {
            "type": "object",
            "properties": {
                "passed": {"type": "boolean"},
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "issues": {"type": "array", "items": {"type": "string"}},
                "recommendations": {"type": "array", "items": {"type": "string"}},
                "details": {"type": "object"},
                "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
                "compliance_status": {"type": "string", "enum": ["compliant", "non-compliant", "needs_review"]}
            },
            "required": ["passed", "score", "issues", "recommendations"]
        }

        # Generate validation result
        response = await self.generate_structured_response(prompt, result_schema, context)

        try:
            result_data = json.loads(response.content)

            return ValidationResult(
                validation_type=rule['name'],
                passed=result_data['passed'],
                score=result_data['score'],
                issues=result_data['issues'],
                recommendations=result_data['recommendations']
            )

        except json.JSONDecodeError as e:
            raise AgentError(f"Invalid validation result format: {e}", self.name)

    def _create_validation_prompt(self, rule: Dict[str, Any], context: BaseContext) -> str:
        """Create validation prompt for a specific rule"""

        # Get relevant data for validation
        tool_results = self._get_tool_results_summary(context)
        workflow_outputs = context.outputs

        prompt = f"""
        Validate the following aspect of the workflow:

        Validation Rule: {rule['name']}
        Description: {rule['description']}
        Criteria: {rule['criteria']}

        Workflow Data:
        - Workflow Type: {context.workflow_type}
        - Tool Results: {tool_results}
        - Workflow Outputs: {json.dumps(workflow_outputs, indent=2)}

        Validation Requirements:
        {rule.get('requirements', 'Standard validation requirements')}

        Please assess:
        1. Whether the rule criteria are met
        2. Data quality and accuracy
        3. Compliance with regulations
        4. Risk assessment
        5. Any issues or concerns
        6. Recommendations for improvement

        Provide a comprehensive validation assessment.
        """

        return prompt

    def _get_tool_results_summary(self, context: BaseContext) -> str:
        """Get summary of tool results for validation"""

        if not context.tool_results:
            return "No tool results available"

        summary = []
        for result in context.tool_results:
            summary.append({
                "tool": result.tool_name,
                "success": result.success,
                "execution_time": result.execution_time,
                "has_outputs": bool(result.outputs),
                "error": result.error_message if result.error_message else None
            })

        return json.dumps(summary, indent=2)

    async def _perform_additional_validations(self, context: BaseContext) -> List[ValidationResult]:
        """Perform additional context-specific validations"""

        additional_results = []

        # Data consistency validation
        consistency_result = await self._validate_data_consistency(context)
        additional_results.append(consistency_result)

        # Calculation accuracy validation
        accuracy_result = await self._validate_calculation_accuracy(context)
        additional_results.append(accuracy_result)

        # Regulatory compliance validation
        compliance_result = await self._validate_regulatory_compliance(context)
        additional_results.append(compliance_result)

        # Risk assessment validation
        risk_result = await self._validate_risk_assessment(context)
        additional_results.append(risk_result)

        return additional_results

    async def _validate_data_consistency(self, context: BaseContext) -> ValidationResult:
        """Validate data consistency across the workflow"""

        prompt = f"""
        Validate data consistency across the workflow:

        Workflow Type: {context.workflow_type}
        Tool Results: {len(context.tool_results)}

        Check for:
        1. Data integrity across all steps
        2. Consistent data formats and types
        3. No missing or corrupted data
        4. Logical consistency in calculations
        5. Proper data lineage

        Assess the overall data consistency and identify any issues.
        """

        # Use generic validation logic
        return await self._execute_generic_validation("data_consistency", prompt, context)

    async def _validate_calculation_accuracy(self, context: BaseContext) -> ValidationResult:
        """Validate calculation accuracy"""

        prompt = f"""
        Validate calculation accuracy in the workflow:

        Workflow Type: {context.workflow_type}

        Check for:
        1. Mathematical accuracy of calculations
        2. Proper use of formulas and models
        3. Correct handling of edge cases
        4. Appropriate rounding and precision
        5. Logical calculation flow

        Verify that all calculations are accurate and appropriate.
        """

        return await self._execute_generic_validation("calculation_accuracy", prompt, context)

    async def _validate_regulatory_compliance(self, context: BaseContext) -> ValidationResult:
        """Validate regulatory compliance"""

        prompt = f"""
        Validate regulatory compliance for the workflow:

        Workflow Type: {context.workflow_type}

        Check compliance with:
        1. Basel III requirements (if applicable)
        2. IFRS 9 standards (if applicable)
        3. Local banking regulations
        4. Internal risk policies
        5. Audit requirements

        Assess compliance status and identify any violations.
        """

        return await self._execute_generic_validation("regulatory_compliance", prompt, context)

    async def _validate_risk_assessment(self, context: BaseContext) -> ValidationResult:
        """Validate risk assessment"""

        prompt = f"""
        Validate risk assessment in the workflow:

        Workflow Type: {context.workflow_type}

        Check:
        1. Proper risk identification
        2. Appropriate risk measurement
        3. Risk mitigation strategies
        4. Conservative assumptions
        5. Scenario analysis (if applicable)

        Assess the quality and completeness of risk assessment.
        """

        return await self._execute_generic_validation("risk_assessment", prompt, context)

    async def _execute_generic_validation(self, validation_type: str, prompt: str, context: BaseContext) -> ValidationResult:
        """Execute a generic validation"""

        # Define validation result schema
        result_schema = {
            "type": "object",
            "properties": {
                "passed": {"type": "boolean"},
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "issues": {"type": "array", "items": {"type": "string"}},
                "recommendations": {"type": "array", "items": {"type": "string"}},
                "details": {"type": "object"}
            },
            "required": ["passed", "score", "issues", "recommendations"]
        }

        # Generate validation result
        response = await self.generate_structured_response(prompt, result_schema, context)

        try:
            result_data = json.loads(response.content)

            return ValidationResult(
                validation_type=validation_type,
                passed=result_data['passed'],
                score=result_data['score'],
                issues=result_data['issues'],
                recommendations=result_data['recommendations']
            )

        except json.JSONDecodeError as e:
            # Return failed validation on parse error
            return ValidationResult(
                validation_type=validation_type,
                passed=False,
                score=0.0,
                issues=[f"Validation result parse error: {str(e)}"],
                recommendations=["Fix validation result format"]
            )

    def _calculate_overall_score(self, validation_results: List[ValidationResult]) -> float:
        """Calculate overall validation score"""

        if not validation_results:
            return 0.0

        # Calculate weighted average
        total_score = sum(result.score for result in validation_results)
        return total_score / len(validation_results)

    def _get_model_validation_rules(self) -> List[Dict[str, Any]]:
        """Get validation rules for model validation workflow"""
        return [
            {
                "name": "model_configuration",
                "description": "Validate model configuration parameters",
                "criteria": "All model parameters are within acceptable ranges",
                "requirements": "Check parameter bounds, data types, and business logic"
            },
            {
                "name": "statistical_tests",
                "description": "Validate statistical test results",
                "criteria": "All statistical tests pass with acceptable p-values",
                "requirements": "Verify test methodology and significance levels"
            },
            {
                "name": "backtesting_results",
                "description": "Validate backtesting performance",
                "criteria": "Model performance meets minimum thresholds",
                "requirements": "Check accuracy, precision, recall, and AUC metrics"
            },
            {
                "name": "documentation_completeness",
                "description": "Validate documentation completeness",
                "criteria": "All required documentation is present and accurate",
                "requirements": "Check model documentation, assumptions, and limitations"
            }
        ]

    def _get_ecl_validation_rules(self) -> List[Dict[str, Any]]:
        """Get validation rules for ECL calculation workflow"""
        return [
            {
                "name": "ifrs9_compliance",
                "description": "Validate IFRS 9 compliance",
                "criteria": "ECL calculations comply with IFRS 9 standards",
                "requirements": "Check staging, calculation methodology, and assumptions"
            },
            {
                "name": "pd_lgd_ead_calculations",
                "description": "Validate PD/LGD/EAD calculations",
                "criteria": "All probability estimates are within valid ranges",
                "requirements": "Check calculation formulas and parameter validity"
            },
            {
                "name": "scenario_analysis",
                "description": "Validate scenario analysis",
                "criteria": "Multiple scenarios are properly weighted",
                "requirements": "Check base, upside, and downside scenarios"
            },
            {
                "name": "data_quality",
                "description": "Validate input data quality",
                "criteria": "Portfolio data is complete and accurate",
                "requirements": "Check data completeness, accuracy, and consistency"
            }
        ]

    def _get_rwa_validation_rules(self) -> List[Dict[str, Any]]:
        """Get validation rules for RWA calculation workflow"""
        return [
            {
                "name": "basel_compliance",
                "description": "Validate Basel III compliance",
                "criteria": "RWA calculations comply with Basel III framework",
                "requirements": "Check standardized approach and risk weights"
            },
            {
                "name": "exposure_classification",
                "description": "Validate exposure classification",
                "criteria": "All exposures are properly classified",
                "requirements": "Check asset classes and risk categories"
            },
            {
                "name": "risk_weight_assignment",
                "description": "Validate risk weight assignment",
                "criteria": "Appropriate risk weights are applied",
                "requirements": "Check risk weight tables and credit ratings"
            },
            {
                "name": "capital_adequacy",
                "description": "Validate capital adequacy ratios",
                "criteria": "Capital ratios meet regulatory minimums",
                "requirements": "Check CET1, Tier 1, and Total capital ratios"
            }
        ]

    def _get_reporting_validation_rules(self) -> List[Dict[str, Any]]:
        """Get validation rules for reporting workflow"""
        return [
            {
                "name": "data_accuracy",
                "description": "Validate report data accuracy",
                "criteria": "All reported data is accurate and complete",
                "requirements": "Check data sources and calculation accuracy"
            },
            {
                "name": "format_compliance",
                "description": "Validate report format compliance",
                "criteria": "Report format meets regulatory requirements",
                "requirements": "Check templates, fields, and data formats"
            },
            {
                "name": "reconciliation",
                "description": "Validate data reconciliation",
                "criteria": "Report data reconciles with source systems",
                "requirements": "Check balance and variance explanations"
            },
            {
                "name": "submission_readiness",
                "description": "Validate submission readiness",
                "criteria": "Report is ready for regulatory submission",
                "requirements": "Check completeness and quality assurance"
            }
        ]
