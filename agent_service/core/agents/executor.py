import json
import asyncio
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from agent_service.core.agents.base import BaseAgent, AgentError
from agent_service.core.schema.context import BaseContext, AgentRole, WorkflowStatus, ToolResult
from agent_service.core.services.llm_client import LLMClientManager
from agent_service.core.tools.banking import BankingToolkit
from agent_service.core.tools.validation import ValidationToolkit

logger = logging.getLogger(__name__)

class ExecutorAgent(BaseAgent):
    """Agent responsible for executing workflow steps using tools"""

    def __init__(self, llm_manager: LLMClientManager):
        super().__init__("executor", AgentRole.EXECUTOR, llm_manager)

        # Initialize tool kits
        self.banking_toolkit = BankingToolkit()
        self.validation_toolkit = ValidationToolkit()

        # Executor-specific configuration
        self.config.update({
            "parallel_execution": False,
            "tool_timeout": 300,  # 5 minutes
            "max_tool_retries": 3,
            "continue_on_error": False
        })

    def get_system_prompt(self) -> str:
        """Get system prompt for executor agent"""
        return """
        You are a Banking Workflow Executor Agent. Your role is to execute planned workflow steps using available tools.

        Key Responsibilities:
        - Execute workflow steps according to the plan
        - Use appropriate tools for each task
        - Handle tool inputs and outputs correctly
        - Manage execution flow and dependencies
        - Report execution status and results
        - Handle errors and implement retry logic

        Available Tool Categories:
        - Banking calculations (ECL, RWA, model validation)
        - Data validation and quality checks
        - Report generation and formatting
        - Statistical analysis and testing
        - Regulatory compliance checks

        Execution Principles:
        - Follow the planned execution order
        - Validate tool inputs before execution
        - Handle tool errors gracefully
        - Provide detailed execution logs
        - Ensure data integrity throughout
        """

    async def execute(self, context: BaseContext) -> BaseContext:
        """Execute workflow steps"""
        try:
            # Update status
            context.update_status(WorkflowStatus.RUNNING)

            # Execute all planned steps
            await self._execute_workflow_steps(context)

            # Update status based on execution results
            if self._all_steps_successful(context):
                context.update_status(WorkflowStatus.AWAITING_REVIEW)
            else:
                context.update_status(WorkflowStatus.FAILED)

            return context

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            context.update_status(WorkflowStatus.FAILED)
            raise AgentError(f"Execution failed: {e}", self.name)

    async def _execute_workflow_steps(self, context: BaseContext) -> None:
        """Execute all workflow steps in order"""

        total_steps = len(context.plan_steps)

        for i, step in enumerate(context.plan_steps):
            try:
                # Update current step
                context.current_step = i

                # Log step start
                logger.info(f"Executing step {i+1}/{total_steps}: {step}")

                # Execute step
                await self._execute_step(step, context)

                # Log step completion
                logger.info(f"Completed step {i+1}/{total_steps}")

            except Exception as e:
                logger.error(f"Step {i+1} failed: {e}")

                # Add error to context
                context.add_agent_message(
                    self.role,
                    f"Step {i+1} failed: {str(e)}",
                    {"step": step, "error": str(e)}
                )

                # Handle error based on configuration
                if not self.config["continue_on_error"]:
                    raise AgentError(f"Step {i+1} failed: {e}", self.name)

                # Continue with next step if configured to do so
                continue

    async def _execute_step(self, step: str, context: BaseContext) -> None:
        """Execute a single workflow step"""

        # Parse step (assuming it's a JSON string with step details)
        try:
            if isinstance(step, str):
                step_data = json.loads(step)
            else:
                step_data = step
        except json.JSONDecodeError:
            # If step is just a string description, create basic step data
            step_data = {
                "step_id": f"step_{context.current_step}",
                "description": step,
                "tools": [],
                "inputs": {}
            }

        # Determine required tools
        tools_to_execute = step_data.get("tools", [])

        # Execute tools
        for tool_name in tools_to_execute:
            await self._execute_tool(tool_name, step_data, context)

    async def _execute_tool(self, tool_name: str, step_data: Dict[str, Any], context: BaseContext) -> None:
        """Execute a specific tool"""

        start_time = datetime.utcnow()

        try:
            # Get tool inputs
            tool_inputs = self._prepare_tool_inputs(tool_name, step_data, context)

            # Execute tool with timeout
            tool_result = await asyncio.wait_for(
                self._call_tool(tool_name, tool_inputs, context),
                timeout=self.config["tool_timeout"]
            )

            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()

            # Create tool result
            result = ToolResult(
                tool_name=tool_name,
                inputs=tool_inputs,
                outputs=tool_result,
                execution_time=execution_time,
                success=True
            )

            # Add to context
            context.add_tool_result(result)

            # Log successful execution
            logger.info(f"Tool {tool_name} executed successfully in {execution_time:.2f}s")

        except asyncio.TimeoutError:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result = ToolResult(
                tool_name=tool_name,
                inputs=tool_inputs,
                outputs={},
                execution_time=execution_time,
                success=False,
                error_message=f"Tool execution timed out after {self.config['tool_timeout']}s"
            )

            context.add_tool_result(result)
            raise AgentError(f"Tool {tool_name} timed out", self.name)

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result = ToolResult(
                tool_name=tool_name,
                inputs=tool_inputs,
                outputs={},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )

            context.add_tool_result(result)
            raise AgentError(f"Tool {tool_name} failed: {e}", self.name)

    def _prepare_tool_inputs(self, tool_name: str, step_data: Dict[str, Any], context: BaseContext) -> Dict[str, Any]:
        """Prepare inputs for tool execution"""

        # Start with step inputs
        tool_inputs = step_data.get("inputs", {}).copy()

        # Add context data
        tool_inputs["workflow_id"] = context.workflow_id
        tool_inputs["workflow_type"] = context.workflow_type

        # Add workflow-specific inputs
        if hasattr(context, 'model_name'):
            tool_inputs["model_name"] = context.model_name

        if hasattr(context, 'portfolio_data'):
            tool_inputs["portfolio_data"] = context.portfolio_data

        # Add results from previous tools
        for tool_result in context.tool_results:
            if tool_result.success and tool_result.outputs:
                # Add outputs with tool name prefix
                for key, value in tool_result.outputs.items():
                    tool_inputs[f"{tool_result.tool_name}_{key}"] = value

        return tool_inputs

    async def _call_tool(self, tool_name: str, tool_inputs: Dict[str, Any], context: BaseContext) -> Dict[str, Any]:
        """Call the specified tool"""

        # Banking tools
        if hasattr(self.banking_toolkit, tool_name):
            tool_func = getattr(self.banking_toolkit, tool_name)
            return await tool_func(tool_inputs)

        # Validation tools
        if hasattr(self.validation_toolkit, tool_name):
            tool_func = getattr(self.validation_toolkit, tool_name)
            return await tool_func(tool_inputs)

        # Generic tools
        if tool_name in self._get_generic_tools():
            return await self._execute_generic_tool(tool_name, tool_inputs, context)

        # Unknown tool
        raise AgentError(f"Unknown tool: {tool_name}", self.name)

    def _get_generic_tools(self) -> List[str]:
        """Get list of generic tools"""
        return [
            "validate_inputs",
            "process_data",
            "perform_calculations",
            "validate_results",
            "generate_summary",
            "format_output"
        ]

    async def _execute_generic_tool(self, tool_name: str, tool_inputs: Dict[str, Any], context: BaseContext) -> Dict[str, Any]:
        """Execute generic tool using LLM"""

        # Create tool execution prompt
        prompt = f"""
        Execute the following tool: {tool_name}

        Tool Inputs:
        {json.dumps(tool_inputs, indent=2)}

        Context:
        - Workflow Type: {context.workflow_type}
        - Current Step: {context.current_step + 1}
        - Previous Results: {len(context.tool_results)} tools executed

        Instructions:
        1. Process the inputs according to the tool's purpose
        2. Perform the required calculations or operations
        3. Validate results for accuracy and compliance
        4. Return structured output with all necessary data
        5. Include any warnings or recommendations

        Return the results in JSON format.
        """

        # Define output schema
        output_schema = {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "results": {"type": "object"},
                "warnings": {"type": "array", "items": {"type": "string"}},
                "recommendations": {"type": "array", "items": {"type": "string"}},
                "metadata": {"type": "object"}
            },
            "required": ["success", "results"]
        }

        # Generate tool result
        response = await self.generate_structured_response(prompt, output_schema, context)

        try:
            result = json.loads(response.content)
            return result
        except json.JSONDecodeError as e:
            raise AgentError(f"Invalid tool output format: {e}", self.name)

    def _all_steps_successful(self, context: BaseContext) -> bool:
        """Check if all executed steps were successful"""

        # Check if we have any tool results
        if not context.tool_results:
            return False

        # Check if all tool results are successful
        return all(result.success for result in context.tool_results)

    async def retry_failed_step(self, context: BaseContext, step_index: int) -> BaseContext:
        """Retry a failed step"""

        if step_index >= len(context.plan_steps):
            raise AgentError(f"Invalid step index: {step_index}", self.name)

        try:
            # Get step to retry
            step = context.plan_steps[step_index]

            # Log retry attempt
            context.add_agent_message(
                self.role,
                f"Retrying step {step_index + 1}",
                {"step": step}
            )

            # Update current step
            context.current_step = step_index

            # Execute step
            await self._execute_step(step, context)

            # Log success
            context.add_agent_message(
                self.role,
                f"Step {step_index + 1} retry successful"
            )

            return context

        except Exception as e:
            logger.error(f"Step retry failed: {e}")
            raise AgentError(f"Step retry failed: {e}", self.name)

    async def execute_custom_step(self, context: BaseContext, step_description: str, tools: List[str]) -> BaseContext:
        """Execute a custom step not in the original plan"""

        try:
            # Create custom step data
            custom_step = {
                "step_id": f"custom_step_{len(context.plan_steps)}",
                "description": step_description,
                "tools": tools,
                "inputs": {}
            }

            # Log custom step execution
            context.add_agent_message(
                self.role,
                f"Executing custom step: {step_description}",
                {"custom_step": custom_step}
            )

            # Execute custom step
            await self._execute_step(custom_step, context)

            # Add to plan for auditing
            context.plan_steps.append(json.dumps(custom_step))

            return context

        except Exception as e:
            logger.error(f"Custom step execution failed: {e}")
            raise AgentError(f"Custom step execution failed: {e}", self.name)
