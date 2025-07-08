import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime

from agent_service.core.schema.context import BaseContext, AgentRole  # Customize if needed
from agent_service.core.services.llm_client import LLMClientManager, LLMResponse
from agent_service.core.utils.logging_config import AuditLogger

logger = logging.getLogger("aura_agent")


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the AURA system.

    Supports common agent lifecycle methods, LLM integration, and error handling.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        role: Optional[AgentRole] = None,
        llm_manager: Optional[LLMClientManager] = None,
    ):
        self.name = name or self.__class__.__name__
        self.role = role or AgentRole.SYSTEM  # Default fallback
        self.llm_manager = llm_manager
        self.logger = logging.getLogger(f"aura_agent.{self.name}")
        self.audit_logger = AuditLogger(f"agent.{self.name}")

        self.state: Dict[str, Any] = {}

        self.config = {
            "max_retries": 3,
            "timeout": 300,
            "model_provider": "openai",
            "temperature": 0.1,
            "max_tokens": 1000,
        }

    @abstractmethod
    async def execute(self, context: BaseContext) -> BaseContext:
        """Main logic to be implemented by concrete agents."""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Define system prompt for the agent."""
        pass

    async def process(self, context: BaseContext) -> BaseContext:
        """Handle agent lifecycle: logging, execution, error capture."""
        start_time = datetime.utcnow()

        try:
            self.audit_logger.log_agent_action(
                workflow_id=context.workflow_id,
                agent_name=self.name,
                action="start",
                inputs={"status": context.status.value, "step": context.current_step},
                outputs={}
            )

            context.current_agent = self.role
            context.add_agent_message(self.role, f"Starting {self.name} execution")

            updated_context = await self.execute(context)

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            self.audit_logger.log_agent_action(
                workflow_id=context.workflow_id,
                agent_name=self.name,
                action="complete",
                inputs={"execution_time": execution_time},
                outputs={"status": updated_context.status.value}
            )

            updated_context.add_agent_message(
                self.role,
                f"Completed {self.name} in {execution_time:.2f}s"
            )

            return updated_context

        except Exception as e:
            self.logger.exception(f"Error in agent {self.name}")
            self.audit_logger.log_error(
                workflow_id=context.workflow_id,
                error_type=type(e).__name__,
                error_message=str(e),
                context={"agent": self.name, "step": context.current_step}
            )

            context.add_agent_message(
                self.role,
                f"Error in {self.name}: {str(e)}",
                {"error_type": type(e).__name__}
            )

            raise

    async def generate_response(self, prompt: str, context: BaseContext, **kwargs) -> LLMResponse:
        """Generate raw LLM response."""
        full_prompt = self._prepare_prompt(prompt, context)
        model_config = {
            "temperature": self.config.get("temperature", 0.1),
            "max_tokens": self.config.get("max_tokens", 1000),
            **kwargs,
        }
        return await self.llm_manager.generate(
            prompt=full_prompt,
            provider=self.config.get("model_provider"),
            **model_config
        )

    async def generate_structured_response(
        self, prompt: str, schema: Dict[str, Any], context: BaseContext, **kwargs
    ) -> LLMResponse:
        """Generate structured response using JSON schema."""
        full_prompt = self._prepare_prompt(prompt, context)
        model_config = {
            "temperature": self.config.get("temperature", 0.1),
            "max_tokens": self.config.get("max_tokens", 1000),
            **kwargs,
        }
        return await self.llm_manager.generate_structured(
            prompt=full_prompt,
            schema=schema,
            provider=self.config.get("model_provider"),
            **model_config
        )

    def _prepare_prompt(self, prompt: str, context: BaseContext) -> str:
        """Combine system prompt and context summary into a full prompt."""
        system_prompt = self.get_system_prompt()
        context_summary = self._get_context_summary(context)

        return f"""
        {system_prompt}

        ## Context
        {context_summary}

        ## Task
        {prompt}

        ## Instructions
        - You are the {self.role.value} agent.
        - Respond with compliance and clarity.
        - Provide explanations when needed.
        - Always be risk-aware.
        """

    def _get_context_summary(self, context: BaseContext) -> str:
        summary = f"""
        - Workflow ID: {context.workflow_id}
        - Workflow Type: {context.workflow_type}
        - Status: {context.status.value}
        - Current Step: {context.current_step}/{len(context.plan_steps)}
        - Messages: {len(context.agent_messages)}
        - Tools Used: {len(context.tool_results)}
        """

        if context.plan_steps:
            summary += f"\n- Plan Steps: {context.plan_steps}"

        if context.human_feedback:
            summary += f"\n- Human Feedback: {context.human_feedback.action} - {context.human_feedback.comments}"

        return summary

    def update_config(self, config: Dict[str, Any]) -> None:
        self.config.update(config)

    def get_state(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role.value,
            "config": self.config,
            "state": self.state,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.state = state.get("state", {})
        if "config" in state:
            self.config.update(state["config"])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


# --- Custom Exceptions ---
class AgentError(Exception):
    def __init__(self, message: str, agent_name: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.agent_name = agent_name
        self.context = context or {}

class AgentTimeoutError(AgentError):
    pass

class AgentValidationError(AgentError):
    pass

class AgentConfigurationError(AgentError):
    pass
