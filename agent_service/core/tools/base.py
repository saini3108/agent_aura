"""
Base tool class for all tools in the AURA system.
"""

import logging
from abc import ABC
from abc import abstractmethod
from typing import Any

logger = logging.getLogger("aura_agent")


class BaseTool(ABC):
    """
    Abstract base class for all tools in the AURA system.

    Tools are reusable components that can be used by agents to perform
    specific tasks like calculations, API calls, or data processing.
    """

    # These should be overridden by concrete implementations
    name: str = "base_tool"
    description: str = "Base tool class"

    def __init__(self, name: str | None = None, description: str | None = None):
        """
        Initialize the base tool.

        Args:
            name: Optional name override for the tool
            description: Optional description override
        """
        self.name = name or self.name
        self.description = description or self.description
        self.logger = logging.getLogger(f"aura_agent.tools.{self.name}")

    @abstractmethod
    async def execute(self, **kwargs) -> dict[str, Any]:
        """
        Execute the tool's main functionality.

        This method must be implemented by all concrete tool classes.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            Dictionary containing the tool's output
        """

    def validate_inputs(self, required_params: list[str], **kwargs) -> None:
        """
        Validate that all required parameters are provided.

        Args:
            required_params: List of required parameter names
            **kwargs: Provided parameters

        Raises:
            ValueError: If any required parameter is missing
        """
        missing_params = [param for param in required_params if param not in kwargs]
        if missing_params:
            raise ValueError(
                f"Tool {self.name} requires parameters: {missing_params}",
            )

    def log_execution_start(self, **kwargs) -> None:
        """Log the start of tool execution."""
        self.logger.info(f"🔧 {self.name} starting execution")
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Tool inputs: {list(kwargs.keys())}")

    def log_execution_end(self, result: dict[str, Any]) -> None:
        """Log the end of tool execution."""
        self.logger.info(f"✅ {self.name} completed execution")
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Tool outputs: {list(result.keys())}")

    def handle_error(self, error: Exception, **kwargs) -> dict[str, Any]:
        """
        Handle errors during tool execution.

        Args:
            error: The exception that occurred
            **kwargs: The inputs that caused the error

        Returns:
            Error result dictionary
        """
        self.logger.exception(f"❌ Error in tool {self.name}: {error}")
        return {
            "success": False,
            "error": {
                "tool": self.name,
                "message": str(error),
                "type": type(error).__name__,
            },
        }

    async def __call__(self, **kwargs) -> dict[str, Any]:
        """
        Callable interface for the tool.

        This provides a convenient way to execute the tool.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            Tool execution result
        """
        return await self.execute(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert tool to dictionary representation.

        Returns:
            Dictionary containing tool metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__,
        }

    def __repr__(self) -> str:
        """String representation of the tool."""
        return f"{self.__class__.__name__}(name='{self.name}')"
