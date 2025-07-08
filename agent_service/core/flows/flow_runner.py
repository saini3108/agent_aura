# agent_service/core/flows/flow_runner.py

import importlib
import logging
from typing import Any

from agent_service.core.schema.flow import FlowInput

logger = logging.getLogger("aura_agent")


class FlowRunner:
    """
    Dynamically loads and runs LangGraph flows based on the module name.

    It expects each module to expose a flow file with a function:
        get_model_validation_flow() -> LangGraph flow object

    Example:
        module='model_validation'
        → imports: agent_service.modules.model_validation.flows.sample_validation_flow
        → calls: get_model_validation_flow()
    """

    def __init__(self, default_flow_name: str = "sample_validation_flow"):
        self.default_flow_name = default_flow_name

    def load_flow(self, module: str):
        """
        Dynamically import and return the LangGraph flow object for a given module.

        Args:
            module (str): The name of the business module (e.g., 'model_validation').

        Returns:
            LangGraph flow object with `.ainvoke(input_state)` method.

        Raises:
            ImportError: If the module or function is not found.
        """
        flow_path = f"agent_service.modules.{module}.flows.{self.default_flow_name}"

        try:
            flow_module = importlib.import_module(flow_path)
            get_flow_fn = getattr(flow_module, "get_model_validation_flow", None)

            def _raise_invalid_type():
                msg = (
                    f"'get_model_validation_flow' not found in module path: {flow_path}"
                )
                raise TypeError(msg)  # noqa: TRY301

            if not callable(get_flow_fn):
                _raise_invalid_type()

            logger.info("📦 Loaded flow from: %s", flow_path)
            return get_flow_fn()

        except Exception as e:
            logger.exception("❌ Failed to load flow: %s", flow_path)
            error_msg = f"Failed to load flow for module '{module}': {e}"
            raise ImportError(error_msg) from e

    async def run(self, module: str, input_data: dict[str, Any]):
        """
        Validate input schema and execute LangGraph flow.

        Args:
            module (str): Business module to run.
            input_data (dict): Initial state for LangGraph.

        Returns:
            dict: Final output state returned from flow.

        Raises:
            ValueError: On input validation failure.
            Exception: If flow execution fails.
        """
        # ✅ Validate state input
        try:
            validated_input = FlowInput(**input_data)
        except Exception as e:
            logger.warning("⚠️ Schema validation failed for module=%s: %s", module, e)
            error_msg = f"Input schema validation failed: {e}"
            raise ValueError(error_msg) from e

        flow = self.load_flow(module)
        logger.info("🚀 Running flow for module='%s' with validated input", module)

        try:
            output_state = await flow.ainvoke(validated_input.dict())
        except Exception as e:
            logger.exception("💥 Flow execution error for module='%s'", module)
            error_msg = f"Flow execution failed: {e}"
            raise RuntimeError(error_msg) from e

        return output_state
