import logging
import sys
import json
from datetime import datetime
from typing import Dict, Any
from pythonjsonlogger import jsonlogger

from agent_service.core.config import settings

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for structured logging"""

    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)

        # Add standard fields
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno

        # Add banking-specific context if available
        if hasattr(record, 'task_id'):
            log_record['task_id'] = record.task_id
        if hasattr(record, 'workflow_id'):
            log_record['workflow_id'] = record.workflow_id
        if hasattr(record, 'agent_name'):
            log_record['agent_name'] = record.agent_name
        if hasattr(record, 'model_name'):
            log_record['model_name'] = record.model_name

def setup_logging() -> None:
    """Setup application logging configuration"""

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)

    if settings.LOG_FORMAT.lower() == "json":
        # JSON formatter for structured logging
        formatter = CustomJsonFormatter(
            fmt="%(timestamp)s %(level)s %(logger)s %(module)s %(function)s %(line)d %(message)s"
        )
    else:
        # Standard formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Configure specific loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("redis").setLevel(logging.INFO)

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)

class AuditLogger:
    """Specialized logger for audit trail in banking operations"""

    def __init__(self, name: str = "audit"):
        self.logger = logging.getLogger(name)

    def log_workflow_start(self, workflow_id: str, workflow_type: str, inputs: Dict[str, Any]) -> None:
        """Log workflow start event"""
        self.logger.info(
            "Workflow started",
            extra={
                "workflow_id": workflow_id,
                "workflow_type": workflow_type,
                "inputs": inputs,
                "event_type": "workflow_start"
            }
        )

    def log_workflow_end(self, workflow_id: str, status: str, outputs: Dict[str, Any]) -> None:
        """Log workflow end event"""
        self.logger.info(
            "Workflow completed",
            extra={
                "workflow_id": workflow_id,
                "status": status,
                "outputs": outputs,
                "event_type": "workflow_end"
            }
        )

    def log_agent_action(self, workflow_id: str, agent_name: str, action: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Log agent action"""
        self.logger.info(
            "Agent action executed",
            extra={
                "workflow_id": workflow_id,
                "agent_name": agent_name,
                "action": action,
                "inputs": inputs,
                "outputs": outputs,
                "event_type": "agent_action"
            }
        )

    def log_human_intervention(self, workflow_id: str, action: str, user_id: str, decision: str) -> None:
        """Log human intervention"""
        self.logger.info(
            "Human intervention",
            extra={
                "workflow_id": workflow_id,
                "action": action,
                "user_id": user_id,
                "decision": decision,
                "event_type": "human_intervention"
            }
        )

    def log_error(self, workflow_id: str, error_type: str, error_message: str, context: Dict[str, Any]) -> None:
        """Log error event"""
        self.logger.error(
            "Workflow error",
            extra={
                "workflow_id": workflow_id,
                "error_type": error_type,
                "error_message": error_message,
                "context": context,
                "event_type": "error"
            }
        )
