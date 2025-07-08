
#!/usr/bin/env python3
"""
Main CLI entry point for Banking AI Platform administrative tasks.

This provides a unified command-line interface for various administrative
and operational tasks including workflow management, health checks, and system utilities.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_service.core.config import settings
from agent_service.core.memory.redis_store import MemoryService
from agent_service.core.services.llm_client import llm_manager
from agent_service.core.utils.logging_config import setup_logging
from agent_service.core.workflows.graph import BankingWorkflowGraph


def setup_cli_logging(debug: bool = False) -> None:
    """Setup logging for CLI execution."""
    setup_logging()


@click.group()
def cli():
    """Banking AI Platform - Administrative CLI"""


@cli.group()
def health():
    """Health check commands for system components"""


@cli.command()
def test_memory():
    """Test memory service functionality."""
    click.echo("🧪 Testing memory service...")
    asyncio.run(_test_memory_service())


@cli.command()
def test_workflow():
    """Test complete banking workflow."""
    click.echo("🧪 Testing model validation workflow...")
    asyncio.run(_test_model_validation_workflow())


@cli.command()
def test_all():
    """Run all tests - memory, workflows, and agents."""
    click.echo("🧪 Running comprehensive test suite...")

    click.echo("\n1️⃣ Testing memory service...")
    asyncio.run(_test_memory_service())

    click.echo("\n2️⃣ Testing banking workflow...")
    asyncio.run(_test_model_validation_workflow())

    click.echo("\n✅ All tests completed!")


async def _test_memory_service():
    """Test memory service functionality."""
    try:
        memory = MemoryService()
        await memory.initialize()

        # Test basic operations
        test_key = "test_cli_key"
        test_value = {"test": "data", "timestamp": "2024-01-01"}

        await memory.cache_set(test_key, test_value, 60)
        retrieved = await memory.cache_get(test_key)

        if retrieved == test_value:
            click.echo("   ✅ Memory service is working correctly")
        else:
            click.echo("   ❌ Memory service test failed")

        await memory.cache_delete(test_key)
        await memory.close()

    except Exception as e:
        click.echo(f"   💥 Memory service test failed: {e}")


async def _test_model_validation_workflow():
    """Test model validation workflow."""
    try:
        memory = MemoryService()
        await memory.initialize()

        workflow_graph = BankingWorkflowGraph(memory)

        # Test workflow data
        test_data = {
            "model_name": "test_credit_model",
            "model_configuration": {
                "algorithm": "logistic_regression",
                "features": ["credit_score", "income", "debt_ratio"],
                "target": "default_probability"
            }
        }

        workflow_id = await workflow_graph.start_workflow(
            workflow_type="model_validation",
            inputs=test_data,
            config={"provider": "mock", "model": "test"}
        )

        click.echo(f"   ✅ Workflow started: {workflow_id}")

        # Check status
        status = await workflow_graph.get_workflow_status(workflow_id)
        click.echo(f"   📊 Status: {status.get('status', 'unknown')}")

        await memory.close()

    except Exception as e:
        click.echo(f"   💥 Workflow test failed: {e}")


@health.command()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def check_all(debug: bool) -> None:
    """Check health of all system components."""
    setup_cli_logging(debug)

    click.echo("🏥 Banking AI Platform - Health Check")
    click.echo("=" * 50)

    overall_healthy = True

    # Check Memory Service
    click.echo("\n💾 Memory Service Health Check:")
    memory_healthy = asyncio.run(_check_memory_health())
    overall_healthy &= memory_healthy

    # Check LLM Providers
    click.echo("\n🧠 LLM Providers Health Check:")
    llm_healthy = _check_llm_health()
    overall_healthy &= llm_healthy

    # Overall status
    click.echo("\n" + "=" * 50)
    if overall_healthy:
        click.echo("✅ All systems healthy!")
        sys.exit(0)
    else:
        click.echo("❌ Some systems are unhealthy!")
        sys.exit(1)


@health.command()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def memory(debug: bool) -> None:
    """Check memory service health."""
    setup_cli_logging(debug)
    healthy = asyncio.run(_check_memory_health())
    sys.exit(0 if healthy else 1)


@health.command()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def llm(debug: bool) -> None:
    """Check LLM providers health."""
    setup_cli_logging(debug)
    healthy = _check_llm_health()
    sys.exit(0 if healthy else 1)


async def _check_memory_health() -> bool:
    """Check memory service health and display results."""
    try:
        memory = MemoryService()
        await memory.initialize()

        healthy = await memory.health_check()

        if healthy:
            click.echo("   ✅ Memory service is healthy")
            if memory.use_redis:
                click.echo("   🔴 Using Redis backend")
            else:
                click.echo("   🧠 Using in-memory backend")
            return True
        else:
            click.echo("   ❌ Memory service is unhealthy")
            return False

        await memory.close()

    except Exception as e:
        click.echo(f"   💥 Memory health check failed: {e}")
        return False


def _check_llm_health() -> bool:
    """Check LLM providers health and display results."""
    try:
        available_providers = llm_manager.get_available_providers()

        if available_providers:
            click.echo("   ✅ LLM providers are available")
            for provider in available_providers:
                click.echo(f"   🤖 {provider}")
            return True
        else:
            click.echo("   ❌ No LLM providers available")
            click.echo("   💡 Set API keys in environment variables")
            return False

    except Exception as e:
        click.echo(f"   💥 LLM health check failed: {e}")
        return False


@cli.group()
def workflows():
    """Workflow management commands"""


@workflows.command()
@click.option("--type", help="Filter by workflow type")
def list_types(type: Optional[str]) -> None:
    """List available workflow types."""
    click.echo("📋 Available Workflow Types:")

    workflow_types = [
        {
            "type": "model_validation",
            "description": "Validate banking models with statistical tests and compliance checks",
            "inputs": ["model_name", "model_configuration"],
            "duration": "5-15 minutes"
        },
        {
            "type": "ecl_calculation",
            "description": "Calculate Expected Credit Loss under IFRS 9 standards",
            "inputs": ["portfolio_data", "pd_curves", "scenarios"],
            "duration": "10-30 minutes"
        },
        {
            "type": "rwa_calculation",
            "description": "Calculate Risk-Weighted Assets under Basel III framework",
            "inputs": ["exposure_data", "risk_weights"],
            "duration": "15-45 minutes"
        },
        {
            "type": "reporting",
            "description": "Generate regulatory and management reports",
            "inputs": ["report_type", "data_sources"],
            "duration": "3-10 minutes"
        }
    ]

    if type:
        workflow_types = [w for w in workflow_types if w["type"] == type]

    for workflow in workflow_types:
        click.echo(f"\n🔄 {workflow['type']}")
        click.echo(f"   📝 {workflow['description']}")
        click.echo(f"   📊 Inputs: {', '.join(workflow['inputs'])}")
        click.echo(f"   ⏱️  Duration: {workflow['duration']}")


@workflows.command()
@click.argument("workflow_type")
@click.option("--debug", is_flag=True, help="Enable debug logging")
def test(workflow_type: str, debug: bool) -> None:
    """Test a workflow with sample data."""
    setup_cli_logging(debug)

    click.echo(f"🧪 Testing workflow: {workflow_type}")

    # Get sample data for workflow type
    test_data = _get_sample_workflow_data(workflow_type)

    if not test_data:
        click.echo(f"❌ Unknown workflow type: {workflow_type}")
        sys.exit(1)

    try:
        result = asyncio.run(_run_test_workflow(workflow_type, test_data))

        click.echo("✅ Workflow test completed successfully!")
        click.echo(f"📊 Result: {json.dumps(result, indent=2, default=str)}")

    except Exception as e:
        click.echo(f"❌ Workflow test failed: {e}")
        sys.exit(1)


def _get_sample_workflow_data(workflow_type: str) -> Optional[Dict[str, Any]]:
    """Get sample data for different workflow types."""

    sample_data = {
        "model_validation": {
            "model_name": "test_credit_model",
            "model_configuration": {
                "algorithm": "logistic_regression",
                "features": ["credit_score", "income", "debt_ratio"],
                "target": "default_probability"
            }
        },
        "ecl_calculation": {
            "portfolio_data": [
                {
                    "account_id": "TEST001",
                    "balance": 50000,
                    "product_type": "personal_loan",
                    "origination_date": "2023-01-15",
                    "maturity_date": "2028-01-15",
                    "credit_score": 720
                }
            ],
            "scenarios": ["base", "stress"]
        },
        "rwa_calculation": {
            "exposure_data": [
                {
                    "exposure_id": "TEST001",
                    "counterparty": "Test_Corp",
                    "exposure_amount": 1000000,
                    "asset_class": "corporate",
                    "rating": "BBB"
                }
            ]
        },
        "reporting": {
            "report_type": "risk_summary",
            "data_sources": ["portfolio_data", "market_data"],
            "template": "monthly_risk_report"
        }
    }

    return sample_data.get(workflow_type)


async def _run_test_workflow(workflow_type: str, test_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run a test workflow."""
    memory = MemoryService()
    await memory.initialize()

    try:
        workflow_graph = BankingWorkflowGraph(memory)

        workflow_id = await workflow_graph.start_workflow(
            workflow_type=workflow_type,
            inputs=test_data,
            config={"provider": "mock", "model": "test"}
        )

        # Wait a moment for processing
        await asyncio.sleep(1)

        # Get results
        status = await workflow_graph.get_workflow_status(workflow_id)
        results = await workflow_graph.get_workflow_results(workflow_id)

        return {
            "workflow_id": workflow_id,
            "status": status,
            "results": results
        }

    finally:
        await memory.close()


@cli.group()
def data():
    """Data management commands"""


@data.command()
@click.option("--workflow-id", help="Filter by workflow ID")
@click.option("--workflow-type", help="Filter by workflow type")
@click.option("--status", help="Filter by status")
@click.option("--limit", default=10, help="Maximum number of records")
@click.option("--output-file", type=click.Path(), help="Save results to file")
def workflows(
    workflow_id: Optional[str],
    workflow_type: Optional[str],
    status: Optional[str],
    limit: int,
    output_file: Optional[str]
) -> None:
    """List workflow history."""
    result = asyncio.run(_get_workflow_history(workflow_id, workflow_type, status, limit))

    if output_file:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        click.echo(f"💾 Results saved to {output_file}")
    else:
        click.echo("📋 Workflow History:")
        for workflow in result:
            click.echo(f"\n🔄 {workflow['workflow_id']}")
            click.echo(f"   📦 Type: {workflow['workflow_type']}")
            click.echo(f"   📊 Status: {workflow['status']}")
            click.echo(f"   📅 Created: {workflow['created_at']}")
            click.echo(f"   🔄 Updated: {workflow['updated_at']}")


async def _get_workflow_history(
    workflow_id: Optional[str],
    workflow_type: Optional[str],
    status: Optional[str],
    limit: int
) -> List[Dict[str, Any]]:
    """Get workflow history from memory service."""
    memory = MemoryService()
    await memory.initialize()

    try:
        if workflow_id:
            # Get specific workflow
            metadata = await memory.get_workflow_metadata(workflow_id)
            return [metadata] if metadata else []

        # Get list of active workflows
        active_workflows = await memory.list_active_workflows()

        results = []
        for wf_id in active_workflows[:limit]:
            metadata = await memory.get_workflow_metadata(wf_id)
            if metadata:
                # Apply filters
                if workflow_type and metadata.get("workflow_type") != workflow_type:
                    continue
                if status and metadata.get("status") != status:
                    continue

                results.append(metadata)

        return results

    finally:
        await memory.close()


@data.command()
@click.argument("workflow_id")
def context(workflow_id: str) -> None:
    """Show workflow context and state."""
    result = asyncio.run(_get_workflow_context(workflow_id))

    if not result:
        click.echo(f"📭 No context found for workflow: {workflow_id}")
        return

    click.echo(f"📋 Workflow Context: {workflow_id}")
    click.echo("=" * 60)
    click.echo(json.dumps(result, indent=2, default=str))


async def _get_workflow_context(workflow_id: str) -> Optional[Dict[str, Any]]:
    """Get workflow context."""
    memory = MemoryService()
    await memory.initialize()

    try:
        context = await memory.load_context(workflow_id)
        return context.dict() if context else None
    finally:
        await memory.close()


@cli.command()
@click.option("--port", default=8000, help="Port to run on")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def serve(port: int, host: str, reload: bool, debug: bool) -> None:
    """Start the Banking AI Platform server."""
    setup_cli_logging(debug)

    click.echo("🚀 Starting Banking AI Platform...")
    click.echo(f"🌐 Server will be available at http://{host}:{port}")
    click.echo(f"📖 API documentation at http://{host}:{port}/docs")

    import uvicorn

    uvicorn.run(
        "agent_service.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="debug" if debug else "info",
    )


@cli.group()
def config():
    """Configuration management commands"""


@config.command()
def show():
    """Show current configuration."""
    click.echo("⚙️  Banking AI Platform Configuration:")
    click.echo("=" * 40)

    config_items = [
        ("Debug Mode", settings.DEBUG),
        ("Memory TTL", f"{settings.MEMORY_TTL} seconds"),
        ("Vector Store Size", settings.VECTOR_STORE_SIZE),
        ("Redis URL", settings.REDIS_URL if hasattr(settings, 'REDIS_URL') else "Not configured"),
        ("Allowed Origins", settings.ALLOWED_ORIGINS),
    ]

    for key, value in config_items:
        click.echo(f"📋 {key}: {value}")


@config.command()
def validate():
    """Validate current configuration."""
    click.echo("✅ Validating configuration...")

    issues = []

    # Check required settings
    if not hasattr(settings, 'MEMORY_TTL') or settings.MEMORY_TTL <= 0:
        issues.append("MEMORY_TTL must be a positive integer")

    if not hasattr(settings, 'VECTOR_STORE_SIZE') or settings.VECTOR_STORE_SIZE <= 0:
        issues.append("VECTOR_STORE_SIZE must be a positive integer")

    if issues:
        click.echo("❌ Configuration issues found:")
        for issue in issues:
            click.echo(f"   • {issue}")
        sys.exit(1)
    else:
        click.echo("✅ Configuration is valid!")


if __name__ == "__main__":
    cli()
