"""
PostgreSQL-based persistent state management.
"""

import json
import logging
from typing import Any

import asyncpg

from agent_service.core.config import settings

logger = logging.getLogger("aura_agent")


class PostgresStore:
    """
    PostgreSQL-based store for persistent state management.

    Provides async interface for storing workflow states, audit logs,
    and persistent data with proper schema management.
    """

    def __init__(self, database_url: str | None = None):
        """
        Initialize Postgres store.

        Args:
            database_url: PostgreSQL connection URL (defaults to config setting)
        """
        self.database_url = database_url or settings.POSTGRES_URL
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Establish PostgreSQL connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                command_timeout=60,
            )

            # Initialize schema
            await self._initialize_schema()
            logger.info("✅ Connected to PostgreSQL")

        except Exception as e:
            logger.exception("❌ Failed to connect to PostgreSQL")
            error_msg = f"PostgreSQL connection failed: {e}"
            raise RuntimeError(error_msg) from e

    async def disconnect(self) -> None:
        """Close PostgreSQL connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("📤 Disconnected from PostgreSQL")

    async def _ensure_connected(self) -> asyncpg.Pool:
        """Ensure PostgreSQL connection pool is established."""
        if self.pool is None:
            await self.connect()
        return self.pool

    async def _initialize_schema(self) -> None:
        """Initialize database schema."""
        pool = await self._ensure_connected()

        async with pool.acquire() as conn:
            # Create workflow_states table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_states (
                    id SERIAL PRIMARY KEY,
                    workflow_id VARCHAR(255) NOT NULL UNIQUE,
                    module_name VARCHAR(255) NOT NULL,
                    input_data JSONB,
                    output_data JSONB,
                    state_data JSONB NOT NULL,
                    status VARCHAR(50) NOT NULL DEFAULT 'active',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """,
            )

            # Create audit_logs table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id SERIAL PRIMARY KEY,
                    workflow_id VARCHAR(255) NOT NULL,
                    agent_name VARCHAR(255),
                    action VARCHAR(255) NOT NULL,
                    details JSONB,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """,
            )

            # Create indexes
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_workflow_states_workflow_id
                ON workflow_states(workflow_id);
            """,
            )

            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_workflow_states_status
                ON workflow_states(status);
            """,
            )

            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_logs_workflow_id
                ON audit_logs(workflow_id);
            """,
            )

            logger.info("📋 Database schema initialized")

    async def save_workflow_state(
        self,
        workflow_id: str,
        module_name: str,
        state_data: dict[str, Any],
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        status: str = "active",
    ) -> int:
        """
        Save workflow state to database.

        Args:
            workflow_id: Unique workflow identifier
            module_name: Name of the module
            state_data: Workflow state dictionary
            status: Workflow status

        Returns:
            Record ID
        """

        pool = await self._ensure_connected()

        try:
            async with pool.acquire() as conn:
                existing = await conn.fetchrow(
                    "SELECT id FROM workflow_states WHERE workflow_id = $1",
                    workflow_id,
                )

                if existing:
                    record_id = await conn.fetchval(
                        """
                        UPDATE workflow_states
                        SET input_data = $1,
                            output_data = $2,
                            status = $3,
                            updated_at = NOW()
                        WHERE workflow_id = $4
                        RETURNING id
                    """,
                        json.dumps(input_data),
                        json.dumps(output_data),
                        status,
                        workflow_id,
                    )
                else:
                    record_id = await conn.fetchval(
                        """
                        INSERT INTO workflow_states (workflow_id, module_name, input_data, output_data, status)
                        VALUES ($1, $2, $3, $4, $5)
                        RETURNING id
                    """,
                        workflow_id,
                        module_name,
                        json.dumps(input_data),
                        json.dumps(output_data),
                        status,
                    )

                logger.debug("💾 Saved workflow state: %s", workflow_id)
                return record_id

        except Exception as e:
            logger.exception("❌ Failed to save workflow state %s", workflow_id)
            raise RuntimeError("PostgreSQL save failed") from e

    async def get_workflow_state(self, workflow_id: str) -> dict[str, Any] | None:
        """
        Retrieve workflow state from database.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            Workflow state or None if not found
        """
        pool = await self._ensure_connected()

        try:
            async with pool.acquire() as conn:
                record = await conn.fetchrow(
                    """
                    SELECT workflow_id,
                        module_name,
                        input_data,
                        output_data,
                        status,
                        created_at,
                        updated_at
                    FROM workflow_states
                    WHERE workflow_id = $1
                """,
                    workflow_id,
                )

                if not record:
                    return None

                result = {
                    "workflow_id": record["workflow_id"],
                    "module_name": record["module_name"],
                    "input_data": (
                        json.loads(record["input_data"])
                        if record["input_data"]
                        else None
                    ),
                    "output_data": (
                        json.loads(record["output_data"])
                        if record["output_data"]
                        else None
                    ),
                    "status": record["status"],
                    "created_at": record["created_at"].isoformat(),
                    "updated_at": record["updated_at"].isoformat(),
                }

                logger.debug("📖 Retrieved workflow state: %s", workflow_id)
                return result

        except Exception as e:
            logger.exception("❌ Failed to get workflow state %s", workflow_id)
            raise RuntimeError("PostgreSQL get failed") from e

    async def log_audit_event(
        self,
        workflow_id: str,
        agent_name: str,
        action: str,
        details: dict[str, Any] | None = None,
    ) -> int:
        """
        Log an audit event.

        Args:
            workflow_id: Workflow identifier
            agent_name: Name of the agent performing the action
            action: Action description
            details: Additional details

        Returns:
            Log record ID
        """
        pool = await self._ensure_connected()

        try:
            async with pool.acquire() as conn:
                record_id = await conn.fetchval(
                    """
                    INSERT INTO audit_logs (workflow_id, agent_name, action, details)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id
                    """,
                    workflow_id,
                    agent_name,
                    action,
                    json.dumps(details) if details else None,
                )

                logger.debug("📝 Logged audit event: %s - %s", workflow_id, action)
                return record_id

        except Exception as e:
            logger.exception("❌ Failed to log audit event %s", workflow_id)
            error_msg = "PostgreSQL audit log failed"
            raise RuntimeError(error_msg) from e

    async def get_audit_logs(self, workflow_id: str) -> list[dict[str, Any]]:
        """
        Retrieve audit logs for a workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            List of audit log entries
        """
        pool = await self._ensure_connected()

        try:
            async with pool.acquire() as conn:
                records = await conn.fetch(
                    """
                    SELECT agent_name, action, details, timestamp
                    FROM audit_logs
                    WHERE workflow_id = $1
                    ORDER BY timestamp ASC
                """,
                    workflow_id,
                )

                result = []
                for record in records:
                    entry = {
                        "agent_name": record["agent_name"],
                        "action": record["action"],
                        "details": (
                            json.loads(record["details"]) if record["details"] else None
                        ),
                        "timestamp": record["timestamp"].isoformat(),
                    }
                    result.append(entry)

                logger.debug(
                    "📋 Retrieved %d audit logs for: %s",
                    len(result),
                    workflow_id,
                )
                return result

        except Exception as e:
            logger.exception("❌ Failed to get audit logs %s", workflow_id)
            error_msg = "PostgreSQL audit get failed"
            raise RuntimeError(error_msg) from e

    async def get_workflow_history(
        self,
        module_name: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get workflow history with optional filtering.

        Args:
            module_name: Filter by module name
            status: Filter by status
            limit: Maximum number of records

        Returns:
            List of workflow records
        """
        pool = await self._ensure_connected()

        try:
            async with pool.acquire() as conn:
                query = "SELECT * FROM workflow_states WHERE 1=1"
                params = []
                param_count = 0

                if module_name:
                    param_count += 1
                    query += f" AND module_name = ${param_count}"
                    params.append(module_name)

                if status:
                    param_count += 1
                    query += f" AND status = ${param_count}"
                    params.append(status)

                query += " ORDER BY updated_at DESC LIMIT $%d" % (param_count + 1)
                params.append(limit)

                records = await conn.fetch(query, *params)

                result = []
                for record in records:
                    state_data = json.loads(record["state_data"])
                    item = {
                        "workflow_id": record["workflow_id"],
                        "module_name": record["module_name"],
                        "state_data": state_data,
                        "status": record["status"],
                        "created_at": record["created_at"].isoformat(),
                        "updated_at": record["updated_at"].isoformat(),
                    }
                    result.append(item)

                logger.debug("📜 Retrieved %d workflow history records", len(result))
                return result

        except Exception as e:
            logger.exception("❌ Failed to get workflow history")
            error_msg = "PostgreSQL history get failed"
            raise RuntimeError(error_msg) from e
