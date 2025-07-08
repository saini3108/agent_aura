
import json
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import redis.asyncio as aioredis
from redis.asyncio import Redis
import logging

from agent_service.core.config import settings
from agent_service.core.schema.context import BaseContext

logger = logging.getLogger(__name__)

class RedisStore:
    """
    Redis-based store for caching and temporary state management.

    Provides async interface for storing workflow state, caching results,
    and managing temporary data with TTL support.
    """

    def __init__(self, redis_url: str | None = None):
        """
        Initialize Redis store.

        Args:
            redis_url: Redis connection URL (defaults to config setting)
        """
        self.redis_url = redis_url or settings.REDIS_URL
        self.redis: Redis | None = None

    async def connect(self) -> None:
        """Establish Redis connection."""
        try:
            self.redis = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            # Test connection
            await self.redis.ping()
            logger.info("✅ Connected to Redis")
        except Exception as e:
            logger.exception("❌ Failed to connect to Redis")
            msg = f"Redis connection failed: {e}"
            raise RuntimeError(msg) from e

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            logger.info("📤 Disconnected from Redis")

    async def _ensure_connected(self) -> Redis:
        """Ensure Redis connection is established."""
        if self.redis is None:
            await self.connect()
        return self.redis

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """
        Set a value in Redis.

        Args:
            key: Redis key
            value: Value to store (will be JSON serialized)
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        redis_client = await self._ensure_connected()

        try:
            # Serialize value to JSON
            serialized_value = json.dumps(value, default=str)

            if ttl:
                result = await redis_client.setex(key, ttl, serialized_value)
            else:
                result = await redis_client.set(key, serialized_value)

            logger.debug("📝 Set Redis key: %s (TTL: %s)", key, ttl)
            return bool(result)

        except Exception as e:
            logger.exception("❌ Failed to set Redis key %s", key)
            msg = f"Redis set failed: {e}"
            raise RuntimeError(msg) from e

    async def get(self, key: str) -> Any | None:
        """
        Get a value from Redis.

        Args:
            key: Redis key

        Returns:
            Deserialized value or None if not found
        """
        redis_client = await self._ensure_connected()

        try:
            value = await redis_client.get(key)
            if value is None:
                return None
            # Deserialize from JSON
            result = json.loads(value)
            logger.debug("📖 Got Redis key: %s", key)
            return result

        except json.JSONDecodeError as e:
            logger.exception(
                "❌ Failed to deserialize Redis value for key %s: %s",
                key,
                e,
            )
            return None
        except Exception as e:
            logger.exception("❌ Failed to get Redis key %s", key)
            msg = f"Redis get failed: {e}"
            raise RuntimeError(msg) from e

    async def delete(self, key: str) -> bool:
        """
        Delete a key from Redis.

        Args:
            key: Redis key to delete

        Returns:
            True if key was deleted
        """
        redis_client = await self._ensure_connected()

        try:
            result = await redis_client.delete(key)
            logger.debug("🗑️ Deleted Redis key: %s", key)
            return bool(result)

        except Exception as e:
            logger.exception("❌ Failed to delete Redis key %s", key)
            msg = f"Redis delete failed: {e}"
            raise RuntimeError(msg) from e

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis.

        Args:
            key: Redis key

        Returns:
            True if key exists
        """
        redis_client = await self._ensure_connected()

        try:
            result = await redis_client.exists(key)
            return bool(result)

        except Exception as e:
            logger.exception("❌ Failed to check Redis key existence %s", key)
            msg = f"Redis exists failed: {e}"
            raise RuntimeError(msg) from e

    async def set_workflow_state(
        self,
        workflow_id: str,
        state: dict[str, Any],
        ttl: int = 3600,
    ) -> bool:
        """
        Store workflow state with a specific TTL.

        Args:
            workflow_id: Unique workflow identifier
            state: Workflow state dictionary
            ttl: Time to live in seconds (default: 1 hour)

        Returns:
            True if successful
        """
        key = f"workflow:{workflow_id}"
        return await self.set(key, state, ttl)

    async def get_workflow_state(self, workflow_id: str) -> dict[str, Any] | None:
        """
        Retrieve workflow state.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            Workflow state or None if not found
        """
        key = f"workflow:{workflow_id}"
        return await self.get(key)

    async def cache_result(self, cache_key: str, result: Any, ttl: int = 900) -> bool:
        """
        Cache a computation result.

        Args:
            cache_key: Unique cache key
            result: Result to cache
            ttl: Time to live in seconds (default: 15 minutes)

        Returns:
            True if successful
        """
        key = f"cache:{cache_key}"
        return await self.set(key, result, ttl)

    async def get_cached_result(self, cache_key: str) -> Any | None:
        """
        Retrieve cached result.

        Args:
            cache_key: Unique cache key

        Returns:
            Cached result or None if not found
        """
        key = f"cache:{cache_key}"
        return await self.get(key)

    async def redis_health_check(self) -> dict[str, Any]:
        """
        Perform Redis health check.

        Returns:
            Health status information
        """
        try:
            redis_client = await self._ensure_connected()

            # Test basic operations
            test_key = "health_check"
            await redis_client.set(test_key, "ok", ex=5)
            value = await redis_client.get(test_key)
            await redis_client.delete(test_key)

            def _raise_redis_health_error():
                error_message = "Redis read/write test failed"
                raise RuntimeError(error_message)

            if value != "ok":
                _raise_redis_health_error()

            # Get Redis info
            info = await redis_client.info()

            return {
                "status": "healthy",
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human"),
            }

        except Exception as e:
            logger.exception("❌ Redis health check failed")
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


class MemoryService:
    """Memory service for workflow context and agent state management"""

    def __init__(self):
        self.redis: Optional[Redis] = None
        self.redis_store: Optional[RedisStore] = None
        self.in_memory_store: Dict[str, Any] = {}
        self.vector_store: Dict[str, Any] = {}  # Simple in-memory vector store
        self.initialized = False
        self.use_redis = False

    async def initialize(self) -> None:
        """Initialize memory service connections with fallback"""
        try:
            # Try to initialize Redis store
            self.redis_store = RedisStore()
            await self.redis_store.connect()

            # Keep backward compatibility with existing Redis client
            self.redis = self.redis_store.redis

            logger.info("Redis connection established")
            self.use_redis = True

        except Exception as e:
            logger.warning(f"Redis not available, using in-memory storage: {e}")
            self.redis = None
            self.redis_store = None
            self.use_redis = False

        self.initialized = True
        logger.info(f"Memory service initialized (Redis: {self.use_redis})")

    async def close(self) -> None:
        """Close memory service connections"""
        if self.redis_store:
            await self.redis_store.disconnect()
        elif self.redis:
            await self.redis.close()
        logger.info("Memory service connections closed")

    async def health_check(self) -> bool:
        """Check if memory service is healthy"""
        if not self.initialized:
            return False

        if self.use_redis and self.redis_store:
            try:
                health_result = await self.redis_store.redis_health_check()
                return health_result.get("status") == "healthy"
            except Exception:
                return False

        # In-memory storage is always available if initialized
        return True

    # Context Management
    async def save_context(self, context: BaseContext) -> None:
        """Save workflow context to memory"""
        if not self.initialized:
            raise RuntimeError("Memory service not initialized")

        try:
            # Serialize context
            context_data = context.model_dump() if hasattr(context, 'model_dump') else context.dict()

            if self.use_redis and self.redis_store:
                # Use RedisStore for better error handling
                await self.redis_store.set_workflow_state(
                    context.workflow_id,
                    context_data,
                    settings.MEMORY_TTL,
                )

                # Save to workflow index
                await self.redis.sadd("workflows:active", context.workflow_id)

                # Save metadata
                metadata = {
                    "workflow_id": context.workflow_id,
                    "workflow_type": context.workflow_type,
                    "status": context.status.value,
                    "created_at": context.created_at.isoformat(),
                    "updated_at": context.updated_at.isoformat()
                }

                await self.redis.hset(
                    f"workflow:meta:{context.workflow_id}",
                    mapping=metadata
                )
            else:
                # Save to in-memory store
                key = f"context:{context.workflow_id}"
                self.in_memory_store[key] = {
                    'data': context_data,
                    'timestamp': datetime.utcnow().isoformat()
                }

                # Maintain active workflows list
                active_key = "workflows:active"
                if active_key not in self.in_memory_store:
                    self.in_memory_store[active_key] = set()
                self.in_memory_store[active_key].add(context.workflow_id)

            logger.debug(f"Context saved for workflow {context.workflow_id}")

        except Exception as e:
            logger.error(f"Failed to save context: {e}")
            raise

    async def load_context(self, workflow_id: str) -> Optional[BaseContext]:
        """Load workflow context from memory"""
        if not self.initialized:
            raise RuntimeError("Memory service not initialized")

        try:
            if self.use_redis and self.redis_store:
                # Use RedisStore for better error handling
                context_data = await self.redis_store.get_workflow_state(workflow_id)

                if not context_data:
                    return None

                context = BaseContext(**context_data)
                logger.debug(f"Context loaded for workflow {workflow_id}")
                return context
            else:
                # Load from in-memory store
                key = f"context:{workflow_id}"
                stored_data = self.in_memory_store.get(key)

                if not stored_data:
                    return None

                context = BaseContext(**stored_data['data'])
                logger.debug(f"Context loaded for workflow {workflow_id}")
                return context

        except Exception as e:
            logger.error(f"Failed to load context: {e}")
            return None

    async def delete_context(self, workflow_id: str) -> bool:
        """Delete workflow context from memory"""
        if not self.initialized:
            raise RuntimeError("Memory service not initialized")

        try:
            if self.use_redis and self.redis_store:
                # Delete context using RedisStore
                key = f"workflow:{workflow_id}"
                deleted = await self.redis_store.delete(key)

                # Remove from active workflows
                await self.redis.srem("workflows:active", workflow_id)

                # Delete metadata
                await self.redis.delete(f"workflow:meta:{workflow_id}")

                logger.debug(f"Context deleted for workflow {workflow_id}")
                return deleted
            else:
                # Delete from in-memory store
                key = f"context:{workflow_id}"
                deleted = key in self.in_memory_store
                if deleted:
                    del self.in_memory_store[key]

                # Remove from active workflows
                active_key = "workflows:active"
                if active_key in self.in_memory_store:
                    self.in_memory_store[active_key].discard(workflow_id)

                logger.debug(f"Context deleted for workflow {workflow_id}")
                return deleted

        except Exception as e:
            logger.error(f"Failed to delete context: {e}")
            return False

    # Agent State Management
    async def save_agent_state(self, workflow_id: str, agent_name: str, state: Dict[str, Any]) -> None:
        """Save agent state"""
        if not self.initialized:
            raise RuntimeError("Memory service not initialized")

        try:
            key = f"agent:{workflow_id}:{agent_name}"

            if self.use_redis and self.redis_store:
                await self.redis_store.set(key, state, settings.MEMORY_TTL)
            else:
                self.in_memory_store[key] = {
                    'data': state,
                    'timestamp': datetime.utcnow().isoformat()
                }

            logger.debug(f"Agent state saved for {agent_name} in workflow {workflow_id}")

        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")
            raise

    async def load_agent_state(self, workflow_id: str, agent_name: str) -> Optional[Dict[str, Any]]:
        """Load agent state"""
        if not self.initialized:
            raise RuntimeError("Memory service not initialized")

        try:
            key = f"agent:{workflow_id}:{agent_name}"

            if self.use_redis and self.redis_store:
                state = await self.redis_store.get(key)
            else:
                stored_data = self.in_memory_store.get(key)
                state = stored_data['data'] if stored_data else None

            if state:
                logger.debug(f"Agent state loaded for {agent_name} in workflow {workflow_id}")

            return state

        except Exception as e:
            logger.error(f"Failed to load agent state: {e}")
            return None

    # Session Management
    async def create_session(self, workflow_id: str, user_id: Optional[str] = None) -> str:
        """Create a new session"""
        if not self.initialized:
            raise RuntimeError("Memory service not initialized")

        try:
            session_id = f"session:{workflow_id}:{datetime.utcnow().timestamp()}"

            session_data = {
                "workflow_id": workflow_id,
                "user_id": user_id or "anonymous",
                "created_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat()
            }

            if self.use_redis and self.redis:
                await self.redis.hset(session_id, mapping=session_data)
                await self.redis.expire(session_id, settings.MEMORY_TTL)
            else:
                self.in_memory_store[session_id] = session_data

            logger.debug(f"Session created: {session_id}")
            return session_id

        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise

    async def update_session_activity(self, session_id: str) -> None:
        """Update session last activity"""
        if not self.initialized:
            raise RuntimeError("Memory service not initialized")

        try:
            if self.use_redis and self.redis:
                await self.redis.hset(
                    session_id,
                    "last_activity",
                    datetime.utcnow().isoformat()
                )
            else:
                if session_id in self.in_memory_store:
                    self.in_memory_store[session_id]["last_activity"] = datetime.utcnow().isoformat()

        except Exception as e:
            logger.error(f"Failed to update session activity: {e}")

    # Workflow Management
    async def list_active_workflows(self) -> List[str]:
        """List active workflow IDs"""
        if not self.initialized:
            raise RuntimeError("Memory service not initialized")

        try:
            if self.use_redis and self.redis:
                workflows = await self.redis.smembers("workflows:active")
                return list(workflows)
            else:
                active_key = "workflows:active"
                if active_key in self.in_memory_store:
                    return list(self.in_memory_store[active_key])
                return []

        except Exception as e:
            logger.error(f"Failed to list active workflows: {e}")
            return []

    async def get_workflow_metadata(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow metadata"""
        if not self.initialized:
            raise RuntimeError("Memory service not initialized")

        try:
            if self.use_redis and self.redis:
                metadata = await self.redis.hgetall(f"workflow:meta:{workflow_id}")
                return metadata if metadata else None
            else:
                # For in-memory, extract metadata from context
                key = f"context:{workflow_id}"
                stored_data = self.in_memory_store.get(key)
                if stored_data and 'data' in stored_data:
                    context_data = stored_data['data']
                    return {
                        "workflow_id": context_data.get("workflow_id"),
                        "workflow_type": context_data.get("workflow_type"),
                        "status": context_data.get("status"),
                        "created_at": context_data.get("created_at"),
                        "updated_at": context_data.get("updated_at")
                    }
                return None

        except Exception as e:
            logger.error(f"Failed to get workflow metadata: {e}")
            return None

    # Vector Store Operations (Simple implementation)
    async def store_vector(self, key: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Store vector with metadata"""
        self.vector_store[key] = {
            "vector": vector,
            "metadata": metadata,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Keep only recent vectors
        if len(self.vector_store) > settings.VECTOR_STORE_SIZE:
            # Remove oldest entries
            sorted_keys = sorted(
                self.vector_store.keys(),
                key=lambda k: self.vector_store[k]["timestamp"]
            )

            for old_key in sorted_keys[:-settings.VECTOR_STORE_SIZE]:
                del self.vector_store[old_key]

    async def search_vectors(self, query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search similar vectors (simplified similarity search)"""
        results = []

        for key, data in self.vector_store.items():
            # Simple cosine similarity calculation
            similarity = self._cosine_similarity(query_vector, data["vector"])

            results.append({
                "key": key,
                "similarity": similarity,
                "metadata": data["metadata"]
            })

        # Sort by similarity and return top results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    # Caching (enhanced with RedisStore)
    async def cache_set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set cache value"""
        if not self.initialized:
            raise RuntimeError("Memory service not initialized")

        try:
            ttl = ttl or settings.MEMORY_TTL

            if self.use_redis and self.redis_store:
                await self.redis_store.cache_result(key, value, ttl)
            else:
                cache_key = f"cache:{key}"
                self.in_memory_store[cache_key] = {
                    'data': value,
                    'timestamp': datetime.utcnow().isoformat(),
                    'ttl': ttl
                }

        except Exception as e:
            logger.error(f"Failed to set cache: {e}")
            raise

    async def cache_get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        if not self.initialized:
            raise RuntimeError("Memory service not initialized")

        try:
            if self.use_redis and self.redis_store:
                return await self.redis_store.get_cached_result(key)
            else:
                cache_key = f"cache:{key}"
                stored_data = self.in_memory_store.get(cache_key)
                if stored_data:
                    # Check TTL for in-memory cache
                    timestamp = datetime.fromisoformat(stored_data['timestamp'])
                    if datetime.utcnow() - timestamp < timedelta(seconds=stored_data.get('ttl', settings.MEMORY_TTL)):
                        return stored_data['data']
                    else:
                        # Remove expired entry
                        del self.in_memory_store[cache_key]
                return None

        except Exception as e:
            logger.error(f"Failed to get cache: {e}")
            return None

    async def cache_delete(self, key: str) -> bool:
        """Delete cache value"""
        if not self.initialized:
            raise RuntimeError("Memory service not initialized")

        try:
            if self.use_redis and self.redis_store:
                cache_key = f"cache:{key}"
                return await self.redis_store.delete(cache_key)
            else:
                cache_key = f"cache:{key}"
                if cache_key in self.in_memory_store:
                    del self.in_memory_store[cache_key]
                    return True
                return False

        except Exception as e:
            logger.error(f"Failed to delete cache: {e}")
            return False
