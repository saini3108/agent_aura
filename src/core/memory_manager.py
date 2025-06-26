"""
Memory Management System
========================

Implements short-term and long-term memory for the ValiCred-AI system
using Redis for short-term and SQLAlchemy for long-term persistence.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import uuid

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """Represents a memory entry"""
    id: str
    workflow_id: str
    step_name: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime] = None
    priority: int = 0

class ShortTermMemory:
    """In-memory short-term storage for active workflows"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_client = None
        self.local_cache: Dict[str, MemoryEntry] = {}
        self.max_local_entries = 1000
        
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Connected to Redis for short-term memory")
            except Exception as e:
                logger.warning(f"Redis connection failed, using local cache: {e}")
                self.redis_client = None
    
    def store(self, workflow_id: str, step_name: str, content: Dict[str, Any], 
              ttl_minutes: int = 60) -> str:
        """Store data in short-term memory"""
        entry_id = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(minutes=ttl_minutes)
        
        entry = MemoryEntry(
            id=entry_id,
            workflow_id=workflow_id,
            step_name=step_name,
            content=content,
            metadata={"ttl_minutes": ttl_minutes},
            created_at=datetime.now(),
            expires_at=expires_at
        )
        
        if self.redis_client:
            try:
                key = f"stm:{workflow_id}:{step_name}:{entry_id}"
                self.redis_client.setex(
                    key, 
                    ttl_minutes * 60, 
                    json.dumps(asdict(entry), default=str)
                )
                return entry_id
            except Exception as e:
                logger.error(f"Redis store failed: {e}")
        
        # Fallback to local cache
        self._cleanup_local_cache()
        self.local_cache[entry_id] = entry
        return entry_id
    
    def retrieve(self, workflow_id: str, step_name: Optional[str] = None) -> List[MemoryEntry]:
        """Retrieve data from short-term memory"""
        results = []
        
        if self.redis_client:
            try:
                pattern = f"stm:{workflow_id}:{step_name or '*'}:*"
                for key in self.redis_client.scan_iter(match=pattern):
                    data = self.redis_client.get(key)
                    if data:
                        entry_dict = json.loads(data)
                        entry_dict['created_at'] = datetime.fromisoformat(entry_dict['created_at'])
                        if entry_dict.get('expires_at'):
                            entry_dict['expires_at'] = datetime.fromisoformat(entry_dict['expires_at'])
                        results.append(MemoryEntry(**entry_dict))
                return results
            except Exception as e:
                logger.error(f"Redis retrieve failed: {e}")
        
        # Fallback to local cache
        for entry in self.local_cache.values():
            if entry.workflow_id == workflow_id:
                if step_name is None or entry.step_name == step_name:
                    if entry.expires_at is None or entry.expires_at > datetime.now():
                        results.append(entry)
        
        return results
    
    def _cleanup_local_cache(self):
        """Clean up expired entries from local cache"""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.local_cache.items()
            if entry.expires_at and entry.expires_at <= now
        ]
        
        for key in expired_keys:
            del self.local_cache[key]
        
        # If cache is too large, remove oldest entries
        if len(self.local_cache) > self.max_local_entries:
            sorted_entries = sorted(
                self.local_cache.items(),
                key=lambda x: x[1].created_at
            )
            entries_to_remove = len(self.local_cache) - self.max_local_entries
            for key, _ in sorted_entries[:entries_to_remove]:
                del self.local_cache[key]

class LongTermMemory:
    """Persistent long-term storage for workflow history and learning"""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url
        self.persistent_store: Dict[str, List[MemoryEntry]] = {}
        
        # In a production setup, this would use SQLAlchemy
        # For now, using in-memory storage with periodic persistence
        
    def store_workflow_completion(self, workflow_id: str, final_state: Dict[str, Any]) -> str:
        """Store completed workflow for long-term learning"""
        entry_id = str(uuid.uuid4())
        
        entry = MemoryEntry(
            id=entry_id,
            workflow_id=workflow_id,
            step_name="workflow_completion",
            content=final_state,
            metadata={
                "completion_time": datetime.now().isoformat(),
                "success": final_state.get('workflow_status') == 'completed'
            },
            created_at=datetime.now()
        )
        
        if workflow_id not in self.persistent_store:
            self.persistent_store[workflow_id] = []
        
        self.persistent_store[workflow_id].append(entry)
        return entry_id
    
    def get_workflow_history(self, limit: int = 50) -> List[MemoryEntry]:
        """Get workflow history for learning and analysis"""
        all_entries = []
        for entries in self.persistent_store.values():
            all_entries.extend(entries)
        
        # Sort by creation time, most recent first
        all_entries.sort(key=lambda x: x.created_at, reverse=True)
        return all_entries[:limit]
    
    def get_similar_workflows(self, current_workflow: Dict[str, Any]) -> List[MemoryEntry]:
        """Find similar past workflows for context"""
        # Simplified similarity matching
        # In production, this would use embedding similarity
        
        current_data_features = current_workflow.get('data', {}).keys()
        similar_workflows = []
        
        for entries in self.persistent_store.values():
            for entry in entries:
                if entry.step_name == "workflow_completion":
                    stored_data_features = entry.content.get('data', {}).keys()
                    
                    # Simple feature overlap calculation
                    overlap = len(set(current_data_features) & set(stored_data_features))
                    if overlap > 0:
                        entry.priority = overlap
                        similar_workflows.append(entry)
        
        # Sort by similarity score
        similar_workflows.sort(key=lambda x: x.priority, reverse=True)
        return similar_workflows[:5]

class MemoryManager:
    """Unified memory management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.short_term = ShortTermMemory(config.get('redis_url'))
        self.long_term = LongTermMemory(config.get('database_url'))
        
    def store_agent_output(self, workflow_id: str, step_name: str, 
                          output: Dict[str, Any], is_final: bool = False) -> str:
        """Store agent output with appropriate persistence"""
        
        # Always store in short-term memory
        entry_id = self.short_term.store(workflow_id, step_name, output)
        
        # Store in long-term memory if it's a final result
        if is_final:
            self.long_term.store_workflow_completion(workflow_id, output)
        
        return entry_id
    
    def get_workflow_context(self, workflow_id: str) -> Dict[str, Any]:
        """Get comprehensive context for a workflow"""
        
        # Get recent short-term memory
        recent_entries = self.short_term.retrieve(workflow_id)
        
        # Get similar workflows from long-term memory
        current_workflow = {"workflow_id": workflow_id}
        if recent_entries:
            current_workflow["data"] = recent_entries[0].content.get("data", {})
        
        similar_workflows = self.long_term.get_similar_workflows(current_workflow)
        
        return {
            "recent_steps": [
                {
                    "step": entry.step_name,
                    "content": entry.content,
                    "timestamp": entry.created_at.isoformat()
                }
                for entry in recent_entries
            ],
            "similar_workflows": [
                {
                    "workflow_id": entry.workflow_id,
                    "completion_status": entry.content.get("workflow_status"),
                    "similarity_score": entry.priority
                }
                for entry in similar_workflows
            ]
        }
    
    def cleanup_expired_data(self):
        """Clean up expired data from memory systems"""
        self.short_term._cleanup_local_cache()
        
        # Long-term cleanup (keep last 1000 workflows)
        all_workflows = list(self.long_term.persistent_store.keys())
        if len(all_workflows) > 1000:
            oldest_workflows = sorted(
                all_workflows,
                key=lambda x: self.long_term.persistent_store[x][0].created_at
            )[:len(all_workflows) - 1000]
            
            for workflow_id in oldest_workflows:
                del self.long_term.persistent_store[workflow_id]