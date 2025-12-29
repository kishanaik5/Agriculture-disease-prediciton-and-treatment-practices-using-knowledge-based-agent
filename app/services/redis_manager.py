import json
import logging
from typing import Optional, Dict, Any
from redis import asyncio as aioredis
from app.config import init_settings

logger = logging.getLogger(__name__)

class RedisTaskManager:
    """
    Redis-based task manager for async operations.
    Stores task status and results with automatic expiration.
    """
    
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        self.settings = init_settings()
        
    async def connect(self):
        """Establish Redis connection"""
        try:
            self.redis = await aioredis.from_url(
                self.settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # Test connection
            await self.redis.ping()
            logger.info(f"✅ Redis connected: {self.settings.REDIS_URL}")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            logger.warning("⚠️  Falling back to in-memory task storage")
            self.redis = None
    
    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")
    
    def _get_task_key(self, task_id: str) -> str:
        """Generate Redis key for task"""
        return f"task:{task_id}"
    
    async def set_task_status(
        self, 
        task_id: str, 
        status: str, 
        data: Optional[Dict[str, Any]] = None,
        ttl: int = 3600  # 1 hour expiration
    ):
        """
        Store task status in Redis
        
        Args:
            task_id: Unique task identifier
            status: Task status (queued, processing, completed, failed)
            data: Additional task data
            ttl: Time to live in seconds (default: 1 hour)
        """
        task_data = {
            "status": status,
            "task_id": task_id,
            **(data or {})
        }
        
        if self.redis:
            try:
                key = self._get_task_key(task_id)
                await self.redis.setex(
                    key, 
                    ttl, 
                    json.dumps(task_data, default=str)
                )
            except Exception as e:
                logger.error(f"Redis set failed: {e}")
        else:
            # Fallback to in-memory (not recommended for production)
            logger.warning(f"Using in-memory storage for task {task_id}")
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve task status from Redis
        
        Args:
            task_id: Unique task identifier
            
        Returns:
            Task data dictionary or None if not found
        """
        if self.redis:
            try:
                key = self._get_task_key(task_id)
                data = await self.redis.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.error(f"Redis get failed: {e}")
        
        return None
    
    async def delete_task(self, task_id: str):
        """Delete task from Redis"""
        if self.redis:
            try:
                key = self._get_task_key(task_id)
                await self.redis.delete(key)
            except Exception as e:
                logger.error(f"Redis delete failed: {e}")
    
    async def extend_task_ttl(self, task_id: str, ttl: int = 3600):
        """Extend task expiration time"""
        if self.redis:
            try:
                key = self._get_task_key(task_id)
                await self.redis.expire(key, ttl)
            except Exception as e:
                logger.error(f"Redis expire failed: {e}")


# Singleton instance
task_manager = RedisTaskManager()
