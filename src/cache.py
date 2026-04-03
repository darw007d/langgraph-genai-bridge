"""
Context Cache Manager for Google GenAI.

Caches system prompts to avoid re-sending them every LLM call.
Google charges input tokens once per cache creation (TTL-based),
then near-free for subsequent calls using the cached content.
"""

import time
import logging
from typing import Optional

logger = logging.getLogger("langgraph-genai-bridge")


class ContextCacheManager:
    """Manages Google GenAI context caches with automatic TTL refresh."""

    def __init__(self, client, model: str = "gemini-2.5-flash", ttl_seconds: int = 3600):
        """
        Args:
            client: google.genai.Client instance
            model: Gemini model name
            ttl_seconds: Cache time-to-live (default 1 hour)
        """
        self.client = client
        self.model = model
        self.ttl_seconds = ttl_seconds
        self._cache_id: Optional[str] = None
        self._cache_ts: float = 0
        self._cached_prompt_hash: Optional[int] = None

    def get_or_create(self, system_prompt: str) -> Optional[str]:
        """
        Get existing cache or create a new one for the system prompt.
        Returns the cache name (model ID) to use in generate_content calls.
        Returns None if caching fails (caller should use uncached model).
        """
        prompt_hash = hash(system_prompt)
        now = time.time()

        # Reuse if same prompt and not expired
        if (self._cache_id
                and self._cached_prompt_hash == prompt_hash
                and (now - self._cache_ts) < self.ttl_seconds * 0.9):  # 90% of TTL
            return self._cache_id

        try:
            from google.genai import types as genai_types

            cache = self.client.caches.create(
                model=self.model,
                config=genai_types.CreateCachedContentConfig(
                    system_instruction=system_prompt,
                    display_name="langgraph-genai-bridge-cache",
                    ttl=f"{self.ttl_seconds}s",
                ),
            )
            self._cache_id = cache.name
            self._cache_ts = now
            self._cached_prompt_hash = prompt_hash
            logger.info(f"Context cache created: {cache.name} (TTL: {self.ttl_seconds}s)")
            return cache.name

        except Exception as e:
            logger.warning(f"Context cache creation failed: {e}")
            self._cache_id = None
            return None

    def invalidate(self):
        """Force cache invalidation (e.g., when system prompt changes)."""
        self._cache_id = None
        self._cache_ts = 0
        self._cached_prompt_hash = None

    @property
    def is_cached(self) -> bool:
        """Check if a valid cache exists."""
        return (self._cache_id is not None
                and (time.time() - self._cache_ts) < self.ttl_seconds * 0.9)
