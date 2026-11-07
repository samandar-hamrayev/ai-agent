from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

_DEFAULT_TTL = 3600  # 1 hour
_MAX_HISTORY = 60  # keep the last N messages per session


def _is_turn_start(message: dict[str, Any]) -> bool:
    """A user message that isn't a tool_result — the only safe place to cut history."""
    if message["role"] != "user":
        return False
    content = message["content"]
    if isinstance(content, str):
        return True
    return not any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)


class ShortTermMemory:
    """Per-session conversation history in Redis, with TTL and safe trimming."""

    def __init__(self, redis: Redis, ttl: int = _DEFAULT_TTL, max_history: int = _MAX_HISTORY) -> None:
        self._redis = redis
        self._ttl = ttl
        self._max_history = max_history

    @staticmethod
    def _key(session_id: str) -> str:
        return f"agent:session:{session_id}:messages"

    async def add_message(self, session_id: str, role: str, content: str | list[dict[str, Any]]) -> None:
        # Content is stored as-is (string or list of blocks) so tool_use / thinking blocks
        # go back to the API as real blocks, not as a JSON-encoded string.
        message = {"role": role, "content": content, "timestamp": datetime.now(UTC).isoformat()}
        key = self._key(session_id)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, json.dumps(message, default=str))
            pipe.ltrim(key, -self._max_history, -1)
            pipe.expire(key, self._ttl)
            await pipe.execute()

    async def get_history(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        start = -limit if limit else 0
        raw = await self._redis.lrange(self._key(session_id), start, -1)
        return [json.loads(m) for m in raw]

    async def get_messages_for_api(self, session_id: str) -> list[dict[str, Any]]:
        """History ready for the Messages API.

        LTRIM can cut a tool_use/tool_result pair in half, which the API rejects. Drop
        everything before the first real user turn so the window always starts cleanly.
        """
        history = [{"role": m["role"], "content": m["content"]} for m in await self.get_history(session_id)]
        for i, message in enumerate(history):
            if _is_turn_start(message):
                return history[i:]
        return []

    async def clear(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))
        logger.info("Cleared memory for session %s", session_id)


class LongTermMemory:
    """ChromaDB-backed semantic memory shared across sessions.

    Chroma's client is synchronous, so the async wrappers push calls onto a worker thread.
    """

    def __init__(self, persist_directory: str | Path, collection_name: str = "agent_memory") -> None:
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("LongTermMemory ready: collection='%s', docs=%d", collection_name, self._collection.count())

    def store(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        doc_id = str(uuid.uuid4())
        meta = {**(metadata or {}), "stored_at": datetime.now(UTC).isoformat()}
        self._collection.add(documents=[text], metadatas=[meta], ids=[doc_id])
        return doc_id

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        total = self._collection.count()
        if total == 0:
            return []
        results = self._collection.query(query_texts=[query], n_results=min(top_k, total))
        entries = []
        for i, doc in enumerate(results["documents"][0] if results["documents"] else []):
            entries.append(
                {
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                }
            )
        return entries

    async def astore(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        return await asyncio.to_thread(self.store, text, metadata)

    async def asearch(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.search, query, top_k)

    @property
    def count(self) -> int:
        return self._collection.count()
