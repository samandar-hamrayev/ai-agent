from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import chromadb
import redis

logger = logging.getLogger(__name__)

_DEFAULT_TTL = 3600      # 1 hour
_MAX_HISTORY = 50        # keep last N messages per session


class ShortTermMemory:
    """Redis-backed conversation memory with TTL and auto-trimming."""

    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = _DEFAULT_TTL) -> None:
        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._ttl = ttl

    def _key(self, session_id: str) -> str:
        return f"agent:session:{session_id}:messages"

    def add_message(self, session_id: str, role: str, content: Any) -> None:
        message = {
            "role": role,
            "content": content if isinstance(content, str) else json.dumps(content, default=str),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        key = self._key(session_id)

        pipe = self._redis.pipeline()
        pipe.rpush(key, json.dumps(message))
        pipe.ltrim(key, -_MAX_HISTORY, -1)
        pipe.expire(key, self._ttl)
        pipe.execute()

    def get_history(self, session_id: str, limit: int | None = None) -> list[dict]:
        key = self._key(session_id)
        if limit:
            raw = self._redis.lrange(key, -limit, -1)
        else:
            raw = self._redis.lrange(key, 0, -1)
        return [json.loads(msg) for msg in raw]

    def clear(self, session_id: str) -> None:
        self._redis.delete(self._key(session_id))
        logger.info("Cleared memory for session %s", session_id)

    def get_messages_for_api(self, session_id: str) -> list[dict]:
        """Return history without timestamps — ready for Claude API."""
        history = self.get_history(session_id)
        return [{"role": m["role"], "content": m["content"]} for m in history]


class LongTermMemory:
    """ChromaDB-backed semantic memory for cross-session knowledge retrieval."""

    def __init__(self, persist_directory: str = "/tmp/agent_chromadb", collection_name: str = "agent_memory") -> None:
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("LongTermMemory ready: collection='%s', docs=%d", collection_name, self._collection.count())

    def store(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        doc_id = str(uuid.uuid4())
        meta = metadata or {}
        meta["stored_at"] = datetime.now(timezone.utc).isoformat()

        self._collection.add(documents=[text], metadatas=[meta], ids=[doc_id])
        return doc_id

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count()),
        )

        entries = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                entries.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                })
        return entries

    @property
    def count(self) -> int:
        return self._collection.count()
