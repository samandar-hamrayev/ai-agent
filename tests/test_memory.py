"""Tests for memory module — ShortTermMemory and LongTermMemory."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────────────
# ShortTermMemory Tests (mocked Redis)
# ──────────────────────────────────────────────────────────────────────

class TestShortTermMemory:
    """Tests for ShortTermMemory with mocked Redis."""

    def _create_memory(self) -> "ShortTermMemory":
        """Create a ShortTermMemory instance with mocked Redis."""
        with patch("agent.memory.redis") as mock_redis_module:
            mock_client = MagicMock()
            mock_pipeline = MagicMock()
            mock_client.pipeline.return_value = mock_pipeline
            mock_redis_module.from_url.return_value = mock_client

            from agent.memory import ShortTermMemory
            memory = ShortTermMemory(redis_url="redis://fake:6379")
            memory._mock_client = mock_client
            memory._mock_pipeline = mock_pipeline
            return memory

    def test_add_message(self):
        memory = self._create_memory()
        memory.add_message("session_1", role="user", content="Hello!")

        # Pipeline should have been called with rpush, ltrim, expire
        pipeline = memory._mock_pipeline
        assert pipeline.rpush.called
        assert pipeline.ltrim.called
        assert pipeline.expire.called
        assert pipeline.execute.called

    def test_add_message_serializes_dict(self):
        memory = self._create_memory()
        memory.add_message("session_1", role="assistant", content={"key": "value"})

        pipeline = memory._mock_pipeline
        call_args = pipeline.rpush.call_args
        # The second arg should be a valid JSON string
        stored = json.loads(call_args[0][1])
        assert stored["role"] == "assistant"

    def test_get_history(self):
        memory = self._create_memory()
        # Setup mock to return serialized messages
        stored_messages = [
            json.dumps({"role": "user", "content": "Hello", "timestamp": "2025-01-01T00:00:00"}),
            json.dumps({"role": "assistant", "content": "Hi!", "timestamp": "2025-01-01T00:00:01"}),
        ]
        memory._mock_client.lrange.return_value = stored_messages

        history = memory.get_history("session_1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_get_history_with_limit(self):
        memory = self._create_memory()
        memory._mock_client.lrange.return_value = []
        memory.get_history("session_1", limit=5)

        # Should use negative indexing for limit
        memory._mock_client.lrange.assert_called_once_with(
            "agent:session:session_1:messages", -5, -1
        )

    def test_clear(self):
        memory = self._create_memory()
        memory.clear("session_1")
        memory._mock_client.delete.assert_called_once_with(
            "agent:session:session_1:messages"
        )

    def test_get_messages_for_api(self):
        memory = self._create_memory()
        stored = [
            json.dumps({"role": "user", "content": "Hi", "timestamp": "2025-01-01T00:00:00"}),
        ]
        memory._mock_client.lrange.return_value = stored

        api_messages = memory.get_messages_for_api("session_1")
        assert len(api_messages) == 1
        assert api_messages[0] == {"role": "user", "content": "Hi"}
        # Should NOT contain timestamp
        assert "timestamp" not in api_messages[0]


# ──────────────────────────────────────────────────────────────────────
# LongTermMemory Tests (real ChromaDB — isolated per test)
# ──────────────────────────────────────────────────────────────────────

class TestLongTermMemory:
    """Tests for LongTermMemory using unique collections per test."""

    _test_dir = Path("/tmp/test_agent_chromadb")
    _test_counter = 0

    def setup_method(self):
        TestLongTermMemory._test_counter += 1

    def teardown_method(self):
        # Clean up after all tests
        pass

    @classmethod
    def teardown_class(cls):
        if cls._test_dir.exists():
            shutil.rmtree(cls._test_dir)

    def _create_memory(self) -> "LongTermMemory":
        import uuid
        from agent.memory import LongTermMemory
        # Use a unique collection name per test to avoid state leaks
        collection_name = f"test_{uuid.uuid4().hex[:8]}"
        return LongTermMemory(
            persist_directory=str(self._test_dir),
            collection_name=collection_name,
        )

    def test_store_and_count(self):
        memory = self._create_memory()
        assert memory.count == 0

        memory.store("Python is a programming language.")
        assert memory.count == 1

        memory.store("FastAPI is a web framework.")
        assert memory.count == 2

    def test_store_returns_id(self):
        memory = self._create_memory()
        doc_id = memory.store("Test entry")
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    def test_store_with_metadata(self):
        memory = self._create_memory()
        doc_id = memory.store(
            "Important fact",
            metadata={"session_id": "abc", "type": "fact"},
        )
        assert doc_id is not None

    def test_search_returns_results(self):
        memory = self._create_memory()
        memory.store("Python is great for data science.")
        memory.store("JavaScript runs in the browser.")
        memory.store("PostgreSQL is a relational database.")

        results = memory.search("data analysis with Python")
        assert len(results) > 0
        # The Python entry should be most relevant
        assert "Python" in results[0]["text"]

    def test_search_empty_collection(self):
        memory = self._create_memory()
        results = memory.search("anything")
        assert results == []

    def test_search_top_k(self):
        memory = self._create_memory()
        for i in range(10):
            memory.store(f"Entry number {i}")

        results = memory.search("entry", top_k=3)
        assert len(results) <= 3

    def test_search_result_structure(self):
        memory = self._create_memory()
        memory.store("Test document", metadata={"source": "test"})

        results = memory.search("test")
        assert len(results) == 1
        assert "text" in results[0]
        assert "metadata" in results[0]
        assert "distance" in results[0]
