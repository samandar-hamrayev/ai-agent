"""Tests for ShortTermMemory (fakeredis) and LongTermMemory (real ChromaDB, marked slow)."""

from __future__ import annotations

import uuid

import pytest

from agent.memory import LongTermMemory, ShortTermMemory


class TestShortTermMemory:
    async def test_roundtrip_keeps_blocks_as_objects(self, stm):
        blocks = [{"type": "tool_use", "id": "t1", "name": "echo", "input": {"message": "hi"}}]
        await stm.add_message("s", "user", "hello")
        await stm.add_message("s", "assistant", blocks)

        messages = await stm.get_messages_for_api("s")
        assert messages == [{"role": "user", "content": "hello"}, {"role": "assistant", "content": blocks}]

    async def test_ttl_is_set(self, stm, redis):
        await stm.add_message("s", "user", "hello")
        assert 0 < await redis.ttl(ShortTermMemory._key("s")) <= 60

    async def test_history_is_trimmed(self, redis):
        memory = ShortTermMemory(redis, max_history=4)
        for i in range(10):
            await memory.add_message("s", "user", f"m{i}")
        history = await memory.get_history("s")
        assert [m["content"] for m in history] == ["m6", "m7", "m8", "m9"]

    async def test_window_never_starts_with_orphan_tool_result(self, redis):
        memory = ShortTermMemory(redis, max_history=3)
        await memory.add_message("s", "user", "goal")
        await memory.add_message("s", "assistant", [{"type": "tool_use", "id": "t", "name": "x", "input": {}}])
        await memory.add_message("s", "user", [{"type": "tool_result", "tool_use_id": "t", "content": "r"}])
        await memory.add_message("s", "assistant", "answer")
        await memory.add_message("s", "user", "follow-up")

        # Raw window is [tool_result, "answer", "follow-up"]; the API view must start at a real user turn.
        assert await memory.get_messages_for_api("s") == [{"role": "user", "content": "follow-up"}]

    async def test_get_history_limit(self, stm):
        for i in range(5):
            await stm.add_message("s", "user", str(i))
        assert [m["content"] for m in await stm.get_history("s", limit=2)] == ["3", "4"]

    async def test_clear(self, stm):
        await stm.add_message("s", "user", "hello")
        await stm.clear("s")
        assert await stm.get_history("s") == []


@pytest.mark.slow
class TestLongTermMemory:
    """Uses ChromaDB's default embedding model (downloaded on first run)."""

    @pytest.fixture
    def memory(self, tmp_path) -> LongTermMemory:
        return LongTermMemory(persist_directory=tmp_path, collection_name=f"test_{uuid.uuid4().hex[:8]}")

    def test_store_and_count(self, memory):
        memory.store("Python is a programming language.")
        memory.store("FastAPI is a web framework.")
        assert memory.count == 2

    def test_search_empty(self, memory):
        assert memory.search("anything") == []

    def test_search_ranks_relevant_first(self, memory):
        memory.store("Python is great for data science.")
        memory.store("JavaScript runs in the browser.")
        memory.store("PostgreSQL is a relational database.")
        results = memory.search("data analysis with Python")
        assert "Python" in results[0]["text"]
        assert {"text", "metadata", "distance"} <= results[0].keys()

    async def test_async_wrappers(self, memory):
        await memory.astore("Redis is an in-memory store.", {"source": "test"})
        results = await memory.asearch("in-memory database", top_k=1)
        assert results[0]["metadata"]["source"] == "test"
