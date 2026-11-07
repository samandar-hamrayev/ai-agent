"""Human-in-the-loop approval for side-effecting tool calls.

The agent loop and the HTTP request that carries the human's decision can land on different
API workers, so the hand-off goes through Redis:

1. the agent marks the call as pending (key with TTL) and blocks on BLPOP;
2. POST /runs/{run_id}/approvals/{tool_use_id} checks the call is pending and RPUSHes the decision;
3. BLPOP returns it — or times out, which counts as a denial.
"""

from __future__ import annotations

import json
import logging
from typing import Literal, Protocol

from pydantic import BaseModel, Field
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

Decision = Literal["approved", "denied", "timed_out"]


class ApprovalDecision(BaseModel):
    decision: Decision
    reason: str | None = Field(default=None, max_length=500)

    @property
    def approved(self) -> bool:
        return self.decision == "approved"


class ApprovalGate(Protocol):
    async def open(self, run_id: str, tool_use_id: str, wait_seconds: int) -> None: ...
    async def wait(self, run_id: str, tool_use_id: str, wait_seconds: int) -> ApprovalDecision: ...
    async def submit(self, run_id: str, tool_use_id: str, decision: ApprovalDecision) -> bool: ...


class RedisApprovalGate:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    @staticmethod
    def _pending_key(run_id: str, tool_use_id: str) -> str:
        return f"agent:approval:{run_id}:{tool_use_id}:pending"

    @staticmethod
    def _decision_key(run_id: str, tool_use_id: str) -> str:
        return f"agent:approval:{run_id}:{tool_use_id}:decision"

    async def open(self, run_id: str, tool_use_id: str, wait_seconds: int) -> None:
        """Mark the call as awaiting a decision (before the client is told about it)."""
        await self._redis.set(self._pending_key(run_id, tool_use_id), "1", ex=wait_seconds + 30)

    async def wait(self, run_id: str, tool_use_id: str, wait_seconds: int) -> ApprovalDecision:
        decision_key = self._decision_key(run_id, tool_use_id)
        try:
            item = await self._redis.blpop([decision_key], timeout=wait_seconds)
        finally:
            await self._redis.delete(self._pending_key(run_id, tool_use_id), decision_key)
        if item is None:
            logger.info("Approval for %s/%s timed out", run_id, tool_use_id)
            return ApprovalDecision(decision="timed_out", reason=f"No decision within {wait_seconds}s.")
        return ApprovalDecision.model_validate(json.loads(item[1]))

    async def submit(self, run_id: str, tool_use_id: str, decision: ApprovalDecision) -> bool:
        """Deliver a decision. Returns False if nothing is waiting for it (unknown or expired)."""
        pending = self._pending_key(run_id, tool_use_id)
        # GETDEL makes the decision single-use: a second POST for the same call is rejected.
        if await self._redis.getdel(pending) is None:
            return False
        decision_key = self._decision_key(run_id, tool_use_id)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.rpush(decision_key, decision.model_dump_json())
            pipe.expire(decision_key, 60)
            await pipe.execute()
        return True
