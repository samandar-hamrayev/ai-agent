"""Durable run history: every agent run and each of its steps, stored in PostgreSQL.

Redis holds the live conversation (short TTL); this is the audit trail you can query later.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, Numeric, String, Text, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, selectinload

from agent.usage import Usage


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    goal: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="running")  # running | completed | failed
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cache_write_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), default=Decimal("0"), server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    steps: Mapped[list[Step]] = relationship(
        back_populates="run", order_by="Step.step_number", cascade="all, delete-orphan"
    )


class Step(Base):
    __tablename__ = "agent_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    step_number: Mapped[int] = mapped_column(Integer)
    thought: Mapped[str | None] = mapped_column(Text, nullable=True)
    actions: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    is_final: Mapped[bool] = mapped_column(Boolean, default=False)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    run: Mapped[Run] = relationship(back_populates="steps")


class RunStore(Protocol):
    async def start_run(self, run_id: str, session_id: str, goal: str) -> None: ...
    async def add_step(self, run_id: str, step: dict[str, Any]) -> None: ...
    async def finish_run(
        self, run_id: str, status: str, final_answer: str | None, model: str | None, usage: Usage | None = None
    ) -> None: ...


class NullRunStore:
    """Used when DATABASE_URL is not configured: the agent still works, nothing is persisted."""

    async def start_run(self, run_id: str, session_id: str, goal: str) -> None:
        return None

    async def add_step(self, run_id: str, step: dict[str, Any]) -> None:
        return None

    async def finish_run(
        self, run_id: str, status: str, final_answer: str | None, model: str | None, usage: Usage | None = None
    ) -> None:
        return None


class SqlRunStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)

    @classmethod
    def from_url(cls, url: str) -> SqlRunStore:
        return cls(create_async_engine(url, pool_pre_ping=True))

    async def create_tables(self) -> None:
        """For tests and local SQLite; production schema is managed by Alembic."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def ping(self) -> bool:
        async with self.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True

    async def start_run(self, run_id: str, session_id: str, goal: str) -> None:
        async with self._sessions.begin() as db:
            db.add(Run(id=run_id, session_id=session_id, goal=goal))

    async def add_step(self, run_id: str, step: dict[str, Any]) -> None:
        async with self._sessions.begin() as db:
            db.add(
                Step(
                    run_id=run_id,
                    step_number=step["step_number"],
                    thought=step.get("thought"),
                    actions=step.get("actions", []),
                    is_final=step.get("is_final", False),
                    usage=step.get("usage"),
                )
            )

    async def finish_run(
        self, run_id: str, status: str, final_answer: str | None, model: str | None, usage: Usage | None = None
    ) -> None:
        async with self._sessions.begin() as db:
            run = await db.get(Run, run_id)
            if run is None:
                return
            run.status = status
            run.final_answer = final_answer
            run.model = model
            run.finished_at = _now()
            if usage is not None:
                run.input_tokens = usage.input_tokens
                run.output_tokens = usage.output_tokens
                run.cache_read_tokens = usage.cache_read_tokens
                run.cache_write_tokens = usage.cache_write_tokens
                run.cost_usd = usage.cost_usd

    async def get_run(self, run_id: str) -> Run | None:
        async with self._sessions() as db:
            return await db.scalar(select(Run).where(Run.id == run_id).options(selectinload(Run.steps)))

    async def list_runs(self, session_id: str | None = None, limit: int = 20) -> list[Run]:
        query = select(Run).order_by(Run.created_at.desc()).limit(limit)
        if session_id:
            query = query.where(Run.session_id == session_id)
        async with self._sessions() as db:
            return list(await db.scalars(query))

    async def close(self) -> None:
        await self.engine.dispose()
