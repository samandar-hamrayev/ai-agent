from __future__ import annotations

import hmac
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import anthropic
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from sse_starlette.sse import EventSourceResponse

from agent.config import Settings, get_settings
from agent.core import Agent, AgentConfig, AgentStep
from agent.memory import LongTermMemory, ShortTermMemory
from agent.planner import Planner
from agent.storage import SqlRunStore
from agent.tool_router import ToolRegistry
from tools import CalculatorTool, CodeExecutorTool, FileTool, WebSearchTool

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class Services:
    agent: Agent
    redis: Redis
    short_term: ShortTermMemory
    store: SqlRunStore | None = None


def build_registry(settings: Settings) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WebSearchTool())
    registry.register(FileTool(workspace=settings.workspace_dir))
    if settings.enable_code_executor:
        registry.register(CodeExecutorTool())
    return registry


async def build_services(settings: Settings) -> Services:
    """Wire tools, memory, persistence and the Claude planner into an Agent."""
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    short_term = ShortTermMemory(redis, ttl=settings.session_ttl_seconds)
    settings.chromadb_path.mkdir(parents=True, exist_ok=True)
    long_term = LongTermMemory(persist_directory=settings.chromadb_path)
    store = SqlRunStore.from_url(settings.database_url) if settings.database_url else None

    if not settings.anthropic_api_key.get_secret_value():
        logger.warning("ANTHROPIC_API_KEY is not set — the agent won't be able to call Claude.")
    planner = Planner(
        client=anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value() or None),
        model=settings.llm_model,
        effort=settings.llm_effort,
        max_tokens=settings.llm_max_tokens,
        fallbacks=settings.llm_fallbacks,
    )
    agent = Agent(
        tool_registry=build_registry(settings),
        short_term_memory=short_term,
        long_term_memory=long_term,
        planner=planner,
        run_store=store,
        config=AgentConfig(max_iterations=settings.max_iterations, save_to_long_term=settings.save_to_long_term),
    )
    return Services(agent=agent, redis=redis, short_term=short_term, store=store)


# --- Request / response schemas ---


class RunRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=5000)
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class HealthResponse(BaseModel):
    status: str
    redis: bool
    database: bool | None
    tools: list[str]


class ToolInfo(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class StepOut(BaseModel):
    step_number: int
    thought: str | None
    actions: list[dict[str, Any]]
    is_final: bool


class RunOut(BaseModel):
    id: str
    session_id: str
    goal: str
    status: str
    final_answer: str | None
    model: str | None
    created_at: str
    finished_at: str | None
    steps: list[StepOut] = []


def _run_out(run: Any, with_steps: bool = False) -> RunOut:
    return RunOut(
        id=run.id,
        session_id=run.session_id,
        goal=run.goal,
        status=run.status,
        final_answer=run.final_answer,
        model=run.model,
        created_at=run.created_at.isoformat(),
        finished_at=run.finished_at.isoformat() if run.finished_at else None,
        steps=[
            StepOut(step_number=s.step_number, thought=s.thought, actions=s.actions, is_final=s.is_final)
            for s in run.steps
        ]
        if with_steps
        else [],
    )


# --- Dependencies ---


def get_services(request: Request) -> Services:
    services: Services | None = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Agent not initialized.")
    return services


def require_api_key(request: Request) -> str:
    """When API_KEY is configured, require it in X-API-Key (constant-time comparison)."""
    settings: Settings = request.app.state.settings
    client_id = request.client.host if request.client else "anonymous"
    if settings.api_key is None:
        return client_id
    supplied = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(supplied.encode(), settings.api_key.get_secret_value().encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key.")
    return "api-key:" + supplied[-6:]


async def rate_limit(
    request: Request,
    client_id: Annotated[str, Depends(require_api_key)],
    services: Annotated[Services, Depends(get_services)],
) -> None:
    """Fixed one-minute window per client, counted in Redis so it holds across workers."""
    limit = request.app.state.settings.rate_limit_per_minute
    if limit == 0:
        return
    key = f"agent:ratelimit:{client_id}:{int(time.time() // 60)}"
    count = await services.redis.incr(key)
    if count == 1:
        await services.redis.expire(key, 60)
    if count > limit:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit exceeded, try again in a minute.")


ServicesDep = Annotated[Services, Depends(get_services)]
Authorized = Depends(require_api_key)


def _require_store(services: Services) -> SqlRunStore:
    if services.store is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run history is disabled (DATABASE_URL not set).")
    return services.store


# --- App factory ---


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        owns_services = services is None
        app.state.services = services or await build_services(settings)
        logger.info("Agent ready with tools: %s", app.state.services.agent.tools.tool_names)
        yield
        if owns_services:
            await app.state.services.redis.aclose()
            if app.state.services.store:
                await app.state.services.store.close()

    app = FastAPI(
        title="AI Agent API",
        description="Autonomous ReAct agent on the Claude API: tool use, Redis + ChromaDB memory, SSE streaming.",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def demo_page() -> str:
        return (_STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.post("/run", dependencies=[Depends(rate_limit)])
    async def run_agent(body: RunRequest, services: ServicesDep) -> EventSourceResponse:
        """Stream the agent's reasoning as Server-Sent Events: run_started → step* → final | error."""
        session_id = body.session_id or str(uuid.uuid4())
        run_id = str(uuid.uuid4())

        async def stream() -> AsyncGenerator[dict[str, str], None]:
            yield {"event": "run_started", "data": json.dumps({"run_id": run_id, "session_id": session_id})}
            try:
                async for step in services.agent.run(body.goal, session_id=session_id, run_id=run_id):
                    yield {"event": "final" if step.is_final else "step", "data": step.model_dump_json()}
            except Exception:
                logger.exception("Run %s failed", run_id)
                error = AgentStep(run_id=run_id, step_number=0, is_final=True, final_answer="Internal error.")
                yield {"event": "error", "data": error.model_dump_json()}

        return EventSourceResponse(stream())

    @app.get("/runs", response_model=list[RunOut], dependencies=[Authorized])
    async def list_runs(
        services: ServicesDep,
        session_id: str | None = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ) -> list[RunOut]:
        runs = await _require_store(services).list_runs(session_id=session_id, limit=limit)
        return [_run_out(r) for r in runs]

    @app.get("/runs/{run_id}", response_model=RunOut, dependencies=[Authorized])
    async def get_run(run_id: str, services: ServicesDep) -> RunOut:
        run = await _require_store(services).get_run(run_id)
        if run is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found.")
        return _run_out(run, with_steps=True)

    @app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Authorized])
    async def clear_session(session_id: str, services: ServicesDep) -> None:
        await services.short_term.clear(session_id)

    @app.get("/tools", response_model=list[ToolInfo], dependencies=[Authorized])
    async def list_tools(services: ServicesDep) -> list[ToolInfo]:
        return [ToolInfo(**schema) for schema in services.agent.tools.get_all_schemas()]

    @app.get("/health", response_model=HealthResponse)
    async def health(services: ServicesDep) -> HealthResponse:
        redis_ok = database_ok = False
        try:
            redis_ok = bool(await services.redis.ping())
        except Exception as e:
            logger.warning("Redis health check failed: %s", e)
        if services.store is not None:
            try:
                database_ok = await services.store.ping()
            except Exception as e:
                logger.warning("Database health check failed: %s", e)
        healthy = redis_ok and (services.store is None or database_ok)
        return HealthResponse(
            status="ok" if healthy else "degraded",
            redis=redis_ok,
            database=database_ok if services.store is not None else None,
            tools=services.agent.tools.tool_names,
        )

    return app


def _configure_logging() -> None:
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


_configure_logging()
app = create_app()
