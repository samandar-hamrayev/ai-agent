from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agent.core import Agent, AgentConfig, AgentStep
from agent.memory import LongTermMemory, ShortTermMemory
from agent.planner import Planner
from agent.tool_router import ToolRegistry
from tools import CalculatorTool, CodeExecutorTool, FileTool, WebSearchTool

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_agent: Agent | None = None


def _build_agent() -> Agent:
    """Wire up tools, memory, planner → Agent."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WebSearchTool())
    registry.register(CodeExecutorTool())
    registry.register(FileTool())
    logger.info("Registered %d tools: %s", len(registry), registry.tool_names)

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    short_term = ShortTermMemory(redis_url=redis_url)

    chromadb_path = os.getenv("CHROMADB_PATH", "/tmp/agent_chromadb")
    long_term = LongTermMemory(persist_directory=chromadb_path)

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        logger.warning("ANTHROPIC_API_KEY is not set — agent won't be able to reason.")

    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    planner = Planner(api_key=api_key, model=model)

    config = AgentConfig(max_iterations=int(os.getenv("MAX_ITERATIONS", "10")), model=model)

    return Agent(
        tool_registry=registry,
        short_term_memory=short_term,
        long_term_memory=long_term,
        planner=planner,
        config=config,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _agent
    logger.info("🚀 Starting AI Agent API...")
    _agent = _build_agent()
    logger.info("✅ Agent ready.")
    yield
    logger.info("🛑 Shutting down.")


app = FastAPI(
    title="AI Agent API",
    description="Autonomous AI Agent with ReAct pattern, tool use, and memory.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / Response schemas ---

class RunRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=5000)
    session_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    agent_ready: bool
    tools: list[str]


class ToolInfo(BaseModel):
    name: str
    description: str
    input_schema: dict


# --- Endpoints ---

@app.post("/run")
async def run_agent(request: RunRequest) -> EventSourceResponse:
    """Stream agent steps via SSE as it reasons through the goal."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    session_id = request.session_id or str(uuid.uuid4())

    async def stream() -> AsyncGenerator[dict, None]:
        try:
            async for step in _agent.run(goal=request.goal, session_id=session_id):
                yield {"event": "agent_step", "data": json.dumps(step.model_dump(), default=str)}
        except Exception as e:
            logger.error("Agent error: %s", e, exc_info=True)
            error = AgentStep(step_number=0, thought=str(e), is_final=True, final_answer=f"Error: {e}")
            yield {"event": "agent_error", "data": json.dumps(error.model_dump(), default=str)}

    return EventSourceResponse(stream())


@app.get("/health", response_model=HealthResponse)
async def health_check():
    if _agent is None:
        return HealthResponse(status="degraded", agent_ready=False, tools=[])
    return HealthResponse(status="ok", agent_ready=True, tools=_agent._tools.tool_names)


@app.get("/tools", response_model=list[ToolInfo])
async def list_tools():
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized.")
    return [
        ToolInfo(name=s["name"], description=s["description"], input_schema=s["input_schema"])
        for s in _agent._tools.get_all_schemas()
    ]
