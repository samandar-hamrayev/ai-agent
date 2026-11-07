# AI Agent — a ReAct agent built from scratch on the Claude API

[![CI](https://github.com/samandar-hamrayev/ai-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/samandar-hamrayev/ai-agent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-SSE-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

An autonomous agent that reasons, calls tools, and remembers — **without LangChain or any agent framework**.
The loop, tool registry, memory layers, persistence and streaming API are all written by hand so every
moving part is visible and testable.

> Ask it *"What's 15% of the sum of the first 20 primes?"* and watch it plan, call `calculator` /
> `web_search` / `file_tool`, read the results and answer — each step streamed live over Server-Sent Events.

<!-- DEMO: record the built-in UI at http://localhost:8000, save it as docs/demo.gif, then replace this comment with:
<p align="center"><img src="docs/demo.gif" alt="Live ReAct trace in the built-in demo UI" width="760"></p>
-->

---

## Why build it from scratch?

Frameworks hide the parts that break in production: how tool calls are replayed, what happens when the
history window cuts a `tool_use` away from its `tool_result`, how a blocking tool affects an async server,
where state lives between requests. This project implements each of those explicitly:

- **ReAct loop** with parallel tool calls, an iteration budget, and clean handling of refusals and API errors
- **Two-tier memory** — Redis for the live conversation, ChromaDB for semantic recall across sessions
- **Durable run history** in PostgreSQL (async SQLAlchemy + Alembic) — every run and step is queryable
- **Streaming API** — FastAPI + SSE, API-key auth and a Redis-backed rate limiter
- **Sandboxing that is honest about its limits** — path-confined file access, resource-limited code execution

## Architecture

```mermaid
flowchart LR
    client([Client / demo UI]) -- "POST /run (SSE)" --> api

    subgraph app [FastAPI service]
        api[API layer<br/>auth · rate limit · SSE]
        agent[Agent core<br/>ReAct loop]
        planner[Planner<br/>Claude Messages API]
        registry[Tool registry]
        api --> agent
        agent --> planner
        agent --> registry
    end

    planner -- "tools + history" --> claude[(Claude API)]
    registry --> tools[calculator · web_search<br/>file_tool · code_executor]

    agent -- "conversation (TTL)" --> redis[(Redis)]
    api -- "rate-limit counters" --> redis
    agent -- "semantic recall" --> chroma[(ChromaDB)]
    agent -- "runs & steps" --> pg[(PostgreSQL)]
```

### One run, step by step

```mermaid
sequenceDiagram
    participant C as Client
    participant A as Agent
    participant M as Memory (Redis / Chroma)
    participant L as Claude
    participant T as Tools

    C->>A: POST /run {goal, session_id}
    A-->>C: event: run_started
    A->>M: load session history + recall similar past goals
    loop until answer or max_iterations
        A->>L: system + history + tool schemas
        L-->>A: thinking + text + tool_use blocks
        alt Claude wants tools
            A->>T: run every tool_use concurrently
            T-->>A: results (errors flagged is_error)
            A->>M: append assistant turn + one tool_result message
            A-->>C: event: step
        else final answer
            A->>M: store goal → answer for future recall
            A-->>C: event: final
        end
    end
```

## Design decisions

| Decision | Why |
|---|---|
| **Assistant turns are replayed verbatim** (thinking, text and `tool_use` blocks stored as JSON objects) | The API needs real `tool_use` blocks to pair with `tool_result`s, and thinking blocks must be passed back unchanged. Storing them as a string would make Claude read its own tool calls as plain text. |
| **All tool calls of a turn run concurrently; results go back in one user message** | Claude can request several tools at once. Splitting results across messages breaks the pairing and teaches the model to stop parallelising. |
| **History is trimmed at turn boundaries** | Redis `LTRIM` keeps the last N messages, which can orphan a `tool_result`. The API view always starts at a real user message. |
| **Async end-to-end** (`AsyncAnthropic`, `redis.asyncio`, async SQLAlchemy) and blocking tools/Chroma on worker threads | One slow tool or search must not stall every other stream on the server. |
| **Postgres is optional** (`NullRunStore` when `DATABASE_URL` is unset) | The agent runs locally with only Redis; history is an add-on, not a dependency of the loop. |
| **Stable, cache-friendly prompt** — no timestamps or IDs in the system prompt, tools sorted by name, `cache_control` on the system block | Prompt caching is a prefix match; any per-request byte in the prefix would invalidate it. |
| **Adaptive thinking + `effort`**, server-side refusal fallback (beta) | Lets the model decide how much to think per step; `LLM_EFFORT` trades depth for cost. Fallbacks can be switched off with `LLM_FALLBACKS=false`. |
| **Failures become data, not exceptions** | Tool errors are returned to Claude with `is_error: true` so it can recover; API errors end the run with a clear status that is persisted. |

## Tech stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude API — tool use, adaptive thinking, prompt caching |
| API | FastAPI, Server-Sent Events (`sse-starlette`), Pydantic v2 / pydantic-settings |
| Short-term memory | Redis (`redis.asyncio`) — per-session history with TTL, rate limiting |
| Long-term memory | ChromaDB — cosine similarity over past goal/answer pairs |
| Persistence | PostgreSQL, SQLAlchemy 2 (async, `asyncpg`), Alembic migrations |
| Quality | pytest (+ fakeredis, aiosqlite), ruff, pre-commit, gitleaks |
| Delivery | Docker (multi-stage, non-root, read-only FS), docker compose, GitHub Actions, Dependabot |

## Quick start

### With Docker (recommended)

```bash
git clone https://github.com/samandar-hamrayev/ai-agent.git
cd ai-agent
cp .env.example .env          # set ANTHROPIC_API_KEY and POSTGRES_PASSWORD
docker compose up --build -d
```

Open **http://localhost:8000** for the live demo UI, or **http://localhost:8000/docs** for the OpenAPI docs.
Migrations run automatically on start.

### Locally

```bash
make install                  # .venv + dev dependencies + pre-commit hooks
docker compose up -d redis postgres
make migrate
make run                      # uvicorn with --reload on :8000
```

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/run` | Run the agent; streams `run_started` → `step`* → `final` (or `error`) |
| `GET` | `/runs?session_id=&limit=` | Recent runs |
| `GET` | `/runs/{run_id}` | A run with all of its steps and tool calls |
| `DELETE` | `/sessions/{session_id}` | Forget a session's conversation |
| `GET` | `/tools` | Registered tools and their JSON schemas |
| `GET` | `/health` | Redis / Postgres status |

```bash
curl -N -X POST localhost:8000/run \
  -H 'Content-Type: application/json' \
  -d '{"goal": "What is (17*23) + 4**5?", "session_id": "demo"}'
```

```text
event: run_started
data: {"run_id": "6f1c…", "session_id": "demo"}

event: step
data: {"step_number": 1, "thought": "I'll compute this.", "actions": [{"tool_name": "calculator",
       "tool_input": {"expression": "(17*23) + 4**5"}, "success": true, "result": "SUCCESS: 1415"}], …}

event: final
data: {"step_number": 2, "is_final": true, "final_answer": "(17×23) + 4⁵ = 1415.", …}
```

Reuse the same `session_id` to continue a conversation; set `API_KEY` to require an `X-API-Key` header.

## Configuration

All settings are environment variables (see [`.env.example`](.env.example)).

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Claude API key |
| `LLM_MODEL` | `claude-opus-5` | Model ID |
| `LLM_EFFORT` | `high` | `low` · `medium` · `high` · `xhigh` · `max` |
| `LLM_FALLBACKS` | `true` | Server-side refusal fallback (beta) |
| `MAX_ITERATIONS` | `10` | ReAct iteration budget per run |
| `REDIS_URL` | `redis://localhost:6379/0` | Conversation memory + rate limiter |
| `DATABASE_URL` | unset | Enables run history (`postgresql+asyncpg://…`) |
| `ENABLE_CODE_EXECUTOR` | `false` | Registers the Python execution tool |
| `API_KEY` | unset | Require `X-API-Key` on every request |
| `RATE_LIMIT_PER_MINUTE` | `20` | Per client, `0` disables |
| `CORS_ORIGINS` | `["http://localhost:8000"]` | Allowed browser origins |

## Tools

| Tool | What it does | Safety |
|---|---|---|
| `calculator` | Arithmetic via an AST walker | No `eval`; exponent capped |
| `web_search` | DuckDuckGo HTML results (no API key) | 15 s timeout, top 5 results |
| `file_tool` | Read/write text files | Confined to `WORKSPACE_DIR`; `..`, absolute paths, sibling-prefix dirs and symlink escapes are rejected |
| `code_executor` | Runs a Python script | Opt-in; see below |

Adding a tool means subclassing `BaseTool` (name, description, JSON schema, `run`) and registering it — the
registry turns it into a Claude tool definition.

## Security model

- **Secrets** come only from the environment; `.env` is git- and docker-ignored; gitleaks runs in pre-commit and CI.
- **Container**: non-root user, read-only root filesystem, `no-new-privileges`, all Linux capabilities dropped.
- **API**: optional API key (constant-time comparison), per-client rate limit in Redis, restricted CORS, input validation on goal and session id.
- **Code execution is defence in depth, not a sandbox.** The child process gets an empty environment, isolated mode (`-I -S`: no site-packages), a throwaway directory and rlimits on CPU, memory, file size and open files. It still runs as the same OS user as the API, so a determined script could read that process's memory or `/proc`. That's why it is **off by default**; a production deployment would run it in a separate, network-less container (gVisor / Firecracker) or use Anthropic's server-side code execution tool.

## Testing

```bash
make test        # fast suite: no network, no embedding model download
make test-all    # + ChromaDB integration tests
make lint
```

The suite (85 tests) drives the full loop with a scripted planner, real Redis semantics (fakeredis), a real
SQL store (SQLite), and the HTTP layer through `httpx.ASGITransport`. CI also runs Alembic
upgrade → downgrade → upgrade against PostgreSQL, builds the Docker image and scans history for secrets.

## Project structure

```text
agent/
  config.py        settings (pydantic-settings)
  core.py          ReAct loop, steps, tool-result plumbing
  planner.py       Claude request building + response parsing
  memory.py        Redis short-term + ChromaDB long-term memory
  storage.py       SQLAlchemy models and run store
  tool_router.py   tool registry, concurrent execution
api/
  main.py          FastAPI app factory, SSE endpoint, auth, rate limit
  static/          zero-dependency demo UI
tools/             calculator, web_search, file_tool, code_executor
alembic/           migrations
tests/             unit + API tests
```

## Roadmap

- [ ] Stream Claude's text token-by-token inside a step
- [ ] Per-tool timeouts and an approval hook for side-effecting tools
- [ ] Token/cost accounting per run (`usage` → Postgres)
- [ ] Run `code_executor` in a dedicated sandbox container

## License

[MIT](LICENSE) © Samandar Hamrayev
