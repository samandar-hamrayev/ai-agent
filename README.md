# 🤖 AI Agent — Autonomous ReAct Agent from Scratch

A **production-grade, framework-free AI Agent** built with pure Python. No LangChain, no LlamaIndex — everything is built from scratch to demonstrate deep understanding of how agentic AI systems work internally.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI + SSE                         │
│                  POST /run (stream)                      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   Agent Core (ReAct Loop)                │
│                                                          │
│   ┌──────┐    ┌──────┐    ┌──────┐    ┌──────────┐      │
│   │THINK │───▶│DECIDE│───▶│ ACT  │───▶│ OBSERVE  │──┐   │
│   └──────┘    └──────┘    └──────┘    └──────────┘  │   │
│       ▲                                              │   │
│       └──────────────────────────────────────────────┘   │
└───────┬────────────────┬────────────────┬───────────────┘
        │                │                │
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Planner    │ │ Tool Router  │ │    Memory     │
│  (Claude API)│ │  (Registry)  │ │ (Redis+Chroma)│
└──────────────┘ └──────┬───────┘ └──────────────┘
                        │
         ┌──────┬───────┼───────┬──────┐
         ▼      ▼       ▼       ▼      ▼
      ┌─────┐┌─────┐┌──────┐┌──────┐
      │Calc ││File ││Search││ Code │
      │     ││Tool ││      ││ Exec │
      └─────┘└─────┘└──────┘└──────┘
```

---

## ReAct Loop — How It Works

The agent follows the **ReAct (Reason + Act)** pattern:

1. **THINK** — The LLM receives the goal + context + available tools
2. **DECIDE** — The LLM decides which tool to use (or provide a final answer)
3. **ACT** — The agent executes the chosen tool
4. **OBSERVE** — The tool result is added to agent memory
5. **REPEAT** — The loop continues until the goal is achieved

### Example

```
Goal: "What is the current Bitcoin price and how much is 0.5 BTC worth?"

Step 1 — THINK:  "I need to search for Bitcoin price first"
Step 1 — ACT:    web_search(query="Bitcoin price USD")
Step 1 — RESULT: "Bitcoin is $67,432 as of today"

Step 2 — THINK:  "Now I can calculate 0.5 × 67432"
Step 2 — ACT:    calculator(expression="0.5 * 67432")
Step 2 — RESULT: "33716.0"

Step 3 — FINAL:  "Bitcoin is $67,432. 0.5 BTC = $33,716 USD."
```

---

## Tech Stack

| Component     | Technology                           |
|---------------|--------------------------------------|
| Language      | Python 3.11                          |
| LLM           | Anthropic Claude API (tool_use)      |
| API           | FastAPI + SSE (Server-Sent Events)   |
| Short Memory  | Redis (conversation history)         |
| Long Memory   | ChromaDB (semantic vector store)     |
| Container     | Docker + docker-compose              |
| Validation    | Pydantic v2                          |

---

## Project Structure

```
ai-agent/
├── agent/
│   ├── __init__.py
│   ├── core.py            # Main agent loop (ReAct pattern)
│   ├── planner.py         # LLM interaction, tool selection
│   ├── memory.py          # Short-term (Redis) + Long-term (ChromaDB)
│   └── tool_router.py     # Tool registry and dispatcher
├── tools/
│   ├── __init__.py
│   ├── base.py            # Abstract BaseTool + ToolResult
│   ├── web_search.py      # DuckDuckGo web search
│   ├── code_executor.py   # Sandboxed Python execution
│   ├── file_tool.py       # Safe file read/write
│   └── calculator.py      # Math expression evaluator
├── api/
│   ├── __init__.py
│   └── main.py            # FastAPI app with SSE streaming
├── tests/
│   ├── test_tools.py      # Tool unit tests
│   ├── test_agent.py      # Agent core tests
│   └── test_memory.py     # Memory tests
├── .env                   # Environment variables
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone and Configure

```bash
git clone https://github.com/your-username/ai-agent.git
cd ai-agent
cp .env.example .env
```

Edit `.env` and set your Anthropic API key:
```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 2. Run with Docker

```bash
docker-compose up --build
```

This starts:
- **Redis** on port 6379 (short-term memory)
- **PostgreSQL** on port 5432 (logs)
- **Agent API** on port 8000

### 3. Run Locally (Development)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start Redis (required)
docker run -d -p 6379:6379 redis:7-alpine

# Start the API
uvicorn api.main:app --reload --port 8000
```

---

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "agent_ready": true,
  "tools": ["calculator", "code_executor", "file_tool", "web_search"]
}
```

### List Tools

```bash
curl http://localhost:8000/tools
```

### Run Agent (SSE Stream)

```bash
curl -N -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"goal": "What is 25 * 4 + 100?"}'
```

Each SSE event contains an `AgentStep`:
```json
{
  "step_number": 1,
  "thought": "I need to calculate this expression.",
  "tool_name": "calculator",
  "tool_input": {"expression": "25 * 4 + 100"},
  "tool_result": "SUCCESS: 200",
  "is_final": false,
  "timestamp": "2025-01-01T00:00:00Z"
}
```

With session continuity:
```bash
curl -N -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"goal": "Write a Python script that prints fibonacci numbers", "session_id": "my-session-123"}'
```

---

## Available Tools

### 🔢 Calculator (`calculator`)
Safely evaluates math expressions using AST parsing. No `eval()` or `exec()`.

**Supports:** `+`, `-`, `*`, `/`, `//`, `%`, `**`, parentheses

```json
{"expression": "(25 * 4) + 100"}
```

### 🔍 Web Search (`web_search`)
Searches the web via DuckDuckGo. No API key required.

```json
{"query": "Python 3.12 new features"}
```

### 💻 Code Executor (`code_executor`)
Runs Python code in an isolated subprocess with a 10-second timeout.

```json
{"code": "print(sum(range(100)))"}
```

### 📁 File Tool (`file_tool`)
Reads and writes files within `/tmp/agent_workspace/` (sandboxed).

```json
{"action": "write", "path": "notes.txt", "content": "Hello!"}
{"action": "read", "path": "notes.txt"}
```

---

## How to Add a Custom Tool

Create a new file in `tools/` that extends `BaseTool`:

```python
from tools.base import BaseTool, ToolResult


class MyCustomTool(BaseTool):
    name: str = "my_tool"
    description: str = "Does something useful."

    def run(self, input_param: str) -> ToolResult:
        # Your logic here
        result = f"Processed: {input_param}"
        return ToolResult(success=True, output=result)

    def get_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "input_param": {
                    "type": "string",
                    "description": "The input to process.",
                }
            },
            "required": ["input_param"],
        }
```

Then register it in `api/main.py`:
```python
from tools.my_tool import MyCustomTool

registry.register(MyCustomTool())
```

---

## Testing

```bash
# Run all tests (no external services needed for most)
python -m pytest tests/ -v

# Skip slow/network tests
python -m pytest tests/ -v -m "not slow and not network"

# Run specific test file
python -m pytest tests/test_tools.py -v
```

---

## Security

- **Code executor** runs in a subprocess with timeout — never uses `eval()`
- **File tool** restricted to `/tmp/agent_workspace/` — directory traversal blocked
- **Calculator** uses AST parsing — no arbitrary code execution
- **API keys** never logged or exposed in responses
- **Input validation** via Pydantic on all endpoints

---

## Memory System

### Short-Term Memory (Redis)
- Stores conversation history per session
- Key: `agent:session:{session_id}:messages`
- TTL: 1 hour (configurable)
- Auto-trimmed to last 50 messages

### Long-Term Memory (ChromaDB)
- Stores important facts as vector embeddings
- Semantic search across all past sessions
- Automatically saves goal-answer pairs
- Injected into system prompt for context

---

## License

MIT
