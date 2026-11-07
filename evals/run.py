"""Run the agent against a fixed task set and report accuracy, steps, latency and cost.

    python -m evals.run                         # all tasks, settings from .env
    python -m evals.run --tags math,files       # a subset
    python -m evals.run --effort medium --offline   # skip tasks that need the internet

Every run spends real API credit: the script prints the task count first and stops once
--max-cost is exceeded. Results are written to evals/results/<timestamp>.json so prompt or
model changes can be compared run against run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import fakeredis

from agent.config import get_settings
from agent.core import Agent, AgentConfig, AgentStep
from agent.memory import ShortTermMemory
from agent.planner import Planner
from agent.tool_router import ToolRegistry
from agent.usage import Usage
from tools import CalculatorTool, FileTool, WebFetchTool, WebSearchTool

TASKS_FILE = Path(__file__).parent / "tasks.json"
RESULTS_DIR = Path(__file__).parent / "results"
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in _NUMBER.findall(text or "")]


def check(spec: dict[str, Any], goal: str, answer: str, workspace: Path) -> tuple[bool, str]:
    """Grade one check. Returns (passed, explanation)."""
    kind, value = spec["type"], spec.get("value")
    if kind == "contains":
        ok = str(value).lower() in answer.lower()
        return ok, f"contains {value!r}"
    if kind == "not_contains":
        ok = str(value).lower() not in answer.lower()
        return ok, f"does not contain {value!r}"
    if kind == "regex":
        return bool(re.search(value, answer, re.IGNORECASE)), f"matches /{value}/"
    if kind == "number":
        # Numbers restated from the question don't count as the answer.
        in_goal = set(_numbers(goal))
        tolerance = spec.get("tolerance", 1e-9)
        found = [n for n in _numbers(answer) if n not in in_goal]
        ok = any(abs(n - float(value)) <= tolerance for n in found)
        return ok, f"number {value} ± {tolerance}"
    if kind == "file_contains":
        path = workspace / spec["path"]
        ok = path.is_file() and str(value) in path.read_text(encoding="utf-8", errors="replace")
        return ok, f"{spec['path']} contains {value!r}"
    raise ValueError(f"Unknown check type: {kind}")


@dataclass
class TaskResult:
    id: str
    passed: bool
    checks: list[dict[str, Any]]
    answer: str
    steps: int
    tool_calls: list[str]
    seconds: float
    usage: Usage
    stop_reason: str | None
    error: str | None = None
    tags: list[str] = field(default_factory=list)


def build_registry(workspace: Path) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (CalculatorTool(), WebSearchTool(), WebFetchTool(), FileTool(workspace=workspace)):
        registry.register(tool)
    return registry


async def run_task(task: dict[str, Any], make_planner: Callable[[], Any], max_iterations: int) -> TaskResult:
    """Run one task in isolation: fresh workspace, fresh conversation, no long-term memory."""
    with tempfile.TemporaryDirectory(prefix=f"eval-{task['id']}-") as tmp:
        workspace = Path(tmp)
        redis = fakeredis.FakeAsyncRedis(decode_responses=True)
        agent = Agent(
            tool_registry=build_registry(workspace),
            short_term_memory=ShortTermMemory(redis),
            long_term_memory=None,
            planner=make_planner(),
            # Evals run unattended: file writes are confined to the temp workspace.
            config=AgentConfig(max_iterations=max_iterations, save_to_long_term=False, require_approval=False),
        )
        started = time.perf_counter()
        steps: list[AgentStep] = []
        error = None
        try:
            async for event in agent.run(task["goal"]):
                if isinstance(event, AgentStep):
                    steps.append(event)
        except Exception as e:  # an eval must report a crash, not stop the whole suite
            error = f"{type(e).__name__}: {e}"
        finally:
            await redis.aclose()
        seconds = time.perf_counter() - started

        final = steps[-1] if steps else None
        answer = (final.final_answer if final else "") or ""
        graded = []
        for spec in task["checks"]:
            ok, why = check(spec, task["goal"], answer, workspace)
            graded.append({"check": why, "passed": ok})
        tools_used = [a.tool_name for s in steps for a in s.actions]
        for expected in task.get("expect_tools", []):
            graded.append({"check": f"used {expected}", "passed": expected in tools_used})

        return TaskResult(
            id=task["id"],
            passed=error is None and all(g["passed"] for g in graded),
            checks=graded,
            answer=answer,
            steps=len(steps),
            tool_calls=tools_used,
            seconds=round(seconds, 2),
            usage=(final.run_usage if final and final.run_usage else Usage()),
            stop_reason=final.stop_reason if final else None,
            error=error,
            tags=task.get("tags", []),
        )


async def run_suite(
    tasks: list[dict[str, Any]],
    make_planner: Callable[[], Any],
    max_iterations: int = 10,
    concurrency: int = 3,
    max_cost: Decimal | None = None,
) -> list[TaskResult]:
    semaphore = asyncio.Semaphore(concurrency)
    spent = Decimal("0")
    results: list[TaskResult] = []

    async def one(task: dict[str, Any]) -> None:
        nonlocal spent
        async with semaphore:
            if max_cost is not None and spent >= max_cost:
                return
            result = await run_task(task, make_planner, max_iterations)
            spent += result.usage.cost_usd
            results.append(result)
            mark = "PASS" if result.passed else "FAIL"
            print(f"  {mark}  {result.id:<28} {result.steps} steps  ${result.usage.cost_usd:.4f}  {result.seconds}s")

    await asyncio.gather(*(one(t) for t in tasks))
    order = {t["id"]: i for i, t in enumerate(tasks)}
    return sorted(results, key=lambda r: order[r.id])


def summarize(results: list[TaskResult]) -> dict[str, Any]:
    total = Usage()
    for r in results:
        total += r.usage
    n = len(results) or 1
    return {
        "tasks": len(results),
        "passed": sum(r.passed for r in results),
        "accuracy": round(sum(r.passed for r in results) / n, 3),
        "avg_steps": round(sum(r.steps for r in results) / n, 2),
        "avg_seconds": round(sum(r.seconds for r in results) / n, 2),
        "total_cost_usd": str(total.cost_usd),
        "avg_cost_usd": str((total.cost_usd / n).quantize(Decimal("0.000001"))),
        "cache_hit_rate": total.cache_hit_rate,
        "input_tokens": total.total_input_tokens,
        "output_tokens": total.output_tokens,
    }


def markdown_report(summary: dict[str, Any], results: list[TaskResult]) -> str:
    lines = [
        f"**Accuracy {summary['passed']}/{summary['tasks']} ({summary['accuracy']:.0%})** · "
        f"avg {summary['avg_steps']} steps · avg {summary['avg_seconds']}s · "
        f"total ${summary['total_cost_usd']} · cache hit {summary['cache_hit_rate']:.0%}",
        "",
        "| Task | Result | Steps | Tools | Cost | Failed checks |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        failed = "; ".join(c["check"] for c in r.checks if not c["passed"]) or (r.error or "")
        lines.append(
            f"| {r.id} | {'✅' if r.passed else '❌'} | {r.steps} | {', '.join(r.tool_calls) or '—'} "
            f"| ${r.usage.cost_usd:.4f} | {failed} |"
        )
    return "\n".join(lines)


def select_tasks(tasks: list[dict[str, Any]], ids: str | None, tags: str | None, offline: bool) -> list[dict[str, Any]]:
    if ids:
        wanted = set(ids.split(","))
        tasks = [t for t in tasks if t["id"] in wanted]
    if tags:
        wanted_tags = set(tags.split(","))
        tasks = [t for t in tasks if wanted_tags & set(t.get("tags", []))]
    if offline:
        tasks = [t for t in tasks if "network" not in t.get("tags", [])]
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", help="comma-separated task ids")
    parser.add_argument("--tags", help="comma-separated tags (any match)")
    parser.add_argument("--offline", action="store_true", help="skip tasks tagged 'network'")
    parser.add_argument("--model", help="override LLM_MODEL")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], help="override LLM_EFFORT")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-cost", type=Decimal, default=Decimal("2.00"), help="stop starting tasks past this USD")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.anthropic_api_key.get_secret_value():
        raise SystemExit("ANTHROPIC_API_KEY is not set — evals call the real API.")
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
    model = args.model or settings.llm_model
    effort = args.effort or settings.llm_effort

    def make_planner() -> Planner:
        return Planner(client, model=model, effort=effort, max_tokens=settings.llm_max_tokens,
                       fallbacks=settings.llm_fallbacks)  # fmt: skip

    tasks = select_tasks(json.loads(TASKS_FILE.read_text()), args.tasks, args.tags, args.offline)
    print(f"Running {len(tasks)} tasks on {model} (effort={effort}), budget ${args.max_cost}")
    results = asyncio.run(run_suite(tasks, make_planner, args.max_iterations, args.concurrency, args.max_cost))
    summary = summarize(results)
    report = markdown_report(summary, results)
    print("\n" + report)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{model}-{effort}.json"
    payload = {
        "model": model,
        "effort": effort,
        "summary": summary,
        "results": [
            {**{k: v for k, v in r.__dict__.items() if k != "usage"}, "usage": r.usage.model_dump(mode="json")}
            for r in results
        ],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
