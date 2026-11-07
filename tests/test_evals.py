"""Tests for the eval harness itself — grading, selection and a scripted end-to-end run (no API calls)."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from evals.run import TASKS_FILE, check, markdown_report, run_suite, select_tasks, summarize
from tests.conftest import ScriptedPlanner, text_turn, tool_turn

KNOWN_CHECKS = {"contains", "not_contains", "regex", "number", "file_contains"}


def test_task_file_is_well_formed():
    tasks = json.loads(TASKS_FILE.read_text())
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids))
    for task in tasks:
        assert task["goal"] and task["checks"]
        assert {c["type"] for c in task["checks"]} <= KNOWN_CHECKS


@pytest.mark.parametrize(
    ("spec", "answer", "expected"),
    [
        ({"type": "contains", "value": "Example Domain"}, "The heading is example domain.", True),
        ({"type": "not_contains", "value": "root:x:"}, "I can't read files outside the workspace.", True),
        ({"type": "not_contains", "value": "root:x:"}, "root:x:0:0:root", False),
        ({"type": "regex", "value": r"\bNOT FOUND\b"}, "NOT FOUND", True),
        ({"type": "number", "value": 1628.89, "tolerance": 0.01}, "You will have $1,628.89.", True),
        ({"type": "number", "value": 4}, "It is 3.", False),
    ],
)
def test_checks(spec, answer, expected, tmp_path):
    assert check(spec, "question", answer, tmp_path)[0] is expected


def test_number_check_ignores_numbers_from_the_question(tmp_path):
    # "7" appears in the goal; an answer that only restates it must not pass.
    assert not check({"type": "number", "value": 7}, "What is 2**2000 mod 7?", "mod 7 is tricky", tmp_path)[0]


def test_file_check(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "t.txt").write_text("alpha beta")
    assert check({"type": "file_contains", "path": "notes/t.txt", "value": "alpha"}, "", "", tmp_path)[0]
    assert not check({"type": "file_contains", "path": "missing.txt", "value": "x"}, "", "", tmp_path)[0]


def test_select_tasks():
    tasks = [{"id": "a", "tags": ["math"]}, {"id": "b", "tags": ["web", "network"]}, {"id": "c", "tags": ["files"]}]
    assert [t["id"] for t in select_tasks(tasks, None, None, offline=True)] == ["a", "c"]
    assert [t["id"] for t in select_tasks(tasks, None, "web,files", offline=False)] == ["b", "c"]
    assert [t["id"] for t in select_tasks(tasks, "c", None, offline=False)] == ["c"]


async def test_scripted_suite_end_to_end():
    tasks = [
        {
            "id": "file-roundtrip",
            "goal": "Write 'alpha beta' to notes/t.txt and count characters.",
            "checks": [
                {"type": "number", "value": 10},
                {"type": "file_contains", "path": "notes/t.txt", "value": "alpha beta"},
            ],
            "expect_tools": ["file_tool"],
        },
        {"id": "wrong", "goal": "What is 2+2?", "checks": [{"type": "number", "value": 4}]},
    ]
    scripts = {
        "file-roundtrip": [
            tool_turn(("w", "file_tool", {"action": "write", "path": "notes/t.txt", "content": "alpha beta"})),
            text_turn("It has 10 characters."),
        ],
        "wrong": [text_turn("It is 5.")],
    }
    queue = [scripts["file-roundtrip"], scripts["wrong"]]
    results = await run_suite(tasks, lambda: ScriptedPlanner(queue.pop(0)), concurrency=1)

    assert [r.passed for r in results] == [True, False]
    assert results[0].tool_calls == ["file_tool"]
    summary = summarize(results)
    assert summary["passed"] == 1 and summary["accuracy"] == 0.5
    assert Decimal(summary["total_cost_usd"]) > 0
    report = markdown_report(summary, results)
    assert "1/2" in report and "❌" in report


async def test_budget_stops_new_tasks():
    tasks = [{"id": f"t{i}", "goal": "q", "checks": [{"type": "contains", "value": "a"}]} for i in range(3)]
    results = await run_suite(tasks, lambda: ScriptedPlanner([text_turn("a")]), concurrency=1, max_cost=Decimal("0"))
    assert results == []
