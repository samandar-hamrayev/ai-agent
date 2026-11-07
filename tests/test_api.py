"""HTTP-level tests: auth, rate limiting, SSE stream shape and run history endpoints."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from pydantic import SecretStr

from agent.config import Settings
from api.main import Services, create_app
from tests.conftest import text_turn, tool_turn


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for chunk in body.replace("\r\n", "\n").strip().split("\n\n"):
        lines = chunk.split("\n")
        event = next((line[7:] for line in lines if line.startswith("event: ")), None)
        data = "\n".join(line[6:] for line in lines if line.startswith("data: "))
        if event:
            events.append((event, json.loads(data)))
    return events


@pytest.fixture
def client_factory(make_agent, redis, stm, store, gate):
    async def _make(turns, **settings_overrides):
        settings = Settings(_env_file=None, **{"rate_limit_per_minute": 100, **settings_overrides})
        services = Services(agent=make_agent(turns), redis=redis, short_term=stm, store=store, approvals=gate)
        app = create_app(settings, services)
        app.state.services = services  # ASGITransport doesn't run lifespan
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url="http://test")

    return _make


async def test_run_streams_steps_and_persists(client_factory):
    async with await client_factory([tool_turn(("t1", "echo", {"message": "hi"})), text_turn("done")]) as client:
        res = await client.post("/run", json={"goal": "say hi", "session_id": "demo"})
        assert res.status_code == 200
        events = parse_sse(res.text)

        names = [e for e, _ in events]
        assert names == ["run_started", "step", "text", "final"]
        run_id = events[0][1]["run_id"]
        assert events[1][1]["actions"][0]["tool_name"] == "echo"
        assert events[2][1]["text"].strip() == "done"
        final = events[3][1]
        assert final["final_answer"] == "done"
        assert final["run_usage"]["input_tokens"] == 2000

        run = (await client.get(f"/runs/{run_id}")).json()
        assert run["status"] == "completed"
        assert run["usage"]["input_tokens"] == 2000
        assert run["usage"]["cache_hit_rate"] == 0.8
        assert float(run["usage"]["cost_usd"]) > 0
        assert [s["step_number"] for s in run["steps"]] == [1, 2]

        listed = (await client.get("/runs", params={"session_id": "demo"})).json()
        assert [r["id"] for r in listed] == [run_id]


async def test_unknown_run_is_404(client_factory):
    async with await client_factory([]) as client:
        assert (await client.get("/runs/nope")).status_code == 404


async def test_api_key_required_when_configured(client_factory):
    async with await client_factory([text_turn("ok")], api_key=SecretStr("s3cret")) as client:
        assert (await client.get("/tools")).status_code == 401
        assert (await client.post("/run", json={"goal": "x"})).status_code == 401
        assert (await client.get("/tools", headers={"X-API-Key": "wrong"})).status_code == 401
        assert (await client.get("/tools", headers={"X-API-Key": "s3cret"})).status_code == 200


async def test_rate_limit(client_factory):
    async with await client_factory([text_turn("1"), text_turn("2")], rate_limit_per_minute=1) as client:
        assert (await client.post("/run", json={"goal": "a"})).status_code == 200
        assert (await client.post("/run", json={"goal": "b"})).status_code == 429


async def test_validation(client_factory):
    async with await client_factory([]) as client:
        assert (await client.post("/run", json={"goal": ""})).status_code == 422
        assert (await client.post("/run", json={"goal": "x", "session_id": "../etc"})).status_code == 422


async def test_health_and_tools(client_factory):
    async with await client_factory([]) as client:
        health = (await client.get("/health")).json()
        assert health == {"status": "ok", "redis": True, "database": True, "tools": ["boom", "echo", "writer"]}
        tools = (await client.get("/tools")).json()
        assert {t["name"] for t in tools} == {"boom", "echo", "writer"}


async def test_clear_session(client_factory, stm):
    await stm.add_message("s1", "user", "hello")
    async with await client_factory([]) as client:
        assert (await client.delete("/sessions/s1")).status_code == 204
    assert await stm.get_history("s1") == []


async def test_demo_page(client_factory):
    async with await client_factory([]) as client:
        res = await client.get("/")
        assert res.status_code == 200
        assert "ReAct" in res.text


async def test_approval_flow_over_http(client_factory, redis):
    turns = [tool_turn(("w1", "writer", {"text": "hello"})), text_turn("written")]
    async with await client_factory(turns) as client:

        async def approve_when_pending():
            for _ in range(200):
                keys = await redis.keys("agent:approval:*:w1:pending")
                if keys:
                    run_id = keys[0].split(":")[2]
                    res = await client.post(f"/runs/{run_id}/approvals/w1", json={"approve": True})
                    assert res.status_code == 204
                    again = await client.post(f"/runs/{run_id}/approvals/w1", json={"approve": False})
                    assert again.status_code == 404  # single use
                    return
                await asyncio.sleep(0.01)
            raise AssertionError("approval was never requested")

        approver = asyncio.create_task(approve_when_pending())
        res = await client.post("/run", json={"goal": "write"})
        await approver

        events = parse_sse(res.text)
        names = [e for e, _ in events]
        assert "approval_required" in names
        request = dict(events)["approval_required"]
        assert request["tool_name"] == "writer" and request["tool_input"] == {"text": "hello"}
        step = dict(events)["step"]
        assert step["actions"][0]["approval"] == "approved"


async def test_unknown_approval_is_404(client_factory):
    async with await client_factory([]) as client:
        res = await client.post("/runs/r/approvals/t", json={"approve": True})
        assert res.status_code == 404


async def test_approvals_disabled(client_factory, make_agent, redis, stm, store):
    settings = Settings(_env_file=None, require_approval=False)
    services = Services(agent=make_agent([]), redis=redis, short_term=stm, store=store, approvals=None)
    app = create_app(settings, services)
    app.state.services = services
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post("/runs/r/approvals/t", json={"approve": True})
        assert res.status_code == 404
        assert "disabled" in res.json()["detail"]
