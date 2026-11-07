"""Tests for token accounting and cost estimation."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from agent.usage import Usage, price_for


def test_opus_5_cost_includes_cache_multipliers():
    # 1M input @ $5, 1M output @ $25, 1M cache write @ $6.25, 1M cache read @ $0.50
    usage = Usage.from_counts("claude-opus-5", 1_000_000, 1_000_000, cache_read=1_000_000, cache_write=1_000_000)
    assert usage.cost_usd == Decimal("36.75")
    assert usage.cost_complete


def test_cache_hit_rate():
    usage = Usage(input_tokens=100, cache_read_tokens=900)
    assert usage.cache_hit_rate == 0.9
    assert Usage().cache_hit_rate == 0.0


def test_unknown_model_flags_incomplete_cost():
    usage = Usage.from_counts("some-future-model", 1000, 1000, 0, 0)
    assert usage.cost_usd == 0
    assert not usage.cost_complete
    assert not (usage + Usage()).cost_complete


def test_longest_prefix_wins():
    assert price_for("claude-opus-5-5") == (Decimal("4"), Decimal("20"))
    assert price_for("claude-opus-5") == (Decimal("5"), Decimal("25"))
    assert price_for("anthropic.claude-sonnet-5") == (Decimal("2"), Decimal("10"))
    assert price_for(None) is None


def test_from_api_bills_each_fallback_iteration_at_its_own_model():
    api_usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=10,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        iterations=[
            SimpleNamespace(
                model="claude-opus-5",
                input_tokens=1_000_000,
                output_tokens=0,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            SimpleNamespace(
                model="claude-opus-4-8",
                input_tokens=0,
                output_tokens=1_000_000,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
        ],
    )
    usage = Usage.from_api(api_usage, "claude-opus-5")
    assert usage.input_tokens == 1_000_000 and usage.output_tokens == 1_000_000
    assert usage.cost_usd == Decimal("30")


def test_from_api_without_usage():
    assert Usage.from_api(None, "claude-opus-5") == Usage()
