"""Token accounting and cost estimation for Claude calls."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel

# USD per million tokens (input, output). Cache writes (5-minute TTL) bill at 1.25x input,
# cache reads at 0.1x input. Source: Anthropic pricing page — update when prices change.
PRICES_PER_MTOK: dict[str, tuple[Decimal, Decimal]] = {
    "claude-fable-5-1": (Decimal("10"), Decimal("50")),
    "claude-fable-5": (Decimal("10"), Decimal("50")),
    "claude-opus-5-5": (Decimal("4"), Decimal("20")),
    "claude-opus-5": (Decimal("5"), Decimal("25")),
    "claude-opus-4-8": (Decimal("5"), Decimal("25")),
    "claude-opus-4-7": (Decimal("5"), Decimal("25")),
    "claude-opus-4-6": (Decimal("5"), Decimal("25")),
    "claude-sonnet-5": (Decimal("2"), Decimal("10")),
    "claude-sonnet-4-6": (Decimal("3"), Decimal("15")),
    "claude-haiku-4-5": (Decimal("1"), Decimal("5")),
}
_CACHE_WRITE_MULTIPLIER = Decimal("1.25")
_CACHE_READ_MULTIPLIER = Decimal("0.1")
_MTOK = Decimal(1_000_000)


def price_for(model: str | None) -> tuple[Decimal, Decimal] | None:
    """Prices for a model id, tolerating provider prefixes / date suffixes (longest match wins)."""
    if not model:
        return None
    for known in sorted(PRICES_PER_MTOK, key=len, reverse=True):
        if known in model:
            return PRICES_PER_MTOK[known]
    return None


class Usage(BaseModel):
    """Token counts for one or more model calls, plus the estimated cost."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: Decimal = Decimal("0")
    # False when a call used a model we have no price for — cost is then a lower bound.
    cost_complete: bool = True

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            cost_complete=self.cost_complete and other.cost_complete,
        )

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Share of prompt tokens served from cache — the number that proves caching works."""
        total = self.total_input_tokens
        return round(self.cache_read_tokens / total, 4) if total else 0.0

    @classmethod
    def from_counts(
        cls, model: str | None, input_tokens: int, output_tokens: int, cache_read: int, cache_write: int
    ) -> Usage:
        prices = price_for(model)
        cost = Decimal("0")
        if prices:
            input_price, output_price = prices
            cost = (
                input_tokens * input_price
                + cache_write * input_price * _CACHE_WRITE_MULTIPLIER
                + cache_read * input_price * _CACHE_READ_MULTIPLIER
                + output_tokens * output_price
            ) / _MTOK
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            cost_usd=cost.quantize(Decimal("0.000001")),
            cost_complete=prices is not None,
        )

    @classmethod
    def from_api(cls, usage: Any, model: str | None) -> Usage:
        """Build from a Messages API `usage` object.

        With server-side fallbacks one response can include several attempts on different
        models; `usage.iterations` lists each one, and the top-level numbers only cover the
        attempt that produced the message — so bill per iteration when they are present.
        """
        if usage is None:
            return cls()
        iterations = getattr(usage, "iterations", None) or []
        if iterations:
            total = cls()
            for it in iterations:
                total += cls.from_counts(
                    getattr(it, "model", None) or model,
                    it.input_tokens or 0,
                    it.output_tokens or 0,
                    getattr(it, "cache_read_input_tokens", 0) or 0,
                    getattr(it, "cache_creation_input_tokens", 0) or 0,
                )
            return total
        return cls.from_counts(
            model,
            usage.input_tokens or 0,
            usage.output_tokens or 0,
            getattr(usage, "cache_read_input_tokens", 0) or 0,
            getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )
