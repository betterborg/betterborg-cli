"""Release-verified standard API token prices used for estimates.

The catalog is deliberately static: estimates stay reproducible and a release
review, rather than a live network request, changes the numbers. Rates are USD
per one million tokens. ``None`` means that the provider does not publish a
separate rate for that token class.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from betterborg_cli.agent_runtime.base import AgentUsage

PRICE_CATALOG_VERSION = "2026-08-26"
PRICE_CATALOG_SOURCES = {
    "anthropic": "https://platform.claude.com/docs/en/about-claude/pricing",
    "openai": "https://developers.openai.com/api/docs/models/compare",
}


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """Standard API rates for one provider model, in USD per 1M tokens."""

    input_per_1m: float
    cache_read_per_1m: float | None
    cache_write_per_1m: float | None
    output_per_1m: float


# Only directly supported agent-provider models are listed. Unknown model IDs
# intentionally stay unpriced instead of silently inheriting a family rate.
MODEL_PRICES: dict[tuple[str, str], ModelPrice] = {
    ("anthropic", "claude-opus-5"): ModelPrice(5.00, 0.50, 6.25, 25.00),
    ("anthropic", "claude-opus-4-8"): ModelPrice(5.00, 0.50, 6.25, 25.00),
    ("anthropic", "claude-opus-4-7"): ModelPrice(5.00, 0.50, 6.25, 25.00),
    ("anthropic", "claude-opus-4-6"): ModelPrice(5.00, 0.50, 6.25, 25.00),
    ("anthropic", "claude-sonnet-4-6"): ModelPrice(3.00, 0.30, 3.75, 15.00),
    ("anthropic", "claude-sonnet-4-5"): ModelPrice(3.00, 0.30, 3.75, 15.00),
    ("anthropic", "claude-haiku-4-5"): ModelPrice(1.00, 0.10, 1.25, 5.00),
    ("openai", "gpt-5.6-sol"): ModelPrice(4.00, 0.40, 5.00, 20.00),
    ("openai", "gpt-5.6-terra"): ModelPrice(2.00, 0.20, 2.50, 12.00),
    ("openai", "gpt-5.6-luna"): ModelPrice(0.20, 0.02, 0.25, 1.20),
    ("openai", "gpt-5.5"): ModelPrice(5.00, 0.50, None, 30.00),
    ("openai", "gpt-5.5-pro"): ModelPrice(30.00, None, None, 180.00),
    ("openai", "gpt-5.4"): ModelPrice(2.50, 0.25, None, 15.00),
    ("openai", "gpt-5.4-mini"): ModelPrice(0.75, 0.075, None, 4.50),
    ("openai", "gpt-5.4-nano"): ModelPrice(0.20, 0.02, None, 1.25),
    ("openai", "gpt-5.4-pro"): ModelPrice(30.00, None, None, 180.00),
    ("openai", "gpt-5.3-codex"): ModelPrice(1.75, 0.175, None, 14.00),
    ("openai", "gpt-5.2"): ModelPrice(1.75, 0.175, None, 14.00),
    ("openai", "gpt-5.2-pro"): ModelPrice(21.00, None, None, 168.00),
    ("openai", "gpt-5.1"): ModelPrice(1.25, 0.125, None, 10.00),
    ("openai", "gpt-5"): ModelPrice(1.25, 0.125, None, 10.00),
    ("openai", "gpt-5-mini"): ModelPrice(0.25, 0.025, None, 2.00),
    ("openai", "gpt-5-nano"): ModelPrice(0.05, 0.005, None, 0.40),
    ("openai", "gpt-5-pro"): ModelPrice(15.00, None, None, 120.00),
}

_SNAPSHOT_SUFFIX = re.compile(r"(?:-\d{8}|-\d{4}-\d{2}-\d{2})$")


def lookup_model_price(provider: str, model: str) -> ModelPrice | None:
    """Return an exact or date-suffixed model's standard API price."""
    key = (provider.strip().casefold(), model.strip().casefold().split(" ")[0])
    direct = MODEL_PRICES.get(key)
    if direct is not None:
        return direct
    normalized_model = _SNAPSHOT_SUFFIX.sub("", key[1])
    return MODEL_PRICES.get((key[0], normalized_model))


def estimate_api_cost_usd(
    provider: str, model: str, usage: AgentUsage
) -> float | None:
    """Price complete token usage, returning unknown for any missing input."""
    price = lookup_model_price(provider, model)
    token_values = (
        usage.tokens_input,
        usage.tokens_cache_read,
        usage.tokens_cache_write,
        usage.tokens_output,
    )
    if price is None or any(value is None for value in token_values):
        return None
    rates = (
        price.input_per_1m,
        price.cache_read_per_1m or price.input_per_1m,
        price.cache_write_per_1m or price.input_per_1m,
        price.output_per_1m,
    )
    return sum(
        token * rate / 1_000_000
        for rate, token in zip(rates, token_values, strict=True)
    )
