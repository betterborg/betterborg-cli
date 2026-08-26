"""Generation-specific execution commitment estimates.

The first release intentionally ships conspicuous dummy priors. Repository-
local completions replace them gradually, reaching a fully local estimate at
five measured tasks in a complexity bucket.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from uuid import UUID

from betterborg_cli.agent_runtime.base import AgentUsage, BillingMode
from betterborg_cli.agent_runtime.pricing import (
    PRICE_CATALOG_SOURCES,
    PRICE_CATALOG_VERSION,
    estimate_api_cost_usd,
)
from betterborg_cli.repository_config import AgentChoice, RepositoryConfig
from betterborg_cli.store.models import (
    TaskCompletionSample,
    TaskComplexity,
    TaskRecord,
)

DUMMY_PRIOR_VERSION = "dummy-v1"
DUMMY_PRIOR_LABEL = (
    "DUMMY DATA — bootstrap prior only; replace with local completion history"
)
LOCAL_BLEND_SAMPLE_COUNT = 5
_LEVELS = tuple(TaskComplexity)
_PHASES = ("coding", "review", "merge")
_DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-8",
    "claude": "claude-opus-4-8",
    "codex": "gpt-5",
    "openai": "gpt-5",
}


@dataclass(frozen=True, slots=True)
class EstimateRange:
    """P50/P80 values in one documented unit; either may be unknown."""

    p50: float | None
    p80: float | None


@dataclass(frozen=True, slots=True)
class UsageRange:
    """P50/P80 aggregate token usage for one execution phase."""

    p50: AgentUsage
    p80: AgentUsage


@dataclass(frozen=True, slots=True)
class ComplexityPrior:
    """Bootstrap time and token assumptions for one task complexity."""

    time_seconds: EstimateRange
    coding_usage: UsageRange
    review_usage: UsageRange
    merge_usage: UsageRange


def _usage(
    tokens_input: int,
    tokens_cache_read: int,
    tokens_cache_write: int,
    tokens_output: int,
) -> AgentUsage:
    return AgentUsage(
        tokens_input=tokens_input,
        tokens_cache_read=tokens_cache_read,
        tokens_cache_write=tokens_cache_write,
        tokens_output=tokens_output,
    )


# These are synthetic, monotonic bootstrap assumptions, not measurements. The
# values are deliberately simple so users cannot mistake precision for evidence.
DUMMY_PRIORS: dict[TaskComplexity, ComplexityPrior] = {
    TaskComplexity.SMALL: ComplexityPrior(
        time_seconds=EstimateRange(1_800.0, 3_600.0),
        coding_usage=UsageRange(
            _usage(80_000, 160_000, 20_000, 16_000),
            _usage(160_000, 320_000, 40_000, 32_000),
        ),
        review_usage=UsageRange(
            _usage(40_000, 80_000, 10_000, 8_000),
            _usage(80_000, 160_000, 20_000, 16_000),
        ),
        merge_usage=UsageRange(
            _usage(40_000, 80_000, 10_000, 8_000),
            _usage(80_000, 160_000, 20_000, 16_000),
        ),
    ),
    TaskComplexity.MEDIUM: ComplexityPrior(
        time_seconds=EstimateRange(3_600.0, 7_200.0),
        coding_usage=UsageRange(
            _usage(160_000, 320_000, 40_000, 32_000),
            _usage(320_000, 640_000, 80_000, 64_000),
        ),
        review_usage=UsageRange(
            _usage(80_000, 160_000, 20_000, 16_000),
            _usage(160_000, 320_000, 40_000, 32_000),
        ),
        merge_usage=UsageRange(
            _usage(80_000, 160_000, 20_000, 16_000),
            _usage(160_000, 320_000, 40_000, 32_000),
        ),
    ),
    TaskComplexity.LARGE: ComplexityPrior(
        time_seconds=EstimateRange(7_200.0, 14_400.0),
        coding_usage=UsageRange(
            _usage(320_000, 640_000, 80_000, 64_000),
            _usage(640_000, 1_280_000, 160_000, 128_000),
        ),
        review_usage=UsageRange(
            _usage(160_000, 320_000, 40_000, 32_000),
            _usage(320_000, 640_000, 80_000, 64_000),
        ),
        merge_usage=UsageRange(
            _usage(160_000, 320_000, 40_000, 32_000),
            _usage(320_000, 640_000, 80_000, 64_000),
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class PhaseBilling:
    """Billing identity for one estimated execution phase."""

    phase: str
    mode: BillingMode | None
    provider: str | None
    model: str | None


def phase_billing_from_config(config: RepositoryConfig) -> tuple[PhaseBilling, ...]:
    """Resolve non-secret agent-phase billing facts from tracked config."""
    return (
        _choice_billing("coding", config.agents.coding),
        _choice_billing("review", config.agents.review),
        _choice_billing("merge", config.agents.merge),
    )


def _choice_billing(phase: str, choice: AgentChoice) -> PhaseBilling:
    adapter = choice.adapter
    if adapter in {"anthropic", "openai"}:
        return PhaseBilling(
            phase, BillingMode.API, adapter, choice.model or _DEFAULT_MODELS[adapter]
        )
    if adapter in {"claude", "codex"}:
        return PhaseBilling(
            phase,
            BillingMode.SUBSCRIPTION,
            None,
            choice.model or _DEFAULT_MODELS[adapter],
        )
    return PhaseBilling(phase, None, None, choice.model)


def estimate_generation(
    generation_id: UUID,
    tasks: Iterable[TaskRecord | TaskComplexity | None],
    samples: Iterable[TaskCompletionSample],
    billing: Iterable[PhaseBilling],
    *,
    priors: Mapping[TaskComplexity, ComplexityPrior] = DUMMY_PRIORS,
) -> dict[str, object]:
    """Estimate total agent work and API cost for one immutable generation."""
    task_items = list(tasks)
    counts = {level: 0 for level in _LEVELS}
    unsized = 0
    for item in task_items:
        complexity = item.complexity if isinstance(item, TaskRecord) else item
        if complexity in counts:
            counts[complexity] += 1
        else:
            unsized += 1

    local = list(samples)
    phase_billings = {item.phase: item for item in billing}
    for phase in _PHASES:
        phase_billings.setdefault(phase, PhaseBilling(phase, None, None, None))
    per_complexity = []
    total_p50 = 0.0
    total_p80 = 0.0
    time_unknown_tasks = unsized
    total_sample_size = 0
    api_p50 = 0.0
    api_p80 = 0.0
    subscription_phases = sorted(
        item.phase
        for item in phase_billings.values()
        if item.mode is BillingMode.SUBSCRIPTION
    )
    unknown_phases = sorted(
        item.phase for item in phase_billings.values() if item.mode is None
    )
    api_present = bool(task_items) and any(
        item.mode is BillingMode.API for item in phase_billings.values()
    )
    api_unknown = bool(task_items and unknown_phases) or bool(
        unsized and api_present
    )

    for level in _LEVELS:
        level_samples = [sample for sample in local if sample.complexity is level]
        durations = [
            sample.duration_seconds
            for sample in level_samples
            if sample.duration_seconds is not None
        ]
        prior = priors.get(level)
        time_range, time_source = _blended_range(
            durations,
            prior.time_seconds if prior is not None else None,
        )
        count = counts[level]
        total_sample_size += len(durations)
        if count:
            if time_range.p50 is None or time_range.p80 is None:
                time_unknown_tasks += count
            else:
                total_p50 += count * time_range.p50
                total_p80 += count * time_range.p80

        usage_ranges: dict[str, tuple[UsageRange | None, int]] = {}
        for phase in _PHASES:
            usages = [
                usage
                for sample in level_samples
                if (usage := getattr(sample, f"{phase}_usage")) is not None
                and _has_complete_tokens(usage)
            ]
            prior_usage = (
                getattr(prior, f"{phase}_usage") if prior is not None else None
            )
            usage_ranges[phase] = (_blended_usage(usages, prior_usage), len(usages))

        per_complexity.append(
            {
                "complexity": level.value,
                "task_count": count,
                "sample_size": len(durations),
                "token_sample_size": {
                    phase: sample_count
                    for phase, (_range, sample_count) in usage_ranges.items()
                },
                "token_source": {
                    phase: _source_for_samples(
                        sample_count,
                        (
                            getattr(prior, f"{phase}_usage")
                            if prior is not None
                            else None
                        ),
                    )
                    for phase, (_range, sample_count) in usage_ranges.items()
                },
                "source": time_source,
                "time": _range_item(time_range, "seconds"),
            }
        )

        if not count:
            continue
        for phase, selection in phase_billings.items():
            if selection.mode is BillingMode.SUBSCRIPTION:
                continue
            if selection.mode is None:
                api_unknown = True
                continue
            usage_range = usage_ranges.get(phase, (None, 0))[0]
            if (
                usage_range is None
                or selection.provider is None
                or selection.model is None
            ):
                api_unknown = True
                continue
            p50 = estimate_api_cost_usd(
                selection.provider, selection.model, usage_range.p50
            )
            p80 = estimate_api_cost_usd(
                selection.provider, selection.model, usage_range.p80
            )
            if p50 is None or p80 is None:
                api_unknown = True
            else:
                api_p50 += count * p50
                api_p80 += count * p80

    api_models = [
        {
            "phase": item.phase,
            "provider": item.provider,
            "model": item.model,
        }
        for item in phase_billings.values()
        if item.mode is BillingMode.API
    ]
    api_range = (
        None
        if not api_present or api_unknown
        else _range_item(EstimateRange(api_p50, api_p80), "USD")
    )
    return {
        "generation_id": str(generation_id),
        "task_mix": {
            **{level.value: counts[level] for level in _LEVELS},
            "unsized": unsized,
        },
        "estimable_tasks": len(task_items) - time_unknown_tasks,
        "sample_size": total_sample_size,
        "per_complexity": per_complexity,
        "time": {
            **_range_item(
                EstimateRange(None, None)
                if time_unknown_tasks
                else EstimateRange(total_p50, total_p80),
                "seconds",
            ),
            "kind": "total_agent_work",
            "calendar_time": False,
            "unknown_tasks": time_unknown_tasks,
        },
        "billing": {
            "api": {
                "estimate": api_range,
                "unknown": api_unknown,
                "models": api_models,
                "pricing_catalog_version": PRICE_CATALOG_VERSION,
                "pricing_sources": PRICE_CATALOG_SOURCES,
            },
            "subscription": {
                "included": bool(subscription_phases),
                "phases": subscription_phases,
                "usd": None,
            },
            "unknown_phases": unknown_phases,
        },
        "provenance": {
            "prior_version": DUMMY_PRIOR_VERSION,
            "prior_label": DUMMY_PRIOR_LABEL,
            "local_blend_sample_count": LOCAL_BLEND_SAMPLE_COUNT,
        },
    }


def _blended_range(
    samples: list[float], prior: EstimateRange | None
) -> tuple[EstimateRange, str]:
    local = (
        EstimateRange(_percentile(samples, 0.5), _percentile(samples, 0.8))
        if samples
        else None
    )
    if local is None:
        return prior or EstimateRange(None, None), (
            "dummy_prior" if prior is not None else "unknown"
        )
    if prior is None or len(samples) >= LOCAL_BLEND_SAMPLE_COUNT:
        return local, "local"
    weight = len(samples) / LOCAL_BLEND_SAMPLE_COUNT
    return (
        EstimateRange(
            _blend(prior.p50, local.p50, weight),
            _blend(prior.p80, local.p80, weight),
        ),
        "dummy_prior+local",
    )


def _blended_usage(
    samples: list[AgentUsage], prior: UsageRange | None
) -> UsageRange | None:
    if not samples:
        return prior
    local = UsageRange(
        _usage_percentile(samples, 0.5),
        _usage_percentile(samples, 0.8),
    )
    if prior is None or len(samples) >= LOCAL_BLEND_SAMPLE_COUNT:
        return local
    weight = len(samples) / LOCAL_BLEND_SAMPLE_COUNT
    return UsageRange(
        _blend_usage(prior.p50, local.p50, weight),
        _blend_usage(prior.p80, local.p80, weight),
    )


def _source_for_samples(sample_size: int, prior: object | None) -> str:
    if sample_size == 0:
        return "dummy_prior" if prior is not None else "unknown"
    if prior is None or sample_size >= LOCAL_BLEND_SAMPLE_COUNT:
        return "local"
    return "dummy_prior+local"


def _usage_percentile(samples: list[AgentUsage], percentile: float) -> AgentUsage:
    return AgentUsage(
        **{
            field: int(
                _percentile(
                    [float(getattr(sample, field)) for sample in samples],
                    percentile,
                )
            )
            for field in _TOKEN_FIELDS
        }
    )


def _blend_usage(prior: AgentUsage, local: AgentUsage, weight: float) -> AgentUsage:
    return AgentUsage(
        **{
            field: int(
                round(
                    _blend(
                        float(getattr(prior, field)),
                        float(getattr(local, field)),
                        weight,
                    )
                )
            )
            for field in _TOKEN_FIELDS
        }
    )


_TOKEN_FIELDS = (
    "tokens_input",
    "tokens_output",
    "tokens_cache_read",
    "tokens_cache_write",
)


def _has_complete_tokens(usage: AgentUsage) -> bool:
    return all(getattr(usage, field) is not None for field in _TOKEN_FIELDS)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _blend(prior: float | None, local: float | None, weight: float) -> float | None:
    if prior is None or local is None:
        return None
    return prior * (1.0 - weight) + local * weight


def _range_item(value: EstimateRange, unit: str) -> dict[str, object]:
    return {"p50": value.p50, "p80": value.p80, "unit": unit}
