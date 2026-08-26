"""Deterministic execution-estimate and provider-pricing contracts."""

from __future__ import annotations

from uuid import uuid4

import pytest

from betterborg_cli.agent_runtime import AgentUsage, BillingMode
from betterborg_cli.agent_runtime.pricing import estimate_api_cost_usd
from betterborg_cli.execution_estimate import (
    DUMMY_PRIORS,
    PhaseBilling,
    estimate_generation,
)
from betterborg_cli.store import TaskCompletionSample, TaskComplexity


def _usage(value: int = 1_000) -> AgentUsage:
    return AgentUsage(
        tokens_input=value,
        tokens_output=value,
        tokens_cache_read=value,
        tokens_cache_write=value,
    )


def _sample(complexity: TaskComplexity, duration: float) -> TaskCompletionSample:
    return TaskCompletionSample(
        generation_id=uuid4(),
        task_id=uuid4(),
        complexity=complexity,
        duration_seconds=duration,
        coding_usage=_usage(int(duration)),
        review_usage=_usage(int(duration)),
    )


def _billing(
    coding: BillingMode | None = BillingMode.SUBSCRIPTION,
    review: BillingMode | None = BillingMode.SUBSCRIPTION,
    *,
    model: str = "gpt-5",
) -> tuple[PhaseBilling, ...]:
    def phase(name: str, mode: BillingMode | None) -> PhaseBilling:
        return PhaseBilling(
            name,
            mode,
            "openai" if mode is BillingMode.API else None,
            model,
        )

    return phase("coding", coding), phase("review", review)


def test_zero_samples_uses_prominently_versioned_dummy_prior() -> None:
    estimate = estimate_generation(
        uuid4(),
        [TaskComplexity.SMALL, TaskComplexity.MEDIUM],
        [],
        _billing(),
    )

    assert estimate["sample_size"] == 0
    assert estimate["task_mix"] == {
        "small": 1,
        "medium": 1,
        "large": 0,
        "unsized": 0,
    }
    assert estimate["time"] == {
        "p50": 5_400.0,
        "p80": 10_800.0,
        "unit": "seconds",
        "kind": "total_agent_work",
        "calendar_time": False,
        "unknown_tasks": 0,
    }
    assert estimate["provenance"]["prior_version"] == "dummy-v1"
    assert estimate["provenance"]["prior_label"].startswith("DUMMY DATA")
    assert [item["source"] for item in estimate["per_complexity"]] == [
        "dummy_prior",
        "dummy_prior",
        "dummy_prior",
    ]


def test_sparse_samples_blend_gradually_and_five_samples_become_local() -> None:
    sparse = estimate_generation(
        uuid4(),
        [TaskComplexity.SMALL],
        [_sample(TaskComplexity.SMALL, 100.0)],
        _billing(),
    )
    sparse_small = sparse["per_complexity"][0]
    assert sparse_small["sample_size"] == 1
    assert sparse_small["source"] == "dummy_prior+local"
    assert sparse_small["time"] == {
        "p50": pytest.approx(1_460.0),
        "p80": pytest.approx(2_900.0),
        "unit": "seconds",
    }

    local = estimate_generation(
        uuid4(),
        [TaskComplexity.SMALL],
        [_sample(TaskComplexity.SMALL, value) for value in range(100, 600, 100)],
        _billing(),
    )
    local_small = local["per_complexity"][0]
    assert local_small["sample_size"] == 5
    assert local_small["source"] == "local"
    assert local_small["time"] == {
        "p50": 300.0,
        "p80": 400.0,
        "unit": "seconds",
    }


def test_missing_prior_and_unsized_tasks_stay_unknown() -> None:
    estimate = estimate_generation(
        uuid4(),
        [TaskComplexity.SMALL, None],
        [],
        _billing(),
        priors={},
    )

    assert estimate["task_mix"]["unsized"] == 1
    assert estimate["estimable_tasks"] == 0
    assert estimate["time"]["p50"] is None
    assert estimate["time"]["p80"] is None
    assert estimate["time"]["unknown_tasks"] == 2
    assert estimate["per_complexity"][0]["source"] == "unknown"


@pytest.mark.parametrize(
    ("billing", "api_unknown", "has_api", "subscription"),
    (
        (_billing(BillingMode.API, BillingMode.API), False, True, False),
        (_billing(), False, False, True),
        (_billing(BillingMode.API, BillingMode.SUBSCRIPTION), False, True, True),
        (
            _billing(BillingMode.API, BillingMode.API, model="unpriced-model"),
            True,
            False,
            False,
        ),
    ),
)
def test_api_subscription_mixed_and_unpriced_billing_are_honest(
    billing: tuple[PhaseBilling, ...],
    api_unknown: bool,
    has_api: bool,
    subscription: bool,
) -> None:
    estimate = estimate_generation(
        uuid4(), [TaskComplexity.SMALL], [], billing
    )

    assert estimate["billing"]["api"]["unknown"] is api_unknown
    assert (estimate["billing"]["api"]["estimate"] is not None) is has_api
    assert estimate["billing"]["subscription"] == {
        "included": subscription,
        "phases": ["review"] if len(set(item.mode for item in billing)) > 1 else (
            ["coding", "review"] if subscription else []
        ),
        "usd": None,
    }


def test_missing_usage_or_unpriced_model_is_not_converted_to_zero() -> None:
    assert estimate_api_cost_usd("openai", "gpt-5", AgentUsage()) is None
    assert estimate_api_cost_usd("openai", "not-released", _usage()) is None
    assert estimate_api_cost_usd("openai", "gpt-5", _usage()) == pytest.approx(
        0.012625
    )


def test_dummy_priors_are_complete_for_each_size() -> None:
    assert set(DUMMY_PRIORS) == set(TaskComplexity)
