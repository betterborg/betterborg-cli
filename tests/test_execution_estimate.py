"""Deterministic execution-estimate and provider-pricing contracts."""

from __future__ import annotations

from uuid import uuid4

import pytest

from betterborg_cli.agent_runtime import AgentUsage, BillingMode
from betterborg_cli.agent_runtime.pricing import (
    ModelPrice,
    estimate_api_cost_usd,
    lookup_model_price,
)
from betterborg_cli.execution_estimate import (
    DUMMY_PRIORS,
    PhaseBilling,
    estimate_generation,
    phase_billing_from_config,
)
from betterborg_cli.repository_config import (
    AgentChoice,
    AgentChoices,
    RepositoryConfig,
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
        merge_usage=_usage(int(duration)),
    )


def _billing(
    coding: BillingMode | None = BillingMode.SUBSCRIPTION,
    review: BillingMode | None = BillingMode.SUBSCRIPTION,
    merge: BillingMode | None = BillingMode.SUBSCRIPTION,
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

    return phase("coding", coding), phase("review", review), phase("merge", merge)


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


def test_unsized_task_makes_an_api_total_unknown_instead_of_partial() -> None:
    estimate = estimate_generation(
        uuid4(),
        [TaskComplexity.SMALL, None],
        [],
        _billing(BillingMode.API, BillingMode.API, BillingMode.API),
    )

    assert estimate["billing"]["api"]["unknown"] is True
    assert estimate["billing"]["api"]["estimate"] is None


@pytest.mark.parametrize(
    ("billing", "api_unknown", "has_api", "subscription_phases"),
    (
        (
            _billing(BillingMode.API, BillingMode.API, BillingMode.API),
            False,
            True,
            [],
        ),
        (_billing(), False, False, ["coding", "merge", "review"]),
        (
            _billing(BillingMode.API, BillingMode.SUBSCRIPTION),
            False,
            True,
            ["merge", "review"],
        ),
        (
            _billing(
                BillingMode.API,
                BillingMode.API,
                BillingMode.API,
                model="unpriced-model",
            ),
            True,
            False,
            [],
        ),
    ),
)
def test_api_subscription_mixed_and_unpriced_billing_are_honest(
    billing: tuple[PhaseBilling, ...],
    api_unknown: bool,
    has_api: bool,
    subscription_phases: list[str],
) -> None:
    estimate = estimate_generation(
        uuid4(), [TaskComplexity.SMALL], [], billing
    )

    assert estimate["billing"]["api"]["unknown"] is api_unknown
    assert (estimate["billing"]["api"]["estimate"] is not None) is has_api
    assert estimate["billing"]["subscription"] == {
        "included": bool(subscription_phases),
        "phases": subscription_phases,
        "usd": None,
    }


def test_merge_agent_usage_and_billing_are_part_of_api_commitment() -> None:
    with_merge = estimate_generation(
        uuid4(),
        [TaskComplexity.SMALL],
        [],
        _billing(BillingMode.API, BillingMode.API, BillingMode.API),
    )
    without_merge_api = estimate_generation(
        uuid4(),
        [TaskComplexity.SMALL],
        [],
        _billing(BillingMode.API, BillingMode.API, BillingMode.SUBSCRIPTION),
    )

    assert [item["phase"] for item in with_merge["billing"]["api"]["models"]] == [
        "coding",
        "review",
        "merge",
    ]
    assert with_merge["billing"]["api"]["estimate"]["p50"] > (
        without_merge_api["billing"]["api"]["estimate"]["p50"]
    )
    assert with_merge["per_complexity"][0]["token_source"]["merge"] == (
        "dummy_prior"
    )


def test_phase_billing_resolves_configured_merge_agent() -> None:
    config = RepositoryConfig(
        version=1,
        repository_id=uuid4(),
        default_branch="main",
        agents=AgentChoices(
            coding=AgentChoice(adapter="codex"),
            review=AgentChoice(adapter="claude"),
            merge=AgentChoice(adapter="openai", model="gpt-5-mini"),
        ),
    )

    assert phase_billing_from_config(config) == (
        PhaseBilling("coding", BillingMode.SUBSCRIPTION, None, "gpt-5.6-sol"),
        PhaseBilling(
            "review", BillingMode.SUBSCRIPTION, None, "claude-opus-5"
        ),
        PhaseBilling("merge", BillingMode.API, "openai", "gpt-5-mini"),
    )


def test_missing_usage_or_unpriced_model_is_not_converted_to_zero() -> None:
    assert estimate_api_cost_usd("openai", "gpt-5", AgentUsage()) is None
    assert estimate_api_cost_usd("openai", "not-released", _usage()) is None
    assert estimate_api_cost_usd("openai", "gpt-5", _usage()) == pytest.approx(
        0.012625
    )


def test_openai_cached_input_is_not_also_charged_as_fresh_input() -> None:
    assert estimate_api_cost_usd(
        "openai",
        "gpt-5",
        AgentUsage(
            tokens_input=0,
            tokens_output=0,
            tokens_cache_read=1_000,
            tokens_cache_write=0,
        ),
    ) == pytest.approx(0.000125)


@pytest.mark.parametrize(
    ("model", "price"),
    (
        ("gpt-5.6-sol", ModelPrice(4.00, 0.40, 5.00, 20.00)),
        ("gpt-5.6-terra", ModelPrice(2.00, 0.20, 2.50, 12.00)),
        ("gpt-5.6-luna", ModelPrice(0.20, 0.02, 0.25, 1.20)),
    ),
)
def test_gpt_5_6_uses_release_verified_rates(
    model: str, price: ModelPrice
) -> None:
    assert lookup_model_price("openai", model) == price


def test_gpt_5_6_sol_fresh_input_and_output_cost() -> None:
    assert estimate_api_cost_usd(
        "openai",
        "gpt-5.6-sol",
        AgentUsage(
            tokens_input=1_000_000,
            tokens_output=1_000_000,
            tokens_cache_read=0,
            tokens_cache_write=0,
        ),
    ) == pytest.approx(24.00)


@pytest.mark.parametrize(
    ("snapshot", "alias"),
    (
        ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5"),
        ("claude-haiku-4-5-20251001", "claude-haiku-4-5"),
    ),
)
def test_anthropic_compact_date_snapshot_uses_alias_price(
    snapshot: str, alias: str
) -> None:
    assert lookup_model_price("anthropic", snapshot) == lookup_model_price(
        "anthropic", alias
    )


def test_dummy_priors_are_complete_for_each_size() -> None:
    assert set(DUMMY_PRIORS) == set(TaskComplexity)
