"""Harness Performance report contract tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID

import pytest

from betterborg_cli.repo_analysis.reporting import (
    build_machine_report,
    render_json_report,
    render_markdown_report,
    render_terminal_report,
)
from betterborg_cli.repo_analysis.scoring import DIMENSIONS
from betterborg_cli.store import RepositoryAnalysis, RepositoryPackage

_REPOSITORY_ID = UUID("10000000-0000-0000-0000-000000000000")
_ANALYSIS_ID = UUID("20000000-0000-0000-0000-000000000000")
_PRIOR_ID = UUID("30000000-0000-0000-0000-000000000000")


def _rubric(score: float) -> dict[str, dict[str, object]]:
    return {
        dimension: {"score": score, "evidence": f"evidence for {dimension}"}
        for dimension in DIMENSIONS
    }


@pytest.fixture
def analysis() -> RepositoryAnalysis:
    return RepositoryAnalysis(
        id=_ANALYSIS_ID,
        repository_id=_REPOSITORY_ID,
        head_sha="abc123",
        summary="A Python monorepo with two independently scored packages.",
        primary_language="python",
        is_monorepo=True,
        overall_score=3.0,
        prior_analysis_id=_PRIOR_ID,
        score_delta=0.5,
        created_at=datetime(2026, 8, 24, tzinfo=UTC),
        analysis_json={
            "command_catalog": {"commands": [{"argv": ["make", "test"]}]},
            "required_secrets": [{"name": "PACKAGE_TOKEN"}],
            "service_dependencies": [],
            "themes": [
                {
                    "id": "theme-ci",
                    "title": "Strengthen CI feedback",
                    "recommendation_ids": ["rec-ci"],
                    "effort": "S",
                    "effort_rationale": "One workflow edit.",
                    "normalized_impact": 0.125,
                    "ranking_score": 0.125,
                    "recommendations": [
                        {
                            "id": "rec-ci",
                            "effective_delta": 1.0,
                            "delta_clamped": False,
                        }
                    ],
                }
            ],
        },
    )


@pytest.fixture
def packages() -> list[RepositoryPackage]:
    first_rubric = _rubric(4)
    second_rubric = _rubric(2)
    first_rubric["ci"]["score"] = 5
    second_rubric["ci"]["score"] = 1
    return [
        RepositoryPackage(
            repository_id=_REPOSITORY_ID,
            analysis_id=_ANALYSIS_ID,
            package_path="packages/worker",
            package_name="worker",
            primary_language="python",
            rubric=second_rubric,
            overall_score=2.0,
        ),
        RepositoryPackage(
            repository_id=_REPOSITORY_ID,
            analysis_id=_ANALYSIS_ID,
            package_path="packages/api",
            package_name="api",
            primary_language="python",
            rubric=first_rubric,
            overall_score=4.0,
        ),
    ]


def test_machine_report_preserves_arithmetic_history_and_ranked_theme_contract(
    analysis: RepositoryAnalysis, packages: list[RepositoryPackage]
) -> None:
    report = build_machine_report(analysis, packages)

    assert report["score"] == 3.0
    assert report["previous_score"] == 2.5
    assert report["delta"] == 0.5
    assert [package["path"] for package in report["packages"]] == [
        "packages/api",
        "packages/worker",
    ]
    dimension_scores = {
        dimension["id"]: dimension["score"] for dimension in report["dimensions"]
    }
    assert dimension_scores == {
        dimension: 3.0 for dimension in DIMENSIONS
    }
    assert report["themes"][0] == {
        "rank": 1,
        "id": "theme-ci",
        "title": "Strengthen CI feedback",
        "effort": "S",
        "effort_label": "S (estimated)",
        "effort_rationale": "One workflow edit.",
        "estimated_impact": 0.125,
        "ranking_score": 0.125,
        "recommendations": [
            {"id": "rec-ci", "effective_delta": 1.0, "delta_clamped": False}
        ],
    }
    assert report["estimated"] is True
    assert report["non_deterministic"] is True


def test_harness_impact_distinguishes_unknown_from_detected_and_not_detected(
    analysis: RepositoryAnalysis, packages: list[RepositoryPackage]
) -> None:
    impact = build_machine_report(analysis, packages)["harness_impact"]

    assert impact["commands"]["status"] == "detected"
    assert impact["environment"] == {
        "status": "unknown",
        "label": "Unknown",
        "summary": "No reliable environment inputs were persisted.",
    }
    assert impact["secrets"]["status"] == "detected"
    assert impact["services"]["status"] == "not_detected"


def test_terminal_markdown_and_json_render_the_same_labeled_report(
    analysis: RepositoryAnalysis, packages: list[RepositoryPackage]
) -> None:
    report = build_machine_report(analysis, packages)

    terminal = render_terminal_report(report)
    markdown = render_markdown_report(report)
    machine = json.loads(render_json_report(report))

    for rendered in (terminal, markdown):
        assert "Harness Performance" in rendered
        assert "3.00/5" in rendered
        assert "Previous: 2.50" in rendered
        assert "Delta: +0.50" in rendered
        assert "██████░░░░" in rendered
        assert "packages/api" in rendered
        assert "Strengthen CI feedback" in rendered
        assert "S (estimated)" in rendered
        assert "Harness Impact" in rendered
        assert "Environment" in rendered
        assert "Unknown" in rendered
        assert "non-deterministic" in rendered
        lowered = rendered.lower()
        assert "readiness" not in lowered
        assert "reproducib" not in lowered

    assert machine == report


def test_report_rejects_packages_from_another_analysis(
    analysis: RepositoryAnalysis, packages: list[RepositoryPackage]
) -> None:
    wrong_package = RepositoryPackage(
        repository_id=_REPOSITORY_ID,
        analysis_id=UUID("40000000-0000-0000-0000-000000000000"),
        package_path=".",
        package_name="wrong",
        primary_language="python",
        rubric=_rubric(3),
        overall_score=3,
    )

    with pytest.raises(ValueError, match="belong to the supplied analysis"):
        build_machine_report(analysis, [*packages, wrong_package])
