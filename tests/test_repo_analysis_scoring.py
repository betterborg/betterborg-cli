"""Repository score and recommendation-theme policy tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from betterborg_cli.agent_runtime.structured import StructuredResultError
from betterborg_cli.repo_analysis.discovery import (
    DiscoveryFile,
    DiscoveryManifest,
)
from betterborg_cli.repo_analysis.scoring import (
    DIMENSIONS,
    WEIGHTS,
    compute_overall_score,
    compute_repo_overall_score,
    rank_recommendation_themes,
    score_repository,
    validate_recommendation,
    validate_recommendation_theme,
)


@pytest.fixture
def manifest() -> DiscoveryManifest:
    return DiscoveryManifest(
        repo_name="acme/widgets",
        files=[
            DiscoveryFile(
                path="pyproject.toml",
                workspace_path="files/pyproject.toml",
                category="manifest",
                size_bytes=100,
                copied_bytes=100,
            ),
            DiscoveryFile(
                path=".github/workflows/ci.yml",
                workspace_path="files/.github/workflows/ci.yml",
                category="ci",
                size_bytes=100,
                copied_bytes=100,
            ),
        ],
    )


@pytest.fixture
def rubric() -> dict[str, dict[str, object]]:
    return {
        dimension: {"score": 3, "evidence": f"evidence for {dimension}"}
        for dimension in DIMENSIONS
    }


@pytest.fixture
def recommendation_payload() -> dict[str, object]:
    return {
        "id": "rec-ci",
        "title": "Run lint in CI",
        "package_path": ".",
        "dimension": "ci",
        "manifest_evidence": [".github/workflows/ci.yml"],
        "estimated_delta": 1.0,
        "effort": "S",
        "overlap_group": None,
    }


@pytest.fixture
def theme_payload() -> dict[str, object]:
    return {
        "id": "theme-ci",
        "title": "Strengthen CI feedback",
        "recommendation_ids": ["rec-ci"],
        "effort": "S",
        "effort_rationale": "One workflow edit.",
    }


def test_scores_all_eight_dimensions_and_repository_packages(
    rubric: dict[str, dict[str, object]],
) -> None:
    second = deepcopy(rubric)
    for cell in second.values():
        cell["score"] = 1

    result = score_repository({".": rubric, "packages/web": second})

    assert tuple(result.packages[0].dimensions) == DIMENSIONS
    assert result.packages[0].overall_score == 3.0
    assert result.packages[1].overall_score == 1.0
    assert result.overall_score == 2.0


def test_canonical_scoring_clamps_dimension_scores_and_penalizes_missing() -> None:
    rubric = {
        "agent_guidance": {"score": 10},
        "documentation": {"score": -2},
    }

    assert compute_overall_score(rubric) == pytest.approx(5.0 / 8.0)
    assert compute_repo_overall_score(
        [{"overall_score": 4.0}, {"overall_score": None}, {"overall_score": 2.0}]
    ) == 3.0


def test_weight_policy_drives_package_score_and_normalized_impact(
    rubric: dict[str, dict[str, object]],
    recommendation_payload: dict[str, object],
    theme_payload: dict[str, object],
    manifest: DiscoveryManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(WEIGHTS, "ci", 3.0)
    for cell in rubric.values():
        cell["score"] = 0
    rubric["ci"]["score"] = 4

    recommendation = validate_recommendation(recommendation_payload, manifest)
    result = rank_recommendation_themes(
        {".": rubric},
        [recommendation],
        [validate_recommendation_theme(theme_payload)],
    )[0]

    assert compute_overall_score(rubric) == pytest.approx(12.0 / 10.0)
    assert result.normalized_impact == pytest.approx(3.0 / 10.0)


@pytest.mark.parametrize("fixture_name", ["recommendation_payload", "theme_payload"])
def test_schemas_require_explicit_labeled_effort(
    fixture_name: str,
    request: pytest.FixtureRequest,
    manifest: DiscoveryManifest,
) -> None:
    payload = deepcopy(request.getfixturevalue(fixture_name))
    del payload["effort"]

    with pytest.raises(StructuredResultError, match="effort"):
        if fixture_name == "recommendation_payload":
            validate_recommendation(payload, manifest)
        else:
            validate_recommendation_theme(payload)


def test_recommendation_requires_evidence_present_in_discovery_manifest(
    recommendation_payload: dict[str, object], manifest: DiscoveryManifest
) -> None:
    recommendation_payload["manifest_evidence"] = ["src/uncopied_secret.py"]

    with pytest.raises(ValueError, match="absent from manifest"):
        validate_recommendation(recommendation_payload, manifest)


def test_over_promised_delta_is_clamped_and_flagged(
    rubric: dict[str, dict[str, object]],
    recommendation_payload: dict[str, object],
    theme_payload: dict[str, object],
    manifest: DiscoveryManifest,
) -> None:
    rubric["ci"]["score"] = 4.5
    recommendation_payload["estimated_delta"] = 3.0
    recommendation = validate_recommendation(recommendation_payload, manifest)
    theme = validate_recommendation_theme(theme_payload)

    result = rank_recommendation_themes({".": rubric}, [recommendation], [theme])[0]

    assert result.recommendations[0].effective_delta == 0.5
    assert result.recommendations[0].delta_clamped is True
    assert result.normalized_impact == pytest.approx(0.5 / 8.0)


def test_overlapping_recommendations_contribute_only_largest_delta(
    rubric: dict[str, dict[str, object]],
    recommendation_payload: dict[str, object],
    theme_payload: dict[str, object],
    manifest: DiscoveryManifest,
) -> None:
    first_payload = deepcopy(recommendation_payload)
    first_payload.update(
        {"id": "rec-ci-lint", "estimated_delta": 1.0, "overlap_group": "ci-gate"}
    )
    second_payload = deepcopy(recommendation_payload)
    second_payload.update(
        {"id": "rec-ci-check", "estimated_delta": 1.5, "overlap_group": "ci-gate"}
    )
    theme_payload["recommendation_ids"] = ["rec-ci-lint", "rec-ci-check"]

    result = rank_recommendation_themes(
        {".": rubric},
        [
            validate_recommendation(first_payload, manifest),
            validate_recommendation(second_payload, manifest),
        ],
        [validate_recommendation_theme(theme_payload)],
    )[0]

    assert result.normalized_impact == pytest.approx(1.5 / 8.0)


def test_overlap_labels_cannot_collide_with_independent_recommendation_keys(
    rubric: dict[str, dict[str, object]],
    recommendation_payload: dict[str, object],
    theme_payload: dict[str, object],
    manifest: DiscoveryManifest,
) -> None:
    rubric["ci"]["score"] = 0
    independent_payload = deepcopy(recommendation_payload)
    independent_payload.update({"id": "alpha", "estimated_delta": 1.0})
    labeled_payload = deepcopy(recommendation_payload)
    labeled_payload.update(
        {
            "id": "beta",
            "estimated_delta": 2.0,
            "overlap_group": "recommendation:alpha",
        }
    )
    theme_payload["recommendation_ids"] = ["alpha", "beta"]

    result = rank_recommendation_themes(
        {".": rubric},
        [
            validate_recommendation(independent_payload, manifest),
            validate_recommendation(labeled_payload, manifest),
        ],
        [validate_recommendation_theme(theme_payload)],
    )[0]

    assert result.normalized_impact == pytest.approx(3.0 / 8.0)


def test_monorepo_impact_divides_by_dimensions_and_package_count(
    rubric: dict[str, dict[str, object]],
    recommendation_payload: dict[str, object],
    theme_payload: dict[str, object],
    manifest: DiscoveryManifest,
) -> None:
    recommendation = validate_recommendation(recommendation_payload, manifest)

    result = rank_recommendation_themes(
        {".": rubric, "packages/web": deepcopy(rubric)},
        [recommendation],
        [validate_recommendation_theme(theme_payload)],
    )[0]

    assert result.normalized_impact == pytest.approx(1.0 / (8.0 * 2.0))


def test_explicit_theme_effort_controls_order_despite_member_effort(
    rubric: dict[str, dict[str, object]],
    recommendation_payload: dict[str, object],
    manifest: DiscoveryManifest,
) -> None:
    large_member = deepcopy(recommendation_payload)
    large_member.update({"id": "rec-large", "effort": "L"})
    small_member = deepcopy(recommendation_payload)
    small_member.update({"id": "rec-small", "effort": "S"})
    small_theme = {
        "id": "theme-small",
        "title": "Small theme with large member label",
        "recommendation_ids": ["rec-large"],
        "effort": "S",
        "effort_rationale": "The changes share one edit.",
    }
    large_theme = {
        "id": "theme-large",
        "title": "Large theme with small member label",
        "recommendation_ids": ["rec-small"],
        "effort": "L",
        "effort_rationale": "Rollout spans several teams.",
    }

    ranked = rank_recommendation_themes(
        {".": rubric},
        [
            validate_recommendation(large_member, manifest),
            validate_recommendation(small_member, manifest),
        ],
        [
            validate_recommendation_theme(large_theme),
            validate_recommendation_theme(small_theme),
        ],
    )

    assert [result.theme.id for result in ranked] == ["theme-small", "theme-large"]
    assert ranked[0].ranking_score == pytest.approx(3 * ranked[1].ranking_score)
