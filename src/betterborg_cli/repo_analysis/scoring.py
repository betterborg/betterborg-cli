"""Canonical repository scores and recommendation-theme ranking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from betterborg_cli.agent_runtime.structured import validate_structured_result
from betterborg_cli.repo_analysis.discovery import DiscoveryManifest

DIMENSIONS = (
    "agent_guidance",
    "documentation",
    "testing",
    "ci",
    "coding_standards",
    "build_ergonomics",
    "type_discipline",
    "deployment",
)
WEIGHTS: dict[str, float] = dict.fromkeys(DIMENSIONS, 1.0)
EFFORT_COST = {"S": 1.0, "M": 2.0, "L": 3.0}

RECOMMENDATION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "BetterBorg repository recommendation",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id",
        "title",
        "package_path",
        "dimension",
        "manifest_evidence",
        "estimated_delta",
        "effort",
        "overlap_group",
    ],
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "title": {"type": "string", "minLength": 1},
        "package_path": {"type": "string", "minLength": 1},
        "dimension": {"type": "string", "enum": list(DIMENSIONS)},
        "manifest_evidence": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        # No schema maximum: promises beyond the available 0--5 headroom are
        # valid inputs which the scoring policy must clamp and flag.
        "estimated_delta": {"type": "number", "minimum": 0},
        "effort": {"type": "string", "enum": list(EFFORT_COST)},
        "overlap_group": {"type": ["string", "null"], "minLength": 1},
    },
}

RECOMMENDATION_THEME_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "BetterBorg recommendation theme",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id",
        "title",
        "recommendation_ids",
        "effort",
        "effort_rationale",
    ],
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "title": {"type": "string", "minLength": 1},
        "recommendation_ids": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "effort": {"type": "string", "enum": list(EFFORT_COST)},
        "effort_rationale": {"type": "string", "minLength": 1},
    },
}


@dataclass(frozen=True, slots=True)
class Recommendation:
    """One manifest-supported, dimension-local improvement."""

    id: str
    title: str
    package_path: str
    dimension: str
    manifest_evidence: tuple[str, ...]
    estimated_delta: float
    effort: str
    overlap_group: str | None


@dataclass(frozen=True, slots=True)
class RecommendationTheme:
    """A ranked group whose explicit effort is authoritative."""

    id: str
    title: str
    recommendation_ids: tuple[str, ...]
    effort: str
    effort_rationale: str


@dataclass(frozen=True, slots=True)
class PackageScore:
    """Canonical score for every dimension and one package overall."""

    path: str
    dimensions: dict[str, float]
    overall_score: float


@dataclass(frozen=True, slots=True)
class RepositoryScore:
    """Canonical package scores and their equal-weight repository mean."""

    packages: tuple[PackageScore, ...]
    overall_score: float


@dataclass(frozen=True, slots=True)
class ScoredRecommendation:
    """A recommendation delta after applying dimension headroom."""

    recommendation: Recommendation
    effective_delta: float
    delta_clamped: bool


@dataclass(frozen=True, slots=True)
class RankedRecommendationTheme:
    """A theme's normalized impact and effort-aware ordering value."""

    theme: RecommendationTheme
    recommendations: tuple[ScoredRecommendation, ...]
    normalized_impact: float
    ranking_score: float


def compute_overall_score(rubric: Mapping[str, Mapping[str, object]]) -> float:
    """Return the policy-weighted mean across all eight canonical dimensions."""
    divisor = sum(WEIGHTS.values())
    return (
        sum(
            _dimension_score(rubric, dimension) * weight
            for dimension, weight in WEIGHTS.items()
        )
        / divisor
        if divisor
        else 0.0
    )


def compute_repo_overall_score(packages: Sequence[Mapping[str, object]]) -> float:
    """Return the mean of present numeric package scores, or zero."""
    scores = [
        float(package["overall_score"])
        for package in packages
        if isinstance(package, Mapping)
        and isinstance(package.get("overall_score"), int | float)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def score_repository(
    package_rubrics: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> RepositoryScore:
    """Compute explicit eight-dimension package scores and the repository mean."""
    if not package_rubrics:
        raise ValueError("at least one package rubric is required")
    packages = tuple(
        PackageScore(
            path=path,
            dimensions={
                dimension: _dimension_score(rubric, dimension)
                for dimension in DIMENSIONS
            },
            overall_score=compute_overall_score(rubric),
        )
        for path, rubric in package_rubrics.items()
    )
    return RepositoryScore(
        packages=packages,
        overall_score=compute_repo_overall_score(
            [{"overall_score": package.overall_score} for package in packages]
        ),
    )


def validate_recommendation(
    payload: Mapping[str, Any], manifest: DiscoveryManifest
) -> Recommendation:
    """Validate a recommendation and require citations from the manifest."""
    validate_structured_result(payload, RECOMMENDATION_SCHEMA)
    known_evidence = {file.path for file in manifest.files}
    unknown_evidence = set(payload["manifest_evidence"]) - known_evidence
    if unknown_evidence:
        names = ", ".join(sorted(unknown_evidence))
        raise ValueError(f"recommendation cites evidence absent from manifest: {names}")
    return Recommendation(
        id=payload["id"],
        title=payload["title"],
        package_path=payload["package_path"],
        dimension=payload["dimension"],
        manifest_evidence=tuple(payload["manifest_evidence"]),
        estimated_delta=float(payload["estimated_delta"]),
        effort=payload["effort"],
        overlap_group=payload["overlap_group"],
    )


def validate_recommendation_theme(payload: Mapping[str, Any]) -> RecommendationTheme:
    """Validate a theme, including its explicit labeled effort and rationale."""
    validate_structured_result(payload, RECOMMENDATION_THEME_SCHEMA)
    return RecommendationTheme(
        id=payload["id"],
        title=payload["title"],
        recommendation_ids=tuple(payload["recommendation_ids"]),
        effort=payload["effort"],
        effort_rationale=payload["effort_rationale"],
    )


def rank_recommendation_themes(
    package_rubrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    recommendations: Sequence[Recommendation],
    themes: Sequence[RecommendationTheme],
) -> list[RankedRecommendationTheme]:
    """Clamp, normalize, and rank themes by impact per explicit theme effort.

    Recommendations sharing a non-null overlap group within one package and
    dimension contribute only their largest delta. Independent groups may add,
    but their combined effect cannot exceed that dimension's remaining
    headroom. Normalized impact is dimension points divided by eight and by the
    total package count.
    """
    if not package_rubrics:
        raise ValueError("at least one package rubric is required")
    recommendations_by_id = _index_unique(recommendations, "recommendation")
    _index_unique(themes, "theme")
    unknown_packages = {
        recommendation.package_path
        for recommendation in recommendations
        if recommendation.package_path not in package_rubrics
    }
    if unknown_packages:
        names = ", ".join(sorted(unknown_packages))
        raise ValueError(f"recommendations reference unknown packages: {names}")

    ranked: list[RankedRecommendationTheme] = []
    for theme in themes:
        unknown_ids = set(theme.recommendation_ids) - recommendations_by_id.keys()
        if unknown_ids:
            names = ", ".join(sorted(unknown_ids))
            raise ValueError(
                f"theme {theme.id!r} references unknown recommendations: {names}"
            )

        scored = tuple(
            _score_recommendation(
                recommendations_by_id[recommendation_id], package_rubrics
            )
            for recommendation_id in theme.recommendation_ids
        )
        dimension_points = _theme_dimension_points(scored, package_rubrics)
        normalized_impact = dimension_points / (
            sum(WEIGHTS.values()) * len(package_rubrics)
        )
        ranked.append(
            RankedRecommendationTheme(
                theme=theme,
                recommendations=scored,
                normalized_impact=normalized_impact,
                ranking_score=normalized_impact / EFFORT_COST[theme.effort],
            )
        )

    return sorted(
        ranked,
        key=lambda result: (
            -result.ranking_score,
            -result.normalized_impact,
            result.theme.id,
        ),
    )


def _dimension_score(
    rubric: Mapping[str, Mapping[str, object]], dimension: str
) -> float:
    cell = rubric.get(dimension) or {}
    if not isinstance(cell, Mapping):
        return 0.0
    try:
        score = float(cell.get("score", 0))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(5.0, score))


def _index_unique(items: Sequence[Any], kind: str) -> dict[str, Any]:
    indexed: dict[str, Any] = {}
    for item in items:
        if item.id in indexed:
            raise ValueError(f"duplicate {kind} ID: {item.id}")
        indexed[item.id] = item
    return indexed


def _score_recommendation(
    recommendation: Recommendation,
    package_rubrics: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> ScoredRecommendation:
    current = _dimension_score(
        package_rubrics[recommendation.package_path], recommendation.dimension
    )
    effective = min(recommendation.estimated_delta, 5.0 - current)
    return ScoredRecommendation(
        recommendation=recommendation,
        effective_delta=effective,
        delta_clamped=effective < recommendation.estimated_delta,
    )


def _theme_dimension_points(
    scored: Sequence[ScoredRecommendation],
    package_rubrics: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> float:
    grouped: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
    for result in scored:
        recommendation = result.recommendation
        target = (recommendation.package_path, recommendation.dimension)
        # Tag both namespaces so a user-provided overlap label cannot collide
        # with the generated key for an independent recommendation.
        group = (
            ("overlap", recommendation.overlap_group)
            if recommendation.overlap_group is not None
            else ("recommendation", recommendation.id)
        )
        grouped.setdefault(target, {})[group] = max(
            result.effective_delta,
            grouped.get(target, {}).get(group, 0.0),
        )

    total = 0.0
    for (package_path, dimension), groups in grouped.items():
        current = _dimension_score(package_rubrics[package_path], dimension)
        total += min(sum(groups.values()), 5.0 - current) * WEIGHTS[dimension]
    return total
