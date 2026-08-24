"""Generate tracked PRDs from canonical ranked recommendation themes."""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from betterborg_cli.agent_runtime.structured import validate_structured_result
from betterborg_cli.repo_analysis.scoring import (
    RECOMMENDATION_SCHEMA,
    RankedRecommendationTheme,
    Recommendation,
    rank_recommendation_themes,
    validate_recommendation_theme,
)
from betterborg_cli.repo_analysis.text_rendering import (
    markdown_code_span,
    markdown_text,
)
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.store import RepositoryAnalysis, SqliteStore

_THEME_KEY_PARTS = re.compile(r"[^a-z0-9]+")
_WINDOWS_RESERVED_BASENAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}
_THEME_FIELDS = (
    "id",
    "title",
    "recommendation_ids",
    "effort",
    "effort_rationale",
)


@dataclass(frozen=True, slots=True)
class ImprovementPrd:
    """One generated, tracked improvement document."""

    theme_key: str
    suggested_borg_name: str
    path: Path
    body_md: str


def resolve_theme_key(theme_id: str) -> str:
    """Resolve an analyzer theme ID to one portable filename key."""
    if not isinstance(theme_id, str) or not theme_id.strip():
        raise ValueError("theme ID must not be empty")
    ascii_id = unicodedata.normalize("NFKD", theme_id).encode(
        "ascii", "ignore"
    ).decode("ascii")
    key = _THEME_KEY_PARTS.sub("-", ascii_id.casefold()).strip("-")
    if not key:
        raise ValueError(f"theme ID {theme_id!r} does not resolve to a filename key")
    if key in _WINDOWS_RESERVED_BASENAMES:
        key = f"{key}-theme"
    return key


def generate_improvement_prds(
    analysis: RepositoryAnalysis,
    paths: RepoPaths,
    suggested_borg_names: Mapping[str, str],
    *,
    store: SqliteStore | None = None,
) -> tuple[ImprovementPrd, ...]:
    """Write one deterministic improvement PRD for every persisted theme.

    Suggested names are keyed by the portable theme keys returned by
    :func:`resolve_theme_key`. The optional store check prevents a suggestion
    from naming a Borg that already belongs to this repository. This function
    writes tracked Markdown only; it never creates a Borg or PRD session.
    """
    themes = _ranked_themes(analysis.analysis_json)
    keyed_themes: list[tuple[str, RankedRecommendationTheme]] = []
    seen_keys: dict[str, str] = {}
    for ranked in themes:
        key = resolve_theme_key(ranked.theme.id)
        if prior_id := seen_keys.get(key):
            raise ValueError(
                f"theme IDs {prior_id!r} and {ranked.theme.id!r} resolve to "
                f"the same key: {key}"
            )
        seen_keys[key] = ranked.theme.id
        keyed_themes.append((key, ranked))

    if any(not isinstance(key, str) for key in suggested_borg_names):
        raise ValueError("suggested Borg name keys must be strings")
    expected_keys = set(seen_keys)
    supplied_keys = set(suggested_borg_names)
    if supplied_keys != expected_keys:
        missing = ", ".join(sorted(expected_keys - supplied_keys)) or "none"
        unknown = ", ".join(sorted(supplied_keys - expected_keys)) or "none"
        raise ValueError(
            "suggested Borg names must match theme keys exactly "
            f"(missing: {missing}; unknown: {unknown})"
        )

    resolved_names: dict[str, str] = {}
    for key in sorted(expected_keys):
        name = suggested_borg_names[key]
        _validate_suggested_name(name, key)
        if name in resolved_names.values():
            raise ValueError(f"suggested Borg name must be unique: {name!r}")
        if store is not None and store.get_borg_by_name(
            analysis.repository_id, name
        ) is not None:
            raise ValueError(
                f"suggested Borg name already exists in this repository: {name!r}"
            )
        resolved_names[key] = name

    # Complete all validation and rendering before creating the destination.
    documents = tuple(
        ImprovementPrd(
            theme_key=key,
            suggested_borg_name=resolved_names[key],
            path=paths.improvement_prds_dir / f"{key}.md",
            body_md=_render_prd(ranked, key, resolved_names[key]),
        )
        for key, ranked in keyed_themes
    )
    publication_directory = _prepare_publication_directory(paths)
    for document in documents:
        _publish_prd(
            publication_directory / document.path.name,
            document.body_md,
        )
    return documents


def _prepare_publication_directory(paths: RepoPaths) -> Path:
    directory = paths.improvement_prds_dir
    resolved = directory.resolve()
    if not resolved.is_relative_to(paths.root):
        raise ValueError(f"improvement PRD directory escapes repository: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    resolved = directory.resolve(strict=True)
    if not resolved.is_relative_to(paths.root):
        raise ValueError(f"improvement PRD directory escapes repository: {directory}")
    return resolved


def _publish_prd(path: Path, body_md: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.urandom(16).hex()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(body_md)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _ranked_themes(payload: Mapping[str, Any]) -> list[RankedRecommendationTheme]:
    raw_packages = payload.get("packages")
    raw_recommendations = payload.get("recommendations")
    raw_themes = payload.get("themes")
    if not isinstance(raw_packages, list) or not raw_packages:
        raise ValueError("analysis does not contain packages for improvement PRDs")
    if not isinstance(raw_recommendations, list):
        raise ValueError("analysis does not contain recommendations")
    if not isinstance(raw_themes, list):
        raise ValueError("analysis does not contain recommendation themes")

    package_rubrics: dict[str, Mapping[str, Mapping[str, object]]] = {}
    for raw in raw_packages:
        if not isinstance(raw, Mapping):
            raise ValueError("analysis contains an invalid package")
        path = raw.get("path")
        rubric = raw.get("rubric")
        if not isinstance(path, str) or not path or not isinstance(rubric, Mapping):
            raise ValueError("analysis contains an invalid package")
        if path in package_rubrics:
            raise ValueError(f"analysis contains duplicate package path: {path}")
        package_rubrics[path] = rubric  # type: ignore[assignment]

    recommendations = [_recommendation(raw) for raw in raw_recommendations]
    themes = []
    for raw in raw_themes:
        if not isinstance(raw, Mapping):
            raise ValueError("analysis contains an invalid recommendation theme")
        canonical = {field: raw.get(field) for field in _THEME_FIELDS}
        themes.append(validate_recommendation_theme(canonical))
    return rank_recommendation_themes(package_rubrics, recommendations, themes)


def _recommendation(raw: object) -> Recommendation:
    if not isinstance(raw, Mapping):
        raise ValueError("analysis contains an invalid recommendation")
    validate_structured_result(raw, RECOMMENDATION_SCHEMA)
    return Recommendation(
        id=raw["id"],
        title=raw["title"],
        package_path=raw["package_path"],
        dimension=raw["dimension"],
        manifest_evidence=tuple(raw["manifest_evidence"]),
        estimated_delta=float(raw["estimated_delta"]),
        effort=raw["effort"],
        overlap_group=raw["overlap_group"],
    )


def _validate_suggested_name(name: object, theme_key: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"suggested Borg name for {theme_key!r} must not be empty")
    if name != name.strip() or "\n" in name or "\r" in name:
        raise ValueError(
            f"suggested Borg name for {theme_key!r} must be a single trimmed line"
        )


def _render_prd(
    ranked: RankedRecommendationTheme, theme_key: str, suggested_name: str
) -> str:
    theme = ranked.theme
    lines = [
        f"# {markdown_text(theme.title)}",
        "",
        f"Theme key: `{theme_key}`",
        "",
        "## Suggested Borg",
        "",
        markdown_text(suggested_name),
        "",
        "## Scope",
        "",
    ]
    for scored in ranked.recommendations:
        recommendation = scored.recommendation
        lines.append(
            f"- **{markdown_text(recommendation.title)}** in "
            f"{markdown_code_span(recommendation.package_path)}"
        )

    lines.extend(
        [
            "",
            "## Estimated effort",
            "",
            f"**{theme.effort}** — {markdown_text(theme.effort_rationale)}",
            "",
            "## Dimension changes",
            "",
            "| Scope | Dimension | Proposed change | Effective change |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for change in ranked.dimension_changes:
        lines.append(
            f"| {markdown_code_span(change.package_path, table_cell=True)} | "
            f"{markdown_code_span(change.dimension, table_cell=True)} | "
            f"+{change.proposed_delta} | +{change.effective_delta} |"
        )

    lines.extend(
        [
            "",
            "## Estimated overall effect",
            "",
            f"**+{ranked.normalized_impact}** repository score.",
            "",
            "## Evidence",
            "",
        ]
    )
    for scored in ranked.recommendations:
        recommendation = scored.recommendation
        evidence = ", ".join(
            markdown_code_span(path) for path in recommendation.manifest_evidence
        )
        lines.append(f"- **{markdown_text(recommendation.title)}:** {evidence}")
    return "\n".join(lines) + "\n"
