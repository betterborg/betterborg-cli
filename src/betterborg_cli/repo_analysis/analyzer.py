"""Validated repository analysis in a bounded discovery workspace."""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from betterborg_cli.agent_runtime.base import (
    AgentAdapter,
    AgentRunSpec,
    AgentStatus,
    CancellationToken,
)
from betterborg_cli.agent_runtime.selection import SelectedAgent
from betterborg_cli.agent_runtime.structured import validate_structured_result
from betterborg_cli.repo_analysis.discovery import (
    DiscoveryLimits,
    DiscoveryManifest,
    build_discovery_workspace,
)
from betterborg_cli.repo_analysis.scoring import (
    DIMENSIONS,
    RECOMMENDATION_SCHEMA,
    RECOMMENDATION_THEME_SCHEMA,
    rank_recommendation_themes,
    score_repository,
    validate_recommendation,
    validate_recommendation_theme,
)
from betterborg_cli.store import (
    Repository,
    RepositoryAnalysis,
    RepositoryPackage,
    SqliteStore,
)

_DIMENSION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["score", "evidence"],
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 5},
        "evidence": {"type": "string", "minLength": 1},
    },
}
_RUBRIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": list(DIMENSIONS),
    "properties": {
        dimension: {"$ref": "#/$defs/dimension"} for dimension in DIMENSIONS
    },
}
ANALYZER_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "BetterBorg bounded repository analysis",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "summary",
        "primary_language",
        "is_monorepo",
        "packages",
        "recommendations",
        "themes",
    ],
    "properties": {
        "summary": {"type": "string", "minLength": 8},
        "primary_language": {"type": "string", "minLength": 1},
        "is_monorepo": {"type": "boolean"},
        "packages": {
            "type": "array",
            "minItems": 1,
            "items": {"$ref": "#/$defs/package"},
        },
        "recommendations": {
            "type": "array",
            "items": {"$ref": "#/$defs/recommendation"},
        },
        "themes": {
            "type": "array",
            "items": {"$ref": "#/$defs/theme"},
        },
    },
    "$defs": {
        "dimension": _DIMENSION_SCHEMA,
        "rubric": _RUBRIC_SCHEMA,
        "package": {
            "type": "object",
            "additionalProperties": False,
            "required": ["path", "name", "primary_language", "rubric"],
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "name": {"type": "string", "minLength": 1},
                "primary_language": {"type": "string", "minLength": 1},
                "rubric": {"$ref": "#/$defs/rubric"},
            },
        },
        "recommendation": {
            key: value
            for key, value in RECOMMENDATION_SCHEMA.items()
            if key != "$schema"
        },
        "theme": {
            key: value
            for key, value in RECOMMENDATION_THEME_SCHEMA.items()
            if key != "$schema"
        },
    },
}

_SYSTEM_PROMPT = """You analyze only the bounded evidence workspace at cwd.
Open analysis_input.json first and inspect only files listed in its files array.
Score every package on the eight required dimensions. Every recommendation must
cite manifest paths, target one package and dimension, and state S/M/L effort.
Group recommendations into themes with an explicit S/M/L theme effort and
rationale. Return only the JSON object required by the supplied schema.
"""
_USER_PROMPT = (
    "Analyze the bounded discovery manifest and copied evidence. Treat omitted "
    "or truncated evidence as uncertainty; do not inspect the raw repository."
)
_DEFAULT_ANALYSIS_MODELS = {
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-5",
}


class AnalyzerError(RuntimeError):
    """Raised when an analyzer run cannot produce a persistent result."""


@dataclass(frozen=True, slots=True)
class AnalyzerConfig:
    """Resource and model settings for one bounded analysis."""

    model: str | None = None
    effort: str | None = None
    limits: DiscoveryLimits = DiscoveryLimits()


def run_analyzer(
    repository: Repository,
    store: SqliteStore,
    agent: AgentAdapter | SelectedAgent,
    *,
    artifact_dir: Path,
    workspace_dir: Path | None = None,
    config: AnalyzerConfig | None = None,
    cancel: CancellationToken | None = None,
) -> RepositoryAnalysis:
    """Run, validate, normalize, and append one successful analysis."""
    stored_repository = store.get_repository(repository.id)
    if stored_repository != repository:
        raise ValueError("repository must already be present in the supplied store")

    artifact_dir = Path(artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    head_sha = _git_head(repository.root)
    resolved_config = config or AnalyzerConfig()
    if resolved_config.effort is not None and agent.name == "anthropic":
        raise AnalyzerError("Anthropic does not support an effort override")

    if workspace_dir is None:
        with tempfile.TemporaryDirectory(prefix="betterborg-analysis-") as temporary:
            return _run_in_workspace(
                repository,
                store,
                agent,
                artifact_dir=artifact_dir,
                workspace_dir=Path(temporary),
                head_sha=head_sha,
                config=resolved_config,
                cancel=cancel,
            )
    return _run_in_workspace(
        repository,
        store,
        agent,
        artifact_dir=artifact_dir,
        workspace_dir=Path(workspace_dir),
        head_sha=head_sha,
        config=resolved_config,
        cancel=cancel,
    )


def _run_in_workspace(
    repository: Repository,
    store: SqliteStore,
    agent: AgentAdapter | SelectedAgent,
    *,
    artifact_dir: Path,
    workspace_dir: Path,
    head_sha: str,
    config: AnalyzerConfig,
    cancel: CancellationToken | None,
) -> RepositoryAnalysis:
    manifest = build_discovery_workspace(
        repository.root,
        workspace_dir,
        limits=config.limits,
    )
    if agent.capabilities.host_capable:
        raise AnalyzerError(
            f"adapter {agent.name!r} is host-capable and cannot be confined to "
            "the bounded discovery workspace; select the 'anthropic' or "
            "'openai' API adapter"
        )
    if not agent.capabilities.tool_allowlist:
        raise AnalyzerError(
            f"adapter {agent.name!r} cannot enforce the bounded analyzer "
            "tool allowlist"
        )
    run_id = uuid4()
    spec = AgentRunSpec(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=_USER_PROMPT,
        schema=ANALYZER_OUTPUT_SCHEMA,
        cwd=workspace_dir.resolve(),
        model=resolve_analysis_model(agent, config.model),
        effort=config.effort,
        allowed_tools=("list_files", "read_file", "search_text"),
        log_path=artifact_dir / f"{run_id}.log",
        result_path=artifact_dir / f"{run_id}.json",
    )
    result = (
        agent.run_contained(spec, cancel=cancel)
        if isinstance(agent, SelectedAgent)
        else agent.run(spec, cancel=cancel)
    )
    if result.status is not AgentStatus.COMPLETED or result.payload is None:
        raise AnalyzerError(result.error or f"analyzer returned {result.status.value}")

    # Adapters validate their output; repeat validation at the persistence edge.
    validate_structured_result(result.payload, ANALYZER_OUTPUT_SCHEMA)
    return _persist_payload(
        repository,
        store,
        result.payload,
        manifest=manifest,
        head_sha=head_sha,
        analysis_id=run_id,
    )


def _persist_payload(
    repository: Repository,
    store: SqliteStore,
    payload: Mapping[str, Any],
    *,
    manifest: DiscoveryManifest,
    head_sha: str,
    analysis_id: UUID,
) -> RepositoryAnalysis:
    packages_raw = payload["packages"]
    package_paths = [package["path"] for package in packages_raw]
    if len(package_paths) != len(set(package_paths)):
        raise AnalyzerError("analyzer returned duplicate package paths")
    if payload["is_monorepo"] != (len(packages_raw) > 1):
        raise AnalyzerError("is_monorepo must match whether multiple packages exist")

    package_rubrics = {
        package["path"]: package["rubric"] for package in packages_raw
    }
    repository_score = score_repository(package_rubrics)
    recommendations = [
        validate_recommendation(recommendation, manifest)
        for recommendation in payload["recommendations"]
    ]
    themes = [validate_recommendation_theme(theme) for theme in payload["themes"]]
    ranked_themes = rank_recommendation_themes(
        package_rubrics, recommendations, themes
    )

    normalized_payload = dict(payload)
    normalized_payload["overall_score"] = repository_score.overall_score
    package_scores = {package.path: package for package in repository_score.packages}
    normalized_payload["packages"] = [
        {
            **dict(package),
            "overall_score": package_scores[package["path"]].overall_score,
        }
        for package in packages_raw
    ]
    normalized_payload["themes"] = [
        {
            "id": result.theme.id,
            "title": result.theme.title,
            "recommendation_ids": list(result.theme.recommendation_ids),
            "effort": result.theme.effort,
            "effort_rationale": result.theme.effort_rationale,
            "normalized_impact": result.normalized_impact,
            "ranking_score": result.ranking_score,
            "recommendations": [
                {
                    "id": scored.recommendation.id,
                    "effective_delta": scored.effective_delta,
                    "delta_clamped": scored.delta_clamped,
                }
                for scored in result.recommendations
            ],
        }
        for result in ranked_themes
    ]

    with store.transaction():
        prior = store.get_prior_ready_analysis(repository.id)
        analysis = RepositoryAnalysis(
            id=analysis_id,
            repository_id=repository.id,
            head_sha=head_sha,
            summary=payload["summary"],
            primary_language=payload["primary_language"],
            is_monorepo=payload["is_monorepo"],
            overall_score=repository_score.overall_score,
            analysis_json=normalized_payload,
            prior_analysis_id=prior.id if prior is not None else None,
            score_delta=(
                repository_score.overall_score - prior.overall_score
                if prior is not None
                else None
            ),
        )
        packages = [
            RepositoryPackage(
                repository_id=repository.id,
                analysis_id=analysis.id,
                package_path=package["path"],
                package_name=package["name"],
                primary_language=package["primary_language"],
                rubric=package["rubric"],
                overall_score=package_scores[package["path"]].overall_score,
            )
            for package in packages_raw
        ]
        store.append_analysis(analysis, packages)
    return analysis


def resolve_analysis_model(
    agent: AgentAdapter | SelectedAgent, configured_model: str | None
) -> str:
    """Resolve an explicit, selected, or provider-default analysis model."""
    if configured_model is not None:
        return configured_model
    if isinstance(agent, SelectedAgent) and agent.model is not None:
        return agent.model
    try:
        return _DEFAULT_ANALYSIS_MODELS[agent.name]
    except KeyError as error:
        raise AnalyzerError(
            f"analysis model must be configured for adapter {agent.name!r}"
        ) from error


def _git_head(repository_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise AnalyzerError("repository does not have a readable Git HEAD") from error
    return result.stdout.strip()
