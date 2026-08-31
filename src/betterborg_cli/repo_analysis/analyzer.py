"""Validated repository analysis in a bounded discovery workspace."""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from betterborg_cli.agent_runtime.api_tools import READ_ONLY_API_TOOLS
from betterborg_cli.agent_runtime.base import (
    AgentAdapter,
    AgentRunSpec,
    AgentStatus,
    CancellationToken,
)
from betterborg_cli.agent_runtime.process import run_captured
from betterborg_cli.agent_runtime.selection import (
    AgentSelectionError,
    SelectedAgent,
    require_read_only_agent,
    resolve_agent_model,
)
from betterborg_cli.agent_runtime.structured import validate_structured_result
from betterborg_cli.progress import RunProgress, StageState
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
_COMMAND_STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["stage", "argv"],
    "properties": {
        "stage": {"type": "string", "minLength": 1},
        "argv": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "cwd": {"type": "string", "minLength": 1},
        "source": {"type": "string", "minLength": 1},
        "uses_services": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "required_secrets": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    },
}
_COMMAND_CATALOG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "commands": {
            "type": "array",
            "items": {"$ref": "#/$defs/command_step"},
        },
        "source": {"type": "string", "minLength": 1},
        "notes": {"type": "string"},
    },
}
_SECRET_REQUIREMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "used_by", "scope"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "used_by": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "scope": {"type": "string", "enum": ["all", "build", "agent"]},
        "source": {"type": "string", "minLength": 1},
        "description": {"type": "string"},
    },
}
_ENVIRONMENT_VARIABLE_NAME_SCHEMA: dict[str, Any] = {
    "type": "string",
    "pattern": r"^[A-Za-z_][A-Za-z0-9_]*$",
}
_SERVICE_PORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["port"],
    "properties": {
        "port": {"type": "integer", "minimum": 1, "maximum": 65535},
        "protocol": {"type": "string", "enum": ["tcp", "udp"]},
        "env": _ENVIRONMENT_VARIABLE_NAME_SCHEMA,
        "source": {"type": "string", "minLength": 1},
    },
}
_SERVICE_DEPENDENCY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "image": {"type": "string", "minLength": 1},
        "port": {"type": "integer", "minimum": 1, "maximum": 65535},
        "ports": {
            "type": "array",
            "items": {"$ref": "#/$defs/service_port"},
        },
        "source": {"type": "string", "minLength": 1},
        "compose_service": {"type": "string", "minLength": 1},
        "url_env": _ENVIRONMENT_VARIABLE_NAME_SCHEMA,
        "env": {
            "type": "array",
            "uniqueItems": True,
            "items": _ENVIRONMENT_VARIABLE_NAME_SCHEMA,
        },
    },
}
_COMPOSE_FILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["path"],
    "properties": {
        "path": {"type": "string", "minLength": 1},
        "source": {"type": "string", "minLength": 1},
        "profiles": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "services": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    },
}
_COMPOSE_CATALOG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "file": {"type": "string", "minLength": 1},
        "files": {
            "type": "array",
            "items": {"$ref": "#/$defs/compose_file"},
        },
        "source": {"type": "string", "minLength": 1},
        "profiles": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "project_name": {"type": "string", "minLength": 1},
        "notes": {"type": "string"},
    },
}
_ENVIRONMENT_COMMAND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["argv"],
    "properties": {
        "argv": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "cwd": {"type": "string", "minLength": 1},
        "source": {"type": "string", "minLength": 1},
    },
}
_ENVIRONMENT_TOOLCHAIN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "version": {
            "type": ["string", "null"],
            "minLength": 1,
        },
        "source": {"type": "string", "minLength": 1},
    },
}
_ENVIRONMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "anyOf": [
        {"required": ["files"]},
        {"required": ["toolchains"]},
        {"required": ["package_managers"]},
        {"required": ["prepare_commands"]},
        {"required": ["materialize_commands"]},
    ],
    "properties": {
        "version": {"type": "integer", "const": 1},
        "source": {"type": "string", "minLength": 1},
        "files": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "toolchains": {
            "type": "array",
            "minItems": 1,
            "items": {"$ref": "#/$defs/environment_toolchain"},
        },
        "package_managers": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "prepare_commands": {
            "type": "array",
            "minItems": 1,
            "items": {"$ref": "#/$defs/environment_command"},
        },
        "materialize_commands": {
            "type": "array",
            "minItems": 1,
            "items": {"$ref": "#/$defs/environment_command"},
        },
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
        "command_catalog": {"$ref": "#/$defs/command_catalog"},
        "environment": {"$ref": "#/$defs/environment"},
        "compose": {"$ref": "#/$defs/compose_catalog"},
        "required_secrets": {
            "type": "array",
            "items": {"$ref": "#/$defs/secret_requirement"},
        },
        "service_dependencies": {
            "type": "array",
            "items": {"$ref": "#/$defs/service_dependency"},
        },
    },
    "$defs": {
        "dimension": _DIMENSION_SCHEMA,
        "rubric": _RUBRIC_SCHEMA,
        "command_step": _COMMAND_STEP_SCHEMA,
        "command_catalog": _COMMAND_CATALOG_SCHEMA,
        "secret_requirement": _SECRET_REQUIREMENT_SCHEMA,
        "service_port": _SERVICE_PORT_SCHEMA,
        "service_dependency": _SERVICE_DEPENDENCY_SCHEMA,
        "compose_file": _COMPOSE_FILE_SCHEMA,
        "compose_catalog": _COMPOSE_CATALOG_SCHEMA,
        "environment_command": _ENVIRONMENT_COMMAND_SCHEMA,
        "environment_toolchain": _ENVIRONMENT_TOOLCHAIN_SCHEMA,
        "environment": _ENVIRONMENT_SCHEMA,
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
rationale. Every reported Harness command, environment input, Compose file,
required secret, and service must cite a manifest path or inherit a source
from its containing catalog/environment/service. Service env contains variable
names only, never values. Omit an optional category when bounded evidence is
insufficient. Return only the JSON object required by the supplied schema.
"""
_USER_PROMPT = (
    "Analyze the bounded discovery manifest and copied evidence. Treat omitted "
    "or truncated evidence as uncertainty; do not inspect the raw repository."
)
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
    progress: RunProgress | None = None,
    stage_key: str = "discover",
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = run_captured,
) -> RepositoryAnalysis:
    """Run, validate, normalize, and append one successful analysis."""
    stored_repository = store.get_repository(repository.id)
    if stored_repository != repository:
        raise ValueError("repository must already be present in the supplied store")

    artifact_dir = Path(artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = config or AnalyzerConfig()
    if resolved_config.effort is not None and agent.name == "anthropic":
        raise AnalyzerError("Anthropic does not support an effort override")

    if progress is not None:
        progress.start(stage_key)
    try:
        head_sha = _git_head(
            repository.root,
            cancel=cancel,
            command_runner=command_runner,
        )
        if workspace_dir is None:
            with tempfile.TemporaryDirectory(
                prefix="betterborg-analysis-"
            ) as temporary:
                return _run_in_workspace(
                    repository,
                    store,
                    agent,
                    artifact_dir=artifact_dir,
                    workspace_dir=Path(temporary),
                    head_sha=head_sha,
                    config=resolved_config,
                    cancel=cancel,
                    progress=progress,
                    discovery_stage_key=stage_key,
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
            progress=progress,
            discovery_stage_key=stage_key,
        )
    except BaseException as error:
        if (
            progress is not None
            and progress.stages[stage_key].state is StageState.RUNNING
        ):
            if _is_interruption(error, cancel):
                progress.stop(stage_key, "interrupted")
            else:
                progress.fail(stage_key, str(error))
        raise


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
    progress: RunProgress | None,
    discovery_stage_key: str,
) -> RepositoryAnalysis:
    manifest = build_discovery_workspace(
        repository.root,
        workspace_dir,
        limits=config.limits,
        cancel=cancel,
    )
    if progress is not None:
        progress.complete(
            discovery_stage_key,
            f"{len(manifest.files)} evidence files",
        )
        progress.start("analyze")
    try:
        require_read_only_agent(agent, role="analyzer", error_factory=AnalyzerError)
        run_id = uuid4()
        spec = AgentRunSpec(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=_USER_PROMPT,
            schema=ANALYZER_OUTPUT_SCHEMA,
            cwd=workspace_dir.resolve(),
            model=resolve_analysis_model(agent, config.model),
            effort=config.effort,
            allowed_tools=READ_ONLY_API_TOOLS,
            log_path=artifact_dir / f"{run_id}.log",
            result_path=artifact_dir / f"{run_id}.json",
            activity_sink=(
                (lambda activity: progress.activity("analyze", activity))
                if progress is not None
                else None
            ),
        )
        result = (
            agent.run_contained(spec, cancel=cancel)
            if isinstance(agent, SelectedAgent)
            else agent.run(spec, cancel=cancel)
        )
        if (
            result.status is AgentStatus.CANCELLED
            and cancel is not None
            and cancel.is_set()
        ):
            raise KeyboardInterrupt
        if result.status is not AgentStatus.COMPLETED or result.payload is None:
            raise AnalyzerError(
                result.error or f"analyzer returned {result.status.value}"
            )

        # Adapters validate their output; repeat validation at the persistence edge.
        validate_structured_result(result.payload, ANALYZER_OUTPUT_SCHEMA)
        analysis = _persist_payload(
            repository,
            store,
            result.payload,
            manifest=manifest,
            head_sha=head_sha,
            analysis_id=run_id,
        )
    except BaseException as error:
        if progress is not None:
            if _is_interruption(error, cancel):
                progress.stop("analyze", "interrupted")
            else:
                progress.fail("analyze", str(error))
        raise
    if progress is not None:
        progress.complete("analyze", f"score {analysis.overall_score:.2f}/5")
    return analysis


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
    _validate_harness_evidence(payload, manifest)
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


def _validate_harness_evidence(
    payload: Mapping[str, Any], manifest: DiscoveryManifest
) -> None:
    """Require every Harness claim to cite bounded discovery evidence."""
    cited_paths: set[str] = set()
    uncited_claims: set[str] = set()

    catalog = payload.get("command_catalog")
    if isinstance(catalog, Mapping):
        _add_source(cited_paths, catalog)
        catalog_source = _source(catalog)
        commands = catalog.get("commands")
        if isinstance(commands, list):
            for command in commands:
                _add_source(cited_paths, command)
                if _source(command) is None and catalog_source is None:
                    uncited_claims.add("command")

    environment = payload.get("environment")
    if isinstance(environment, Mapping):
        _add_source(cited_paths, environment)
        files = environment.get("files")
        environment_sources = {_source(environment)}
        if isinstance(files, list):
            file_sources = {path for path in files if isinstance(path, str)}
            cited_paths.update(file_sources)
            environment_sources.update(file_sources)
        environment_sources.discard(None)
        for key in ("toolchains", "prepare_commands", "materialize_commands"):
            records = environment.get(key)
            if isinstance(records, list):
                for record in records:
                    _add_source(cited_paths, record)
                    if _source(record) is None and not environment_sources:
                        uncited_claims.add(f"environment {key[:-1]}")
        package_managers = environment.get("package_managers")
        if (
            isinstance(package_managers, list)
            and package_managers
            and not environment_sources
        ):
            uncited_claims.add("environment package manager")

    compose = payload.get("compose")
    if isinstance(compose, Mapping):
        _add_source(cited_paths, compose)
        compose_source = _source(compose)
        primary_file = compose.get("file")
        if isinstance(primary_file, str):
            cited_paths.add(primary_file)
        files = compose.get("files")
        if isinstance(files, list):
            for compose_file in files:
                _add_source(cited_paths, compose_file)
                if isinstance(compose_file, Mapping):
                    path = compose_file.get("path")
                    if isinstance(path, str):
                        cited_paths.add(path)
                    elif _source(compose_file) is None and compose_source is None:
                        uncited_claims.add("Compose file")

    for key in ("required_secrets", "service_dependencies"):
        records = payload.get(key)
        if not isinstance(records, list):
            continue
        for record in records:
            _add_source(cited_paths, record)
            record_source = _source(record)
            if record_source is None:
                claim = "required secret" if key == "required_secrets" else "service"
                uncited_claims.add(claim)
            if key == "service_dependencies" and isinstance(record, Mapping):
                ports = record.get("ports")
                if isinstance(ports, list):
                    for port in ports:
                        _add_source(cited_paths, port)
                        if _source(port) is None and record_source is None:
                            uncited_claims.add("service port")

    if uncited_claims:
        names = ", ".join(sorted(uncited_claims))
        raise AnalyzerError(f"Harness input lacks bounded evidence for: {names}")

    known_paths = {file.path for file in manifest.files}
    unknown_sources = {
        source
        for source in cited_paths
        if not _source_is_in_manifest(source, known_paths)
    }
    if unknown_sources:
        names = ", ".join(sorted(unknown_sources))
        raise AnalyzerError(
            f"Harness input cites evidence absent from manifest: {names}"
        )


def _add_source(paths: set[str], value: object) -> None:
    if (source := _source(value)) is not None:
        paths.add(source)


def _source(value: object) -> str | None:
    if isinstance(value, Mapping) and isinstance(value.get("source"), str):
        return value["source"]
    return None


def _source_is_in_manifest(source: str, known_paths: set[str]) -> bool:
    """Accept a discovered file or an anchored location within that file."""
    if source in known_paths:
        return True

    path, marker, anchor = source.partition("#")
    if marker and path in known_paths and anchor:
        return True

    for path in known_paths:
        prefix = f"{path}/"
        if source.startswith(prefix):
            segments = source[len(prefix) :].split("/")
            return all(segment not in ("", ".", "..") for segment in segments)
    return False


def resolve_analysis_model(
    agent: AgentAdapter | SelectedAgent, configured_model: str | None
) -> str:
    """Resolve an explicit, selected, or provider-default analysis model."""
    try:
        return resolve_agent_model(agent, configured_model)
    except AgentSelectionError as error:
        raise AnalyzerError(
            f"analysis model must be configured for adapter {agent.name!r}"
        ) from error


def _git_head(
    repository_root: Path,
    *,
    cancel: CancellationToken | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = run_captured,
) -> str:
    command = ["git", "-C", str(repository_root), "rev-parse", "HEAD"]
    try:
        result = command_runner(
            command,
            check=True,
            cancel=cancel,
        )
    except subprocess.CalledProcessError as error:
        raise AnalyzerError("repository does not have a readable Git HEAD") from error
    if result.returncode == -1 and cancel is not None and cancel.is_set():
        raise KeyboardInterrupt
    head = result.stdout.strip()
    if result.returncode != 0 or not head:
        raise AnalyzerError("repository does not have a readable Git HEAD")
    return head


def _is_interruption(
    error: BaseException,
    cancel: CancellationToken | None,
) -> bool:
    return isinstance(error, KeyboardInterrupt) or (
        cancel is not None and cancel.is_set()
    )
