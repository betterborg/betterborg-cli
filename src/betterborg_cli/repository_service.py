"""Repository registration and explicit analysis orchestration."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from betterborg_cli.agent_runtime.base import AgentAdapter
from betterborg_cli.agent_runtime.selection import SelectedAgent
from betterborg_cli.repo_analysis import (
    PROMPT_ROLES,
    ImprovementPrd,
    PromptGeneration,
    build_machine_report,
    generate_improvement_prds,
    generate_role_prompts,
    render_markdown_report,
    resolve_theme_key,
    run_analyzer,
)
from betterborg_cli.repo_paths import RepoPaths, ensure_managed_gitignore
from betterborg_cli.repository_config import (
    CONFIG_FILENAME,
    CONFIG_VERSION,
    RepositoryConfig,
    load_repository_config,
)
from betterborg_cli.repository_files import (
    RepositoryPathError,
    publish_repository_text,
)
from betterborg_cli.store import Operation, Repository, RepositoryAnalysis, SqliteStore

_INITIALIZED_OPERATION = "repository.initialized"

AnalysisAgent: TypeAlias = AgentAdapter | SelectedAgent
AgentFactory: TypeAlias = Callable[[RepositoryConfig], AnalysisAgent]


class RepositoryInitializationError(RuntimeError):
    """Raised when repository initialization cannot produce every output."""


@dataclass(frozen=True, slots=True)
class RepositoryInitialization:
    """Outcome of one idempotent repository initialization request."""

    repository: Repository
    analysis: RepositoryAnalysis
    initialized: bool
    score_path: Path
    prompts: tuple[PromptGeneration, ...] = ()
    improvement_prds: tuple[ImprovementPrd, ...] = ()


@dataclass(frozen=True, slots=True)
class RepositoryReanalysis:
    """Outcome of one explicit analysis of an initialized repository."""

    repository: Repository
    analysis: RepositoryAnalysis
    previous_analysis: RepositoryAnalysis
    score_path: Path
    prompts: tuple[PromptGeneration, ...]
    improvement_prds: tuple[ImprovementPrd, ...]


class RepositoryService:
    """Own repository identity and its generated analysis outputs."""

    def __init__(
        self,
        paths: RepoPaths,
        store: SqliteStore,
        agent_factory: AgentFactory,
    ) -> None:
        self.paths = paths
        self.store = store
        self._agent_factory = agent_factory

    def initialize(self) -> RepositoryInitialization:
        """Register and analyze a repository once, resuming partial attempts."""
        repository, config = self._ensure_repository()
        ensure_managed_gitignore(self.paths)

        analysis = self.store.get_prior_ready_analysis(repository.id)
        if self._is_initialized(repository):
            if analysis is None:
                raise RepositoryInitializationError(
                    "repository initialization record has no persisted analysis"
                )
            return RepositoryInitialization(
                repository=repository,
                analysis=analysis,
                initialized=False,
                score_path=self.paths.score_report,
            )

        agent = self._agent_factory(config)
        if analysis is None:
            analysis = run_analyzer(
                repository,
                self.store,
                agent,
                artifact_dir=self.paths.artifacts_dir / "analysis",
            )

        self._write_score(analysis)
        prompt_runs = self._generate_missing_prompts(repository, analysis, agent)
        _require_complete_prompts(prompt_runs)

        improvement_prds = generate_improvement_prds(
            analysis,
            self.paths,
            _suggested_borg_names(analysis),
        )
        self.store.append_operation(
            Operation(
                repository_id=repository.id,
                kind=_INITIALIZED_OPERATION,
                payload={"analysis_id": str(analysis.id)},
            )
        )
        return RepositoryInitialization(
            repository=repository,
            analysis=analysis,
            initialized=True,
            score_path=self.paths.score_report,
            prompts=prompt_runs,
            improvement_prds=improvement_prds,
        )

    def analyze(self) -> RepositoryReanalysis:
        """Append an analysis and refresh outputs for an initialized repository."""
        repository, config = self._registered_repository()
        if self.store.get_prior_ready_analysis(repository.id) is None:
            raise RepositoryInitializationError(
                "repository initialization record has no persisted analysis"
            )
        agent = self._agent_factory(config)
        analysis = run_analyzer(
            repository,
            self.store,
            agent,
            artifact_dir=self.paths.artifacts_dir / "analysis",
        )
        previous_analysis = self.store.get_prior_ready_analysis(
            repository.id,
            before_analysis_id=analysis.id,
        )
        if previous_analysis is None or analysis.score_delta is None:
            raise RepositoryInitializationError(
                "explicit analysis has no persisted predecessor"
            )

        self._write_score(analysis)
        prompt_runs = tuple(
            generate_role_prompts(
                repository,
                analysis,
                self.store,
                agent,
                artifact_dir=self.paths.artifacts_dir / "prompts",
                roles=PROMPT_ROLES,
            )
        )
        _require_complete_prompts(prompt_runs)

        improvement_prds = generate_improvement_prds(
            analysis,
            self.paths,
            _suggested_borg_names(analysis),
        )
        return RepositoryReanalysis(
            repository=repository,
            analysis=analysis,
            previous_analysis=previous_analysis,
            score_path=self.paths.score_report,
            prompts=prompt_runs,
            improvement_prds=improvement_prds,
        )

    def _registered_repository(self) -> tuple[Repository, RepositoryConfig]:
        if not (self.paths.tracked_dir / CONFIG_FILENAME).is_file():
            raise RepositoryInitializationError(
                "repository is not initialized; run 'borg init' first"
            )
        config = load_repository_config(self.paths)
        repository = self.store.get_repository(config.repository_id)
        if repository is None or not self._is_initialized(repository):
            raise RepositoryInitializationError(
                "repository is not initialized; run 'borg init' first"
            )
        if repository.root != self.paths.root:
            raise RepositoryInitializationError(
                "tracked repository identity belongs to a different repository root"
            )
        return repository, config

    def _ensure_repository(self) -> tuple[Repository, RepositoryConfig]:
        config_path = self.paths.tracked_dir / CONFIG_FILENAME
        if not config_path.exists():
            self._write_initial_config(Repository(root=self.paths.root))
        config = load_repository_config(self.paths)
        repository = Repository(root=self.paths.root, id=config.repository_id)

        stored = self.store.get_repository(repository.id)
        if stored is None:
            self.store.add_repository(repository)
        elif stored.root != repository.root:
            raise RepositoryInitializationError(
                "tracked repository identity belongs to a different repository root"
            )
        else:
            repository = stored
        return repository, config

    def _write_initial_config(self, repository: Repository) -> None:
        tracked_dir = self.paths.tracked_dir
        if not tracked_dir.resolve().is_relative_to(self.paths.root):
            raise RepositoryInitializationError(
                f"tracked Borg directory escapes repository: {tracked_dir}"
            )
        tracked_dir.mkdir(parents=True, exist_ok=True)
        if not tracked_dir.resolve(strict=True).is_relative_to(self.paths.root):
            raise RepositoryInitializationError(
                f"tracked Borg directory escapes repository: {tracked_dir}"
            )
        default_branch = _default_branch(self.paths.root)
        body = (
            f"version = {CONFIG_VERSION}\n\n"
            "[repository]\n"
            f'id = "{repository.id}"\n'
            f"default_branch = {json.dumps(default_branch, ensure_ascii=False)}\n"
        )
        try:
            with (tracked_dir / CONFIG_FILENAME).open(
                "x", encoding="utf-8", errors="strict", newline="\n"
            ) as config_file:
                config_file.write(body)
        except FileExistsError:
            # Another initializer won the creation race; its identity is canonical.
            return

    def _is_initialized(self, repository: Repository) -> bool:
        return any(
            operation.kind == _INITIALIZED_OPERATION
            for operation in self.store.list_operations(repository.id)
        )

    def _write_score(self, analysis: RepositoryAnalysis) -> None:
        packages = self.store.list_packages(analysis.id)
        report = render_markdown_report(build_machine_report(analysis, packages))
        _publish_text(self.paths.score_report, report, root=self.paths.root)

    def _generate_missing_prompts(
        self,
        repository: Repository,
        analysis: RepositoryAnalysis,
        agent: AnalysisAgent,
    ) -> tuple[PromptGeneration, ...]:
        latest = self.store.get_latest_generated_prompts(repository.id)
        missing_roles = tuple(
            role
            for role in PROMPT_ROLES
            if role not in latest
            or not (self.paths.prompts_dir / f"{role}.system.md").is_file()
        )
        if not missing_roles:
            return ()
        return tuple(
            generate_role_prompts(
                repository,
                analysis,
                self.store,
                agent,
                artifact_dir=self.paths.artifacts_dir / "prompts",
                roles=missing_roles,
            )
        )


def _default_branch(repository_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "symbolic-ref", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RepositoryInitializationError(
            "cannot initialize a repository while Git HEAD is detached"
        ) from error
    branch = result.stdout.strip()
    if not branch:
        raise RepositoryInitializationError("cannot determine the default Git branch")
    return branch


def _require_complete_prompts(prompt_runs: tuple[PromptGeneration, ...]) -> None:
    failures = [run for run in prompt_runs if not run.ok]
    if not failures:
        return
    details = "; ".join(
        f"{run.role}: {run.error or 'generation failed'}" for run in failures
    )
    raise RepositoryInitializationError(
        f"repository prompt generation was incomplete: {details}"
    )


def _suggested_borg_names(
    analysis: RepositoryAnalysis,
) -> Mapping[str, str]:
    themes = analysis.analysis_json.get("themes")
    if not isinstance(themes, list):
        raise RepositoryInitializationError(
            "persisted analysis has no recommendation themes"
        )
    suggestions: dict[str, str] = {}
    for theme in themes:
        if not isinstance(theme, Mapping) or not isinstance(theme.get("id"), str):
            raise RepositoryInitializationError(
                "persisted analysis contains an invalid recommendation theme"
            )
        key = resolve_theme_key(theme["id"])
        suggestions[key] = key
    return suggestions


def _publish_text(path: Path, body: str, *, root: Path) -> None:
    try:
        publish_repository_text(path, body, root=root, overwrite=True)
    except RepositoryPathError as error:
        raise RepositoryInitializationError(str(error)) from error
