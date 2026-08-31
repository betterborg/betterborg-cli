"""Repository registration and explicit analysis orchestration."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from betterborg_cli.agent_runtime.base import AgentAdapter, CancellationToken
from betterborg_cli.agent_runtime.process import run_captured
from betterborg_cli.agent_runtime.selection import SelectedAgent
from betterborg_cli.progress import ChildSpec, RunProgress, StageSpec
from betterborg_cli.repo_analysis import (
    PROMPT_ROLES,
    ImprovementPrd,
    PromptGeneration,
    build_machine_report,
    generate_improvement_prds,
    generate_role_prompts,
    get_durable_role_prompt,
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
from betterborg_cli.repository_files import RepositoryPathError, publish_repository_text
from betterborg_cli.store import Operation, Repository, RepositoryAnalysis, SqliteStore

_INITIALIZED_OPERATION = "repository.initialized"
_PROMPTS_STAGE_KEY = "prompts"

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
        *,
        cancel: CancellationToken | None = None,
        progress: RunProgress | None = None,
    ) -> None:
        self.paths = paths
        self.store = store
        self._agent_factory = agent_factory
        self.cancel = cancel
        self.progress = progress
        if progress is not None:
            progress.declare(StageSpec("discover", "Discover evidence"))
            progress.declare(StageSpec("analyze", "Analyze repository"))
            progress.declare(
                StageSpec(
                    _PROMPTS_STAGE_KEY,
                    "Generate role prompts",
                    tuple(
                        ChildSpec(role, f"{role.title()} prompt")
                        for role in PROMPT_ROLES
                    ),
                )
            )

    def initialize(self) -> RepositoryInitialization:
        """Register and analyze a repository once, resuming partial attempts."""
        repository, config = self._ensure_repository()
        ensure_managed_gitignore(self.paths)

        analysis = self.store.get_prior_ready_analysis(repository.id)
        if analysis is not None:
            self._seed_retained_analysis(analysis)
        retained_prompt_roles = self._seed_retained_prompts(repository)
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
                cancel=self.cancel,
                progress=self.progress,
            )

        self._write_score(analysis)
        prompt_runs = self._generate_missing_prompts(
            repository,
            analysis,
            agent,
            retained_prompt_roles=retained_prompt_roles,
        )
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
            cancel=self.cancel,
            progress=self.progress,
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
                cancel=self.cancel,
                progress=self.progress,
                stage_key=_PROMPTS_STAGE_KEY,
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
        default_branch = _default_branch(self.paths.root, cancel=self.cancel)
        body = (
            f"version = {CONFIG_VERSION}\n\n"
            "[repository]\n"
            f'id = "{repository.id}"\n'
            f"default_branch = {json.dumps(default_branch, ensure_ascii=False)}\n"
        )
        try:
            publish_repository_text(
                self.paths.tracked_dir / CONFIG_FILENAME,
                body,
                root=self.paths.root,
                overwrite=False,
            )
        except FileExistsError:
            # Another initializer won the creation race; its identity is canonical.
            return
        except RepositoryPathError as error:
            raise RepositoryInitializationError(str(error)) from error

    def _is_initialized(self, repository: Repository) -> bool:
        return any(
            operation.kind == _INITIALIZED_OPERATION
            for operation in self.store.list_operations(repository.id)
        )

    def _seed_retained_analysis(self, analysis: RepositoryAnalysis) -> None:
        if self.progress is None:
            return
        self.progress.seed_completed(
            "discover",
            f"evidence retained for analysis {analysis.id}",
        )
        self.progress.seed_completed(
            "analyze",
            f"score {analysis.overall_score:.2f}/5",
        )

    def _write_score(self, analysis: RepositoryAnalysis) -> None:
        packages = self.store.list_packages(analysis.id)
        report = render_markdown_report(build_machine_report(analysis, packages))
        _publish_text(self.paths.score_report, report, root=self.paths.root)

    def _seed_retained_prompts(self, repository: Repository) -> frozenset[str]:
        retained_roles = frozenset(
            role
            for role in PROMPT_ROLES
            if get_durable_role_prompt(
                repository,
                self.store,
                role=role,
                path=self.paths.prompts_dir / f"{role}.system.md",
            )
            is not None
        )
        if self.progress is None:
            return retained_roles
        for role in PROMPT_ROLES:
            if role in retained_roles:
                self.progress.seed_child_completed(
                    _PROMPTS_STAGE_KEY,
                    role,
                    "prompt retained",
                )
        if len(retained_roles) == len(PROMPT_ROLES):
            self.progress.seed_completed(
                _PROMPTS_STAGE_KEY,
                f"{len(PROMPT_ROLES)} prompts retained",
            )
        return retained_roles

    def _generate_missing_prompts(
        self,
        repository: Repository,
        analysis: RepositoryAnalysis,
        agent: AnalysisAgent,
        *,
        retained_prompt_roles: frozenset[str],
    ) -> tuple[PromptGeneration, ...]:
        missing_roles = tuple(
            role for role in PROMPT_ROLES if role not in retained_prompt_roles
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
                cancel=self.cancel,
                progress=self.progress,
                stage_key=_PROMPTS_STAGE_KEY,
            )
        )


def _default_branch(
    repository_root: Path,
    *,
    cancel: CancellationToken | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = run_captured,
) -> str:
    command = ["git", "-C", str(repository_root), "symbolic-ref", "--short", "HEAD"]
    try:
        result = command_runner(
            command,
            check=True,
            cancel=cancel,
        )
    except subprocess.CalledProcessError as error:
        raise RepositoryInitializationError(
            "cannot initialize a repository while Git HEAD is detached"
        ) from error
    if result.returncode == -1 and cancel is not None and cancel.is_set():
        raise KeyboardInterrupt
    branch = result.stdout.strip()
    if result.returncode != 0:
        raise RepositoryInitializationError(
            "cannot initialize a repository while Git HEAD is detached"
        )
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
