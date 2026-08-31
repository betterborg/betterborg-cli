"""Generate stable role prompts without coupling failures to analysis state."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
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
from betterborg_cli.agent_runtime.selection import SelectedAgent
from betterborg_cli.agent_runtime.structured import validate_structured_result
from betterborg_cli.progress import AgentActivity, RunProgress, StageState
from betterborg_cli.repo_analysis.analyzer import (
    AnalyzerError,
    resolve_analysis_model,
)
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_files import RepositoryPathError, read_repository_text
from betterborg_cli.store import (
    GeneratedPrompt,
    Repository,
    RepositoryAnalysis,
    SqliteStore,
)

PROMPT_ROLES = ("coding", "review", "merge")

PROMPT_MANAGER_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "BetterBorg repository prompt",
    "type": "object",
    "additionalProperties": False,
    "required": ["body_md"],
    "properties": {
        "body_md": {
            "type": "string",
            "minLength": 32,
            "description": "The complete role system prompt as Markdown.",
        },
        "notes": {
            "type": "string",
            "description": "Optional generation notes retained in agent artifacts.",
        },
    },
}

_COMMON_SYSTEM_PROMPT = """You are a principal prompt engineer. Generate the
complete {role} system prompt for the repository at cwd. Ground every command,
path, convention, and invariant in the persisted analyzer report or files you
inspect. Preserve the engineering discipline, testing expectations, completion
standard, and structured output contract appropriate to the role. Do not invent
commands or repository facts. Return only the object required by the schema;
body_md must start with a Markdown heading and contain no front matter or
preamble.

{role_requirements}
"""

_ROLE_REQUIREMENTS = {
    "coding": """The coding prompt must cover mission, runtime inputs, concrete
source and test layout, exact build/lint/test commands, reuse and locality,
meaningful tests using established fixtures, commit conventions, completion,
and the coding agent's final result contract.""",
    "review": """The review prompt must cover mission, runtime inputs, review
method by file class, duplication/over-abstraction/orphaned-code lenses, test
value and established test infrastructure, verification commands, sensitive
paths, blocker/major/minor severity, approval criteria, and the review result
contract. It must instruct the reviewer to inspect rather than edit.""",
    "merge": """The merge prompt must cover mission, rebase inputs, conflict
resolution using surrounding code, append-only migrations when present,
regeneration of generated code and lock files using discovered commands, Git
rules, post-merge verification, fail-loud criteria, and the merge result
contract.""",
}


@dataclass(frozen=True, slots=True)
class PromptManagerConfig:
    """Model settings for repository prompt generation."""

    model: str | None = None
    effort: str | None = None


@dataclass(frozen=True, slots=True)
class PromptGeneration:
    """The isolated outcome of generating one role prompt."""

    role: str
    ok: bool
    path: Path
    prompt: GeneratedPrompt | None = None
    error: str | None = None

    @property
    def version(self) -> int | None:
        """Return the persisted version when generation succeeded."""
        return self.prompt.version if self.prompt is not None else None


def generate_role_prompts(
    repository: Repository,
    analysis: RepositoryAnalysis,
    store: SqliteStore,
    agent: AgentAdapter | SelectedAgent,
    *,
    artifact_dir: Path,
    config: PromptManagerConfig | None = None,
    roles: Iterable[str] | None = None,
    cancel: CancellationToken | None = None,
    progress: RunProgress | None = None,
    stage_key: str = "prompts",
) -> list[PromptGeneration]:
    """Generate and persist independent coding, review, and merge prompts.

    The analysis is a prerequisite, not part of this transaction. A failed role
    therefore returns a failed outcome while the successful score and any other
    successful role prompts remain durable and inspectable.
    """
    _validate_persisted_inputs(repository, analysis, store)
    selected_roles = _validate_roles(roles)
    resolved_config = config or PromptManagerConfig()
    if resolved_config.effort is not None and agent.name == "anthropic":
        raise AnalyzerError("Anthropic does not support an effort override")
    model = resolve_analysis_model(agent, resolved_config.model)
    try:
        paths = RepoPaths.discover(repository.root, cancel=cancel)
    except BaseException:
        if cancel is not None and cancel.is_set():
            raise KeyboardInterrupt from None
        raise
    if cancel is not None and cancel.is_set():
        raise KeyboardInterrupt
    if paths.root != repository.root:
        raise ValueError("repository root does not match its discovered Git root")

    artifact_dir = Path(artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _prepare_stable_prompt_directory(paths)
    prior_prompts = store.get_latest_generated_prompts(repository.id)

    _raise_if_cancelled(cancel)
    if progress is not None:
        progress.start(stage_key)

    def generate(role: str) -> PromptGeneration:
        prompt_path = paths.prompts_dir / f"{role}.system.md"
        try:
            if progress is not None:
                progress.start_child(stage_key, role)
            _raise_if_cancelled(cancel)
            outcome = _generate_one_role(
                role=role,
                repository=repository,
                analysis=analysis,
                store=store,
                agent=agent,
                artifact_dir=artifact_dir,
                prompt_path=prompt_path,
                prior_prompt=prior_prompts.get(role),
                model=model,
                effort=resolved_config.effort,
                cancel=cancel,
                activity_sink=(
                    lambda activity: progress.child_activity(
                        stage_key, role, activity
                    )
                    if progress is not None
                    else None
                ),
            )
        except BaseException as error:
            retained = get_durable_role_prompt(
                repository,
                store,
                role=role,
                path=prompt_path,
                analysis_id=analysis.id,
            )
            if retained is not None:
                outcome = PromptGeneration(
                    role=role,
                    ok=True,
                    path=prompt_path,
                    prompt=retained,
                )
                _complete_prompt_child(progress, stage_key, outcome)
                return outcome
            _terminalize_prompt_child(
                progress,
                stage_key,
                role,
                error=str(error) or type(error).__name__,
                interrupted=_is_interruption(error, cancel),
            )
            raise

        retained = (
            get_durable_role_prompt(
                repository,
                store,
                role=role,
                path=prompt_path,
                analysis_id=analysis.id,
            )
            if outcome.ok
            else None
        )
        if retained is not None:
            outcome = PromptGeneration(
                role=role,
                ok=True,
                path=prompt_path,
                prompt=retained,
            )
            _complete_prompt_child(progress, stage_key, outcome)
            return outcome
        if outcome.ok:
            outcome = PromptGeneration(
                role=role,
                ok=False,
                path=prompt_path,
                error="prompt publication could not be reconciled",
            )
        _terminalize_prompt_child(
            progress,
            stage_key,
            role,
            error=outcome.error or "generation failed",
            interrupted=cancel is not None and cancel.is_set(),
        )
        return outcome

    outcomes: list[PromptGeneration] = []
    errors: list[BaseException] = []
    try:
        with ThreadPoolExecutor(max_workers=len(selected_roles)) as executor:
            futures = [executor.submit(generate, role) for role in selected_roles]
            for future in futures:
                try:
                    outcomes.append(future.result())
                except BaseException as error:
                    errors.append(error)
    except BaseException as error:
        errors.append(error)
        raise
    finally:
        _terminalize_prompt_parent(
            progress,
            stage_key,
            selected_roles,
            outcomes,
            errors,
            cancel,
        )
    if errors:
        if any(_is_interruption(error, cancel) for error in errors):
            raise KeyboardInterrupt
        raise errors[0]
    if cancel is not None and cancel.is_set() and any(not run.ok for run in outcomes):
        raise KeyboardInterrupt
    by_role = {outcome.role: outcome for outcome in outcomes}
    return [by_role[role] for role in selected_roles]


def _generate_one_role(
    *,
    role: str,
    repository: Repository,
    analysis: RepositoryAnalysis,
    store: SqliteStore,
    agent: AgentAdapter | SelectedAgent,
    artifact_dir: Path,
    prompt_path: Path,
    prior_prompt: GeneratedPrompt | None,
    model: str,
    effort: str | None,
    cancel: CancellationToken | None,
    activity_sink: Callable[[AgentActivity], None] | None,
) -> PromptGeneration:
    spec = AgentRunSpec(
        system_prompt=_system_prompt(role),
        user_prompt=_render_user_prompt(
            role=role,
            analysis_json=analysis.analysis_json,
            prior_prompt=prior_prompt,
        ),
        schema=PROMPT_MANAGER_OUTPUT_SCHEMA,
        cwd=repository.root,
        model=model,
        effort=effort,
        allowed_tools=READ_ONLY_API_TOOLS,
        log_path=artifact_dir / f"{analysis.id}.{role}.log",
        result_path=artifact_dir / f"{analysis.id}.{role}.json",
        activity_sink=activity_sink,
    )
    try:
        result = agent.run(spec, cancel=cancel)
    except Exception as error:  # adapters are isolated at the role boundary
        return PromptGeneration(
            role=role,
            ok=False,
            path=prompt_path,
            error=f"adapter crashed: {error}",
        )
    if result.status is not AgentStatus.COMPLETED or result.payload is None:
        return PromptGeneration(
            role=role,
            ok=False,
            path=prompt_path,
            error=result.error or f"prompt manager returned {result.status.value}",
        )

    try:
        validate_structured_result(result.payload, PROMPT_MANAGER_OUTPUT_SCHEMA)
        body_md = result.payload["body_md"]
        if not body_md.strip():
            raise ValueError("prompt manager returned an empty body_md")
        with _stable_prompt_publication(
            prompt_path,
            body_md,
            repository.root,
        ) as publish:
            with store.transaction():
                prompt = store.append_generated_prompt(
                    repository_id=repository.id,
                    analysis_id=analysis.id,
                    role=role,
                    body_md=body_md,
                )
                publish()
    except Exception as error:
        return PromptGeneration(
            role=role,
            ok=False,
            path=prompt_path,
            error=f"prompt could not be recorded: {error}",
        )
    return PromptGeneration(
        role=role,
        ok=True,
        path=prompt_path,
        prompt=prompt,
    )


def get_durable_role_prompt(
    repository: Repository,
    store: SqliteStore,
    *,
    role: str,
    path: Path,
    analysis_id: UUID | None = None,
) -> GeneratedPrompt | None:
    """Return the latest prompt only when its stable file matches metadata."""
    prompt = store.get_latest_generated_prompts(repository.id).get(role)
    if prompt is None or (
        analysis_id is not None and prompt.analysis_id != analysis_id
    ):
        return None
    try:
        body = read_repository_text(path, root=repository.root)
    except (OSError, UnicodeError, RepositoryPathError):
        return None
    return prompt if body == prompt.body_md else None


def _complete_prompt_child(
    progress: RunProgress | None,
    stage_key: str,
    outcome: PromptGeneration,
) -> None:
    if progress is not None:
        progress.complete_child(
            stage_key,
            outcome.role,
            f"prompt v{outcome.version}",
        )


def _terminalize_prompt_child(
    progress: RunProgress | None,
    stage_key: str,
    role: str,
    *,
    error: str,
    interrupted: bool,
) -> None:
    if progress is None:
        return
    child = progress.stages[stage_key].children[role]
    if child.state is not StageState.RUNNING:
        return
    if interrupted:
        progress.stop_child(stage_key, role, "interrupted")
    else:
        progress.fail_child(stage_key, role, error)


def _terminalize_prompt_parent(
    progress: RunProgress | None,
    stage_key: str,
    selected_roles: tuple[str, ...],
    outcomes: list[PromptGeneration],
    errors: list[BaseException],
    cancel: CancellationToken | None,
) -> None:
    if progress is None:
        return
    parent = progress.stages[stage_key]
    if parent.state is not StageState.RUNNING:
        return
    children = [parent.children[role] for role in selected_roles]
    if any(child.state is StageState.RUNNING for child in children):
        return
    if any(child.state is StageState.STOPPED for child in children) or any(
        _is_interruption(error, cancel) for error in errors
    ):
        progress.stop(stage_key, "interrupted")
    elif errors or any(not outcome.ok for outcome in outcomes):
        progress.fail(stage_key, "prompt generation incomplete")
    else:
        progress.complete(stage_key, f"{len(parent.children)} prompts")


def _is_interruption(
    error: BaseException,
    cancel: CancellationToken | None,
) -> bool:
    return isinstance(error, KeyboardInterrupt) or (
        cancel is not None and cancel.is_set()
    )


def _raise_if_cancelled(cancel: CancellationToken | None) -> None:
    if cancel is not None and cancel.is_set():
        raise KeyboardInterrupt


def _render_user_prompt(
    *,
    role: str,
    analysis_json: dict[str, Any],
    prior_prompt: GeneratedPrompt | None,
) -> str:
    parts = [
        "# Persisted repository analysis",
        "```json",
        json.dumps(analysis_json, indent=2, sort_keys=True),
        "```",
    ]
    if prior_prompt is not None:
        parts.extend(
            [
                "",
                f"# Prior {role} prompt",
                "Keep what still applies and refresh facts changed by the analysis.",
                "",
                prior_prompt.body_md,
            ]
        )
    return "\n".join(parts)


def _system_prompt(role: str) -> str:
    return _COMMON_SYSTEM_PROMPT.format(
        role=role,
        role_requirements=_ROLE_REQUIREMENTS[role],
    )


def _validate_persisted_inputs(
    repository: Repository,
    analysis: RepositoryAnalysis,
    store: SqliteStore,
) -> None:
    if store.get_repository(repository.id) != repository:
        raise ValueError("repository must already be present in the supplied store")
    if analysis.repository_id != repository.id:
        raise ValueError("analysis does not belong to the supplied repository")
    if store.get_analysis(analysis.id) != analysis:
        raise ValueError("analysis must already be present in the supplied store")


def _validate_roles(roles: Iterable[str] | None) -> tuple[str, ...]:
    selected = tuple(PROMPT_ROLES if roles is None else roles)
    if not selected:
        raise ValueError("at least one prompt role must be selected")
    if len(selected) != len(set(selected)):
        raise ValueError("prompt roles must not contain duplicates")
    unknown = [role for role in selected if role not in PROMPT_ROLES]
    if unknown:
        raise ValueError(f"unknown prompt role: {unknown[0]!r}")
    return selected


def _prepare_stable_prompt_directory(paths: RepoPaths) -> None:
    prompt_directory = paths.prompts_dir
    resolved = prompt_directory.resolve()
    if not resolved.is_relative_to(paths.root):
        raise ValueError(
            f"stable prompt directory escapes repository: {prompt_directory}"
        )
    prompt_directory.mkdir(parents=True, exist_ok=True)
    resolved = prompt_directory.resolve(strict=True)
    if not resolved.is_relative_to(paths.root):
        raise ValueError(
            f"stable prompt directory escapes repository: {prompt_directory}"
        )


@contextmanager
def _stable_prompt_publication(
    path: Path,
    body_md: str,
    repository_root: Path,
) -> Iterator[Callable[[], None]]:
    resolved_directory = path.parent.resolve(strict=True)
    if not resolved_directory.is_relative_to(repository_root):
        raise ValueError(f"stable prompt path escapes repository: {path}")
    path = resolved_directory / path.name
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    backup = path.with_name(f".{path.name}.{uuid4().hex}.bak")
    prior_moved = False
    published = False

    def publish() -> None:
        nonlocal prior_moved, published
        if path.exists() or path.is_symlink():
            os.replace(path, backup)
            prior_moved = True
        try:
            os.replace(temporary, path)
        except BaseException:
            if prior_moved:
                os.replace(backup, path)
                prior_moved = False
            raise
        published = True

    try:
        temporary.write_text(body_md, encoding="utf-8")
        try:
            yield publish
            if not published:
                raise RuntimeError("stable prompt was not published")
        except BaseException as error:
            if published:
                try:
                    if prior_moved:
                        os.replace(backup, path)
                        prior_moved = False
                    else:
                        path.unlink(missing_ok=True)
                    published = False
                except BaseException as rollback_error:
                    raise RuntimeError(
                        f"{error}; stable prompt rollback failed: {rollback_error}"
                    ) from rollback_error
            raise
        else:
            backup.unlink(missing_ok=True)
            prior_moved = False
    finally:
        temporary.unlink(missing_ok=True)
