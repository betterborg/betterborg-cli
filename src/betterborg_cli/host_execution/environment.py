"""Reusable host caches and checkout-local environment materialization."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import stat
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from urllib.parse import quote
from uuid import UUID

from betterborg_cli.agent_runtime import CancellationToken
from betterborg_cli.agent_runtime.process import run_captured
from betterborg_cli.host_execution._locking import path_lock
from betterborg_cli.host_execution.git import SafeGit
from betterborg_cli.host_execution.guard import PrimaryCheckoutGuard
from betterborg_cli.host_execution.preflight import HostCommand, HostPreflightPlan
from betterborg_cli.host_execution.scheduler import TaskActivitySink
from betterborg_cli.progress import AgentActivity, AgentActivityKind
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.store import (
    EnvironmentAttempt,
    ExecutionAttemptStatus,
    SqliteStore,
    TaskClaim,
    TaskRuntime,
    TaskRuntimeStatus,
)
from betterborg_cli.store.models import utcnow

_CACHE_CONTRACT_VERSION = 1
_SAFE_HOST_ENVIRONMENT = (
    "LANG",
    "LC_ALL",
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "TMPDIR",
)


class EnvironmentMaterializationError(RuntimeError):
    """Raised when a claimed task cannot safely consume its environment."""


@dataclass(frozen=True, slots=True)
class EnvironmentMaterialization:
    """Result of preparing a cache and materializing one exact checkout."""

    fingerprint: str
    cache_path: Path
    preparation_reused: bool
    materialization_reused: bool
    environment: Mapping[str, str] = field(repr=False, hash=False)


@dataclass(frozen=True, slots=True)
class _EnvironmentDescriptor:
    relative_path: Path
    content: bytes


@dataclass(frozen=True, slots=True)
class _PathSnapshot:
    kind: str
    content: bytes | None = None
    mode: int | None = None
    link_target: str | None = None


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
ActivitySink = Callable[[AgentActivity], None]
Clock = Callable[[], datetime]


def environment_fingerprint(plan: HostPreflightPlan, worktree: Path) -> str:
    """Fingerprint analyzer inputs using their bytes in one exact worktree."""
    descriptors = _environment_descriptors(plan, worktree)
    return _fingerprint_descriptors(plan, descriptors)


def _environment_descriptors(
    plan: HostPreflightPlan, worktree: Path
) -> tuple[_EnvironmentDescriptor, ...]:
    root = Path(worktree).resolve()
    if not root.is_dir():
        raise EnvironmentMaterializationError(
            f"task worktree does not exist: {root}"
        )

    descriptors: list[_EnvironmentDescriptor] = []
    for source in plan.environment_files:
        try:
            relative = source.relative_to(plan.repository_root)
        except ValueError as error:
            raise EnvironmentMaterializationError(
                f"environment descriptor is outside the repository: {source}"
            ) from error
        candidate = root / relative
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise EnvironmentMaterializationError(
                f"environment descriptor is missing from task worktree: {relative}"
            ) from error
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise EnvironmentMaterializationError(
                f"environment descriptor escapes task worktree: {relative}"
            )
        try:
            content = resolved.read_bytes()
        except OSError as error:
            raise EnvironmentMaterializationError(
                f"unable to read environment descriptor {relative}: {error}"
            ) from error
        descriptors.append(
            _EnvironmentDescriptor(relative_path=relative, content=content)
        )
    return tuple(descriptors)


def _fingerprint_descriptors(
    plan: HostPreflightPlan,
    descriptors: Sequence[_EnvironmentDescriptor],
) -> str:
    files = [
        {
            "path": descriptor.relative_path.as_posix(),
            "sha256": hashlib.sha256(descriptor.content).hexdigest(),
        }
        for descriptor in descriptors
    ]

    payload = {
        "contract": _CACHE_CONTRACT_VERSION,
        "files": sorted(files, key=lambda item: item["path"]),
        "materialize_commands": _command_payload(plan.materialize_commands),
        "package_managers": sorted(plan.package_managers),
        "prepare_commands": _command_payload(plan.prepare_commands),
        # Host caches are machine-local.  Scoping by the trusted checkout root
        # prevents identical manifests in two repositories from sharing state.
        "repository_root": str(plan.repository_root.resolve()),
        "secret_requirements": sorted(
            (
                {
                    "name": secret.name,
                    "scope": secret.scope,
                    "used_by": sorted(secret.used_by),
                }
                for secret in plan.secret_requirements
            ),
            key=lambda item: item["name"],
        ),
        "toolchains": sorted(
            (
                {"name": executable.name, "version": executable.version}
                for executable in plan.executables
            ),
            key=lambda item: item["name"],
        ),
    }
    encoded = json.dumps(
        payload, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def package_manager_cache_environment(
    cache_path: Path, package_managers: Sequence[str]
) -> dict[str, str]:
    """Return stable package-manager cache variables for one fingerprint."""
    cache = Path(cache_path).resolve()
    managers = {manager.lower() for manager in package_managers}
    result = {
        "BETTERBORG_ENVIRONMENT_ROOT": str(cache),
        "XDG_CACHE_HOME": str(cache / "xdg" / "cache"),
        "XDG_DATA_HOME": str(cache / "xdg" / "data"),
        "XDG_STATE_HOME": str(cache / "xdg" / "state"),
    }

    if managers & {"npm", "node"}:
        result.update(
            {
                "COREPACK_HOME": str(cache / "corepack"),
                "npm_config_cache": str(cache / "npm" / "cache"),
            }
        )
    if "pnpm" in managers:
        store = str(cache / "pnpm" / "store")
        result.update(
            {
                "COREPACK_HOME": str(cache / "corepack"),
                "PNPM_HOME": str(cache / "pnpm" / "home"),
                "PNPM_STORE_DIR": store,
                "npm_config_cache": str(cache / "npm" / "cache"),
                "npm_config_store_dir": store,
                "pnpm_config_store_dir": store,
            }
        )
    if "yarn" in managers:
        result.update(
            {
                "COREPACK_HOME": str(cache / "corepack"),
                "YARN_CACHE_FOLDER": str(cache / "yarn" / "cache"),
                "YARN_ENABLE_GLOBAL_CACHE": "true",
                "YARN_GLOBAL_FOLDER": str(cache / "yarn" / "berry"),
            }
        )
    if managers & {"pip", "python"}:
        result["PIP_CACHE_DIR"] = str(cache / "pip" / "cache")
    if "uv" in managers:
        result["UV_CACHE_DIR"] = str(cache / "uv" / "cache")
    if "poetry" in managers:
        result.update(
            {
                "POETRY_CACHE_DIR": str(cache / "poetry" / "cache"),
                "POETRY_DATA_DIR": str(cache / "poetry" / "data"),
                "POETRY_VIRTUALENVS_IN_PROJECT": "true",
            }
        )
    if managers & {"cargo", "rust"}:
        result.update(
            {
                "CARGO_HOME": str(cache / "cargo"),
                "RUSTUP_HOME": str(cache / "rustup"),
            }
        )
    if "go" in managers:
        result.update(
            {
                "GOCACHE": str(cache / "go" / "cache"),
                "GOMODCACHE": str(cache / "go" / "pkg" / "mod"),
                "GOPATH": str(cache / "go"),
            }
        )
    if managers & {"bundler", "bundle", "ruby"}:
        # BUNDLE_PATH and GEM_HOME are installation locations, not download
        # caches.  Setting either here would put the consumer's installed gems
        # in shared state instead of its own checkout.
        result["BUNDLE_USER_CACHE"] = str(cache / "bundler")
    return result


class HostEnvironmentManager:
    """Prepare one reusable cache and materialize every claimed worktree."""

    def __init__(
        self,
        repository_root: Path,
        *,
        cache_root: Path | None = None,
        preparation_root: Path | None = None,
        environment: Mapping[str, str] | None = None,
        command_runner: CommandRunner | None = None,
        activity: ActivitySink | None = None,
        clock: Clock = utcnow,
        cancel: CancellationToken | None = None,
        git: SafeGit | None = None,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self._paths = RepoPaths.discover(self.repository_root, cancel=cancel)
        if self._paths.root != self.repository_root:
            raise EnvironmentMaterializationError(
                "environment manager must be bound to the Git worktree root"
            )
        self.cache_root = Path(
            cache_root or self._paths.state_dir / "environment-cache"
        ).resolve()
        default_preparation_root = (
            self.repository_root.parent
            / ".betterborg-environments"
            / self.repository_root.name
        )
        self.preparation_root = Path(
            preparation_root or default_preparation_root
        ).resolve()
        if git is not None and git.cwd != self.repository_root:
            raise EnvironmentMaterializationError(
                "environment manager Git binding must match repository"
            )
        self._git = git or SafeGit(self.repository_root, cancel=cancel)
        self._validate_managed_paths()
        self._environment = dict(os.environ if environment is None else environment)
        self._run = command_runner or run_captured
        self._activity = activity
        self._cancel = cancel
        self._clock = clock
        self._guard = PrimaryCheckoutGuard(
            self.repository_root, git=self._git
        )

    def prepare_reusable_caches(
        self,
        store: SqliteStore,
        plan: HostPreflightPlan,
        run_id: UUID,
        owner_token: str,
        worktrees: Sequence[tuple[UUID, Path]],
        *,
        secret_values: Mapping[str, str] | None = None,
        activity: TaskActivitySink | None = None,
    ) -> tuple[str, ...]:
        """Prepare every distinct task fingerprint before claims may dispatch."""
        if plan.repository_root.resolve() != self.repository_root:
            raise EnvironmentMaterializationError(
                "preflight plan belongs to a different repository"
            )
        if not plan.prepare_commands:
            return ()

        prepared: list[str] = []
        for task_id, source_worktree in worktrees:
            source = Path(source_worktree).resolve()
            descriptors = _environment_descriptors(plan, source)
            fingerprint = _fingerprint_descriptors(plan, descriptors)
            cache_path = self._cache_path(fingerprint)
            cache_path.mkdir(parents=True, exist_ok=True)
            base_environment = self._base_command_environment(plan, cache_path)
            command_environments = self._command_environments(
                plan, base_environment, secret_values or {}
            )
            with _preparation_lock(cache_path):
                marker = cache_path / ".betterborg-prepared"
                completed = store.find_completed_environment_attempt(
                    fingerprint, kind="prepare"
                )
                if completed is not None and _prepared_marker_matches(
                    marker, fingerprint
                ):
                    continue
                self._record_attempt(
                    store,
                    None,
                    owner_token,
                    run_id=run_id,
                    task_id=task_id,
                    kind="prepare",
                    fingerprint=fingerprint,
                    commands=plan.prepare_commands,
                    worktree=None,
                    preparation_source=source,
                    preparation_descriptors=descriptors,
                    cache_path=cache_path,
                    completion_marker=marker,
                    command_environments=command_environments,
                    prepared_before_dispatch=True,
                    activity=(
                        partial(activity, task_id)
                        if activity is not None
                        else None
                    ),
                )
            prepared.append(fingerprint)
        return tuple(prepared)

    def materialize_claimed_task(
        self,
        store: SqliteStore,
        plan: HostPreflightPlan,
        claim: TaskClaim,
        owner_token: str,
        *,
        secret_values: Mapping[str, str] | None = None,
        activity: ActivitySink | None = None,
        task_transition: Callable[..., TaskRuntime] | None = None,
    ) -> EnvironmentMaterialization:
        """Move one claimed task through environment setup into coding.

        A successful attempt is reused only for the exact descriptor
        fingerprint.  A descriptor edit therefore creates a new cache and a
        new checkout-local materialization before any consumer can run.
        """
        if plan.repository_root.resolve() != self.repository_root:
            raise EnvironmentMaterializationError(
                "preflight plan belongs to a different repository"
            )
        runtime = store.get_task_runtime(claim.task_id)
        if runtime is None or runtime.worktree_path is None:
            raise EnvironmentMaterializationError(
                "claimed task has no persisted worktree"
            )
        worktree = Path(runtime.worktree_path).resolve()
        preserving_active_phase = runtime.status in {
            TaskRuntimeStatus.CODING,
            TaskRuntimeStatus.MERGING,
        }
        reclaimed_agent_work = (
            runtime.status is TaskRuntimeStatus.CLAIMED
            and claim.resume_phase != TaskRuntimeStatus.ENVIRONMENT.value
        )
        if runtime.status is TaskRuntimeStatus.CLAIMED:
            runtime = self._transition_claimed_task(
                store,
                claim,
                owner_token,
                expected_status=TaskRuntimeStatus.CLAIMED,
                new_status=TaskRuntimeStatus.ENVIRONMENT,
                resume_phase=claim.resume_phase,
                task_transition=task_transition,
            )
        elif runtime.status not in {
            TaskRuntimeStatus.ENVIRONMENT,
            TaskRuntimeStatus.CODING,
            TaskRuntimeStatus.MERGING,
        }:
            raise EnvironmentMaterializationError(
                "task must be claimed, resuming its environment phase, or "
                "rematerializing before sanity"
            )

        try:
            self._assert_task_worktree(worktree, runtime.branch)
            self._guard.assert_clean("task environment materialization")
            if not preserving_active_phase and not reclaimed_agent_work:
                self._assert_no_tracked_changes(
                    worktree, "before environment materialization"
                )
            descriptors = _environment_descriptors(plan, worktree)
            fingerprint = _fingerprint_descriptors(plan, descriptors)
            cache_path = self._cache_path(fingerprint)
            cache_path.mkdir(parents=True, exist_ok=True)
            base_environment = self._base_command_environment(plan, cache_path)
            command_environments = self._command_environments(
                plan, base_environment, secret_values or {}
            )
            preparation_reused = self._ensure_prepared(
                store,
                plan,
                claim,
                owner_token,
                fingerprint=fingerprint,
                cache_path=cache_path,
                source_worktree=worktree,
                descriptors=descriptors,
                command_environments=command_environments,
                activity=activity,
            )
            materialization_reused = self._materialize_worktree(
                store,
                plan,
                claim,
                owner_token,
                fingerprint=fingerprint,
                worktree=worktree,
                cache_path=cache_path,
                command_environments=command_environments,
                activity=activity,
            )
        except BaseException as error:
            self._raise_if_cancelled(error)
            self._block_environment_task(
                store,
                claim,
                owner_token,
                error,
                task_transition=task_transition,
            )
            raise

        if not preserving_active_phase:
            self._transition_claimed_task(
                store,
                claim,
                owner_token,
                expected_status=TaskRuntimeStatus.ENVIRONMENT,
                new_status=TaskRuntimeStatus.CODING,
                resume_phase=(claim.resume_phase if reclaimed_agent_work else None),
                task_transition=task_transition,
            )
        return EnvironmentMaterialization(
            fingerprint=fingerprint,
            cache_path=cache_path,
            preparation_reused=preparation_reused,
            materialization_reused=materialization_reused,
            environment=MappingProxyType(dict(base_environment)),
        )

    def _ensure_prepared(
        self,
        store: SqliteStore,
        plan: HostPreflightPlan,
        claim: TaskClaim,
        owner_token: str,
        *,
        fingerprint: str,
        cache_path: Path,
        source_worktree: Path,
        descriptors: Sequence[_EnvironmentDescriptor],
        command_environments: Mapping[
            str, tuple[Mapping[str, str], Sequence[str]]
        ],
        activity: ActivitySink | None,
    ) -> bool:
        if not plan.prepare_commands:
            return False
        with _preparation_lock(cache_path):
            marker = cache_path / ".betterborg-prepared"
            completed = store.find_completed_environment_attempt(
                fingerprint, kind="prepare"
            )
            if completed is not None and _prepared_marker_matches(
                marker, fingerprint
            ):
                return True
            self._record_attempt(
                store,
                claim,
                owner_token,
                kind="prepare",
                fingerprint=fingerprint,
                commands=plan.prepare_commands,
                worktree=None,
                preparation_source=source_worktree,
                preparation_descriptors=descriptors,
                cache_path=cache_path,
                completion_marker=marker,
                command_environments=command_environments,
                activity=activity,
            )
            return False

    def _materialize_worktree(
        self,
        store: SqliteStore,
        plan: HostPreflightPlan,
        claim: TaskClaim,
        owner_token: str,
        *,
        fingerprint: str,
        worktree: Path,
        cache_path: Path,
        command_environments: Mapping[
            str, tuple[Mapping[str, str], Sequence[str]]
        ],
        activity: ActivitySink | None,
    ) -> bool:
        marker = self._materialization_marker(worktree)
        if (
            store.find_completed_environment_attempt(
                fingerprint, kind="materialize", task_id=claim.task_id
            )
            is not None
            and _prepared_marker_matches(marker, fingerprint)
        ):
            return True

        # A failed or intervening materialization may already have changed
        # ignored checkout-local dependencies.  Invalidate the prior state
        # before running so a later A -> B -> A transition cannot reuse A.
        _invalidate_marker(marker)

        # Analyzer materialization is preferred.  Preparation is the safe
        # per-checkout fallback because outputs from the disposable preparation
        # worktree are deliberately never copied into a task checkout.
        commands = plan.materialize_commands or plan.prepare_commands
        self._record_attempt(
            store,
            claim,
            owner_token,
            kind="materialize",
            fingerprint=fingerprint,
            commands=commands,
            worktree=worktree,
            cache_path=cache_path,
            completion_marker=marker,
            command_environments=command_environments,
            activity=activity,
        )
        return False

    def _record_attempt(
        self,
        store: SqliteStore,
        claim: TaskClaim | None,
        owner_token: str,
        *,
        run_id: UUID | None = None,
        task_id: UUID | None = None,
        kind: str,
        fingerprint: str,
        commands: Sequence[HostCommand],
        worktree: Path | None,
        cache_path: Path,
        command_environments: Mapping[
            str, tuple[Mapping[str, str], Sequence[str]]
        ],
        preparation_source: Path | None = None,
        preparation_descriptors: Sequence[_EnvironmentDescriptor] = (),
        completion_marker: Path | None = None,
        prepared_before_dispatch: bool = False,
        activity: ActivitySink | None = None,
    ) -> None:
        if claim is not None:
            run_id = claim.run_id
            task_id = claim.task_id
        if run_id is None or task_id is None:
            raise AssertionError("environment attempt requires run and task identity")
        claim_token = claim.claim_token if claim is not None else None
        prior = [
            attempt
            for attempt in store.list_environment_attempts(task_id)
            if attempt.kind == kind
        ]
        mask_values = tuple(
            sorted(
                {
                    value
                    for command in commands
                    for value in command_environments[command.stage][1]
                },
                key=len,
                reverse=True,
            )
        )
        started_at = self._clock()
        attempt = EnvironmentAttempt(
            run_id=run_id,
            claim_id=claim.id if claim is not None else None,
            task_id=task_id,
            kind=kind,
            attempt_number=len(prior) + 1,
            fingerprint=fingerprint,
            status=ExecutionAttemptStatus.RUNNING,
            commands=[
                [redact_secrets(argument, mask_values) for argument in command.argv]
                for command in commands
            ],
            started_at=started_at,
            finished_at=None,
        )
        store.append_environment_attempt(
            attempt,
            owner_token,
            claim_token,
            now=started_at,
        )

        started = time.monotonic()
        try:
            if preparation_source is None:
                if worktree is None:
                    raise AssertionError("materialization worktree is required")
                with self._guard.protect(str(task_id), f"environment {kind}"):
                    results = self._run_commands(
                        commands,
                        worktree=worktree,
                        command_environments=command_environments,
                        activity=activity,
                    )
                    if completion_marker is not None:
                        _write_marker(completion_marker, fingerprint)
            else:
                if worktree is not None:
                    raise AssertionError(
                        "preparation cannot use a caller-provided worktree"
                    )
                results = self._run_preparation_commands(
                    commands,
                    fingerprint=fingerprint,
                    source_worktree=preparation_source,
                    descriptors=preparation_descriptors,
                    command_environments=command_environments,
                    completion_marker=completion_marker,
                    activity=activity,
                )
        except BaseException as error:
            duration = time.monotonic() - started
            redacted = redact_secrets(str(error), mask_values)
            failure_result: dict[str, object] = {
                "cache_path": str(cache_path),
            }
            if prepared_before_dispatch:
                failure_result["prepared_before_dispatch"] = True
            store.complete_environment_attempt(
                attempt.id,
                owner_token,
                claim_token,
                status=ExecutionAttemptStatus.FAILED,
                result=failure_result,
                error=redacted,
                duration_seconds=duration,
                now=self._clock(),
            )
            self._raise_if_cancelled(error)
            if isinstance(error, EnvironmentMaterializationError):
                raise EnvironmentMaterializationError(redacted) from error
            raise EnvironmentMaterializationError(redacted) from error

        completed_result: dict[str, object] = {
            "cache_path": str(cache_path),
            "commands": results,
        }
        if prepared_before_dispatch:
            completed_result["prepared_before_dispatch"] = True
        store.complete_environment_attempt(
            attempt.id,
            owner_token,
            claim_token,
            status=ExecutionAttemptStatus.COMPLETED,
            result=completed_result,
            duration_seconds=time.monotonic() - started,
            now=self._clock(),
        )

    def _run_preparation_commands(
        self,
        commands: Sequence[HostCommand],
        *,
        fingerprint: str,
        source_worktree: Path,
        descriptors: Sequence[_EnvironmentDescriptor],
        command_environments: Mapping[
            str, tuple[Mapping[str, str], Sequence[str]]
        ],
        completion_marker: Path | None,
        activity: ActivitySink | None,
    ) -> list[dict[str, object]]:
        with self._guard.protect(fingerprint, "environment prepare"):
            with self._preparation_worktree(
                fingerprint, source_worktree, descriptors
            ) as preparation_worktree:
                results = self._run_commands(
                    commands,
                    worktree=preparation_worktree,
                    command_environments=command_environments,
                    activity=activity,
                )
            if completion_marker is not None:
                _write_marker(completion_marker, fingerprint)
            return results

    def _run_commands(
        self,
        commands: Sequence[HostCommand],
        *,
        worktree: Path,
        command_environments: Mapping[
            str, tuple[Mapping[str, str], Sequence[str]]
        ],
        activity: ActivitySink | None = None,
    ) -> list[dict[str, object]]:
        before = self._tracked_state(worktree)
        results: list[dict[str, object]] = []
        for command in commands:
            cwd = command_cwd(worktree, command.cwd)
            environment, mask_values = command_environments[command.stage]
            self._report_command(command.argv, mask_values, activity=activity)
            try:
                completed = self._run(
                    list(command.argv),
                    cwd=cwd,
                    env=dict(environment),
                    check=False,
                    cancel=self._cancel,
                )
            except (OSError, subprocess.SubprocessError) as error:
                self._raise_if_cancelled(error)
                raise EnvironmentMaterializationError(
                    f"unable to run {command.argv[0]!r}: {error}"
                ) from error
            self._raise_if_cancelled()
            stdout = redact_secrets(completed.stdout or "", mask_values)
            stderr = redact_secrets(completed.stderr or "", mask_values)
            results.append(
                {
                    "argv": [
                        redact_secrets(argument, mask_values)
                        for argument in command.argv
                    ],
                    "cwd": command.cwd,
                    "returncode": completed.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                }
            )
            if completed.returncode != 0:
                detail = stderr.strip() or stdout.strip() or "no command output"
                raise EnvironmentMaterializationError(
                    f"environment command {command.argv!r} failed with exit code "
                    f"{completed.returncode}: {detail}"
                )
        after = self._tracked_state(worktree)
        if after != before:
            details = after[1].replace("\0", "\n").strip() or "HEAD changed"
            raise EnvironmentMaterializationError(
                "worktree has unexpected tracked changes after environment "
                f"command: {details}"
            )
        return results

    def _report_command(
        self,
        command: Sequence[str],
        mask_values: Sequence[str],
        *,
        activity: ActivitySink | None = None,
    ) -> None:
        """Publish one redacted environment command without affecting setup."""
        sink = activity if activity is not None else self._activity
        if sink is None:
            return
        redacted = [redact_secrets(argument, mask_values) for argument in command]
        try:
            sink(
                AgentActivity(AgentActivityKind.COMMAND, shlex.join(redacted))
            )
        except Exception:
            return

    def _raise_if_cancelled(self, cause: BaseException | None = None) -> None:
        """Keep cancellation distinct from an environment setup failure."""
        if self._cancel is None or not self._cancel.is_set():
            return
        if cause is None:
            raise KeyboardInterrupt
        raise KeyboardInterrupt from cause

    def _base_command_environment(
        self,
        plan: HostPreflightPlan,
        cache_path: Path,
    ) -> dict[str, str]:
        base_environment = {
            name: self._environment[name]
            for name in _SAFE_HOST_ENVIRONMENT
            if name in self._environment
        }
        base_environment.update(
            package_manager_cache_environment(cache_path, plan.package_managers)
        )
        home = cache_path / "home"
        home.mkdir(parents=True, exist_ok=True)
        base_environment.update(
            {
                "GIT_TERMINAL_PROMPT": "0",
                "HOME": str(home),
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            }
        )
        return base_environment

    def _command_environments(
        self,
        plan: HostPreflightPlan,
        base_environment: Mapping[str, str],
        secret_values: Mapping[str, str],
    ) -> dict[str, tuple[dict[str, str], tuple[str, ...]]]:
        stages = {
            command.stage
            for command in (*plan.prepare_commands, *plan.materialize_commands)
        }
        mask_values = declared_secret_mask_values(plan, secret_values)
        environments: dict[str, tuple[dict[str, str], tuple[str, ...]]] = {}
        for stage in stages:
            environment = dict(base_environment)
            secrets, _ = command_secret_environment(
                plan, stage, secret_values
            )
            environment.update(secrets)
            environments[stage] = (
                environment,
                mask_values,
            )
        return environments

    @contextmanager
    def _preparation_worktree(
        self,
        fingerprint: str,
        source_worktree: Path,
        descriptors: Sequence[_EnvironmentDescriptor],
    ) -> Iterator[Path]:
        identity = fingerprint.removeprefix("sha256:")
        path = self.preparation_root / identity
        self.preparation_root.mkdir(parents=True, exist_ok=True)
        self._remove_stale_preparation_worktree(path)
        source_sha = self._git.for_worktree(source_worktree).head_sha()
        try:
            self._git.run(
                ["worktree", "add", "--detach", str(path), source_sha]
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise EnvironmentMaterializationError(
                f"unable to create preparation worktree {path}: {error}"
            ) from error

        active_error: BaseException | None = None
        try:
            with _descriptor_overlay(path, descriptors):
                yield path
        except BaseException as error:
            active_error = error
            raise
        finally:
            try:
                self._git.remove_worktree(path)
            except (OSError, subprocess.SubprocessError) as error:
                cleanup = EnvironmentMaterializationError(
                    "unable to clean preparation worktree without discarding "
                    f"changes: {path}: {error}"
                )
                if active_error is None:
                    raise cleanup from error
                active_error.add_note(str(cleanup))

    def _remove_stale_preparation_worktree(self, path: Path) -> None:
        entries = self._git.worktree_list()
        registered = any(
            Path(entry.get("path", "")).resolve() == path for entry in entries
        )
        if registered:
            try:
                self._git.remove_worktree(path)
            except (OSError, subprocess.SubprocessError) as error:
                raise EnvironmentMaterializationError(
                    "stale preparation worktree contains changes; preserving "
                    f"{path} and blocking execution"
                ) from error
        if path.exists():
            raise EnvironmentMaterializationError(
                f"preparation path exists but is not a managed worktree: {path}"
            )

    def _assert_no_tracked_changes(self, worktree: Path, when: str) -> None:
        output = self._git.for_worktree(worktree).run(
            ["status", "--porcelain=v1", "-z", "-uno"]
        ).stdout
        if output:
            details = output.replace("\0", "\n").strip()
            raise EnvironmentMaterializationError(
                f"worktree has unexpected tracked changes {when}: "
                f"{details}"
            )

    def _tracked_state(self, worktree: Path) -> tuple[str, str, str]:
        git = self._git.for_worktree(worktree)
        status = git.run(["status", "--porcelain=v1", "-z", "-uno"]).stdout
        diff = git.run(["diff", "--binary", "HEAD", "--"]).stdout
        return git.head_sha(), status, diff

    def _assert_task_worktree(self, worktree: Path, branch: str | None) -> None:
        expected_branch = f"refs/heads/{branch}" if branch is not None else None
        if not any(
            Path(entry.get("path", "")).resolve() == worktree
            and entry.get("branch") == expected_branch
            for entry in self._git.worktree_list()
        ):
            raise EnvironmentMaterializationError(
                "claimed task path is not its registered BetterBorg worktree: "
                f"{worktree}"
            )

    def _cache_path(self, fingerprint: str) -> Path:
        return self.cache_root / fingerprint.removeprefix("sha256:")

    def _materialization_marker(self, worktree: Path) -> Path:
        marker = worktree / ".borg/state/environment-materialization"
        parent = marker.parent.resolve()
        if not parent.is_relative_to(worktree.resolve()):
            raise EnvironmentMaterializationError(
                "checkout-local environment marker escapes task worktree"
            )
        if not self._git.for_worktree(worktree).is_ignored(marker):
            raise EnvironmentMaterializationError(
                "checkout-local environment marker is not ignored by Git"
            )
        return marker

    def _validate_managed_paths(self) -> None:
        if self.preparation_root == self.repository_root or (
            self.preparation_root.is_relative_to(self.repository_root)
        ):
            raise EnvironmentMaterializationError(
                "preparation worktrees must be outside the repository checkout"
            )
        if self.cache_root.is_relative_to(self.repository_root) and not (
            self.cache_root == self._paths.state_dir
            or self.cache_root.is_relative_to(self._paths.state_dir)
        ):
            raise EnvironmentMaterializationError(
                "repository-local caches must be under ignored .borg/state"
            )
        if self.cache_root.is_relative_to(
            self.repository_root
        ) and not self._git.is_ignored(self.cache_root):
            raise EnvironmentMaterializationError(
                "repository-local environment cache is not ignored by Git"
            )

    def _block_environment_task(
        self,
        store: SqliteStore,
        claim: TaskClaim,
        owner_token: str,
        error: BaseException,
        *,
        task_transition: Callable[..., TaskRuntime] | None = None,
    ) -> None:
        runtime = store.get_task_runtime(claim.task_id)
        if runtime is None or runtime.status not in {
            TaskRuntimeStatus.ENVIRONMENT,
            TaskRuntimeStatus.CODING,
            TaskRuntimeStatus.MERGING,
        }:
            return
        reason = str(error) or error.__class__.__name__
        try:
            self._transition_claimed_task(
                store,
                claim,
                owner_token,
                expected_status=runtime.status,
                new_status=TaskRuntimeStatus.BLOCKED,
                state_reason=reason,
                task_transition=task_transition,
            )
        except BaseException as transition_error:
            error.add_note(
                f"task could not be durably blocked: {transition_error}"
            )

    def _transition_claimed_task(
        self,
        store: SqliteStore,
        claim: TaskClaim,
        owner_token: str,
        *,
        expected_status: TaskRuntimeStatus,
        new_status: TaskRuntimeStatus,
        task_transition: Callable[..., TaskRuntime] | None,
        **changes: object,
    ) -> TaskRuntime:
        """Use the scheduler-owned transition seam when one is available."""
        if task_transition is not None:
            return task_transition(expected_status, new_status, **changes)
        return store.transition_task_runtime(
            claim.run_id,
            owner_token,
            claim.id,
            claim.claim_token,
            expected_status=expected_status,
            new_status=new_status,
            now=self._clock(),
            **changes,
        )


def _command_payload(commands: Sequence[HostCommand]) -> list[dict[str, object]]:
    return [
        {"argv": list(command.argv), "cwd": command.cwd, "stage": command.stage}
        for command in commands
    ]


def command_cwd(worktree: Path, value: str) -> Path:
    """Resolve one declared command cwd without allowing checkout escape."""
    portable = PurePosixPath(value)
    if portable.is_absolute() or ".." in portable.parts or "\\" in value:
        raise EnvironmentMaterializationError(
            f"command cwd is not repository-relative: {value!r}"
        )
    candidate = (worktree / portable).resolve()
    if not candidate.is_relative_to(worktree) or not candidate.is_dir():
        raise EnvironmentMaterializationError(
            f"command cwd is missing from worktree: {value!r}"
        )
    return candidate


def command_secret_environment(
    plan: HostPreflightPlan,
    stage: str,
    secret_values: Mapping[str, str],
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Return only build secrets declared for one command stage."""
    environment: dict[str, str] = {}
    mask_values: list[str] = []
    for secret in plan.secret_requirements:
        if secret.scope not in {"all", "build"} or stage not in secret.used_by:
            continue
        value = secret_values.get(secret.name)
        if value is None:
            raise EnvironmentMaterializationError(
                f"build-scoped secret value is unavailable: {secret.name}"
            )
        environment[secret.name] = value
        if value:
            mask_values.append(value)
    return environment, tuple(sorted(set(mask_values), key=len, reverse=True))


def declared_secret_mask_values(
    plan: HostPreflightPlan, secret_values: Mapping[str, str]
) -> tuple[str, ...]:
    """Return supplied values for secrets declared by the validated plan."""
    declared = set(plan.required_secret_names)
    return tuple(
        sorted(
            {
                value
                for name, value in secret_values.items()
                if name in declared and value
            },
            key=len,
            reverse=True,
        )
    )


def redact_secrets(value: str, mask_values: Sequence[str]) -> str:
    """Redact raw, JSON-escaped, and URL-encoded forms of secret values."""
    variants: set[str] = set()
    for secret in mask_values:
        if not secret:
            continue
        variants.update(
            {
                secret,
                json.dumps(secret)[1:-1],
                quote(secret, safe=""),
            }
        )
    redacted = value
    for secret in sorted(variants, key=len, reverse=True):
        redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _prepared_marker_matches(marker: Path, fingerprint: str) -> bool:
    try:
        return marker.read_text(encoding="utf-8").strip() == fingerprint
    except OSError:
        return False


def _write_marker(marker: Path, fingerprint: str) -> None:
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{fingerprint}\n", encoding="utf-8")
    except OSError as error:
        raise EnvironmentMaterializationError(
            f"unable to record environment state marker {marker}: {error}"
        ) from error


def _invalidate_marker(marker: Path) -> None:
    try:
        marker.unlink(missing_ok=True)
    except OSError as error:
        raise EnvironmentMaterializationError(
            f"unable to invalidate environment state marker {marker}: {error}"
        ) from error


@contextmanager
def _preparation_lock(cache_path: Path) -> Iterator[None]:
    with path_lock(cache_path / ".betterborg-preparation.lock"):
        yield


@contextmanager
def _descriptor_overlay(
    worktree: Path, descriptors: Sequence[_EnvironmentDescriptor]
) -> Iterator[None]:
    originals: list[tuple[Path, _PathSnapshot]] = []
    try:
        for descriptor in descriptors:
            destination = worktree / descriptor.relative_path
            parent = destination.parent.resolve()
            if not parent.is_relative_to(worktree):
                raise EnvironmentMaterializationError(
                    "environment descriptor escapes preparation worktree: "
                    f"{descriptor.relative_path}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            original = _snapshot_path(destination)
            originals.append((destination, original))
            _replace_with_regular_file(destination, descriptor.content)
        yield
    finally:
        for destination, original in reversed(originals):
            _restore_path(destination, original)


def _snapshot_path(path: Path) -> _PathSnapshot:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return _PathSnapshot(kind="missing")
    except OSError as error:
        raise EnvironmentMaterializationError(
            f"unable to inspect preparation descriptor {path}: {error}"
        ) from error
    if stat.S_ISLNK(metadata.st_mode):
        return _PathSnapshot(kind="symlink", link_target=os.readlink(path))
    if stat.S_ISREG(metadata.st_mode):
        return _PathSnapshot(
            kind="file",
            content=path.read_bytes(),
            mode=stat.S_IMODE(metadata.st_mode),
        )
    raise EnvironmentMaterializationError(
        f"preparation descriptor is not a file or symlink: {path}"
    )


def _replace_with_regular_file(path: Path, content: bytes) -> None:
    try:
        path.unlink(missing_ok=True)
        path.write_bytes(content)
    except OSError as error:
        raise EnvironmentMaterializationError(
            f"unable to materialize preparation descriptor {path}: {error}"
        ) from error


def _restore_path(path: Path, snapshot: _PathSnapshot) -> None:
    try:
        path.unlink(missing_ok=True)
        if snapshot.kind == "file":
            if snapshot.content is None or snapshot.mode is None:
                raise AssertionError("regular-file snapshot is incomplete")
            path.write_bytes(snapshot.content)
            path.chmod(snapshot.mode)
        elif snapshot.kind == "symlink":
            if snapshot.link_target is None:
                raise AssertionError("symlink snapshot is incomplete")
            path.symlink_to(snapshot.link_target)
    except OSError as error:
        raise EnvironmentMaterializationError(
            f"unable to restore preparation descriptor {path}: {error}"
        ) from error
