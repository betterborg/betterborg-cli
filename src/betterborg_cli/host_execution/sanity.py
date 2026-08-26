"""Locked post-merge sanity, project-base advancement, and cleanup."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from betterborg_cli.host_execution.compose import (
    ComposeStack,
    ComposeStackError,
    HostComposeManager,
    service_url_environment,
)
from betterborg_cli.host_execution.environment import (
    EnvironmentMaterializationError,
    HostEnvironmentManager,
)
from betterborg_cli.host_execution.git import SafeGit, UnsafeGitError
from betterborg_cli.host_execution.guard import (
    PrimaryCheckoutContaminationError,
    PrimaryCheckoutGuard,
)
from betterborg_cli.host_execution.merge import MergeTip
from betterborg_cli.host_execution.preflight import HostCommand, HostPreflightPlan
from betterborg_cli.host_execution.scheduler import ScheduledTaskContext
from betterborg_cli.host_execution.worktrees import HostWorktreeManager, WorktreeError
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.store import ExecutionEvent, TaskRuntime, TaskRuntimeStatus

RepositoryLockFactory = Callable[[], AbstractContextManager[None]]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]

_SAFE_HOST_ENVIRONMENT = (
    "LANG",
    "LC_ALL",
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "TMPDIR",
)
_SANITY_COMPLETED_EVENT = "sanity.completed"
_BASE_ADVANCE_STARTED_EVENT = "base.advance_started"
_BASE_ADVANCED_EVENT = "base.advanced"


class SanityPhaseError(RuntimeError):
    """Raised when a merged task cannot safely advance its project base."""


@dataclass(frozen=True, slots=True)
class SanityCommandResult:
    """Redacted result of one catalog command in declared order."""

    command: HostCommand
    returncode: int
    stdout: str = field(default="", repr=False)
    stderr: str = field(default="", repr=False)


@dataclass(frozen=True, slots=True)
class HostSanityResult:
    """Terminal outcome of the post-merge publication gate."""

    status: TaskRuntimeStatus
    reason: str
    commit_sha: str | None = None
    commands: tuple[SanityCommandResult, ...] = ()

    def __post_init__(self) -> None:
        if self.status is TaskRuntimeStatus.DONE:
            if self.commit_sha is None:
                raise ValueError("successful sanity requires a published commit")
        elif self.status is not TaskRuntimeStatus.BLOCKED:
            raise ValueError("sanity result must be done or blocked")
        elif self.commit_sha is not None:
            raise ValueError("blocked sanity cannot expose a published commit")


class HostSanityPhase:
    """Run the final host-only gate and publish one exact merge tip.

    The shared repository lock covers descriptor rematerialization, task-owned
    service startup and teardown, catalog execution, and the compare-and-swap
    fast-forward. This makes a successful sanity result inseparable from the
    project-base decision it authorizes.
    """

    def __init__(
        self,
        repository_root: Path,
        plan: HostPreflightPlan,
        *,
        environment_manager: HostEnvironmentManager,
        compose_manager: HostComposeManager,
        worktree_manager: HostWorktreeManager,
        repository_lock: RepositoryLockFactory,
        environment: Mapping[str, str] | None = None,
        command_runner: CommandRunner | None = None,
        timeout_seconds: float = 600,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        paths = RepoPaths.discover(self.repository_root)
        if paths.root != self.repository_root:
            raise SanityPhaseError("sanity phase must use the primary checkout")
        if plan.repository_root.resolve() != self.repository_root:
            raise SanityPhaseError("preflight plan belongs to another repository")
        if not callable(repository_lock):
            raise TypeError("repository lock must be a context manager factory")
        if timeout_seconds <= 0:
            raise ValueError("sanity command timeout must be positive")
        self.plan = plan
        self._environment_manager = environment_manager
        self._compose_manager = compose_manager
        self._worktree_manager = worktree_manager
        self._repository_lock = repository_lock
        source_environment = os.environ if environment is None else environment
        self._environment = {
            name: source_environment[name]
            for name in _SAFE_HOST_ENVIRONMENT
            if name in source_environment
        }
        self._run = command_runner or subprocess.run
        self._timeout_seconds = timeout_seconds
        self._git = SafeGit(self.repository_root)
        self._guard = PrimaryCheckoutGuard(self.repository_root)

    def run(
        self,
        context: ScheduledTaskContext,
        tip: MergeTip,
        *,
        secret_values: Mapping[str, str] | None = None,
    ) -> HostSanityResult:
        """Sanity-check and publish ``tip``, or durably block the task."""
        commands: tuple[SanityCommandResult, ...] = ()
        try:
            runtime, worktree = self._runtime_and_worktree(context, tip)
            with self._guard.protect(self._task_ref(context), "sanity"):
                with self._repository_lock():
                    published, commands = self._run_locked(
                        context,
                        runtime,
                        worktree,
                        tip,
                        secret_values or {},
                    )
        except (
            ComposeStackError,
            EnvironmentMaterializationError,
            PrimaryCheckoutContaminationError,
            SanityPhaseError,
            UnsafeGitError,
            WorktreeError,
            OSError,
            subprocess.SubprocessError,
        ) as error:
            reason = _redact(
                _error_text(error),
                tuple((secret_values or {}).values()),
            )
            return self._block(context, reason, commands)

        context.transition(
            TaskRuntimeStatus.MERGING,
            TaskRuntimeStatus.DONE,
            resume_phase="done",
            state_reason=f"advanced {tip.project_branch} to {published}",
        )
        completed = context.store.get_task_runtime(context.claim.task_id)
        if completed is None:
            raise SanityPhaseError("completed task runtime disappeared")
        self._worktree_manager.cleanup_task_worktree(completed)
        return HostSanityResult(
            TaskRuntimeStatus.DONE,
            f"sanity passed and advanced {tip.project_branch} to {published}",
            published,
            commands,
        )

    def _run_locked(
        self,
        context: ScheduledTaskContext,
        runtime: TaskRuntime,
        worktree: Path,
        tip: MergeTip,
        secret_values: Mapping[str, str],
    ) -> tuple[str, tuple[SanityCommandResult, ...]]:
        current_base = self._resolve_project_tip(tip.project_branch)
        self._verify_tip(runtime, worktree, tip)
        if current_base == tip.commit_sha:
            if not self._advance_was_attested(context, tip):
                raise SanityPhaseError(
                    "project base is already at the merge tip without a durable "
                    "BetterBorg advancement attestation"
                )
            return tip.commit_sha, ()
        if current_base != tip.base_commit:
            raise SanityPhaseError(
                "project base moved after the merge tip was produced; rerun the "
                "merge phase before sanity"
            )

        materialization = self._environment_manager.materialize_claimed_task(
            context.store,
            self.plan,
            context.claim,
            context.owner_token,
            secret_values=secret_values,
        )
        stack: ComposeStack | None = None
        commands: tuple[SanityCommandResult, ...] = ()
        active_error: BaseException | None = None
        try:
            stack = self._compose_manager.start_claimed_stack(
                context.store,
                self.plan,
                context.claim,
                context.owner_token,
            )
            service_environment = service_url_environment(self.plan.services)
            if stack is not None:
                service_environment.update(stack.environment)
            commands = self._run_commands(
                worktree,
                service_environment=service_environment,
                secret_values=secret_values,
            )
            failure = next(
                (result for result in commands if result.returncode != 0), None
            )
            if failure is not None:
                detail = failure.stderr.strip() or failure.stdout.strip()
                raise SanityPhaseError(
                    "sanity command failed with exit code "
                    f"{failure.returncode}: {shlex.join(failure.command.argv)}"
                    + (f": {detail[-4000:]}" if detail else "")
                )
            if not commands:
                raise SanityPhaseError("sanity command catalog is empty")
            if not SafeGit(worktree).is_clean():
                raise SanityPhaseError(
                    "sanity commands changed tracked or untracked task files"
                )
        except BaseException as error:
            active_error = error
        finally:
            if stack is not None:
                try:
                    self._compose_manager.stop_claimed_stack(
                        context.store,
                        stack,
                        context.claim,
                        context.owner_token,
                    )
                except BaseException as cleanup_error:
                    if active_error is None:
                        active_error = cleanup_error
                    else:
                        active_error.add_note(
                            f"task-owned Compose cleanup also failed: {cleanup_error}"
                        )
        if active_error is not None:
            raise active_error

        masks = self._mask_values(secret_values)
        self._append_event(
            context,
            _SANITY_COMPLETED_EVENT,
            {
                "project_branch": tip.project_branch,
                "base_commit": tip.base_commit,
                "commit_sha": tip.commit_sha,
                "environment_fingerprint": materialization.fingerprint,
                "commands": [
                    {
                        "argv": [_redact(arg, masks) for arg in result.command.argv],
                        "cwd": result.command.cwd,
                        "returncode": result.returncode,
                        "stdout": result.stdout[-4000:],
                        "stderr": result.stderr[-4000:],
                    }
                    for result in commands
                ],
            },
        )
        self._append_event(
            context,
            _BASE_ADVANCE_STARTED_EVENT,
            {
                "project_branch": tip.project_branch,
                "from_commit": tip.base_commit,
                "commit_sha": tip.commit_sha,
            },
        )
        if not self._git.fast_forward_branch(tip.project_branch, tip.commit_sha):
            raise SanityPhaseError(
                f"could not fast-forward {tip.project_branch!r} to the sanity "
                "checked merge tip"
            )
        self._append_event(
            context,
            _BASE_ADVANCED_EVENT,
            {
                "project_branch": tip.project_branch,
                "from_commit": tip.base_commit,
                "commit_sha": tip.commit_sha,
            },
        )
        return tip.commit_sha, commands

    def _run_commands(
        self,
        worktree: Path,
        *,
        service_environment: Mapping[str, str],
        secret_values: Mapping[str, str],
    ) -> tuple[SanityCommandResult, ...]:
        results: list[SanityCommandResult] = []
        masks = self._mask_values(secret_values)
        for command in self.plan.commands:
            cwd = _command_cwd(worktree, command.cwd)
            environment = dict(self._environment)
            environment["CI"] = "true"
            environment.update(service_environment)
            environment.update(
                self._command_secrets(command, secret_values=secret_values)
            )
            try:
                completed = self._run(
                    list(command.argv),
                    cwd=cwd,
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_seconds,
                )
                result = SanityCommandResult(
                    command,
                    completed.returncode,
                    _redact(completed.stdout or "", masks),
                    _redact(completed.stderr or "", masks),
                )
            except subprocess.TimeoutExpired as error:
                result = SanityCommandResult(
                    command,
                    -1,
                    _redact(_output(error.stdout), masks),
                    _redact(
                        _output(error.stderr)
                        + f"sanity command timed out after {self._timeout_seconds}s",
                        masks,
                    ),
                )
            except OSError as error:
                result = SanityCommandResult(
                    command,
                    -1,
                    "",
                    _redact(f"sanity command could not run: {error}", masks),
                )
            results.append(result)
            if result.returncode != 0:
                break
        return tuple(results)

    def _command_secrets(
        self,
        command: HostCommand,
        *,
        secret_values: Mapping[str, str],
    ) -> dict[str, str]:
        environment: dict[str, str] = {}
        for secret in self.plan.secret_requirements:
            if (
                secret.scope not in {"all", "build"}
                or command.stage not in secret.used_by
            ):
                continue
            value = secret_values.get(secret.name)
            if value is None:
                raise SanityPhaseError(
                    f"build-scoped secret value is unavailable: {secret.name}"
                )
            environment[secret.name] = value
        return environment

    def _mask_values(self, secret_values: Mapping[str, str]) -> tuple[str, ...]:
        declared = set(self.plan.required_secret_names)
        return tuple(
            value for name, value in secret_values.items() if name in declared and value
        )

    def _runtime_and_worktree(
        self, context: ScheduledTaskContext, tip: MergeTip
    ) -> tuple[TaskRuntime, Path]:
        runtime = context.store.get_task_runtime(context.claim.task_id)
        if runtime is None or runtime.status is not TaskRuntimeStatus.MERGING:
            raise SanityPhaseError("task must be in its merging phase for sanity")
        if runtime.branch != tip.task_branch or runtime.worktree_path is None:
            raise SanityPhaseError("merge tip does not match the claimed worktree")
        worktree = Path(runtime.worktree_path).resolve()
        if not worktree.is_dir():
            raise SanityPhaseError("merged task worktree is missing")
        expected_project = self._project_branch(context)
        if tip.project_branch != expected_project:
            raise SanityPhaseError("merge tip belongs to another project base")
        return runtime, worktree

    @staticmethod
    def _verify_tip(runtime: TaskRuntime, worktree: Path, tip: MergeTip) -> None:
        git = SafeGit(worktree)
        if git.current_branch() != runtime.branch:
            raise SanityPhaseError("merged worktree is on the wrong branch")
        if git.head_sha() != tip.commit_sha:
            raise SanityPhaseError("merged worktree no longer matches its merge tip")
        if not git.is_clean():
            raise SanityPhaseError("merged worktree has uncommitted changes")
        if not git.is_ancestor(tip.approved_commit, tip.commit_sha):
            raise SanityPhaseError("merge tip does not contain its approved commit")
        if not git.is_ancestor(tip.base_commit, tip.commit_sha):
            raise SanityPhaseError("merge tip does not contain its resolved base")

    def _resolve_project_tip(self, project_branch: str) -> str:
        result = self._git.run(
            ["rev-parse", "--verify", f"refs/heads/{project_branch}^{{commit}}"],
            check=False,
        )
        value = result.stdout.strip()
        if result.returncode != 0 or not value:
            raise SanityPhaseError(f"project base does not resolve: {project_branch!r}")
        return value

    def _advance_was_attested(
        self, context: ScheduledTaskContext, tip: MergeTip
    ) -> bool:
        return any(
            event.kind == _BASE_ADVANCE_STARTED_EVENT
            and event.payload.get("project_branch") == tip.project_branch
            and event.payload.get("from_commit") == tip.base_commit
            and event.payload.get("commit_sha") == tip.commit_sha
            for event in context.store.list_task_execution_events(context.claim.task_id)
        )

    def _project_branch(self, context: ScheduledTaskContext) -> str:
        generation = context.store.get_task_generation(context.runtime.generation_id)
        if generation is None:
            raise SanityPhaseError("task generation is missing")
        borg = context.store.get_borg(generation.borg_id)
        if borg is None or borg.id != generation.borg_id:
            raise SanityPhaseError("task Borg is missing or mismatched")
        branch = f"project/{borg.name}"
        if not self._git.is_valid_branch_name(branch):
            raise SanityPhaseError(f"invalid project branch: {branch!r}")
        return branch

    def _task_ref(self, context: ScheduledTaskContext) -> str:
        runtime = context.store.get_task_runtime(context.claim.task_id)
        if runtime is not None:
            task = next(
                (
                    item
                    for item in context.store.list_task_records(runtime.generation_id)
                    if item.id == context.claim.task_id
                ),
                None,
            )
            if task is not None:
                return task.task_ref
        return str(context.claim.task_id)

    def _append_event(
        self,
        context: ScheduledTaskContext,
        kind: str,
        payload: dict[str, object],
    ) -> None:
        payload = {"claim_id": str(context.claim.id), **payload}
        context.store.append_claim_execution_event(
            ExecutionEvent(
                run_id=context.claim.run_id,
                task_id=context.claim.task_id,
                kind=kind,
                payload=payload,
                created_at=context.clock(),
            ),
            context.owner_token,
            context.claim.claim_token,
            now=context.clock(),
        )

    def _block(
        self,
        context: ScheduledTaskContext,
        reason: str,
        commands: tuple[SanityCommandResult, ...],
    ) -> HostSanityResult:
        runtime = context.store.get_task_runtime(context.claim.task_id)
        if runtime is not None and runtime.status is TaskRuntimeStatus.MERGING:
            context.transition(
                TaskRuntimeStatus.MERGING,
                TaskRuntimeStatus.BLOCKED,
                resume_phase="merging",
                state_reason=reason,
            )
        return HostSanityResult(TaskRuntimeStatus.BLOCKED, reason, commands=commands)


def _command_cwd(worktree: Path, value: str) -> Path:
    portable = PurePosixPath(value)
    if portable.is_absolute() or ".." in portable.parts:
        raise SanityPhaseError(f"sanity command cwd is unsafe: {value!r}")
    candidate = (worktree / portable).resolve()
    if not candidate.is_relative_to(worktree) or not candidate.is_dir():
        raise SanityPhaseError(f"sanity command cwd is missing: {value!r}")
    return candidate


def _output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return (
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    )


def _error_text(error: BaseException) -> str:
    message = str(error) or error.__class__.__name__
    notes = getattr(error, "__notes__", ())
    return "\n".join((message, *(str(note) for note in notes)))


def _redact(value: str, mask_values: Sequence[str]) -> str:
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


__all__ = [
    "HostSanityPhase",
    "HostSanityResult",
    "SanityCommandResult",
    "SanityPhaseError",
]
