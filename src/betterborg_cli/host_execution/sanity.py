"""Locked post-merge sanity, project-base advancement, and cleanup."""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path

from betterborg_cli.agent_runtime import CancellationToken, run_captured
from betterborg_cli.host_execution.compose import (
    ComposeStack,
    ComposeStackError,
    HostComposeManager,
    service_url_environment,
)
from betterborg_cli.host_execution.environment import (
    EnvironmentMaterializationError,
    HostEnvironmentManager,
    command_cwd,
    command_secret_environment,
    declared_secret_mask_values,
    redact_secrets,
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
from betterborg_cli.progress import AgentActivity, AgentActivityKind
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.store import ExecutionEvent, TaskRuntime, TaskRuntimeStatus

RepositoryLockFactory = Callable[[], AbstractContextManager[None]]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]

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
        command_runner: CommandRunner | None = None,
        timeout_seconds: float = 600,
        cancel: CancellationToken | None = None,
        git: SafeGit | None = None,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        paths = RepoPaths.discover(self.repository_root, cancel=cancel)
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
        self._run = command_runner or run_captured
        self._timeout_seconds = timeout_seconds
        if git is not None and git.cwd != self.repository_root:
            raise SanityPhaseError("sanity phase Git binding must match repository")
        self._git = git or SafeGit(self.repository_root, cancel=cancel)
        self._guard = PrimaryCheckoutGuard(
            self.repository_root, git=self._git
        )

    def run(
        self,
        context: ScheduledTaskContext,
        tip: MergeTip,
        *,
        secret_values: Mapping[str, str] | None = None,
        existing_stack: ComposeStack | None = None,
    ) -> HostSanityResult:
        """Sanity-check and publish ``tip``, or durably block the task."""
        commands: list[SanityCommandResult] = []
        stack_to_stop = existing_stack
        try:
            runtime, worktree = self._runtime_and_worktree(context, tip)
            with self._guard.protect(self._task_ref(context), "sanity"):
                with self._repository_lock():
                    # From this point the locked sanity gate owns the supplied
                    # agent-phase stack.  It retires that stack before starting
                    # a fresh sanity stack, and tears the fresh stack down before
                    # recording success or advancing the shared project base.
                    locked_stack = stack_to_stop
                    stack_to_stop = None
                    published = self._run_locked(
                        context,
                        runtime,
                        worktree,
                        tip,
                        secret_values or {},
                        commands,
                        existing_stack=locked_stack,
                    )
                    cleanup_runtime = context.store.get_task_runtime(
                        context.claim.task_id
                    )
                    if cleanup_runtime is None:
                        raise SanityPhaseError(
                            "task runtime disappeared before cleanup"
                        )
                    if not self._worktree_manager.cleanup_published_task_worktree(
                        cleanup_runtime
                    ):
                        raise SanityPhaseError(
                            "published task worktree was not eligible for cleanup"
                        )
                    context.transition(
                        TaskRuntimeStatus.MERGING,
                        TaskRuntimeStatus.DONE,
                        resume_phase="done",
                        state_reason=(
                            f"advanced {tip.project_branch} to {published}"
                        ),
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
            ValueError,
        ) as error:
            cleanup_detail = ""
            if stack_to_stop is not None:
                try:
                    self._compose_manager.stop_claimed_stack(
                        context.store,
                        stack_to_stop,
                        context.claim,
                        context.owner_token,
                        cancel=context.cancel,
                        activity=context.activity,
                    )
                except BaseException as cleanup_error:
                    cleanup_detail = (
                        "; task-owned Compose cleanup also failed: "
                        f"{cleanup_error}"
                    )
                stack_to_stop = None
            reason = redact_secrets(
                _error_text(error) + cleanup_detail,
                declared_secret_mask_values(self.plan, secret_values or {}),
            )
            return self._block(context, reason, tuple(commands))

        return HostSanityResult(
            TaskRuntimeStatus.DONE,
            f"sanity passed and advanced {tip.project_branch} to {published}",
            published,
            tuple(commands),
        )

    def _run_locked(
        self,
        context: ScheduledTaskContext,
        runtime: TaskRuntime,
        worktree: Path,
        tip: MergeTip,
        secret_values: Mapping[str, str],
        command_results: list[SanityCommandResult],
        *,
        existing_stack: ComposeStack | None,
    ) -> str:
        prior_stack = existing_stack
        sanity_stack = None
        commands: tuple[SanityCommandResult, ...] = ()
        materialization = None
        already_advanced = False
        active_error: BaseException | None = None
        try:
            current_base = self._resolve_project_tip(tip.project_branch)
            if current_base == tip.commit_sha:
                if not self._advance_was_attested(context, tip):
                    raise SanityPhaseError(
                        "project base is already at the merge tip without a durable "
                        "BetterBorg advancement attestation"
                    )
                if worktree.exists():
                    if not worktree.is_dir():
                        raise SanityPhaseError(
                            "merged task worktree is not a directory"
                        )
                    self._verify_tip(runtime, worktree, tip)
                already_advanced = True
            else:
                if current_base != tip.base_commit:
                    raise SanityPhaseError(
                        "project base moved after the merge tip was produced; rerun "
                        "the merge phase before sanity"
                    )
                if not worktree.is_dir():
                    raise SanityPhaseError("merged task worktree is missing")
                self._verify_tip(runtime, worktree, tip)

                # Agent phases may have changed build inputs or mutated service
                # state.  Remove their images, volumes, and containers before
                # rematerializing and rebuilding services from the merged tip.
                if prior_stack is not None:
                    self._compose_manager.stop_claimed_stack(
                        context.store,
                        prior_stack,
                        context.claim,
                        context.owner_token,
                        cancel=context.cancel,
                        activity=context.activity,
                    )
                    prior_stack = None
                materialization = self._environment_manager.materialize_claimed_task(
                    context.store,
                    self.plan,
                    context.claim,
                    context.owner_token,
                    secret_values=secret_values,
                    task_transition=context.transition,
                )
                sanity_stack = self._compose_manager.start_claimed_sanity_stack(
                    context.store,
                    self.plan,
                    context.claim,
                    context.owner_token,
                    cancel=context.cancel,
                    activity=context.activity,
                )
                service_environment = service_url_environment(self.plan.services)
                if sanity_stack is not None:
                    service_environment.update(sanity_stack.environment)
                commands = self._run_commands(
                    worktree,
                    materialization_environment=materialization.environment,
                    service_environment=service_environment,
                    secret_values=secret_values,
                    cancel=context.cancel,
                    activity=context.activity_sink("sanity"),
                )
                command_results.extend(commands)
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
                if not self._git.for_worktree(worktree).is_clean():
                    raise SanityPhaseError(
                        "sanity commands changed tracked or untracked task files"
                    )
        except BaseException as error:
            active_error = error
        finally:
            stack_to_stop = sanity_stack or prior_stack
            if stack_to_stop is not None:
                try:
                    self._compose_manager.stop_claimed_stack(
                        context.store,
                        stack_to_stop,
                        context.claim,
                        context.owner_token,
                        cancel=context.cancel,
                        activity=context.activity,
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
        if already_advanced:
            return tip.commit_sha
        if materialization is None:
            raise SanityPhaseError("sanity materialization did not complete")

        masks = declared_secret_mask_values(self.plan, secret_values)
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
                        "argv": [
                            redact_secrets(arg, masks) for arg in result.command.argv
                        ],
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
        return tip.commit_sha

    def _run_commands(
        self,
        worktree: Path,
        *,
        materialization_environment: Mapping[str, str],
        service_environment: Mapping[str, str],
        secret_values: Mapping[str, str],
        cancel: CancellationToken,
        activity: Callable[[AgentActivity], None] | None,
    ) -> tuple[SanityCommandResult, ...]:
        results: list[SanityCommandResult] = []
        masks = declared_secret_mask_values(self.plan, secret_values)
        for command in self.plan.commands:
            cwd = command_cwd(worktree, command.cwd)
            environment = dict(materialization_environment)
            environment["CI"] = "true"
            environment.update(service_environment)
            command_secrets, _ = command_secret_environment(
                self.plan, command.stage, secret_values
            )
            environment.update(command_secrets)
            redacted_command = HostCommand(
                redact_secrets(command.stage, masks),
                tuple(redact_secrets(argument, masks) for argument in command.argv),
                redact_secrets(command.cwd, masks),
                redact_secrets(command.evidence, masks),
            )
            self._report_command(command.argv, masks, activity)
            try:
                completed = self._run(
                    list(command.argv),
                    cwd=cwd,
                    env=environment,
                    check=False,
                    timeout=self._timeout_seconds,
                    cancel=cancel,
                )
                self._raise_if_cancelled(cancel)
                result = SanityCommandResult(
                    redacted_command,
                    completed.returncode,
                    redact_secrets(completed.stdout or "", masks),
                    redact_secrets(completed.stderr or "", masks),
                )
            except subprocess.TimeoutExpired as error:
                self._raise_if_cancelled(cancel, error)
                result = SanityCommandResult(
                    redacted_command,
                    -1,
                    redact_secrets(_output(error.stdout), masks),
                    redact_secrets(
                        _output(error.stderr)
                        + f"sanity command timed out after {self._timeout_seconds}s",
                        masks,
                    ),
                )
            except OSError as error:
                self._raise_if_cancelled(cancel, error)
                result = SanityCommandResult(
                    redacted_command,
                    -1,
                    "",
                    redact_secrets(f"sanity command could not run: {error}", masks),
                )
            results.append(result)
            if result.returncode != 0:
                break
        return tuple(results)

    @staticmethod
    def _report_command(
        command: Sequence[str],
        masks: Sequence[str],
        activity: Callable[[AgentActivity], None] | None,
    ) -> None:
        if activity is None:
            return
        redacted = [redact_secrets(argument, masks) for argument in command]
        try:
            activity(AgentActivity(AgentActivityKind.COMMAND, shlex.join(redacted)))
        except Exception:
            return

    @staticmethod
    def _raise_if_cancelled(
        cancel: CancellationToken,
        cause: BaseException | None = None,
    ) -> None:
        if not cancel.is_set():
            return
        if cause is None:
            raise KeyboardInterrupt
        raise KeyboardInterrupt from cause

    def _runtime_and_worktree(
        self, context: ScheduledTaskContext, tip: MergeTip
    ) -> tuple[TaskRuntime, Path]:
        runtime = context.store.get_task_runtime(context.claim.task_id)
        if runtime is None or runtime.status is not TaskRuntimeStatus.MERGING:
            raise SanityPhaseError("task must be in its merging phase for sanity")
        if runtime.branch != tip.task_branch or runtime.worktree_path is None:
            raise SanityPhaseError("merge tip does not match the claimed worktree")
        worktree = Path(runtime.worktree_path).resolve()
        expected_project = self._project_branch(context)
        if tip.project_branch != expected_project:
            raise SanityPhaseError("merge tip belongs to another project base")
        return runtime, worktree

    def _verify_tip(
        self, runtime: TaskRuntime, worktree: Path, tip: MergeTip
    ) -> None:
        git = self._git.for_worktree(worktree)
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
        else:
            context.reconcile_progress()
        return HostSanityResult(TaskRuntimeStatus.BLOCKED, reason, commands=commands)


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


__all__ = [
    "HostSanityPhase",
    "HostSanityResult",
    "SanityCommandResult",
    "SanityPhaseError",
]
