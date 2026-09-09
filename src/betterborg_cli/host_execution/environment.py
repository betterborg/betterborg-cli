"""Checkout-local environment preparation for claimed task worktrees."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from urllib.parse import quote

from betterborg_cli.agent_runtime import CancellationToken
from betterborg_cli.agent_runtime.process import run_captured
from betterborg_cli.host_execution.git import SafeGit
from betterborg_cli.host_execution.guard import PrimaryCheckoutGuard
from betterborg_cli.host_execution.preflight import (
    HostCommand,
    HostPreflightPlan,
    selected_preparation_commands,
)
from betterborg_cli.progress import AgentActivity, AgentActivityKind
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_config import PreparationMode
from betterborg_cli.store import (
    EnvironmentAttempt,
    ExecutionAttemptStatus,
    SqliteStore,
    TaskClaim,
    TaskRuntime,
    TaskRuntimeStatus,
)
from betterborg_cli.store.models import utcnow

_PREPARATION_CONTRACT_VERSION = 1


class EnvironmentMaterializationError(RuntimeError):
    """Raised when a claimed task cannot safely consume its environment."""


@dataclass(frozen=True, slots=True)
class EnvironmentMaterialization:
    """Result of preparing one exact checkout."""

    preparation_key: str
    materialization_reused: bool
    environment: Mapping[str, str] = field(repr=False, hash=False)
    #: Why the checkout is not prepared, when it is not. A task that ran
    #: without its toolchain must not read like one that ran with it, so this
    #: reaches the task's outcome and the agent working in the checkout.
    preparation_note: str | None = None


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
ActivitySink = Callable[[AgentActivity], None]
Clock = Callable[[], datetime]


def materialization_marker(paths: RepoPaths, worktree: Path) -> Path:
    """Locate the marker recording what one checkout has materialized.

    While Betterborg's own files live inside the repository the marker is a
    checkout-local file the managed ignore block hides, alongside the ignored
    dependencies it describes. Once they live outside it, nothing of
    Betterborg's may be written into a checkout at all, so the marker joins
    the rest of its state and is keyed by the checkout it speaks for.
    """
    if paths.tracked_in_repository:
        return worktree / ".betterborg/state/environment-materialization"
    key = hashlib.sha256(str(Path(worktree).resolve()).encode("utf-8")).hexdigest()
    return paths.state_dir / "environment-markers" / key


def discard_materialization_marker(paths: RepoPaths, worktree: Path) -> None:
    """Forget what a checkout materialized before it is replaced.

    A checkout-local marker leaves with the checkout that holds it. One kept
    outside every checkout has to be discarded deliberately, or a freshly
    minted worktree would inherit the claim of the one it replaced.
    """
    if paths.tracked_in_repository:
        return
    _invalidate_marker(materialization_marker(paths, worktree))


def _preparation_key(plan: HostPreflightPlan) -> str:
    """Key a prepared worktree by the commands that would prepare it."""
    payload = {
        "commands": _command_payload(
            selected_preparation_commands(
                prepare_commands=plan.prepare_commands,
                materialize_commands=plan.materialize_commands,
            )
        ),
        "contract": _PREPARATION_CONTRACT_VERSION,
    }
    encoded = json.dumps(
        payload, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class HostEnvironmentManager:
    """Prepare every claimed worktree by running the repository's commands."""

    def __init__(
        self,
        repository_root: Path,
        *,
        environment: Mapping[str, str] | None = None,
        command_runner: CommandRunner | None = None,
        activity: ActivitySink | None = None,
        clock: Clock = utcnow,
        cancel: CancellationToken | None = None,
        git: SafeGit | None = None,
        preparation: PreparationMode = PreparationMode.REQUIRED,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.preparation = PreparationMode(preparation)
        self._paths = RepoPaths.discover(self.repository_root, cancel=cancel)
        if self._paths.root != self.repository_root:
            raise EnvironmentMaterializationError(
                "environment manager must be bound to the Git worktree root"
            )
        if git is not None and git.cwd != self.repository_root:
            raise EnvironmentMaterializationError(
                "environment manager Git binding must match repository"
            )
        self._git = git or SafeGit(self.repository_root, cancel=cancel)
        self._environment = dict(os.environ if environment is None else environment)
        self._run = command_runner or run_captured
        self._activity = activity
        self._cancel = cancel
        self._clock = clock
        self._guard = PrimaryCheckoutGuard(
            self.repository_root, git=self._git
        )

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
        force_preparation: bool = False,
    ) -> EnvironmentMaterialization:
        """Move one claimed task through environment setup into coding.

        A successful attempt is reused only for the exact preparation key, so
        an edit to the command that prepares a worktree prepares it again.
        A caller holding a tree the key cannot speak for — a merge tip, which
        the same commands produce a different install from — asks for
        preparation outright.
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
            preparation_key = _preparation_key(plan)
            base_environment = self._base_command_environment()
            command_environments = self._command_environments(
                plan, base_environment, secret_values or {}
            )
            materialization_reused = False
            preparation_note: str | None = None
            if self.preparation is PreparationMode.SKIPPED:
                preparation_note = (
                    "preparation is skipped by configuration; the checkout "
                    "holds only what its commit tracks"
                )
            else:
                try:
                    materialization_reused = self._materialize_worktree(
                        store,
                        plan,
                        claim,
                        owner_token,
                        preparation_key=preparation_key,
                        worktree=worktree,
                        command_environments=command_environments,
                        force_preparation=force_preparation,
                        activity=activity,
                    )
                except EnvironmentMaterializationError as error:
                    if self.preparation is PreparationMode.REQUIRED:
                        raise
                    # Optional preparation survives a command that fails and
                    # leaves the checkout as it found it. It cannot survive
                    # one that wrote into the checkout: those writes are not
                    # the task's work and would be graded as if they were,
                    # and discarding them needs the destructive Git the
                    # worktree guard deliberately withholds. So this refuses
                    # exactly what it cannot clean up, and says which it was.
                    self._assert_no_tracked_changes(
                        worktree, "after preparation failed"
                    )
                    preparation_note = f"preparation did not complete: {error}"
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
            preparation_key=preparation_key,
            materialization_reused=materialization_reused,
            environment=MappingProxyType(dict(base_environment)),
            preparation_note=preparation_note,
        )

    def _materialize_worktree(
        self,
        store: SqliteStore,
        plan: HostPreflightPlan,
        claim: TaskClaim,
        owner_token: str,
        *,
        preparation_key: str,
        worktree: Path,
        command_environments: Mapping[
            str, tuple[Mapping[str, str], Sequence[str]]
        ],
        force_preparation: bool,
        activity: ActivitySink | None,
    ) -> bool:
        marker = self._materialization_marker(worktree)
        if (
            not force_preparation
            and store.find_completed_environment_attempt(
                preparation_key, kind="materialize", task_id=claim.task_id
            )
            is not None
            and _prepared_marker_matches(marker, preparation_key)
        ):
            return True

        # A failed or intervening preparation may already have changed ignored
        # checkout-local dependencies.  Invalidate the prior state before
        # running so a later A -> B -> A transition cannot reuse A.
        _invalidate_marker(marker)

        self._record_attempt(
            store,
            claim,
            owner_token,
            preparation_key=preparation_key,
            commands=selected_preparation_commands(
                prepare_commands=plan.prepare_commands,
                materialize_commands=plan.materialize_commands,
            ),
            worktree=worktree,
            completion_marker=marker,
            command_environments=command_environments,
            activity=activity,
        )
        return False

    def _record_attempt(
        self,
        store: SqliteStore,
        claim: TaskClaim,
        owner_token: str,
        *,
        preparation_key: str,
        commands: Sequence[HostCommand],
        worktree: Path,
        command_environments: Mapping[
            str, tuple[Mapping[str, str], Sequence[str]]
        ],
        completion_marker: Path,
        activity: ActivitySink | None = None,
    ) -> None:
        task_id = claim.task_id
        prior = [
            attempt
            for attempt in store.list_environment_attempts(task_id)
            if attempt.kind == "materialize"
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
            run_id=claim.run_id,
            claim_id=claim.id,
            task_id=task_id,
            kind="materialize",
            attempt_number=len(prior) + 1,
            fingerprint=preparation_key,
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
            claim.claim_token,
            now=started_at,
        )

        started = time.monotonic()
        try:
            with self._guard.protect(str(task_id), "environment materialize"):
                results = self._run_commands(
                    commands,
                    worktree=worktree,
                    command_environments=command_environments,
                    activity=activity,
                )
                _write_marker(completion_marker, preparation_key)
        except BaseException as error:
            duration = time.monotonic() - started
            redacted = redact_secrets(str(error), mask_values)
            store.complete_environment_attempt(
                attempt.id,
                owner_token,
                claim.claim_token,
                status=ExecutionAttemptStatus.FAILED,
                error=redacted,
                duration_seconds=duration,
                now=self._clock(),
            )
            self._raise_if_cancelled(error)
            if isinstance(error, EnvironmentMaterializationError):
                raise EnvironmentMaterializationError(redacted) from error
            raise EnvironmentMaterializationError(redacted) from error

        store.complete_environment_attempt(
            attempt.id,
            owner_token,
            claim.claim_token,
            status=ExecutionAttemptStatus.COMPLETED,
            result={"commands": results},
            duration_seconds=time.monotonic() - started,
            now=self._clock(),
        )

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

    def _base_command_environment(self) -> dict[str, str]:
        """Return the operator environment every repository command runs in.

        A repository builds on this machine, in this shell, with these
        toolchains and these warm caches. Only the credential prompt is
        suppressed: a command runs with no timeout and inherits a stdin, so
        one reaching a private dependency would otherwise block forever.
        """
        base_environment = dict(self._environment)
        base_environment["GIT_TERMINAL_PROMPT"] = "0"
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
            environments[stage] = (
                command_secret_environment(
                    plan, stage, base_environment, secret_values
                ),
                mask_values,
            )
        return environments

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
                "claimed task path is not its registered Betterborg worktree: "
                f"{worktree}"
            )

    def _materialization_marker(self, worktree: Path) -> Path:
        marker = materialization_marker(self._paths, worktree)
        if not self._paths.tracked_in_repository:
            # No ignore rule of the repository's has anything to say about a
            # marker outside every checkout, but it still has to be somewhere
            # Betterborg owns: a symlinked marker directory pointing back into
            # the repository would put the file in the working tree that the
            # relocation exists to keep empty.
            parent = marker.parent.resolve()
            if not parent.is_relative_to(self._paths.tracked_dir.resolve()):
                raise EnvironmentMaterializationError(
                    "environment marker escapes the tracked directory"
                )
            return marker
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
    base_environment: Mapping[str, str],
    secret_values: Mapping[str, str],
) -> dict[str, str]:
    """Return one stage's environment holding only the secrets it declared.

    Commands run in the operator's environment, so a declared secret the
    stage did not name has to be subtracted as well as withheld. Both halves
    live here because two call sites compose their own environment and two
    independently written filters would disagree the next time the secret
    model changes.
    """
    environment = dict(base_environment)
    for secret in plan.secret_requirements:
        environment.pop(secret.name, None)
    for secret in plan.secret_requirements:
        if secret.scope not in {"all", "build"} or stage not in secret.used_by:
            continue
        value = secret_values.get(secret.name)
        if value is None:
            raise EnvironmentMaterializationError(
                f"build-scoped secret value is unavailable: {secret.name}"
            )
        environment[secret.name] = value
    return environment


def redacted_dropped_command_summary(
    plan: HostPreflightPlan, secret_values: Mapping[str, str]
) -> str:
    """Return the dropped-check summary with declared secret values masked.

    The summary quotes a catalogued command's argv and its evidence, and every
    surface that reports a run carries it: the terminal, the progress line, the
    task's durable state reason, the MCP payload, and the pull request body
    that gets pushed. It is masked wherever it is read, like every other
    quotation of the analysis.
    """

    return redact_secrets(
        plan.dropped_command_summary,
        declared_secret_mask_values(plan, secret_values),
    )


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


def _prepared_marker_matches(marker: Path, preparation_key: str) -> bool:
    try:
        return marker.read_text(encoding="utf-8").strip() == preparation_key
    except OSError:
        return False


def _write_marker(marker: Path, preparation_key: str) -> None:
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{preparation_key}\n", encoding="utf-8")
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
