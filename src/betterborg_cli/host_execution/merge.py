"""Guarded project-base merging with conflict-only agent fallback."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from betterborg_cli.agent_runtime import (
    AgentAdapter,
    AgentResult,
    AgentRunSpec,
    AgentStatus,
    BillingMode,
)
from betterborg_cli.host_execution._agent_phase import (
    AgentAttemptArtifacts,
    HostAgentPhaseError,
    VerifiedTaskInputs,
    cancelled_agent_reason,
    current_branch,
    require_ready_worktree,
    result_summary,
    verified_task_inputs,
)
from betterborg_cli.host_execution.coding import CODING_RESULT_SCHEMA
from betterborg_cli.host_execution.git import SafeGit, UnsafeGitError
from betterborg_cli.host_execution.guard import (
    PrimaryCheckoutContaminationError,
    PrimaryCheckoutGuard,
)
from betterborg_cli.host_execution.scheduler import ScheduledTaskContext
from betterborg_cli.planning import TaskDigestDriftError
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.store import (
    AgentAttempt,
    ExecutionAttemptStatus,
    ExecutionEvent,
    TaskRuntime,
    TaskRuntimeStatus,
)

MERGE_RESULT_SCHEMA: dict[str, Any] = CODING_RESULT_SCHEMA

_MERGE_IDENTITY = {
    "GIT_AUTHOR_NAME": "BetterBorg",
    "GIT_AUTHOR_EMAIL": "betterborg@example.invalid",
    "GIT_COMMITTER_NAME": "BetterBorg",
    "GIT_COMMITTER_EMAIL": "betterborg@example.invalid",
}
_MERGE_STARTED_EVENT = "merge.started"
_MERGE_COMPLETED_EVENT = "merge.completed"

RepositoryLockFactory = Callable[[], AbstractContextManager[None]]


class MergePhaseError(RuntimeError):
    """Raised when an approved task cannot be merged safely."""


@dataclass(frozen=True, slots=True)
class HostMergeConfig:
    """Provider and artifact settings for conflict-resolution attempts."""

    model: str
    billing_mode: BillingMode = BillingMode.API
    effort: str | None = None
    allowed_tools: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)
    artifact_root: Path | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("merge model must not be empty")
        object.__setattr__(self, "billing_mode", BillingMode(self.billing_mode))
        object.__setattr__(self, "allowed_tools", tuple(self.allowed_tools))
        if self.artifact_root is not None:
            object.__setattr__(self, "artifact_root", Path(self.artifact_root))


@dataclass(frozen=True, slots=True)
class MergeTip:
    """A verified task tip ready for rematerialization and sanity."""

    task_branch: str
    project_branch: str
    approved_commit: str
    base_commit: str
    commit_sha: str
    agent_used: bool


@dataclass(frozen=True, slots=True)
class HostMergeResult:
    """Outcome of producing, but not publishing, one task merge tip."""

    status: TaskRuntimeStatus
    reason: str
    tip: MergeTip | None = None
    interrupted: bool = False

    def __post_init__(self) -> None:
        if self.status is TaskRuntimeStatus.MERGING:
            if self.tip is None:
                raise ValueError("a successful merge result requires a tip")
        elif self.tip is not None:
            raise ValueError("a terminal merge result cannot expose a tip")
        if self.interrupted and self.status is not TaskRuntimeStatus.BLOCKED:
            raise ValueError("only a blocked merge result may be interrupted")


class HostMergePhase:
    """Merge the current project base into one approved task worktree.

    The injected repository lock is shared with the later sanity/base-publish
    phase. It is held only while the moving project ref is resolved and the
    initial Git merge is started; an agent is never run while holding it.
    """

    def __init__(
        self,
        repository_root: Path,
        adapter: AgentAdapter,
        *,
        config: HostMergeConfig,
        repository_lock: RepositoryLockFactory,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self._paths = RepoPaths.discover(self.repository_root)
        if self._paths.root != self.repository_root:
            raise MergePhaseError(
                "merge phase must be bound to the primary Git checkout"
            )
        if not callable(repository_lock):
            raise TypeError("repository lock must be a context manager factory")
        self._adapter = adapter
        self._config = config
        self._repository_lock = repository_lock
        self._guard = PrimaryCheckoutGuard(self.repository_root)
        self._primary_git = SafeGit(self.repository_root)
        self.artifact_root = Path(
            config.artifact_root
            or self._paths.artifacts_dir / "host-execution"
        ).resolve()
        if self.artifact_root.is_relative_to(self.repository_root) and not (
            self.artifact_root == self._paths.state_dir
            or self.artifact_root.is_relative_to(self._paths.state_dir)
        ):
            raise MergePhaseError(
                "repository-local merge artifacts must be under .borg/state"
            )

    def run(
        self,
        context: ScheduledTaskContext,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> HostMergeResult:
        """Produce an attested merge tip without advancing the project base."""
        try:
            runtime, worktree = require_ready_worktree(
                self.repository_root,
                self._primary_git,
                context,
                expected_statuses={TaskRuntimeStatus.MERGING},
            )
            inputs = verified_task_inputs(
                self.repository_root,
                context,
                worktree,
                prompt_role="merge",
            )
            project_branch = self._project_branch(context)
            approved_commit = self._approved_commit(context)
            self._guard.before_phase(inputs.task.task_ref, "merge")
        except (
            HostAgentPhaseError,
            MergePhaseError,
            PrimaryCheckoutContaminationError,
            TaskDigestDriftError,
            UnsafeGitError,
            OSError,
            subprocess.SubprocessError,
        ) as error:
            return self._block(context, str(error) or error.__class__.__name__)

        try:
            outcome = self._merge(
                context,
                runtime,
                worktree,
                inputs,
                project_branch=project_branch,
                approved_commit=approved_commit,
                environment=environment,
            )
        except (
            HostAgentPhaseError,
            MergePhaseError,
            UnsafeGitError,
            OSError,
            subprocess.SubprocessError,
        ) as error:
            outcome = HostMergeResult(
                TaskRuntimeStatus.BLOCKED,
                str(error) or error.__class__.__name__,
            )

        try:
            self._guard.after_phase(inputs.task.task_ref, "merge")
        except PrimaryCheckoutContaminationError as error:
            outcome = HostMergeResult(TaskRuntimeStatus.BLOCKED, str(error))

        if context.cancel.is_set() and outcome.interrupted:
            return outcome
        if outcome.status is TaskRuntimeStatus.MERGING:
            return outcome
        return self._transition(context, outcome.status, outcome.reason)

    def _merge(
        self,
        context: ScheduledTaskContext,
        runtime: TaskRuntime,
        worktree: Path,
        inputs: VerifiedTaskInputs,
        *,
        project_branch: str,
        approved_commit: str,
        environment: Mapping[str, str] | None,
    ) -> HostMergeResult:
        git = SafeGit(worktree)
        if current_branch(git) != runtime.branch:
            raise MergePhaseError("task worktree is on the wrong branch")

        merge_head = self._merge_head(git)
        unresolved = git.unmerged_paths()
        if merge_head is not None:
            with self._repository_lock():
                base_commit = self._resolve_project_tip(project_branch)
                starting_commit = git.head_sha()
                if merge_head != base_commit:
                    raise MergePhaseError(
                        "active merge does not match the current project base"
                    )
                if self._attested_tip_agent_used(
                    context,
                    git,
                    approved_commit=approved_commit,
                    project_branch=project_branch,
                    task_branch=runtime.branch or "",
                    target_base=base_commit,
                    commit_sha=starting_commit,
                ) is None:
                    raise MergePhaseError(
                        "active merge did not start from an attested task tip"
                    )
                if not self._has_started_attestation(
                    context,
                    approved_commit=approved_commit,
                    project_branch=project_branch,
                    task_branch=runtime.branch or "",
                    starting_commit=starting_commit,
                    base_commit=base_commit,
                ):
                    raise MergePhaseError(
                        "active merge lacks a durable host attestation"
                    )
            return self._invoke_agent(
                context,
                runtime,
                git,
                inputs,
                project_branch=project_branch,
                approved_commit=approved_commit,
                base_commit=base_commit,
                unresolved=unresolved,
                environment=environment,
            )
        if unresolved:
            raise MergePhaseError(
                "task worktree has unresolved paths without an active merge: "
                + ", ".join(unresolved[:20])
            )
        if not git.is_clean():
            raise MergePhaseError(
                "task worktree has uncommitted changes; preserving it and "
                "refusing to merge"
            )

        with self._repository_lock():
            base_commit = self._resolve_project_tip(project_branch)
            current = git.head_sha()
            agent_used = self._attested_tip_agent_used(
                context,
                git,
                approved_commit=approved_commit,
                project_branch=project_branch,
                task_branch=runtime.branch or "",
                target_base=base_commit,
                commit_sha=current,
            )
            if agent_used is None:
                raise MergePhaseError(
                    "approved task commit no longer matches task worktree"
                )
            if git.is_ancestor(base_commit, current):
                outcome = self._verified_tip_locked(
                    git,
                    runtime,
                    project_branch=project_branch,
                    approved_commit=approved_commit,
                    base_commit=base_commit,
                    agent_used=agent_used,
                )
                if not agent_used and not self._has_completed_host_attestation(
                    context,
                    approved_commit=approved_commit,
                    project_branch=project_branch,
                    task_branch=runtime.branch or "",
                    base_commit=base_commit,
                    commit_sha=current,
                ):
                    self._record_merge_event(
                        context,
                        kind=_MERGE_COMPLETED_EVENT,
                        approved_commit=approved_commit,
                        project_branch=project_branch,
                        task_branch=runtime.branch or "",
                        starting_commit=current,
                        base_commit=base_commit,
                        commit_sha=current,
                    )
                return outcome
            merge_date = f"@{int(context.clock().timestamp())} +0000"
            expected_commit = self._expected_clean_merge_commit(
                git,
                starting_commit=current,
                base_commit=base_commit,
                merge_date=merge_date,
            )
            self._record_merge_event(
                context,
                kind=_MERGE_STARTED_EVENT,
                approved_commit=approved_commit,
                project_branch=project_branch,
                task_branch=runtime.branch or "",
                starting_commit=current,
                base_commit=base_commit,
                expected_commit=expected_commit,
                merge_date=merge_date,
            )
            merge_environment = self._merge_environment()
            merge_environment.update(
                {
                    "GIT_AUTHOR_DATE": merge_date,
                    "GIT_COMMITTER_DATE": merge_date,
                }
            )
            merged = git.run(
                [
                    "merge",
                    "--no-edit",
                    "-m",
                    self._clean_merge_message(base_commit),
                    base_commit,
                ],
                check=False,
                env=merge_environment,
            )
            unresolved = git.unmerged_paths()
            if merged.returncode == 0 and not unresolved:
                outcome = self._verified_tip_locked(
                    git,
                    runtime,
                    project_branch=project_branch,
                    approved_commit=approved_commit,
                    base_commit=base_commit,
                    agent_used=False,
                )
                assert outcome.tip is not None
                if outcome.tip.commit_sha != expected_commit:
                    raise MergePhaseError(
                        "clean merge did not create its pre-attested commit"
                    )
                self._record_merge_event(
                    context,
                    kind=_MERGE_COMPLETED_EVENT,
                    approved_commit=approved_commit,
                    project_branch=project_branch,
                    task_branch=runtime.branch or "",
                    starting_commit=current,
                    base_commit=base_commit,
                    commit_sha=outcome.tip.commit_sha,
                )
                return outcome

        if not unresolved:
            detail = (merged.stderr or merged.stdout).strip()[-4000:]
            return HostMergeResult(
                TaskRuntimeStatus.BLOCKED,
                "Git could not merge the project base without reporting "
                f"conflicted paths; merge agent was not invoked: {detail}",
            )
        return self._invoke_agent(
            context,
            runtime,
            git,
            inputs,
            project_branch=project_branch,
            approved_commit=approved_commit,
            base_commit=base_commit,
            unresolved=unresolved,
            environment=environment,
        )

    def _invoke_agent(
        self,
        context: ScheduledTaskContext,
        runtime: TaskRuntime,
        git: SafeGit,
        inputs: VerifiedTaskInputs,
        *,
        project_branch: str,
        approved_commit: str,
        base_commit: str,
        unresolved: tuple[str, ...],
        environment: Mapping[str, str] | None,
    ) -> HostMergeResult:
        resumed = self._resume_agent_tip(
            context,
            runtime,
            git,
            project_branch=project_branch,
            approved_commit=approved_commit,
            base_commit=base_commit,
        )
        if resumed is not None:
            return resumed

        attempts = context.store.list_agent_attempts(context.claim.task_id)
        attempt_number = 1 + sum(item.phase == "merge" for item in attempts)
        attempt_id = uuid4()
        attempt_dir = (
            self.artifact_root
            / str(context.claim.task_id)
            / f"merge-{attempt_number:03d}-{attempt_id.hex}"
        )
        artifacts = AgentAttemptArtifacts(
            self.repository_root, attempt_dir, git.cwd, "merge"
        )
        user_prompt = _render_merge_prompt(
            inputs,
            task_branch=runtime.branch or "",
            project_branch=project_branch,
            approved_commit=approved_commit,
            base_commit=base_commit,
            unresolved=unresolved,
        )
        try:
            attempt_dir.mkdir(parents=True, exist_ok=False)
            artifacts.write_text("system-prompt.md", inputs.system_prompt)
            artifacts.write_text("user-prompt.md", user_prompt)
            artifacts.write_text(
                "result-schema.json",
                json.dumps(MERGE_RESULT_SCHEMA, indent=2, sort_keys=True) + "\n",
            )
        except OSError as error:
            return HostMergeResult(
                TaskRuntimeStatus.BLOCKED,
                f"unable to create merge artifacts: {error}",
            )

        log_path = attempt_dir / "merge.log"
        result_path = attempt_dir / "merge.result.json"
        started_at = context.clock()
        attempt = AgentAttempt(
            id=attempt_id,
            run_id=context.claim.run_id,
            claim_id=context.claim.id,
            task_id=context.claim.task_id,
            phase="merge",
            review_round=runtime.review_round,
            attempt_number=attempt_number,
            adapter=self._adapter.name,
            model=self._config.model,
            billing_mode=self._config.billing_mode,
            status=ExecutionAttemptStatus.RUNNING,
            log_path=artifacts.reference(log_path),
            started_at=started_at,
            finished_at=None,
        )
        context.store.append_agent_attempt(
            attempt,
            context.owner_token,
            context.claim.claim_token,
            now=started_at,
        )
        spec = AgentRunSpec(
            system_prompt=inputs.system_prompt,
            user_prompt=user_prompt,
            schema=MERGE_RESULT_SCHEMA,
            cwd=git.cwd,
            model=self._config.model,
            log_path=log_path,
            result_path=result_path,
            allowed_tools=self._config.allowed_tools,
            env={**self._config.environment, **(environment or {})},
            effort=self._config.effort,
            billing_mode=self._config.billing_mode,
        )
        operational_error: BaseException | None = None
        try:
            result = self._adapter.run(spec, cancel=context.cancel)
        except BaseException as error:
            operational_error = error
            result = AgentResult(
                status=(
                    AgentStatus.CANCELLED
                    if context.cancel.is_set()
                    else AgentStatus.FAILED
                ),
                log_path=log_path,
                error=f"{type(error).__name__}: {error}",
                billing_mode=self._config.billing_mode,
                provider=self._adapter.name,
                model=self._config.model,
            )

        cancellation_reason = cancelled_agent_reason(
            result,
            context.cancel,
            phase="merge",
        )
        outcome = self._classify_agent(
            result,
            runtime=runtime,
            git=git,
            project_branch=project_branch,
            approved_commit=approved_commit,
            base_commit=base_commit,
            operational_error=operational_error,
            cancellation_reason=cancellation_reason,
        )
        durable_result = dict(result.payload or {})
        durable_result["_betterborg"] = {
            "artifact_dir": artifacts.reference(attempt_dir),
            "approved_commit": approved_commit,
            "base_commit": base_commit,
            "commit_sha": outcome.tip.commit_sha if outcome.tip else None,
            "outcome_status": outcome.status.value,
            "outcome_reason": outcome.reason,
            "project_branch": project_branch,
            "task_branch": runtime.branch,
            "provider": result.provider or self._adapter.name,
            "model": result.model or self._config.model,
            "billing_mode": result.billing_mode.value,
        }
        try:
            artifacts.finish(result, durable_result)
        except OSError as error:
            outcome = HostMergeResult(
                TaskRuntimeStatus.BLOCKED,
                f"artifact persistence failed: {error}",
            )
            durable_result["_betterborg"]["outcome_status"] = (
                outcome.status.value
            )
            durable_result["_betterborg"]["outcome_reason"] = outcome.reason
            durable_result["_betterborg"]["commit_sha"] = None

        terminal_status = {
            AgentStatus.COMPLETED: ExecutionAttemptStatus.COMPLETED,
            AgentStatus.CANCELLED: ExecutionAttemptStatus.CANCELLED,
            AgentStatus.FAILED: ExecutionAttemptStatus.FAILED,
        }[result.status]
        context.store.complete_agent_attempt(
            attempt.id,
            context.owner_token,
            context.claim.claim_token,
            status=terminal_status,
            result_path=(
                artifacts.reference(result_path) if result_path.is_file() else None
            ),
            result=durable_result,
            summary=result_summary(result),
            duration_seconds=result.duration_seconds,
            usage=result.usage,
            now=context.clock(),
        )
        return outcome

    def _classify_agent(
        self,
        result: AgentResult,
        *,
        runtime: TaskRuntime,
        git: SafeGit,
        project_branch: str,
        approved_commit: str,
        base_commit: str,
        operational_error: BaseException | None,
        cancellation_reason: str | None,
    ) -> HostMergeResult:
        if operational_error is not None:
            return HostMergeResult(TaskRuntimeStatus.BLOCKED, str(operational_error))
        try:
            branch = current_branch(git)
        except (HostAgentPhaseError, UnsafeGitError, OSError) as error:
            return HostMergeResult(TaskRuntimeStatus.BLOCKED, str(error))
        if branch != runtime.branch:
            return HostMergeResult(
                TaskRuntimeStatus.BLOCKED,
                "merge agent changed the task branch",
            )
        if result.status is AgentStatus.CANCELLED:
            return HostMergeResult(
                TaskRuntimeStatus.BLOCKED,
                cancellation_reason or "merge agent was interrupted",
                interrupted=True,
            )
        if result.status is AgentStatus.FAILED:
            return HostMergeResult(
                TaskRuntimeStatus.FAILED, result.error or "merge agent failed"
            )
        payload_status = (result.payload or {}).get("status")
        if payload_status != "completed":
            status = (
                TaskRuntimeStatus.FAILED
                if payload_status == "failed"
                else TaskRuntimeStatus.BLOCKED
            )
            return HostMergeResult(
                status, f"merge agent reported {payload_status or 'no status'}"
            )
        try:
            return self._verified_tip(
                git,
                runtime,
                project_branch=project_branch,
                approved_commit=approved_commit,
                base_commit=base_commit,
                agent_used=True,
            )
        except (HostAgentPhaseError, MergePhaseError, UnsafeGitError) as error:
            return HostMergeResult(TaskRuntimeStatus.BLOCKED, str(error))

    def _verified_tip(
        self,
        git: SafeGit,
        runtime: TaskRuntime,
        *,
        project_branch: str,
        approved_commit: str,
        base_commit: str,
        agent_used: bool,
    ) -> HostMergeResult:
        with self._repository_lock():
            return self._verified_tip_locked(
                git,
                runtime,
                project_branch=project_branch,
                approved_commit=approved_commit,
                base_commit=base_commit,
                agent_used=agent_used,
            )

    def _verified_tip_locked(
        self,
        git: SafeGit,
        runtime: TaskRuntime,
        *,
        project_branch: str,
        approved_commit: str,
        base_commit: str,
        agent_used: bool,
    ) -> HostMergeResult:
        """Verify a tip while the caller holds the repository lock."""
        if self._resolve_project_tip(project_branch) != base_commit:
            raise MergePhaseError(
                "project base moved while the merge phase was running"
            )
        branch = current_branch(git)
        if branch != runtime.branch:
            raise MergePhaseError("merge agent changed the task branch")
        unresolved = git.unmerged_paths()
        if unresolved:
            formatted = "\n".join(f"  - {path}" for path in unresolved[:20])
            raise MergePhaseError(
                "merge completed with unresolved paths; worktree preserved:\n"
                + formatted
            )
        if not git.is_clean():
            raise MergePhaseError(
                "merge left uncommitted changes in the task worktree; "
                "worktree preserved"
            )
        commit_sha = git.head_sha()
        if not git.is_ancestor(approved_commit, commit_sha):
            raise MergePhaseError(
                "merge tip does not descend from the approved task commit"
            )
        if not git.is_ancestor(base_commit, commit_sha):
            raise MergePhaseError(
                "merge tip does not descend from the resolved project base"
            )
        return HostMergeResult(
            TaskRuntimeStatus.MERGING,
            f"produced merge tip {commit_sha}",
            MergeTip(
                task_branch=runtime.branch or "",
                project_branch=project_branch,
                approved_commit=approved_commit,
                base_commit=base_commit,
                commit_sha=commit_sha,
                agent_used=agent_used,
            ),
        )

    def _resume_agent_tip(
        self,
        context: ScheduledTaskContext,
        runtime: TaskRuntime,
        git: SafeGit,
        *,
        project_branch: str,
        approved_commit: str,
        base_commit: str,
    ) -> HostMergeResult | None:
        completed = [
            attempt
            for attempt in context.store.list_agent_attempts(context.claim.task_id)
            if attempt.phase == "merge"
            and attempt.status is ExecutionAttemptStatus.COMPLETED
        ]
        if not completed:
            return None
        metadata = (completed[-1].result or {}).get("_betterborg")
        if not isinstance(metadata, Mapping):
            return None
        if metadata.get("outcome_status") != TaskRuntimeStatus.MERGING.value:
            return None
        if metadata.get("base_commit") != base_commit:
            return None
        commit_sha = metadata.get("commit_sha")
        if not isinstance(commit_sha, str) or git.head_sha() != commit_sha:
            raise MergePhaseError(
                "completed merge attempt no longer matches task worktree"
            )
        return self._verified_tip(
            git,
            runtime,
            project_branch=project_branch,
            approved_commit=approved_commit,
            base_commit=base_commit,
            agent_used=True,
        )

    def _project_branch(self, context: ScheduledTaskContext) -> str:
        generation = context.store.get_task_generation(context.runtime.generation_id)
        if generation is None:
            raise MergePhaseError("task generation is missing")
        borg = context.store.get_borg(generation.borg_id)
        if borg is None:
            raise MergePhaseError("task Borg is missing")
        if borg.id != generation.borg_id:
            raise MergePhaseError("task Borg does not match its generation")
        branch = f"project/{borg.name}"
        if not self._primary_git.is_valid_branch_name(branch):
            raise MergePhaseError(f"invalid project branch: {branch!r}")
        return branch

    def _approved_commit(self, context: ScheduledTaskContext) -> str:
        reviews = [
            attempt
            for attempt in context.store.list_agent_attempts(context.claim.task_id)
            if attempt.phase == "review"
            and attempt.status is ExecutionAttemptStatus.COMPLETED
            and (attempt.result or {}).get("status") == "approved"
        ]
        if not reviews:
            raise MergePhaseError("merge requires a completed review approval")
        metadata = (reviews[-1].result or {}).get("_betterborg")
        if not isinstance(metadata, Mapping) or metadata.get(
            "outcome_status"
        ) != TaskRuntimeStatus.MERGING.value:
            raise MergePhaseError("review approval lacks a merging attestation")
        commit_sha = metadata.get("commit_sha")
        if not isinstance(commit_sha, str) or not commit_sha:
            raise MergePhaseError("review approval lacks a commit attestation")
        return commit_sha

    def _attested_tip_agent_used(
        self,
        context: ScheduledTaskContext,
        git: SafeGit,
        *,
        approved_commit: str,
        project_branch: str,
        task_branch: str,
        target_base: str,
        commit_sha: str,
    ) -> bool | None:
        """Return agent provenance for an exact trusted tip, if any."""
        if commit_sha == approved_commit:
            return False
        agent = self._completed_merge_attestation(
            context,
            approved_commit=approved_commit,
            project_branch=project_branch,
            task_branch=task_branch,
            commit_sha=commit_sha,
        )
        if agent is not None:
            attested_base = agent.get("base_commit")
            if isinstance(attested_base, str) and git.is_ancestor(
                attested_base, target_base
            ):
                return True
        host = self._host_merge_attestation(
            context,
            kind=_MERGE_COMPLETED_EVENT,
            approved_commit=approved_commit,
            project_branch=project_branch,
            task_branch=task_branch,
            commit_sha=commit_sha,
        )
        if host is not None:
            attested_base = host.get("base_commit")
            if isinstance(attested_base, str) and git.is_ancestor(
                attested_base, target_base
            ):
                return False
        started = self._host_merge_attestation(
            context,
            kind=_MERGE_STARTED_EVENT,
            approved_commit=approved_commit,
            project_branch=project_branch,
            task_branch=task_branch,
            expected_commit=commit_sha,
        )
        if started is not None:
            attested_base = started.get("base_commit")
            if isinstance(attested_base, str) and git.is_ancestor(
                attested_base, target_base
            ):
                return False
        return None

    def _expected_clean_merge_commit(
        self,
        git: SafeGit,
        *,
        starting_commit: str,
        base_commit: str,
        merge_date: str,
    ) -> str | None:
        """Create the exact clean merge object before its branch can move."""
        if git.is_ancestor(starting_commit, base_commit):
            return base_commit
        merged_tree = git.run(
            ["merge-tree", "--write-tree", starting_commit, base_commit],
            check=False,
        )
        if merged_tree.returncode == 1:
            return None
        tree_sha = merged_tree.stdout.splitlines()[0].strip()
        if merged_tree.returncode != 0 or not tree_sha:
            detail = (merged_tree.stderr or merged_tree.stdout).strip()[-4000:]
            raise MergePhaseError(
                "Git could not prepare the clean merge attestation: " + detail
            )
        environment = self._merge_environment()
        environment.update(
            {
                "GIT_AUTHOR_DATE": merge_date,
                "GIT_COMMITTER_DATE": merge_date,
            }
        )
        expected = git.run(
            [
                "commit-tree",
                tree_sha,
                "-p",
                starting_commit,
                "-p",
                base_commit,
                "-m",
                self._clean_merge_message(base_commit),
            ],
            env=environment,
        ).stdout.strip()
        if not expected:
            raise MergePhaseError("Git did not create a clean merge attestation")
        return expected

    @staticmethod
    def _clean_merge_message(base_commit: str) -> str:
        return f"Merge project base {base_commit}"

    def _completed_merge_attestation(
        self,
        context: ScheduledTaskContext,
        *,
        approved_commit: str,
        project_branch: str,
        task_branch: str,
        commit_sha: str,
    ) -> Mapping[str, Any] | None:
        for attempt in reversed(
            context.store.list_agent_attempts(context.claim.task_id)
        ):
            if (
                attempt.phase != "merge"
                or attempt.status is not ExecutionAttemptStatus.COMPLETED
            ):
                continue
            metadata = (attempt.result or {}).get("_betterborg")
            if (
                isinstance(metadata, Mapping)
                and metadata.get("outcome_status")
                == TaskRuntimeStatus.MERGING.value
                and metadata.get("approved_commit") == approved_commit
                and metadata.get("project_branch") == project_branch
                and metadata.get("task_branch") == task_branch
                and metadata.get("commit_sha") == commit_sha
            ):
                return metadata
        return None

    def _record_merge_event(
        self,
        context: ScheduledTaskContext,
        *,
        kind: str,
        approved_commit: str,
        project_branch: str,
        task_branch: str,
        starting_commit: str,
        base_commit: str,
        commit_sha: str | None = None,
        expected_commit: str | None = None,
        merge_date: str | None = None,
    ) -> None:
        recorded_at = context.clock()
        if merge_date is None:
            merge_date = f"@{int(recorded_at.timestamp())} +0000"
        context.store.append_claim_execution_event(
            ExecutionEvent(
                run_id=context.claim.run_id,
                task_id=context.claim.task_id,
                kind=kind,
                payload={
                    "claim_id": str(context.claim.id),
                    "approved_commit": approved_commit,
                    "project_branch": project_branch,
                    "task_branch": task_branch,
                    "starting_commit": starting_commit,
                    "base_commit": base_commit,
                    "commit_sha": commit_sha,
                    "expected_commit": expected_commit,
                    "merge_date": merge_date,
                },
                created_at=recorded_at,
            ),
            context.owner_token,
            context.claim.claim_token,
            now=recorded_at,
        )

    def _host_merge_attestation(
        self,
        context: ScheduledTaskContext,
        *,
        kind: str,
        **expected: str,
    ) -> Mapping[str, Any] | None:
        for event in reversed(
            context.store.list_task_execution_events(
                context.claim.task_id, kind=kind
            )
        ):
            if all(
                event.payload.get(name) == value
                for name, value in expected.items()
            ):
                return event.payload
        return None

    def _has_started_attestation(
        self,
        context: ScheduledTaskContext,
        *,
        approved_commit: str,
        project_branch: str,
        task_branch: str,
        starting_commit: str,
        base_commit: str,
    ) -> bool:
        return (
            self._host_merge_attestation(
                context,
                kind=_MERGE_STARTED_EVENT,
                approved_commit=approved_commit,
                project_branch=project_branch,
                task_branch=task_branch,
                starting_commit=starting_commit,
                base_commit=base_commit,
            )
            is not None
        )

    def _has_completed_host_attestation(
        self,
        context: ScheduledTaskContext,
        *,
        approved_commit: str,
        project_branch: str,
        task_branch: str,
        base_commit: str,
        commit_sha: str,
    ) -> bool:
        return (
            self._host_merge_attestation(
                context,
                kind=_MERGE_COMPLETED_EVENT,
                approved_commit=approved_commit,
                project_branch=project_branch,
                task_branch=task_branch,
                base_commit=base_commit,
                commit_sha=commit_sha,
            )
            is not None
        )

    def _resolve_project_tip(self, project_branch: str) -> str:
        result = self._primary_git.run(
            [
                "rev-parse",
                "--verify",
                f"refs/heads/{project_branch}^{{commit}}",
            ],
            check=False,
        )
        commit_sha = result.stdout.strip()
        if result.returncode != 0 or not commit_sha:
            raise MergePhaseError(
                f"project base does not resolve: {project_branch!r}"
            )
        return commit_sha

    @staticmethod
    def _merge_head(git: SafeGit) -> str | None:
        result = git.run(
            ["rev-parse", "--verify", "--quiet", "MERGE_HEAD"], check=False
        )
        commit_sha = result.stdout.strip()
        return commit_sha if result.returncode == 0 and commit_sha else None

    def _merge_environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        environment.update(self._config.environment)
        for name, value in _MERGE_IDENTITY.items():
            environment.setdefault(name, value)
        return environment

    def _transition(
        self,
        context: ScheduledTaskContext,
        status: TaskRuntimeStatus,
        reason: str,
    ) -> HostMergeResult:
        if status not in {TaskRuntimeStatus.BLOCKED, TaskRuntimeStatus.FAILED}:
            raise MergePhaseError(f"invalid terminal merge status: {status.value}")
        context.store.transition_task_runtime(
            context.claim.run_id,
            context.owner_token,
            context.claim.id,
            context.claim.claim_token,
            expected_status=TaskRuntimeStatus.MERGING,
            new_status=status,
            resume_phase="merging",
            state_reason=reason,
            now=context.clock(),
        )
        return HostMergeResult(status, reason)

    def _block(
        self, context: ScheduledTaskContext, reason: str
    ) -> HostMergeResult:
        runtime = context.store.get_task_runtime(context.claim.task_id)
        if runtime is None:
            raise MergePhaseError(reason)
        if runtime.status is TaskRuntimeStatus.BLOCKED:
            return HostMergeResult(TaskRuntimeStatus.BLOCKED, reason)
        if runtime.status is not TaskRuntimeStatus.MERGING:
            raise MergePhaseError(reason)
        return self._transition(context, TaskRuntimeStatus.BLOCKED, reason)


def _render_merge_prompt(
    inputs: VerifiedTaskInputs,
    *,
    task_branch: str,
    project_branch: str,
    approved_commit: str,
    base_commit: str,
    unresolved: tuple[str, ...],
) -> str:
    sections = [
        "Resolve the active Git merge in the current task worktree.",
        "Preserve the intent of both the approved task commit and current project "
        "base. Resolve every conflicted path, stage the resolutions, and create "
        "the merge commit before returning completed. Do not abort the merge, "
        "switch branches, or modify the primary checkout.",
        "",
        f"Task branch: {task_branch}",
        f"Project branch: {project_branch}",
        f"Approved task commit: {approved_commit}",
        f"Project base commit: {base_commit}",
        "Unresolved paths:",
        *(f"- {path}" for path in unresolved),
        "",
        f"Task file: {inputs.task_path.as_posix()}",
        f"Task digest: {inputs.task.digest}",
        "",
        "## Assigned task",
        "",
        inputs.task_markdown.rstrip(),
    ]
    if inputs.dependencies:
        sections.extend(["", "## Dependency tasks"])
        for task, path, markdown in inputs.dependencies:
            sections.extend(
                [
                    "",
                    f"### {task.task_ref}",
                    f"Dependency file: {path.as_posix()}",
                    f"Dependency digest: {task.digest}",
                    "",
                    markdown.rstrip(),
                ]
            )
    return "\n".join(sections).rstrip() + "\n"


__all__ = [
    "HostMergeConfig",
    "HostMergePhase",
    "HostMergeResult",
    "MERGE_RESULT_SCHEMA",
    "MergePhaseError",
    "MergeTip",
]
