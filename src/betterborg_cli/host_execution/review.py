"""Durable review and fix loop for completed host coding attempts."""

from __future__ import annotations

import json
from collections.abc import Mapping
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
    current_branch,
    require_ready_worktree,
    result_summary,
    verified_task_inputs,
)
from betterborg_cli.host_execution.coding import CODING_RESULT_SCHEMA
from betterborg_cli.host_execution.git import SafeGit
from betterborg_cli.host_execution.guard import PrimaryCheckoutGuard
from betterborg_cli.host_execution.scheduler import ScheduledTaskContext
from betterborg_cli.planning import TaskDigestDriftError
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.store import (
    AgentAttempt,
    ExecutionAttemptStatus,
    TaskRuntime,
    TaskRuntimeStatus,
)

REVIEW_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "task_file": {"type": "string", "minLength": 1},
        "status": {
            "type": "string",
            "enum": [
                "approved",
                "issues_found",
                "failed",
                "environment_refresh_required",
            ],
        },
        "summary": {"type": "string", "minLength": 1},
        "issues_file": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "environment_refresh": {"type": ["object", "null"]},
    },
    "required": ["task_file", "status", "summary", "issues_file", "findings"],
}


class ReviewFixPhaseError(RuntimeError):
    """Raised when the review/fix lifecycle cannot safely proceed."""


@dataclass(frozen=True, slots=True)
class HostReviewFixConfig:
    """Provider, artifact, and pass-limit settings for review and fixes."""

    review_model: str
    fix_model: str | None = None
    review_passes: int = 3
    review_billing_mode: BillingMode = BillingMode.API
    fix_billing_mode: BillingMode | None = None
    review_effort: str | None = None
    fix_effort: str | None = None
    review_allowed_tools: tuple[str, ...] = ()
    fix_allowed_tools: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)
    artifact_root: Path | None = None

    def __post_init__(self) -> None:
        if not self.review_model.strip():
            raise ValueError("review model must not be empty")
        if self.fix_model is not None and not self.fix_model.strip():
            raise ValueError("fix model must not be empty")
        if self.review_passes < 1:
            raise ValueError("review passes must be positive")
        object.__setattr__(
            self, "review_billing_mode", BillingMode(self.review_billing_mode)
        )
        if self.fix_billing_mode is not None:
            object.__setattr__(
                self, "fix_billing_mode", BillingMode(self.fix_billing_mode)
            )
        object.__setattr__(
            self, "review_allowed_tools", tuple(self.review_allowed_tools)
        )
        object.__setattr__(
            self, "fix_allowed_tools", tuple(self.fix_allowed_tools)
        )
        if self.artifact_root is not None:
            object.__setattr__(self, "artifact_root", Path(self.artifact_root))

    @property
    def resolved_fix_model(self) -> str:
        return self.fix_model or self.review_model

    @property
    def resolved_fix_billing_mode(self) -> BillingMode:
        return self.fix_billing_mode or self.review_billing_mode


class HostReviewFixPhase:
    """Review a coding commit and run bounded, commit-producing fix turns."""

    def __init__(
        self,
        repository_root: Path,
        review_adapter: AgentAdapter,
        *,
        config: HostReviewFixConfig,
        fix_adapter: AgentAdapter | None = None,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self._paths = RepoPaths.discover(self.repository_root)
        if self._paths.root != self.repository_root:
            raise ReviewFixPhaseError(
                "review phase must be bound to the primary Git checkout"
            )
        self._review_adapter = review_adapter
        self._fix_adapter = fix_adapter or review_adapter
        self._config = config
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
            raise ReviewFixPhaseError(
                "repository-local review artifacts must be under .borg/state"
            )

    def run(
        self,
        context: ScheduledTaskContext,
        *,
        environment: Mapping[str, str] | None = None,
        review_environment: Mapping[str, str] | None = None,
        fix_environment: Mapping[str, str] | None = None,
    ) -> TaskRuntimeStatus:
        """Drive REVIEW/FIX until approval, a cap, or another durable stop."""
        while True:
            try:
                runtime, worktree = require_ready_worktree(
                    self.repository_root,
                    self._primary_git,
                    context,
                    expected_statuses={
                        TaskRuntimeStatus.REVIEW,
                        TaskRuntimeStatus.FIX,
                    },
                )
                inputs = verified_task_inputs(
                    self.repository_root,
                    context,
                    worktree,
                    prompt_role=(
                        "review"
                        if runtime.status is TaskRuntimeStatus.REVIEW
                        else "coding"
                    ),
                )
                base_commit, current_commit = self._declared_commits(
                    context, worktree
                )
            except (
                HostAgentPhaseError,
                ReviewFixPhaseError,
                TaskDigestDriftError,
                OSError,
            ) as error:
                return self._block(
                    context, str(error) or error.__class__.__name__
                )

            resumed = self._resume_completed_attempt(
                context, runtime, worktree
            )
            if resumed is not None:
                if resumed not in {
                    TaskRuntimeStatus.REVIEW,
                    TaskRuntimeStatus.FIX,
                }:
                    return resumed
                continue

            if runtime.status is TaskRuntimeStatus.REVIEW:
                status = self._run_review(
                    context,
                    runtime,
                    worktree,
                    inputs,
                    base_commit=base_commit,
                    current_commit=current_commit,
                    environment={
                        **(environment or {}),
                        **(review_environment or {}),
                    },
                )
            else:
                status = self._run_fix(
                    context,
                    runtime,
                    worktree,
                    inputs,
                    base_commit=base_commit,
                    current_commit=current_commit,
                    environment={
                        **(environment or {}),
                        **(fix_environment or {}),
                    },
                )
            if status not in {TaskRuntimeStatus.REVIEW, TaskRuntimeStatus.FIX}:
                return status

    def _run_review(
        self,
        context: ScheduledTaskContext,
        runtime: TaskRuntime,
        worktree: Path,
        inputs: VerifiedTaskInputs,
        *,
        base_commit: str,
        current_commit: str,
        environment: Mapping[str, str] | None,
    ) -> TaskRuntimeStatus:
        user_prompt = _render_review_prompt(
            inputs,
            branch=runtime.branch or "",
            base_commit=base_commit,
            current_commit=current_commit,
            review_round=runtime.review_round,
        )
        return self._invoke(
            context,
            runtime,
            worktree,
            phase="review",
            adapter=self._review_adapter,
            model=self._config.review_model,
            billing_mode=self._config.review_billing_mode,
            effort=self._config.review_effort,
            allowed_tools=self._config.review_allowed_tools,
            schema=REVIEW_RESULT_SCHEMA,
            system_prompt=inputs.system_prompt,
            user_prompt=user_prompt,
            base_commit=base_commit,
            current_commit=current_commit,
            environment=environment,
        )

    def _run_fix(
        self,
        context: ScheduledTaskContext,
        runtime: TaskRuntime,
        worktree: Path,
        inputs: VerifiedTaskInputs,
        *,
        base_commit: str,
        current_commit: str,
        environment: Mapping[str, str] | None,
    ) -> TaskRuntimeStatus:
        findings = self._findings_for_fix(context, runtime.review_round)
        user_prompt = _render_fix_prompt(
            inputs,
            findings=findings,
            review_round=runtime.review_round,
        )
        return self._invoke(
            context,
            runtime,
            worktree,
            phase="fix",
            adapter=self._fix_adapter,
            model=self._config.resolved_fix_model,
            billing_mode=self._config.resolved_fix_billing_mode,
            effort=self._config.fix_effort,
            allowed_tools=self._config.fix_allowed_tools,
            schema=CODING_RESULT_SCHEMA,
            system_prompt=inputs.system_prompt,
            user_prompt=user_prompt,
            base_commit=base_commit,
            current_commit=current_commit,
            environment=environment,
        )

    def _invoke(
        self,
        context: ScheduledTaskContext,
        runtime: TaskRuntime,
        worktree: Path,
        *,
        phase: str,
        adapter: AgentAdapter,
        model: str,
        billing_mode: BillingMode,
        effort: str | None,
        allowed_tools: tuple[str, ...],
        schema: Mapping[str, Any],
        system_prompt: str,
        user_prompt: str,
        base_commit: str,
        current_commit: str,
        environment: Mapping[str, str] | None,
    ) -> TaskRuntimeStatus:
        attempts = context.store.list_agent_attempts(context.claim.task_id)
        attempt_number = 1 + sum(item.phase == phase for item in attempts)
        attempt_id = uuid4()
        attempt_dir = (
            self.artifact_root
            / str(context.claim.task_id)
            / f"{phase}-{attempt_number:03d}-{attempt_id.hex}"
        )
        artifacts = AgentAttemptArtifacts(
            self.repository_root, attempt_dir, worktree, phase
        )
        try:
            attempt_dir.mkdir(parents=True, exist_ok=False)
            artifacts.write_text("system-prompt.md", system_prompt)
            artifacts.write_text("user-prompt.md", user_prompt)
            artifacts.write_text(
                "result-schema.json",
                json.dumps(schema, indent=2, sort_keys=True) + "\n",
            )
        except OSError as error:
            return self._block(
                context, f"unable to create {phase} artifacts: {error}"
            )

        log_path = attempt_dir / f"{phase}.log"
        result_path = attempt_dir / f"{phase}.result.json"
        started_at = context.clock()
        attempt = AgentAttempt(
            id=attempt_id,
            run_id=context.claim.run_id,
            claim_id=context.claim.id,
            task_id=context.claim.task_id,
            phase=phase,
            review_round=runtime.review_round,
            attempt_number=attempt_number,
            adapter=adapter.name,
            model=model,
            billing_mode=billing_mode,
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
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            cwd=worktree,
            model=model,
            log_path=log_path,
            result_path=result_path,
            allowed_tools=allowed_tools,
            env={**self._config.environment, **(environment or {})},
            effort=effort,
            billing_mode=billing_mode,
        )
        git = SafeGit(worktree)
        before_status = git.run(
            ["status", "--porcelain=v1", "-z", "-uall"]
        ).stdout
        operational_error: BaseException | None = None
        try:
            with self._guard.protect(str(context.claim.task_id), phase):
                result = adapter.run(spec, cancel=context.cancel)
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
                billing_mode=billing_mode,
                provider=adapter.name,
                model=model,
            )
        try:
            final_commit = git.head_sha()
            actual_branch = current_branch(git)
            after_status = git.run(
                ["status", "--porcelain=v1", "-z", "-uall"]
            ).stdout
        except BaseException as error:
            operational_error = operational_error or error
            final_commit = current_commit
            actual_branch = runtime.branch or ""
            after_status = before_status

        if phase == "review":
            outcome = self._classify_review(
                result,
                runtime=runtime,
                expected_commit=current_commit,
                final_commit=final_commit,
                expected_branch=runtime.branch or "",
                actual_branch=actual_branch,
                before_status=before_status,
                after_status=after_status,
                operational_error=operational_error,
            )
        else:
            outcome = self._classify_fix(
                result,
                runtime=runtime,
                previous_commit=current_commit,
                final_commit=final_commit,
                expected_branch=runtime.branch or "",
                actual_branch=actual_branch,
                git=git,
                after_status=after_status,
                operational_error=operational_error,
            )

        durable_result = dict(result.payload or {})
        durable_result["_betterborg"] = {
            "artifact_dir": artifacts.reference(attempt_dir),
            "base_commit": base_commit,
            "prior_commit": current_commit,
            "commit_sha": final_commit,
            "outcome_status": outcome.status.value,
            "outcome_reason": outcome.reason,
            "review_round": runtime.review_round,
            "next_review_round": outcome.review_round,
            "provider": result.provider or adapter.name,
            "model": result.model or model,
            "billing_mode": result.billing_mode.value,
        }
        try:
            artifacts.finish(result, durable_result)
        except OSError as error:
            outcome = _PhaseOutcome(
                TaskRuntimeStatus.BLOCKED,
                f"artifact persistence failed: {error}",
                runtime.review_round,
                runtime.status.value,
            )
            durable_result["_betterborg"]["outcome_status"] = outcome.status.value
            durable_result["_betterborg"]["outcome_reason"] = outcome.reason
            durable_result["_betterborg"]["next_review_round"] = (
                outcome.review_round
            )

        terminal_attempt_status = {
            AgentStatus.COMPLETED: ExecutionAttemptStatus.COMPLETED,
            AgentStatus.CANCELLED: ExecutionAttemptStatus.CANCELLED,
            AgentStatus.FAILED: ExecutionAttemptStatus.FAILED,
        }[result.status]
        context.store.complete_agent_attempt(
            attempt.id,
            context.owner_token,
            context.claim.claim_token,
            status=terminal_attempt_status,
            result_path=(
                artifacts.reference(result_path) if result_path.is_file() else None
            ),
            result=durable_result,
            summary=result_summary(result),
            duration_seconds=result.duration_seconds,
            usage=result.usage,
            now=context.clock(),
        )
        return self._transition(context, runtime.status, outcome)

    def _classify_review(
        self,
        result: AgentResult,
        *,
        runtime: TaskRuntime,
        expected_commit: str,
        final_commit: str,
        expected_branch: str,
        actual_branch: str,
        before_status: str,
        after_status: str,
        operational_error: BaseException | None,
    ) -> _PhaseOutcome:
        if operational_error is not None:
            return _blocked_outcome(runtime, str(operational_error))
        if actual_branch != expected_branch or final_commit != expected_commit:
            return _blocked_outcome(runtime, "review agent changed the task branch")
        if after_status != before_status:
            return _blocked_outcome(runtime, "review agent modified the task worktree")
        if result.status is AgentStatus.CANCELLED:
            return _blocked_outcome(runtime, "review agent was interrupted")
        if result.status is AgentStatus.FAILED:
            return _PhaseOutcome(
                TaskRuntimeStatus.FAILED,
                result.error or "review agent failed",
                runtime.review_round,
                "review",
            )
        payload = result.payload or {}
        payload_status = payload.get("status")
        findings = payload.get("findings")
        if payload_status == "approved":
            if findings:
                return _blocked_outcome(
                    runtime, "review approval included unresolved findings"
                )
            return _PhaseOutcome(
                TaskRuntimeStatus.MERGING,
                str(payload.get("summary") or "review approved"),
                runtime.review_round,
                "merging",
            )
        if payload_status == "issues_found":
            if not isinstance(findings, list) or not findings:
                return _blocked_outcome(
                    runtime, "review reported issues without findings"
                )
            next_round = runtime.review_round + 1
            if next_round >= self._config.review_passes:
                return _PhaseOutcome(
                    TaskRuntimeStatus.BLOCKED,
                    f"review pass limit {self._config.review_passes} reached",
                    next_round,
                    "review",
                )
            return _PhaseOutcome(
                TaskRuntimeStatus.FIX,
                f"review round {next_round} requested fixes",
                next_round,
                "fix",
            )
        if payload_status == "failed":
            return _PhaseOutcome(
                TaskRuntimeStatus.FAILED,
                str(payload.get("summary") or "review agent could not review"),
                runtime.review_round,
                "review",
            )
        return _blocked_outcome(
            runtime, f"review agent reported {payload_status or 'no status'}"
        )

    @staticmethod
    def _classify_fix(
        result: AgentResult,
        *,
        runtime: TaskRuntime,
        previous_commit: str,
        final_commit: str,
        expected_branch: str,
        actual_branch: str,
        git: SafeGit,
        after_status: str,
        operational_error: BaseException | None,
    ) -> _PhaseOutcome:
        if operational_error is not None:
            return _PhaseOutcome(
                TaskRuntimeStatus.BLOCKED,
                str(operational_error),
                runtime.review_round,
                "fix",
            )
        if actual_branch != expected_branch:
            reason = "fix agent changed the task branch"
        elif result.status is AgentStatus.CANCELLED:
            reason = "fix agent was interrupted"
        elif result.status is AgentStatus.FAILED:
            return _PhaseOutcome(
                TaskRuntimeStatus.FAILED,
                result.error or "fix agent failed",
                runtime.review_round,
                "fix",
            )
        elif (result.payload or {}).get("status") != "completed":
            payload_status = (result.payload or {}).get("status")
            if payload_status == "failed":
                return _PhaseOutcome(
                    TaskRuntimeStatus.FAILED,
                    "fix agent reported failed",
                    runtime.review_round,
                    "fix",
                )
            reason = f"fix agent reported {payload_status or 'no status'}"
        elif final_commit == previous_commit:
            reason = "fix completed without producing a commit; worktree preserved"
        elif not git.is_ancestor(previous_commit, final_commit):
            reason = "fix commit does not descend from the reviewed commit"
        elif after_status:
            reason = "fix agent left uncommitted work after its commit"
        else:
            return _PhaseOutcome(
                TaskRuntimeStatus.REVIEW,
                f"fix committed {final_commit}",
                runtime.review_round,
                "review",
            )
        return _PhaseOutcome(
            TaskRuntimeStatus.BLOCKED,
            reason,
            runtime.review_round,
            "fix",
        )

    def _resume_completed_attempt(
        self,
        context: ScheduledTaskContext,
        runtime: TaskRuntime,
        worktree: Path,
    ) -> TaskRuntimeStatus | None:
        phase = runtime.status.value
        completed = [
            attempt
            for attempt in context.store.list_agent_attempts(context.claim.task_id)
            if attempt.phase == phase
            and attempt.review_round == runtime.review_round
            and attempt.status is ExecutionAttemptStatus.COMPLETED
        ]
        if not completed:
            return None
        metadata = (completed[-1].result or {}).get("_betterborg")
        if not isinstance(metadata, Mapping):
            return self._block(
                context,
                f"completed {phase} attempt lacks durable outcome; refusing replay",
            )
        try:
            outcome = _PhaseOutcome(
                status=TaskRuntimeStatus(str(metadata.get("outcome_status"))),
                reason=str(
                    metadata.get("outcome_reason") or f"resumed {phase} outcome"
                ),
                review_round=int(
                    metadata.get("next_review_round", runtime.review_round)
                ),
                resume_phase=str(metadata.get("outcome_status") or phase),
            )
        except (TypeError, ValueError):
            return self._block(
                context, f"completed {phase} attempt has invalid durable outcome"
            )
        if outcome.status in {TaskRuntimeStatus.REVIEW, TaskRuntimeStatus.MERGING}:
            commit_sha = metadata.get("commit_sha")
            if (
                not isinstance(commit_sha, str)
                or SafeGit(worktree).head_sha() != commit_sha
            ):
                return self._block(
                    context,
                    f"completed {phase} commit no longer matches task worktree",
                )
        if outcome.status is TaskRuntimeStatus.BLOCKED:
            outcome = _PhaseOutcome(
                outcome.status,
                outcome.reason,
                outcome.review_round,
                phase,
            )
        return self._transition(context, runtime.status, outcome)

    def _declared_commits(
        self, context: ScheduledTaskContext, worktree: Path
    ) -> tuple[str, str]:
        attestations: list[tuple[str, str]] = []
        for attempt in context.store.list_agent_attempts(context.claim.task_id):
            if attempt.phase not in {"coding", "fix"}:
                continue
            metadata = (attempt.result or {}).get("_betterborg")
            if not isinstance(metadata, Mapping):
                continue
            base_commit = metadata.get("base_commit")
            commit_sha = metadata.get("commit_sha")
            if isinstance(base_commit, str) and isinstance(commit_sha, str):
                attestations.append((base_commit, commit_sha))
        if not attestations:
            raise ReviewFixPhaseError(
                "review requires a completed coding commit attestation"
            )
        base_commit = attestations[0][0]
        current_commit = attestations[-1][1]
        git = SafeGit(worktree)
        if git.head_sha() != current_commit:
            raise ReviewFixPhaseError(
                "declared coding/fix commit no longer matches task worktree"
            )
        if not git.is_ancestor(base_commit, current_commit):
            raise ReviewFixPhaseError(
                "task commit does not descend from its declared coding base"
            )
        if not git.is_clean():
            raise ReviewFixPhaseError(
                "task worktree has uncommitted changes outside its declared commit"
            )
        return base_commit, current_commit

    def _findings_for_fix(
        self, context: ScheduledTaskContext, review_round: int
    ) -> tuple[str, ...]:
        reviews = [
            attempt
            for attempt in context.store.list_agent_attempts(context.claim.task_id)
            if attempt.phase == "review"
            and attempt.status is ExecutionAttemptStatus.COMPLETED
            and isinstance(attempt.result, Mapping)
            and (attempt.result or {}).get("status") == "issues_found"
        ]
        if not reviews:
            raise ReviewFixPhaseError(
                f"fix round {review_round} has no persisted review findings"
            )
        findings = (reviews[-1].result or {}).get("findings")
        if not isinstance(findings, list) or not all(
            isinstance(finding, str) and finding.strip() for finding in findings
        ):
            raise ReviewFixPhaseError("persisted review findings are invalid")
        return tuple(findings)

    def _transition(
        self,
        context: ScheduledTaskContext,
        expected_status: TaskRuntimeStatus,
        outcome: _PhaseOutcome,
    ) -> TaskRuntimeStatus:
        context.store.transition_task_runtime(
            context.claim.run_id,
            context.owner_token,
            context.claim.id,
            context.claim.claim_token,
            expected_status=expected_status,
            new_status=outcome.status,
            resume_phase=outcome.resume_phase,
            review_round=outcome.review_round,
            state_reason=outcome.reason,
            now=context.clock(),
        )
        return outcome.status

    def _block(self, context: ScheduledTaskContext, reason: str) -> TaskRuntimeStatus:
        runtime = context.store.get_task_runtime(context.claim.task_id)
        if runtime is None:
            raise ReviewFixPhaseError(reason)
        if runtime.status is TaskRuntimeStatus.BLOCKED:
            return TaskRuntimeStatus.BLOCKED
        if runtime.status not in {TaskRuntimeStatus.REVIEW, TaskRuntimeStatus.FIX}:
            raise ReviewFixPhaseError(reason)
        return self._transition(
            context,
            runtime.status,
            _PhaseOutcome(
                TaskRuntimeStatus.BLOCKED,
                reason,
                runtime.review_round,
                runtime.status.value,
            ),
        )


@dataclass(frozen=True, slots=True)
class _PhaseOutcome:
    status: TaskRuntimeStatus
    reason: str
    review_round: int
    resume_phase: str


def _blocked_outcome(runtime: TaskRuntime, reason: str) -> _PhaseOutcome:
    return _PhaseOutcome(
        TaskRuntimeStatus.BLOCKED,
        reason,
        runtime.review_round,
        runtime.status.value,
    )


def _render_review_prompt(
    inputs: VerifiedTaskInputs,
    *,
    branch: str,
    base_commit: str,
    current_commit: str,
    review_round: int,
) -> str:
    sections = [
        "Review the assigned implementation without modifying the worktree.",
        "Compare the declared base commit with the current task commit and return "
        "only the required structured result.",
        "",
        f"Task file: {inputs.task_path.as_posix()}",
        f"Task digest: {inputs.task.digest}",
        f"Task branch: {branch}",
        f"Declared base commit: {base_commit}",
        f"Current task commit: {current_commit}",
        f"Review round: {review_round}",
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


def _render_fix_prompt(
    inputs: VerifiedTaskInputs,
    *,
    findings: tuple[str, ...],
    review_round: int,
) -> str:
    sections = [
        "Fix every persisted review finding in the current worktree. Keep the "
        "change in scope, run relevant verification, and commit the fix before "
        "returning completed.",
        "",
        f"Task file: {inputs.task_path.as_posix()}",
        f"Task digest: {inputs.task.digest}",
        f"Fix round: {review_round}",
        "",
        "## Review findings",
        "",
        *(f"- {finding}" for finding in findings),
        "",
        "## Assigned task",
        "",
        inputs.task_markdown.rstrip(),
    ]
    return "\n".join(sections).rstrip() + "\n"


__all__ = [
    "HostReviewFixConfig",
    "HostReviewFixPhase",
    "REVIEW_RESULT_SCHEMA",
    "ReviewFixPhaseError",
]
