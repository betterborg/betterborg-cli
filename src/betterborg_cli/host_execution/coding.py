"""Digest-bound coding-agent execution in claimed host worktrees."""

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

CODING_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "task_file": {"type": "string", "minLength": 1},
        "status": {
            "type": "string",
            "enum": [
                "completed",
                "blocked",
                "partial",
                "failed",
                "environment_refresh_required",
            ],
        },
        "summary": {"type": "string", "minLength": 1},
        "changed_files": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "tests_run": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "follow_ups": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "blockers": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "environment_refresh": {"type": ["object", "null"]},
    },
    "required": [
        "task_file",
        "status",
        "summary",
        "changed_files",
        "tests_run",
        "follow_ups",
        "blockers",
    ],
}


class CodingPhaseError(RuntimeError):
    """Raised when a claimed task is not safe or ready for coding."""


@dataclass(frozen=True, slots=True)
class HostCodingConfig:
    """Provider and artifact settings for coding-agent attempts."""

    model: str
    billing_mode: BillingMode = BillingMode.API
    effort: str | None = None
    allowed_tools: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)
    artifact_root: Path | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("coding model must not be empty")
        object.__setattr__(self, "billing_mode", BillingMode(self.billing_mode))
        object.__setattr__(self, "allowed_tools", tuple(self.allowed_tools))
        if self.artifact_root is not None:
            object.__setattr__(self, "artifact_root", Path(self.artifact_root))


class HostCodingPhase:
    """Run one coding attempt without claiming or preparing task work."""

    def __init__(
        self,
        repository_root: Path,
        adapter: AgentAdapter,
        *,
        config: HostCodingConfig,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self._paths = RepoPaths.discover(self.repository_root)
        if self._paths.root != self.repository_root:
            raise CodingPhaseError(
                "coding phase must be bound to the primary Git checkout"
            )
        self._adapter = adapter
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
            raise CodingPhaseError(
                "repository-local coding artifacts must be under .borg/state"
            )

    def run(
        self,
        context: ScheduledTaskContext,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> TaskRuntimeStatus:
        """Invoke coding once, persist its outcome, and require a new commit."""
        try:
            runtime, worktree = self._require_ready_worktree(context)
            inputs = self._verified_inputs(context, worktree)
        except (
            CodingPhaseError,
            HostAgentPhaseError,
            TaskDigestDriftError,
            OSError,
        ) as error:
            return self._block(context, str(error) or error.__class__.__name__)

        resumed = self._resume_completed_attempt(context, worktree)
        if resumed is not None:
            return resumed

        worktree_git = SafeGit(worktree)
        base_head = worktree_git.head_sha()
        attempt_number = 1 + sum(
            attempt.phase == "coding"
            for attempt in context.store.list_agent_attempts(context.claim.task_id)
        )
        attempt_id = uuid4()
        attempt_dir = (
            self.artifact_root
            / str(context.claim.task_id)
            / f"coding-{attempt_number:03d}-{attempt_id.hex}"
        )
        try:
            attempt_dir.mkdir(parents=True, exist_ok=False)
            artifacts = AgentAttemptArtifacts(
                self.repository_root, attempt_dir, worktree, "coding"
            )
            user_prompt = _render_user_prompt(inputs)
            artifacts.write_text("system-prompt.md", inputs.system_prompt)
            artifacts.write_text("user-prompt.md", user_prompt)
            artifacts.write_text(
                "result-schema.json",
                json.dumps(CODING_RESULT_SCHEMA, indent=2, sort_keys=True) + "\n",
            )
        except OSError as error:
            return self._block(context, f"unable to create coding artifacts: {error}")

        log_path = attempt_dir / "coding.log"
        result_path = attempt_dir / "coding.result.json"
        started_at = context.clock()
        attempt = AgentAttempt(
            id=attempt_id,
            run_id=context.claim.run_id,
            claim_id=context.claim.id,
            task_id=context.claim.task_id,
            phase="coding",
            attempt_number=attempt_number,
            adapter=self._adapter.name,
            model=self._config.model,
            billing_mode=self._config.billing_mode,
            status=ExecutionAttemptStatus.RUNNING,
            log_path=self._artifact_ref(log_path),
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
            schema=CODING_RESULT_SCHEMA,
            cwd=worktree,
            model=self._config.model,
            log_path=log_path,
            result_path=result_path,
            allowed_tools=self._config.allowed_tools,
            env={**self._config.environment, **(environment or {})},
            effort=self._config.effort,
            billing_mode=self._config.billing_mode,
        )
        result: AgentResult
        operational_error: BaseException | None = None
        try:
            with self._guard.protect(inputs.task.task_ref, "coding"):
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

        try:
            final_head = worktree_git.head_sha()
            branch = current_branch(worktree_git)
        except BaseException as error:
            operational_error = operational_error or error
            final_head = base_head
            branch = runtime.branch or ""

        outcome = self._classify(
            result,
            base_head=base_head,
            final_head=final_head,
            expected_branch=runtime.branch or "",
            actual_branch=branch,
            git=worktree_git,
            operational_error=operational_error,
        )
        durable_result = dict(result.payload or {})
        durable_result["_betterborg"] = {
            "artifact_dir": self._artifact_ref(attempt_dir),
            "base_commit": base_head,
            "commit_sha": final_head if final_head != base_head else None,
            "outcome_status": outcome[0].value,
            "outcome_reason": outcome[1],
            "provider": result.provider or self._adapter.name,
            "model": result.model or self._config.model,
            "billing_mode": result.billing_mode.value,
        }
        try:
            artifacts.finish(result, durable_result)
        except OSError as error:
            outcome = (
                TaskRuntimeStatus.BLOCKED,
                f"artifact persistence failed: {error}",
            )
            durable_result["_betterborg"]["outcome_status"] = outcome[0].value
            durable_result["_betterborg"]["outcome_reason"] = outcome[1]

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
                self._artifact_ref(result_path) if result_path.is_file() else None
            ),
            result=durable_result,
            summary=result_summary(result),
            duration_seconds=result.duration_seconds,
            usage=result.usage,
            now=context.clock(),
        )
        return self._transition(context, outcome[0], outcome[1])

    def _require_ready_worktree(
        self, context: ScheduledTaskContext
    ) -> tuple[TaskRuntime, Path]:
        return require_ready_worktree(
            self.repository_root,
            self._primary_git,
            context,
            expected_statuses={TaskRuntimeStatus.CODING},
        )

    def _verified_inputs(
        self, context: ScheduledTaskContext, worktree: Path
    ) -> VerifiedTaskInputs:
        return verified_task_inputs(
            self.repository_root,
            context,
            worktree,
            prompt_role="coding",
        )

    def _resume_completed_attempt(
        self, context: ScheduledTaskContext, worktree: Path
    ) -> TaskRuntimeStatus | None:
        completed = [
            attempt
            for attempt in context.store.list_agent_attempts(context.claim.task_id)
            if attempt.phase == "coding"
            and attempt.status is ExecutionAttemptStatus.COMPLETED
        ]
        if not completed:
            return None
        latest = completed[-1]
        result = latest.result or {}
        metadata = result.get("_betterborg")
        if not isinstance(metadata, Mapping):
            return self._block(
                context,
                "completed coding attempt lacks durable commit attestation; "
                "refusing to replay it",
            )
        status_value = metadata.get("outcome_status")
        reason = str(metadata.get("outcome_reason") or "resumed coding outcome")
        try:
            status = TaskRuntimeStatus(str(status_value))
        except ValueError:
            return self._block(context, "completed coding attempt has invalid outcome")
        if status is TaskRuntimeStatus.REVIEW:
            commit_sha = metadata.get("commit_sha")
            head_matches = (
                isinstance(commit_sha, str)
                and SafeGit(worktree).head_sha() == commit_sha
            )
            if not head_matches:
                return self._block(
                    context, "completed coding commit no longer matches task worktree"
                )
        return self._transition(context, status, reason)

    @staticmethod
    def _classify(
        result: AgentResult,
        *,
        base_head: str,
        final_head: str,
        expected_branch: str,
        actual_branch: str,
        git: SafeGit,
        operational_error: BaseException | None,
    ) -> tuple[TaskRuntimeStatus, str]:
        if operational_error is not None:
            return TaskRuntimeStatus.BLOCKED, str(operational_error)
        if actual_branch != expected_branch:
            return TaskRuntimeStatus.BLOCKED, "coding agent changed the task branch"
        if result.status is AgentStatus.CANCELLED:
            return TaskRuntimeStatus.BLOCKED, "coding agent was interrupted"
        if result.status is AgentStatus.FAILED:
            return TaskRuntimeStatus.FAILED, result.error or "coding agent failed"
        payload_status = (result.payload or {}).get("status")
        if payload_status != "completed":
            status = (
                TaskRuntimeStatus.FAILED
                if payload_status == "failed"
                else TaskRuntimeStatus.BLOCKED
            )
            return status, f"coding agent reported {payload_status or 'no status'}"
        if final_head == base_head:
            return (
                TaskRuntimeStatus.BLOCKED,
                "coding completed without producing a commit; worktree preserved",
            )
        if not git.is_ancestor(base_head, final_head):
            return (
                TaskRuntimeStatus.BLOCKED,
                "coding commit does not descend from the claimed worktree HEAD",
            )
        return TaskRuntimeStatus.REVIEW, f"coding committed {final_head}"

    def _artifact_ref(self, path: Path) -> str:
        return AgentAttemptArtifacts(
            self.repository_root, path.parent, self.repository_root, "coding"
        ).reference(path)

    def _transition(
        self,
        context: ScheduledTaskContext,
        status: TaskRuntimeStatus,
        reason: str,
    ) -> TaskRuntimeStatus:
        if status is TaskRuntimeStatus.CODING:
            return status
        context.store.transition_task_runtime(
            context.claim.run_id,
            context.owner_token,
            context.claim.id,
            context.claim.claim_token,
            expected_status=TaskRuntimeStatus.CODING,
            new_status=status,
            state_reason=reason,
            now=context.clock(),
        )
        return status

    def _block(self, context: ScheduledTaskContext, reason: str) -> TaskRuntimeStatus:
        runtime = context.store.get_task_runtime(context.claim.task_id)
        if runtime is None:
            raise CodingPhaseError(reason)
        if runtime.status is TaskRuntimeStatus.BLOCKED:
            return TaskRuntimeStatus.BLOCKED
        if runtime.status not in {
            TaskRuntimeStatus.CLAIMED,
            TaskRuntimeStatus.ENVIRONMENT,
            TaskRuntimeStatus.CODING,
        }:
            raise CodingPhaseError(reason)
        context.store.transition_task_runtime(
            context.claim.run_id,
            context.owner_token,
            context.claim.id,
            context.claim.claim_token,
            expected_status=runtime.status,
            new_status=TaskRuntimeStatus.BLOCKED,
            state_reason=reason,
            now=context.clock(),
        )
        return TaskRuntimeStatus.BLOCKED


def _render_user_prompt(inputs: VerifiedTaskInputs) -> str:
    sections = [
        "Implement the assigned task in the current worktree. Commit all required "
        "changes before returning completed.",
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
    "CODING_RESULT_SCHEMA",
    "CodingPhaseError",
    "HostCodingConfig",
    "HostCodingPhase",
]
