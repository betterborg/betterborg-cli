"""Durable review and fix loop for completed host coding attempts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from betterborg_cli.agent_runtime import (
    AgentAdapter,
    AgentResult,
    AgentRunSpec,
    AgentStatus,
    BillingMode,
    CancellationToken,
)
from betterborg_cli.host_execution._agent_phase import (
    EXISTING_TEST_REVIEW_RULE,
    EXISTING_TEST_RULE,
    REVIEW_FINDING_RULE,
    AgentAttemptArtifacts,
    HostAgentPhaseError,
    VerifiedTaskInputs,
    cancelled_agent_reason,
    current_branch,
    require_ready_worktree,
    result_summary,
    verified_task_inputs,
)
from betterborg_cli.host_execution.coding import (
    CODING_RESULT_SCHEMA,
    reviewable_coding_statuses,
)
from betterborg_cli.host_execution.git import SafeGit
from betterborg_cli.host_execution.guard import PrimaryCheckoutGuard
from betterborg_cli.host_execution.scheduler import ScheduledTaskContext
from betterborg_cli.planning import TaskDigestDriftError
from betterborg_cli.planning.convergence import assess_convergence, drain_evidence
from betterborg_cli.planning.findings_ledger import (
    REPEATS_SCHEMA,
    RESOLVED_SCHEMA,
    open_execution_findings,
    open_findings,
    reconcile_execution_ledger,
)
from betterborg_cli.planning.grants import (
    EXECUTION_GRANT_BUDGET,
    TASK_REVIEW_LOOP,
    assess_grant,
    grant_account,
)
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_config import BlockedTaskPolicy
from betterborg_cli.store import (
    AgentAttempt,
    ExecutionAttemptStatus,
    ExecutionLedgerFinding,
    ReviewAssessment,
    TaskRuntime,
    TaskRuntimeStatus,
)

#: Severities that hold a task. A batch whose every fault is minor is one the
#: Supervisor approves, saying what the faults are, and a commit is held to the
#: same rule.
_HOLDING_SEVERITIES = frozenset({"blocker", "major"})

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
            ],
        },
        "summary": {"type": "string", "minLength": 1},
        "issues_file": {"type": "string"},
        "resolved": RESOLVED_SCHEMA,
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["severity", "message", "repeats"],
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["blocker", "major", "minor"],
                    },
                    "message": {
                        "type": "string",
                        "minLength": 1,
                        "pattern": r"\S",
                    },
                    "suggestion": {"type": "string"},
                    "repeats": REPEATS_SCHEMA,
                },
            },
        },
    },
    "required": [
        "task_file",
        "status",
        "summary",
        "issues_file",
        "findings",
        "resolved",
    ],
}


class ReviewFixPhaseError(RuntimeError):
    """Raised when the review/fix lifecycle cannot safely proceed."""


@dataclass(frozen=True, slots=True)
class HostReviewFixConfig:
    """Provider, artifact, and review-budget settings for review and fixes.

    ``review_passes`` is the minimum number of review rounds a task gets.
    ``grant_budget`` is how many rounds past it the loop may spend closing
    nothing before the task blocks.
    """

    review_model: str
    fix_model: str | None = None
    review_passes: int = 3
    grant_budget: int = EXECUTION_GRANT_BUDGET
    review_billing_mode: BillingMode = BillingMode.API
    fix_billing_mode: BillingMode | None = None
    review_effort: str | None = None
    fix_effort: str | None = None
    review_allowed_tools: tuple[str, ...] = ()
    fix_allowed_tools: tuple[str, ...] = ()
    blocked_tasks: BlockedTaskPolicy = BlockedTaskPolicy.STOP
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)
    artifact_root: Path | None = None

    def __post_init__(self) -> None:
        if not self.review_model.strip():
            raise ValueError("review model must not be empty")
        if self.fix_model is not None and not self.fix_model.strip():
            raise ValueError("fix model must not be empty")
        if self.review_passes < 1:
            raise ValueError("review passes must be positive")
        # Zero buys nothing and is legal; below zero would stop a task short of
        # the passes it was told to run.
        if self.grant_budget < 0:
            raise ValueError("review grant budget must not be negative")
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
    """Review a coding commit, fixing it while its grant budget holds."""

    def __init__(
        self,
        repository_root: Path,
        review_adapter: AgentAdapter,
        *,
        config: HostReviewFixConfig,
        fix_adapter: AgentAdapter | None = None,
        cancel: CancellationToken | None = None,
        git: SafeGit | None = None,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self._paths = RepoPaths.discover(self.repository_root, cancel=cancel)
        if self._paths.root != self.repository_root:
            raise ReviewFixPhaseError(
                "review phase must be bound to the primary Git checkout"
            )
        self._review_adapter = review_adapter
        self._fix_adapter = fix_adapter or review_adapter
        self._config = config
        if git is not None and git.cwd != self.repository_root:
            raise ReviewFixPhaseError(
                "review phase Git binding must match repository"
            )
        self._primary_git = git or SafeGit(self.repository_root, cancel=cancel)
        self._guard = PrimaryCheckoutGuard(
            self.repository_root, git=self._primary_git
        )
        self.artifact_root = Path(
            config.artifact_root
            or self._paths.artifacts_dir / "host-execution"
        ).resolve()
        if self.artifact_root.is_relative_to(self.repository_root) and not (
            self.artifact_root == self._paths.state_dir
            or self.artifact_root.is_relative_to(self._paths.state_dir)
        ):
            raise ReviewFixPhaseError(
                "repository-local review artifacts must be under .betterborg/state"
            )

    def run(
        self,
        context: ScheduledTaskContext,
        *,
        environment: Mapping[str, str] | None = None,
        review_environment: Mapping[str, str] | None = None,
        fix_environment: Mapping[str, str] | None = None,
    ) -> TaskRuntimeStatus:
        """Drive REVIEW/FIX to approval, a spent budget, or a durable stop."""
        while True:
            try:
                runtime, worktree = require_ready_worktree(
                    self._paths,
                    self._primary_git,
                    context,
                    expected_statuses={
                        TaskRuntimeStatus.REVIEW,
                        TaskRuntimeStatus.FIX,
                    },
                )
                inputs = verified_task_inputs(
                    self._paths,
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
                # Only a review records an assessment, so only a review pays
                # for finding out whose record it goes in.
                borg_id = (
                    _assessment_borg_id(context)
                    if runtime.status is TaskRuntimeStatus.REVIEW
                    else None
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
                    borg_id=borg_id,
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
            if context.cancel.is_set() or status not in {
                TaskRuntimeStatus.REVIEW,
                TaskRuntimeStatus.FIX,
            }:
                return status

    def _run_review(
        self,
        context: ScheduledTaskContext,
        runtime: TaskRuntime,
        worktree: Path,
        inputs: VerifiedTaskInputs,
        *,
        borg_id: UUID,
        base_commit: str,
        current_commit: str,
        environment: Mapping[str, str] | None,
    ) -> TaskRuntimeStatus:
        user_prompt = _render_review_prompt(
            inputs,
            branch=runtime.branch or "",
            base_commit=base_commit,
            current_commit=current_commit,
            review_round=_ledger_round(runtime),
            open_ledger=open_execution_findings(
                context.store, context.claim.task_id
            ),
            unfinished=_unfinished_coding_report(context),
        )
        return self._invoke(
            context,
            runtime,
            worktree,
            phase="review",
            borg_id=borg_id,
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
            # The review that raised these findings advanced the counter as it
            # requested the fix, so the runtime already carries that review's
            # ledger round and the round line agrees with the rows below it.
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
        # The review phase's alone: it is the only one that records an
        # assessment, and the only one given a Borg to record it under.
        borg_id: UUID | None = None,
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
            activity_sink=context.activity_sink(phase),
        )
        git = self._primary_git.for_worktree(worktree)
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
        cancellation_reason = cancelled_agent_reason(
            result,
            context.cancel,
            phase=phase,
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

        ledger: tuple[ExecutionLedgerFinding, ...] = ()
        assessment: ReviewAssessment | None = None
        if phase == "review":
            classified = self._classify_review(
                result,
                runtime=runtime,
                borg_id=borg_id,
                task_id=context.claim.task_id,
                attempt_id=attempt_id,
                ledger=context.store.list_execution_ledger_findings(
                    context.claim.task_id
                ),
                # Read here rather than in the classifier, which keeps every
                # read of this round's own history outside the durable step
                # that appends to it.
                recorded=context.store.list_review_assessments(
                    borg_id, loop=TASK_REVIEW_LOOP, task_id=context.claim.task_id
                ),
                expected_commit=current_commit,
                final_commit=final_commit,
                expected_branch=runtime.branch or "",
                actual_branch=actual_branch,
                before_status=before_status,
                after_status=after_status,
                operational_error=operational_error,
                cancellation_reason=cancellation_reason,
            )
            outcome = classified.outcome
            ledger = classified.ledger
            assessment = classified.assessment
        else:
            outcome = self._classify_fix(
                result,
                reviewable=reviewable_coding_statuses(
                    self._config.blocked_tasks
                ),
                runtime=runtime,
                previous_commit=current_commit,
                final_commit=final_commit,
                expected_branch=runtime.branch or "",
                actual_branch=actual_branch,
                git=git,
                after_status=after_status,
                operational_error=operational_error,
                cancellation_reason=cancellation_reason,
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
        # One durable step for the round's findings, the grant it spent and the
        # attempt that produced them: an interruption cannot leave objections
        # recorded against an attempt that never finished, and a resume replays
        # a completed attempt's outcome without re-running the classifier, so
        # rows written after it would never be written at all. The assessment
        # goes with them for the same reason: a round whose objections are
        # recorded without its snapshot denies the round after it the refund it
        # earned. A round the artifact write has already turned into a block has
        # no round after it, and records what it found all the same rather than
        # being carved out of the rule.
        with context.store.transaction():
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
            context.store.record_execution_ledger_findings(ledger)
            if assessment is not None:
                context.store.record_review_assessment(assessment)
        if outcome.status is runtime.status:
            return outcome.status
        return self._transition(context, runtime.status, outcome)

    def _classify_review(
        self,
        result: AgentResult,
        *,
        runtime: TaskRuntime,
        borg_id: UUID,
        task_id: UUID,
        attempt_id: UUID,
        ledger: Sequence[ExecutionLedgerFinding],
        recorded: Sequence[ReviewAssessment],
        expected_commit: str,
        final_commit: str,
        expected_branch: str,
        actual_branch: str,
        before_status: str,
        after_status: str,
        operational_error: BaseException | None,
        cancellation_reason: str | None,
    ) -> _ReviewClassification:
        if operational_error is not None:
            return _ReviewClassification(
                _blocked_outcome(runtime, str(operational_error))
            )
        if actual_branch != expected_branch or final_commit != expected_commit:
            return _ReviewClassification(
                _blocked_outcome(runtime, "review agent changed the task branch")
            )
        if after_status != before_status:
            return _ReviewClassification(
                _blocked_outcome(runtime, "review agent modified the task worktree")
            )
        if result.status is AgentStatus.CANCELLED:
            return _ReviewClassification(
                _PhaseOutcome(
                    TaskRuntimeStatus.REVIEW,
                    cancellation_reason or "review agent was interrupted",
                    runtime.review_round,
                    "review",
                )
            )
        if result.status is AgentStatus.FAILED:
            return _ReviewClassification(
                _PhaseOutcome(
                    TaskRuntimeStatus.FAILED,
                    result.error or "review agent failed",
                    runtime.review_round,
                    "review",
                )
            )
        payload = result.payload or {}
        payload_status = payload.get("status")
        if payload_status == "failed":
            return _ReviewClassification(
                _PhaseOutcome(
                    TaskRuntimeStatus.FAILED,
                    str(payload.get("summary") or "review agent could not review"),
                    runtime.review_round,
                    "review",
                )
            )
        if payload_status not in {"approved", "issues_found"}:
            return _ReviewClassification(
                _blocked_outcome(
                    runtime,
                    f"review agent reported {payload_status or 'no status'}",
                )
            )
        # The runtime counts the rounds behind it while the ledger numbers the
        # round in hand, so the count this review leaves behind is its own
        # ledger round. One number, and the rows, the reason and the assessment
        # all read it.
        review_round = _ledger_round(runtime)
        try:
            declared = _declared_findings(
                payload,
                task_id=task_id,
                attempt_id=attempt_id,
                review_round=review_round,
            )
            resolved = _declared_resolved(payload)
        except (TypeError, ValueError) as error:
            return _ReviewClassification(
                _blocked_outcome(runtime, f"review declarations are invalid: {error}")
            )
        approved = payload_status == "approved"
        if approved and any(
            finding.severity in _HOLDING_SEVERITIES for finding, _ in declared
        ):
            return _ReviewClassification(
                _blocked_outcome(
                    runtime, "review approval included blocker or major findings"
                )
            )
        if not approved and not declared:
            return _ReviewClassification(
                _blocked_outcome(runtime, "review reported issues without findings")
            )
        reconciled = tuple(
            reconcile_execution_ledger(
                ledger,
                findings=declared,
                resolved=resolved,
                attempt_id=attempt_id,
                review_round=review_round,
                approved=approved,
            )
        )
        # Two questions of the same round, and they are not the same question:
        # what the round cost decides whether the task gets another pass, and
        # whether its argument is closing in is a judgement on the shape of it
        # that the record keeps.
        convergence = assess_convergence(reconciled)
        snapshot = len(open_findings(reconciled))
        grant = assess_grant(
            review_round=review_round,
            minimum=self._config.review_passes,
            budget=self._config.grant_budget,
            snapshot=snapshot,
            recorded=recorded,
        )
        assessment = ReviewAssessment(
            borg_id=borg_id,
            loop=TASK_REVIEW_LOOP,
            # The task, which is the scope these passes share: the configured
            # minimum belongs to every task in the run and cannot carry one
            # task's grants. Its attempt is left unnamed because the record's
            # attempt column points at planning attempts alone.
            task_id=task_id,
            round=review_round,
            minimum=self._config.review_passes,
            converging=convergence.converging,
            open_findings=snapshot,
            refunded=grant.refunded,
            evidence=drain_evidence(convergence),
        )
        if approved:
            return _ReviewClassification(
                _PhaseOutcome(
                    TaskRuntimeStatus.MERGING,
                    str(payload.get("summary") or "review approved"),
                    runtime.review_round,
                    "merging",
                ),
                reconciled,
                assessment,
            )
        if not grant.continues:
            return _ReviewClassification(
                _PhaseOutcome(
                    TaskRuntimeStatus.BLOCKED,
                    # The account says what the rounds cost, so the reason
                    # names the state and lets it: a task held to its minimum
                    # with a budget of zero spent no grants, and saying a
                    # budget was spent would be untrue of the commonest stop
                    # an operator configures.
                    "review ended without approval. "
                    f"{grant_account([*recorded, assessment]).sentence()}",
                    review_round,
                    "review",
                ),
                reconciled,
                assessment,
            )
        return _ReviewClassification(
            _PhaseOutcome(
                TaskRuntimeStatus.FIX,
                f"review round {review_round} requested fixes",
                review_round,
                "fix",
            ),
            reconciled,
            assessment,
        )

    @staticmethod
    def _classify_fix(
        result: AgentResult,
        *,
        reviewable: frozenset[str],
        runtime: TaskRuntime,
        previous_commit: str,
        final_commit: str,
        expected_branch: str,
        actual_branch: str,
        git: SafeGit,
        after_status: str,
        operational_error: BaseException | None,
        cancellation_reason: str | None,
    ) -> _PhaseOutcome:
        if operational_error is not None:
            return _PhaseOutcome(
                TaskRuntimeStatus.BLOCKED,
                str(operational_error),
                runtime.review_round,
                "fix",
            )
        payload_status = (result.payload or {}).get("status")
        if actual_branch != expected_branch:
            reason = "fix agent changed the task branch"
        elif result.status is AgentStatus.CANCELLED:
            return _PhaseOutcome(
                TaskRuntimeStatus.FIX,
                cancellation_reason or "fix agent was interrupted",
                runtime.review_round,
                "fix",
            )
        elif result.status is AgentStatus.FAILED:
            return _PhaseOutcome(
                TaskRuntimeStatus.FAILED,
                result.error or "fix agent failed",
                runtime.review_round,
                "fix",
            )
        elif payload_status == "failed":
            return _PhaseOutcome(
                TaskRuntimeStatus.FAILED,
                "fix agent reported failed",
                runtime.review_round,
                "fix",
            )
        elif payload_status not in reviewable:
            reason = f"fix agent reported {payload_status or 'no status'}"
        elif final_commit == previous_commit:
            reason = (
                f"fix reported {payload_status} without producing a commit; "
                "worktree preserved"
            )
        elif not git.is_ancestor(previous_commit, final_commit):
            reason = "fix commit does not descend from the reviewed commit"
        elif after_status:
            reason = "fix agent left uncommitted work after its commit"
        elif payload_status == "completed":
            return _PhaseOutcome(
                TaskRuntimeStatus.REVIEW,
                f"fix committed {final_commit}",
                runtime.review_round,
                "review",
            )
        else:
            return _PhaseOutcome(
                TaskRuntimeStatus.REVIEW,
                f"fix reported {payload_status} and committed {final_commit}",
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
                or self._primary_git.for_worktree(worktree).head_sha()
                != commit_sha
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
        attestations: list[tuple[int, int, int, str, str]] = []
        for attempt in context.store.list_agent_attempts(context.claim.task_id):
            if (
                attempt.phase not in {"coding", "fix"}
                or attempt.status is not ExecutionAttemptStatus.COMPLETED
            ):
                continue
            metadata = (attempt.result or {}).get("_betterborg")
            if not isinstance(metadata, Mapping):
                continue
            base_commit = metadata.get("base_commit")
            commit_sha = metadata.get("commit_sha")
            if isinstance(base_commit, str) and isinstance(commit_sha, str):
                attestations.append(
                    (
                        attempt.review_round,
                        0 if attempt.phase == "coding" else 1,
                        attempt.attempt_number,
                        base_commit,
                        commit_sha,
                    )
                )
        if not attestations:
            raise ReviewFixPhaseError(
                "review requires a completed coding commit attestation"
            )
        attestations.sort(key=lambda item: item[:3])
        base_commit = attestations[0][3]
        current_commit = attestations[-1][4]
        git = self._primary_git.for_worktree(worktree)
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
    ) -> tuple[ExecutionLedgerFinding, ...]:
        """Return every objection still standing against the task's commit.

        The last review's payload holds only what that round said. A finding an
        earlier round raised and no later round repeated is still open, and a
        fixer shown less than the ledger holds against it answers less.
        """
        findings = tuple(
            open_execution_findings(context.store, context.claim.task_id)
        )
        if not findings:
            raise ReviewFixPhaseError(
                f"fix round {review_round} has no open review findings"
            )
        return findings

    def _transition(
        self,
        context: ScheduledTaskContext,
        expected_status: TaskRuntimeStatus,
        outcome: _PhaseOutcome,
    ) -> TaskRuntimeStatus:
        context.transition(
            expected_status,
            outcome.status,
            resume_phase=outcome.resume_phase,
            review_round=outcome.review_round,
            state_reason=outcome.reason,
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


@dataclass(frozen=True, slots=True)
class _ReviewClassification:
    """A review's outcome beside the ledger and grant its round leaves behind.

    Only a review that actually reviewed reconciles, so every other outcome
    carries neither a ledger nor an assessment: a reviewer that could not
    review, an agent that failed or was cancelled, a changed branch and a
    modified worktree all leave the rows as the round before them left them,
    because objections recorded from a review that never formed them are
    objections no later round answers — and a round with nothing to weigh has
    no grant to account for either.
    """

    outcome: _PhaseOutcome
    ledger: tuple[ExecutionLedgerFinding, ...] = ()
    assessment: ReviewAssessment | None = None


def _assessment_borg_id(context: ScheduledTaskContext) -> UUID:
    """Return the Borg this task's recorded assessments are scoped under.

    The record is a Borg's, and a task claim names only its run, so the run is
    where the loop finds out whose rounds it is accounting for.
    """
    run = context.store.get_execution_run(context.claim.run_id)
    if run is None:
        raise ReviewFixPhaseError(
            f"execution run {context.claim.run_id} is no longer recorded"
        )
    return run.borg_id


def _ledger_round(runtime: TaskRuntime) -> int:
    """Return the ledger round a review of this runtime runs as.

    The ledger numbers its rounds from one, as a planning ledger does, so that
    every prompt, reason and row speaks one numbering. The task runtime counts
    the review rounds behind it instead, from zero.
    """
    return runtime.review_round + 1


def _blocked_outcome(runtime: TaskRuntime, reason: str) -> _PhaseOutcome:
    return _PhaseOutcome(
        TaskRuntimeStatus.BLOCKED,
        reason,
        runtime.review_round,
        runtime.status.value,
    )


def _unfinished_coding_report(
    context: ScheduledTaskContext,
) -> tuple[str, tuple[str, ...]] | None:
    """Return the coding agent's own account of what it did not finish.

    A commit reaches review whenever coding left one, including when the agent
    said the task is not done. Review is the first reader able to weigh that
    claim against the tree, and it cannot weigh what it is not told.
    """
    attempts = [
        attempt
        for attempt in context.store.list_agent_attempts(context.claim.task_id)
        if attempt.phase == "coding"
    ]
    if not attempts:
        return None
    result = attempts[-1].result or {}
    status = result.get("status")
    if not isinstance(status, str) or status == "completed":
        return None
    notes = tuple(
        text
        for key in ("blockers", "follow_ups")
        for item in (result.get(key) or ())
        if (text := str(item).strip())
    )
    return status, notes


def _declared_findings(
    payload: Mapping[str, Any],
    *,
    task_id: UUID,
    attempt_id: UUID,
    review_round: int,
) -> tuple[tuple[ExecutionLedgerFinding, str | None], ...]:
    """Build this round's findings, each beside the objection it restates."""

    raw = payload.get("findings")
    if not isinstance(raw, list):
        raise ValueError("findings must be a list")
    declared: list[tuple[ExecutionLedgerFinding, str | None]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("every finding must be an object")
        # Both declarations are required, so both are read as required. An
        # absent severity is the sharper one: it is the field this loop
        # branches on, and read as empty it would pass for one that does not
        # hold the task.
        for required in ("severity", "message", "repeats"):
            if required not in item:
                raise ValueError(f"every finding declares its {required}")
        repeats = item["repeats"]
        if repeats is not None and not isinstance(repeats, str):
            raise ValueError("a finding repeats one id or nothing")
        suggestion = str(item.get("suggestion") or "").strip()
        declared.append(
            (
                ExecutionLedgerFinding(
                    task_id=task_id,
                    attempt_id=attempt_id,
                    first_seen_round=review_round,
                    last_seen_round=review_round,
                    severity=str(item["severity"]),
                    message=str(item["message"]).strip(),
                    suggestion=suggestion or None,
                ),
                repeats,
            )
        )
    return tuple(declared)


def _declared_resolved(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the ids a review says its round closed.

    An id the ledger cannot place is left to the reconciliation rather than
    refused here: the row it meant stays open, which costs the loop a grant and
    lets no blocker through.
    """

    raw = payload.get("resolved")
    if not isinstance(raw, list):
        raise ValueError("resolved must be a list")
    return tuple(str(item) for item in raw)


def _ledger_lines(findings: Sequence[ExecutionLedgerFinding]) -> list[str]:
    """Render open objections as the list an agent names ids from."""

    return [
        f"- {finding.id} ({finding.severity}, first raised in round "
        f"{finding.first_seen_round}): {finding.message}"
        + (f" (suggestion: {finding.suggestion})" if finding.suggestion else "")
        for finding in findings
    ]


def _render_review_prompt(
    inputs: VerifiedTaskInputs,
    *,
    branch: str,
    base_commit: str,
    current_commit: str,
    review_round: int,
    open_ledger: Sequence[ExecutionLedgerFinding] = (),
    unfinished: tuple[str, tuple[str, ...]] | None = None,
) -> str:
    sections = [
        "Review the assigned implementation without modifying the worktree.",
        "Compare the declared base commit with the current task commit and return "
        "only the required structured result.",
        "",
        EXISTING_TEST_REVIEW_RULE,
        "",
        REVIEW_FINDING_RULE,
        "",
        f"Task file: {inputs.task_path.as_posix()}",
        f"Task digest: {inputs.task.digest}",
        f"Task branch: {branch}",
        f"Declared base commit: {base_commit}",
        f"Current task commit: {current_commit}",
        f"Review round: {review_round}",
    ]
    if open_ledger:
        sections.extend(
            [
                "",
                "## Open findings this commit has to answer",
                "",
                "Earlier rounds raised these and no round has closed them. "
                "Judge each against the tree in front of you, and name it by "
                "the id shown here in resolved or in repeats.",
                "",
                *_ledger_lines(open_ledger),
            ]
        )
    if unfinished is not None:
        status, notes = unfinished
        sections.extend(
            [
                "",
                "## The coding agent did not report the task finished",
                "",
                f"It committed this work and returned status {status!r}. Judge "
                "the commit on the assigned task as you would any other, and "
                "read what follows as the coder's own account rather than as "
                "findings: report only what you can still see in the tree.",
            ]
        )
        if notes:
            sections.extend(["", *(f"- {note}" for note in notes)])
    sections.extend(
        [
            "",
            "## Assigned task",
            "",
            inputs.task_markdown.rstrip(),
        ]
    )
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
    findings: Sequence[ExecutionLedgerFinding],
    review_round: int,
) -> str:
    sections = [
        "Fix every open review finding in the current worktree. Keep the "
        "change in scope, run relevant verification, and commit the fix before "
        "returning completed.",
        "",
        EXISTING_TEST_RULE,
        "",
        f"Task file: {inputs.task_path.as_posix()}",
        f"Task digest: {inputs.task.digest}",
        f"Fix round: {review_round}",
        "",
        "## Open review findings",
        "",
        "Every objection still standing against this commit, including ones an "
        "earlier round raised that the latest review did not repeat.",
        "",
        *_ledger_lines(findings),
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
