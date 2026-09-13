"""Durable Supervisor review and bounded Project Manager revision lifecycle."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from betterborg_cli.agent_runtime.base import AgentAdapter, CancellationToken
from betterborg_cli.agent_runtime.selection import (
    AgentSelectionError,
    SelectedAgent,
    require_read_only_agent,
    resolve_agent_model,
)
from betterborg_cli.planning.convergence import assess_convergence, drain_evidence
from betterborg_cli.planning.findings_ledger import (
    REPEATS_SCHEMA,
    RESOLVED_SCHEMA,
    open_findings,
    open_task_findings,
    reconcile_task_ledger,
    task_ledger_json,
)
from betterborg_cli.planning.grants import (
    PLANNING_GRANT_BUDGET,
    GrantAccount,
    assess_grant,
    planning_grant_account,
)
from betterborg_cli.planning.pm import (
    PM_OUTPUT_RETRY_MINIMUM,
    ProjectManagerCancelled,
    ProjectManagerError,
    ProjectManagerLoop,
    approved_plan_digest,
    task_batch_semantic_digest,
)
from betterborg_cli.planning.task_publication import (
    TaskPublication,
    TaskPublicationCancelled,
    TaskPublicationError,
    TaskPublisher,
)
from betterborg_cli.planning.task_validation import (
    TaskGraphValidationError,
    validate_task_graph,
)
from betterborg_cli.planning.turns import (
    DurablePlanningTurns,
    planning_attempt_duration,
    planning_attempt_result,
    planning_request_change_attempts,
)
from betterborg_cli.progress import ChildSpec, RunProgress, StageSpec, StageState
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.store import (
    Borg,
    BorgState,
    PlanApproval,
    PlanningAttempt,
    PlanningAttemptStatus,
    Repository,
    ReviewAssessment,
    SqliteStore,
    TaskBatch,
    TaskDependency,
    TaskFinding,
    TaskGeneration,
    TaskGenerationStatus,
    TaskRecord,
)

#: Review rounds a task batch gets before every further round is a grant,
#: where its repository configures no minimum of its own.
SUPERVISOR_ROUND_MINIMUM = 3
_SUPERVISOR_PHASE = "supervisor_review"
_PUBLICATION_DETAIL = "publishing approved tasks"
_RETAINED_APPROVAL_RESULT = "approval retained; task publication pending"

_NONBLANK_STRING: dict[str, Any] = {
    "type": "string",
    "minLength": 1,
    "pattern": r"\S",
}

SUPERVISOR_REVIEW_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "summary", "findings", "resolved"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approve", "request_changes"],
        },
        "summary": _NONBLANK_STRING,
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
                    "message": _NONBLANK_STRING,
                    "suggestion": _NONBLANK_STRING,
                    "task_ref": _NONBLANK_STRING,
                    "repeats": REPEATS_SCHEMA,
                },
            },
        },
    },
}

_SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor reviewing a complete,
deterministically valid Project Manager task batch for an approved plan. Judge
plan coverage, task coherence, foundation ownership, reuse instead of
duplication, dependency ordering, meaningful tests, and simplicity. Approve
only a complete batch that is ready for publication. Betterborg performs the
delivery around the batch: branching, worktrees, commits, review, merge, and
the repository's own checks. Every task changes the repository, so never hold
the batch to work Betterborg already does, and reject a task that has nothing
to commit. Do not modify files or redesign the batch; return actionable
findings for the Project Manager. Your decision and your findings have to
agree: return request_changes only while holding at least one blocker or major
finding, and return approve only while holding none. A batch whose every fault
is minor is one you approve, saying what the faults are. Account for the open
findings you were given: list in resolved the id of every one this batch
closes, and on each finding of your own set repeats to the id of the open
finding it raises again, or null when the objection is new. Both are always
required, so a review that closes nothing sends an empty list and a new finding
sends null. An open finding you neither resolve nor repeat stays open. Every
revision mints fresh task references, so an open finding's
raised_against_task_ref records where the objection started and is never a
reference to reuse: a finding of your own names a task in the batch under
review or no task at all. Return only the required JSON object.
"""


#: A decision that contradicts its own findings is sent back the way the
#: Architect's contract failures are, rather than ending decomposition on the
#: turn that made it.
SUPERVISOR_DECISION_ROUND_CAP = 3

_DECISION_CORRECTION = """

## Rejected review

Your last review never reached the Project Manager:

{error}

Review the batch again and return the whole result. A decision of
request_changes carries at least one blocker or major finding; a decision of
approve carries none. Minor findings ride along with either.
"""


class SupervisorError(RuntimeError):
    """Raised when task review cannot reach a durable outcome."""


class SupervisorCancelled(SupervisorError):
    """Raised after preserving enough Supervisor state to resume later."""


@dataclass(frozen=True, slots=True)
class SupervisorResult:
    """The one latest reviewed batch and its durable handoff state."""

    borg: Borg
    approval: PlanApproval
    batch: TaskBatch
    generation: TaskGeneration
    tasks: tuple[TaskRecord, ...]
    dependencies: tuple[TaskDependency, ...]
    findings: tuple[TaskFinding, ...]
    attempt: PlanningAttempt
    publication: TaskPublication | None


def supervisor_grant_account(
    store: SqliteStore, borg_id: UUID, *, plan_approval_id: UUID
) -> GrantAccount:
    """Account for what this approval's review rounds cost its task batch."""

    return planning_grant_account(
        store,
        borg_id,
        loop=_SUPERVISOR_PHASE,
        plan_approval_id=plan_approval_id,
    )


def _review_round_phrase(review_round: int) -> str:
    """Name the round and nothing else.

    No loop knows whether a round is its last: that depends on findings the
    round has not produced yet. And a budget number in a reviewer's prompt
    reads as a deadline it is being held to, so where the loop stands belongs
    to the operator's surfaces.
    """

    return f"in round {review_round}."


class SupervisorLoop:
    """Review valid PM batches and revise while the grant budget holds."""

    def __init__(
        self,
        repository: Repository,
        borg: Borg,
        store: SqliteStore,
        agent: AgentAdapter | SelectedAgent,
        *,
        pm_agent: AgentAdapter | SelectedAgent | None = None,
        approved_plan: Mapping[str, Any] | None = None,
        plan_approval: PlanApproval | None = None,
        artifact_dir: Path | None = None,
        model: str | None = None,
        pm_model: str | None = None,
        cancel: CancellationToken | None = None,
        progress: RunProgress | None = None,
        dirty_borg_documents: Sequence[Path] = (),
        worktrees_root: Path | None = None,
        review_rounds: int = SUPERVISOR_ROUND_MINIMUM,
        pm_output_retries: int = PM_OUTPUT_RETRY_MINIMUM,
        grant_budget: int = PLANNING_GRANT_BUDGET,
    ) -> None:
        if cancel is not None and cancel.is_set():
            raise SupervisorCancelled("Supervisor run cancelled")
        if not isinstance(review_rounds, int) or review_rounds < 1:
            raise SupervisorError(
                "Supervisor review rounds must be a whole number of at least 1"
            )
        # Zero buys nothing and is legal; below zero would stop the loop short
        # of the minimum it was told to run.
        if not isinstance(grant_budget, int) or grant_budget < 0:
            raise SupervisorError(
                "Supervisor grant budget must be a whole number of at least 0"
            )
        # Refused here as well as by the loop it is carried to, so a caller
        # learns it at construction like the two limits beside it rather than
        # part-way through a decomposition round.
        if not isinstance(pm_output_retries, int) or pm_output_retries < 1:
            raise SupervisorError(
                "Project Manager output retries must be a whole number of at "
                "least 1"
            )
        self.review_rounds = review_rounds
        self.pm_output_retries = pm_output_retries
        self.grant_budget = grant_budget
        project_manager = pm_agent or agent
        require_read_only_agent(
            agent, role="Supervisor", error_factory=SupervisorError
        )
        require_read_only_agent(
            project_manager, role="Project Manager", error_factory=SupervisorError
        )
        try:
            resolved_model = resolve_agent_model(agent, model)
            resolved_pm_model = resolve_agent_model(project_manager, pm_model)
        except AgentSelectionError as error:
            raise SupervisorError(str(error)) from error

        paths = RepoPaths.discover(repository.root, cancel=cancel)
        if paths.root != repository.root:
            raise ValueError("repository root does not match its discovered Git root")
        self.repository = repository
        self.borg_id = borg.id
        self.store = store
        self.agent = agent
        self.pm_agent = project_manager
        self._supplied_plan = dict(approved_plan) if approved_plan is not None else None
        self._supplied_approval = plan_approval
        self.artifact_dir = Path(
            artifact_dir or paths.artifacts_dir / "planning" / str(borg.id)
        ).resolve()
        self.model = resolved_model
        self.pm_model = resolved_pm_model
        self.cancel = cancel
        self.progress = progress
        self.dirty_borg_documents = tuple(dirty_borg_documents)
        self.worktrees_root = worktrees_root
        if progress is not None:
            if "project-manager" not in progress.stages:
                progress.declare(StageSpec("project-manager", "Project Manager"))
            if "supervisor" not in progress.stages:
                progress.declare(StageSpec("supervisor", "Supervisor"))
        self._turns = DurablePlanningTurns(
            repository,
            borg,
            store,
            agent,
            role="Supervisor",
            model=resolved_model,
            artifact_dir=self.artifact_dir,
            error_factory=SupervisorError,
            cancelled_error_factory=SupervisorCancelled,
            cancel=cancel,
            progress=progress,
            stage_key="supervisor" if progress is not None else None,
            dirty_borg_documents=dirty_borg_documents,
            worktrees_root=worktrees_root,
        )

    def run(self) -> SupervisorResult:
        """Resume review and PM revision turns until approval or exhaustion."""
        try:
            approval = self._approval()
            plan = self._approved_plan(approval)
            self._seed_project_manager_progress(approval)
            self._declare_revision_progress(approval)
            if self._approved_publication_is_pending(approval):
                self._start_supervisor_progress(approval)
                self._show_publication_progress()
            terminal = self._terminal_result(approval)
            if terminal is not None:
                self._seed_revision_progress(approval)
                self._complete_or_seed_supervisor_progress(terminal.attempt)
                return terminal

            self._seed_revision_progress(approval)
            self._start_supervisor_progress(approval)
            result = self._run(approval, plan)
            if self.progress is not None:
                self.progress.complete(
                    "supervisor", result.attempt.summary or "review complete"
                )
            return result
        except (
            ProjectManagerCancelled,
            SupervisorCancelled,
            KeyboardInterrupt,
        ) as error:
            if isinstance(error, KeyboardInterrupt):
                self._cancel_interrupted_review()
            approval_retained = (
                isinstance(error, KeyboardInterrupt)
                and self._approved_publication_is_pending(self._approval())
            )
            self._reconcile_progress(
                _RETAINED_APPROVAL_RESULT if approval_retained else str(error),
                stopped=True,
                always_reconcile=True,
            )
            if (
                approval_retained
                and self._approved_publication_is_pending(self._approval())
            ):
                raise SupervisorCancelled(_RETAINED_APPROVAL_RESULT) from error
            raise
        except Exception as error:
            self._reconcile_progress(
                str(error),
                stopped=self.cancel is not None and self.cancel.is_set(),
            )
            raise

    def _run(
        self, approval: PlanApproval, plan: dict[str, Any]
    ) -> SupervisorResult:
        """Execute the active Supervisor parent through all revision cycles."""

        decision_correction = ""
        decision_rounds = 0
        while True:
            borg = self._turns.current_borg()
            if borg.state is BorgState.PM_WORKING:
                child_key = self._active_revision_key(approval)
                initial_work = (
                    child_key is None and not self._completed_reviews(approval)
                )
                if not initial_work and child_key is None:
                    raise SupervisorError(
                        "Project Manager revision requires a rejected "
                        "Supervisor attempt"
                    )
                if initial_work:
                    self._start_project_manager_progress()
                else:
                    self._start_revision_progress(child_key)
                try:
                    revised = ProjectManagerLoop(
                        self.repository,
                        borg,
                        self.store,
                        self.pm_agent,
                        approved_plan=plan,
                        plan_approval=approval,
                        artifact_dir=self.artifact_dir,
                        model=self.pm_model,
                        cancel=self.cancel,
                        progress=self.progress,
                        stage_key=(
                            "project-manager" if initial_work else "supervisor"
                        ),
                        child_key=None if initial_work else child_key,
                        dirty_borg_documents=self.dirty_borg_documents,
                        worktrees_root=self.worktrees_root,
                        output_retries=self.pm_output_retries,
                        grant_budget=self.grant_budget,
                    ).run()
                except ProjectManagerCancelled as error:
                    raise SupervisorCancelled(str(error)) from error
                except ProjectManagerError as error:
                    raise SupervisorError(str(error)) from error
                if revised.borg.state is not BorgState.SUPERVISOR_WORKING:
                    raise SupervisorError(
                        "Project Manager revision did not return to Supervisor"
                    )
                self._start_supervisor_progress(approval)
                continue
            if borg.state is not BorgState.SUPERVISOR_WORKING:
                raise SupervisorError(
                    f"Borg {borg.name!r} cannot run Supervisor from state "
                    f"{borg.state.value!r}"
                )
            if self.cancel is not None and self.cancel.is_set():
                raise SupervisorCancelled("Supervisor run cancelled")

            batch, generation, tasks, dependencies = self._latest_graph(approval)
            try:
                validate_task_graph(plan, tasks, dependencies)
            except TaskGraphValidationError as error:
                raise SupervisorError(
                    "Supervisor requires a deterministically valid PM batch: "
                    + str(error)
                ) from error
            self._require_revision_progress(batch, approval)

            review_round = len(self._completed_reviews(approval)) + 1
            attempt, payload = self._turns.run(
                phase=_SUPERVISOR_PHASE,
                round_number=self._turns.next_round(_SUPERVISOR_PHASE),
                schema=SUPERVISOR_REVIEW_SCHEMA,
                system_prompt=_SUPERVISOR_SYSTEM_PROMPT,
                user_prompt=(
                    "Read the complete approved plan and task-review context from "
                    ".betterborg/state/planning/context/manifest.json. "
                    "Review task batch "
                    f"{batch.id} "
                    + _review_round_phrase(review_round)
                    + decision_correction
                ),
                current_plan=json.dumps(
                    self._review_context(plan, batch, tasks),
                    indent=2,
                    sort_keys=True,
                ),
                turn_name="task review",
                request_context={
                    "approved_plan_digest": approval.plan_digest,
                    "batch_digest": batch.digest,
                    "batch_id": str(batch.id),
                    "generation_id": str(generation.id),
                    "plan_approval_id": str(approval.id),
                    "review_round": review_round,
                },
            )
            try:
                declared = self._findings(
                    payload, attempt, batch, tasks, review_round
                )
            except SupervisorError as error:
                self.store.complete_planning_attempt(
                    attempt.id,
                    status=PlanningAttemptStatus.FAILED,
                    result=payload,
                    summary=str(error),
                )
                decision_rounds += 1
                if decision_rounds >= SUPERVISOR_DECISION_ROUND_CAP:
                    raise
                # A failed attempt is not a completed review, so this costs the
                # batch none of its rounds and leaves the revision check
                # reading the same last decision it read before.
                decision_correction = _DECISION_CORRECTION.format(error=error)
                continue

            decision_correction = ""
            decision = payload["decision"]
            # Reconciled before the decision is taken, and written with it, so
            # no reader is a round behind the objection it is judging.
            findings = tuple(finding for finding, _ in declared)
            ledger = reconcile_task_ledger(
                self.store.list_task_ledger_findings(
                    self.borg_id, plan_approval_id=approval.id
                ),
                findings=declared,
                resolved=payload["resolved"],
                attempt_id=attempt.id,
                plan_approval_id=approval.id,
                review_round=review_round,
                approved=decision == "approve",
            )
            # Two questions of the same round, and they are not the same
            # question: what the round cost decides whether the loop gets
            # another, and whether its argument is closing in is a judgement on
            # the shape of it that the record keeps.
            convergence = assess_convergence(ledger)
            snapshot = len(open_findings(ledger))
            grant = assess_grant(
                review_round=review_round,
                minimum=self.review_rounds,
                budget=self.grant_budget,
                snapshot=snapshot,
                recorded=self.store.list_review_assessments(
                    self.borg_id,
                    loop=_SUPERVISOR_PHASE,
                    plan_approval_id=approval.id,
                ),
            )
            assessment = ReviewAssessment(
                borg_id=self.borg_id,
                loop=_SUPERVISOR_PHASE,
                # The approval, and not the batch under review: this loop
                # compares its rounds across every batch the approval produced,
                # and a scope that changed each round would be no scope at all.
                plan_approval_id=approval.id,
                attempt_id=attempt.id,
                round=review_round,
                minimum=self.review_rounds,
                converging=convergence.converging,
                open_findings=snapshot,
                refunded=grant.refunded,
                evidence=drain_evidence(convergence),
            )

            if decision == "approve":
                next_state = BorgState.READY_TO_EXECUTE
            elif grant.continues:
                next_state = BorgState.PM_WORKING
            else:
                next_state = BorgState.BLOCKED

            with self.store.transaction():
                completed = self.store.complete_planning_attempt(
                    attempt.id,
                    status=PlanningAttemptStatus.COMPLETED,
                    result=payload,
                    summary=str(payload["summary"]).strip(),
                )
                for finding in findings:
                    self.store.append_task_finding(finding)
                self.store.record_task_ledger_findings(ledger)
                self.store.record_review_assessment(assessment)
                if decision != "approve":
                    borg = self._turns.transition(borg, next_state)

            publication = None
            if decision == "approve":
                self._show_publication_progress()
                try:
                    publication = TaskPublisher(
                        self.repository,
                        self.store,
                        cancel=self.cancel,
                    ).publish(generation.id)
                    generation = publication.generation
                except TaskPublicationCancelled as error:
                    raise SupervisorCancelled(_RETAINED_APPROVAL_RESULT) from error
                except TaskPublicationError as error:
                    raise SupervisorError(
                        f"approved task publication failed: {error}"
                    ) from error
                borg = self._turns.transition(borg, BorgState.READY_TO_EXECUTE)

            if next_state is not BorgState.PM_WORKING:
                return SupervisorResult(
                    borg=borg,
                    approval=approval,
                    batch=batch,
                    generation=generation,
                    tasks=tasks,
                    dependencies=dependencies,
                    findings=findings,
                    attempt=completed,
                    publication=publication,
                )
            self._declare_revision_progress(approval)

    def _approval(self) -> PlanApproval:
        approvals = self.store.list_plan_approvals(self.borg_id)
        if not approvals:
            raise SupervisorError("Supervisor requires an approved plan")
        approval = self._supplied_approval or approvals[-1]
        if approval not in approvals or approval != approvals[-1]:
            raise SupervisorError("Supervisor requires the latest persisted approval")
        return approval

    def _approved_plan(self, approval: PlanApproval) -> dict[str, Any]:
        plan = self._supplied_plan
        if plan is None and isinstance(approval.manifest.get("plan"), dict):
            plan = dict(approval.manifest["plan"])
        if plan is None:
            plan = next(
                (
                    dict(attempt.result)
                    for attempt in reversed(
                        self.store.list_planning_attempts(self.borg_id)
                    )
                    if attempt.phase == "architect_plan"
                    and attempt.status is PlanningAttemptStatus.COMPLETED
                    and attempt.result is not None
                    and approved_plan_digest(attempt.result) == approval.plan_digest
                ),
                None,
            )
        if plan is None:
            raise SupervisorError("approved plan content is unavailable")
        if approved_plan_digest(plan) != approval.plan_digest:
            raise SupervisorError("approved plan content does not match its digest")
        return plan

    def _latest_graph(
        self, approval: PlanApproval
    ) -> tuple[
        TaskBatch,
        TaskGeneration,
        tuple[TaskRecord, ...],
        tuple[TaskDependency, ...],
    ]:
        batch = next(
            (
                item
                for item in reversed(self.store.list_task_batches(self.borg_id))
                if item.plan_approval_id == approval.id
            ),
            None,
        )
        if batch is None:
            raise SupervisorError("Supervisor requires a PM task batch")
        generation = next(
            (
                item
                for item in reversed(self.store.list_task_generations(self.borg_id))
                if item.batch_id == batch.id
            ),
            None,
        )
        if generation is None:
            raise SupervisorError("Supervisor task batch has no generation")
        if generation.status is not TaskGenerationStatus.PREPARING:
            raise SupervisorError(
                "Supervisor can only review a preparing task generation"
            )
        tasks = tuple(self.store.list_task_records(generation.id))
        dependencies = tuple(self.store.list_task_dependencies(generation.id))
        return batch, generation, tasks, dependencies

    def _review_context(
        self,
        plan: dict[str, Any],
        batch: TaskBatch,
        tasks: tuple[TaskRecord, ...],
    ) -> dict[str, Any]:
        approval_batch_ids = {
            candidate.id
            for candidate in self.store.list_task_batches(self.borg_id)
            if candidate.plan_approval_id == batch.plan_approval_id
        }
        history = [
            {
                "batch_id": str(finding.batch_id),
                "message": finding.message,
                "round": finding.round,
                "severity": finding.severity,
                "suggestion": finding.suggestion,
                "task_ref": finding.task_ref,
            }
            for finding in self.store.list_task_findings(self.borg_id)
            if finding.batch_id in approval_batch_ids
        ]
        return {
            "approved_plan": plan,
            # The history says everything every round said, with no lifecycle
            # on any of it and no id to name a row by. The open ledger is the
            # list a reviewer names ids from.
            "open_supervisor_findings": [
                task_ledger_json(row)
                for row in open_task_findings(
                    self.store, self.borg_id, batch.plan_approval_id
                )
            ],
            "prior_supervisor_findings": history,
            "task_batch": {
                "digest": batch.digest,
                "id": str(batch.id),
                "summary": batch.summary,
                "tasks": [
                    {"task_ref": task.task_ref, "task": task.task} for task in tasks
                ],
            },
        }

    def _findings(
        self,
        payload: dict[str, Any],
        attempt: PlanningAttempt,
        batch: TaskBatch,
        tasks: tuple[TaskRecord, ...],
        review_round: int,
    ) -> tuple[tuple[TaskFinding, str | None], ...]:
        """Build this round's findings, each beside the objection it restates."""

        raw_findings = payload["findings"]
        if payload["decision"] == "request_changes" and not raw_findings:
            raise SupervisorError("Supervisor request_changes must include findings")
        known_refs = {task.task_ref for task in tasks}
        declared: list[tuple[TaskFinding, str | None]] = []
        for item in raw_findings:
            task_ref = item.get("task_ref")
            if task_ref is not None and task_ref not in known_refs:
                raise SupervisorError(
                    f"Supervisor finding references unknown task {task_ref!r}"
                )
            finding = TaskFinding(
                borg_id=self.borg_id,
                batch_id=batch.id,
                attempt_id=attempt.id,
                round=review_round,
                severity=item["severity"],
                message=item["message"].strip(),
                suggestion=(
                    item["suggestion"].strip()
                    if item.get("suggestion") is not None
                    else None
                ),
                task_ref=task_ref,
            )
            declared.append((finding, item["repeats"]))
        actionable = any(
            finding.severity in {"blocker", "major"} for finding, _ in declared
        )
        if payload["decision"] == "approve" and actionable:
            raise SupervisorError(
                "Supervisor cannot approve with blocker or major findings"
            )
        if payload["decision"] == "request_changes" and not actionable:
            raise SupervisorError(
                "Supervisor request_changes requires a blocker or major finding"
            )
        return tuple(declared)

    def _completed_reviews(self, approval: PlanApproval) -> list[PlanningAttempt]:
        return [
            attempt
            for attempt in self._turns.attempts(_SUPERVISOR_PHASE)
            if attempt.status is PlanningAttemptStatus.COMPLETED
            and attempt.request.get("plan_approval_id") == str(approval.id)
            and attempt.request.get("approved_plan_digest") == approval.plan_digest
        ]

    def _pm_attempts(self, approval: PlanApproval) -> list[PlanningAttempt]:
        return [
            attempt
            for attempt in self._turns.attempts("pm_tasks")
            if attempt.status is PlanningAttemptStatus.COMPLETED
            and attempt.request.get("plan_approval_id") == str(approval.id)
            and attempt.request.get("approved_plan_digest") == approval.plan_digest
        ]

    def _initial_pm_attempt(self, approval: PlanApproval) -> PlanningAttempt | None:
        return next(
            (
                attempt
                for attempt in self._pm_attempts(approval)
                if attempt.request.get("base_batch_id") is None
            ),
            None,
        )

    def _revision_reviews(self, approval: PlanApproval) -> list[PlanningAttempt]:
        return planning_request_change_attempts(
            self._completed_reviews(approval), _SUPERVISOR_PHASE
        )

    @staticmethod
    def _revision_key(review: PlanningAttempt) -> str:
        return f"pm-revision:{review.id}"

    def _revision_attempt(
        self, approval: PlanApproval, review: PlanningAttempt
    ) -> PlanningAttempt | None:
        batch_id = review.request.get("batch_id")
        return next(
            (
                attempt
                for attempt in self._pm_attempts(approval)
                if attempt.started_at >= (review.finished_at or review.started_at)
                and attempt.request.get("base_batch_id") == batch_id
            ),
            None,
        )

    def _active_revision_key(self, approval: PlanApproval) -> str | None:
        for review in reversed(self._revision_reviews(approval)):
            if self._revision_attempt(approval, review) is None:
                return self._revision_key(review)
        return None

    def _revisions_with_work(self, approval: PlanApproval) -> set[str]:
        """Identify the rejections a revision belongs to, run or under way.

        The rejection that blocked is followed by neither, and a child
        declared for it would stay pending forever, refusing to let its parent
        be seeded when the record is read back.
        """
        reviews = self._revision_reviews(approval)
        revising = self._turns.current_borg().state is BorgState.PM_WORKING
        return {
            review.id
            for index, review in enumerate(reviews)
            if self._revision_attempt(approval, review) is not None
            or (revising and index == len(reviews) - 1)
        }

    def _declare_revision_progress(self, approval: PlanApproval) -> None:
        if self.progress is None:
            return
        with_work = self._revisions_with_work(approval)
        for number, review in enumerate(self._revision_reviews(approval), start=1):
            if review.id not in with_work:
                continue
            key = self._revision_key(review)
            if key not in self.progress.stages["supervisor"].children:
                self.progress.declare_child(
                    "supervisor", ChildSpec(key, f"Project Manager revision {number}")
                )

    def _seed_revision_progress(self, approval: PlanApproval) -> None:
        if self.progress is None:
            return
        for review in self._revision_reviews(approval):
            attempt = self._revision_attempt(approval, review)
            if attempt is None:
                continue
            key = self._revision_key(review)
            child = self.progress.stages["supervisor"].children[key]
            if child.state is StageState.PENDING:
                self.progress.seed_child_completed(
                    "supervisor",
                    key,
                    planning_attempt_result(attempt, default="task batch ready"),
                    planning_attempt_duration(attempt),
                )

    def _start_revision_progress(self, child_key: str | None) -> None:
        if self.progress is None or child_key is None:
            return
        child = self.progress.stages["supervisor"].children[child_key]
        if child.state is StageState.PENDING:
            self.progress.start_child("supervisor", child_key)

    def _start_project_manager_progress(self) -> None:
        if self.progress is None:
            return
        record = self.progress.stages["project-manager"]
        if record.state is StageState.PENDING:
            self.progress.start("project-manager")

    def _seed_project_manager_progress(self, approval: PlanApproval) -> None:
        if self.progress is None:
            return
        attempt = self._initial_pm_attempt(approval)
        record = self.progress.stages["project-manager"]
        if attempt is not None and record.state is StageState.PENDING:
            self.progress.seed_completed(
                "project-manager",
                planning_attempt_result(attempt, default="task batch ready"),
                planning_attempt_duration(attempt),
            )

    def _start_supervisor_progress(self, approval: PlanApproval) -> None:
        if self.progress is None:
            return
        record = self.progress.stages["supervisor"]
        if record.state is StageState.PENDING and (
            self._initial_pm_attempt(approval) is not None
            or bool(self._completed_reviews(approval))
        ):
            self.progress.start("supervisor")

    def _seed_supervisor_progress(self, attempt: PlanningAttempt) -> None:
        if self.progress is None:
            return
        record = self.progress.stages["supervisor"]
        if record.state is StageState.PENDING:
            self.progress.seed_completed(
                "supervisor",
                planning_attempt_result(attempt, default="review complete"),
                planning_attempt_duration(attempt),
            )

    def _complete_or_seed_supervisor_progress(
        self, attempt: PlanningAttempt
    ) -> None:
        if self.progress is None:
            return
        record = self.progress.stages["supervisor"]
        result = planning_attempt_result(attempt, default="review complete")
        if record.state is StageState.RUNNING:
            self.progress.complete("supervisor", result)
        elif record.state is StageState.PENDING:
            self._seed_supervisor_progress(attempt)

    def _show_publication_progress(self) -> None:
        if self.progress is None:
            return
        record = self.progress.stages["supervisor"]
        if record.state is StageState.RUNNING:
            self.progress.update("supervisor", _PUBLICATION_DETAIL)

    def _approved_publication_is_pending(self, approval: PlanApproval) -> bool:
        borg = self._turns.current_borg()
        return borg.state is BorgState.SUPERVISOR_WORKING and any(
            (attempt.result or {}).get("decision") == "approve"
            for attempt in self._completed_reviews(approval)
        )

    def _cancel_interrupted_review(self) -> None:
        running = next(
            (
                attempt
                for attempt in reversed(self._turns.attempts(_SUPERVISOR_PHASE))
                if attempt.status is PlanningAttemptStatus.RUNNING
            ),
            None,
        )
        if running is not None:
            self._turns.cancel_attempt(running)

    def _finish_progress(self, result: str, *, stopped: bool) -> None:
        if self.progress is None:
            return
        project_manager = self.progress.stages["project-manager"]
        if project_manager.state is StageState.RUNNING:
            if stopped:
                self.progress.stop("project-manager", result)
            else:
                self.progress.fail("project-manager", result)
        for child in self.progress.stages["supervisor"].children.values():
            if child.state is not StageState.RUNNING:
                continue
            if stopped:
                self.progress.stop_child("supervisor", child.key, result)
            else:
                self.progress.fail_child("supervisor", child.key, result)
        if self.progress.stages["supervisor"].state is StageState.RUNNING:
            if stopped:
                self.progress.stop("supervisor", result)
            else:
                self.progress.fail("supervisor", result)

    def _reconcile_progress(
        self,
        result: str,
        *,
        stopped: bool,
        always_reconcile: bool = False,
    ) -> None:
        if self.progress is None and not always_reconcile:
            return
        project_manager = (
            None
            if self.progress is None
            else self.progress.stages["project-manager"]
        )
        record = (
            None if self.progress is None else self.progress.stages["supervisor"]
        )
        if (
            not always_reconcile
            and project_manager is not None
            and record is not None
            and project_manager.state is not StageState.RUNNING
            and record.state is not StageState.RUNNING
        ):
            return
        approval = self._approval()
        initial_attempt = self._initial_pm_attempt(approval)
        if (
            project_manager is not None
            and project_manager.state is StageState.RUNNING
            and initial_attempt is not None
        ):
            self.progress.complete(
                "project-manager",
                planning_attempt_result(initial_attempt, default="task batch ready"),
            )
        try:
            terminal = self._terminal_result(approval)
        except (SupervisorCancelled, SupervisorError):
            terminal = None
        if self.progress is None:
            return
        if terminal is not None:
            self.progress.complete(
                "supervisor", terminal.attempt.summary or "review complete"
            )
        else:
            self._finish_progress(result, stopped=stopped)

    def _require_revision_progress(
        self, batch: TaskBatch, approval: PlanApproval
    ) -> None:
        previous = next(
            (
                attempt
                for attempt in reversed(self._completed_reviews(approval))
                if (attempt.result or {}).get("decision") == "request_changes"
            ),
            None,
        )
        if previous is None:
            return
        previous_batch_id = previous.request.get("batch_id")
        previous_batch = next(
            (
                candidate
                for candidate in self.store.list_task_batches(self.borg_id)
                if str(candidate.id) == previous_batch_id
            ),
            None,
        )
        if previous_batch is None:
            raise SupervisorError("rejected Supervisor batch is no longer available")
        if task_batch_semantic_digest(
            [task.task for task in self._tasks_for_batch(batch)]
        ) == task_batch_semantic_digest(
            [task.task for task in self._tasks_for_batch(previous_batch)]
        ):
            raise SupervisorError(
                "Project Manager revision made no progress against the rejected batch"
            )

    def _tasks_for_batch(self, batch: TaskBatch) -> tuple[TaskRecord, ...]:
        generation = next(
            (
                candidate
                for candidate in reversed(
                    self.store.list_task_generations(self.borg_id)
                )
                if candidate.batch_id == batch.id
            ),
            None,
        )
        if generation is None:
            raise SupervisorError(f"task batch {batch.id} has no durable generation")
        return tuple(self.store.list_task_records(generation.id))

    def _terminal_result(
        self, approval: PlanApproval
    ) -> SupervisorResult | None:
        borg = self._turns.current_borg()
        if borg.state not in {
            BorgState.SUPERVISOR_WORKING,
            BorgState.READY_TO_EXECUTE,
            BorgState.BLOCKED,
        }:
            return None
        completed_reviews = self._completed_reviews(approval)
        attempt = next(
            (
                item
                for item in reversed(completed_reviews)
                if (
                    borg.state
                    in {BorgState.SUPERVISOR_WORKING, BorgState.READY_TO_EXECUTE}
                    and (item.result or {}).get("decision") == "approve"
                )
                or (
                    borg.state is BorgState.BLOCKED
                    and (item.result or {}).get("decision") == "request_changes"
                )
            ),
            None,
        )
        if attempt is None:
            return None
        batch_id = attempt.request.get("batch_id")
        batch = next(
            (
                item
                for item in self.store.list_task_batches(self.borg_id)
                if str(item.id) == batch_id
            ),
            None,
        )
        generation_id = attempt.request.get("generation_id")
        generation = next(
            (
                item
                for item in self.store.list_task_generations(self.borg_id)
                if str(item.id) == generation_id
            ),
            None,
        )
        if batch is None or generation is None:
            return None
        publication = None
        if borg.state in {
            BorgState.SUPERVISOR_WORKING,
            BorgState.READY_TO_EXECUTE,
        }:
            current = self.store.get_current_task_generation(self.borg_id)
            publication_is_durable = (
                current is not None and current.id == generation.id
            )
            if (
                publication_is_durable
                and self.cancel is not None
                and self.cancel.is_set()
            ):
                generation = current
            else:
                try:
                    publication = TaskPublisher(
                        self.repository,
                        self.store,
                        cancel=self.cancel,
                    ).reconcile(self.borg_id)
                    if (
                        publication is None
                        or publication.generation.id != generation.id
                    ):
                        return None
                    generation = publication.generation
                except TaskPublicationCancelled as error:
                    raise SupervisorCancelled(_RETAINED_APPROVAL_RESULT) from error
                except TaskPublicationError as error:
                    raise SupervisorError(
                        f"approved task publication failed: {error}"
                    ) from error
            if borg.state is BorgState.SUPERVISOR_WORKING:
                borg = self._turns.transition(borg, BorgState.READY_TO_EXECUTE)
        if (
            borg.state is BorgState.READY_TO_EXECUTE
            and self.store.get_current_task_generation(self.borg_id) != generation
        ):
            return None
        return SupervisorResult(
            borg=borg,
            approval=approval,
            batch=batch,
            generation=generation,
            tasks=tuple(self.store.list_task_records(generation.id)),
            dependencies=tuple(self.store.list_task_dependencies(generation.id)),
            findings=tuple(
                finding
                for finding in self.store.list_task_findings(
                    self.borg_id, batch_id=batch.id
                )
                if finding.attempt_id == attempt.id
            ),
            attempt=attempt,
            publication=publication,
        )


__all__ = [
    "SUPERVISOR_REVIEW_SCHEMA",
    "SUPERVISOR_ROUND_MINIMUM",
    "SupervisorCancelled",
    "SupervisorError",
    "SupervisorLoop",
    "SupervisorResult",
    "supervisor_grant_account",
]
