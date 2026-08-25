"""Durable Supervisor review and bounded Project Manager revision lifecycle."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from betterborg_cli.agent_runtime.base import AgentAdapter, CancellationToken
from betterborg_cli.agent_runtime.selection import (
    AgentSelectionError,
    SelectedAgent,
    resolve_agent_model,
)
from betterborg_cli.planning.pm import (
    ProjectManagerCancelled,
    ProjectManagerError,
    ProjectManagerLoop,
    approved_plan_digest,
    task_batch_semantic_digest,
)
from betterborg_cli.planning.task_publication import (
    TaskPublicationError,
    TaskPublisher,
)
from betterborg_cli.planning.task_validation import (
    TaskGraphValidationError,
    validate_task_graph,
)
from betterborg_cli.planning.turns import DurablePlanningTurns, require_read_only_agent
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.store import (
    Borg,
    BorgState,
    PlanApproval,
    PlanningAttempt,
    PlanningAttemptStatus,
    Repository,
    SqliteStore,
    TaskBatch,
    TaskDependency,
    TaskFinding,
    TaskGeneration,
    TaskGenerationStatus,
    TaskRecord,
)

SUPERVISOR_ROUND_CAP = 3
_SUPERVISOR_PHASE = "supervisor_review"

_NONBLANK_STRING: dict[str, Any] = {
    "type": "string",
    "minLength": 1,
    "pattern": r"\S",
}

SUPERVISOR_REVIEW_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "summary", "findings"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approve", "request_changes"],
        },
        "summary": _NONBLANK_STRING,
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["severity", "message"],
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["blocker", "major", "minor"],
                    },
                    "message": _NONBLANK_STRING,
                    "suggestion": _NONBLANK_STRING,
                    "task_ref": _NONBLANK_STRING,
                },
            },
        },
    },
}

_SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor reviewing a complete,
deterministically valid Project Manager task batch for an approved plan. Judge
plan coverage, task coherence, foundation ownership, reuse instead of
duplication, dependency ordering, meaningful tests, delivery scope, and
simplicity. Approve only a complete batch that is ready for publication. Do not
modify files or redesign the batch; return actionable findings for the Project
Manager. Return only the required JSON object.
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


class SupervisorLoop:
    """Review valid PM batches and request no more than three PM cycles."""

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
        dirty_borg_documents: Sequence[Path] = (),
        worktrees_root: Path | None = None,
    ) -> None:
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

        paths = RepoPaths.discover(repository.root)
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
        self.dirty_borg_documents = tuple(dirty_borg_documents)
        self.worktrees_root = worktrees_root
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
            dirty_borg_documents=dirty_borg_documents,
            worktrees_root=worktrees_root,
        )

    def run(self) -> SupervisorResult:
        """Resume review and PM revision turns until approval or exhaustion."""
        approval = self._approval()
        plan = self._approved_plan(approval)
        terminal = self._terminal_result(approval)
        if terminal is not None:
            return terminal

        while True:
            borg = self._turns.current_borg()
            if borg.state is BorgState.PM_WORKING:
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
                        dirty_borg_documents=self.dirty_borg_documents,
                        worktrees_root=self.worktrees_root,
                    ).run()
                except ProjectManagerCancelled as error:
                    raise SupervisorCancelled(str(error)) from error
                except ProjectManagerError as error:
                    raise SupervisorError(str(error)) from error
                if revised.borg.state is not BorgState.SUPERVISOR_WORKING:
                    raise SupervisorError(
                        "Project Manager revision did not return to Supervisor"
                    )
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
            if review_round > SUPERVISOR_ROUND_CAP:
                raise SupervisorError(
                    "Supervisor review round cap was already exhausted"
                )
            attempt, payload = self._turns.run(
                phase=_SUPERVISOR_PHASE,
                round_number=self._turns.next_round(_SUPERVISOR_PHASE),
                schema=SUPERVISOR_REVIEW_SCHEMA,
                system_prompt=_SUPERVISOR_SYSTEM_PROMPT,
                user_prompt=(
                    "Read the complete approved plan and task-review context from "
                    ".borg/state/planning/context/manifest.json. Review task batch "
                    f"{batch.id} in round {review_round} of "
                    f"{SUPERVISOR_ROUND_CAP}."
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
                findings = self._findings(payload, attempt, batch, tasks, review_round)
            except SupervisorError as error:
                self.store.complete_planning_attempt(
                    attempt.id,
                    status=PlanningAttemptStatus.FAILED,
                    result=payload,
                    summary=str(error),
                )
                raise

            decision = payload["decision"]
            if decision == "approve":
                next_state = BorgState.TASKS_APPROVAL_PENDING
            elif review_round < SUPERVISOR_ROUND_CAP:
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
                borg = self._turns.transition(borg, next_state)

            if decision == "approve":
                try:
                    generation = (
                        TaskPublisher(self.repository, self.store)
                        .publish(generation.id)
                        .generation
                    )
                except TaskPublicationError as error:
                    raise SupervisorError(
                        f"approved task publication failed: {error}"
                    ) from error

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
                )

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
    ) -> tuple[TaskFinding, ...]:
        raw_findings = payload["findings"]
        if payload["decision"] == "request_changes" and not raw_findings:
            raise SupervisorError("Supervisor request_changes must include findings")
        known_refs = {task.task_ref for task in tasks}
        findings: list[TaskFinding] = []
        for item in raw_findings:
            task_ref = item.get("task_ref")
            if task_ref is not None and task_ref not in known_refs:
                raise SupervisorError(
                    f"Supervisor finding references unknown task {task_ref!r}"
                )
            findings.append(
                TaskFinding(
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
            )
        actionable = any(
            finding.severity in {"blocker", "major"} for finding in findings
        )
        if payload["decision"] == "approve" and actionable:
            raise SupervisorError(
                "Supervisor cannot approve with blocker or major findings"
            )
        if payload["decision"] == "request_changes" and not actionable:
            raise SupervisorError(
                "Supervisor request_changes requires a blocker or major finding"
            )
        return tuple(findings)

    def _completed_reviews(self, approval: PlanApproval) -> list[PlanningAttempt]:
        return [
            attempt
            for attempt in self._turns.attempts(_SUPERVISOR_PHASE)
            if attempt.status is PlanningAttemptStatus.COMPLETED
            and attempt.request.get("plan_approval_id") == str(approval.id)
            and attempt.request.get("approved_plan_digest") == approval.plan_digest
        ]

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
            BorgState.TASKS_APPROVAL_PENDING,
            BorgState.BLOCKED,
        }:
            return None
        completed_reviews = self._completed_reviews(approval)
        if (
            borg.state is BorgState.BLOCKED
            and len(completed_reviews) < SUPERVISOR_ROUND_CAP
        ):
            return None
        attempt = next(
            (
                item
                for item in reversed(completed_reviews)
                if (
                    borg.state is BorgState.TASKS_APPROVAL_PENDING
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
        if borg.state is BorgState.TASKS_APPROVAL_PENDING:
            try:
                publication = TaskPublisher(
                    self.repository, self.store
                ).reconcile(self.borg_id)
                if publication is None or publication.generation.id != generation.id:
                    return None
                generation = publication.generation
            except TaskPublicationError as error:
                raise SupervisorError(
                    f"approved task publication failed: {error}"
                ) from error
        if (
            borg.state is BorgState.TASKS_APPROVAL_PENDING
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
        )


__all__ = [
    "SUPERVISOR_REVIEW_SCHEMA",
    "SUPERVISOR_ROUND_CAP",
    "SupervisorCancelled",
    "SupervisorError",
    "SupervisorLoop",
    "SupervisorResult",
]
