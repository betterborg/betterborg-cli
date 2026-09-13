"""Digest-bound, resumable Project Manager task decomposition."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from betterborg_cli.agent_runtime.base import AgentAdapter, CancellationToken
from betterborg_cli.agent_runtime.selection import (
    AgentSelectionError,
    SelectedAgent,
    require_read_only_agent,
    resolve_agent_model,
)
from betterborg_cli.planning.findings_ledger import (
    open_task_findings,
    task_ledger_json,
)
from betterborg_cli.planning.grants import (
    PLANNING_GRANT_BUDGET,
    assess_grant,
)
from betterborg_cli.planning.task_render import (
    render_task_markdown,
    task_markdown_digest,
)
from betterborg_cli.planning.task_validation import (
    TASK_NAME_PATTERN,
    TASK_REFERENCE_PATTERN,
    NonProgressingTaskRepairError,
    TaskGraphFinding,
    TaskGraphValidationError,
    build_plan_element_catalog,
    task_graph_findings,
    validate_task_graph,
    validate_task_repair_progress,
)
from betterborg_cli.planning.turns import (
    DurablePlanningTurns,
    planning_attempt_duration,
    planning_attempt_result,
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
    TaskComplexity,
    TaskDependency,
    TaskGeneration,
    TaskRecord,
)

#: Attempts a Project Manager cycle gets at a valid task graph before its
#: grants begin. Past it an attempt is free when it strictly reduced the set of
#: deterministic findings left by the last attempt of its cycle that produced
#: any, and introduced none of its own; every other attempt spends a grant.
PM_OUTPUT_RETRY_MINIMUM = 3
_PM_PHASE = "pm_tasks"

_NONBLANK_STRING: dict[str, Any] = {
    "type": "string",
    "minLength": 1,
    "pattern": r"\S",
}
_NONBLANK_STRINGS: dict[str, Any] = {
    "type": "array",
    "items": _NONBLANK_STRING,
}

PROJECT_MANAGER_TASKS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["tasks"],
    "properties": {
        "summary": _NONBLANK_STRING,
        "tasks": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "stage",
                    "stem",
                    "title",
                    "why",
                    "scope",
                    "implementation_notes",
                    "acceptance_criteria",
                    "tests",
                    "dependencies",
                    "out_of_scope",
                    "plan_refs",
                    "estimate_complexity",
                ],
                "properties": {
                    "stage": {
                        "type": "string",
                        "pattern": TASK_NAME_PATTERN,
                        "maxLength": 32,
                    },
                    "stem": {
                        "type": "string",
                        "pattern": TASK_NAME_PATTERN,
                        "maxLength": 32,
                    },
                    "repository": _NONBLANK_STRING,
                    "title": {**_NONBLANK_STRING, "maxLength": 120},
                    "why": _NONBLANK_STRING,
                    "scope": {**_NONBLANK_STRINGS, "minItems": 1},
                    "implementation_notes": _NONBLANK_STRINGS,
                    "acceptance_criteria": {
                        **_NONBLANK_STRINGS,
                        "minItems": 1,
                    },
                    "tests": {**_NONBLANK_STRINGS, "minItems": 1},
                    "dependencies": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "pattern": TASK_REFERENCE_PATTERN,
                        },
                    },
                    "out_of_scope": _NONBLANK_STRINGS,
                    "plan_refs": {
                        **_NONBLANK_STRINGS,
                        "minItems": 1,
                        "uniqueItems": True,
                    },
                    "estimate_complexity": {
                        "type": "string",
                        "enum": ["small", "medium", "large"],
                    },
                },
            },
        },
    },
}

_PROJECT_MANAGER_SYSTEM_PROMPT = """You are the Project Manager for an approved
implementation plan. Decompose the whole plan into concrete, independently
shippable coding tasks. Do not modify files. Every task must stand alone with a
specific rationale, scope, notes, acceptance criteria, tests, dependencies,
exclusions, plan references, and S/M/L complexity. Assign every required plan
reference to exactly one task and use only dependency task identities present
in this batch. Within one stage, a task may depend only on a task whose stem
sorts before its own, so number the stems of a stage in the order they must
run. Betterborg performs the delivery around your tasks: branching, worktrees,
commits, review, merge, and the repository's own checks. Never write a task for
any of that. Every task changes the repository, and one with nothing to commit
is not a task. A revision is given every Supervisor objection its batch still
owes an answer for, including ones raised before the batch you are revising,
because a later review saying nothing about an objection is not agreement. Where
one names a task, that is the task it was first raised against: a label on where
the objection started, which may name a task no current batch holds, so answer
the objection rather than looking its reference up. Return only the required JSON
object.
"""


class ProjectManagerError(RuntimeError):
    """Raised when PM decomposition cannot produce a durable valid batch."""


class ProjectManagerCancelled(ProjectManagerError):
    """Raised after preserving enough PM state to resume later."""


@dataclass(frozen=True, slots=True)
class ProjectManagerResult:
    """One persisted PM batch and its immutable preparing generation."""

    borg: Borg
    approval: PlanApproval
    batch: TaskBatch
    generation: TaskGeneration
    tasks: tuple[TaskRecord, ...]
    dependencies: tuple[TaskDependency, ...]
    attempt: PlanningAttempt


def approved_plan_digest(plan: Mapping[str, Any]) -> str:
    """Return the canonical digest used to bind approval and PM records."""
    encoded = json.dumps(
        plan,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def task_batch_semantic_digest(tasks: Sequence[Mapping[str, Any]]) -> str:
    """Return an order-insensitive digest of a batch's task content."""

    def canonicalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: canonicalize(item) for key, item in value.items()}
        if isinstance(value, list):
            items = [canonicalize(item) for item in value]
            return sorted(
                items,
                key=lambda item: json.dumps(
                    item,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            )
        return value

    canonical_tasks = canonicalize(list(tasks))
    return approved_plan_digest({"tasks": canonical_tasks})


def _retry_kind(base_batch: TaskBatch | None) -> str:
    """Name the kind of attempt a cycle has run out of.

    One wording for one stop, wherever the loop notices it: a cycle with a batch
    to revise ran out of revisions, and the first cycle of an approval ran out of
    attempts at an output.
    """

    return "revision" if base_batch is not None else "output"


def _repair_progressed(
    previous: Sequence[TaskGraphFinding],
    repaired: Sequence[TaskGraphFinding],
) -> bool:
    """Whether an attempt earned its refund by repairing the one before it.

    The test itself is the product's own, and stricter than a falling count: an
    attempt progresses by removing at least one of the findings left for it
    while introducing none. An attempt with nothing to compare against has
    removed nothing and earns nothing: a cycle's first attempt is one such, and
    so is any attempt whose cycle has yet to produce a finding at all.
    """

    try:
        validate_task_repair_progress(previous, repaired)
    except NonProgressingTaskRepairError:
        return False
    return True


class ProjectManagerLoop:
    """Persist one approved plan's task batch while the grant budget holds."""

    def __init__(
        self,
        repository: Repository,
        borg: Borg,
        store: SqliteStore,
        agent: AgentAdapter | SelectedAgent,
        *,
        approved_plan: Mapping[str, Any] | None = None,
        plan_approval: PlanApproval | None = None,
        artifact_dir: Path | None = None,
        model: str | None = None,
        cancel: CancellationToken | None = None,
        progress: RunProgress | None = None,
        stage_key: str = "project-manager",
        child_key: str | None = None,
        dirty_borg_documents: Sequence[Path] = (),
        worktrees_root: Path | None = None,
        output_retries: int = PM_OUTPUT_RETRY_MINIMUM,
        grant_budget: int = PLANNING_GRANT_BUDGET,
    ) -> None:
        if cancel is not None and cancel.is_set():
            raise ProjectManagerCancelled("Project Manager run cancelled")
        if not isinstance(output_retries, int) or output_retries < 1:
            raise ProjectManagerError(
                "Project Manager output retries must be a whole number of at "
                "least 1"
            )
        # Zero buys nothing and is legal; below zero would stop the cycle short
        # of the minimum it was told to run.
        if not isinstance(grant_budget, int) or grant_budget < 0:
            raise ProjectManagerError(
                "Project Manager grant budget must be a whole number of at least 0"
            )
        require_read_only_agent(
            agent, role="Project Manager", error_factory=ProjectManagerError
        )
        try:
            resolved_model = resolve_agent_model(agent, model)
        except AgentSelectionError as error:
            raise ProjectManagerError(str(error)) from error

        paths = RepoPaths.discover(repository.root, cancel=cancel)
        if paths.root != repository.root:
            raise ValueError("repository root does not match its discovered Git root")
        self.repository = repository
        self.borg_id = borg.id
        self.store = store
        self.agent = agent
        self._supplied_plan = dict(approved_plan) if approved_plan is not None else None
        self._supplied_approval = plan_approval
        self.artifact_dir = Path(
            artifact_dir or paths.artifacts_dir / "planning" / str(borg.id)
        ).resolve()
        self.model = resolved_model
        self.cancel = cancel
        self.progress = progress
        self.stage_key = stage_key
        self.child_key = child_key
        self.output_retries = output_retries
        self.grant_budget = grant_budget
        if progress is not None:
            if child_key is None and stage_key not in progress.stages:
                progress.declare(StageSpec(stage_key, "Project Manager"))
            elif child_key is not None:
                if stage_key not in progress.stages:
                    raise ValueError(
                        "Project Manager revision parent must already be declared"
                    )
                if child_key not in progress.stages[stage_key].children:
                    progress.declare_child(
                        stage_key,
                        ChildSpec(child_key, "Project Manager revision"),
                    )
        self._turns = DurablePlanningTurns(
            repository,
            borg,
            store,
            agent,
            role="Project Manager",
            model=resolved_model,
            artifact_dir=self.artifact_dir,
            error_factory=ProjectManagerError,
            cancelled_error_factory=ProjectManagerCancelled,
            cancel=cancel,
            progress=progress,
            stage_key=stage_key if progress is not None else None,
            child_key=child_key if progress is not None else None,
            dirty_borg_documents=dirty_borg_documents,
            worktrees_root=worktrees_root,
        )

    def run(self) -> ProjectManagerResult:
        """Resume or generate until a complete validated batch is persisted."""
        try:
            approval = self._approval()
            terminal = self._terminal_result(approval)
            if terminal is not None:
                self._seed_progress(terminal.attempt)
                return terminal

            self._start_progress()
            result = self._run(approval)
            self._complete_progress(result.attempt)
            return result
        except (ProjectManagerCancelled, KeyboardInterrupt) as error:
            self._reconcile_progress(str(error), stopped=True)
            raise
        except Exception as error:
            self._reconcile_progress(
                str(error),
                stopped=self.cancel is not None and self.cancel.is_set(),
            )
            raise

    def _run(self, approval: PlanApproval) -> ProjectManagerResult:
        """Execute fresh Project Manager work after its progress record starts."""
        plan = self._approved_plan(approval)

        borg = self._turns.current_borg()
        if borg.state is BorgState.PLAN_APPROVAL_PENDING:
            borg = self._turns.transition(borg, BorgState.PM_WORKING)
        elif borg.state is not BorgState.PM_WORKING:
            raise ProjectManagerError(
                f"Borg {borg.name!r} cannot run Project Manager from state "
                f"{borg.state.value!r}"
            )

        base_batch = self._latest_batch(approval)
        annotated_plan = self._annotated_plan(plan)
        if base_batch is not None:
            annotated_plan["_betterborg_task_revision"] = {
                "batch_digest": base_batch.digest,
                "batch_id": str(base_batch.id),
                # The open ledger, not this batch's snapshot: carry-forward
                # keeps an objection alive, so an objection raised two rounds
                # ago and not repeated since is one this revision still owes
                # an answer for.
                "findings": [
                    task_ledger_json(row)
                    for row in open_task_findings(
                        self.store, self.borg_id, approval.id
                    )
                ],
                "summary": base_batch.summary,
                "tasks": [
                    {"task_ref": task.task_ref, "task": task.task}
                    for task in self._tasks_for_batch(base_batch)
                ],
            }
        while True:
            if self.cancel is not None and self.cancel.is_set():
                raise ProjectManagerCancelled("Project Manager run cancelled")
            self._require_retry_budget(approval, base_batch)
            feedback = self._latest_feedback(approval, base_batch)
            request_context = {
                "plan_approval_id": str(approval.id),
                "approved_plan_digest": approval.plan_digest,
            }
            if base_batch is not None:
                request_context.update(
                    {
                        "base_batch_id": str(base_batch.id),
                        "base_batch_digest": base_batch.digest,
                    }
                )
            try:
                attempt, payload = self._turns.run(
                    phase=_PM_PHASE,
                    round_number=self._turns.next_round(_PM_PHASE),
                    schema=PROJECT_MANAGER_TASKS_SCHEMA,
                    system_prompt=_PROJECT_MANAGER_SYSTEM_PROMPT,
                    user_prompt=self._user_prompt(feedback),
                    current_plan=json.dumps(
                        annotated_plan, indent=2, sort_keys=True
                    ),
                    turn_name="task batch",
                    request_context=request_context,
                )
            except ProjectManagerCancelled:
                raise
            except ProjectManagerError:
                if self._retryable_contract_failure(approval, base_batch):
                    continue
                raise

            generation, batch, tasks, dependencies = self._materialize_graph(
                approval, attempt, payload
            )
            if base_batch is not None and task_batch_semantic_digest(
                [task.task for task in tasks]
            ) == task_batch_semantic_digest(
                [task.task for task in self._tasks_for_batch(base_batch)]
            ):
                summary = (
                    "Project Manager revision made no semantic progress against "
                    f"task batch {base_batch.id}"
                )
                self.store.complete_planning_attempt(
                    attempt.id,
                    status=PlanningAttemptStatus.FAILED,
                    result=payload,
                    summary=summary,
                )
                # A batch that already passed validation leaves no findings to
                # weigh, so this attempt records nothing and an attempt with no
                # record earned no refund.
                if not self._cycle_continues(approval, base_batch):
                    raise ProjectManagerError(
                        f"Project Manager exhausted {_retry_kind(base_batch)} "
                        f"retries: {summary}"
                    )
                continue
            try:
                validate_task_graph(plan, tasks, dependencies)
            except TaskGraphValidationError as error:
                summary = str(error)
                attempts = self._attempts_for_cycle(approval, base_batch)
                previous = self._comparable_findings(
                    approval,
                    plan,
                    [item for item in attempts if item.id != attempt.id],
                )
                progressed = _repair_progressed(previous, error.findings)
                grant = assess_grant(
                    review_round=len(attempts),
                    minimum=self.output_retries,
                    budget=self.grant_budget,
                    progressed=progressed,
                    recorded=self._assessments_for_cycle(approval, base_batch),
                )
                # One durable step for the attempt and what it earned: a refund
                # recorded without its attempt would pay for work that never
                # happened, and an attempt completed without its refund is one
                # the guard charges for on the next entry.
                with self.store.transaction():
                    self.store.complete_planning_attempt(
                        attempt.id,
                        status=PlanningAttemptStatus.FAILED,
                        result=payload,
                        summary=summary,
                    )
                    self.store.record_review_assessment(
                        self._assessment(
                            approval,
                            base_batch,
                            attempt,
                            round_number=len(attempts),
                            progressed=progressed,
                            refunded=grant.refunded,
                            previous=previous,
                            findings=error.findings,
                        )
                    )
                if not grant.continues:
                    raise ProjectManagerError(
                        f"Project Manager exhausted {_retry_kind(base_batch)} "
                        f"retries: {summary}"
                    ) from error
                continue

            with self.store.transaction():
                summary = str(
                    payload.get("summary") or f"{len(tasks)} generated task(s)"
                ).strip()
                completed = self.store.complete_planning_attempt(
                    attempt.id,
                    status=PlanningAttemptStatus.COMPLETED,
                    result=payload,
                    summary=summary,
                )
                self.store.append_task_batch(batch)
                self.store.add_task_generation(generation, tasks, dependencies)
                borg = self._turns.transition(borg, BorgState.SUPERVISOR_WORKING)
            return ProjectManagerResult(
                borg=borg,
                approval=approval,
                batch=batch,
                generation=generation,
                tasks=tasks,
                dependencies=dependencies,
                attempt=completed,
            )

    def _start_progress(self) -> None:
        if self.progress is None:
            return
        if self.child_key is None:
            record = self.progress.stages[self.stage_key]
            if record.state is StageState.PENDING:
                self.progress.start(self.stage_key)
            elif record.state is not StageState.RUNNING:
                raise ProjectManagerError(
                    f"Project Manager progress {self.stage_key!r} is already terminal"
                )
            return
        child = self.progress.stages[self.stage_key].children[self.child_key]
        if child.state is StageState.PENDING:
            self.progress.start_child(self.stage_key, self.child_key)
        elif child.state is not StageState.RUNNING:
            raise ProjectManagerError(
                "Project Manager revision progress "
                f"{self.child_key!r} is already terminal"
            )

    def _seed_progress(self, attempt: PlanningAttempt) -> None:
        if self.progress is None:
            return
        result = planning_attempt_result(attempt, default="task batch ready")
        duration = planning_attempt_duration(attempt)
        if self.child_key is None:
            record = self.progress.stages[self.stage_key]
            if record.state is StageState.PENDING:
                self.progress.seed_completed(self.stage_key, result, duration)
            return
        child = self.progress.stages[self.stage_key].children[self.child_key]
        if child.state is StageState.PENDING:
            self.progress.seed_child_completed(
                self.stage_key, self.child_key, result, duration
            )

    def _complete_progress(self, attempt: PlanningAttempt) -> None:
        if self.progress is None:
            return
        result = planning_attempt_result(attempt, default="task batch ready")
        if self.child_key is None:
            self.progress.complete(self.stage_key, result)
        else:
            self.progress.complete_child(self.stage_key, self.child_key, result)

    def _reconcile_progress(self, result: str, *, stopped: bool) -> None:
        if self.progress is None:
            return
        record = (
            self.progress.stages[self.stage_key]
            if self.child_key is None
            else self.progress.stages[self.stage_key].children[self.child_key]
        )
        if record.state is not StageState.RUNNING:
            return
        approval = self._approval()
        terminal = self._terminal_result(approval)
        if terminal is not None:
            self._complete_progress(terminal.attempt)
        elif self.child_key is None:
            operation = self.progress.stop if stopped else self.progress.fail
            operation(self.stage_key, result)
        else:
            operation = (
                self.progress.stop_child if stopped else self.progress.fail_child
            )
            operation(self.stage_key, self.child_key, result)

    def _approval(self) -> PlanApproval:
        approvals = self.store.list_plan_approvals(self.borg_id)
        if not approvals:
            raise ProjectManagerError("Project Manager requires an approved plan")
        approval = self._supplied_approval or approvals[-1]
        if approval not in approvals:
            raise ProjectManagerError("supplied plan approval is not persisted")
        if approval != approvals[-1]:
            raise ProjectManagerError(
                "Project Manager requires the latest plan approval"
            )
        return approval

    def _approved_plan(self, approval: PlanApproval) -> dict[str, Any]:
        plan = self._supplied_plan
        if plan is None:
            manifest_plan = approval.manifest.get("plan")
            if isinstance(manifest_plan, dict):
                plan = dict(manifest_plan)
            else:
                plan = self._approved_architect_plan(approval.plan_digest)
        if plan is None:
            raise ProjectManagerError(
                "approved plan content is not available in planning history"
            )
        digest = approved_plan_digest(plan)
        if digest != approval.plan_digest:
            raise ProjectManagerError(
                "approved plan digest mismatch: "
                f"approval has {approval.plan_digest!r}, content has {digest!r}"
            )
        return plan

    def _approved_architect_plan(self, digest: str) -> dict[str, Any] | None:
        for attempt in reversed(self.store.list_planning_attempts(self.borg_id)):
            if (
                attempt.phase == "architect_plan"
                and attempt.status is PlanningAttemptStatus.COMPLETED
                and attempt.result is not None
                and approved_plan_digest(attempt.result) == digest
            ):
                return dict(attempt.result)
        return None

    @staticmethod
    def _annotated_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
        annotated = dict(plan)
        annotated["_betterborg_plan_refs"] = [
            dataclasses.asdict(element) for element in build_plan_element_catalog(plan)
        ]
        return annotated

    @staticmethod
    def _user_prompt(feedback: str | None) -> str:
        prompt = (
            "Read the approved plan named by current_plan in "
            ".betterborg/state/planning/context/manifest.json. Emit one complete "
            "project-wide task batch covering every approved phase and required "
            "_betterborg_plan_refs element."
        )
        if feedback is not None:
            prompt += " Repair the previous rejected output: " + feedback
        return prompt

    def _attempts_for(self, approval: PlanApproval) -> list[PlanningAttempt]:
        return [
            item
            for item in self._turns.attempts(_PM_PHASE)
            if item.request.get("plan_approval_id") == str(approval.id)
            and item.request.get("approved_plan_digest") == approval.plan_digest
        ]

    def _attempts_for_cycle(
        self, approval: PlanApproval, base_batch: TaskBatch | None
    ) -> list[PlanningAttempt]:
        base_batch_id = str(base_batch.id) if base_batch is not None else None
        return [
            item
            for item in self._attempts_for(approval)
            if item.request.get("base_batch_id") == base_batch_id
        ]

    def _latest_feedback(
        self, approval: PlanApproval, base_batch: TaskBatch | None
    ) -> str | None:
        failure = next(
            (
                item.summary
                for item in reversed(self._attempts_for_cycle(approval, base_batch))
                if item.status is PlanningAttemptStatus.FAILED and item.summary
            ),
            None,
        )
        if failure is not None:
            return failure
        if base_batch is None:
            return None
        rows = open_task_findings(self.store, self.borg_id, approval.id)
        if not rows:
            return "Revise the complete prior task batch without dropping coverage."
        # A reference names the task the objection was first raised against.
        # For this round's own findings that is a task of the batch being
        # revised; for a carried one the batch that held it is gone, which is
        # why it is labelled rather than offered as a reference to use.
        return "Supervisor findings: " + "; ".join(
            f"{row.severity}"
            + (f" [raised against {row.task_ref}]" if row.task_ref else "")
            + f": {row.message}"
            + (f" ({row.suggestion})" if row.suggestion else "")
            for row in rows
        )

    def _require_retry_budget(
        self, approval: PlanApproval, base_batch: TaskBatch | None
    ) -> None:
        """Refuse the next attempt of a cycle whose last one failed with no
        grants left.

        Read on every entry, which is what a resumed run enters through, so it
        reads the records the attempts left rather than judging them again: an
        attempt whose record is missing earned no refund, which is how the
        failures with nothing to weigh are counted without a rule of their own.

        A cycle whose last attempt did not fail is left alone, as it was before
        there was a budget: an interrupted attempt is one the turn machinery
        resumes rather than one this loop refuses, and a cycle whose grants are
        spent is stopped by the attempt after it instead.
        """

        attempts = self._attempts_for_cycle(approval, base_batch)
        if not attempts:
            return
        latest = attempts[-1]
        if latest.status is not PlanningAttemptStatus.FAILED:
            return
        if self._cycle_continues(approval, base_batch, attempts=attempts):
            return
        raise ProjectManagerError(
            f"Project Manager exhausted {_retry_kind(base_batch)} retries: "
            + (latest.summary or "last attempt failed")
        )

    def _cycle_continues(
        self,
        approval: PlanApproval,
        base_batch: TaskBatch | None,
        *,
        attempts: Sequence[PlanningAttempt] | None = None,
    ) -> bool:
        """Whether this cycle's unrefunded attempts leave room for another.

        Where the reads with nothing of their own to weigh get their answer. The
        validation branch reaches the same answer from the same rule, in the call
        that also decides what its attempt earned, so the count and the comparand
        are one everywhere: attempts, minus the refunds they recorded, against
        the minimum plus the budget.
        """

        if attempts is None:
            attempts = self._attempts_for_cycle(approval, base_batch)
        recorded = self._assessments_for_cycle(approval, base_batch)
        latest = next(
            (item for item in recorded if item.round == len(attempts)), None
        )
        return assess_grant(
            review_round=len(attempts),
            minimum=self.output_retries,
            budget=self.grant_budget,
            progressed=latest is not None and bool(latest.refunded),
            recorded=recorded,
        ).continues

    def _assessments_for_cycle(
        self, approval: PlanApproval, base_batch: TaskBatch | None
    ) -> list[ReviewAssessment]:
        """Return what this cycle's attempts recorded, and no other cycle's.

        Scoped the way the attempt count is scoped, because an assessment of a
        revision cycle is evidence about the batch it revised and nothing about
        the batch before it.
        """

        batch_id = None if base_batch is None else base_batch.id
        return [
            item
            for item in self.store.list_review_assessments(
                self.borg_id, loop=_PM_PHASE, plan_approval_id=approval.id
            )
            if item.batch_id == batch_id
        ]

    def _assessment(
        self,
        approval: PlanApproval,
        base_batch: TaskBatch | None,
        attempt: PlanningAttempt,
        *,
        round_number: int,
        progressed: bool,
        refunded: bool | None,
        previous: Sequence[TaskGraphFinding],
        findings: Sequence[TaskGraphFinding],
    ) -> ReviewAssessment:
        """Build the record of what one validation failure showed."""

        return ReviewAssessment(
            borg_id=self.borg_id,
            loop=_PM_PHASE,
            # A plan approval and the batch being revised, which is the cycle
            # the attempt count already respects. A first cycle revises no
            # batch and names none.
            plan_approval_id=approval.id,
            batch_id=None if base_batch is None else base_batch.id,
            attempt_id=attempt.id,
            round=round_number,
            minimum=self.output_retries,
            # This loop's own test and the only question it asks: its findings
            # come from a validator rather than an argument, so there is no
            # reviewer's verdict to read and no attempt of it is ever steered.
            converging=progressed,
            # No snapshot, because the test is a comparison of finding sets and
            # not a count. A zero here would read as an attempt that left
            # nothing open, which is the one thing a failed validation is not.
            open_findings=None,
            refunded=refunded,
            # The two sides the test compared, rather than the delta between
            # them: the rule that subtracts them has one owner, and a stored
            # copy of its arithmetic is a second answer waiting to disagree.
            evidence={
                "previous": [finding.identity for finding in previous],
                "repaired": [finding.identity for finding in findings],
            },
        )

    def _comparable_findings(
        self,
        approval: PlanApproval,
        plan: Mapping[str, Any],
        attempts: Sequence[PlanningAttempt],
    ) -> tuple[TaskGraphFinding, ...]:
        """Recompute the findings of the last of these attempts that had any.

        The findings are a pure function of the payload an attempt persisted and
        the approved plan, so a resumed run recomputes them rather than reading
        a set it never stored. An attempt that produced none is skipped rather
        than compared against: an unparseable response leaves nothing to
        recompute from, and a revision that changed nothing semantically matched
        a batch that had already passed validation, so charging the next attempt
        for landing after either would charge it for the gap.
        """

        for attempt in reversed(attempts):
            findings = self._recomputed_findings(approval, plan, attempt)
            if findings:
                return findings
        return ()

    def _recomputed_findings(
        self,
        approval: PlanApproval,
        plan: Mapping[str, Any],
        attempt: PlanningAttempt,
    ) -> tuple[TaskGraphFinding, ...]:
        """Recompute one attempt's deterministic findings from its payload."""

        payload = attempt.result
        if not isinstance(payload, Mapping):
            return ()
        try:
            _, _, tasks, dependencies = self._materialize_graph(
                approval, attempt, dict(payload)
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            # A response that never parsed is already handled above, so what
            # reaches here is a payload the schema admitted that still will not
            # build a graph. Evidence that cannot be read is evidence of
            # nothing: the attempt removed nothing, and earns nothing.
            return ()
        return task_graph_findings(plan, tasks, dependencies)

    def _retryable_contract_failure(
        self, approval: PlanApproval, base_batch: TaskBatch | None
    ) -> bool:
        """Whether a malformed response is worth one more attempt.

        A response this loop could not parse records nothing, so the attempt it
        spent is one the budget counts, and a cycle that cannot stop producing
        them stops where every other kind of grinding stops.
        """

        attempts = self._attempts_for_cycle(approval, base_batch)
        if not attempts or not self._cycle_continues(
            approval, base_batch, attempts=attempts
        ):
            return False
        latest = attempts[-1]
        summary = (latest.summary or "").casefold()
        return latest.status is PlanningAttemptStatus.FAILED and (
            "structured result validation failed" in summary
            or "invalid structured result" in summary
            or "unable to extract" in summary
            or "no parseable json" in summary
        )

    def _latest_batch(self, approval: PlanApproval) -> TaskBatch | None:
        return next(
            (
                item
                for item in reversed(self.store.list_task_batches(self.borg_id))
                if item.plan_approval_id == approval.id
            ),
            None,
        )

    def _tasks_for_batch(self, batch: TaskBatch) -> tuple[TaskRecord, ...]:
        generation = next(
            (
                item
                for item in reversed(self.store.list_task_generations(self.borg_id))
                if item.batch_id == batch.id
            ),
            None,
        )
        if generation is None:
            raise ProjectManagerError(
                f"task batch {batch.id} has no durable generation"
            )
        return tuple(self.store.list_task_records(generation.id))

    def _materialize_graph(
        self,
        approval: PlanApproval,
        attempt: PlanningAttempt,
        payload: dict[str, Any],
    ) -> tuple[
        TaskGeneration,
        TaskBatch,
        tuple[TaskRecord, ...],
        tuple[TaskDependency, ...],
    ]:
        generation_id = uuid4()
        tasks: list[TaskRecord] = []
        logical_tasks: dict[str, TaskRecord] = {}
        task_manifest: list[dict[str, Any]] = []
        for position, raw_task in enumerate(payload["tasks"], start=1):
            task = dict(raw_task)
            task_id = uuid4()
            digest = task_markdown_digest(render_task_markdown(task))
            record = TaskRecord(
                id=task_id,
                generation_id=generation_id,
                borg_id=self.borg_id,
                task_ref=f"T-{task_id.hex}",
                stage=task["stage"],
                stem=task["stem"],
                position=position,
                title=task["title"],
                complexity=TaskComplexity(task["estimate_complexity"]),
                digest=digest,
                task=task,
                manifest={
                    "approved_plan_digest": approval.plan_digest,
                    "task.md": digest,
                },
            )
            tasks.append(record)
            logical_tasks[f"{record.stage}/{record.stem}"] = record
            task_manifest.append(
                {
                    "digest": digest,
                    "path": (
                        f".betterborg/tasks/{self._turns.current_borg().name}/"
                        f"{generation_id}/{record.stage}/{record.stem}.md"
                    ),
                    "position": position,
                    "task_ref": record.task_ref,
                }
            )

        dependencies: list[TaskDependency] = []
        dependency_refs: list[tuple[str, str]] = []
        for task in tasks:
            for raw_dependency in task.task["dependencies"]:
                prerequisite = logical_tasks.get(raw_dependency)
                dependencies.append(
                    TaskDependency(
                        generation_id=generation_id,
                        task_id=task.id,
                        depends_on_task_id=(
                            prerequisite.id if prerequisite is not None else uuid4()
                        ),
                    )
                )
                dependency_refs.append(
                    (
                        task.task_ref,
                        prerequisite.task_ref if prerequisite else raw_dependency,
                    )
                )

        batch_manifest = {
            "approved_plan_digest": approval.plan_digest,
            "plan_approval_id": str(approval.id),
            "tasks": task_manifest,
        }
        batch = TaskBatch(
            borg_id=self.borg_id,
            plan_approval_id=approval.id,
            attempt_id=attempt.id,
            round=len(self.store.list_task_batches(self.borg_id)) + 1,
            summary=str(
                payload.get("summary") or f"{len(tasks)} generated task(s)"
            ).strip(),
            digest=approved_plan_digest(payload),
            manifest=batch_manifest,
        )
        generation_manifest = {
            **batch_manifest,
            "batch_digest": batch.digest,
            "dependencies": [
                {
                    "depends_on_task_ref": prerequisite_ref,
                    "task_ref": task_ref,
                }
                for task_ref, prerequisite_ref in dependency_refs
            ],
        }
        generation = TaskGeneration(
            id=generation_id,
            borg_id=self.borg_id,
            plan_approval_id=approval.id,
            batch_id=batch.id,
            digest=approved_plan_digest(generation_manifest),
            manifest=generation_manifest,
        )
        return generation, batch, tuple(tasks), tuple(dependencies)

    def _terminal_result(
        self, approval: PlanApproval
    ) -> ProjectManagerResult | None:
        if self._turns.current_borg().state is BorgState.PM_WORKING:
            return None
        batch = next(
            (
                item
                for item in reversed(self.store.list_task_batches(self.borg_id))
                if item.plan_approval_id == approval.id
            ),
            None,
        )
        if batch is None or batch.attempt_id is None:
            return None
        generation = next(
            (
                item
                for item in reversed(self.store.list_task_generations(self.borg_id))
                if item.batch_id == batch.id
            ),
            None,
        )
        attempt = next(
            (
                item
                for item in self.store.list_planning_attempts(self.borg_id)
                if item.id == batch.attempt_id
                and item.status is PlanningAttemptStatus.COMPLETED
            ),
            None,
        )
        if generation is None or attempt is None:
            return None
        return ProjectManagerResult(
            borg=self._turns.current_borg(),
            approval=approval,
            batch=batch,
            generation=generation,
            tasks=tuple(self.store.list_task_records(generation.id)),
            dependencies=tuple(self.store.list_task_dependencies(generation.id)),
            attempt=attempt,
        )


__all__ = [
    "PM_OUTPUT_RETRY_MINIMUM",
    "PROJECT_MANAGER_TASKS_SCHEMA",
    "ProjectManagerCancelled",
    "ProjectManagerError",
    "ProjectManagerLoop",
    "ProjectManagerResult",
    "approved_plan_digest",
    "task_batch_semantic_digest",
]
