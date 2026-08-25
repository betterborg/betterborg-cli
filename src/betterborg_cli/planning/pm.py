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
    resolve_agent_model,
)
from betterborg_cli.planning.task_validation import (
    TaskGraphValidationError,
    build_plan_element_catalog,
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
    TaskComplexity,
    TaskDependency,
    TaskGeneration,
    TaskRecord,
)

PM_OUTPUT_RETRY_CAP = 3
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
                        "pattern": "^[0-9]{2}-[a-z0-9]+(?:-[a-z0-9]+)*$",
                        "minLength": 4,
                        "maxLength": 32,
                    },
                    "stem": {
                        "type": "string",
                        "pattern": "^[0-9]{2}-[a-z0-9]+(?:-[a-z0-9]+)*$",
                        "minLength": 4,
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
                            "pattern": (
                                "^[0-9]{2}-[a-z0-9]+(?:-[a-z0-9]+)*/"
                                "[0-9]{2}-[a-z0-9]+(?:-[a-z0-9]+)*$"
                            ),
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
in this batch. Return only the required JSON object.
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


class ProjectManagerLoop:
    """Generate and persist a complete task batch for one approved plan."""

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
        dirty_borg_documents: Sequence[Path] = (),
        worktrees_root: Path | None = None,
    ) -> None:
        require_read_only_agent(
            agent, role="Project Manager", error_factory=ProjectManagerError
        )
        try:
            resolved_model = resolve_agent_model(agent, model)
        except AgentSelectionError as error:
            raise ProjectManagerError(str(error)) from error

        paths = RepoPaths.discover(repository.root)
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
            dirty_borg_documents=dirty_borg_documents,
            worktrees_root=worktrees_root,
        )

    def run(self) -> ProjectManagerResult:
        """Resume or generate until a complete validated batch is persisted."""
        approval = self._approval()
        plan = self._approved_plan(approval)
        terminal = self._terminal_result(approval)
        if terminal is not None:
            return terminal

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
                "findings": [
                    {
                        "message": finding.message,
                        "round": finding.round,
                        "severity": finding.severity,
                        "suggestion": finding.suggestion,
                        "task_ref": finding.task_ref,
                    }
                    for finding in self.store.list_task_findings(
                        self.borg_id, batch_id=base_batch.id
                    )
                ],
                "summary": base_batch.summary,
                "tasks": [
                    task.task for task in self._tasks_for_batch(base_batch)
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
            if base_batch is not None and [task.task for task in tasks] == [
                task.task for task in self._tasks_for_batch(base_batch)
            ]:
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
                if len(self._attempts_for_cycle(approval, base_batch)) >= (
                    PM_OUTPUT_RETRY_CAP
                ):
                    raise ProjectManagerError(
                        "Project Manager exhausted revision retries: " + summary
                    )
                continue
            try:
                validate_task_graph(plan, tasks, dependencies)
            except TaskGraphValidationError as error:
                summary = str(error)
                self.store.complete_planning_attempt(
                    attempt.id,
                    status=PlanningAttemptStatus.FAILED,
                    result=payload,
                    summary=summary,
                )
                if len(self._attempts_for_cycle(approval, base_batch)) >= (
                    PM_OUTPUT_RETRY_CAP
                ):
                    raise ProjectManagerError(
                        "Project Manager exhausted output retries: " + summary
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
            ".borg/state/planning/context/manifest.json. Emit one complete "
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
        findings = self.store.list_task_findings(
            self.borg_id, batch_id=base_batch.id
        )
        if not findings:
            return "Revise the complete prior task batch without dropping coverage."
        return "Supervisor findings: " + "; ".join(
            f"{finding.severity}: {finding.message}"
            + (f" ({finding.suggestion})" if finding.suggestion else "")
            for finding in findings
        )

    def _require_retry_budget(
        self, approval: PlanApproval, base_batch: TaskBatch | None
    ) -> None:
        attempts = self._attempts_for_cycle(approval, base_batch)
        if len(attempts) < PM_OUTPUT_RETRY_CAP or not attempts:
            return
        latest = attempts[-1]
        if latest.status is not PlanningAttemptStatus.FAILED:
            return
        kind = "revision" if base_batch is not None else "output"
        raise ProjectManagerError(
            f"Project Manager exhausted {kind} retries: "
            + (latest.summary or "last attempt failed")
        )

    def _retryable_contract_failure(
        self, approval: PlanApproval, base_batch: TaskBatch | None
    ) -> bool:
        attempts = self._attempts_for_cycle(approval, base_batch)
        if len(attempts) >= PM_OUTPUT_RETRY_CAP or not attempts:
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
            digest = approved_plan_digest(task)
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
                manifest={"approved_plan_digest": approval.plan_digest},
            )
            tasks.append(record)
            logical_tasks[f"{record.stage}/{record.stem}"] = record
            task_manifest.append(
                {
                    "digest": digest,
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
    "PM_OUTPUT_RETRY_CAP",
    "PROJECT_MANAGER_TASKS_SCHEMA",
    "ProjectManagerCancelled",
    "ProjectManagerError",
    "ProjectManagerLoop",
    "ProjectManagerResult",
    "approved_plan_digest",
]
