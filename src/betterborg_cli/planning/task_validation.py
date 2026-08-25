"""Deterministic validation for decomposed task-generation graphs."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from betterborg_cli.store import TaskComplexity, TaskDependency, TaskRecord

_TASK_NAME = re.compile(r"^[0-9]{2}-[a-z0-9]+(?:-[a-z0-9]+)*$")
_VALID_COMPLEXITIES = frozenset(complexity.value for complexity in TaskComplexity)


@dataclass(frozen=True, slots=True)
class PlanElement:
    """One stable, addressable element of an approved plan."""

    ref: str
    pointer: str
    phase: str
    repository: str | None
    kind: str
    required: bool


@dataclass(frozen=True, slots=True)
class TaskGraphFinding:
    """One deterministic defect in a proposed task graph."""

    rule: str
    message: str
    task_refs: tuple[str, ...] = ()
    plan_refs: tuple[str, ...] = ()

    @property
    def identity(self) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        """Return the stable identity used to measure repair progress."""
        return (
            self.rule,
            tuple(sorted(set(self.task_refs))),
            tuple(sorted(set(self.plan_refs))),
        )


class TaskGraphValidationError(ValueError):
    """Raised when a proposed task graph fails deterministic validation."""

    def __init__(self, findings: Iterable[TaskGraphFinding]) -> None:
        self.findings = tuple(findings)
        details = "; ".join(
            f"{finding.rule}: {finding.message}" for finding in self.findings
        )
        super().__init__(f"task graph validation failed: {details}")


class NonProgressingTaskRepairError(ValueError):
    """Raised when a deterministic repair does not strictly reduce defects."""


def build_plan_element_catalog(plan: Mapping[str, Any]) -> list[PlanElement]:
    """Build stable plan-element references used by task ownership checks.

    Deliverables, contracts, acceptance criteria, and a nonblank test strategy
    are required ownership atoms. File and supporting-prose references remain
    informational so the traceability contract does not duplicate a plan's
    complete file inventory.
    """
    repositories = [
        item.get("id")
        for item in plan.get("repositories", [])
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    ]
    sole_repository = repositories[0] if len(repositories) == 1 else None
    elements: list[PlanElement] = []

    for phase_index, phase in enumerate(plan.get("phases", [])):
        if not isinstance(phase, Mapping) or not isinstance(phase.get("name"), str):
            continue
        stage = phase["name"]
        for field, kind in (
            ("deliverables", "deliverable"),
            ("contracts", "contract"),
            ("acceptance_criteria", "acceptance"),
        ):
            for item_index, item in enumerate(phase.get(field, []) or []):
                repository = item.get("repo") if isinstance(item, Mapping) else None
                elements.append(
                    PlanElement(
                        ref=_plan_ref(phase_index, kind, item_index),
                        pointer=f"/phases/{phase_index}/{field}/{item_index}",
                        phase=stage,
                        repository=repository or sole_repository,
                        kind=kind,
                        required=True,
                    )
                )

        for item_index, item in enumerate(phase.get("files_touched", []) or []):
            if not isinstance(item, Mapping):
                continue
            elements.append(
                PlanElement(
                    ref=_plan_ref(phase_index, "file", item_index),
                    pointer=f"/phases/{phase_index}/files_touched/{item_index}",
                    phase=stage,
                    repository=item.get("repo") or sole_repository,
                    kind="file",
                    required=False,
                )
            )

        test_strategy = phase.get("test_strategy")
        if isinstance(test_strategy, str) and test_strategy.strip():
            elements.append(
                PlanElement(
                    ref=_plan_ref(phase_index, "test"),
                    pointer=f"/phases/{phase_index}/test_strategy",
                    phase=stage,
                    repository=None,
                    kind="test_strategy",
                    required=True,
                )
            )

        for field, kind in (
            ("constraints", "constraint"),
            ("dependencies_on", "dependency"),
        ):
            for item_index, _item in enumerate(phase.get(field, []) or []):
                elements.append(
                    PlanElement(
                        ref=_plan_ref(phase_index, kind, item_index),
                        pointer=f"/phases/{phase_index}/{field}/{item_index}",
                        phase=stage,
                        repository=None,
                        kind=kind,
                        required=False,
                    )
                )

        for field, kind in (("goal", "goal"), ("technical_approach", "approach")):
            value = phase.get(field)
            if isinstance(value, str) and value.strip():
                elements.append(
                    PlanElement(
                        ref=_plan_ref(phase_index, kind),
                        pointer=f"/phases/{phase_index}/{field}",
                        phase=stage,
                        repository=None,
                        kind=kind,
                        required=False,
                    )
                )

    for field, prefix, kind in (
        ("risks", "RISK", "risk"),
        ("code_pointers", "CONTEXT", "code_pointer"),
        ("open_questions", "QUESTION", "open_question"),
    ):
        for item_index, _item in enumerate(plan.get(field, []) or []):
            elements.append(
                PlanElement(
                    ref=f"{prefix}.{item_index + 1}",
                    pointer=f"/{field}/{item_index}",
                    phase="project",
                    repository=None,
                    kind=kind,
                    required=False,
                )
            )

    return elements


def task_graph_findings(
    plan: Mapping[str, Any],
    tasks: Iterable[TaskRecord],
    dependencies: Iterable[TaskDependency],
) -> tuple[TaskGraphFinding, ...]:
    """Return every deterministic defect in a complete proposed generation."""
    task_rows = tuple(tasks)
    dependency_rows = tuple(dependencies)
    findings: list[TaskGraphFinding] = []
    phases = [
        phase
        for phase in plan.get("phases", [])
        if isinstance(phase, Mapping) and isinstance(phase.get("name"), str)
    ]
    phase_order = {phase["name"]: index for index, phase in enumerate(phases)}
    phase_names = set(phase_order)
    declared_dependencies = {
        phase["name"]: [
            dependency
            for dependency in phase.get("dependencies_on", []) or []
            if isinstance(dependency, str) and dependency in phase_names
        ]
        for phase in phases
    }
    ancestors_by_phase = {
        phase_name: _phase_ancestors(phase_name, declared_dependencies)
        for phase_name in phase_names
    }
    repositories_by_phase = {
        phase["name"]: {
            repository
            for repository in phase.get("repositories", []) or []
            if isinstance(repository, str)
        }
        for phase in phases
    }
    catalog = build_plan_element_catalog(plan)
    elements_by_ref = {element.ref: element for element in catalog}
    repositories = {
        item["id"]
        for item in plan.get("repositories", [])
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    sole_repository = next(iter(repositories)) if len(repositories) == 1 else None

    tasks_by_id: dict[UUID, TaskRecord] = {}
    logical_ids: dict[tuple[str, str], TaskRecord] = {}
    task_refs: dict[str, TaskRecord] = {}
    owners: dict[str, list[TaskRecord]] = {}

    for task in task_rows:
        duplicate_id = tasks_by_id.get(task.id)
        if duplicate_id is not None:
            findings.append(
                _dependency_finding(
                    "task.record_id.duplicate",
                    "task record ID is not unique within the generation",
                    duplicate_id,
                    task,
                )
            )
        tasks_by_id[task.id] = task
        _validate_task_identity(
            task,
            phase_order=phase_order,
            logical_ids=logical_ids,
            task_refs=task_refs,
            findings=findings,
        )
        _validate_task_complexity(task, findings)

        task_repository = task.task.get("repository") or sole_repository
        phase_repositories = repositories_by_phase.get(task.stage, set())
        phase_repository_conflict = (
            isinstance(task_repository, str)
            and bool(phase_repositories)
            and task_repository not in phase_repositories
        )
        repository_missing = len(repositories) > 1 and not isinstance(
            task_repository, str
        )
        repository_unknown = (
            task_repository is not None and task_repository not in repositories
        )
        if repository_missing:
            findings.append(
                _finding(
                    "task.repository.missing",
                    "task in a multi-repository plan must name its repository",
                    task,
                )
            )
        elif repository_unknown:
            findings.append(
                _finding(
                    "task.repository.unknown",
                    "task repository is not declared by the approved plan",
                    task,
                )
            )
        elif phase_repository_conflict:
            findings.append(
                _finding(
                    "task.repository.boundary",
                    "task repository is not written by its approved-plan stage",
                    task,
                )
            )
        task_repository_invalid = (
            repository_missing or repository_unknown or phase_repository_conflict
        )

        plan_refs = task.task.get("plan_refs")
        if not isinstance(plan_refs, list) or not plan_refs:
            findings.append(
                _finding(
                    "task.traceability.missing",
                    "task must declare the approved plan elements it owns",
                    task,
                )
            )
            continue
        seen_refs: set[str] = set()
        for plan_ref in plan_refs:
            if not isinstance(plan_ref, str) or plan_ref not in elements_by_ref:
                findings.append(
                    TaskGraphFinding(
                        rule="task.traceability.unknown_ref",
                        message=f"task references unknown plan element {plan_ref!r}",
                        task_refs=(task.task_ref,),
                        plan_refs=(str(plan_ref),),
                    )
                )
                continue
            if plan_ref in seen_refs:
                findings.append(
                    TaskGraphFinding(
                        rule="task.traceability.duplicate_ref",
                        message="task lists the same plan element more than once",
                        task_refs=(task.task_ref,),
                        plan_refs=(plan_ref,),
                    )
                )
                continue
            seen_refs.add(plan_ref)
            element = elements_by_ref[plan_ref]
            repository_conflict = (
                task_repository_invalid
                or (
                    element.repository is not None
                    and task_repository is not None
                    and element.repository != task_repository
                )
            )
            if element.phase != task.stage or repository_conflict:
                consumes_ancestor = (
                    not repository_conflict
                    and element.phase
                    in ancestors_by_phase.get(task.stage, frozenset())
                )
                if consumes_ancestor:
                    # An ancestor citation records consumption of an upstream
                    # contract. It is valid traceability, but only a citation
                    # from the element's own phase can establish ownership.
                    continue
                findings.append(
                    TaskGraphFinding(
                        rule="task.traceability.boundary",
                        message=(
                            "task claims a plan element outside its stage or "
                            "repository boundary"
                        ),
                        task_refs=(task.task_ref,),
                        plan_refs=(plan_ref,),
                    )
                )
                continue
            owners.setdefault(plan_ref, []).append(task)

    for element in catalog:
        if not element.required:
            continue
        element_owners = owners.get(element.ref, [])
        if not element_owners:
            findings.append(
                TaskGraphFinding(
                    rule="task.traceability.unowned",
                    message="required approved-plan element has no valid task owner",
                    plan_refs=(element.ref,),
                )
            )
        elif len(element_owners) > 1:
            findings.append(
                TaskGraphFinding(
                    rule="task.traceability.duplicate_owner",
                    message="required approved-plan element has multiple task owners",
                    task_refs=tuple(sorted(task.task_ref for task in element_owners)),
                    plan_refs=(element.ref,),
                )
            )

    dependencies_by_task: dict[UUID, list[UUID]] = {
        task_id: [] for task_id in tasks_by_id
    }
    seen_edges: set[tuple[UUID, UUID]] = set()
    for dependency in dependency_rows:
        edge = (dependency.task_id, dependency.depends_on_task_id)
        if edge in seen_edges:
            task = tasks_by_id.get(dependency.task_id)
            findings.append(
                TaskGraphFinding(
                    rule="task.dependency.duplicate",
                    message="task dependency is declared more than once",
                    task_refs=(task.task_ref,) if task is not None else (),
                )
            )
            continue
        seen_edges.add(edge)
        task = tasks_by_id.get(dependency.task_id)
        prerequisite = tasks_by_id.get(dependency.depends_on_task_id)
        if task is None or prerequisite is None:
            known = task if task is not None else prerequisite
            findings.append(
                TaskGraphFinding(
                    rule="task.dependency.dangling",
                    message="task dependency does not resolve within the generation",
                    task_refs=(known.task_ref,) if known is not None else (),
                )
            )
            continue
        if (
            dependency.generation_id != task.generation_id
            or dependency.generation_id != prerequisite.generation_id
        ):
            findings.append(
                _dependency_finding(
                    "task.dependency.generation_mismatch",
                    "task dependency crosses generation boundaries",
                    task,
                    prerequisite,
                )
            )
            continue
        dependencies_by_task[task.id].append(prerequisite.id)
        if task.id == prerequisite.id:
            findings.append(
                _finding(
                    "task.dependency.self",
                    "task cannot depend on itself",
                    task,
                )
            )
            continue
        task_phase = phase_order.get(task.stage)
        prerequisite_phase = phase_order.get(prerequisite.stage)
        if (
            task_phase is not None
            and prerequisite_phase is not None
            and prerequisite_phase > task_phase
        ):
            findings.append(
                _dependency_finding(
                    "task.dependency.phase_inversion",
                    "task dependency points to a later approved-plan stage",
                    task,
                    prerequisite,
                )
            )
        elif (
            task.stage == prerequisite.stage
            and prerequisite.stem >= task.stem
        ):
            findings.append(
                _dependency_finding(
                    "task.dependency.same_stage_order",
                    "same-stage dependency must point to a lexically earlier stem",
                    task,
                    prerequisite,
                )
            )

    findings.extend(_cycle_findings(tasks_by_id, dependencies_by_task))
    return tuple(findings)


def validate_task_graph(
    plan: Mapping[str, Any],
    tasks: Iterable[TaskRecord],
    dependencies: Iterable[TaskDependency],
) -> None:
    """Reject a proposed task generation containing deterministic defects."""
    findings = task_graph_findings(plan, tasks, dependencies)
    if findings:
        raise TaskGraphValidationError(findings)


def validate_task_repair_progress(
    previous: Iterable[TaskGraphFinding],
    repaired: Iterable[TaskGraphFinding],
) -> None:
    """Require a repair to remove findings without introducing replacements."""
    previous_identities = {finding.identity for finding in previous}
    repaired_identities = {finding.identity for finding in repaired}
    introduced = repaired_identities - previous_identities
    remaining = repaired_identities & previous_identities
    if introduced or not remaining < previous_identities:
        removed = previous_identities - remaining
        raise NonProgressingTaskRepairError(
            "task repair did not strictly reduce deterministic findings without "
            "introducing new findings "
            f"(removed={len(removed)}, introduced={len(introduced)})"
        )


def _plan_ref(phase_index: int, kind: str, item_index: int | None = None) -> str:
    if item_index is None:
        return f"P{phase_index + 1}.{kind}"
    return f"P{phase_index + 1}.{kind}.{item_index + 1}"


def _phase_ancestors(
    phase_name: str, declared_dependencies: Mapping[str, list[str]]
) -> frozenset[str]:
    ancestors: set[str] = set()
    pending = list(declared_dependencies.get(phase_name, []))
    while pending:
        ancestor = pending.pop()
        if ancestor == phase_name or ancestor in ancestors:
            continue
        ancestors.add(ancestor)
        pending.extend(declared_dependencies.get(ancestor, []))
    return frozenset(ancestors)


def _validate_task_identity(
    task: TaskRecord,
    *,
    phase_order: Mapping[str, int],
    logical_ids: dict[tuple[str, str], TaskRecord],
    task_refs: dict[str, TaskRecord],
    findings: list[TaskGraphFinding],
) -> None:
    for field, value in (("stage", task.stage), ("stem", task.stem)):
        if len(value) > 32 or _TASK_NAME.fullmatch(value) is None:
            findings.append(
                _finding(
                    f"task.{field}.invalid",
                    f"task {field} must use NN-kebab format and be at most "
                    "32 characters",
                    task,
                )
            )
    if task.stage not in phase_order:
        findings.append(
            _finding(
                "task.stage.unknown",
                "task stage is not an approved plan phase",
                task,
            )
        )
    logical_id = (task.stage, task.stem)
    other = logical_ids.get(logical_id)
    if other is not None:
        findings.append(
            _dependency_finding(
                "task.id.duplicate",
                "task stage/stem identity is not unique within the generation",
                other,
                task,
            )
        )
    else:
        logical_ids[logical_id] = task
    other = task_refs.get(task.task_ref)
    if other is not None:
        findings.append(
            _dependency_finding(
                "task.ref.duplicate",
                "task reference is not unique within the generation",
                other,
                task,
            )
        )
    else:
        task_refs[task.task_ref] = task


def _validate_task_complexity(
    task: TaskRecord, findings: list[TaskGraphFinding]
) -> None:
    complexity = (
        task.complexity.value
        if isinstance(task.complexity, TaskComplexity)
        else task.complexity
    )
    declared_complexity = task.task.get("estimate_complexity")
    invalid = complexity not in _VALID_COMPLEXITIES or (
        declared_complexity is not None
        and (
            declared_complexity not in _VALID_COMPLEXITIES
            or declared_complexity != complexity
        )
    )
    if invalid:
        findings.append(
            _finding(
                "task.complexity.invalid",
                "task complexity must consistently be small, medium, or large",
                task,
            )
        )


def _finding(rule: str, message: str, task: TaskRecord) -> TaskGraphFinding:
    return TaskGraphFinding(rule=rule, message=message, task_refs=(task.task_ref,))


def _dependency_finding(
    rule: str,
    message: str,
    task: TaskRecord,
    prerequisite: TaskRecord,
) -> TaskGraphFinding:
    return TaskGraphFinding(
        rule=rule,
        message=message,
        task_refs=tuple(sorted((task.task_ref, prerequisite.task_ref))),
    )


def _cycle_findings(
    tasks_by_id: Mapping[UUID, TaskRecord],
    dependencies_by_task: Mapping[UUID, list[UUID]],
) -> list[TaskGraphFinding]:
    findings: list[TaskGraphFinding] = []
    state: dict[UUID, int] = {}
    for root_id in tasks_by_id:
        if root_id in state:
            continue
        state[root_id] = 1
        path = [root_id]
        path_indexes = {root_id: 0}
        frames: list[tuple[UUID, int]] = [(root_id, 0)]
        while frames:
            task_id, dependency_index = frames[-1]
            dependencies = dependencies_by_task.get(task_id, [])
            if dependency_index >= len(dependencies):
                frames.pop()
                path_indexes.pop(task_id, None)
                path.pop()
                state[task_id] = 2
                continue
            dependency_id = dependencies[dependency_index]
            frames[-1] = (task_id, dependency_index + 1)
            dependency_state = state.get(dependency_id, 0)
            if dependency_state == 0:
                state[dependency_id] = 1
                path_indexes[dependency_id] = len(path)
                path.append(dependency_id)
                frames.append((dependency_id, 0))
            elif dependency_state == 1:
                cycle = path[path_indexes[dependency_id] :]
                findings.append(
                    TaskGraphFinding(
                        rule="task.dependency.cycle",
                        message="task dependency graph contains a cycle",
                        task_refs=tuple(
                            sorted(tasks_by_id[item].task_ref for item in cycle)
                        ),
                    )
                )
    return findings


__all__ = [
    "NonProgressingTaskRepairError",
    "PlanElement",
    "TaskGraphFinding",
    "TaskGraphValidationError",
    "build_plan_element_catalog",
    "task_graph_findings",
    "validate_task_graph",
    "validate_task_repair_progress",
]
