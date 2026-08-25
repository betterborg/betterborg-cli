"""Deterministic contracts for decomposed task graphs."""

from collections.abc import Iterable
from uuid import UUID, uuid4

import pytest

from betterborg_cli.planning import (
    NonProgressingTaskRepairError,
    TaskGraphFinding,
    TaskGraphValidationError,
    build_plan_element_catalog,
    task_graph_findings,
    validate_task_graph,
    validate_task_repair_progress,
)
from betterborg_cli.store import TaskComplexity, TaskDependency, TaskRecord


def _plan() -> dict:
    return {
        "repositories": [{"id": "repo"}],
        "phases": [
            {
                "name": "01-foundation",
                "goal": "Lay the foundation.",
                "technical_approach": "Add the base contract.",
                "deliverables": ["Foundation"],
                "contracts": [{"spec": "Stable API", "repo": "repo"}],
                "acceptance_criteria": ["Foundation works"],
                "files_touched": [
                    {"path": "foundation.py", "role": "new", "repo": "repo"}
                ],
                "test_strategy": "Run unit tests.",
                "constraints": [],
                "dependencies_on": [],
            },
            {
                "name": "02-consumer",
                "goal": "Use the foundation.",
                "technical_approach": "Build on the stable API.",
                "deliverables": ["Consumer"],
                "contracts": [],
                "acceptance_criteria": ["Consumer works"],
                "files_touched": [
                    {"path": "consumer.py", "role": "new", "repo": "repo"}
                ],
                "test_strategy": "Run integration tests.",
                "constraints": [],
                "dependencies_on": ["01-foundation"],
            },
        ],
    }


def _required_refs(plan: dict, stage: str) -> list[str]:
    return [
        element.ref
        for element in build_plan_element_catalog(plan)
        if element.required and element.phase == stage
    ]


def _task(
    generation_id: UUID,
    *,
    stage: str,
    stem: str,
    position: int,
    refs: Iterable[str],
    complexity: TaskComplexity = TaskComplexity.SMALL,
    declared_complexity: str | None = None,
    repository: str = "repo",
) -> TaskRecord:
    task = {"plan_refs": list(refs), "repository": repository}
    if declared_complexity is not None:
        task["estimate_complexity"] = declared_complexity
    return TaskRecord(
        generation_id=generation_id,
        borg_id=UUID(int=1),
        task_ref=f"task-{position}",
        stage=stage,
        stem=stem,
        position=position,
        title=f"Task {position}",
        complexity=complexity,
        digest=f"sha256:task-{position}",
        task=task,
        manifest={},
    )


def _valid_graph() -> tuple[dict, list[TaskRecord], list[TaskDependency]]:
    plan = _plan()
    generation_id = uuid4()
    foundation = _task(
        generation_id,
        stage="01-foundation",
        stem="01-build",
        position=1,
        refs=_required_refs(plan, "01-foundation"),
    )
    consumer = _task(
        generation_id,
        stage="02-consumer",
        stem="01-build",
        position=2,
        refs=_required_refs(plan, "02-consumer"),
        complexity=TaskComplexity.MEDIUM,
    )
    dependency = TaskDependency(
        generation_id=generation_id,
        task_id=consumer.id,
        depends_on_task_id=foundation.id,
    )
    return plan, [foundation, consumer], [dependency]


def _rules(findings: Iterable[TaskGraphFinding]) -> set[str]:
    return {finding.rule for finding in findings}


def test_complete_graph_has_one_owner_per_required_element() -> None:
    plan, tasks, dependencies = _valid_graph()

    validate_task_graph(plan, tasks, dependencies)

    required = [
        element for element in build_plan_element_catalog(plan) if element.required
    ]
    assert len(required) == 7
    assert not task_graph_findings(plan, tasks, dependencies)


def test_missing_and_duplicate_plan_element_owners_are_rejected() -> None:
    plan = _plan()
    generation_id = uuid4()
    foundation_refs = _required_refs(plan, "01-foundation")
    first = _task(
        generation_id,
        stage="01-foundation",
        stem="01-first",
        position=1,
        refs=foundation_refs[:1],
    )
    second = _task(
        generation_id,
        stage="01-foundation",
        stem="02-second",
        position=2,
        refs=foundation_refs[:1],
    )

    rules = _rules(task_graph_findings(plan, [first, second], []))

    assert "task.traceability.unowned" in rules
    assert "task.traceability.duplicate_owner" in rules


def test_unknown_and_out_of_stage_plan_references_are_rejected() -> None:
    plan, tasks, dependencies = _valid_graph()
    tasks[0].task["plan_refs"].extend(
        ["P99.deliverable.1", _required_refs(plan, "02-consumer")[0]]
    )

    rules = _rules(task_graph_findings(plan, tasks, dependencies))

    assert "task.traceability.unknown_ref" in rules
    assert "task.traceability.boundary" in rules


def test_direct_and_transitive_ancestor_references_are_valid_consumption() -> None:
    plan, tasks, dependencies = _valid_graph()
    ancestor_ref = _required_refs(plan, "01-foundation")[0]
    tasks[1].task["plan_refs"].append(ancestor_ref)

    final_phase = {
        "name": "03-final",
        "goal": "Finish the workflow.",
        "technical_approach": "Use the consumer.",
        "deliverables": ["Final workflow"],
        "contracts": [],
        "acceptance_criteria": ["Workflow is finished"],
        "files_touched": [{"path": "final.py", "role": "new", "repo": "repo"}],
        "test_strategy": "Run end-to-end tests.",
        "constraints": [],
        "dependencies_on": ["02-consumer"],
    }
    plan["phases"].append(final_phase)
    final = _task(
        tasks[0].generation_id,
        stage="03-final",
        stem="01-build",
        position=3,
        refs=[*_required_refs(plan, "03-final"), ancestor_ref],
    )
    dependencies.append(
        TaskDependency(
            generation_id=tasks[0].generation_id,
            task_id=final.id,
            depends_on_task_id=tasks[1].id,
        )
    )

    validate_task_graph(plan, [*tasks, final], dependencies)


def test_task_from_unrelated_repository_cannot_own_phase_elements() -> None:
    plan = {
        "repositories": [{"id": "a"}, {"id": "b"}],
        "phases": [
            {
                "name": "01-foundation",
                "repositories": ["a"],
                "deliverables": ["Foundation"],
                "contracts": [],
                "acceptance_criteria": ["Foundation works"],
                "files_touched": [
                    {"path": "foundation.py", "role": "new", "repo": "a"}
                ],
                "test_strategy": "Run unit tests.",
                "dependencies_on": [],
            }
        ],
    }
    task = _task(
        uuid4(),
        stage="01-foundation",
        stem="01-build",
        position=1,
        refs=_required_refs(plan, "01-foundation"),
        repository="b",
    )

    rules = _rules(task_graph_findings(plan, [task], []))

    assert "task.repository.boundary" in rules
    assert "task.traceability.boundary" in rules
    assert "task.traceability.unowned" in rules


def test_writing_repository_task_can_own_consumed_repository_contract() -> None:
    plan = {
        "repositories": [{"id": "primary"}, {"id": "secondary"}],
        "phases": [
            {
                "name": "01-integration",
                "repositories": ["primary"],
                "deliverables": ["Integration"],
                "contracts": [
                    {
                        "kind": "config",
                        "spec": "secondary.enabled: bool",
                        "repo": "secondary",
                    }
                ],
                "acceptance_criteria": ["Integration works"],
                "files_touched": [
                    {"path": "integration.py", "role": "new", "repo": "primary"},
                    {"path": "settings.py", "role": "read", "repo": "secondary"},
                ],
                "test_strategy": "Run integration tests.",
                "dependencies_on": [],
            }
        ],
    }
    task = _task(
        uuid4(),
        stage="01-integration",
        stem="01-build",
        position=1,
        refs=_required_refs(plan, "01-integration"),
        repository="primary",
    )

    validate_task_graph(plan, [task], [])


def test_dangling_and_forward_same_stage_dependencies_are_rejected() -> None:
    plan, tasks, dependencies = _valid_graph()
    foundation = tasks[0]
    later = _task(
        foundation.generation_id,
        stage="01-foundation",
        stem="02-later",
        position=3,
        refs=[],
    )
    dependencies.extend(
        [
            TaskDependency(
                generation_id=foundation.generation_id,
                task_id=foundation.id,
                depends_on_task_id=later.id,
            ),
            TaskDependency(
                generation_id=foundation.generation_id,
                task_id=foundation.id,
                depends_on_task_id=uuid4(),
            ),
        ]
    )

    rules = _rules(task_graph_findings(plan, [*tasks, later], dependencies))

    assert "task.dependency.same_stage_order" in rules
    assert "task.dependency.dangling" in rules


def test_same_stage_dependency_uses_the_complete_lexical_stem() -> None:
    plan = _plan()
    generation_id = uuid4()
    prerequisite = _task(
        generation_id,
        stage="01-foundation",
        stem="01-a",
        position=1,
        refs=_required_refs(plan, "01-foundation"),
    )
    dependent = _task(
        generation_id,
        stage="01-foundation",
        stem="01-b",
        position=2,
        refs=["P1.goal"],
    )
    dependency = TaskDependency(
        generation_id=generation_id,
        task_id=dependent.id,
        depends_on_task_id=prerequisite.id,
    )

    rules = _rules(task_graph_findings(plan, [prerequisite, dependent], [dependency]))

    assert "task.dependency.same_stage_order" not in rules


def test_dependency_cycles_are_rejected_without_recursion() -> None:
    plan, tasks, dependencies = _valid_graph()
    dependencies.append(
        TaskDependency(
            generation_id=tasks[0].generation_id,
            task_id=tasks[0].id,
            depends_on_task_id=tasks[1].id,
        )
    )

    with pytest.raises(TaskGraphValidationError) as error:
        validate_task_graph(plan, tasks, dependencies)

    assert "task.dependency.cycle" in _rules(error.value.findings)
    assert "task.dependency.phase_inversion" in _rules(error.value.findings)


@pytest.mark.parametrize(
    ("stage", "stem", "declared_complexity", "expected_rule"),
    [
        ("foundation", "01-build", None, "task.stage.invalid"),
        ("01-foundation", "build_task", None, "task.stem.invalid"),
        ("01-foundation", "01-build", "enormous", "task.complexity.invalid"),
    ],
)
def test_invalid_names_and_complexity_are_rejected(
    stage: str,
    stem: str,
    declared_complexity: str | None,
    expected_rule: str,
) -> None:
    plan = _plan()
    task = _task(
        uuid4(),
        stage=stage,
        stem=stem,
        position=1,
        refs=_required_refs(plan, "01-foundation"),
        declared_complexity=declared_complexity,
    )

    assert expected_rule in _rules(task_graph_findings(plan, [task], []))


def test_deterministic_repairs_must_strictly_reduce_stable_findings() -> None:
    missing = TaskGraphFinding(
        rule="task.traceability.unowned",
        message="missing",
        plan_refs=("P1.deliverable.1",),
    )
    dangling = TaskGraphFinding(
        rule="task.dependency.dangling",
        message="dangling",
        task_refs=("task-1",),
    )
    introduced = TaskGraphFinding(
        rule="task.dependency.cycle",
        message="cycle",
        task_refs=("task-1", "task-2"),
    )

    validate_task_repair_progress([missing, dangling], [dangling])
    validate_task_repair_progress([missing], [])
    with pytest.raises(NonProgressingTaskRepairError):
        validate_task_repair_progress([missing, dangling], [missing, dangling])
    with pytest.raises(NonProgressingTaskRepairError):
        validate_task_repair_progress([missing], [introduced])
