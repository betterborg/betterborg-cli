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


def test_project_context_references_are_valid_non_owning_citations() -> None:
    plan, tasks, dependencies = _valid_graph()
    plan["risks"] = ["The upstream API may change."]
    plan["code_pointers"] = [{"path": "foundation.py", "line": 1}]
    plan["open_questions"] = ["Should the integration be optional?"]
    tasks[1].task["plan_refs"].extend(["RISK.1", "CONTEXT.1", "QUESTION.1"])

    validate_task_graph(plan, tasks, dependencies)


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


def test_repository_contract_must_be_owned_by_its_writing_repository() -> None:
    plan = {
        "repositories": [{"id": "a"}, {"id": "b"}],
        "phases": [
            {
                "name": "01-both",
                "repositories": ["a", "b"],
                "deliverables": ["Both outputs"],
                "contracts": [
                    {"kind": "config", "spec": "b.setting", "repo": "b"}
                ],
                "acceptance_criteria": ["Both work"],
                "files_touched": [
                    {"path": "a.py", "role": "new", "repo": "a"},
                    {"path": "b.py", "role": "new", "repo": "b"},
                ],
                "test_strategy": "Test both.",
                "dependencies_on": [],
            }
        ],
    }
    task = _task(
        uuid4(),
        stage="01-both",
        stem="01-own-all",
        position=1,
        refs=_required_refs(plan, "01-both"),
        repository="a",
    )
    findings = task_graph_findings(plan, [task], [])
    contract_ref = next(
        element.ref
        for element in build_plan_element_catalog(plan)
        if element.kind == "contract"
    )

    assert any(
        finding.rule == "task.traceability.boundary"
        and finding.plan_refs == (contract_ref,)
        for finding in findings
    )
    assert any(
        finding.rule == "task.traceability.unowned"
        and finding.plan_refs == (contract_ref,)
        for finding in findings
    )


def test_every_repository_written_by_a_phase_requires_task_coverage() -> None:
    plan = {
        "repositories": [{"id": "a"}, {"id": "b"}],
        "phases": [
            {
                "name": "01-both",
                "repositories": ["a", "b"],
                "deliverables": ["Both outputs"],
                "contracts": [],
                "acceptance_criteria": ["Both work"],
                "files_touched": [
                    {"path": "a.py", "role": "new", "repo": "a"},
                    {"path": "b.py", "role": "new", "repo": "b"},
                ],
                "test_strategy": "Test both.",
                "dependencies_on": [],
            }
        ],
    }
    task = _task(
        uuid4(),
        stage="01-both",
        stem="01-build-a",
        position=1,
        refs=_required_refs(plan, "01-both"),
        repository="a",
    )

    findings = task_graph_findings(plan, [task], [])

    assert any(
        finding.rule == "task.repository.uncovered"
        and finding.dependency_refs
        == ("stage", "01-both", "repository", "b")
        for finding in findings
    )

    repository_b_file = next(
        element.ref
        for element in build_plan_element_catalog(plan)
        if element.kind == "file" and element.repository == "b"
    )
    second_task = _task(
        task.generation_id,
        stage="01-both",
        stem="02-build-b",
        position=2,
        refs=[repository_b_file],
        repository="b",
    )
    validate_task_graph(plan, [task, second_task], [])


def test_task_records_from_different_generations_are_rejected() -> None:
    plan = _plan()
    first_generation = uuid4()
    foundation_refs = _required_refs(plan, "01-foundation")
    first = _task(
        first_generation,
        stage="01-foundation",
        stem="01-first",
        position=1,
        refs=foundation_refs[:2],
    )
    second = _task(
        uuid4(),
        stage="01-foundation",
        stem="02-second",
        position=2,
        refs=foundation_refs[2:],
    )
    consumer = _task(
        first_generation,
        stage="02-consumer",
        stem="01-build",
        position=3,
        refs=_required_refs(plan, "02-consumer"),
    )

    findings = task_graph_findings(plan, [first, second, consumer], [])

    assert _rules(findings) == {"task.generation.mismatch"}


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


def test_removing_one_of_multiple_dangling_edges_counts_as_repair_progress() -> None:
    plan, tasks, dependencies = _valid_graph()
    first_dangling = TaskDependency(
        generation_id=tasks[0].generation_id,
        task_id=tasks[0].id,
        depends_on_task_id=uuid4(),
    )
    second_dangling = TaskDependency(
        generation_id=tasks[0].generation_id,
        task_id=tasks[0].id,
        depends_on_task_id=uuid4(),
    )
    previous = task_graph_findings(
        plan,
        tasks,
        [*dependencies, first_dangling, second_dangling],
    )
    repaired = task_graph_findings(
        plan,
        tasks,
        [*dependencies, second_dangling],
    )
    dangling_identities = {
        finding.identity
        for finding in previous
        if finding.rule == "task.dependency.dangling"
    }

    assert len(dangling_identities) == 2
    validate_task_repair_progress(previous, repaired)


def test_dangling_edge_identity_survives_generation_reconstruction() -> None:
    plan = _plan()
    previous_generation = uuid4()
    previous_foundation = _task(
        previous_generation,
        stage="01-foundation",
        stem="01-build",
        position=1,
        refs=_required_refs(plan, "01-foundation")[1:],
    )
    previous_consumer = _task(
        previous_generation,
        stage="02-consumer",
        stem="01-build",
        position=2,
        refs=_required_refs(plan, "02-consumer"),
    )
    previous = task_graph_findings(
        plan,
        [previous_foundation, previous_consumer],
        [
            TaskDependency(
                generation_id=previous_generation,
                task_id=previous_consumer.id,
                depends_on_task_id=previous_foundation.id,
            ),
            TaskDependency(
                generation_id=previous_generation,
                task_id=previous_foundation.id,
                depends_on_task_id=uuid4(),
            ),
        ],
    )

    repaired_generation = uuid4()
    repaired_foundation = _task(
        repaired_generation,
        stage="01-foundation",
        stem="01-build",
        position=1,
        refs=_required_refs(plan, "01-foundation"),
    )
    repaired_consumer = _task(
        repaired_generation,
        stage="02-consumer",
        stem="01-build",
        position=2,
        refs=_required_refs(plan, "02-consumer"),
    )
    repaired = task_graph_findings(
        plan,
        [repaired_foundation, repaired_consumer],
        [
            TaskDependency(
                generation_id=repaired_generation,
                task_id=repaired_consumer.id,
                depends_on_task_id=repaired_foundation.id,
            ),
            TaskDependency(
                generation_id=repaired_generation,
                task_id=repaired_foundation.id,
                depends_on_task_id=uuid4(),
            ),
        ],
    )

    assert _rules(previous) == {
        "task.traceability.unowned",
        "task.dependency.dangling",
    }
    assert _rules(repaired) == {"task.dependency.dangling"}
    validate_task_repair_progress(previous, repaired)


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


def test_cycle_identity_survives_generation_reconstruction_and_reordering() -> None:
    plan = _plan()
    plan["phases"].append(
        {
            "name": "03-final",
            "deliverables": ["Final workflow"],
            "contracts": [],
            "acceptance_criteria": ["Workflow is finished"],
            "files_touched": [],
            "test_strategy": "Run end-to-end tests.",
            "dependencies_on": ["02-consumer"],
        }
    )

    def generation_findings(
        generation_id: UUID, *, complete_final: bool, reverse_tasks: bool
    ) -> tuple[TaskGraphFinding, ...]:
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
        )
        final_refs = _required_refs(plan, "03-final")
        final = _task(
            generation_id,
            stage="03-final",
            stem="01-build",
            position=3,
            refs=final_refs if complete_final else final_refs[1:],
        )
        dependencies = [
            TaskDependency(
                generation_id=generation_id,
                task_id=final.id,
                depends_on_task_id=consumer.id,
            ),
            TaskDependency(
                generation_id=generation_id,
                task_id=final.id,
                depends_on_task_id=foundation.id,
            ),
            TaskDependency(
                generation_id=generation_id,
                task_id=consumer.id,
                depends_on_task_id=foundation.id,
            ),
            TaskDependency(
                generation_id=generation_id,
                task_id=foundation.id,
                depends_on_task_id=final.id,
            ),
        ]
        tasks = [foundation, consumer, final]
        if reverse_tasks:
            tasks.reverse()
        return task_graph_findings(plan, tasks, dependencies)

    previous = generation_findings(
        uuid4(), complete_final=False, reverse_tasks=True
    )
    repaired = generation_findings(
        uuid4(), complete_final=True, reverse_tasks=False
    )
    previous_cycles = [
        finding.identity
        for finding in previous
        if finding.rule == "task.dependency.cycle"
    ]
    repaired_cycles = [
        finding.identity
        for finding in repaired
        if finding.rule == "task.dependency.cycle"
    ]

    assert previous_cycles == repaired_cycles == [
        ("task.dependency.cycle", ("task-1", "task-2", "task-3"), (), ())
    ]
    validate_task_repair_progress(previous, repaired)


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


def test_removing_one_of_repeated_duplicate_refs_counts_as_repair_progress() -> None:
    plan, tasks, dependencies = _valid_graph()
    repeated_ref = tasks[0].task["plan_refs"][0]
    tasks[0].task["plan_refs"].extend([repeated_ref, repeated_ref])
    previous = task_graph_findings(plan, tasks, dependencies)
    tasks[0].task["plan_refs"].pop()
    repaired = task_graph_findings(plan, tasks, dependencies)

    assert sum(
        finding.rule == "task.traceability.duplicate_ref" for finding in previous
    ) == 2
    assert sum(
        finding.rule == "task.traceability.duplicate_ref" for finding in repaired
    ) == 1
    validate_task_repair_progress(previous, repaired)


def test_removing_one_of_repeated_dependency_edges_counts_as_progress() -> None:
    plan, tasks, dependencies = _valid_graph()
    previous = task_graph_findings(
        plan,
        tasks,
        [*dependencies, dependencies[0], dependencies[0]],
    )
    repaired = task_graph_findings(
        plan,
        tasks,
        [*dependencies, dependencies[0]],
    )

    assert sum(finding.rule == "task.dependency.duplicate" for finding in previous) == 2
    assert sum(finding.rule == "task.dependency.duplicate" for finding in repaired) == 1
    validate_task_repair_progress(previous, repaired)


def test_moving_duplicate_dependency_to_another_edge_is_not_progress() -> None:
    plan, tasks, dependencies = _valid_graph()
    alternate = _task(
        tasks[0].generation_id,
        stage="01-foundation",
        stem="02-alternate",
        position=3,
        refs=["P1.goal"],
    )
    alternate_dependency = TaskDependency(
        generation_id=tasks[0].generation_id,
        task_id=tasks[1].id,
        depends_on_task_id=alternate.id,
    )
    previous = task_graph_findings(
        plan,
        [*tasks, alternate],
        [*dependencies, dependencies[0], dependencies[0]],
    )
    repaired = task_graph_findings(
        plan,
        [*tasks, alternate],
        [*dependencies, alternate_dependency, alternate_dependency],
    )
    previous_duplicate_refs = [
        finding.dependency_refs
        for finding in previous
        if finding.rule == "task.dependency.duplicate"
    ]
    repaired_duplicate_refs = [
        finding.dependency_refs
        for finding in repaired
        if finding.rule == "task.dependency.duplicate"
    ]

    assert previous_duplicate_refs == [
        ("dependent", "task-2", "prerequisite", "task-1"),
        ("dependent", "task-2", "prerequisite", "task-1"),
    ]
    assert repaired_duplicate_refs == [
        ("dependent", "task-2", "prerequisite", "task-3")
    ]
    with pytest.raises(NonProgressingTaskRepairError):
        validate_task_repair_progress(previous, repaired)
