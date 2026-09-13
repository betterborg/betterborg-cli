"""Deterministic and agent lifecycle contracts for task decomposition."""

import json
from collections.abc import Iterable
from dataclasses import replace
from io import StringIO
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from betterborg_cli.agent_runtime.base import CancellationToken
from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.agent_runtime.retry import DEFAULT_SCHEMA_MAX_ATTEMPTS
from betterborg_cli.agent_runtime.structured import (
    StructuredResultError,
    validate_structured_result,
)
from betterborg_cli.planning import (
    SUPERVISOR_REVIEW_SCHEMA,
    NonProgressingTaskRepairError,
    ProjectManagerError,
    ProjectManagerLoop,
    SupervisorCancelled,
    SupervisorError,
    SupervisorLoop,
    TaskGraphFinding,
    TaskGraphValidationError,
    TaskPublisher,
    approved_plan_digest,
    build_plan_element_catalog,
    task_graph_findings,
    validate_task_graph,
    validate_task_repair_progress,
)
from betterborg_cli.planning.findings_ledger import open_task_findings
from betterborg_cli.planning.supervisor import SUPERVISOR_DECISION_ROUND_CAP
from betterborg_cli.progress import RunProgress, StageState
from betterborg_cli.store import (
    Borg,
    BorgState,
    FindingStatus,
    PlanApproval,
    PlanningAttempt,
    PlanningAttemptStatus,
    SqliteStore,
    TaskBatch,
    TaskComplexity,
    TaskDependency,
    TaskGenerationStatus,
    TaskLedgerFinding,
    TaskRecord,
)


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


def _pm_payload(plan: dict, *, revision: str = "") -> dict:
    def task(
        stage: str,
        stem: str,
        refs: list[str],
        *,
        dependencies: list[str],
        complexity: str,
    ) -> dict:
        return {
            "stage": stage,
            "stem": stem,
            "repository": "repo",
            "title": f"Build {stage}{revision}",
            "why": "This task owns one independently testable plan slice.",
            "scope": [f"Implement the concrete {stage} deliverable."],
            "implementation_notes": [],
            "acceptance_criteria": [f"The {stage} behavior works."],
            "tests": [f"Cover the {stage} behavior with a focused test."],
            "dependencies": dependencies,
            "out_of_scope": [],
            "plan_refs": refs,
            "estimate_complexity": complexity,
        }

    return {
        "summary": "Two dependency-ordered tasks cover the approved plan.",
        "tasks": [
            task(
                "01-foundation",
                "01-build",
                _required_refs(plan, "01-foundation"),
                dependencies=[],
                complexity="small",
            ),
            task(
                "02-consumer",
                "01-build",
                _required_refs(plan, "02-consumer"),
                dependencies=["01-foundation/01-build"],
                complexity="medium",
            ),
        ],
    }


def _approve_plan(
    store: SqliteStore, borg: Borg, plan: dict
) -> tuple[PlanApproval, Borg]:
    approval = PlanApproval(
        borg_id=borg.id,
        plan_digest=approved_plan_digest(plan),
        manifest={"plan.json": approved_plan_digest(plan)},
        approved_by="test operator",
    )
    store.append_plan_approval(approval)
    approved_borg = store.compare_and_set_borg_state(
        borg.id,
        expected_state=borg.state,
        expected_version=borg.state_version,
        new_state=BorgState.PLAN_APPROVAL_PENDING,
    )
    return approval, approved_borg


def _planning_context(spec) -> dict:
    manifest = json.loads(
        (spec.cwd / ".betterborg/state/planning/context/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    return json.loads(
        (spec.cwd / manifest["current_plan"]).read_text(encoding="utf-8")
    )


def _review_response(
    decision: str, message: str = "The first task needs a narrower scope."
):
    def respond(spec):
        context = _planning_context(spec)
        task_ref = context["task_batch"]["tasks"][0]["task_ref"]
        findings = []
        if decision == "request_changes":
            findings.append(
                {
                    "severity": "major",
                    "message": message,
                    "suggestion": "Keep the task independently testable.",
                    "task_ref": task_ref,
                    "repeats": None,
                }
            )
        return {
            "decision": decision,
            "summary": f"Supervisor decided to {decision}.",
            "findings": findings,
            "resolved": [],
        }

    return respond


def _rules(findings: Iterable[TaskGraphFinding]) -> set[str]:
    return {finding.rule for finding in findings}


def test_a_validation_failure_names_what_it_is_about() -> None:
    """A rule alone asks the reader to search; the reference asks them to act.

    The Project Manager repairs its own rejected batch from this text and
    nothing else, so a finding that keeps the element to itself spends the
    retry budget on guessing which one it meant.
    """
    error = TaskGraphValidationError(
        [
            TaskGraphFinding(
                rule="task.traceability.unowned",
                message="required approved-plan element has no valid task owner",
                plan_refs=("phase/02-loader/deliverable/1",),
            ),
            TaskGraphFinding(
                rule="task.dependency.same_stage_order",
                message="same-stage dependency must point to a lexically earlier stem",
                task_refs=("02-loader/01-cache",),
                dependency_refs=("02-loader/09-reset",),
            ),
        ]
    )

    detail = str(error)

    assert "phase/02-loader/deliverable/1" in detail
    assert "02-loader/01-cache" in detail
    assert "02-loader/09-reset" in detail


def test_a_validation_failure_without_references_reads_as_before() -> None:
    error = TaskGraphValidationError(
        [TaskGraphFinding(rule="task.batch.empty", message="batch has no tasks")]
    )

    assert str(error) == (
        "task graph validation failed: task.batch.empty: batch has no tasks"
    )


def test_pm_generates_complete_digest_bound_batch_and_persists_attempt(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    plan = _plan()
    payload = _pm_payload(plan)

    def complete_batch(spec):
        manifest = json.loads(
            (
                spec.cwd / ".betterborg/state/planning/context/manifest.json"
            ).read_text(encoding="utf-8")
        )
        annotated_plan = json.loads(
            (spec.cwd / manifest["current_plan"]).read_text(encoding="utf-8")
        )
        required_refs = {
            item["ref"]
            for item in annotated_plan["_betterborg_plan_refs"]
            if item["required"]
        }
        assert required_refs == {
            *payload["tasks"][0]["plan_refs"],
            *payload["tasks"][1]["plan_refs"],
        }
        # The same-stage ordering rule is enforced on the output, so the PM is
        # told it before writing rather than discovering it by rejection: it
        # constrains how stems are named, which is not recoverable by editing
        # one dependency.
        instruction = " ".join(spec.system_prompt.split())
        assert "a task may depend only on a task whose stem sorts before" in (
            instruction
        )
        return payload

    adapter = MockAdapter(name="openai").queue(MockResponse(dynamic=complete_batch))
    database = committed_git_repo.parent / "pm-complete.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "pm-complete"
        )
        plan_attempt = PlanningAttempt(
            borg_id=borg.id,
            phase="architect_plan",
            round=1,
            adapter="mock",
            model="test-model",
        )
        store.append_planning_attempt(plan_attempt)
        store.complete_planning_attempt(
            plan_attempt.id,
            status=PlanningAttemptStatus.COMPLETED,
            result=plan,
            summary="Approved plan candidate.",
        )
        approval, borg = _approve_plan(store, borg, plan)

        result = ProjectManagerLoop(
            repository,
            borg,
            store,
            adapter,
        ).run()

        assert result.borg.state is BorgState.SUPERVISOR_WORKING
        assert result.batch.plan_approval_id == approval.id
        assert result.batch.manifest["approved_plan_digest"] == approval.plan_digest
        assert (
            result.generation.manifest["approved_plan_digest"]
            == approval.plan_digest
        )
        assert [task.task for task in result.tasks] == payload["tasks"]
        assert [task.complexity for task in result.tasks] == [
            TaskComplexity.SMALL,
            TaskComplexity.MEDIUM,
        ]
        assert len(result.dependencies) == 1
        validate_task_graph(plan, result.tasks, result.dependencies)
        attempts = [
            item
            for item in store.list_planning_attempts(borg.id)
            if item.phase == "pm_tasks"
        ]
        assert len(attempts) == 1
        assert attempts[0].status is PlanningAttemptStatus.COMPLETED
        assert attempts[0].result == payload
        assert attempts[0].request["plan_approval_id"] == str(approval.id)
        assert (
            attempts[0].request["approved_plan_digest"] == approval.plan_digest
        )

    with SqliteStore.open(database) as reopened:
        persisted = [
            item
            for item in reopened.list_planning_attempts(borg.id)
            if item.phase == "pm_tasks"
        ]
        assert persisted[0].status is PlanningAttemptStatus.COMPLETED
        assert persisted[0].result == payload
        assert len(reopened.list_task_batches(borg.id)) == 1


def test_pm_retries_malformed_output_with_persisted_feedback(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    plan = _plan()
    malformed = _pm_payload(plan)
    malformed["tasks"][0].pop("tests")

    def repaired_batch(spec):
        assert "Repair the previous rejected output" in spec.user_prompt
        assert "structured result validation failed" in spec.user_prompt
        return _pm_payload(plan)

    adapter = MockAdapter(name="openai")
    for _attempt in range(DEFAULT_SCHEMA_MAX_ATTEMPTS):
        adapter.queue(MockResponse(payload=malformed))
    adapter.queue(MockResponse(dynamic=repaired_batch))
    database = committed_git_repo.parent / "pm-retry.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "pm-retry"
        )
        _approval, borg = _approve_plan(store, borg, plan)

        result = ProjectManagerLoop(
            repository,
            borg,
            store,
            adapter,
            approved_plan=plan,
        ).run()

        assert result.borg.state is BorgState.SUPERVISOR_WORKING
        assert len(adapter.calls) == DEFAULT_SCHEMA_MAX_ATTEMPTS + 1
        attempts = store.list_planning_attempts(borg.id)
        assert [item.status for item in attempts] == [
            PlanningAttemptStatus.FAILED,
            PlanningAttemptStatus.COMPLETED,
        ]
        assert attempts[0].result is None
        assert "structured result validation failed" in (attempts[0].summary or "")
        assert attempts[1].result == _pm_payload(plan)


def test_pm_resumes_completed_provider_turn_without_replay(
    committed_git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    persist_planning_context,
) -> None:
    plan = _plan()
    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload=_pm_payload(plan))
    )
    database = committed_git_repo.parent / "pm-resume.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "pm-resume"
        )
        _approval, borg = _approve_plan(store, borg, plan)
        loop = ProjectManagerLoop(
            repository,
            borg,
            store,
            adapter,
            approved_plan=plan,
        )
        original_complete = store.complete_planning_attempt
        interrupted = False

        def interrupt_after_result(attempt_id, **kwargs):
            nonlocal interrupted
            attempt = next(
                item
                for item in store.list_planning_attempts(borg.id)
                if item.id == attempt_id
            )
            if (
                attempt.phase == "pm_tasks"
                and kwargs["status"] is PlanningAttemptStatus.COMPLETED
                and not interrupted
            ):
                interrupted = True
                raise RuntimeError("simulated terminal interruption")
            return original_complete(attempt_id, **kwargs)

        with monkeypatch.context() as interruption:
            interruption.setattr(
                store, "complete_planning_attempt", interrupt_after_result
            )
            with pytest.raises(RuntimeError, match="terminal interruption"):
                loop.run()

        running = store.list_planning_attempts(borg.id)[-1]
        assert running.status is PlanningAttemptStatus.RUNNING
        assert Path(running.request["result_path"]).is_file()
        assert store.list_task_batches(borg.id) == []
        assert len(adapter.calls) == 1

        resumed = loop.run()

        assert resumed.borg.state is BorgState.SUPERVISOR_WORKING
        assert len(adapter.calls) == 1
        assert loop.run() == resumed
        assert len(adapter.calls) == 1


def test_pm_rejects_plan_content_that_does_not_match_approval_digest(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    plan = _plan()
    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload=_pm_payload(plan))
    )
    database = committed_git_repo.parent / "pm-plan-binding.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "pm-plan-binding"
        )
        _approval, borg = _approve_plan(store, borg, plan)
        changed_plan = _plan()
        changed_plan["phases"][0]["deliverables"] = ["Changed foundation"]

        with pytest.raises(ProjectManagerError, match="digest mismatch"):
            ProjectManagerLoop(
                repository,
                borg,
                store,
                adapter,
                approved_plan=changed_plan,
            ).run()

        assert adapter.calls == []
        assert store.list_planning_attempts(borg.id) == []
        assert store.list_task_batches(borg.id) == []


def test_supervisor_approves_one_validated_batch_for_publication_handoff(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    plan = _plan()
    database = committed_git_repo.parent / "supervisor-approve.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "supervisor-approve"
        )
        approval, borg = _approve_plan(store, borg, plan)
        pm_result = ProjectManagerLoop(
            repository,
            borg,
            store,
            MockAdapter(name="openai").queue(
                MockResponse(payload=_pm_payload(plan))
            ),
            approved_plan=plan,
        ).run()
        supervisor = MockAdapter(name="openai").queue(
            MockResponse(dynamic=_review_response("approve"))
        )
        loop = SupervisorLoop(
            repository,
            pm_result.borg,
            store,
            supervisor,
            approved_plan=plan,
        )

        result = loop.run()

        assert result.borg.state is BorgState.TASKS_APPROVAL_PENDING
        assert result.approval == approval
        assert result.batch == pm_result.batch
        assert result.generation.status is TaskGenerationStatus.CURRENT
        assert store.get_current_task_generation(borg.id) == result.generation
        assert len(supervisor.calls) == 1
        assert loop.run() == result
        assert len(supervisor.calls) == 1


def test_supervisor_restart_reconciles_current_publication_after_commit(
    committed_git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    persist_planning_context,
) -> None:
    plan = _plan()
    database = committed_git_repo.parent / "supervisor-publication-reconcile.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "supervisor-reconcile"
        )
        _approval, borg = _approve_plan(store, borg, plan)
        initial = ProjectManagerLoop(
            repository,
            borg,
            store,
            MockAdapter(name="openai").queue(
                MockResponse(payload=_pm_payload(plan))
            ),
            approved_plan=plan,
        ).run()
        stale = (
            committed_git_repo
            / ".betterborg/tasks/supervisor-reconcile"
            / str(uuid4())
        )
        stale.mkdir(parents=True)
        (stale / "prior.md").write_text("# Prior generation\n", encoding="utf-8")
        supervisor = MockAdapter(name="openai").queue(
            MockResponse(dynamic=_review_response("approve"))
        )
        original_checkpoint = TaskPublisher._checkpoint

        def interrupt_after_commit(self, point: str) -> None:
            if point == "after_db_commit":
                raise RuntimeError("simulated post-commit crash")
            original_checkpoint(self, point)

        with monkeypatch.context() as interruption:
            interruption.setattr(TaskPublisher, "_checkpoint", interrupt_after_commit)
            with pytest.raises(RuntimeError, match="post-commit crash"):
                SupervisorLoop(
                    repository,
                    initial.borg,
                    store,
                    supervisor,
                    approved_plan=plan,
                ).run()

        current = store.get_current_task_generation(borg.id)
        assert current is not None
        assert stale.is_dir()
        assert len(supervisor.calls) == 1

    with SqliteStore.open(database) as reopened:
        resumed_supervisor = MockAdapter(name="openai")
        resumed_borg = reopened.get_borg(borg.id)
        assert resumed_borg is not None

        result = SupervisorLoop(
            repository,
            resumed_borg,
            reopened,
            resumed_supervisor,
            approved_plan=plan,
        ).run()

        assert result.generation.id == current.id
        assert [path.name for path in stale.parent.iterdir()] == [str(current.id)]
        assert resumed_supervisor.calls == []


def test_supervisor_publication_cancellation_retains_approval_and_resumes(
    committed_git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    persist_planning_context,
) -> None:
    plan = _plan()
    database = committed_git_repo.parent / "supervisor-publication-cancel.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "supervisor-publication-cancel"
        )
        _approval, borg = _approve_plan(store, borg, plan)
        initial = ProjectManagerLoop(
            repository,
            borg,
            store,
            MockAdapter(name="openai").queue(
                MockResponse(payload=_pm_payload(plan))
            ),
            approved_plan=plan,
        ).run()
        supervisor = MockAdapter(name="openai").queue(
            MockResponse(dynamic=_review_response("approve"))
        )
        cancel = CancellationToken()
        interrupted_progress = RunProgress(stream=StringIO())
        original_checkpoint = TaskPublisher._checkpoint

        def cancel_before_commit(self, point: str) -> None:
            original_checkpoint(self, point)
            if point == "before_db_commit":
                cancel.cancel()

        with monkeypatch.context() as interruption:
            interruption.setattr(
                TaskPublisher, "_checkpoint", cancel_before_commit
            )
            with pytest.raises(
                SupervisorCancelled,
                match="approval retained; task publication pending",
            ):
                SupervisorLoop(
                    repository,
                    initial.borg,
                    store,
                    supervisor,
                    approved_plan=plan,
                    cancel=cancel,
                    progress=interrupted_progress,
                ).run()

        persisted_borg = store.get_borg(borg.id)
        assert persisted_borg is not None
        assert persisted_borg.state is BorgState.SUPERVISOR_WORKING
        attempts = [
            attempt
            for attempt in store.list_planning_attempts(borg.id)
            if attempt.phase == "supervisor_review"
        ]
        assert len(attempts) == 1
        assert attempts[0].status is PlanningAttemptStatus.COMPLETED
        generation = store.list_task_generations(borg.id)[0]
        assert generation.status is TaskGenerationStatus.PREPARING
        assert len(supervisor.calls) == 1
        interrupted_supervisor = interrupted_progress.stages["supervisor"]
        assert interrupted_supervisor.state is StageState.STOPPED
        assert interrupted_supervisor.detail == "publishing approved tasks"
        assert interrupted_supervisor.result == (
            "approval retained; task publication pending"
        )
        interrupted_progress.close()

        resumed_supervisor = MockAdapter(name="openai")
        resumed_progress = RunProgress(stream=StringIO())
        publication_progress: list[tuple[StageState, str | None]] = []

        def observe_resumed_publication(self, point: str) -> None:
            if point == "before_db_commit":
                record = resumed_progress.stages["supervisor"]
                publication_progress.append((record.state, record.detail))
            original_checkpoint(self, point)

        monkeypatch.setattr(
            TaskPublisher, "_checkpoint", observe_resumed_publication
        )
        resumed = SupervisorLoop(
            repository,
            persisted_borg,
            store,
            resumed_supervisor,
            approved_plan=plan,
            progress=resumed_progress,
        ).run()

        assert resumed.borg.state is BorgState.READY_TO_EXECUTE
        assert resumed.generation.id == generation.id
        assert resumed.generation.status is TaskGenerationStatus.CURRENT
        assert resumed.attempt == attempts[0]
        assert resumed_supervisor.calls == []
        assert publication_progress == [
            (StageState.RUNNING, "publishing approved tasks")
        ]
        assert resumed_progress.stages["project-manager"].state is (
            StageState.COMPLETED
        )
        assert resumed_progress.stages["project-manager"].retained is True
        assert resumed_progress.stages["supervisor"].state is StageState.COMPLETED
        assert resumed_progress.stages["supervisor"].retained is False
        resumed_progress.close()


@pytest.mark.parametrize(
    "interrupt_point",
    [
        "agent",
        "structured_validation",
        "after_turn",
        "findings",
        "before_completion",
    ],
)
def test_supervisor_interrupt_before_attempt_completion_cancels_attempt(
    committed_git_repo: Path,
    interrupt_point: str,
    monkeypatch: pytest.MonkeyPatch,
    persist_planning_context,
) -> None:
    plan = _plan()
    database = committed_git_repo.parent / "supervisor-turn-interrupt.sqlite3"

    def interrupt_turn(*_args) -> None:
        raise KeyboardInterrupt

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "supervisor-turn-interrupt"
        )
        _approval, borg = _approve_plan(store, borg, plan)
        initial = ProjectManagerLoop(
            repository,
            borg,
            store,
            MockAdapter(name="openai").queue(
                MockResponse(payload=_pm_payload(plan))
            ),
            approved_plan=plan,
        ).run()
        response = (
            MockResponse(dynamic=interrupt_turn)
            if interrupt_point == "agent"
            else MockResponse(dynamic=_review_response("approve"))
        )
        supervisor = MockAdapter(name="openai").queue(response)
        progress = RunProgress(stream=StringIO())
        loop = SupervisorLoop(
            repository,
            initial.borg,
            store,
            supervisor,
            approved_plan=plan,
            progress=progress,
        )

        if interrupt_point == "structured_validation":
            monkeypatch.setattr(
                "betterborg_cli.planning.turns.validate_structured_result",
                interrupt_turn,
            )
        elif interrupt_point == "after_turn":
            original_run = loop._turns.run

            def interrupt_after_turn(**kwargs):
                original_run(**kwargs)
                raise KeyboardInterrupt

            monkeypatch.setattr(loop._turns, "run", interrupt_after_turn)
        elif interrupt_point == "findings":
            monkeypatch.setattr(loop, "_findings", interrupt_turn)
        elif interrupt_point == "before_completion":
            original_complete = store.complete_planning_attempt

            def interrupt_completion(attempt_id, **kwargs):
                if kwargs["status"] is PlanningAttemptStatus.COMPLETED:
                    raise KeyboardInterrupt
                return original_complete(attempt_id, **kwargs)

            monkeypatch.setattr(
                store, "complete_planning_attempt", interrupt_completion
            )

        with pytest.raises(KeyboardInterrupt):
            loop.run()

        attempts = [
            attempt
            for attempt in store.list_planning_attempts(borg.id)
            if attempt.phase == "supervisor_review"
        ]
        assert len(attempts) == 1
        assert attempts[0].status is PlanningAttemptStatus.CANCELLED
        assert attempts[0].summary == "Supervisor run cancelled"
        assert len(supervisor.calls) == 1
        assert progress.stages["supervisor"].state is StageState.STOPPED
        progress.close()


def test_supervisor_cancellation_between_current_and_ready_completes_progress(
    committed_git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    persist_planning_context,
) -> None:
    plan = _plan()
    database = committed_git_repo.parent / "supervisor-current-ready-cancel.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "supervisor-current-ready-cancel"
        )
        _approval, borg = _approve_plan(store, borg, plan)
        initial = ProjectManagerLoop(
            repository,
            borg,
            store,
            MockAdapter(name="openai").queue(
                MockResponse(payload=_pm_payload(plan))
            ),
            approved_plan=plan,
        ).run()
        cancel = CancellationToken()
        progress = RunProgress(stream=StringIO())
        loop = SupervisorLoop(
            repository,
            initial.borg,
            store,
            MockAdapter(name="openai").queue(
                MockResponse(dynamic=_review_response("approve"))
            ),
            approved_plan=plan,
            cancel=cancel,
            progress=progress,
        )
        original_transition = loop._turns.transition
        interrupted = False

        def interrupt_before_ready(borg: Borg, state: BorgState) -> Borg:
            nonlocal interrupted
            if state is BorgState.READY_TO_EXECUTE and not interrupted:
                interrupted = True
                cancel.cancel()
                raise KeyboardInterrupt
            return original_transition(borg, state)

        monkeypatch.setattr(loop._turns, "transition", interrupt_before_ready)

        with pytest.raises(KeyboardInterrupt):
            loop.run()

        persisted = store.get_borg(borg.id)
        assert persisted is not None
        assert persisted.state is BorgState.READY_TO_EXECUTE
        current = store.get_current_task_generation(borg.id)
        assert current is not None
        assert current.status is TaskGenerationStatus.CURRENT
        attempts = [
            attempt
            for attempt in store.list_planning_attempts(borg.id)
            if attempt.phase == "supervisor_review"
        ]
        assert len(attempts) == 1
        assert attempts[0].status is PlanningAttemptStatus.COMPLETED
        assert progress.stages["supervisor"].state is StageState.COMPLETED
        progress.close()


def test_supervisor_post_commit_cancellation_completes_durable_publication(
    committed_git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    persist_planning_context,
) -> None:
    plan = _plan()
    database = committed_git_repo.parent / "supervisor-post-commit-cancel.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "supervisor-post-commit-cancel"
        )
        _approval, borg = _approve_plan(store, borg, plan)
        initial = ProjectManagerLoop(
            repository,
            borg,
            store,
            MockAdapter(name="openai").queue(
                MockResponse(payload=_pm_payload(plan))
            ),
            approved_plan=plan,
        ).run()
        cancel = CancellationToken()
        original_checkpoint = TaskPublisher._checkpoint

        def cancel_after_commit(self, point: str) -> None:
            original_checkpoint(self, point)
            if point == "after_db_commit":
                cancel.cancel()

        with monkeypatch.context() as interruption:
            interruption.setattr(
                TaskPublisher, "_checkpoint", cancel_after_commit
            )
            result = SupervisorLoop(
                repository,
                initial.borg,
                store,
                MockAdapter(name="openai").queue(
                    MockResponse(dynamic=_review_response("approve"))
                ),
                approved_plan=plan,
                cancel=cancel,
            ).run()

        assert cancel.is_set()
        assert result.borg.state is BorgState.READY_TO_EXECUTE
        assert result.generation.status is TaskGenerationStatus.CURRENT
        assert result.attempt.status is PlanningAttemptStatus.COMPLETED


def test_supervisor_persists_findings_and_runs_bounded_pm_revision(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    plan = _plan()
    initial_payload = _pm_payload(plan)
    revised_payload = _pm_payload(plan)
    revised_payload["tasks"][0]["title"] = "Build a narrow foundation"
    database = committed_git_repo.parent / "supervisor-revise.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "supervisor-revise"
        )
        _approval, borg = _approve_plan(store, borg, plan)
        initial = ProjectManagerLoop(
            repository,
            borg,
            store,
            MockAdapter(name="openai").queue(
                MockResponse(payload=initial_payload)
            ),
            approved_plan=plan,
        ).run()

        def revise_from_findings(spec):
            context = _planning_context(spec)["_betterborg_task_revision"]
            assert context["batch_id"] == str(initial.batch.id)
            assert context["findings"][0]["severity"] == "major"
            assert "narrower scope" in context["findings"][0]["message"]
            task_ref = context["findings"][0]["raised_against_task_ref"]
            referenced_task = next(
                task for task in context["tasks"] if task["task_ref"] == task_ref
            )
            assert referenced_task["task"]["title"] == initial.tasks[0].title
            assert f"[raised against {task_ref}]" in spec.user_prompt
            assert "Supervisor findings" in spec.user_prompt
            return revised_payload

        pm = MockAdapter(name="openai").queue(
            MockResponse(dynamic=revise_from_findings)
        )
        supervisor = MockAdapter(name="openai")
        supervisor.queue(
            MockResponse(dynamic=_review_response("request_changes"))
        )
        supervisor.queue(MockResponse(dynamic=_review_response("approve")))

        result = SupervisorLoop(
            repository,
            initial.borg,
            store,
            supervisor,
            pm_agent=pm,
            approved_plan=plan,
        ).run()

        assert result.borg.state is BorgState.TASKS_APPROVAL_PENDING
        assert result.batch.id != initial.batch.id
        assert result.tasks[0].title == "Build a narrow foundation"
        assert len(store.list_task_batches(borg.id)) == 2
        persisted_findings = store.list_task_findings(
            borg.id, batch_id=initial.batch.id
        )
        assert len(persisted_findings) == 1
        assert persisted_findings[0].message.endswith("narrower scope.")
        assert len(pm.calls) == 1
        assert len(supervisor.calls) == 2
        assert store.get_current_task_generation(borg.id) == result.generation


def test_fresh_progress_finishes_project_manager_before_supervisor_starts(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    plan = _plan()
    adapter = MockAdapter(name="openai")
    adapter.queue(MockResponse(payload=_pm_payload(plan)))
    adapter.queue(MockResponse(dynamic=_review_response("approve")))
    progress = RunProgress(stream=StringIO())
    database = committed_git_repo.parent / "fresh-pm-progress.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "fresh-pm-progress"
        )
        _approval, borg = _approve_plan(store, borg, plan)
        borg = store.compare_and_set_borg_state(
            borg.id,
            expected_state=borg.state,
            expected_version=borg.state_version,
            new_state=BorgState.PM_WORKING,
        )

        result = SupervisorLoop(
            repository,
            borg,
            store,
            adapter,
            pm_agent=adapter,
            approved_plan=plan,
            progress=progress,
        ).run()

        assert result.borg.state is BorgState.READY_TO_EXECUTE
        project_manager = progress.stages["project-manager"]
        supervisor = progress.stages["supervisor"]
        assert project_manager.state is StageState.COMPLETED
        assert project_manager.retained is False
        assert project_manager.started_at is not None
        assert supervisor.state is StageState.COMPLETED
        assert supervisor.started_at is not None
        assert project_manager.finished_at <= supervisor.started_at
        assert supervisor.children == {}
    progress.close()


def test_two_pm_revision_children_reconstruct_from_rejected_attempt_ids(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    plan = _plan()
    payloads = [_pm_payload(plan) for _ in range(3)]
    payloads[1]["tasks"][0]["title"] = "Foundation revision one"
    payloads[2]["tasks"][0]["title"] = "Foundation revision two"
    pm = MockAdapter(name="openai")
    pm.queue(MockResponse(payload=payloads[0]))
    pm.queue(MockResponse(payload=payloads[1]))
    pm.queue(MockResponse(raise_error=RuntimeError("revision interrupted")))
    supervisor = MockAdapter(name="openai")
    supervisor.queue(MockResponse(dynamic=_review_response("request_changes")))
    supervisor.queue(MockResponse(dynamic=_review_response("request_changes")))
    database = committed_git_repo.parent / "pm-progress-resume.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "pm-progress-resume"
        )
        _approval, borg = _approve_plan(store, borg, plan)
        borg = store.compare_and_set_borg_state(
            borg.id,
            expected_state=borg.state,
            expected_version=borg.state_version,
            new_state=BorgState.PM_WORKING,
        )
        interrupted_progress = RunProgress(stream=StringIO())

        with pytest.raises(SupervisorError, match="revision interrupted"):
            SupervisorLoop(
                repository,
                borg,
                store,
                supervisor,
                pm_agent=pm,
                approved_plan=plan,
                progress=interrupted_progress,
            ).run()

        reviews = [
            attempt
            for attempt in store.list_planning_attempts(borg.id)
            if attempt.phase == "supervisor_review"
        ]
        keys = [f"pm-revision:{attempt.id}" for attempt in reviews]
        assert len(keys) == 2
        assert len(set(keys)) == 2
        children = interrupted_progress.stages["supervisor"].children
        assert children[keys[0]].state is StageState.COMPLETED
        assert children[keys[1]].state is StageState.FAILED
        assert interrupted_progress.stages["supervisor"].state is StageState.FAILED
        interrupted_progress.close()

    with SqliteStore.open(database) as reopened:
        resumed_borg = reopened.get_borg(borg.id)
        assert resumed_borg is not None
        pm.queue(MockResponse(payload=payloads[2]))
        supervisor.queue(MockResponse(dynamic=_review_response("approve")))
        resumed_progress = RunProgress(
            stream=StringIO(), attempt_history_limit=1
        )

        result = SupervisorLoop(
            repository,
            resumed_borg,
            reopened,
            supervisor,
            pm_agent=pm,
            approved_plan=plan,
            progress=resumed_progress,
        ).run()

        assert result.borg.state is BorgState.READY_TO_EXECUTE
        project_manager = resumed_progress.stages["project-manager"]
        assert project_manager.state is StageState.COMPLETED
        assert project_manager.retained is True
        assert project_manager.started_at is None
        children = resumed_progress.stages["supervisor"].children
        assert list(children) == keys
        assert children[keys[0]].state is StageState.COMPLETED
        assert children[keys[0]].retained is True
        assert children[keys[0]].started_at is None
        assert children[keys[1]].state is StageState.COMPLETED
        assert children[keys[1]].retained is False
        assert children[keys[1]].started_at is not None
        assert resumed_progress.stages["supervisor"].state is StageState.COMPLETED
        bounded = resumed_progress.child_render_state("supervisor")
        assert [item.key for item in bounded.children] == [keys[1]]
        assert bounded.earlier_attempt_count == 1
        assert len(pm.calls) == 4
        assert len(supervisor.calls) == 3
        resumed_progress.close()


def test_supervisor_rejects_nonprogressing_pm_revisions(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    plan = _plan()
    payload = _pm_payload(plan)
    database = committed_git_repo.parent / "supervisor-no-progress.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "supervisor-no-progress"
        )
        _approval, borg = _approve_plan(store, borg, plan)
        initial = ProjectManagerLoop(
            repository,
            borg,
            store,
            MockAdapter(name="openai").queue(MockResponse(payload=payload)),
            approved_plan=plan,
        ).run()
        pm = MockAdapter(name="openai")
        for _ in range(3):
            pm.queue(MockResponse(payload=payload))
        supervisor = MockAdapter(name="openai").queue(
            MockResponse(dynamic=_review_response("request_changes"))
        )

        with pytest.raises(
            SupervisorError, match="exhausted revision retries.*no semantic progress"
        ):
            SupervisorLoop(
                repository,
                initial.borg,
                store,
                supervisor,
                pm_agent=pm,
                approved_plan=plan,
            ).run()

        assert store.get_borg(borg.id).state is BorgState.PM_WORKING
        assert len(store.list_task_batches(borg.id)) == 1
        revision_attempts = [
            attempt
            for attempt in store.list_planning_attempts(borg.id)
            if attempt.phase == "pm_tasks"
            and attempt.request.get("base_batch_id") == str(initial.batch.id)
        ]
        assert len(revision_attempts) == 3
        assert all(
            attempt.status is PlanningAttemptStatus.FAILED
            and "no semantic progress" in (attempt.summary or "")
            for attempt in revision_attempts
        )
        with pytest.raises(SupervisorError, match="exhausted revision retries"):
            SupervisorLoop(
                repository,
                store.get_borg(borg.id),
                store,
                supervisor,
                pm_agent=pm,
                approved_plan=plan,
            ).run()
        assert len(pm.calls) == 3


def test_supervisor_rejects_order_only_pm_revisions(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    plan = _plan()
    payload = _pm_payload(plan)
    reordered = _pm_payload(plan)
    reordered["tasks"].reverse()
    for task in reordered["tasks"]:
        task["plan_refs"].reverse()
        task["scope"].reverse()
        task["dependencies"].reverse()
    database = committed_git_repo.parent / "supervisor-order-no-progress.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "supervisor-order-no-progress"
        )
        _approval, borg = _approve_plan(store, borg, plan)
        initial = ProjectManagerLoop(
            repository,
            borg,
            store,
            MockAdapter(name="openai").queue(MockResponse(payload=payload)),
            approved_plan=plan,
        ).run()
        pm = MockAdapter(name="openai")
        for _ in range(3):
            pm.queue(MockResponse(payload=reordered))
        supervisor = MockAdapter(name="openai").queue(
            MockResponse(dynamic=_review_response("request_changes"))
        )

        with pytest.raises(
            SupervisorError, match="exhausted revision retries.*no semantic progress"
        ):
            SupervisorLoop(
                repository,
                initial.borg,
                store,
                supervisor,
                pm_agent=pm,
                approved_plan=plan,
            ).run()

        assert len(store.list_task_batches(borg.id)) == 1
        assert len(pm.calls) == 3


def test_supervisor_resumes_completed_provider_turn_without_replay(
    committed_git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    persist_planning_context,
) -> None:
    plan = _plan()
    database = committed_git_repo.parent / "supervisor-resume.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "supervisor-resume"
        )
        _approval, borg = _approve_plan(store, borg, plan)
        initial = ProjectManagerLoop(
            repository,
            borg,
            store,
            MockAdapter(name="openai").queue(
                MockResponse(payload=_pm_payload(plan))
            ),
            approved_plan=plan,
        ).run()
        adapter = MockAdapter(name="openai").queue(
            MockResponse(dynamic=_review_response("approve"))
        )
        loop = SupervisorLoop(
            repository,
            initial.borg,
            store,
            adapter,
            approved_plan=plan,
        )
        original_complete = store.complete_planning_attempt
        interrupted = False

        def interrupt_after_result(attempt_id, **kwargs):
            nonlocal interrupted
            attempt = next(
                item
                for item in store.list_planning_attempts(borg.id)
                if item.id == attempt_id
            )
            if (
                attempt.phase == "supervisor_review"
                and kwargs["status"] is PlanningAttemptStatus.COMPLETED
                and not interrupted
            ):
                interrupted = True
                raise RuntimeError("simulated Supervisor interruption")
            return original_complete(attempt_id, **kwargs)

        with monkeypatch.context() as interruption:
            interruption.setattr(
                store, "complete_planning_attempt", interrupt_after_result
            )
            with pytest.raises(RuntimeError, match="Supervisor interruption"):
                loop.run()

        running = store.list_planning_attempts(borg.id)[-1]
        assert running.status is PlanningAttemptStatus.RUNNING
        assert Path(running.request["result_path"]).is_file()
        assert len(adapter.calls) == 1

        resumed = loop.run()

        assert resumed.borg.state is BorgState.TASKS_APPROVAL_PENDING
        assert len(adapter.calls) == 1


def test_supervisor_cancellation_preserves_resumable_batch(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    plan = _plan()
    database = committed_git_repo.parent / "supervisor-cancel.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "supervisor-cancel"
        )
        _approval, borg = _approve_plan(store, borg, plan)
        initial = ProjectManagerLoop(
            repository,
            borg,
            store,
            MockAdapter(name="openai").queue(
                MockResponse(payload=_pm_payload(plan))
            ),
            approved_plan=plan,
        ).run()
        cancel = CancellationToken()
        cancel.cancel()
        supervisor = MockAdapter(name="openai")

        with pytest.raises(SupervisorCancelled, match="cancelled"):
            SupervisorLoop(
                repository,
                initial.borg,
                store,
                supervisor,
                approved_plan=plan,
                cancel=cancel,
            ).run()

        assert store.get_borg(borg.id).state is BorgState.SUPERVISOR_WORKING
        assert store.list_task_generations(borg.id)[0].status is (
            TaskGenerationStatus.PREPARING
        )
        assert supervisor.calls == []


def test_supervisor_blocks_after_bounded_review_exhaustion(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    plan = _plan()
    database = committed_git_repo.parent / "supervisor-cap.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "supervisor-cap"
        )
        _approval, borg = _approve_plan(store, borg, plan)
        initial = ProjectManagerLoop(
            repository,
            borg,
            store,
            MockAdapter(name="openai").queue(
                MockResponse(payload=_pm_payload(plan))
            ),
            approved_plan=plan,
        ).run()
        pm = MockAdapter(name="openai")
        for revision in (1, 2):
            payload = _pm_payload(plan)
            payload["tasks"][0]["title"] = f"Foundation revision {revision}"
            pm.queue(MockResponse(payload=payload))
        supervisor = MockAdapter(name="openai")
        for round_number in range(1, 4):
            supervisor.queue(
                MockResponse(
                    dynamic=_review_response(
                        "request_changes",
                        f"Round {round_number} still has a scope defect.",
                    )
                )
            )

        result = SupervisorLoop(
            repository,
            initial.borg,
            store,
            supervisor,
            pm_agent=pm,
            approved_plan=plan,
            grant_budget=0,
        ).run()

        assert result.borg.state is BorgState.BLOCKED
        assert len(supervisor.calls) == 3
        assert len(pm.calls) == 2
        assert len(store.list_task_batches(borg.id)) == 3
        assert len(store.list_task_findings(borg.id)) == 3
        assert store.get_current_task_generation(borg.id) is None
        assert all(
            generation.status is TaskGenerationStatus.PREPARING
            for generation in store.list_task_generations(borg.id)
        )


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
        (
            "task.dependency.cycle",
            ("position:1", "position:2", "position:3"),
            (),
            (),
        )
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


def test_fixing_duplicate_task_ref_preserves_surviving_finding_identities() -> None:
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
    second = replace(
        _task(
            generation_id,
            stage="01-foundation",
            stem="02-second",
            position=2,
            refs=foundation_refs[:1],
        ),
        task_ref=first.task_ref,
    )
    consumer = _task(
        generation_id,
        stage="02-consumer",
        stem="01-build",
        position=3,
        refs=_required_refs(plan, "02-consumer"),
    )

    previous = task_graph_findings(plan, [first, second, consumer], [])
    repaired = task_graph_findings(
        plan,
        [first, replace(second, task_ref="task-2"), consumer],
        [],
    )
    previous_duplicate_owner = next(
        finding
        for finding in previous
        if finding.rule == "task.traceability.duplicate_owner"
    )
    repaired_duplicate_owner = next(
        finding
        for finding in repaired
        if finding.rule == "task.traceability.duplicate_owner"
    )

    assert previous_duplicate_owner.identity == repaired_duplicate_owner.identity
    assert sum(finding.rule == "task.ref.duplicate" for finding in previous) == 1
    assert all(finding.rule != "task.ref.duplicate" for finding in repaired)
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


def test_a_lowered_decomposition_budget_blocks_on_its_only_round(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    """Three rounds is a default, not the only answer a project may give."""
    plan = _plan()
    database = committed_git_repo.parent / "supervisor-lowered.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "supervisor-lowered"
        )
        _approval, borg = _approve_plan(store, borg, plan)
        pm_result = ProjectManagerLoop(
            repository,
            borg,
            store,
            MockAdapter(name="openai").queue(
                MockResponse(payload=_pm_payload(plan))
            ),
            approved_plan=plan,
        ).run()
        supervisor = MockAdapter(name="openai").queue(
            MockResponse(dynamic=_review_response("request_changes"))
        )

        result = SupervisorLoop(
            repository,
            pm_result.borg,
            store,
            supervisor,
            approved_plan=plan,
            review_rounds=1,
            grant_budget=0,
        ).run()

        assert result.borg.state is BorgState.BLOCKED
        assert len(supervisor.calls) == 1
        assert "in round 1." in supervisor.calls[0].user_prompt
        assert store.list_task_findings(borg.id)


def test_a_blocked_decomposition_reports_its_record_under_any_budget(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    """Whether a rejection revised or blocked was settled when it completed.

    Counting the record against a number raised since would deny the plainly
    terminal record and answer with an error naming a state.
    """
    plan = _plan()
    database = committed_git_repo.parent / "supervisor-blocked-raised.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "supervisor-blocked-raised"
        )
        _approval, borg = _approve_plan(store, borg, plan)
        pm_result = ProjectManagerLoop(
            repository,
            borg,
            store,
            MockAdapter(name="openai").queue(
                MockResponse(payload=_pm_payload(plan))
            ),
            approved_plan=plan,
        ).run()
        supervisor = MockAdapter(name="openai").queue(
            MockResponse(dynamic=_review_response("request_changes"))
        )
        first = SupervisorLoop(
            repository,
            pm_result.borg,
            store,
            supervisor,
            approved_plan=plan,
            review_rounds=1,
            grant_budget=0,
        ).run()
        assert first.borg.state is BorgState.BLOCKED

        blocked = store.get_borg(borg.id)
        assert blocked is not None
        again = SupervisorLoop(
            repository,
            blocked,
            store,
            supervisor,
            approved_plan=plan,
            review_rounds=5,
            progress=RunProgress(stream=StringIO()),
        ).run()

        assert again.borg.state is BorgState.BLOCKED
        assert len(supervisor.calls) == 1


def test_a_decomposition_budget_below_one_is_refused_at_construction(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    plan = _plan()
    database = committed_git_repo.parent / "supervisor-zero-budget.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "supervisor-zero-budget"
        )
        _approval, borg = _approve_plan(store, borg, plan)
        with pytest.raises(SupervisorError, match="at least 1"):
            SupervisorLoop(
                repository,
                borg,
                store,
                MockAdapter(name="openai"),
                approved_plan=plan,
                review_rounds=0,
            )


def test_lowering_the_decomposition_budget_does_not_strand_a_revision_under_way(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    """The budget bounds what happens next, never what already happened.

    A run interrupted mid-revision is resumable, and the CLI says so. Refusing
    the round that revision leads to would make the advertised resume
    impossible, with no way back but restoring a number nothing names.
    """
    plan = _plan()
    database = committed_git_repo.parent / "supervisor-strand.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "supervisor-strand"
        )
        _approval, borg = _approve_plan(store, borg, plan)
        pm = MockAdapter(name="openai").queue(
            MockResponse(payload=_pm_payload(plan))
        )
        pm_result = ProjectManagerLoop(
            repository, borg, store, pm, approved_plan=plan
        ).run()

        # Two rejections spent, and the run dies before the third revision.
        supervisor = MockAdapter(name="openai")
        for _ in range(2):
            supervisor.queue(MockResponse(dynamic=_review_response("request_changes")))
        pm.queue(MockResponse(payload=_pm_payload(plan, revision=" One.")))
        with pytest.raises((SupervisorError, RuntimeError)):
            SupervisorLoop(
                repository,
                pm_result.borg,
                store,
                supervisor,
                pm_agent=pm,
                approved_plan=plan,
                review_rounds=3,
            ).run()
        interrupted = store.get_borg(borg.id)
        assert interrupted is not None
        assert interrupted.state is BorgState.PM_WORKING

        # The operator lowers the budget below the rounds already spent, then
        # resumes as the run told them to.
        pm.queue(MockResponse(payload=_pm_payload(plan, revision=" Two.")))
        supervisor.queue(MockResponse(dynamic=_review_response("request_changes")))
        resumed = SupervisorLoop(
            repository,
            interrupted,
            store,
            supervisor,
            pm_agent=pm,
            approved_plan=plan,
            review_rounds=2,
            grant_budget=0,
        ).run()

        assert resumed.borg.state is BorgState.BLOCKED
        assert "in round 3." in supervisor.calls[-1].user_prompt
        assert "of 2." not in supervisor.calls[-1].user_prompt


def test_a_raised_budget_reconstructs_every_revision_it_paid_for(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    """History is history, however the budget has moved since.

    Read back through a bound the record outgrew, the revisions past that
    bound disappear from the account of the run, and the reader is shown a
    decomposition that took fewer attempts than it did.
    """
    plan = _plan()
    database = committed_git_repo.parent / "supervisor-raised.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "supervisor-raised"
        )
        _approval, borg = _approve_plan(store, borg, plan)
        pm = MockAdapter(name="openai").queue(
            MockResponse(payload=_pm_payload(plan))
        )
        pm_result = ProjectManagerLoop(
            repository, borg, store, pm, approved_plan=plan
        ).run()

        # Three rejections, each answered by a revision, then an approval:
        # more rounds than the default budget allows.
        supervisor = MockAdapter(name="openai")
        for index in range(3):
            supervisor.queue(
                MockResponse(dynamic=_review_response("request_changes"))
            )
            pm.queue(
                MockResponse(payload=_pm_payload(plan, revision=f" R{index}."))
            )
        supervisor.queue(MockResponse(dynamic=_review_response("approve")))

        first = SupervisorLoop(
            repository,
            pm_result.borg,
            store,
            supervisor,
            pm_agent=pm,
            approved_plan=plan,
            review_rounds=5,
        ).run()
        assert first.borg.state is BorgState.TASKS_APPROVAL_PENDING
        assert len(supervisor.calls) == 4

        finished = store.get_borg(borg.id)
        assert finished is not None
        progress = RunProgress(stream=StringIO())
        again = SupervisorLoop(
            repository,
            finished,
            store,
            supervisor,
            pm_agent=pm,
            approved_plan=plan,
            review_rounds=5,
            progress=progress,
        ).run()

        assert again == first
        assert len(supervisor.calls) == 4
        children = progress.stages["supervisor"].children
        assert len(children) == 3
        assert all(
            child.state is StageState.COMPLETED for child in children.values()
        )


@pytest.mark.parametrize("budget", [0, -1, 1.5])
def test_a_decomposition_budget_that_is_not_a_whole_number_above_zero_is_refused(
    committed_git_repo: Path,
    persist_planning_context,
    budget: object,
) -> None:
    """The loop is handed this directly as well as through configuration."""
    plan = _plan()
    database = committed_git_repo.parent / f"supervisor-budget-{budget}.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, f"supervisor-budget-{str(budget).strip('-.')}"
        )
        _approval, borg = _approve_plan(store, borg, plan)
        with pytest.raises(SupervisorError, match="whole number"):
            SupervisorLoop(
                repository,
                borg,
                store,
                MockAdapter(name="openai"),
                approved_plan=plan,
                review_rounds=budget,
            )


def test_no_stem_clears_the_schema_and_fails_the_graph_check_after_it() -> None:
    """The Project Manager's schema and the graph check agree on every stem.

    Both read the same constant, but the schema searches while the check
    matches in full, so an anchor generous about a trailing newline divides
    them on that one input and costs a decomposition round to discover.
    """

    from betterborg_cli.agent_runtime.structured import (
        StructuredResultError,
        validate_structured_result,
    )
    from betterborg_cli.planning.pm import PROJECT_MANAGER_TASKS_SCHEMA
    from betterborg_cli.planning.task_validation import _TASK_NAME

    schema = {
        "type": "object",
        "required": ["stem"],
        "properties": {
            "stem": PROJECT_MANAGER_TASKS_SCHEMA["properties"]["tasks"]["items"][
                "properties"
            ]["stem"]
        },
    }

    for stem in ("01-setup", "01-abc\n", "01-abc\r\n", "01--bad", "01-bad-"):
        try:
            validate_structured_result({"stem": stem}, schema)
        except StructuredResultError:
            continue
        assert _TASK_NAME.fullmatch(stem) is not None, (
            f"{stem!r} cleared the schema and is refused by the graph check"
        )


def test_every_role_that_writes_a_phase_name_is_told_its_shape() -> None:
    """The Tech Lead writes phase names as surely as the Architect does.

    Its findings are not only a verdict: the revision turn is told to address
    every persisted finding, so a suggestion that renames a phase becomes the
    name. A Tech Lead that has not been told the shape can ask for one the
    Architect is then refused for using, and a finding that cannot be
    satisfied blocks the run once the rounds run out.
    """
    from betterborg_cli.planning.architect import _PLAN_SYSTEM_PROMPT
    from betterborg_cli.planning.tech_lead import _TECH_LEAD_SYSTEM_PROMPT

    shape = (
        "A phase name is two digits then lowercase words of letters and "
        "digits, all joined by single hyphens and at most 32 characters"
    )
    for prompt in (_PLAN_SYSTEM_PROMPT, _TECH_LEAD_SYSTEM_PROMPT):
        assert shape in " ".join(prompt.split())


def test_every_planning_role_is_told_what_betterborg_performs() -> None:
    """A role that does not know the harness holds the plan to the harness.

    Betterborg creates the branch and worktree, commits, reviews, merges and
    runs the checks. Told to none of them, a plan grows a delivery-preflight
    phase and the batch grows a task whose whole scope is verifying a baseline
    and creating a branch, which has nothing to commit. Told only to the two
    that write, the two that judge reject the plan for the omission, which is
    the same run lost at the other end.
    """
    from betterborg_cli.planning.architect import _PLAN_SYSTEM_PROMPT
    from betterborg_cli.planning.pm import _PROJECT_MANAGER_SYSTEM_PROMPT
    from betterborg_cli.planning.supervisor import _SUPERVISOR_SYSTEM_PROMPT
    from betterborg_cli.planning.tech_lead import _TECH_LEAD_SYSTEM_PROMPT

    boundary = "branching, worktrees, commits, review, merge"
    for prompt in (
        _PLAN_SYSTEM_PROMPT,
        _PROJECT_MANAGER_SYSTEM_PROMPT,
        _TECH_LEAD_SYSTEM_PROMPT,
        _SUPERVISOR_SYSTEM_PROMPT,
    ):
        flattened = " ".join(prompt.split())
        assert "Betterborg performs the delivery" in flattened
        assert boundary in flattened

    plan_prompt = " ".join(_PLAN_SYSTEM_PROMPT.split())
    assert "Plan the product change only" in plan_prompt

    pm_prompt = " ".join(_PROJECT_MANAGER_SYSTEM_PROMPT.split())
    assert "Never write a task for any of that" in pm_prompt
    assert "one with nothing to commit is not a task" in pm_prompt

    # The two that judge are told not to require what the two that write were
    # told to leave out.
    for prompt in (_TECH_LEAD_SYSTEM_PROMPT, _SUPERVISOR_SYSTEM_PROMPT):
        assert "never hold" in " ".join(prompt.split())


def test_every_role_the_ledger_reaches_is_told_what_it_owes() -> None:
    """A required field with no instruction behind it comes back empty.

    The schema can make a reviewer send `resolved` and `repeats`; only the
    instruction makes it send anything in them, and a ledger never told what
    closed never drains. The fixer needs the other half: that an objection
    stays open until it is answered, and that the reference a carried one
    carries is a label rather than a task to look up.
    """
    from betterborg_cli.planning.pm import _PROJECT_MANAGER_SYSTEM_PROMPT
    from betterborg_cli.planning.supervisor import _SUPERVISOR_SYSTEM_PROMPT
    from betterborg_cli.planning.tech_lead import _TECH_LEAD_SYSTEM_PROMPT

    for prompt in (_TECH_LEAD_SYSTEM_PROMPT, _SUPERVISOR_SYSTEM_PROMPT):
        flattened = " ".join(prompt.split())
        assert "list in resolved the id of every one" in flattened
        assert (
            "set repeats to the id of the open finding it raises again, or "
            "null when the objection is new" in flattened
        )
        # Unconditional, because making the requirement depend on the decision
        # would cost provider enforcement of every field at once.
        assert "Both are always required" in flattened
        assert (
            "An open finding you neither resolve nor repeat stays open"
            in flattened
        )

    fixer = " ".join(_PROJECT_MANAGER_SYSTEM_PROMPT.split())
    assert "every Supervisor objection its batch still owes an answer for" in fixer
    assert (
        "may name a task no current batch holds, so answer the objection "
        "rather than looking its reference up" in fixer
    )


def _ledger_review(
    decision: str,
    *,
    handed: list[list[dict]],
    message: str = "",
    severity: str = "major",
    closes_open: bool = False,
    repeats_open: bool = False,
):
    """Review the batch in hand, recording the open ledger it was given.

    Both declarations are answered against the ledger the round is handed,
    because the ids only exist once an earlier round has raised them.
    """

    def respond(spec):
        context = _planning_context(spec)
        ledger = context["open_supervisor_findings"]
        handed.append(ledger)
        findings = []
        if decision == "request_changes":
            findings.append(
                {
                    "severity": severity,
                    "message": message,
                    "suggestion": "Keep the task independently testable.",
                    "task_ref": context["task_batch"]["tasks"][0]["task_ref"],
                    "repeats": ledger[0]["id"] if repeats_open else None,
                }
            )
        return {
            "decision": decision,
            "summary": f"Supervisor decided to {decision}.",
            "findings": findings,
            "resolved": [row["id"] for row in ledger] if closes_open else [],
        }

    return respond


def _recording_pm(payloads: Iterable[dict], handed: list[tuple[dict, str]]):
    """Revise as scripted, recording the revision context and prompt each time."""
    adapter = MockAdapter(name="openai")
    for payload in payloads:

        def respond(spec, payload=payload):
            revision = _planning_context(spec)["_betterborg_task_revision"]
            handed.append((revision, spec.user_prompt))
            return payload

        adapter.queue(MockResponse(dynamic=respond))
    return adapter


def test_a_supervisor_objection_outlives_the_batch_its_reference_named(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    """A carried reference is a label on where the objection started.

    Every revision mints fresh task ids and derives the references from them,
    so the round that inherits an objection reviews tasks whose references are
    all new. The fixer still owes an answer for the objection, so both places
    the findings reach the Project Manager read the ledger rather than the one
    batch's snapshot.
    """
    plan = _plan()
    reviews: list[list[dict]] = []
    revisions: list[tuple[dict, str]] = []
    database = committed_git_repo.parent / "supervisor-ledger.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "supervisor-ledger"
        )
        approval, borg = _approve_plan(store, borg, plan)
        initial = ProjectManagerLoop(
            repository,
            borg,
            store,
            MockAdapter(name="openai").queue(
                MockResponse(payload=_pm_payload(plan))
            ),
            approved_plan=plan,
        ).run()

        supervisor = MockAdapter(name="openai")
        supervisor.queue(
            MockResponse(
                dynamic=_ledger_review(
                    "request_changes",
                    message="The first task needs a narrower scope.",
                    handed=reviews,
                )
            )
        )
        supervisor.queue(
            MockResponse(
                dynamic=_ledger_review(
                    "request_changes",
                    message="The second task duplicates the first.",
                    handed=reviews,
                )
            )
        )
        supervisor.queue(
            MockResponse(
                dynamic=_ledger_review("approve", handed=reviews)
            )
        )
        pm = _recording_pm(
            (
                _pm_payload(plan, revision=" v2"),
                _pm_payload(plan, revision=" v3"),
            ),
            revisions,
        )

        result = SupervisorLoop(
            repository,
            initial.borg,
            store,
            supervisor,
            pm_agent=pm,
            approved_plan=plan,
        ).run()

        assert result.borg.state is BorgState.READY_TO_EXECUTE
        batches = store.list_task_batches(borg.id)
        assert [batch.id for batch in batches][0] == initial.batch.id
        assert len(batches) == 3

        # The second round was handed the first round's objection, labelled
        # with the batch its reference belonged to and not with a reference
        # into the batch under review.
        assert len(reviews[1]) == 1
        carried = reviews[1][0]
        assert carried["message"] == "The first task needs a narrower scope."
        assert carried["first_raised_in_round"] == 1
        assert carried["raised_against_batch_id"] == str(initial.batch.id)
        assert carried["raised_against_task_ref"] == initial.tasks[0].task_ref
        # Against the tasks of the batch under review, which are reached
        # through its generation: a batch id finds no task records at all, and
        # an empty set would make the comparison below say nothing.
        reviewed = next(
            item
            for item in store.list_task_generations(borg.id)
            if item.batch_id == batches[1].id
        )
        live_refs = {
            task.task_ref for task in store.list_task_records(reviewed.id)
        }
        assert live_refs
        assert carried["raised_against_task_ref"] not in live_refs
        # The first round had nothing to answer for, so the label it went on to
        # raise could only have come from the batch it was reviewing.
        assert reviews[0] == []

        # The second revision is shown both objections, while the snapshot of
        # the batch it is revising holds only the newer one.
        revision, prompt = revisions[1]
        assert [item["message"] for item in revision["findings"]] == [
            "The first task needs a narrower scope.",
            "The second task duplicates the first.",
        ]
        assert "The first task needs a narrower scope." in prompt
        assert [
            finding.message
            for finding in store.list_task_findings(
                borg.id, batch_id=batches[1].id
            )
        ] == ["The second task duplicates the first."]
        assert f"[raised against {carried['raised_against_task_ref']}]" in prompt

        # One ledger spans every batch of the approval, and the approval closes it.
        rows = store.list_task_ledger_findings(
            borg.id, plan_approval_id=approval.id
        )
        assert [row.message for row in rows] == [
            "The first task needs a narrower scope.",
            "The second task duplicates the first.",
        ]
        assert [row.batch_id for row in rows] == [initial.batch.id, batches[1].id]
        assert all(row.status is FindingStatus.RESOLVED for row in rows)
        assert all(row.last_seen_round == 3 for row in rows)
        assert open_task_findings(store, borg.id, approval.id) == []


def _seed_other_approval_objection(store: SqliteStore, borg) -> TaskLedgerFinding:
    """Leave one open objection under a plan approval this run is not running.

    Every table in this plan carries its scope as a column, so a row belonging
    to another approval is neither handed to a reviewer nor moved by one.
    """
    other = PlanApproval(
        borg_id=borg.id,
        plan_digest="sha256:another-approval",
        manifest={"plan.json": "sha256:another-approval"},
        approved_by="test operator",
    )
    batch = TaskBatch(
        borg_id=borg.id,
        plan_approval_id=other.id,
        round=1,
        digest="sha256:another-batch",
        manifest={"task-refs": ["elsewhere"]},
        summary="A batch of another approval.",
    )
    attempt = PlanningAttempt(
        borg_id=borg.id,
        phase="supervisor_review",
        round=1,
        adapter="mock",
        model="test-model",
    )
    stranded = TaskLedgerFinding(
        borg_id=borg.id,
        plan_approval_id=other.id,
        batch_id=batch.id,
        attempt_id=attempt.id,
        first_seen_round=1,
        last_seen_round=1,
        severity="blocker",
        message="An objection of another approval entirely.",
    )
    with store.transaction():
        store.append_plan_approval(other)
        store.append_task_batch(batch)
        store.append_planning_attempt(attempt)
        store.record_task_ledger_findings([stranded])
    return stranded


def test_a_supervisor_review_closes_and_repeats_the_rows_it_names(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    """The declarations a Supervisor makes are what move its ledger.

    A round that says it closed an objection closes it, and one that raises an
    objection again moves the row already holding it rather than adding a
    second, so the loop can see which of its rounds have already tried.
    """
    plan = _plan()
    reviews: list[list[dict]] = []
    revisions: list[tuple[dict, str]] = []
    database = committed_git_repo.parent / "supervisor-declarations.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "supervisor-declarations"
        )
        # Seeded before the approval this run uses, so the run's own approval
        # stays the latest one.
        elsewhere = _seed_other_approval_objection(store, borg)
        approval, borg = _approve_plan(store, borg, plan)
        initial = ProjectManagerLoop(
            repository,
            borg,
            store,
            MockAdapter(name="openai").queue(
                MockResponse(payload=_pm_payload(plan))
            ),
            approved_plan=plan,
        ).run()

        supervisor = MockAdapter(name="openai")
        supervisor.queue(
            MockResponse(
                dynamic=_ledger_review(
                    "request_changes",
                    message="The first task needs a narrower scope.",
                    handed=reviews,
                )
            )
        )
        supervisor.queue(
            MockResponse(
                dynamic=_ledger_review(
                    "request_changes",
                    message="The second task duplicates the first.",
                    closes_open=True,
                    handed=reviews,
                )
            )
        )
        supervisor.queue(
            MockResponse(
                dynamic=_ledger_review(
                    "request_changes",
                    message="The duplication is still there.",
                    severity="blocker",
                    repeats_open=True,
                    handed=reviews,
                )
            )
        )
        pm = _recording_pm(
            (
                _pm_payload(plan, revision=" v2"),
                _pm_payload(plan, revision=" v3"),
            ),
            revisions,
        )

        result = SupervisorLoop(
            repository,
            initial.borg,
            store,
            supervisor,
            pm_agent=pm,
            approved_plan=plan,
            grant_budget=0,
        ).run()

        assert result.borg.state is BorgState.BLOCKED

        # The round that closed the first objection was handed it; the round
        # after was handed only what its predecessor left standing.
        assert [row["message"] for row in reviews[1]] == [
            "The first task needs a narrower scope."
        ]
        assert [row["message"] for row in reviews[2]] == [
            "The second task duplicates the first."
        ]

        rows = {
            row.message: row
            for row in store.list_task_ledger_findings(
                borg.id, plan_approval_id=approval.id
            )
        }
        assert set(rows) == {
            "The first task needs a narrower scope.",
            "The second task duplicates the first.",
        }
        closed = rows["The first task needs a narrower scope."]
        assert closed.status is FindingStatus.RESOLVED
        assert closed.last_seen_round == 2
        repeated = rows["The second task duplicates the first."]
        assert repeated.status is FindingStatus.REGRESSED
        assert repeated.first_seen_round == 2
        assert repeated.last_seen_round == 3
        # The repeat moved the row rather than adding one, so a blocker
        # restatement could not escalate what was first recorded as a major.
        assert repeated.severity == "major"
        assert [
            row.message for row in open_task_findings(store, borg.id, approval.id)
        ] == ["The second task duplicates the first."]

        # Each row names the review that last established its status, so a
        # reader of the table can tell which round answered for it.
        by_round = {
            attempt.request["review_round"]: attempt.id
            for attempt in store.list_planning_attempts(borg.id)
            if attempt.phase == "supervisor_review"
            and attempt.status is PlanningAttemptStatus.COMPLETED
        }
        assert closed.attempt_id == by_round[2]
        assert repeated.attempt_id == by_round[3]

        # The objection of another approval was never handed to a round and was
        # never moved by one, so its own loop still owes an answer for it.
        assert all(
            elsewhere.message not in {row["message"] for row in ledger}
            for ledger in reviews
        )
        untouched = store.list_task_ledger_findings(
            borg.id, plan_approval_id=elsewhere.plan_approval_id
        )
        assert untouched == [elsewhere]


@pytest.mark.parametrize(
    ("mutate", "missing"),
    [
        pytest.param(
            lambda payload: payload.pop("resolved"), "resolved", id="resolved"
        ),
        pytest.param(
            lambda payload: payload["findings"][0].pop("repeats"),
            "repeats",
            id="repeats",
        ),
    ],
)
def test_a_supervisor_review_omitting_a_ledger_declaration_fails_its_schema(
    mutate, missing: str
) -> None:
    """Requiring both is what forces a reviewer to answer rather than omit."""
    payload = {
        "decision": "request_changes",
        "summary": "The first task needs a narrower scope.",
        "findings": [
            {
                "severity": "major",
                "message": "The first task needs a narrower scope.",
                "repeats": None,
            }
        ],
        "resolved": [],
    }
    validate_structured_result(payload, SUPERVISOR_REVIEW_SCHEMA)

    mutate(payload)
    with pytest.raises(
        StructuredResultError, match=f"missing required property '{missing}'"
    ):
        validate_structured_result(payload, SUPERVISOR_REVIEW_SCHEMA)


def _grant_review(
    decision: str,
    *,
    raised: int = 1,
    close_open: bool = False,
    severity: str = "major",
    label: str = "",
):
    """Review the batch in hand, closing the ledger rows it was handed.

    A round that closes more than it raises is what lowers the open count a
    refund compares on.
    """

    def respond(spec):
        context = _planning_context(spec)
        ledger = context["open_supervisor_findings"]
        task_ref = context["task_batch"]["tasks"][0]["task_ref"]
        return {
            "decision": decision,
            "summary": f"Supervisor decided to {decision}.",
            "findings": [
                {
                    "severity": severity,
                    "message": f"{label}Objection {index}.",
                    "suggestion": "Keep the task independently testable.",
                    "task_ref": task_ref,
                    "repeats": None,
                }
                for index in range(raised)
            ],
            "resolved": [row["id"] for row in ledger] if close_open else [],
        }

    return respond


def _decomposition_assessments(store, borg_id, approval):
    return [
        (row.round, row.open_findings, row.refunded, row.converging)
        for row in store.list_review_assessments(
            borg_id, loop="supervisor_review", plan_approval_id=approval.id
        )
    ]


def test_a_draining_decomposition_runs_past_its_minimum_and_approves(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    """Past the minimum every round is a grant, and a productive one is free.

    The batch the Supervisor is closing findings against reaches approval with
    no operator action, where its counter would have stopped it on the round
    the minimum named.
    """
    plan = _plan()
    database = committed_git_repo.parent / "supervisor-granted.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "supervisor-granted"
        )
        approval, borg = _approve_plan(store, borg, plan)
        pm = MockAdapter(name="openai").queue(
            MockResponse(payload=_pm_payload(plan))
        )
        initial = ProjectManagerLoop(
            repository, borg, store, pm, approved_plan=plan
        ).run()
        for revision in (1, 2):
            payload = _pm_payload(plan)
            payload["tasks"][0]["title"] = f"Foundation revision {revision}"
            pm.queue(MockResponse(payload=payload))
        supervisor = MockAdapter(name="openai")
        supervisor.queue(
            MockResponse(dynamic=_grant_review("request_changes", raised=3))
        )
        supervisor.queue(
            MockResponse(
                dynamic=_grant_review(
                    "request_changes", raised=1, close_open=True, label="Narrower: "
                )
            )
        )
        supervisor.queue(
            MockResponse(
                dynamic=_grant_review("approve", raised=0, close_open=True)
            )
        )

        result = SupervisorLoop(
            repository,
            initial.borg,
            store,
            supervisor,
            pm_agent=pm,
            approved_plan=plan,
            review_rounds=1,
            grant_budget=10,
        ).run()

        assert result.borg.state is BorgState.TASKS_APPROVAL_PENDING
        assert len(supervisor.calls) == 3
        assert _decomposition_assessments(store, borg.id, approval) == [
            (1, 3, None, False),
            (2, 1, True, True),
            (3, 0, True, True),
        ]


@pytest.mark.parametrize("review_rounds", [1, 2])
def test_the_decision_allowance_ends_a_granted_round_the_same_way(
    committed_git_repo: Path,
    persist_planning_context,
    review_rounds: int,
) -> None:
    """The Supervisor gets three tries at its own decision contract, no more.

    The allowance spans the loop's whole life and was sized when a loop could
    not exceed its configured rounds. Nothing here moves it, so a round that
    keeps contradicting itself ends the run needing a person whether the round
    was granted or inside the minimum.
    """
    plan = _plan()
    database = (
        committed_git_repo.parent / f"supervisor-decision-{review_rounds}.sqlite3"
    )
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, f"supervisor-decision-{review_rounds}"
        )
        approval, borg = _approve_plan(store, borg, plan)
        pm = MockAdapter(name="openai").queue(
            MockResponse(payload=_pm_payload(plan))
        )
        initial = ProjectManagerLoop(
            repository, borg, store, pm, approved_plan=plan
        ).run()
        pm.queue(MockResponse(payload=_pm_payload(plan, revision=" One.")))
        supervisor = MockAdapter(name="openai")
        supervisor.queue(MockResponse(dynamic=_grant_review("request_changes")))
        for _ in range(SUPERVISOR_DECISION_ROUND_CAP):
            supervisor.queue(
                MockResponse(
                    dynamic=_grant_review("request_changes", severity="minor")
                )
            )

        with pytest.raises(SupervisorError, match="blocker or major finding"):
            SupervisorLoop(
                repository,
                initial.borg,
                store,
                supervisor,
                pm_agent=pm,
                approved_plan=plan,
                review_rounds=review_rounds,
                grant_budget=5,
            ).run()

        assert len(supervisor.calls) == 1 + SUPERVISOR_DECISION_ROUND_CAP
        # The round that never reached a decision was never assessed, so it
        # earned no refund either.
        assert [
            number
            for number, *_ in _decomposition_assessments(store, borg.id, approval)
        ] == [1]


@pytest.mark.parametrize("budget", [-1, 1.5])
def test_a_grant_budget_that_is_not_a_whole_number_at_or_above_zero_is_refused(
    committed_git_repo: Path,
    persist_planning_context,
    budget: object,
) -> None:
    """Below zero would stop the loop short of the minimum it was told to run."""
    plan = _plan()
    database = committed_git_repo.parent / f"supervisor-grant-{budget}.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, f"supervisor-grant-{str(budget).strip('-.')}"
        )
        _approval, borg = _approve_plan(store, borg, plan)
        with pytest.raises(SupervisorError, match="at least 0"):
            SupervisorLoop(
                repository,
                borg,
                store,
                MockAdapter(name="openai"),
                approved_plan=plan,
                grant_budget=budget,
            )
