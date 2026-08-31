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
from betterborg_cli.planning import (
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
from betterborg_cli.progress import RunProgress, StageState
from betterborg_cli.store import (
    Borg,
    BorgState,
    PlanApproval,
    PlanningAttempt,
    PlanningAttemptStatus,
    SqliteStore,
    TaskComplexity,
    TaskDependency,
    TaskGenerationStatus,
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


def _pm_payload(plan: dict) -> dict:
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
            "title": f"Build {stage}",
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
                }
            )
        return {
            "decision": decision,
            "summary": f"Supervisor decided to {decision}.",
            "findings": findings,
        }

    return respond


def _rules(findings: Iterable[TaskGraphFinding]) -> set[str]:
    return {finding.rule for finding in findings}


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
        assert len(adapter.calls) == 2
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
            task_ref = context["findings"][0]["task_ref"]
            referenced_task = next(
                task for task in context["tasks"] if task["task_ref"] == task_ref
            )
            assert referenced_task["task"]["title"] == initial.tasks[0].title
            assert f"[{task_ref}]" in spec.user_prompt
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
