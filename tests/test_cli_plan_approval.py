"""CLI contracts for digest-bound approval and automatic decomposition."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from betterborg_cli import cli as cli_module
from betterborg_cli import workflow_service as workflow_service_module
from betterborg_cli.agent_runtime import CancellationToken, run_captured
from betterborg_cli.agent_runtime.api_tools import ApiAgentRole
from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.cli import cli
from betterborg_cli.planning import (
    TaskPublisher,
    approved_plan_digest,
    build_plan_element_catalog,
)
from betterborg_cli.planning import task_publication as publication_module
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.repository_files import (
    RepositoryGitVisibilityError,
    require_git_trackable,
)
from betterborg_cli.store import (
    BorgState,
    PlanningAttempt,
    PlanningAttemptStatus,
    SqliteStore,
    TaskGenerationStatus,
)


def _seed_approval_pending(
    root: Path,
    planning_cli_repository,
    name: str,
    plan: dict,
):
    repository, paths = planning_cli_repository(root, name)
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, name)
        assert borg is not None
        attempt = PlanningAttempt(
            borg_id=borg.id,
            phase="architect_plan",
            round=1,
            adapter="mock",
            model="test-model",
        )
        store.append_planning_attempt(attempt)
        store.complete_planning_attempt(
            attempt.id,
            status=PlanningAttemptStatus.COMPLETED,
            result=plan,
            summary="Ready for operator approval.",
        )
        store.compare_and_set_borg_state(
            borg.id,
            expected_state=borg.state,
            expected_version=borg.state_version,
            new_state=BorgState.PLAN_APPROVAL_PENDING,
        )
    return repository, attempt, paths


def _pm_tasks(plan: dict, *, title: str = "Document the release workflow") -> dict:
    required_refs = [
        element.ref for element in build_plan_element_catalog(plan) if element.required
    ]
    return {
        "summary": "One task covers the approved plan.",
        "tasks": [
            {
                "stage": "01-release-workflow",
                "stem": "01-document-release",
                "title": title,
                "why": "The approved workflow needs an executable task.",
                "scope": ["Document the release path."],
                "implementation_notes": [],
                "acceptance_criteria": ["The release path is documented."],
                "tests": ["Assert the documented public workflow."],
                "dependencies": [],
                "out_of_scope": [],
                "plan_refs": required_refs,
                "estimate_complexity": "small",
            }
        ],
    }


def _review(decision: str, message: str = "The task is ready.") -> dict:
    findings = []
    if decision == "request_changes":
        findings.append(
            {
                "severity": "major",
                "message": message,
                "suggestion": "Make the task independently verifiable.",
            }
        )
    return {
        "decision": decision,
        "summary": message,
        "findings": findings,
    }


def test_approved_plan_trackability_uses_run_token(
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    real_process_harness: Any,
    monkeypatch,
) -> None:
    plan = planning_plan_response()
    repository, _attempt, paths = _seed_approval_pending(
        committed_git_repo,
        planning_cli_repository,
        "approval-trackability",
        plan,
    )
    approval_cancel = CancellationToken()
    errors: list[BaseException] = []
    commands: list[tuple[str, ...]] = []

    def blocked_trackability(
        path: Path,
        *,
        root: Path,
        cancel: CancellationToken | None = None,
    ) -> None:
        assert cancel is approval_cancel

        def runner(
            command: list[str], **kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            commands.append(tuple(command))
            assert kwargs["cancel"] is cancel
            return run_captured(
                real_process_harness.resistant_argv("approval-trackability"),
                check=kwargs["check"],
                cancel=kwargs["cancel"],
            )

        require_git_trackable(
            path,
            root=root,
            cancel=cancel,
            command_runner=runner,
        )

    monkeypatch.setattr(
        workflow_service_module,
        "require_git_trackable",
        blocked_trackability,
    )

    def approve() -> None:
        try:
            with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
                borg = store.get_borg_by_name(repository.id, "approval-trackability")
                assert borg is not None
                workflow_service_module.bind_plan_approval(
                    paths,
                    store,
                    borg,
                    cancel=approval_cancel,
                )
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=approve)
    worker.start()
    real_process_harness.wait_for_marker("approval-trackability.parent.pid")
    real_process_harness.wait_for_marker("approval-trackability.child.pid")
    approval_cancel.cancel()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RepositoryGitVisibilityError)
    assert [command[3:5] for command in commands] == [
        ("check-ignore", "--quiet")
    ]
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "approval-trackability")
        assert borg is not None
        assert borg.state is BorgState.PLAN_APPROVAL_PENDING
        assert store.list_plan_approvals(borg.id) == []
    real_process_harness.assert_tree_absent("approval-trackability")


def test_plan_approve_binds_exact_digest_publishes_golden_and_reaches_ready(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    configure_interactive_cli,
    monkeypatch,
) -> None:
    plan = planning_plan_response()
    repository, plan_attempt, paths = _seed_approval_pending(
        committed_git_repo, planning_cli_repository, "approved-plan", plan
    )
    adapter = MockAdapter(name="openai")
    adapter.queue(MockResponse(payload=_pm_tasks(plan)))
    adapter.queue(MockResponse(payload=_review("approve")))
    configure_interactive_cli(
        repository.root,
        adapter,
        InteractiveIO(
            prompt=lambda _message: None,
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        ),
        state_home=repository.root.parent / ".approval-state",
    )
    selected_roles: list[ApiAgentRole] = []

    def select(config, role, selected_paths, *, interactive):
        selected_roles.append(role)
        return adapter

    monkeypatch.setattr(cli_module, "select_agent", select)

    shown = cli_runner.invoke(cli, ["plan", "show", "approved-plan", "--json"])
    assert shown.exit_code == 0, shown.output
    assert json.loads(shown.output) == plan

    result = cli_runner.invoke(cli, ["plan", "approve", "approved-plan", "--yes"])

    assert result.exit_code == 0, result.output
    assert "Borg 'approved-plan' is ready to execute." in result.output
    assert ".borg/plans/approved-plan.md" in result.output
    assert ".borg/tasks/approved-plan/" in result.output
    project_manager_line = result.output.index("completed Project Manager")
    supervisor_line = result.output.index("completed Supervisor")
    assert project_manager_line < supervisor_line
    assert selected_roles == [ApiAgentRole.PLANNING]
    approved_markdown = """# Release workflow

Add a small, tested release workflow.

## Overall approach

Extend the existing repository conventions and verify public behavior.

## Phases

### 01-release-workflow — Add release workflow

**Goal:** Document and test the release path.

**Technical approach:** Update the tracked README convention.

**Files touched:**
- `README.md` (modified) — Document the release workflow.

**Test strategy:** Assert the documented public workflow.

**Acceptance criteria:**
- The release path is documented.

**Deliverables:**
- Release workflow documentation.

## Code pointers

- `README.md` — It owns repository guidance.
"""
    plan_path = repository.root / ".borg/plans/approved-plan.md"
    assert plan_path.read_text(encoding="utf-8") == approved_markdown
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "approved-plan")
        assert borg is not None
        assert borg.state is BorgState.READY_TO_EXECUTE
        approvals = store.list_plan_approvals(borg.id)
        assert len(approvals) == 1
        approval = approvals[0]
        assert approval.attempt_id == plan_attempt.id
        assert approval.plan_digest == approved_plan_digest(plan)
        assert approval.manifest["plan"] == plan
        assert approval.manifest["plan_path"] == ".borg/plans/approved-plan.md"
        assert approval.manifest["plan.md"] == "sha256:" + hashlib.sha256(
            approved_markdown.encode("utf-8")
        ).hexdigest()
        generation = store.get_current_task_generation(borg.id)
        assert generation is not None
        assert generation.status is TaskGenerationStatus.CURRENT
        tasks = store.list_task_records(generation.id)
        assert len(tasks) == 1
        assert tasks[0].manifest["approved_plan_digest"] == approval.plan_digest
        task_path = repository.root / generation.manifest["tasks"][0]["path"]
        assert task_path.is_file()


def test_plan_approve_interruption_resumes_without_reapproval_or_pm_rerun(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    configure_interactive_cli,
) -> None:
    plan = planning_plan_response()
    repository, _attempt, paths = _seed_approval_pending(
        committed_git_repo, planning_cli_repository, "resume-approval", plan
    )
    adapter = MockAdapter(name="openai")
    adapter.queue(MockResponse(payload=_pm_tasks(plan)))
    adapter.queue(MockResponse(raise_error=RuntimeError("review interrupted")))
    configure_interactive_cli(
        repository.root,
        adapter,
        InteractiveIO(
            prompt=lambda _message: None,
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        ),
        state_home=repository.root.parent / ".resume-approval-state",
    )

    interrupted = cli_runner.invoke(
        cli, ["plan", "approve", "resume-approval", "--yes"]
    )

    assert interrupted.exit_code == 1
    assert "borg plan approve resume-approval" in interrupted.output
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "resume-approval")
        assert borg is not None
        assert borg.state is BorgState.SUPERVISOR_WORKING
        assert len(store.list_plan_approvals(borg.id)) == 1
        assert len(store.list_task_batches(borg.id)) == 1

    adapter.queue(MockResponse(payload=_review("approve")))
    resumed = cli_runner.invoke(
        cli, ["plan", "approve", "resume-approval", "--yes"]
    )

    assert resumed.exit_code == 0, resumed.output
    assert "ready to execute" in resumed.output
    assert "completed Project Manager" in resumed.output
    assert "[retained]" in resumed.output
    assert resumed.output.index("completed Project Manager") < resumed.output.index(
        "completed Supervisor"
    )
    assert len(adapter.calls) == 3
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "resume-approval")
        assert borg is not None
        assert borg.state is BorgState.READY_TO_EXECUTE
        assert len(store.list_plan_approvals(borg.id)) == 1
        assert len(store.list_task_batches(borg.id)) == 1


def test_plan_approve_resumes_publication_before_becoming_ready(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    configure_interactive_cli,
    monkeypatch,
) -> None:
    plan = planning_plan_response()
    repository, _attempt, paths = _seed_approval_pending(
        committed_git_repo, planning_cli_repository, "resume-publication", plan
    )
    adapter = MockAdapter(name="openai")
    adapter.queue(MockResponse(payload=_pm_tasks(plan)))
    adapter.queue(MockResponse(payload=_review("approve")))
    configure_interactive_cli(
        repository.root,
        adapter,
        InteractiveIO(
            prompt=lambda _message: None,
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        ),
        state_home=repository.root.parent / ".resume-publication-state",
    )
    original_checkpoint = TaskPublisher._checkpoint

    def interrupt_before_commit(self, point: str) -> None:
        if point == "before_db_commit":
            raise RuntimeError("publication interrupted before commit")
        original_checkpoint(self, point)

    monkeypatch.setattr(TaskPublisher, "_checkpoint", interrupt_before_commit)
    interrupted = cli_runner.invoke(
        cli, ["plan", "approve", "resume-publication", "--yes"]
    )

    assert interrupted.exit_code == 1
    assert "borg plan approve resume-publication" in interrupted.output
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "resume-publication")
        assert borg is not None
        assert borg.state is BorgState.SUPERVISOR_WORKING
        assert store.get_current_task_generation(borg.id) is None
        generations = store.list_task_generations(borg.id)
        assert len(generations) == 1
        assert generations[0].status is TaskGenerationStatus.PREPARING
        assert len(store.list_plan_approvals(borg.id)) == 1
        assert len(store.list_task_batches(borg.id)) == 1

    monkeypatch.setattr(TaskPublisher, "_checkpoint", original_checkpoint)
    resumed = cli_runner.invoke(
        cli, ["plan", "approve", "resume-publication", "--yes"]
    )

    assert resumed.exit_code == 0, resumed.output
    assert "ready to execute" in resumed.output
    assert len(adapter.calls) == 2
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "resume-publication")
        assert borg is not None
        assert borg.state is BorgState.READY_TO_EXECUTE
        current = store.get_current_task_generation(borg.id)
        assert current is not None
        assert current.status is TaskGenerationStatus.CURRENT
        assert len(store.list_task_generations(borg.id)) == 1
        assert len(store.list_plan_approvals(borg.id)) == 1
        assert len(store.list_task_batches(borg.id)) == 1


def test_plan_approve_publication_cancellation_reports_retained_approval(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    configure_interactive_cli,
    real_process_harness: Any,
    monkeypatch,
) -> None:
    plan = planning_plan_response()
    repository, _attempt, paths = _seed_approval_pending(
        committed_git_repo,
        planning_cli_repository,
        "cancel-publication",
        plan,
    )
    adapter = MockAdapter(name="openai")
    adapter.queue(MockResponse(payload=_pm_tasks(plan)))
    adapter.queue(MockResponse(payload=_review("approve")))
    configure_interactive_cli(
        repository.root,
        adapter,
        InteractiveIO(
            prompt=lambda _message: None,
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        ),
        state_home=repository.root.parent / ".cancel-publication-state",
    )
    observed_cancel: list[CancellationToken] = []
    original_require_git_trackable = publication_module.require_git_trackable

    def blocked_trackability(
        path: Path,
        *,
        root: Path,
        cancel: CancellationToken | None = None,
        command_runner=run_captured,
    ) -> None:
        assert cancel is not None
        assert command_runner is run_captured
        observed_cancel.append(cancel)

        def runner(
            command: list[str], **kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            return run_captured(
                real_process_harness.resistant_argv("cli-task-publication"),
                check=kwargs["check"],
                cancel=kwargs["cancel"],
            )

        require_git_trackable(
            path,
            root=root,
            cancel=cancel,
            command_runner=runner,
        )

    monkeypatch.setattr(
        publication_module,
        "require_git_trackable",
        blocked_trackability,
    )

    def cancel_publication() -> None:
        real_process_harness.wait_for_marker("cli-task-publication.parent.pid")
        real_process_harness.wait_for_marker("cli-task-publication.child.pid")
        observed_cancel[0].cancel()

    canceller = threading.Thread(target=cancel_publication)
    canceller.start()
    interrupted = cli_runner.invoke(
        cli, ["plan", "approve", "cancel-publication", "--yes"]
    )
    canceller.join(timeout=2)

    assert not canceller.is_alive()
    assert interrupted.exit_code == 1
    assert "was interrupted" in interrupted.output
    assert "publishing approved tasks" in interrupted.output
    assert "completed approval was retained" in interrupted.output
    assert len(adapter.calls) == 2
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "cancel-publication")
        assert borg is not None
        assert borg.state is BorgState.SUPERVISOR_WORKING
        attempts = [
            attempt
            for attempt in store.list_planning_attempts(borg.id)
            if attempt.phase == "supervisor_review"
        ]
        assert len(attempts) == 1
        assert attempts[0].status is PlanningAttemptStatus.COMPLETED
        assert store.get_current_task_generation(borg.id) is None
    real_process_harness.assert_tree_absent("cli-task-publication")

    monkeypatch.setattr(
        publication_module,
        "require_git_trackable",
        original_require_git_trackable,
    )
    resumed = cli_runner.invoke(
        cli, ["plan", "approve", "cancel-publication", "--yes"]
    )

    assert resumed.exit_code == 0, resumed.output
    assert "ready to execute" in resumed.output
    assert len(adapter.calls) == 2
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "cancel-publication")
        assert borg is not None
        assert borg.state is BorgState.READY_TO_EXECUTE
        current = store.get_current_task_generation(borg.id)
        assert current is not None
        assert current.status is TaskGenerationStatus.CURRENT


def test_plan_approve_post_commit_cancellation_keeps_ready_outcome(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    configure_interactive_cli,
    monkeypatch,
) -> None:
    plan = planning_plan_response()
    repository, _attempt, paths = _seed_approval_pending(
        committed_git_repo,
        planning_cli_repository,
        "post-commit-cancel",
        plan,
    )
    adapter = MockAdapter(name="openai")
    adapter.queue(MockResponse(payload=_pm_tasks(plan)))
    adapter.queue(MockResponse(payload=_review("approve")))
    configure_interactive_cli(
        repository.root,
        adapter,
        InteractiveIO(
            prompt=lambda _message: None,
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        ),
        state_home=repository.root.parent / ".post-commit-cancel-state",
    )
    original_checkpoint = TaskPublisher._checkpoint

    def cancel_after_commit(self, point: str) -> None:
        original_checkpoint(self, point)
        if point == "after_db_commit":
            assert self.cancel is not None
            self.cancel.cancel()

    monkeypatch.setattr(TaskPublisher, "_checkpoint", cancel_after_commit)

    result = cli_runner.invoke(
        cli, ["plan", "approve", "post-commit-cancel", "--yes"]
    )

    assert result.exit_code == 0, result.output
    assert "ready to execute" in result.output
    assert len(adapter.calls) == 2
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "post-commit-cancel")
        assert borg is not None
        assert borg.state is BorgState.READY_TO_EXECUTE
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


def test_plan_approve_reports_bounded_decomposition_block_without_task_gate(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    configure_interactive_cli,
) -> None:
    plan = planning_plan_response()
    repository, _attempt, paths = _seed_approval_pending(
        committed_git_repo, planning_cli_repository, "blocked-tasks", plan
    )
    adapter = MockAdapter(name="openai")
    for response in (
        _pm_tasks(plan),
        _review("request_changes", "Round one is incomplete."),
        _pm_tasks(plan, title="Document the release workflow revision one"),
        _review("request_changes", "Round two is incomplete."),
        _pm_tasks(plan, title="Document the release workflow revision two"),
        _review("request_changes", "Round three is incomplete."),
    ):
        adapter.queue(MockResponse(payload=response))
    configure_interactive_cli(
        repository.root,
        adapter,
        InteractiveIO(
            prompt=lambda _message: None,
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        ),
        state_home=repository.root.parent / ".blocked-tasks-state",
    )

    result = cli_runner.invoke(cli, ["plan", "approve", "blocked-tasks", "--yes"])

    assert result.exit_code == 0, result.output
    assert "Task decomposition blocked" in result.output
    assert "approval pending" not in result.output.casefold()
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "blocked-tasks")
        assert borg is not None
        assert borg.state is BorgState.BLOCKED
        assert store.get_current_task_generation(borg.id) is None
        assert len(store.list_plan_approvals(borg.id)) == 1


def test_plan_commands_remove_extra_gates_and_standalone_decomposition(
    cli_runner: CliRunner,
) -> None:
    result = cli_runner.invoke(cli, ["plan", "--help"])

    assert result.exit_code == 0
    assert "approve" in result.output
    assert "reject" not in result.output
    assert "decompose" not in result.output
    assert "task-approve" not in result.output
