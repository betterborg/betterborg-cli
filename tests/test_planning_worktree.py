"""Detached planning worktree and context-materialization contracts."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import betterborg_cli.planning.worktree as worktree_module
from betterborg_cli.agent_runtime import CancellationToken, run_captured
from betterborg_cli.agent_runtime.mock import MockAdapter
from betterborg_cli.planning import (
    ArchitectLoop,
    PlanningWorktreeError,
    ProjectManagerLoop,
    SupervisorLoop,
    TechLeadLoop,
    materialize_planning_worktree,
)
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.repo_analysis import (
    build_machine_report,
    render_markdown_report,
)
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.store import (
    Borg,
    PlanChangeRequest,
    PlanningAttempt,
    PlanningFinding,
    PlanningQuestion,
    PrdSession,
    Repository,
    SqliteStore,
)


def test_materializes_detached_planning_context_without_touching_primary(
    committed_git_repo: Path,
    write_repository_config,
    persist_repository_analysis,
) -> None:
    repository = Repository(root=committed_git_repo)
    borg = Borg(repository_id=repository.id, name="safe-planning")
    write_repository_config(committed_git_repo, repository)
    prd_path = Path(".betterborg/prds/safe-planning.md")
    (committed_git_repo / prd_path).parent.mkdir(parents=True)
    (committed_git_repo / prd_path).write_text(
        "# Confirmed PRD\n\nKeep the primary checkout unchanged.\n",
        encoding="utf-8",
    )
    deliberate_path = Path(".betterborg/notes/operator-context.md")
    (committed_git_repo / deliberate_path).parent.mkdir(parents=True)
    (committed_git_repo / deliberate_path).write_text(
        "# Operator context\n\nThis untracked Borg document is deliberate.\n",
        encoding="utf-8",
    )
    (committed_git_repo / ".betterborg/notes/not-supplied.md").write_text(
        "must stay out of the planning worktree\n", encoding="utf-8"
    )
    (committed_git_repo / "unrelated.py").write_text(
        "PRIMARY_DIRTY = True\n", encoding="utf-8"
    )

    database = committed_git_repo.parent / f"{committed_git_repo.name}-planning.db"
    with SqliteStore.open(database) as store:
        store.add_repository(repository)
        store.add_borg(borg)
        store.add_prd_session(
            PrdSession(
                repository_id=repository.id,
                borg_id=borg.id,
                prd_path=prd_path,
            )
        )
        analysis, packages = persist_repository_analysis(store, repository)
        prompts = {
            role: store.append_generated_prompt(
                repository_id=repository.id,
                analysis_id=analysis.id,
                role=role,
                body_md=f"# {role.title()} guidance\n\nUse the repository commands.\n",
            )
            for role in ("coding", "review", "merge")
        }
        attempt = PlanningAttempt(
            borg_id=borg.id,
            phase="tech_review",
            round=1,
            adapter="mock",
            model="planning-model",
        )
        store.append_planning_attempt(attempt)
        question = PlanningQuestion(
            borg_id=borg.id,
            attempt_id=attempt.id,
            round=1,
            questions=[{"id": "platforms", "question": "Which platforms?"}],
        )
        store.append_planning_question(question)
        answered = store.answer_planning_question(
            question.id,
            [{"q_id": "platforms", "answer": "Linux and macOS."}],
        )
        finding = PlanningFinding(
            borg_id=borg.id,
            attempt_id=attempt.id,
            round=1,
            severity="major",
            message="Rollback is not explicit.",
            suggestion="Add a recovery step.",
        )
        change_request = PlanChangeRequest(
            borg_id=borg.id,
            round=2,
            note="Keep migrations forward-only.",
            decided_by="operator",
        )
        store.append_planning_finding(finding)
        store.append_plan_change_request(change_request)

        status_before = _git(committed_git_repo, "status", "--short")
        worktrees_before = _git(committed_git_repo, "worktree", "list", "--porcelain")
        primary_prd_before = (committed_git_repo / prd_path).read_text(encoding="utf-8")
        current_plan = "# Current plan\n\n1. Preserve the checkout.\n"

        with materialize_planning_worktree(
            repository,
            borg,
            store,
            current_plan=current_plan,
            dirty_borg_documents=[deliberate_path],
        ) as worktree:
            assert not worktree.is_relative_to(committed_git_repo)
            assert worktree.is_dir()
            assert _git(worktree, "rev-parse", "--is-inside-work-tree") == "true\n"
            detached = subprocess.run(
                ["git", "-C", str(worktree), "symbolic-ref", "-q", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
            assert detached.returncode == 1
            assert (worktree / "README.md").is_file()
            assert not (worktree / "unrelated.py").exists()
            assert not (worktree / ".betterborg/notes/not-supplied.md").exists()
            assert (worktree / deliberate_path).read_text(encoding="utf-8") == (
                committed_git_repo / deliberate_path
            ).read_text(encoding="utf-8")
            assert (
                worktree / prd_path
            ).read_text(encoding="utf-8") == primary_prd_before
            tracked = worktree / ".betterborg/config.toml"
            assert tracked.read_text(encoding="utf-8") == (
                committed_git_repo / ".betterborg/config.toml"
            ).read_text(encoding="utf-8")
            assert (worktree / ".betterborg/score.md").read_text(encoding="utf-8") == (
                render_markdown_report(build_machine_report(analysis, packages))
            )
            for role, prompt in prompts.items():
                assert (
                    worktree / f".betterborg/prompts/{role}.system.md"
                ).read_text(encoding="utf-8") == prompt.body_md
            assert (worktree / ".betterborg/plans/safe-planning.md").read_text(
                encoding="utf-8"
            ) == current_plan

            context = worktree / ".betterborg/state/planning/context"
            identity = _json(context / "repository.json")
            assert identity["repository"]["id"] == str(repository.id)
            assert identity["repository"]["head_sha"] == _git(
                committed_git_repo, "rev-parse", "HEAD"
            ).strip()
            persisted_analysis = _json(context / "analysis.json")
            assert persisted_analysis["id"] == str(analysis.id)
            assert persisted_analysis["analysis"] == analysis.analysis_json
            questions = _json(context / "questions.json")
            assert questions[0]["id"] == str(answered.id)
            assert questions[0]["answers"] == answered.answers
            changes = _json(context / "change-requests.json")
            assert changes[0]["note"] == change_request.note
            findings = _json(context / "findings.json")
            assert findings[0]["message"] == finding.message
            manifest = _json(context / "manifest.json")
            assert manifest["confirmed_prd"] == prd_path.as_posix()
            assert manifest["current_plan"] == ".betterborg/plans/safe-planning.md"
            assert manifest["dirty_borg_documents"] == [deliberate_path.as_posix()]
            assert set(manifest["prompts"]) == {"coding", "review", "merge"}

            assert _git(committed_git_repo, "status", "--short") == status_before
            assert (committed_git_repo / prd_path).read_text(
                encoding="utf-8"
            ) == primary_prd_before
            materialized_path = worktree

        assert not materialized_path.exists()
        assert _git(committed_git_repo, "status", "--short") == status_before
        assert (
            _git(committed_git_repo, "worktree", "list", "--porcelain")
            == worktrees_before
        )


def test_rejects_dirty_source_outside_borg_documents(
    committed_git_repo: Path, write_repository_config
) -> None:
    repository = Repository(root=committed_git_repo)
    borg = Borg(repository_id=repository.id, name="contained-planning")
    write_repository_config(committed_git_repo, repository)
    (committed_git_repo / "dirty-source.py").write_text(
        "DIRTY = True\n", encoding="utf-8"
    )
    database = committed_git_repo.parent / f"{committed_git_repo.name}-guard.db"

    with SqliteStore.open(database) as store:
        store.add_repository(repository)
        store.add_borg(borg)

        with pytest.raises(
            PlanningWorktreeError,
            match=r"repository-relative \.betterborg file",
        ):
            with materialize_planning_worktree(
                repository,
                borg,
                store,
                dirty_borg_documents=[Path("dirty-source.py")],
            ):
                pytest.fail("unsafe dirty source reached a planning worktree")


def test_surfaces_worktree_removal_failure(
    committed_git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    persist_planning_context,
) -> None:
    database = committed_git_repo.parent / f"{committed_git_repo.name}-cleanup.db"
    original_run_git = worktree_module._run_git

    def fail_worktree_removal(
        root: Path, *arguments: str, **kwargs: Any
    ) -> None:
        if arguments[:2] == ("worktree", "remove"):
            raise subprocess.CalledProcessError(
                1,
                ["git", *arguments],
                stderr="simulated cleanup failure",
            )
        original_run_git(root, *arguments, **kwargs)

    materialized_path: Path | None = None
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "cleanup-failure"
        )
        try:
            with monkeypatch.context() as cleanup_failure:
                cleanup_failure.setattr(
                    worktree_module, "_run_git", fail_worktree_removal
                )
                with pytest.raises(
                    PlanningWorktreeError,
                    match="unable to remove planning worktree",
                ):
                    with materialize_planning_worktree(
                        repository, borg, store
                    ) as materialized_path:
                        assert materialized_path.is_dir()
            assert materialized_path is not None
            assert materialized_path.is_dir()
        finally:
            if materialized_path is not None and materialized_path.exists():
                original_run_git(
                    committed_git_repo,
                    "worktree",
                    "remove",
                    "--force",
                    str(materialized_path),
                )


def test_preserves_caller_error_when_worktree_removal_also_fails(
    committed_git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    persist_planning_context,
) -> None:
    database = committed_git_repo.parent / f"{committed_git_repo.name}-body-error.db"
    original_run_git = worktree_module._run_git

    def fail_worktree_removal(
        root: Path, *arguments: str, **kwargs: Any
    ) -> None:
        if arguments[:2] == ("worktree", "remove"):
            raise subprocess.CalledProcessError(1, ["git", *arguments])
        original_run_git(root, *arguments, **kwargs)

    materialized_path: Path | None = None
    caller_error = OSError("architect validation failed")
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "caller-failure"
        )
        try:
            with monkeypatch.context() as cleanup_failure:
                cleanup_failure.setattr(
                    worktree_module, "_run_git", fail_worktree_removal
                )
                with pytest.raises(
                    OSError, match="architect validation failed"
                ) as caught:
                    with materialize_planning_worktree(
                        repository, borg, store
                    ) as materialized_path:
                        raise caller_error
            assert caught.value is caller_error
            assert any(
                "unable to remove planning worktree" in note
                for note in caught.value.__notes__
            )
        finally:
            if materialized_path is not None and materialized_path.exists():
                original_run_git(
                    committed_git_repo,
                    "worktree",
                    "remove",
                    "--force",
                    str(materialized_path),
                )


@pytest.mark.parametrize(
    ("blocked_operation", "dirty_document"),
    [
        ("status", True),
        ("revision", False),
        ("creation", False),
    ],
)
def test_cancellation_reaps_each_planning_worktree_git_process(
    committed_git_repo: Path,
    persist_planning_context,
    real_process_harness: Any,
    blocked_operation: str,
    dirty_document: bool,
) -> None:
    database = committed_git_repo.parent / f"planning-{blocked_operation}.db"
    cancel = CancellationToken()
    errors: list[BaseException] = []
    document = Path(".betterborg/notes/cancel.md")
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo,
            store,
            f"cancel-{blocked_operation}",
        )
    if dirty_document:
        (committed_git_repo / document).parent.mkdir(parents=True)
        (committed_git_repo / document).write_text("cancel me\n", encoding="utf-8")

    def runner(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        arguments = command[3:]
        target_prefix = {
            "status": ["status", "--porcelain"],
            "revision": ["rev-parse", "--verify"],
            "creation": ["worktree", "add"],
        }[blocked_operation]
        is_target = arguments[:2] == target_prefix
        if is_target:
            return run_captured(
                real_process_harness.resistant_argv(
                    f"planning-{blocked_operation}"
                ),
                check=kwargs["check"],
                cancel=kwargs["cancel"],
            )
        return run_captured(command, **kwargs)

    def materialize() -> None:
        try:
            with SqliteStore.open(database) as store:
                with materialize_planning_worktree(
                    repository,
                    borg,
                    store,
                    dirty_borg_documents=[document] if dirty_document else (),
                    cancel=cancel,
                    command_runner=runner,
                ):
                    pytest.fail("cancelled Git operation exposed a worktree")
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=materialize)
    worker.start()
    real_process_harness.wait_for_marker(
        f"planning-{blocked_operation}.parent.pid"
    )
    real_process_harness.wait_for_marker(
        f"planning-{blocked_operation}.child.pid"
    )
    cancel.cancel()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(
        errors[0], PlanningWorktreeError | subprocess.CalledProcessError
    )
    real_process_harness.assert_tree_absent(f"planning-{blocked_operation}")


@pytest.mark.parametrize(
    "role",
    ["architect", "project-manager", "tech-lead", "supervisor"],
)
def test_cancellation_reaps_planning_constructor_repository_discovery(
    committed_git_repo: Path,
    persist_planning_context,
    real_process_harness: Any,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    database = committed_git_repo.parent / f"planning-{role}-discovery.db"
    constructor_cancel = CancellationToken()
    errors: list[BaseException] = []
    adapter = MockAdapter(name="openai")
    marker = f"planning-{role}-discovery"
    original_discover = RepoPaths.discover

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo,
            store,
            f"{role}-discovery",
        )

        def blocked_discover(
            start: Path | None = None,
            *,
            cancel: CancellationToken | None = None,
            command_runner: Any = run_captured,
        ) -> RepoPaths:
            del command_runner
            assert cancel is constructor_cancel

            def runner(
                _command: list[str], **kwargs: Any
            ) -> subprocess.CompletedProcess[str]:
                assert kwargs["cancel"] is cancel
                return run_captured(
                    real_process_harness.resistant_argv(marker),
                    check=kwargs["check"],
                    cancel=kwargs["cancel"],
                )

            return original_discover(
                start,
                cancel=cancel,
                command_runner=runner,
            )

        monkeypatch.setattr(RepoPaths, "discover", blocked_discover)

        def construct() -> None:
            try:
                _construct_planning_role(
                    role,
                    repository,
                    borg,
                    store,
                    adapter,
                    constructor_cancel,
                )
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=construct)
        worker.start()
        real_process_harness.wait_for_marker(f"{marker}.parent.pid")
        real_process_harness.wait_for_marker(f"{marker}.child.pid")
        constructor_cancel.cancel()
        worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert adapter.calls == []
    real_process_harness.assert_tree_absent(marker)


def test_cancelled_completed_creation_is_removed(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    database = committed_git_repo.parent / "planning-creation-race.db"
    cancel = CancellationToken()
    destination: Path | None = None
    worktrees_before = _git(committed_git_repo, "worktree", "list", "--porcelain")

    def runner(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        nonlocal destination
        result = run_captured(command, **kwargs)
        if command[3:5] != ["worktree", "add"]:
            return result
        destination = Path(command[-2])
        assert result.returncode == 0
        cancel.cancel()
        return subprocess.CompletedProcess(
            result.args,
            -1,
            result.stdout,
            result.stderr,
        )

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "creation-race"
        )
        with pytest.raises(
            PlanningWorktreeError,
            match="unable to materialize planning worktree",
        ):
            with materialize_planning_worktree(
                repository,
                borg,
                store,
                cancel=cancel,
                command_runner=runner,
            ):
                pytest.fail("cancelled worktree creation reached the caller")

    assert destination is not None
    assert not destination.exists()
    assert (
        _git(committed_git_repo, "worktree", "list", "--porcelain")
        == worktrees_before
    )


def test_cancelled_completed_removal_is_not_retried(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    database = committed_git_repo.parent / "planning-removal-race.db"
    cancel = CancellationToken()
    removal_calls: list[dict[str, Any]] = []

    def runner(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        result = run_captured(command, **kwargs)
        if command[3:5] != ["worktree", "remove"]:
            return result
        removal_calls.append(dict(kwargs))
        assert result.returncode == 0
        cancel.cancel()
        return subprocess.CompletedProcess(
            result.args,
            -1,
            result.stdout,
            result.stderr,
        )

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "removal-race"
        )
        with materialize_planning_worktree(
            repository,
            borg,
            store,
            cancel=cancel,
            command_runner=runner,
        ) as worktree:
            assert worktree.is_dir()

    assert not worktree.exists()
    assert removal_calls == [
        {
            "check": True,
            "cancel": cancel,
            "terminate_on_cancel": True,
            "deadline": None,
        }
    ]


def test_already_cancelled_removal_uses_shared_deadline_and_registered_force(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    database = committed_git_repo.parent / "planning-cancelled-removal.db"
    cancel = CancellationToken(grace_seconds=0.5)
    removal_calls: list[dict[str, Any]] = []

    def runner(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        if command[3:5] == ["worktree", "remove"]:
            removal_calls.append(dict(kwargs))
        return run_captured(command, **kwargs)

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "cancelled-removal"
        )
        with materialize_planning_worktree(
            repository,
            borg,
            store,
            cancel=cancel,
            command_runner=runner,
        ) as worktree:
            assert worktree.is_dir()
            cancel.cancel()
            shared_deadline = cancel.force_deadline

        assert not worktree.exists()

    assert shared_deadline is not None
    assert removal_calls == [
        {
            "check": True,
            "cancel": cancel,
            "terminate_on_cancel": False,
            "deadline": shared_deadline,
        }
    ]


def test_cancelled_removal_reaps_resistant_tree_by_shared_deadline(
    committed_git_repo: Path,
    real_process_harness: Any,
) -> None:
    cancel = CancellationToken(grace_seconds=0.25)
    cancel.cancel()
    shared_deadline = cancel.force_deadline
    calls: list[dict[str, Any]] = []

    def runner(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        assert command[3:5] == ["worktree", "remove"]
        calls.append(dict(kwargs))
        return run_captured(
            real_process_harness.resistant_argv("planning-removal"),
            **kwargs,
        )

    started = time.monotonic()
    with pytest.raises(subprocess.CalledProcessError):
        worktree_module._remove_worktree(
            committed_git_repo,
            committed_git_repo.parent / "simulated-planning-worktree",
            cancel=cancel,
            command_runner=runner,
        )

    assert time.monotonic() - started < 1.5
    assert calls == [
        {
            "check": True,
            "cancel": cancel,
            "terminate_on_cancel": False,
            "deadline": shared_deadline,
        }
    ]
    real_process_harness.assert_tree_absent("planning-removal")


def test_records_token_aware_planning_git_command_inventory(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    database = committed_git_repo.parent / "planning-command-inventory.db"
    cancel = CancellationToken()
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    document = Path(".betterborg/notes/inventory.md")

    def runner(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        calls.append((tuple(command), dict(kwargs)))
        return run_captured(command, **kwargs)

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "command-inventory"
        )
        (committed_git_repo / document).parent.mkdir(parents=True)
        (committed_git_repo / document).write_text(
            "inventory\n", encoding="utf-8"
        )
        with materialize_planning_worktree(
            repository,
            borg,
            store,
            dirty_borg_documents=[document],
            cancel=cancel,
            command_runner=runner,
        ):
            pass

    operations = [command[3:5] for command, _kwargs in calls]
    assert operations == [
        ("rev-parse", "--show-toplevel"),
        ("status", "--porcelain"),
        ("rev-parse", "--verify"),
        ("worktree", "add"),
        ("worktree", "remove"),
    ]
    assert all(kwargs["cancel"] is cancel for _command, kwargs in calls)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _construct_planning_role(
    role: str,
    repository: Repository,
    borg: Borg,
    store: SqliteStore,
    adapter: MockAdapter,
    cancel: CancellationToken,
) -> object:
    shared = {
        "model": "planning-model",
        "cancel": cancel,
    }
    io = InteractiveIO(
        prompt=lambda _message: None,
        confirm=lambda _message, _default: False,
        write=lambda _message: None,
    )
    if role == "architect":
        return ArchitectLoop(
            repository,
            borg,
            store,
            adapter,
            io=io,
            **shared,
        )
    if role == "project-manager":
        return ProjectManagerLoop(
            repository,
            borg,
            store,
            adapter,
            **shared,
        )
    if role == "tech-lead":
        return TechLeadLoop(
            repository,
            borg,
            store,
            adapter,
            io=io,
            **shared,
        )
    if role == "supervisor":
        return SupervisorLoop(
            repository,
            borg,
            store,
            adapter,
            **shared,
        )
    raise AssertionError(f"unknown planning role {role}")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
