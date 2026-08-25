"""Real-Git contracts for guarded host worktree allocation."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from betterborg_cli.host_execution import (
    HostWorktreeManager,
    PrimaryCheckoutContaminationError,
    PrimaryCheckoutGuard,
    SafeGit,
    UnsafeGitError,
    WorktreeError,
)
from betterborg_cli.planning import render_task_markdown
from betterborg_cli.store import (
    Borg,
    ExecutionOwnershipError,
    PlanApproval,
    Repository,
    SqliteStore,
    StaleTaskRuntimeError,
)
from betterborg_cli.store.models import TaskRuntimeStatus, utcnow


@pytest.mark.parametrize(
    "arguments",
    [
        ["reset", "HEAD~1"],
        ["reset", "--soft", "HEAD~1"],
        ["reset", "--hard", "HEAD"],
        ["checkout", "--", "README.md"],
        ["checkout", "HEAD", "README.md"],
        ["clean", "-fd"],
        ["push", "--force", "origin", "main"],
        ["rebase", "main"],
        ["branch", "-D", "task"],
        ["branch", "-df", "task"],
        ["branch", "--move", "old", "new"],
        ["branch", "-mnew", "old"],
        ["commit", "--amend", "-m", "rewrite"],
        ["worktree", "remove", "--force", "elsewhere"],
        ["worktree", "add", "-Bproject/demo", "elsewhere", "HEAD~1"],
        ["fetch", "origin", "+main:refs/heads/project/demo"],
        ["fetch", "--refmap=+refs/heads/*:refs/heads/*", "origin"],
        ["fetch", "--update-head-ok", "origin", "main:main"],
    ],
)
def test_safe_git_rejects_destructive_operations(
    committed_git_repo: Path, arguments: list[str]
) -> None:
    with pytest.raises(UnsafeGitError):
        SafeGit(committed_git_repo).run(arguments)


def test_safe_git_rejects_parent_discovery_from_nested_path(
    committed_git_repo: Path,
) -> None:
    nested = committed_git_repo / "nested"
    nested.mkdir()
    with pytest.raises(UnsafeGitError, match="nested path"):
        SafeGit(nested).run(["status", "--porcelain"])


def test_safe_git_strips_repository_routing_environment(
    committed_git_repo: Path, tmp_path: Path
) -> None:
    other = tmp_path / "other"
    _git(tmp_path, "init", "--quiet", str(other))

    result = SafeGit(committed_git_repo).run(
        ["rev-parse", "--show-toplevel"],
        env={
            "PATH": os.environ["PATH"],
            "GIT_DIR": str(other / ".git"),
            "GIT_WORK_TREE": str(other),
        },
    )

    assert Path(result.stdout.strip()).resolve() == committed_git_repo


def test_primary_guard_detects_dirt_and_phase_contamination(
    committed_git_repo: Path,
) -> None:
    guard = PrimaryCheckoutGuard(committed_git_repo)
    guard.assert_clean()

    guard.before_phase("07-host/task", "coding")
    (committed_git_repo / "escaped.py").write_text("ESCAPED = True\n")
    with pytest.raises(
        PrimaryCheckoutContaminationError, match="task work was preserved"
    ) as caught:
        guard.after_phase("07-host/task", "coding")
    assert "escaped.py" in str(caught.value)

    with pytest.raises(PrimaryCheckoutContaminationError, match="before it started"):
        guard.assert_clean()


def test_primary_guard_protect_rejects_dirty_phase_entry(
    committed_git_repo: Path,
) -> None:
    (committed_git_repo / "README.md").write_text("changed once\n")
    entered_phase = False

    with pytest.raises(
        PrimaryCheckoutContaminationError, match="before it started"
    ) as caught:
        with PrimaryCheckoutGuard(committed_git_repo).protect(
            "07-host/task", "coding"
        ):
            entered_phase = True
            (committed_git_repo / "README.md").write_text("changed twice\n")

    assert not entered_phase
    assert "README.md" in str(caught.value)


def test_primary_guard_detects_rename_into_ignored_state_directory(
    committed_git_repo: Path,
) -> None:
    state_directory = committed_git_repo / ".borg/state"
    state_directory.mkdir(parents=True)
    _git(committed_git_repo, "mv", "README.md", ".borg/state/README.md")

    with pytest.raises(
        PrimaryCheckoutContaminationError, match="before it started"
    ) as caught:
        PrimaryCheckoutGuard(committed_git_repo).assert_clean()

    assert "README.md -> .borg/state/README.md" in str(caught.value)


def test_primary_guard_does_not_parse_arrow_in_filename_as_rename(
    committed_git_repo: Path,
) -> None:
    filename = "outside -> .borg-state-hidden"
    (committed_git_repo / filename).write_text("preserve me\n")

    with pytest.raises(PrimaryCheckoutContaminationError) as caught:
        PrimaryCheckoutGuard(
            committed_git_repo,
            ignored_prefixes=(".borg-state-hidden",),
        ).assert_clean()

    assert filename in str(caught.value)


def test_primary_guard_detects_clean_commit_during_phase(
    committed_git_repo: Path,
) -> None:
    guard = PrimaryCheckoutGuard(committed_git_repo)
    original_head = _git(committed_git_repo, "rev-parse", "HEAD").strip()
    guard.before_phase("07-host/task", "coding")

    (committed_git_repo / "escaped.py").write_text("ESCAPED = True\n")
    _git(committed_git_repo, "add", "escaped.py")
    _git(committed_git_repo, "commit", "-m", "escaped commit")

    with pytest.raises(
        PrimaryCheckoutContaminationError, match="HEAD changed"
    ) as caught:
        guard.after_phase("07-host/task", "coding")
    assert original_head in str(caught.value)
    assert _git(committed_git_repo, "status", "--porcelain") == ""


def test_primary_guard_detects_clean_branch_switch_during_phase(
    committed_git_repo: Path,
) -> None:
    guard = PrimaryCheckoutGuard(committed_git_repo)
    original_branch = _git(
        committed_git_repo, "rev-parse", "--abbrev-ref", "HEAD"
    ).strip()
    _git(committed_git_repo, "branch", "escaped-branch")
    guard.before_phase("07-host/task", "coding")

    _git(committed_git_repo, "checkout", "escaped-branch")

    with pytest.raises(
        PrimaryCheckoutContaminationError, match="branch changed"
    ) as caught:
        guard.after_phase("07-host/task", "coding")
    assert f"{original_branch} -> escaped-branch" in str(caught.value)
    assert _git(committed_git_repo, "status", "--porcelain") == ""


def test_allocates_persists_reuses_and_cleans_task_worktree(
    committed_git_repo: Path,
    approved_task_generation,
) -> None:
    database = committed_git_repo.parent / f"{committed_git_repo.name}-host.sqlite3"
    repository = Repository(root=committed_git_repo)
    borg = Borg(repository_id=repository.id, name="executor")
    approval = PlanApproval(
        borg_id=borg.id,
        plan_digest="sha256:plan",
        manifest={"plan.md": "sha256:plan"},
    )
    body = {
        "stage": "07-host-execution",
        "stem": "03-worktrees",
        "title": "Manage guarded host worktrees",
        "why": "Task execution must not modify the primary checkout.",
        "scope": ["Allocate a sibling worktree."],
        "implementation_notes": [],
        "acceptance_criteria": ["The branch identity is durable."],
        "tests": ["Use a real Git repository."],
        "dependencies": [],
        "out_of_scope": [],
        "plan_refs": ["P1.deliverable.3"],
        "estimate_complexity": "medium",
    }
    with SqliteStore.open(database) as store:
        store.add_repository(repository)
        store.add_borg(borg)
        store.append_plan_approval(approval)
        fixture = approved_task_generation(
            store,
            borg,
            approval,
            body=body,
            round_number=1,
            task_ref="worktrees",
        )
        generation, task = fixture.generation, fixture.task
        durable_root = (
            committed_git_repo / ".borg/tasks" / borg.name / str(generation.id)
        )
        task_path = durable_root / task.stage / f"{task.stem}.md"
        task_path.parent.mkdir(parents=True)
        task_path.write_text(render_task_markdown(body), encoding="utf-8")
        store._promote_published_task_generation(
            generation.id, durable_root=durable_root
        )

    _git(committed_git_repo, "add", ".borg")
    _git(committed_git_repo, "commit", "-m", "add task")

    now = utcnow()
    worktree_root = (
        committed_git_repo.parent / f"{committed_git_repo.name}-task-worktrees"
    )
    manager = HostWorktreeManager(
        committed_git_repo,
        worktree_root,
        source_branch=SafeGit(committed_git_repo).current_branch(),
    )
    with SqliteStore.open(database) as store:
        acquisition = store.acquire_execution_run(
            borg.id,
            generation.id,
            lease_duration=timedelta(minutes=5),
            now=now,
        )
        assert acquisition.owner_token is not None
        with pytest.raises(ExecutionOwnershipError):
            manager.prepare_current_task_worktrees(
                store,
                run_id=acquisition.run_id,
                owner_token="x" * 43,
                generation_id=generation.id,
                project_name=borg.name,
                now=now,
            )
        assert not SafeGit(committed_git_repo).branch_exists("project/executor")
        first = manager.prepare_current_task_worktrees(
            store,
            run_id=acquisition.run_id,
            owner_token=acquisition.owner_token,
            generation_id=generation.id,
            project_name=borg.name,
            now=now,
        )
        second = manager.prepare_current_task_worktrees(
            store,
            run_id=acquisition.run_id,
            owner_token=acquisition.owner_token,
            generation_id=generation.id,
            project_name=borg.name,
            now=now,
        )
        runtime = store.get_task_runtime(task.id)
        claim = store.claim_dependency_ready_task(
            acquisition.run_id,
            acquisition.owner_token,
            lease_duration=timedelta(minutes=2),
            now=now,
        )
        assert claim is not None
        with pytest.raises(StaleTaskRuntimeError, match="cannot be replaced"):
            store.transition_task_runtime(
                acquisition.run_id,
                acquisition.owner_token,
                claim.id,
                claim.claim_token,
                expected_status=TaskRuntimeStatus.CLAIMED,
                new_status=TaskRuntimeStatus.ENVIRONMENT,
                branch=f"{runtime.branch}-replacement",
                worktree_path=f"{runtime.worktree_path}-replacement",
                now=now,
            )

    assert first == second
    assert len(first) == 1
    spec = first[0]
    assert re.fullmatch(
        r"betterborg-tasks/07-host-execution/03-worktrees-[0-9a-f]{16}",
        spec.branch,
    )
    assert runtime is not None
    assert runtime.branch == spec.branch
    assert runtime.worktree_path == str(spec.path)
    assert spec.path.parent == worktree_root / "07-host-execution"
    assert SafeGit(spec.path).current_branch() == spec.branch
    assert SafeGit(committed_git_repo).branch_exists("project/executor")

    assert not manager.cleanup_task_worktree(
        replace(runtime, status=TaskRuntimeStatus.BLOCKED)
    )
    assert spec.path.exists()
    assert manager.cleanup_task_worktree(
        replace(runtime, status=TaskRuntimeStatus.DONE)
    )
    assert not spec.path.exists()
    assert SafeGit(committed_git_repo).branch_exists(spec.branch)


def test_project_base_only_fast_forwards_and_preserves_dirty_completed_work(
    committed_git_repo: Path,
) -> None:
    worktree_root = committed_git_repo.parent / (
        f"{committed_git_repo.name}-project-worktrees"
    )
    source = SafeGit(committed_git_repo).current_branch()
    manager = HostWorktreeManager(
        committed_git_repo, worktree_root, source_branch=source
    )
    project = manager.ensure_project_base("demo")
    old_tip = _git(committed_git_repo, "rev-parse", project).strip()

    (committed_git_repo / "advance.txt").write_text("advance\n")
    _git(committed_git_repo, "add", "advance.txt")
    _git(committed_git_repo, "commit", "-m", "advance source")
    assert manager.ensure_project_base("demo") == project
    assert _git(committed_git_repo, "rev-parse", project).strip() != old_tip

    branch = "betterborg-tasks/07-host-execution/dirty-0123456789abcdef"
    path = worktree_root / "07-host-execution/dirty-0123456789abcdef"
    SafeGit(committed_git_repo).add_worktree(path, branch, base=project)
    (path / "uncommitted.txt").write_text("preserve me\n")
    from betterborg_cli.store import TaskRuntime

    runtime = TaskRuntime(
        generation_id=uuid4(),
        task_id=uuid4(),
        status=TaskRuntimeStatus.DONE,
        branch=branch,
        worktree_path=str(path),
    )
    with pytest.raises(WorktreeError, match="dirty; preserving"):
        manager.cleanup_task_worktree(runtime)
    assert (path / "uncommitted.txt").read_text() == "preserve me\n"


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
