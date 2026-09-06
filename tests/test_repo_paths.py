"""Filesystem contracts for repository-local Betterborg data."""

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch

from betterborg_cli import repository_files as repository_files_module
from betterborg_cli.agent_runtime import CancellationToken, run_captured
from betterborg_cli.repo_paths import (
    HOME_VARIABLE,
    MANAGED_IGNORE_BEGIN,
    MANAGED_IGNORE_END,
    BetterborgHomeError,
    RepoPaths,
    ensure_managed_gitignore,
)


@pytest.mark.parametrize(
    "module",
    (
        "betterborg_cli.repo_paths",
        "betterborg_cli.repository_config",
        "betterborg_cli.repository_files",
        "betterborg_cli.workspace_trust",
    ),
)
def test_repository_modules_support_cold_imports(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_discover_uses_nearest_git_root(git_repo: Path) -> None:
    nested_repo = git_repo / "packages" / "nearest"
    child = nested_repo / "src" / "package"
    child.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", str(nested_repo)], check=True)

    paths = RepoPaths.discover(child)

    assert paths.root == nested_repo
    assert paths.tracked_dir == nested_repo / ".betterborg"
    assert paths.state_dir == nested_repo / ".betterborg" / "state"
    assert paths.artifacts_dir == nested_repo / ".betterborg" / "state" / "artifacts"
    assert paths.improvement_prds_dir == (
        nested_repo / ".betterborg" / "prds" / "improvements"
    )


def test_discover_forwards_cancellation_to_the_registered_runner(
    git_repo: Path,
) -> None:
    cancel = CancellationToken()
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, f"{git_repo}\n", "")

    paths = RepoPaths.discover(
        git_repo,
        cancel=cancel,
        command_runner=runner,
    )

    assert paths.root == git_repo
    assert calls == [
        (
            ["git", "-C", str(git_repo), "rev-parse", "--show-toplevel"],
            {"check": True, "cancel": cancel},
        )
    ]


def test_discover_cancellation_reaps_the_git_process_tree(
    git_repo: Path,
    real_process_harness: Any,
) -> None:
    cancel = CancellationToken(grace_seconds=0.05)
    errors: list[BaseException] = []

    def runner(_command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return run_captured(
            real_process_harness.resistant_argv("repo-root"),
            cancel=kwargs["cancel"],
            check=kwargs["check"],
        )

    def discover() -> None:
        try:
            RepoPaths.discover(git_repo, cancel=cancel, command_runner=runner)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=discover)
    worker.start()
    real_process_harness.wait_for_marker("repo-root.parent.pid")
    real_process_harness.wait_for_marker("repo-root.child.pid")
    cancel.cancel()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    real_process_harness.assert_tree_absent("repo-root")


def test_managed_ignore_keeps_documents_trackable_and_ignores_state(
    git_repo: Path,
) -> None:
    paths = RepoPaths.discover(git_repo)
    ensure_managed_gitignore(paths)
    document = paths.tracked_dir / "project.md"
    state = paths.state_dir / "betterborg.sqlite3"
    artifact = paths.artifacts_dir / "result.json"
    for candidate in (document, state, artifact):
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text("test\n", encoding="utf-8")

    status = subprocess.run(
        ["git", "-C", str(git_repo), "status", "--short", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = subprocess.run(
        ["git", "-C", str(git_repo), "check-ignore", "--quiet", str(document)],
        check=False,
    )

    assert "?? .betterborg/project.md" in status.stdout
    assert ".betterborg/state" not in status.stdout
    assert tracked.returncode == 1
    for ignored_path in (state, artifact):
        ignored = subprocess.run(
            [
                "git",
                "-C",
                str(git_repo),
                "check-ignore",
                "--quiet",
                str(ignored_path),
            ],
            check=False,
        )
        assert ignored.returncode == 0


def test_undeclared_home_keeps_every_path_inside_the_repository(
    git_repo: Path,
) -> None:
    paths = RepoPaths.discover(git_repo)

    tracked = git_repo / ".betterborg"
    assert paths.tracked_dir == tracked
    assert paths.tracked_in_repository
    assert paths.tracked_root == git_repo
    assert paths.state_dir == tracked / "state"
    assert paths.artifacts_dir == tracked / "state" / "artifacts"
    assert paths.task_staging_dir == tracked / "state" / "task-staging"
    assert paths.prompts_dir == tracked / "prompts"
    assert paths.prds_dir == tracked / "prds"
    assert paths.improvement_prds_dir == tracked / "prds" / "improvements"
    assert paths.plans_dir == tracked / "plans"
    assert paths.tasks_dir == tracked / "tasks"
    assert paths.score_report == tracked / "score.md"


def test_declared_home_moves_every_derived_path_together(
    git_repo: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: MonkeyPatch,
) -> None:
    home = tmp_path_factory.mktemp("betterborg-home")
    monkeypatch.setenv(HOME_VARIABLE, str(home))

    paths = RepoPaths.discover(git_repo)

    assert paths.root == git_repo
    assert paths.tracked_dir == home
    assert not paths.tracked_in_repository
    assert paths.tracked_root == home
    assert paths.state_dir == home / "state"
    assert paths.artifacts_dir == home / "state" / "artifacts"
    assert paths.task_staging_dir == home / "state" / "task-staging"
    assert paths.prompts_dir == home / "prompts"
    assert paths.prds_dir == home / "prds"
    assert paths.improvement_prds_dir == home / "prds" / "improvements"
    assert paths.plans_dir == home / "plans"
    assert paths.tasks_dir == home / "tasks"
    assert paths.score_report == home / "score.md"
    # The worktrees directory is already a sibling of the repository and is
    # not the operator's to place.
    assert paths.worktrees_dir == git_repo.parent / ".betterborg-worktrees" / (
        git_repo.name
    )


def test_declared_home_inside_the_repository_is_refused(
    git_repo: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    inside = git_repo / "nested" / "home"
    monkeypatch.setenv(HOME_VARIABLE, str(inside))

    with pytest.raises(BetterborgHomeError) as failure:
        RepoPaths.discover(git_repo)

    message = str(failure.value)
    assert HOME_VARIABLE in message
    assert "resolves inside the repository" in message
    assert str(git_repo) in message


def test_declared_home_reached_by_symlink_into_the_repository_is_refused(
    git_repo: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: MonkeyPatch,
) -> None:
    link = tmp_path_factory.mktemp("outside") / "looks-outside"
    link.symlink_to(git_repo / "inside-home", target_is_directory=True)
    monkeypatch.setenv(HOME_VARIABLE, str(link))

    with pytest.raises(BetterborgHomeError, match="resolves inside the repository"):
        RepoPaths.discover(git_repo)


def test_declared_home_containing_the_repository_is_refused(
    git_repo: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(HOME_VARIABLE, str(git_repo.parent))

    with pytest.raises(BetterborgHomeError) as failure:
        RepoPaths.discover(git_repo)

    message = str(failure.value)
    assert HOME_VARIABLE in message
    assert "contains the repository" in message
    assert str(git_repo) in message


def test_declared_home_reached_by_symlink_around_the_repository_is_refused(
    git_repo: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: MonkeyPatch,
) -> None:
    link = tmp_path_factory.mktemp("outside") / "looks-beside"
    link.symlink_to(git_repo.parent, target_is_directory=True)
    monkeypatch.setenv(HOME_VARIABLE, str(link))

    with pytest.raises(BetterborgHomeError, match="contains the repository"):
        RepoPaths.discover(git_repo)


def test_declared_home_must_be_absolute(
    git_repo: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(HOME_VARIABLE, "betterborg-home")

    with pytest.raises(BetterborgHomeError, match="must name an absolute path"):
        RepoPaths.discover(git_repo)


def test_declared_home_leaves_the_repository_ignore_file_alone(
    git_repo: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: MonkeyPatch,
) -> None:
    prior = "dist/\n*.log\n"
    (git_repo / ".gitignore").write_text(prior, encoding="utf-8")
    monkeypatch.setenv(HOME_VARIABLE, str(tmp_path_factory.mktemp("home")))
    paths = RepoPaths.discover(git_repo)

    ensure_managed_gitignore(paths)

    assert paths.gitignore.read_text(encoding="utf-8") == prior


def test_in_checkout_names_a_tracked_file_the_same_way_from_either_home(
    git_repo: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv(HOME_VARIABLE, raising=False)
    inside = RepoPaths.discover(git_repo)
    monkeypatch.setenv(HOME_VARIABLE, str(tmp_path_factory.mktemp("home")))
    relocated = RepoPaths.discover(git_repo)

    # Every checkout Betterborg prepares carries its context under
    # ``.betterborg``, wherever this repository's own tracked directory sits.
    for paths in (inside, relocated):
        assert paths.in_checkout(paths.prds_dir / "sentinel.md") == Path(
            ".betterborg/prds/sentinel.md"
        )
        assert paths.in_checkout(paths.plans_dir / "sentinel.md") == Path(
            ".betterborg/plans/sentinel.md"
        )
        assert paths.in_checkout(paths.score_report) == Path(".betterborg/score.md")


def test_worktrees_are_placed_in_sibling_repository_directory(git_repo: Path) -> None:
    paths = RepoPaths.discover(git_repo)

    expected = git_repo.parent / ".betterborg-worktrees" / git_repo.name
    assert paths.worktrees_dir == expected
    assert paths.worktrees_dir.parent.parent == git_repo.parent


def test_managed_ignore_update_is_idempotent_and_preserves_existing_rules(
    git_repo: Path,
) -> None:
    paths = RepoPaths.discover(git_repo)
    paths.gitignore.write_text("dist/\n*.log\n", encoding="utf-8")

    ensure_managed_gitignore(paths)
    first_update = paths.gitignore.read_text(encoding="utf-8")
    ensure_managed_gitignore(paths)

    assert paths.gitignore.read_text(encoding="utf-8") == first_update
    assert first_update.startswith("dist/\n*.log\n\n")
    assert first_update.count(MANAGED_IGNORE_BEGIN) == 1
    assert first_update.count(MANAGED_IGNORE_END) == 1
    assert first_update.count(".betterborg/state/\n") == 1


@pytest.mark.parametrize("interrupt_after_replacement", [False, True])
def test_managed_ignore_interruption_preserves_prior_or_complete_canonical_bytes(
    git_repo: Path,
    monkeypatch: MonkeyPatch,
    interrupt_after_replacement: bool,
) -> None:
    paths = RepoPaths.discover(git_repo)
    prior = "dist/\n*.log\n"
    canonical = (
        "dist/\n*.log\n\n"
        f"{MANAGED_IGNORE_BEGIN}\n"
        ".betterborg/state/\n"
        f"{MANAGED_IGNORE_END}\n"
    )
    paths.gitignore.write_text(prior, encoding="utf-8")
    original_replace = os.replace

    def interrupt(source: Path, destination: Path) -> None:
        if interrupt_after_replacement:
            original_replace(source, destination)
        raise KeyboardInterrupt("managed ignore publication interrupted")

    monkeypatch.setattr(repository_files_module.os, "replace", interrupt)

    with pytest.raises(
        KeyboardInterrupt,
        match="managed ignore publication interrupted",
    ):
        ensure_managed_gitignore(paths)

    expected = canonical if interrupt_after_replacement else prior
    assert paths.gitignore.read_text(encoding="utf-8") == expected
    assert list(git_repo.glob(".gitignore.*.tmp")) == []


def test_manages_accepts_only_paths_inside_the_worktrees_directory(
    git_repo: Path,
) -> None:
    paths = RepoPaths.discover(git_repo)

    assert paths.manages(paths.worktrees_dir / "planning" / "borg-run")
    # A sibling whose name merely begins with the worktrees directory's own
    # name is outside it, which a string-prefix test would get wrong.
    adjacent = paths.worktrees_dir.parent / f"{paths.worktrees_dir.name}-elsewhere"
    assert not paths.manages(adjacent)
    assert not paths.manages(paths.root)
    assert not paths.manages(git_repo.parent / "unrelated-checkout")


def test_manages_resolves_a_symlink_out_of_the_worktrees_directory(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    paths = RepoPaths.discover(git_repo)
    foreign = tmp_path / "foreign-checkout"
    foreign.mkdir()
    planted = paths.worktrees_dir / "planning"
    planted.mkdir(parents=True)
    link = planted / "looks-managed"
    link.symlink_to(foreign, target_is_directory=True)

    # Resolving first is what stops a symlink planted inside the worktrees
    # directory from borrowing the repository's trust.
    assert not paths.manages(link)
