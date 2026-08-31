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
    MANAGED_IGNORE_BEGIN,
    MANAGED_IGNORE_END,
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
