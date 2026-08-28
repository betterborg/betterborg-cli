"""Filesystem contracts for repository-local BetterBorg data."""

import subprocess
from pathlib import Path

from betterborg_cli.repo_paths import (
    MANAGED_IGNORE_BEGIN,
    MANAGED_IGNORE_END,
    RepoPaths,
    ensure_managed_gitignore,
)


def test_discover_uses_nearest_git_root(git_repo: Path) -> None:
    nested_repo = git_repo / "packages" / "nearest"
    child = nested_repo / "src" / "package"
    child.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", str(nested_repo)], check=True)

    paths = RepoPaths.discover(child)

    assert paths.root == nested_repo
    assert paths.tracked_dir == nested_repo / ".borg"
    assert paths.state_dir == nested_repo / ".borg" / "state"
    assert paths.artifacts_dir == nested_repo / ".borg" / "state" / "artifacts"


def test_managed_ignore_keeps_documents_trackable_and_ignores_state(
    git_repo: Path,
) -> None:
    paths = RepoPaths.discover(git_repo)
    ensure_managed_gitignore(paths)
    document = paths.tracked_dir / "project.md"
    state = paths.state_dir / "borg.sqlite3"
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

    assert "?? .borg/project.md" in status.stdout
    assert ".borg/state" not in status.stdout
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
    assert first_update.count(".borg/state/\n") == 1
