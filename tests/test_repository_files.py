"""Repository-owned file publication contracts."""

from pathlib import Path

import pytest

from betterborg_cli.repository_files import (
    RepositoryGitVisibilityError,
    RepositoryPathError,
    require_git_trackable,
)


def test_git_trackability_normalizes_relative_and_absolute_paths(
    committed_git_repo: Path,
) -> None:
    relative = Path(".borg/plans/visible.md")
    absolute = committed_git_repo / relative

    require_git_trackable(relative, root=committed_git_repo)
    require_git_trackable(absolute, root=committed_git_repo)


def test_git_trackability_reports_ignored_repository_path(
    committed_git_repo: Path,
) -> None:
    (committed_git_repo / ".gitignore").write_text(
        ".borg/state/\n", encoding="utf-8"
    )
    ignored = committed_git_repo / ".borg/state/ignored.md"

    with pytest.raises(
        RepositoryGitVisibilityError,
        match=r"repository path is ignored by Git: \.borg/state/ignored\.md",
    ):
        require_git_trackable(ignored, root=committed_git_repo)


def test_git_trackability_rejects_path_outside_repository(
    committed_git_repo: Path,
) -> None:
    outside = committed_git_repo.parent / "outside.md"

    with pytest.raises(RepositoryPathError, match="repository path escapes root"):
        require_git_trackable(outside, root=committed_git_repo)
