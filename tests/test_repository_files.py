"""Repository-owned file publication contracts."""

import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from betterborg_cli.agent_runtime import CancellationToken, run_captured
from betterborg_cli.repository_files import (
    RepositoryGitVisibilityError,
    RepositoryPathError,
    read_repository_text,
    require_git_trackable,
)


def test_git_trackability_normalizes_relative_and_absolute_paths(
    committed_git_repo: Path,
) -> None:
    relative = Path(".borg/plans/visible.md")
    absolute = committed_git_repo / relative

    require_git_trackable(relative, root=committed_git_repo)
    require_git_trackable(absolute, root=committed_git_repo)


def test_git_trackability_forwards_cancellation_to_registered_runner(
    committed_git_repo: Path,
) -> None:
    cancel = CancellationToken()
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 1, "", "")

    require_git_trackable(
        Path("visible.md"),
        root=committed_git_repo,
        cancel=cancel,
        command_runner=runner,
    )

    assert calls == [
        (
            [
                "git",
                "-C",
                str(committed_git_repo),
                "check-ignore",
                "--quiet",
                "--",
                "visible.md",
            ],
            {"check": False, "cancel": cancel},
        )
    ]


def test_git_visibility_cancellation_reaps_tree_before_later_mutation(
    committed_git_repo: Path,
    real_process_harness: Any,
) -> None:
    cancel = CancellationToken(grace_seconds=0.05)
    errors: list[BaseException] = []
    mutation = committed_git_repo / "mutation-started"

    def runner(_command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return run_captured(
            real_process_harness.resistant_argv("git-visibility"),
            cancel=kwargs["cancel"],
            check=kwargs["check"],
        )

    def verify_then_mutate() -> None:
        try:
            require_git_trackable(
                Path("visible.md"),
                root=committed_git_repo,
                cancel=cancel,
                command_runner=runner,
            )
            mutation.write_text("started\n", encoding="utf-8")
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=verify_then_mutate)
    worker.start()
    real_process_harness.wait_for_marker("git-visibility.parent.pid")
    real_process_harness.wait_for_marker("git-visibility.child.pid")
    cancel.cancel()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RepositoryGitVisibilityError)
    assert not mutation.exists()
    real_process_harness.assert_tree_absent("git-visibility")


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


def test_repository_text_rejects_symlink_to_file_outside_repository(
    committed_git_repo: Path,
) -> None:
    outside = committed_git_repo.parent / "host-secret.md"
    outside.write_text("host secret\n", encoding="utf-8")
    linked = committed_git_repo / "linked.md"
    linked.symlink_to(outside)

    with pytest.raises(RepositoryPathError, match="not a regular file"):
        read_repository_text(linked, root=committed_git_repo)
