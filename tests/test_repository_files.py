"""Repository-owned file publication contracts."""

import os
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch

from betterborg_cli import repository_files as repository_files_module
from betterborg_cli.agent_runtime import CancellationToken, run_captured
from betterborg_cli.repository_files import (
    RepositoryGitVisibilityError,
    RepositoryPathError,
    publish_repository_text,
    read_repository_text,
    require_git_trackable,
)


def test_git_trackability_normalizes_relative_and_absolute_paths(
    committed_git_repo: Path,
) -> None:
    relative = Path(".betterborg/plans/visible.md")
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
        ".betterborg/state/\n", encoding="utf-8"
    )
    ignored = committed_git_repo / ".betterborg/state/ignored.md"

    with pytest.raises(
        RepositoryGitVisibilityError,
        match=r"repository path is ignored by Git: \.betterborg/state/ignored\.md",
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


@pytest.mark.parametrize("interrupt_after_claim", [False, True])
def test_atomic_replacement_exposes_only_complete_destination_bytes(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    interrupt_after_claim: bool,
) -> None:
    destination = tmp_path / "managed.txt"
    destination.write_text("prior bytes\n", encoding="utf-8")
    original_replace = os.replace

    def interrupt(source: Path, target: Path) -> None:
        assert source.parent == destination.parent
        assert target == destination
        if interrupt_after_claim:
            original_replace(source, target)
        raise KeyboardInterrupt("publication interrupted")

    monkeypatch.setattr(repository_files_module.os, "replace", interrupt)

    with pytest.raises(KeyboardInterrupt, match="publication interrupted"):
        publish_repository_text(
            destination,
            "complete replacement\n",
            root=tmp_path,
            overwrite=True,
        )

    expected = "complete replacement\n" if interrupt_after_claim else "prior bytes\n"
    assert destination.read_text(encoding="utf-8") == expected
    assert list(tmp_path.glob(".managed.txt.*.tmp")) == []


@pytest.mark.parametrize("interrupt_after_claim", [False, True])
def test_atomic_creation_exposes_no_destination_or_complete_bytes(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    interrupt_after_claim: bool,
) -> None:
    destination = tmp_path / "config.toml"
    original_link = os.link

    def interrupt(source: Path, target: Path) -> None:
        assert source.parent == destination.parent
        assert target == destination
        if interrupt_after_claim:
            original_link(source, target)
        raise KeyboardInterrupt("publication interrupted")

    monkeypatch.setattr(repository_files_module.os, "link", interrupt)

    with pytest.raises(KeyboardInterrupt, match="publication interrupted"):
        publish_repository_text(
            destination,
            'version = 1\nname = "complete"\n',
            root=tmp_path,
            overwrite=False,
        )

    if interrupt_after_claim:
        assert destination.read_text(encoding="utf-8") == (
            'version = 1\nname = "complete"\n'
        )
    else:
        assert not destination.exists()
    assert list(tmp_path.glob(".config.toml.*.tmp")) == []


def test_atomic_creation_preserves_a_concurrent_winner_and_its_temporary_file(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    destination = tmp_path / "config.toml"
    unrelated_temporary = tmp_path / ".config.toml.concurrent.tmp"
    unrelated_temporary.write_text("owned by another initializer\n", encoding="utf-8")
    original_link = os.link

    def publish_winner_then_lose(source: Path, target: Path) -> None:
        winner_temporary = tmp_path / ".winner.tmp"
        winner_temporary.write_text("complete winner\n", encoding="utf-8")
        original_link(winner_temporary, target)
        winner_temporary.unlink()
        original_link(source, target)

    monkeypatch.setattr(
        repository_files_module.os,
        "link",
        publish_winner_then_lose,
    )

    with pytest.raises(FileExistsError):
        publish_repository_text(
            destination,
            "losing contents\n",
            root=tmp_path,
            overwrite=False,
        )

    assert destination.read_text(encoding="utf-8") == "complete winner\n"
    assert unrelated_temporary.read_text(encoding="utf-8") == (
        "owned by another initializer\n"
    )
    assert list(tmp_path.glob(".config.toml.*.tmp")) == [unrelated_temporary]


def test_atomic_publication_rejects_a_symlinked_parent_outside_the_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RepositoryPathError, match="escapes repository"):
        publish_repository_text(
            Path("linked/config.toml"),
            "complete config\n",
            root=root,
            overwrite=False,
        )

    assert list(outside.iterdir()) == []
