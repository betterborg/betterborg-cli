"""Contracts for machine-local workspace trust."""

import json
import stat
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from betterborg_cli.agent_runtime import CancellationToken, run_captured
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.workspace_trust import (
    TrustStore,
    UntrustedWorkspaceError,
    WorkspaceIdentity,
    require_workspace_trust,
)


def _trust_path_outside(repository: Path) -> Path:
    return repository.parent / f"{repository.name}-machine-state" / "trust.json"


def test_fingerprint_is_stable_and_changes_for_a_different_repository_path(
    git_repo: Path,
) -> None:
    paths = RepoPaths.discover(git_repo)
    first = WorkspaceIdentity.discover(paths)

    assert WorkspaceIdentity.discover(paths) == first

    moved_repo = git_repo.parent / f"{git_repo.name}-moved"
    subprocess.run(
        ["git", "clone", "--quiet", str(git_repo), str(moved_repo)], check=True
    )
    moved = WorkspaceIdentity.discover(RepoPaths.discover(moved_repo))

    assert moved.repository_path != first.repository_path
    assert moved.git_common_dir != first.git_common_dir
    assert moved.fingerprint != first.fingerprint


def test_identity_and_trust_forward_one_token_and_runner(git_repo: Path) -> None:
    paths = RepoPaths.discover(git_repo)
    cancel = CancellationToken()
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, ".git\n", "")

    direct = WorkspaceIdentity.discover(
        paths,
        cancel=cancel,
        command_runner=runner,
    )
    trusted = require_workspace_trust(
        paths,
        store=TrustStore(_trust_path_outside(git_repo)),
        explicit=True,
        cancel=cancel,
        command_runner=runner,
    )

    assert direct == trusted
    assert [kwargs for _command, kwargs in calls] == [
        {"check": True, "cancel": cancel},
        {"check": True, "cancel": cancel},
    ]


def test_trust_cancellation_reaps_identity_git_process_tree(
    git_repo: Path,
    real_process_harness: Any,
) -> None:
    paths = RepoPaths.discover(git_repo)
    cancel = CancellationToken(grace_seconds=0.05)
    errors: list[BaseException] = []

    def runner(_command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return run_captured(
            real_process_harness.resistant_argv("trust-identity"),
            cancel=kwargs["cancel"],
            check=kwargs["check"],
        )

    def trust() -> None:
        try:
            require_workspace_trust(
                paths,
                store=TrustStore(_trust_path_outside(git_repo)),
                explicit=True,
                cancel=cancel,
                command_runner=runner,
            )
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=trust)
    worker.start()
    real_process_harness.wait_for_marker("trust-identity.parent.pid")
    real_process_harness.wait_for_marker("trust-identity.child.pid")
    cancel.cancel()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert not TrustStore(_trust_path_outside(git_repo)).path.exists()
    real_process_harness.assert_tree_absent("trust-identity")


def test_changed_git_common_dir_invalidates_existing_trust(git_repo: Path) -> None:
    identity = WorkspaceIdentity.discover(RepoPaths.discover(git_repo))
    store = TrustStore(_trust_path_outside(git_repo))
    store.trust(identity)

    relocated_git_dir = git_repo.parent / f"{git_repo.name}-git-dir"
    (git_repo / ".git").rename(relocated_git_dir)
    (git_repo / ".git").write_text(f"gitdir: {relocated_git_dir}\n", encoding="utf-8")
    changed = WorkspaceIdentity.discover(RepoPaths.discover(git_repo))

    assert changed.repository_path == identity.repository_path
    assert changed.git_common_dir != identity.git_common_dir
    assert changed.fingerprint != identity.fingerprint
    assert store.is_trusted(identity)
    assert not store.is_trusted(changed)


def test_explicit_trust_is_persisted_outside_repository_with_owner_permissions(
    git_repo: Path,
) -> None:
    paths = RepoPaths.discover(git_repo)
    trust_path = _trust_path_outside(git_repo)
    store = TrustStore(trust_path)

    identity = require_workspace_trust(paths, store=store, explicit=True)

    assert trust_path.exists()
    assert not trust_path.is_relative_to(git_repo)
    assert stat.S_IMODE(trust_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(trust_path.parent.stat().st_mode) == 0o700
    document = json.loads(trust_path.read_text(encoding="utf-8"))
    assert document["workspaces"][identity.fingerprint] == {
        "git_common_dir": str(identity.git_common_dir),
        "repository_path": str(identity.repository_path),
    }


def test_interactive_trust_describes_host_access_and_persists(
    git_repo: Path,
) -> None:
    prompts: list[str] = []
    store = TrustStore(_trust_path_outside(git_repo))

    identity = require_workspace_trust(
        RepoPaths.discover(git_repo),
        store=store,
        interactive=True,
        confirm=lambda prompt: prompts.append(prompt) or True,
    )

    assert store.is_trusted(identity)
    assert len(prompts) == 1
    assert "read and modify files" in prompts[0]
    assert "execute commands on this machine" in prompts[0]


def test_rejected_interactive_trust_is_not_persisted(git_repo: Path) -> None:
    store = TrustStore(_trust_path_outside(git_repo))

    with pytest.raises(UntrustedWorkspaceError, match="was not trusted"):
        require_workspace_trust(
            RepoPaths.discover(git_repo),
            store=store,
            interactive=True,
            confirm=lambda _prompt: False,
        )

    assert not store.path.exists()


def test_noninteractive_rejection_happens_before_repository_context_load(
    git_repo: Path,
) -> None:
    config = git_repo / ".betterborg" / "config.json"
    prompt = git_repo / ".betterborg" / "agent-prompt.md"
    config.parent.mkdir()
    config.write_text('{"unsafe": true}\n', encoding="utf-8")
    prompt.write_text("repository-controlled prompt\n", encoding="utf-8")
    loaded: list[Path] = []

    def load_repository_context() -> None:
        for candidate in (config, prompt):
            candidate.read_text(encoding="utf-8")
            loaded.append(candidate)

    with pytest.raises(UntrustedWorkspaceError, match="not trusted"):
        require_workspace_trust(
            RepoPaths.discover(git_repo),
            store=TrustStore(_trust_path_outside(git_repo)),
        )
        load_repository_context()

    assert loaded == []


def test_insecure_existing_trust_file_fails_closed(git_repo: Path) -> None:
    store = TrustStore(_trust_path_outside(git_repo))
    identity = WorkspaceIdentity.discover(RepoPaths.discover(git_repo))
    store.trust(identity)
    store.path.chmod(0o644)

    with pytest.raises(RuntimeError, match="must have mode 600"):
        store.is_trusted(identity)


def test_trust_path_inside_repository_is_rejected(git_repo: Path) -> None:
    with pytest.raises(RuntimeError, match="stored outside the repository"):
        require_workspace_trust(
            RepoPaths.discover(git_repo),
            store=TrustStore(git_repo / ".betterborg" / "state" / "trust.json"),
            explicit=True,
        )
