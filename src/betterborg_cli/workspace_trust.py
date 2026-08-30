"""Machine-local trust decisions for host-accessing workspace operations."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from betterborg_cli.agent_runtime.base import CancellationToken
from betterborg_cli.agent_runtime.process import run_captured
from betterborg_cli.repo_paths import RepoPaths

_TRUST_FORMAT_VERSION = 1
_TRUST_DIRECTORY = "betterborg"
_TRUST_FILENAME = "trusted-workspaces.json"


class UntrustedWorkspaceError(RuntimeError):
    """Raised when a workspace operation lacks machine-local trust."""


@dataclass(frozen=True)
class WorkspaceIdentity:
    """Paths and stable fingerprint that identify one Git workspace."""

    repository_path: Path
    git_common_dir: Path
    fingerprint: str

    @classmethod
    def discover(
        cls,
        paths: RepoPaths,
        *,
        cancel: CancellationToken | None = None,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] = run_captured,
    ) -> WorkspaceIdentity:
        """Resolve the repository's Git storage and derive its fingerprint."""
        command = [
            "git",
            "-C",
            str(paths.root),
            "rev-parse",
            "--git-common-dir",
        ]
        try:
            result = command_runner(
                command,
                check=True,
                cancel=cancel,
            )
        except subprocess.CalledProcessError as error:
            raise ValueError(
                f"cannot resolve Git common directory for {paths.root}"
            ) from error
        if result.returncode != 0:
            error = subprocess.CalledProcessError(
                result.returncode,
                command,
                output=result.stdout,
                stderr=result.stderr,
            )
            raise ValueError(
                f"cannot resolve Git common directory for {paths.root}"
            ) from error

        common_dir = Path(result.stdout.strip())
        if not common_dir.is_absolute():
            common_dir = paths.root / common_dir
        repository_path = paths.root.resolve()
        common_dir = common_dir.resolve()
        fingerprint_input = json.dumps(
            {
                "git_common_dir": str(common_dir),
                "repository_path": str(repository_path),
                "version": _TRUST_FORMAT_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        fingerprint = hashlib.sha256(fingerprint_input).hexdigest()
        return cls(
            repository_path=repository_path,
            git_common_dir=common_dir,
            fingerprint=fingerprint,
        )


class TrustStore:
    """Owner-only, machine-local persistence for workspace trust."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else self.default_path()

    @staticmethod
    def default_path() -> Path:
        """Return the trust file beneath the user's local state directory."""
        state_home = os.environ.get("XDG_STATE_HOME")
        root = Path(state_home) if state_home else Path.home() / ".local" / "state"
        return root.expanduser() / _TRUST_DIRECTORY / _TRUST_FILENAME

    def is_trusted(self, identity: WorkspaceIdentity) -> bool:
        """Return whether the exact workspace identity was previously trusted."""
        workspaces = self._read()["workspaces"]
        return identity.fingerprint in workspaces

    def trust(self, identity: WorkspaceIdentity) -> None:
        """Persist trust for one exact workspace identity."""
        document = self._read()
        document["workspaces"][identity.fingerprint] = {
            "git_common_dir": str(identity.git_common_dir),
            "repository_path": str(identity.repository_path),
        }
        self._write(document)

    def _read(self) -> dict[str, Any]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"version": _TRUST_FORMAT_VERSION, "workspaces": {}}

        self._require_owner_only(self.path, 0o600)
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid workspace trust file: {self.path}") from error
        if (
            not isinstance(document, dict)
            or document.get("version") != _TRUST_FORMAT_VERSION
            or not isinstance(document.get("workspaces"), dict)
        ):
            raise RuntimeError(f"unsupported workspace trust file: {self.path}")
        return document

    def _write(self, document: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.path.parent.chmod(0o700)
        temporary_path: Path | None = None
        try:
            descriptor, name = tempfile.mkstemp(
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
            )
            temporary_path = Path(name)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as trust_file:
                json.dump(document, trust_file, indent=2, sort_keys=True)
                trust_file.write("\n")
                trust_file.flush()
                os.fsync(trust_file.fileno())
            temporary_path.replace(self.path)
            self.path.chmod(0o600)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _require_owner_only(path: Path, expected_mode: int) -> None:
        details = path.stat()
        mode = stat.S_IMODE(details.st_mode)
        if mode != expected_mode:
            raise RuntimeError(
                f"workspace trust file must have mode {expected_mode:o}: {path}"
            )
        if hasattr(os, "getuid") and details.st_uid != os.getuid():
            raise RuntimeError(
                f"workspace trust file is not owned by this user: {path}"
            )


def require_workspace_trust(
    paths: RepoPaths,
    *,
    store: TrustStore | None = None,
    explicit: bool = False,
    interactive: bool = False,
    confirm: Callable[[str], bool] | None = None,
    cancel: CancellationToken | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = run_captured,
) -> WorkspaceIdentity:
    """Require trust before a caller loads repository-controlled context."""
    identity = WorkspaceIdentity.discover(
        paths,
        cancel=cancel,
        command_runner=command_runner,
    )
    trust_store = store or TrustStore()
    if trust_store.path.resolve().is_relative_to(identity.repository_path):
        raise RuntimeError(
            "workspace trust must be stored outside the repository: "
            f"{trust_store.path}"
        )
    if trust_store.is_trusted(identity):
        return identity
    if explicit:
        trust_store.trust(identity)
        return identity
    if not interactive or confirm is None:
        raise UntrustedWorkspaceError(
            f"workspace is not trusted on this machine: {paths.root}. "
            "Run 'borg trust --yes' to trust it explicitly."
        )

    consequence = (
        f"Trust workspace {identity.repository_path}? BetterBorg's host-capable "
        "agents may read and modify files and execute commands on this machine."
    )
    if not confirm(consequence):
        raise UntrustedWorkspaceError(
            f"workspace was not trusted: {identity.repository_path}"
        )
    trust_store.trust(identity)
    return identity
