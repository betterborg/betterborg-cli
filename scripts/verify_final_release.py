"""Verify one synchronized PyPI, GitHub, and npm release without publishing."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

try:
    from scripts import (
        reconcile_npm_release,
        verify_github_release,
        verify_public_installations,
        verify_pypi_release,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
    import reconcile_npm_release
    import verify_github_release
    import verify_public_installations
    import verify_pypi_release

REPOSITORY = "betterborg/betterborg-cli"


class FinalReleaseVerificationError(RuntimeError):
    """The synchronized release is partial, invalid, or unsafe to resume."""


@dataclass(frozen=True)
class VerificationResult:
    """Read-only result for all three ordered publication surfaces."""

    complete: bool
    remaining: tuple[str, ...]


def _fail(message: str) -> NoReturn:
    raise FinalReleaseVerificationError(message)


def _pypi_status(
    version: str,
    registry_artifacts: Path,
    fixture: Path | None,
) -> str:
    pypi_fixture = fixture / "pypi.json" if fixture is not None else None
    try:
        return verify_pypi_release.publication_action(
            version,
            registry_artifacts,
            fixture=pypi_fixture,
        )
    except verify_pypi_release.ReleaseVerificationError as error:
        _fail(f"PyPI verification failed: {error}")


def _npm_status(
    version: str,
    registry_artifacts: Path,
    fixture: Path | None,
) -> str:
    npm_fixture = fixture / "npm.json" if fixture is not None else None
    tarball = registry_artifacts / f"betterborg-cli-{version}.tgz"
    try:
        return reconcile_npm_release.publication_action(
            version,
            tarball,
            fixture=npm_fixture,
        )
    except reconcile_npm_release.NpmReconciliationError as error:
        _fail(f"npm verification failed: {error}")


def _npm_present(version: str, fixture: Path | None) -> bool:
    npm_fixture = fixture / "npm.json" if fixture is not None else None
    try:
        return reconcile_npm_release.version_exists(version, fixture=npm_fixture)
    except reconcile_npm_release.NpmReconciliationError as error:
        _fail(f"npm verification failed: {error}")


def _github_snapshot(
    version: str,
    repository: str,
    reviewed_sha: str | None,
    fixture: Path | None,
    download_directory: Path,
) -> verify_github_release.ReleaseSnapshot | None:
    try:
        if fixture is not None:
            return verify_github_release.fixture_snapshot(fixture / "github")
        if reviewed_sha is None:
            _fail("--reviewed-sha is required for live verification")
        download_directory.mkdir()
        return verify_github_release.download_snapshot(
            repository,
            version,
            reviewed_sha,
            download_directory,
        )
    except verify_github_release.GitHubReleaseVerificationError as error:
        _fail(f"GitHub verification failed: {error}")


def _surface_result(
    version: str,
    registry_artifacts: Path,
    github_artifacts: Path | None,
    repository: str,
    reviewed_sha: str | None,
    fixture: Path | None,
    download_directory: Path,
) -> VerificationResult:
    pypi = _pypi_status(version, registry_artifacts, fixture)
    snapshot = _github_snapshot(
        version,
        repository,
        reviewed_sha,
        fixture,
        download_directory,
    )
    github_started = snapshot is not None
    if pypi == "publish":
        npm_present = _npm_present(version, fixture)
        if github_started or npm_present:
            _fail(
                "publication order violation: GitHub or npm exists before the "
                "reviewed PyPI version"
            )
        return VerificationResult(
            False,
            (
                "publish the reviewed PyPI wheel and source distribution",
                "verify both public PyPI SHA-256 digests",
            ),
        )

    if snapshot is not None and github_artifacts is None:
        _fail(
            "--github-artifacts is required once GitHub publication has started; "
            f"download binary-release-{version} from the reviewed workflow run"
        )
    try:
        if github_artifacts is not None:
            verify_github_release.compare_reviewed_assets(snapshot, github_artifacts)
        github = verify_github_release.verify_snapshot(version, snapshot)
    except verify_github_release.GitHubReleaseVerificationError as error:
        _fail(f"GitHub verification failed: {error}")

    if not github.complete:
        npm_present = _npm_present(version, fixture)
        if npm_present:
            _fail(
                "publication order violation: npm exists before the GitHub "
                "Release is complete"
            )
        return VerificationResult(False, github.remaining)

    npm = _npm_status(version, registry_artifacts, fixture)
    if npm == "publish":
        return VerificationResult(
            False,
            (
                "publish the reviewed npm tarball",
                "verify the public npm SHA-512 integrity",
            ),
        )
    return VerificationResult(True, ())


class _FixtureRunner:
    """Exercise exact smoke commands locally without invoking package clients."""

    _git_commands = (
        ["git", "init", "--initial-branch=main", "."],
        ["git", "config", "user.name", "Release Smoke"],
        ["git", "config", "user.email", "release-smoke@betterborg.com"],
        ["git", "add", "README.md"],
        ["git", "commit", "-m", "Initialize fixture"],
    )

    def __init__(self, version: str, credential: str, leak: str | None) -> None:
        self.version = version
        self.credential = credential
        self.leak = leak
        self.git_indexes = {method: 0 for method in ("curl", "uvx", "npx")}
        self.smoke_indexes = {method: 0 for method in ("curl", "uvx", "npx")}

    def _smoke_commands(self, method: str, fixture: Path) -> tuple[list[str], ...]:
        shapes = verify_public_installations.command_shapes(self.version)
        if method == "curl":
            installer = fixture / "install.sh"
            prefix = [str(fixture / "home/.local/bin/betterborg")]
            return (
                [*shapes["curl"], "--output", str(installer)],
                ["sh", str(installer)],
                [*prefix, "version"],
                [*prefix, "init", "--yes", "--json"],
            )
        prefix = list(shapes[method])
        return (
            [*prefix, "version"],
            [*prefix, "init", "--yes", "--json"],
        )

    def __call__(
        self, command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        directory = Path(str(kwargs["cwd"]))
        # Repository commands run inside the fixture; installation commands run
        # beside it, so that machine state never lands within the repository.
        fixture = directory.parent if directory.name == "repo" else directory
        method = fixture.name
        if method not in self.git_indexes:
            return subprocess.CompletedProcess(command, 1, b"", b"unknown fixture")

        if command[:1] == ["git"]:
            index = self.git_indexes[method]
            if index >= len(self._git_commands) or command != self._git_commands[index]:
                return subprocess.CompletedProcess(
                    command, 1, b"", b"unexpected git command"
                )
            self.git_indexes[method] += 1
            return subprocess.CompletedProcess(command, 0, b"", b"")

        expected = self._smoke_commands(method, fixture)
        index = self.smoke_indexes[method]
        if index >= len(expected) or command != expected[index]:
            return subprocess.CompletedProcess(
                command, 1, b"", b"unexpected smoke command"
            )
        self.smoke_indexes[method] += 1
        if method == "curl" and index == 0:
            Path(command[-1]).write_text("#!/bin/sh\n", encoding="utf-8")

        stdout = b""
        stderr = b""
        if command[-1:] == ["version"]:
            stdout = f"betterborg {self.version}\n".encode()
        elif command[-3:] == ["init", "--yes", "--json"]:
            stdout = b'{"initialized":true}\n'
            if self.leak == "stdout":
                stdout = self.credential.encode()
            elif self.leak == "stderr":
                stderr = self.credential.encode()
            elif self.leak == "file":
                (fixture / "leaked-state").write_text(
                    self.credential, encoding="utf-8"
                )
        return subprocess.CompletedProcess(command, 0, stdout, stderr)

    def assert_complete(self) -> None:
        if set(self.git_indexes.values()) != {len(self._git_commands)}:
            _fail(
                "fixture did not exercise all three isolated Git "
                "initialization paths"
            )
        expected = {"curl": 4, "uvx": 2, "npx": 2}
        if self.smoke_indexes != expected:
            _fail("fixture did not exercise every exact-version public smoke command")


def _fixture_smoke(path: Path) -> tuple[str, str | None]:
    try:
        payload = json.loads((path / "smoke.json").read_text(encoding="utf-8"))
        credential = payload["credential"]
        leak = payload.get("leak")
    except (
        OSError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        _fail(f"could not read protected-smoke fixture: {error}")
    if not isinstance(credential, str) or len(credential) < 12:
        _fail("protected-smoke fixture credential is missing or too short")
    if leak not in {None, "stdout", "stderr", "file"}:
        _fail("protected-smoke fixture leak must be stdout, stderr, file, or null")
    return credential, leak


def verify_final_release(
    version: str,
    registry_artifacts: Path,
    github_artifacts: Path | None = None,
    *,
    repository: str = REPOSITORY,
    reviewed_sha: str | None = None,
    fixture: Path | None = None,
    attempts: int = 6,
    retry_delay: float = 10.0,
) -> VerificationResult:
    """Verify ordered public bytes, then smoke all exact installation paths."""
    with tempfile.TemporaryDirectory(prefix="betterborg-final-release-") as temporary:
        temporary_root = Path(temporary)
        result = _surface_result(
            version,
            registry_artifacts,
            github_artifacts,
            repository,
            reviewed_sha,
            fixture,
            temporary_root / "github",
        )
        if not result.complete:
            return result

        runner = subprocess.run
        protected_context = None
        fixture_runner = None
        if fixture is not None:
            credential, leak = _fixture_smoke(fixture)
            fixture_runner = _FixtureRunner(version, credential, leak)
            runner = fixture_runner
            protected_context = (credential, {})
        try:
            verify_public_installations.verify_installations(
                version,
                temporary_root / "installations",
                attempts=attempts,
                retry_delay=retry_delay,
                runner=runner,
                protected_context=protected_context,
            )
        except verify_public_installations.PublicInstallationError as error:
            _fail(f"public installation verification failed: {error}")
        if fixture_runner is not None:
            fixture_runner.assert_complete()
        return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--registry-artifacts", required=True, type=Path)
    parser.add_argument(
        "--github-artifacts",
        type=Path,
        help=(
            "reviewed binary-release artifact directory; required after GitHub "
            "publication starts"
        ),
    )
    parser.add_argument("--repository", default=REPOSITORY)
    parser.add_argument("--reviewed-sha")
    parser.add_argument(
        "--fixture",
        type=Path,
        help="local pypi.json, github/, npm.json, and smoke.json fixture root",
    )
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--retry-delay", type=float, default=10.0)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        result = verify_final_release(
            arguments.version,
            arguments.registry_artifacts,
            arguments.github_artifacts,
            repository=arguments.repository,
            reviewed_sha=arguments.reviewed_sha,
            fixture=arguments.fixture,
            attempts=arguments.attempts,
            retry_delay=arguments.retry_delay,
        )
    except FinalReleaseVerificationError as error:
        print(f"final release verification failed: {error}", file=sys.stderr)
        return 1
    if not result.complete:
        print(f"Betterborg {arguments.version} publication is partial. Next steps:")
        for step in result.remaining:
            print(f"- {step}")
        return 2
    print(
        f"verified synchronized Betterborg {arguments.version} across PyPI, "
        "GitHub, npm, curl, uvx, and npx"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
