"""Smoke exact curl, uvx, and npx releases in isolated trusted repositories."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

PROVIDER_VARIABLE = "OPENAI_API_KEY"
OTHER_PROVIDER_VARIABLE = "ANTHROPIC_API_KEY"
INSTALL_URL = (
    "https://github.com/betterborg/betterborg-cli/"
    "releases/download/v{version}/install.sh"
)


class PublicInstallationError(RuntimeError):
    """A public installation path failed its protected smoke contract."""


@dataclass(frozen=True)
class CommandCapture:
    label: str
    stdout: bytes
    stderr: bytes


Runner = Callable[..., subprocess.CompletedProcess[bytes]]


def _fail(message: str) -> NoReturn:
    raise PublicInstallationError(message)


def command_shapes(version: str) -> dict[str, tuple[str, ...]]:
    """Return the exact-version public commands exercised by the smoke."""
    return {
        "curl": (
            "curl",
            "--proto",
            "=https",
            "--tlsv1.2",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            INSTALL_URL.format(version=version),
        ),
        "uvx": (
            "uvx",
            "--refresh",
            "--from",
            f"betterborg=={version}",
            "borg",
        ),
        "npx": ("npx", "--yes", f"@betterborg/cli@{version}"),
    }


def _credential_markers(credential: str) -> tuple[bytes, ...]:
    raw = credential.encode()
    standard = base64.b64encode(raw)
    urlsafe = base64.urlsafe_b64encode(raw)
    return tuple(
        marker
        for marker in {
            raw,
            standard,
            standard.rstrip(b"="),
            urlsafe,
            urlsafe.rstrip(b"="),
            urllib.parse.quote(credential, safe="").encode(),
        }
        if marker
    )


def _assert_no_credential_leak(
    credential: str, captures: list[CommandCapture], root: Path
) -> None:
    markers = _credential_markers(credential)
    for capture in captures:
        for stream_name, body in (
            ("stdout", capture.stdout),
            ("stderr", capture.stderr),
        ):
            if any(marker in body for marker in markers):
                _fail(f"provider credential leaked in {capture.label} {stream_name}")
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(marker in os.fsencode(relative) for marker in markers):
            _fail("provider credential leaked in fixture path")
        if path.is_symlink():
            if any(marker in os.fsencode(os.readlink(path)) for marker in markers):
                _fail("provider credential leaked in fixture symlink")
        elif path.is_file():
            try:
                body = path.read_bytes()
            except OSError:
                _fail(f"could not inspect public smoke fixture file {relative}")
            if any(marker in body for marker in markers):
                _fail(f"provider credential leaked in fixture file {relative}")


def _run(
    runner: Runner,
    command: list[str],
    *,
    label: str,
    captures: list[CommandCapture],
    cwd: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = runner(
            command,
            cwd=cwd,
            env=env,
            check=False,
            capture_output=True,
        )
    except OSError as error:
        _fail(f"{label} could not start: {type(error).__name__}")
    captures.append(CommandCapture(label, completed.stdout, completed.stderr))
    return completed


def _initialize_git(
    fixture: Path,
    environment: dict[str, str],
    captures: list[CommandCapture],
    runner: Runner,
) -> None:
    fixture.mkdir()
    (fixture / "README.md").write_text("# BetterBorg public smoke\n", encoding="utf-8")
    commands = (
        ["git", "init", "--initial-branch=main", "."],
        ["git", "config", "user.name", "Release Smoke"],
        ["git", "config", "user.email", "release-smoke@betterborg.com"],
        ["git", "add", "README.md"],
        ["git", "commit", "-m", "Initialize fixture"],
    )
    for command in commands:
        completed = _run(
            runner,
            command,
            label=" ".join(command[:2]),
            captures=captures,
            cwd=fixture,
            env=environment,
        )
        if completed.returncode != 0:
            _fail(f"fixture setup failed: {' '.join(command[:2])}")


def _commands_for_installation(
    method: str,
    version: str,
    fixture: Path,
    environment: dict[str, str],
    captures: list[CommandCapture],
    runner: Runner,
) -> tuple[list[str], dict[str, str]]:
    shapes = command_shapes(version)
    if method == "uvx":
        return list(shapes[method]), environment
    if method == "npx":
        return list(shapes[method]), environment

    installer = fixture / "install.sh"
    download = [*shapes["curl"], "--output", str(installer)]
    completed = _run(
        runner,
        download,
        label="exact-version curl installer download",
        captures=captures,
        cwd=fixture,
        env=environment,
    )
    if completed.returncode != 0:
        _fail("exact-version curl installer download failed")
    install_environment = dict(environment)
    install_environment["BETTERBORG_VERSION"] = version
    completed = _run(
        runner,
        ["sh", str(installer)],
        label="checksum-verifying curl installation",
        captures=captures,
        cwd=fixture,
        env=install_environment,
    )
    if completed.returncode != 0:
        _fail("checksum-verifying curl installation failed")
    return [str(fixture / "home/.local/bin/borg")], environment


def verify_installations(
    version: str,
    root: Path,
    *,
    attempts: int = 6,
    retry_delay: float = 10.0,
    runner: Runner = subprocess.run,
) -> None:
    """Run each public source in a fresh repository with isolated machine state."""
    if attempts < 1:
        _fail("attempts must be at least one")
    credential = os.environ.get(PROVIDER_VARIABLE, "")
    if len(credential) < 12:
        _fail(f"{PROVIDER_VARIABLE} is missing or too short for the protected smoke")
    root.mkdir(parents=True, exist_ok=False)
    base_environment = dict(os.environ)
    base_environment.pop(PROVIDER_VARIABLE, None)
    base_environment.pop(OTHER_PROVIDER_VARIABLE, None)

    for method in ("curl", "uvx", "npx"):
        fixture = root / method
        captures: list[CommandCapture] = []
        environment = dict(base_environment)
        environment.update(
            {
                "HOME": str(fixture / "home"),
                "NO_COLOR": "1",
                "XDG_CACHE_HOME": str(fixture / "cache"),
                "XDG_DATA_HOME": str(fixture / "data"),
                "XDG_STATE_HOME": str(fixture / "state"),
            }
        )
        _initialize_git(fixture, environment, captures, runner)
        prefix, command_environment = _commands_for_installation(
            method, version, fixture, environment, captures, runner
        )
        expected = f"borg {version}".encode()
        for attempt in range(1, attempts + 1):
            completed = _run(
                runner,
                [*prefix, "version"],
                label=f"{method} exact-version check attempt {attempt}",
                captures=captures,
                cwd=fixture,
                env=command_environment,
            )
            _assert_no_credential_leak(credential, captures, fixture)
            if completed.returncode == 0 and completed.stdout.strip() == expected:
                break
            if attempt != attempts:
                time.sleep(retry_delay)
        else:
            _fail(f"{method} did not run exact BetterBorg {version}")

        init_environment = dict(command_environment)
        init_environment[PROVIDER_VARIABLE] = credential
        initialized = _run(
            runner,
            [*prefix, "init", "--yes", "--json"],
            label=f"{method} trusted provider initialization",
            captures=captures,
            cwd=fixture,
            env=init_environment,
        )
        _assert_no_credential_leak(credential, captures, fixture)
        if initialized.returncode != 0:
            _fail(f"{method} provider initialization failed")
        try:
            payload = json.loads(initialized.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError):
            _fail(f"{method} provider initialization did not emit valid JSON")
        if not isinstance(payload, dict) or payload.get("initialized") is not True:
            _fail(f"{method} provider initialization was not fresh")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--retry-delay", type=float, default=10.0)
    arguments = parser.parse_args()
    try:
        with tempfile.TemporaryDirectory(
            prefix="betterborg-public-installations-"
        ) as temporary:
            verify_installations(
                arguments.version,
                Path(temporary) / "fixtures",
                attempts=arguments.attempts,
                retry_delay=arguments.retry_delay,
            )
    except PublicInstallationError as error:
        print(f"public installation verification failed: {error}", file=sys.stderr)
        return 1
    print(f"verified curl, uvx, and npx for BetterBorg {arguments.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
