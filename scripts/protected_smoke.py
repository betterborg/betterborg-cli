"""Shared subprocess and credential-safety contract for protected release smokes."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

PROVIDER_VARIABLE = "OPENAI_API_KEY"
OTHER_PROVIDER_VARIABLE = "ANTHROPIC_API_KEY"


class ProtectedSmokeError(RuntimeError):
    """A protected release smoke violated its safety or behavior contract."""


@dataclass(frozen=True)
class CommandCapture:
    """Captured output from one protected-smoke subprocess."""

    label: str
    stdout: bytes
    stderr: bytes


Runner = Callable[..., subprocess.CompletedProcess[bytes]]


def fail(message: str) -> NoReturn:
    """Fail a protected smoke without exposing subprocess output."""
    raise ProtectedSmokeError(message)


def credential_markers(credential: str) -> tuple[bytes, ...]:
    """Return raw and encoded forms that must not survive a smoke."""
    raw = credential.encode()
    standard_base64 = base64.b64encode(raw)
    urlsafe_base64 = base64.urlsafe_b64encode(raw)
    candidates = {
        raw,
        standard_base64,
        standard_base64.rstrip(b"="),
        urlsafe_base64,
        urlsafe_base64.rstrip(b"="),
        urllib.parse.quote(credential, safe="").encode(),
    }
    return tuple(candidate for candidate in candidates if candidate)


def assert_no_credential_leak(
    credential: str,
    captures: list[CommandCapture],
    roots: tuple[Path, ...],
) -> None:
    """Reject credential representations in output or fixture state."""
    markers = credential_markers(credential)
    for capture in captures:
        for stream_name, content in (
            ("stdout", capture.stdout),
            ("stderr", capture.stderr),
        ):
            if any(marker in content for marker in markers):
                fail(f"provider credential leaked in {capture.label} {stream_name}")

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            relative_path = path.relative_to(root)
            if any(marker in os.fsencode(relative_path) for marker in markers):
                fail("provider credential leaked in fixture path")
            if path.is_symlink():
                if any(marker in os.fsencode(os.readlink(path)) for marker in markers):
                    fail("provider credential leaked in fixture symlink")
                continue
            if not path.is_file():
                continue
            try:
                content = path.read_bytes()
            except OSError:
                fail(f"could not inspect fixture file {relative_path}")
            if any(marker in content for marker in markers):
                fail(f"provider credential leaked in fixture file {relative_path}")


def run_command(
    runner: Runner,
    command: list[str],
    *,
    label: str,
    captures: list[CommandCapture],
    credential: str,
    roots: tuple[Path, ...],
    cwd: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[bytes]:
    """Run, capture, and immediately scan one protected-smoke command."""
    try:
        completed = runner(
            command,
            cwd=cwd,
            env=env,
            check=False,
            capture_output=True,
        )
    except OSError as error:
        fail(f"{label} could not start: {type(error).__name__}")
    captures.append(CommandCapture(label, completed.stdout, completed.stderr))
    assert_no_credential_leak(credential, captures, roots)
    return completed


def protected_environment() -> tuple[str, dict[str, str]]:
    """Return the provider credential and a copy of the environment without it."""
    credential = os.environ.get(PROVIDER_VARIABLE, "")
    if len(credential) < 12:
        fail(f"{PROVIDER_VARIABLE} is missing or too short for the protected smoke")
    environment = dict(os.environ)
    environment.pop(PROVIDER_VARIABLE, None)
    environment.pop(OTHER_PROVIDER_VARIABLE, None)
    return credential, environment


def initialize_git_fixture(
    fixture: Path,
    environment: dict[str, str],
    captures: list[CommandCapture],
    credential: str,
    runner: Runner,
) -> None:
    """Create the common fresh trusted repository used by release smokes."""
    fixture.mkdir()
    (fixture / "README.md").write_text(
        "# BetterBorg release fixture\n", encoding="utf-8"
    )
    commands = (
        ["git", "init", "--initial-branch=main", "."],
        ["git", "config", "user.name", "Release Smoke"],
        ["git", "config", "user.email", "release-smoke@betterborg.com"],
        ["git", "add", "README.md"],
        ["git", "commit", "-m", "Initialize fixture"],
    )
    for command in commands:
        label = " ".join(command[:2])
        completed = run_command(
            runner,
            command,
            label=label,
            captures=captures,
            credential=credential,
            roots=(fixture,),
            cwd=fixture,
            env=environment,
        )
        if completed.returncode != 0:
            fail(f"fixture setup failed: {label}")


def verify_cli_initialization(
    prefix: list[str],
    *,
    method: str,
    version: str,
    fixture: Path,
    environment: dict[str, str],
    captures: list[CommandCapture],
    credential: str,
    runner: Runner,
    attempts: int,
    retry_delay: float,
) -> None:
    """Check one exact CLI version and initialize one fresh provider fixture."""
    expected_version_output = f"borg {version}".encode()
    for attempt in range(1, attempts + 1):
        completed = run_command(
            runner,
            [*prefix, "version"],
            label=f"{method} exact-version check attempt {attempt}",
            captures=captures,
            credential=credential,
            roots=(fixture,),
            cwd=fixture,
            env=environment,
        )
        if (
            completed.returncode == 0
            and completed.stdout.strip() == expected_version_output
        ):
            break
        if attempt != attempts:
            time.sleep(retry_delay)
    else:
        fail(f"exact-version {method} check did not observe BetterBorg {version}")

    init_environment = dict(environment)
    init_environment[PROVIDER_VARIABLE] = credential
    initialized = run_command(
        runner,
        [*prefix, "init", "--yes", "--json"],
        label=f"{method} trusted provider initialization",
        captures=captures,
        credential=credential,
        roots=(fixture,),
        cwd=fixture,
        env=init_environment,
    )
    if initialized.returncode != 0:
        fail(f"{method} provider initialization failed")
    try:
        payload = json.loads(initialized.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError):
        fail(f"{method} provider initialization did not emit valid JSON")
    if not isinstance(payload, dict) or payload.get("initialized") is not True:
        fail(f"{method} provider initialization was not fresh")
