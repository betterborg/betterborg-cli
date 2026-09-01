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
from uuid import uuid4

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


def redact(content: bytes, credential: str) -> str:
    """Return decoded output with every credential representation removed."""
    for marker in credential_markers(credential):
        content = content.replace(marker, b"[REDACTED]")
    return content.decode("utf-8", errors="replace").strip()


def fail_with_output(
    message: str, capture: CommandCapture, credential: str
) -> NoReturn:
    """Fail a protected smoke, reporting one command's scrubbed output.

    Every capture has already passed ``assert_no_credential_leak`` before a
    caller can reach this point, so the credential provably does not appear in
    it; redacting again here is defence in depth. An opaque smoke failure is
    undiagnosable from a CI log, which costs more than this discloses.
    """
    details = [
        f"{stream}: {text}"
        for stream, text in (
            ("stdout", redact(capture.stdout, credential)),
            ("stderr", redact(capture.stderr, credential)),
        )
        if text
    ]
    if not details:
        fail(message)
    joined = "\n".join(details)
    fail(f"{message}\n{joined}")


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
    repository: Path,
    environment: dict[str, str],
    captures: list[CommandCapture],
    credential: str,
    runner: Runner,
) -> None:
    """Create the common fresh repository used by release smokes.

    The repository is a directory of its own so that a smoke's machine state —
    home, caches, and the workspace trust store — can sit beside it rather than
    inside it. The CLI refuses to record trust for a repository within that
    repository, so a fixture that nests them cannot be initialized at all.
    """
    repository.mkdir(parents=True)
    (repository / "README.md").write_text(
        "# Betterborg release fixture\n", encoding="utf-8"
    )
    tracked_directory = repository / ".betterborg"
    tracked_directory.mkdir()
    (tracked_directory / "config.toml").write_text(
        "version = 1\n\n"
        "[repository]\n"
        f'id = "{uuid4()}"\n'
        'default_branch = "main"\n\n'
        "[agents.analysis]\n"
        'adapter = "openai"\n'
        'model = "gpt-5.6-luna"\n'
        'effort = "low"\n',
        encoding="utf-8",
    )
    commands = (
        ["git", "init", "--initial-branch=main", "."],
        ["git", "config", "user.name", "Release Smoke"],
        ["git", "config", "user.email", "release-smoke@betterborg.com"],
        ["git", "add", "README.md", ".betterborg/config.toml"],
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
            roots=(repository,),
            cwd=repository,
            env=environment,
        )
        if completed.returncode != 0:
            fail_with_output(
                f"fixture setup failed: {label}", captures[-1], credential
            )


def verify_cli_initialization(
    prefix: list[str],
    *,
    method: str,
    version: str,
    repository: Path,
    scan_root: Path,
    environment: dict[str, str],
    captures: list[CommandCapture],
    credential: str,
    runner: Runner,
    attempts: int,
    retry_delay: float,
) -> None:
    """Check one exact CLI version and initialize one fresh provider fixture.

    The CLI runs inside ``repository``; ``scan_root`` is the wider fixture tree
    the credential scan walks, so machine state written outside the repository
    is still checked for leaks.
    """
    expected_version_output = f"betterborg {version}".encode()
    for attempt in range(1, attempts + 1):
        completed = run_command(
            runner,
            [*prefix, "version"],
            label=f"{method} exact-version check attempt {attempt}",
            captures=captures,
            credential=credential,
            roots=(scan_root,),
            cwd=repository,
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
        fail_with_output(
            f"exact-version {method} check did not observe Betterborg {version}",
            captures[-1],
            credential,
        )

    init_environment = dict(environment)
    init_environment[PROVIDER_VARIABLE] = credential
    initialized = run_command(
        runner,
        [*prefix, "init", "--yes", "--json"],
        label=f"{method} trusted provider initialization",
        captures=captures,
        credential=credential,
        roots=(scan_root,),
        cwd=repository,
        env=init_environment,
    )
    if initialized.returncode != 0:
        fail_with_output(
            f"{method} provider initialization failed", captures[-1], credential
        )
    try:
        payload = json.loads(initialized.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError):
        fail_with_output(
            f"{method} provider initialization did not emit valid JSON",
            captures[-1],
            credential,
        )
    if not isinstance(payload, dict) or payload.get("initialized") is not True:
        fail_with_output(
            f"{method} provider initialization was not fresh",
            captures[-1],
            credential,
        )
