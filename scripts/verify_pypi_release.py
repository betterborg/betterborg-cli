"""Verify an exact public release without retaining provider credentials."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

PROVIDER_VARIABLE = "OPENAI_API_KEY"
OTHER_PROVIDER_VARIABLE = "ANTHROPIC_API_KEY"


class ReleaseVerificationError(RuntimeError):
    """A public release did not satisfy the protected smoke contract."""


@dataclass(frozen=True)
class CommandCapture:
    label: str
    stdout: bytes
    stderr: bytes


def _fail(message: str) -> NoReturn:
    raise ReleaseVerificationError(message)


def _run(
    command: list[str],
    *,
    label: str,
    captures: list[CommandCapture],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
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


def _credential_markers(credential: str) -> tuple[bytes, ...]:
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


def _assert_no_credential_leak(
    credential: str,
    captures: list[CommandCapture],
    roots: tuple[Path, ...],
) -> None:
    markers = _credential_markers(credential)
    for capture in captures:
        for stream_name, content in (
            ("stdout", capture.stdout),
            ("stderr", capture.stderr),
        ):
            if any(marker in content for marker in markers):
                _fail(f"provider credential leaked in {capture.label} {stream_name}")

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            relative_path = path.relative_to(root)
            if any(marker in os.fsencode(relative_path) for marker in markers):
                _fail("provider credential leaked in release fixture path")
            if path.is_symlink():
                if any(marker in os.fsencode(os.readlink(path)) for marker in markers):
                    _fail("provider credential leaked in release fixture symlink")
                continue
            if not path.is_file():
                continue
            try:
                content = path.read_bytes()
            except OSError:
                _fail(f"could not inspect release fixture file {relative_path}")
            if any(marker in content for marker in markers):
                _fail(
                    "provider credential leaked in release fixture file "
                    f"{path.relative_to(root)}"
                )


def _uvx_command(version: str, *borg_arguments: str) -> list[str]:
    return [
        "uvx",
        "--refresh",
        "--from",
        f"betterborg=={version}",
        "borg",
        *borg_arguments,
    ]


def verify_release(
    version: str,
    fixture_root: Path,
    *,
    attempts: int = 12,
    retry_delay: float = 10.0,
) -> None:
    """Run version and init smoke checks against one exact PyPI version."""
    credential = os.environ.get(PROVIDER_VARIABLE, "")
    if len(credential) < 12:
        _fail(
            f"{PROVIDER_VARIABLE} is missing or too short for the protected smoke"
        )
    if attempts < 1:
        _fail("attempts must be at least one")

    fixture_root.mkdir(parents=True, exist_ok=False)
    captures: list[CommandCapture] = []
    readme = fixture_root / "README.md"
    readme.write_text("# BetterBorg release fixture\n", encoding="utf-8")
    safe_environment = dict(os.environ)
    safe_environment.pop(PROVIDER_VARIABLE, None)
    safe_environment.pop(OTHER_PROVIDER_VARIABLE, None)

    for command, label in (
        (["git", "init", "--initial-branch=main", str(fixture_root)], "git init"),
        (
            [
                "git",
                "-C",
                str(fixture_root),
                "config",
                "user.name",
                "Release Smoke",
            ],
            "git user name",
        ),
        (
            [
                "git",
                "-C",
                str(fixture_root),
                "config",
                "user.email",
                "release-smoke@betterborg.com",
            ],
            "git user email",
        ),
        (["git", "-C", str(fixture_root), "add", "README.md"], "git add"),
        (
            [
                "git",
                "-C",
                str(fixture_root),
                "commit",
                "-m",
                "Initialize fixture",
            ],
            "git commit",
        ),
    ):
        completed = _run(
            command,
            label=label,
            captures=captures,
            env=safe_environment,
        )
        if completed.returncode != 0:
            _assert_no_credential_leak(credential, captures, (fixture_root,))
            _fail(f"{label} failed with exit code {completed.returncode}")

    child_environment = dict(safe_environment)
    child_environment[PROVIDER_VARIABLE] = credential
    child_environment["XDG_STATE_HOME"] = str(fixture_root / ".release-state")
    child_environment["NO_COLOR"] = "1"

    expected_version_output = f"borg {version}".encode()
    version_completed: subprocess.CompletedProcess[bytes] | None = None
    for attempt in range(1, attempts + 1):
        version_completed = _run(
            _uvx_command(version, "version"),
            label=f"exact-version check attempt {attempt}",
            captures=captures,
            cwd=fixture_root,
            env=child_environment,
        )
        _assert_no_credential_leak(credential, captures, (fixture_root,))
        if (
            version_completed.returncode == 0
            and version_completed.stdout.strip() == expected_version_output
        ):
            break
        if attempt != attempts:
            time.sleep(retry_delay)
    else:
        _fail("exact-version uvx check did not observe the reviewed public release")

    initialized = _run(
        _uvx_command(version, "init", "--yes", "--json"),
        label="fixture repository init",
        captures=captures,
        cwd=fixture_root,
        env=child_environment,
    )
    _assert_no_credential_leak(credential, captures, (fixture_root,))
    if initialized.returncode != 0:
        _fail(f"fixture repository init failed with exit code {initialized.returncode}")
    try:
        payload = json.loads(initialized.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail("fixture repository init did not emit valid JSON")
    if not isinstance(payload, dict) or payload.get("initialized") is not True:
        _fail("fixture repository init did not report a new initialization")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="exact reviewed PyPI version")
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--retry-delay", type=float, default=10.0)
    arguments = parser.parse_args()
    try:
        with tempfile.TemporaryDirectory(prefix="betterborg-release-") as temporary:
            verify_release(
                arguments.version,
                Path(temporary) / "fixture",
                attempts=arguments.attempts,
                retry_delay=arguments.retry_delay,
            )
    except ReleaseVerificationError as error:
        print(f"release verification failed: {error}", file=os.sys.stderr)
        return 1
    print(f"verified betterborg {arguments.version} from PyPI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
