"""Verify an exact public release without retaining provider credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import NoReturn

try:
    from scripts import protected_smoke
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
    import protected_smoke

PYPI_RELEASE_URL = "https://pypi.org/pypi/betterborg/{version}/json"


ReleaseVerificationError = protected_smoke.ProtectedSmokeError


def _fail(message: str) -> NoReturn:
    protected_smoke.fail(message)


def _uvx_command(version: str, *borg_arguments: str) -> list[str]:
    return [
        "uvx",
        "--refresh",
        "--from",
        f"betterborg=={version}",
        "borg",
        *borg_arguments,
    ]


def _expected_distribution_names(version: str) -> set[str]:
    return {
        f"betterborg-{version}-py3-none-any.whl",
        f"betterborg-{version}.tar.gz",
    }


def _local_distribution_digests(
    version: str, artifact_directory: Path
) -> dict[str, str]:
    digests: dict[str, str] = {}
    for filename in sorted(_expected_distribution_names(version)):
        path = artifact_directory / filename
        try:
            body = path.read_bytes()
        except OSError:
            _fail(f"reviewed distribution is missing or unreadable: {filename}")
        digests[filename] = hashlib.sha256(body).hexdigest()
    return digests


def _distribution_digests_from_payload(payload: object) -> dict[str, str] | None:
    if payload is None:
        return None
    try:
        files = payload["urls"]
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        _fail("public PyPI release metadata is malformed")
    if not isinstance(files, list):
        _fail("public PyPI release metadata is malformed")

    digests: dict[str, str] = {}
    for item in files:
        if not isinstance(item, dict):
            _fail("public PyPI release metadata is malformed")
        filename = item.get("filename")
        item_digests = item.get("digests")
        digest = item_digests.get("sha256") if isinstance(item_digests, dict) else None
        if not isinstance(filename, str) or not isinstance(digest, str):
            _fail("public PyPI release metadata is malformed")
        normalized_digest = digest.casefold()
        valid_digest = len(normalized_digest) == 64 and all(
            character in "0123456789abcdef" for character in normalized_digest
        )
        if not valid_digest or filename in digests:
            _fail("public PyPI release metadata is malformed")
        digests[filename] = normalized_digest
    return digests


def _public_distribution_digests(version: str) -> dict[str, str] | None:
    quoted_version = urllib.parse.quote(version, safe="")
    request = urllib.request.Request(
        PYPI_RELEASE_URL.format(version=quoted_version),
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        _fail("could not retrieve public PyPI release metadata")
    except (OSError, ValueError):
        _fail("could not retrieve public PyPI release metadata")

    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail("public PyPI release metadata is malformed")
    return _distribution_digests_from_payload(payload)


def _fixture_distribution_digests(path: Path) -> dict[str, str] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"could not read PyPI verification fixture: {error}")
    return _distribution_digests_from_payload(payload)


def _compare_public_digests(
    reviewed: dict[str, str], public: dict[str, str]
) -> None:
    if public.keys() != reviewed.keys():
        _fail(
            "public PyPI artifact names do not match the reviewed distributions; "
            "the version is immutable, so prepare a new version"
        )
    for filename, reviewed_digest in reviewed.items():
        if public[filename] != reviewed_digest:
            _fail(
                f"public PyPI digest mismatch for {filename}; "
                "the version is immutable, so prepare a new version"
            )


def _verify_public_digests(version: str, artifact_directory: Path) -> None:
    reviewed = _local_distribution_digests(version, artifact_directory)
    public = _public_distribution_digests(version)
    if public is None:
        _fail("reviewed PyPI version is not public")
    _compare_public_digests(reviewed, public)


def publication_action(
    version: str,
    artifact_directory: Path,
    *,
    fixture: Path | None = None,
) -> str:
    """Return ``publish`` or ``skip`` after comparing immutable PyPI bytes."""
    reviewed = _local_distribution_digests(version, artifact_directory)
    public = (
        _fixture_distribution_digests(fixture)
        if fixture is not None
        else _public_distribution_digests(version)
    )
    if public is None:
        return "publish"
    _compare_public_digests(reviewed, public)
    return "skip"


def verify_public_artifacts(
    version: str,
    artifact_directory: Path,
    *,
    attempts: int,
    retry_delay: float,
) -> None:
    """Wait for one reviewed wheel and sdist to become publicly visible."""
    if attempts < 1:
        _fail("attempts must be at least one")
    reviewed = _local_distribution_digests(version, artifact_directory)
    for attempt in range(1, attempts + 1):
        public = _public_distribution_digests(version)
        if public is not None:
            _compare_public_digests(reviewed, public)
            return
        if attempt != attempts:
            time.sleep(retry_delay)
    _fail("reviewed PyPI version did not become publicly visible")


def verify_release(
    version: str,
    fixture_root: Path,
    artifact_directory: Path = Path("dist"),
    *,
    attempts: int = 12,
    retry_delay: float = 10.0,
) -> None:
    """Compare artifacts and run smoke checks against one exact PyPI version."""
    if attempts < 1:
        _fail("attempts must be at least one")
    _verify_public_digests(version, artifact_directory)

    credential, safe_environment = protected_smoke.protected_environment()
    captures: list[protected_smoke.CommandCapture] = []
    protected_smoke.initialize_git_fixture(
        fixture_root, safe_environment, captures, credential, subprocess.run
    )

    child_environment = dict(safe_environment)
    child_environment["XDG_STATE_HOME"] = str(fixture_root / ".release-state")
    child_environment["NO_COLOR"] = "1"

    protected_smoke.verify_cli_initialization(
        _uvx_command(version),
        method="uvx",
        version=version,
        fixture=fixture_root,
        environment=child_environment,
        captures=captures,
        credential=credential,
        runner=subprocess.run,
        attempts=attempts,
        retry_delay=retry_delay,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="exact reviewed PyPI version")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("dist"),
        help="directory containing the reviewed wheel and source distribution",
    )
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--retry-delay", type=float, default=10.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-publication", action="store_true")
    mode.add_argument("--artifacts-only", action="store_true")
    parser.add_argument("--github-output", type=Path)
    arguments = parser.parse_args()
    try:
        if arguments.plan_publication:
            action = publication_action(arguments.version, arguments.artifacts)
            if arguments.github_output is not None:
                with arguments.github_output.open("a", encoding="utf-8") as output:
                    output.write(f"action={action}\n")
            print(json.dumps({"action": action}, sort_keys=True))
            return 0
        if arguments.artifacts_only:
            verify_public_artifacts(
                arguments.version,
                arguments.artifacts,
                attempts=arguments.attempts,
                retry_delay=arguments.retry_delay,
            )
        else:
            with tempfile.TemporaryDirectory(
                prefix="betterborg-release-"
            ) as temporary:
                verify_release(
                    arguments.version,
                    Path(temporary) / "fixture",
                    arguments.artifacts,
                    attempts=arguments.attempts,
                    retry_delay=arguments.retry_delay,
                )
    except ReleaseVerificationError as error:
        print(f"release verification failed: {error}", file=sys.stderr)
        return 1
    print(f"verified betterborg {arguments.version} from PyPI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
