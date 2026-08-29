"""Digest-gate one immutable npm package version without publishing it."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import NoReturn

PACKAGE_NAME = "@betterborg/cli"
REGISTRY_URL = "https://registry.npmjs.org/{package}/{version}"


class NpmReconciliationError(RuntimeError):
    """An npm version cannot be safely published or resumed."""


def _fail(message: str) -> NoReturn:
    raise NpmReconciliationError(message)


def package_integrity(tarball: Path) -> str:
    """Return the npm SHA-512 subresource-integrity value for a tarball."""
    try:
        body = tarball.read_bytes()
    except OSError as error:
        _fail(f"could not read reviewed npm package {tarball}: {error}")
    digest = base64.b64encode(hashlib.sha512(body).digest()).decode("ascii")
    return f"sha512-{digest}"


def _integrity_from_payload(payload: object, version: str) -> str:
    if not isinstance(payload, dict) or not isinstance(payload.get("dist"), dict):
        _fail("public npm release metadata is malformed")
    name = payload.get("name")
    public_version = payload.get("version")
    integrity = payload["dist"].get("integrity")
    if (
        name != PACKAGE_NAME
        or public_version != version
        or not isinstance(integrity, str)
        or not integrity.startswith("sha512-")
    ):
        _fail("public npm release metadata is malformed")
    return integrity


def _public_integrity(version: str) -> str | None:
    url = REGISTRY_URL.format(
        package=urllib.parse.quote(PACKAGE_NAME, safe=""),
        version=urllib.parse.quote(version, safe=""),
    )
    request = urllib.request.Request(
        url, headers={"Accept": "application/json"}, method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        _fail("could not retrieve public npm release metadata")
    except (OSError, ValueError):
        _fail("could not retrieve public npm release metadata")
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail("public npm release metadata is malformed")
    return _integrity_from_payload(payload, version)


def _fixture_integrity(path: Path, version: str) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail(f"could not read npm reconciliation fixture: {error}")
    if payload is None:
        return None
    return _integrity_from_payload(payload, version)


def publication_action(
    version: str, tarball: Path, *, fixture: Path | None = None
) -> str:
    """Return ``publish`` or ``skip`` after comparing immutable bytes."""
    reviewed = package_integrity(tarball)
    public = (
        _fixture_integrity(fixture, version)
        if fixture is not None
        else _public_integrity(version)
    )
    if public is None:
        return "publish"
    if public != reviewed:
        _fail(
            "public npm digest mismatch; the version is immutable, so prepare "
            "a new version"
        )
    return "skip"


def require_public_match(
    version: str,
    tarball: Path,
    *,
    attempts: int,
    retry_delay: float,
) -> None:
    if attempts < 1:
        _fail("attempts must be at least one")
    reviewed = package_integrity(tarball)
    for attempt in range(1, attempts + 1):
        public = _public_integrity(version)
        if public is not None:
            if public != reviewed:
                _fail(
                    "public npm digest mismatch; the version is immutable, so "
                    "prepare a new version"
                )
            return
        if attempt != attempts:
            time.sleep(retry_delay)
    _fail("reviewed npm version did not become publicly visible")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--tarball", required=True, type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--require-present", action="store_true")
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--retry-delay", type=float, default=10.0)
    parser.add_argument("--github-output", type=Path)
    arguments = parser.parse_args()
    try:
        if arguments.require_present:
            if arguments.fixture is not None:
                _fail("fixture reconciliation cannot wait for publication")
            require_public_match(
                arguments.version,
                arguments.tarball,
                attempts=arguments.attempts,
                retry_delay=arguments.retry_delay,
            )
            action = "skip"
        else:
            action = publication_action(
                arguments.version,
                arguments.tarball,
                fixture=arguments.fixture,
            )
        if arguments.github_output is not None:
            with arguments.github_output.open("a", encoding="utf-8") as output:
                output.write(f"action={action}\n")
    except NpmReconciliationError as error:
        print(f"npm reconciliation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"action": action}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
