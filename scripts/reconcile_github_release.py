"""Digest-gate and reconcile an immutable Betterborg GitHub Release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from release_artifacts import (
    INSTALLER_FILENAME,
    TARGETS,
    ReleaseArtifactError,
    build_manifest,
    sha256,
)


class ReleaseReconciliationError(RuntimeError):
    """A GitHub Release cannot be safely reconciled."""


@dataclass(frozen=True)
class RemoteRelease:
    draft: bool
    assets: dict[str, str]


@dataclass(frozen=True)
class ReconciliationPlan:
    create_draft: bool
    upload: tuple[str, ...]
    publish_draft: bool


def _fail(message: str) -> NoReturn:
    raise ReleaseReconciliationError(message)


def expected_assets(version: str, directory: Path) -> dict[str, str]:
    """Validate the manifest and return every asset's local digest."""
    manifest_path = directory / "release-manifest.json"
    try:
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        generated = build_manifest(version, directory)
    except (OSError, json.JSONDecodeError, ReleaseArtifactError) as error:
        _fail(f"invalid release manifest: {error}")
    if recorded != generated:
        _fail("release-manifest.json does not match the reviewed binaries")

    filenames = [
        name
        for target in TARGETS
        for name in (target.filename, f"{target.filename}.sha256")
    ]
    filenames.extend((manifest_path.name, INSTALLER_FILENAME))
    try:
        return {name: sha256(directory / name) for name in filenames}
    except ReleaseArtifactError as error:
        _fail(str(error))


def plan_reconciliation(
    local: dict[str, str],
    remote: RemoteRelease | None,
    *,
    publish: bool,
) -> ReconciliationPlan:
    """Plan mutations only after every extant remote digest matches."""
    if remote is None:
        return ReconciliationPlan(True, tuple(local), publish)

    unexpected = remote.assets.keys() - local.keys()
    if unexpected:
        _fail(f"GitHub Release has unexpected assets: {sorted(unexpected)}")
    for filename, digest in remote.assets.items():
        if local[filename] != digest:
            _fail(
                f"GitHub Release digest mismatch for {filename}; "
                "the release is immutable, so prepare a new version"
            )

    missing = tuple(name for name in local if name not in remote.assets)
    if missing and not remote.draft:
        _fail("published GitHub Release is partial; prepare a new version")
    return ReconciliationPlan(False, missing, publish and remote.draft)


def _gh(command: list[str], *, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["gh", *command], check=False, capture_output=True, text=not binary
    )
    if completed.returncode != 0:
        stderr = completed.stderr
        detail = (
            stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
        ).strip()
        _fail(f"gh {' '.join(command[:2])} failed: {detail or 'unknown error'}")
    return completed.stdout


def _remote_release(repository: str, tag: str) -> RemoteRelease | None:
    completed = subprocess.run(
        ["gh", "api", f"repos/{repository}/releases/tags/{tag}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        try:
            error = json.loads(completed.stderr)
        except json.JSONDecodeError:
            error = {}
        if error.get("status") == "404" or "HTTP 404" in completed.stderr:
            return None
        _fail("could not inspect the existing GitHub Release")
    try:
        payload = json.loads(completed.stdout)
        draft = payload["draft"]
        assets = payload["assets"]
    except (KeyError, TypeError, json.JSONDecodeError):
        _fail("GitHub Release metadata is malformed")
    if not isinstance(draft, bool) or not isinstance(assets, list):
        _fail("GitHub Release metadata is malformed")

    digests: dict[str, str] = {}
    for asset in assets:
        try:
            name = asset["name"]
            url = asset["url"]
        except (KeyError, TypeError):
            _fail("GitHub Release asset metadata is malformed")
        if not isinstance(name, str) or not isinstance(url, str) or name in digests:
            _fail("GitHub Release asset metadata is malformed")
        body = _gh(
            ["api", url, "--header", "Accept: application/octet-stream"],
            binary=True,
        )
        assert isinstance(body, bytes)
        digests[name] = hashlib.sha256(body).hexdigest()
    return RemoteRelease(draft, digests)


def _fixture_release(path: Path) -> RemoteRelease | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail(f"could not read reconciliation fixture: {error}")
    if payload is None:
        return None
    try:
        draft = payload["draft"]
        assets = payload["assets"]
    except (KeyError, TypeError):
        _fail("reconciliation fixture is malformed")
    if not isinstance(draft, bool) or not isinstance(assets, dict) or not all(
        isinstance(name, str) and isinstance(digest, str)
        for name, digest in assets.items()
    ):
        _fail("reconciliation fixture is malformed")
    return RemoteRelease(draft, assets)


def reconcile(
    version: str,
    directory: Path,
    repository: str,
    *,
    publish: bool,
    fixture: Path | None,
) -> ReconciliationPlan:
    """Inspect, digest-gate, and optionally perform a release reconciliation."""
    local = expected_assets(version, directory)
    tag = f"v{version}"
    remote = _fixture_release(fixture) if fixture else _remote_release(repository, tag)
    plan = plan_reconciliation(local, remote, publish=publish)
    if not publish:
        return plan
    if fixture is not None:
        _fail("fixture reconciliation can never publish")

    if plan.create_draft:
        _gh(
            [
                "release",
                "create",
                tag,
                "--repo",
                repository,
                "--verify-tag",
                "--draft",
                "--title",
                f"Betterborg {version}",
            ]
        )
    for filename in plan.upload:
        _gh(
            [
                "release",
                "upload",
                tag,
                str(directory / filename),
                "--repo",
                repository,
            ]
        )
    if plan.publish_draft:
        _gh(
            [
                "release",
                "edit",
                tag,
                "--repo",
                repository,
                "--draft=false",
                "--latest",
            ]
        )
    return plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--fixture", type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if not arguments.repository and arguments.fixture is None:
        print(
            "release reconciliation failed: --repository is required",
            file=sys.stderr,
        )
        return 1
    try:
        plan = reconcile(
            arguments.version,
            arguments.artifacts,
            arguments.repository,
            publish=arguments.publish,
            fixture=arguments.fixture,
        )
    except ReleaseReconciliationError as error:
        print(f"release reconciliation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(plan.__dict__, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
