"""Verify a published Betterborg binary GitHub Release without mutating it."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from release_artifacts import (
    INSTALLER_FILENAME,
    SCHEMA_VERSION,
    TARGETS,
    ReleaseArtifactError,
    build_manifest,
    sha256,
    validate_installer,
)


class GitHubReleaseVerificationError(RuntimeError):
    """The public release cannot be accepted or safely resumed."""


@dataclass(frozen=True)
class ReleaseSnapshot:
    tag: str
    draft: bool
    assets: dict[str, Path]
    verified_attestations: frozenset[str]
    published_attestations: frozenset[str]


@dataclass(frozen=True)
class VerificationResult:
    complete: bool
    remaining: tuple[str, ...]


def _fail(message: str) -> NoReturn:
    raise GitHubReleaseVerificationError(message)


def _expected_names() -> tuple[str, ...]:
    names = [
        name
        for target in TARGETS
        for name in (target.filename, f"{target.filename}.sha256")
    ]
    names.extend(("release-manifest.json", INSTALLER_FILENAME))
    return tuple(names)


def _mismatch(detail: str) -> NoReturn:
    _fail(
        f"release asset digest mismatch ({detail}); the release is immutable, "
        "so prepare a new version"
    )


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        _mismatch("release-manifest.json has malformed UTF-8 encoding")
    except (OSError, json.JSONDecodeError) as error:
        _fail(
            "release-manifest.json is invalid; the release is immutable, "
            f"so prepare a new version: {error}"
        )
    if not isinstance(recorded, dict):
        _mismatch("release-manifest.json is not an object")
    return recorded


def _read_checksum_digest(path: Path, filename: str) -> str:
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        _mismatch(f"checksum sidecar for {filename} is malformed")
    except OSError as error:
        _fail(f"could not read release checksum {path}: {error}")
    suffix = f"  {filename}\n"
    digest = content[:64]
    if (
        len(content) != 64 + len(suffix)
        or content[64:] != suffix
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        _mismatch(f"checksum sidecar for {filename} is malformed")
    return digest


def _validate_available_assets(
    version: str,
    assets: dict[str, Path],
) -> dict[str, object] | None:
    """Reject every mismatch observable in a complete or partial draft."""
    manifest_path = assets.get("release-manifest.json")
    recorded = _read_manifest(manifest_path) if manifest_path else None
    installer_path = assets.get(INSTALLER_FILENAME)
    if installer_path is not None:
        try:
            validate_installer(installer_path)
        except ReleaseArtifactError as error:
            _mismatch(str(error))
    entries: list[object] | None = None
    if recorded is not None:
        if set(recorded) != {"schema_version", "version", "artifacts"}:
            _mismatch("release-manifest.json has unexpected fields")
        if recorded["schema_version"] != SCHEMA_VERSION:
            _mismatch("release-manifest.json has the wrong schema version")
        if recorded["version"] != version:
            _mismatch("release-manifest.json has the wrong release version")
        entries = recorded["artifacts"]
        if not isinstance(entries, list) or len(entries) != len(TARGETS):
            _mismatch("release-manifest.json has the wrong target set")

    for index, target in enumerate(TARGETS):
        binary = assets.get(target.filename)
        checksum = assets.get(f"{target.filename}.sha256")
        binary_digest = sha256(binary) if binary else None
        binary_size = binary.stat().st_size if binary else None
        manifest_digest = None

        if entries is not None:
            entry = entries[index]
            if not isinstance(entry, dict) or set(entry) != {
                "filename",
                "os",
                "arch",
                "sha256",
                "size",
            }:
                _mismatch("release-manifest.json has malformed target metadata")
            if (
                entry["filename"] != target.filename
                or entry["os"] != target.operating_system
                or entry["arch"] != target.architecture
            ):
                _mismatch("release-manifest.json target metadata differs")
            manifest_digest = entry["sha256"]
            size = entry["size"]
            if (
                not isinstance(manifest_digest, str)
                or len(manifest_digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in manifest_digest
                )
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
            ):
                _mismatch("release-manifest.json has malformed digest metadata")
            if binary_digest is not None and (
                manifest_digest != binary_digest or size != binary_size
            ):
                _mismatch(f"release-manifest.json does not match {target.filename}")

        if checksum is not None:
            checksum_digest = _read_checksum_digest(checksum, target.filename)
            expected_digest = binary_digest or manifest_digest
            if expected_digest is not None and checksum_digest != expected_digest:
                _mismatch(f"checksum sidecar does not match {target.filename}")
    return recorded


def _validate_complete_assets(version: str, assets: dict[str, Path]) -> None:
    recorded = _validate_available_assets(version, assets)
    assert recorded is not None
    manifest_path = assets["release-manifest.json"]

    directory = manifest_path.parent
    try:
        generated = build_manifest(version, directory)
    except ReleaseArtifactError as error:
        _fail(
            "release asset digest mismatch; the release is immutable, "
            f"so prepare a new version: {error}"
        )
    if recorded != generated:
        _fail(
            "release-manifest.json digest or metadata mismatch; the release is "
            "immutable, so prepare a new version"
        )


def verify_snapshot(
    version: str,
    snapshot: ReleaseSnapshot | None,
) -> VerificationResult:
    """Validate a downloaded snapshot and describe safe publication work left."""
    expected = _expected_names()
    expected_set = set(expected)
    if snapshot is None:
        remaining = ["create the draft GitHub Release"]
        remaining.extend(f"upload release asset {name}" for name in expected)
        remaining.extend(
            f"publish GitHub artifact attestation for {name}" for name in expected
        )
        remaining.append("publish the draft GitHub Release")
        return VerificationResult(False, tuple(remaining))

    expected_tag = f"v{version}"
    if snapshot.tag != expected_tag:
        _fail(
            f"release tag mismatch: expected {expected_tag}, found {snapshot.tag}; "
            "stop and review the selected version"
        )

    unexpected = snapshot.assets.keys() - expected_set
    if unexpected:
        _fail(
            "GitHub Release has unexpected immutable assets: "
            f"{sorted(unexpected)}; prepare a new version"
        )

    missing = tuple(name for name in expected if name not in snapshot.assets)
    if missing:
        _validate_available_assets(version, snapshot.assets)
    if missing and not snapshot.draft:
        _fail(
            "published GitHub Release is partial and immutable; missing assets "
            f"{list(missing)}; prepare a new version"
        )
    if not missing:
        _validate_complete_assets(version, snapshot.assets)

    unexpected_attestations = (
        snapshot.verified_attestations | snapshot.published_attestations
    ) - expected_set
    if unexpected_attestations:
        _fail(
            "verification fixture has attestations for unexpected assets: "
            f"{sorted(unexpected_attestations)}"
        )
    missing_attestations = tuple(
        name for name in expected if name not in snapshot.published_attestations
    )
    unverified_attestations = tuple(
        name
        for name in expected
        if name not in snapshot.verified_attestations
    )

    remaining = [f"upload release asset {name}" for name in missing]
    remaining.extend(
        f"publish GitHub artifact attestation for {name}"
        for name in missing_attestations
    )
    remaining.extend(
        f"verify GitHub artifact attestation for {name}"
        for name in unverified_attestations
    )
    if snapshot.draft:
        remaining.append("publish the draft GitHub Release")
    return VerificationResult(not remaining, tuple(remaining))


def compare_reviewed_assets(
    snapshot: ReleaseSnapshot | None,
    reviewed_directory: Path,
) -> None:
    """Compare every available public asset with the reviewed build-once bytes."""
    expected = _expected_names()
    try:
        reviewed = {name: sha256(reviewed_directory / name) for name in expected}
    except ReleaseArtifactError as error:
        _fail(f"reviewed GitHub artifact set is incomplete: {error}")
    unexpected = {
        path.name
        for path in reviewed_directory.iterdir()
        if path.is_file() and path.name not in expected
    }
    if unexpected:
        _fail(
            "reviewed GitHub artifact set has unexpected files: "
            f"{sorted(unexpected)}"
        )
    if snapshot is None:
        return
    for name, public_path in snapshot.assets.items():
        if name not in reviewed or sha256(public_path) != reviewed[name]:
            _mismatch(f"public {name} does not match the reviewed build-once asset")


def _run(command: list[str], *, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=not binary,
    )
    if completed.returncode != 0:
        stderr = completed.stderr
        detail = (
            stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
        ).strip()
        _fail(f"{' '.join(command[:3])} failed: {detail or 'unknown error'}")
    return completed.stdout


def _release_metadata(repository: str, tag: str) -> dict[str, object] | None:
    output = _run(
        [
            "gh",
            "api",
            f"repos/{repository}/releases?per_page=100",
            "--paginate",
            "--slurp",
        ]
    )
    assert isinstance(output, str)
    try:
        pages = json.loads(output)
    except json.JSONDecodeError:
        _fail("GitHub Release metadata is malformed")
    if not isinstance(pages, list) or not all(
        isinstance(page, list) for page in pages
    ):
        _fail("GitHub Release metadata is malformed")

    matches: list[dict[str, object]] = []
    for page in pages:
        for release in page:
            if not isinstance(release, dict):
                _fail("GitHub Release metadata is malformed")
            if release.get("tag_name") == tag:
                matches.append(release)
    if len(matches) > 1:
        _fail(f"multiple GitHub Releases use tag {tag}")
    return matches[0] if matches else None


def _source_digest(repository: str, tag: str) -> str:
    output = _run(["gh", "api", f"repos/{repository}/commits/{tag}"])
    assert isinstance(output, str)
    try:
        digest = json.loads(output)["sha"]
    except (KeyError, TypeError, json.JSONDecodeError):
        _fail("reviewed release tag metadata is malformed")
    if (
        not isinstance(digest, str)
        or len(digest) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        _fail("reviewed release tag metadata is malformed")
    return digest


def _has_attestation(repository: str, digest: str) -> bool:
    output = _run(
        ["gh", "api", f"repos/{repository}/attestations/sha256:{digest}"]
    )
    assert isinstance(output, str)
    try:
        payload = json.loads(output)
        attestations = payload["attestations"]
    except (KeyError, TypeError, json.JSONDecodeError):
        _fail("GitHub artifact attestation metadata is malformed")
    if not isinstance(attestations, list):
        _fail("GitHub artifact attestation metadata is malformed")
    return bool(attestations)


def _verify_attestation(
    repository: str,
    path: Path,
    source_digest: str,
) -> None:
    output = _run(
        [
            "gh",
            "attestation",
            "verify",
            str(path),
            "--repo",
            repository,
            "--signer-workflow",
            f"{repository}/.github/workflows/binary-release.yml",
            "--source-digest",
            source_digest,
            "--source-ref",
            "refs/heads/main",
        ]
    )
    assert isinstance(output, str)


def _download_snapshot(
    repository: str,
    version: str,
    reviewed_sha: str,
    directory: Path,
) -> ReleaseSnapshot | None:
    tag = f"v{version}"
    source_digest = _source_digest(repository, tag)
    if source_digest != reviewed_sha:
        _fail(
            f"reviewed release tag moved: expected {tag} at {reviewed_sha}, "
            f"found {source_digest}; the release cannot be accepted"
        )
    payload = _release_metadata(repository, tag)
    if payload is None:
        return None
    try:
        remote_tag = payload["tag_name"]
        draft = payload["draft"]
        remote_assets = payload["assets"]
    except KeyError:
        _fail("GitHub Release metadata is malformed")
    if (
        not isinstance(remote_tag, str)
        or not isinstance(draft, bool)
        or not isinstance(remote_assets, list)
    ):
        _fail("GitHub Release metadata is malformed")

    assets: dict[str, Path] = {}
    for metadata in remote_assets:
        try:
            name = metadata["name"]
            url = metadata["url"]
        except (KeyError, TypeError):
            _fail("GitHub Release asset metadata is malformed")
        if (
            not isinstance(name, str)
            or not isinstance(url, str)
            or Path(name).name != name
            or name in assets
        ):
            _fail("GitHub Release asset metadata is malformed")
        if name not in _expected_names():
            _fail(
                f"GitHub Release has unexpected immutable asset {name}; "
                "prepare a new version"
            )
        body = _run(
            ["gh", "api", url, "--header", "Accept: application/octet-stream"],
            binary=True,
        )
        assert isinstance(body, bytes)
        path = directory / name
        path.write_bytes(body)
        assets[name] = path

    verified_attestations: set[str] = set()
    published_attestations: set[str] = set()
    _validate_available_assets(version, assets)
    # GitHub CLI needs the subject bytes to verify the signature and provenance.
    # Do not credit API records inferred for assets that have not been downloaded.
    for name, path in assets.items():
        digest = sha256(path)
        if _has_attestation(repository, digest):
            try:
                _verify_attestation(repository, path, source_digest)
            except GitHubReleaseVerificationError as error:
                _fail(
                    f"attestation digest or provenance mismatch for {name}; "
                    "the release cannot be accepted: "
                    f"{error}"
                )
            published_attestations.add(name)
            verified_attestations.add(name)

    return ReleaseSnapshot(
        remote_tag,
        draft,
        assets,
        frozenset(verified_attestations),
        frozenset(published_attestations),
    )


def _fixture_snapshot(path: Path) -> ReleaseSnapshot | None:
    metadata_path = path / "release.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail(f"could not read release fixture: {error}")
    if payload is None:
        return None
    try:
        tag = payload["tag_name"]
        draft = payload["draft"]
        attestations = payload["attestations"]
    except (KeyError, TypeError):
        _fail("release fixture metadata is malformed")
    if (
        not isinstance(tag, str)
        or not isinstance(draft, bool)
        or not isinstance(attestations, list)
        or not all(isinstance(name, str) for name in attestations)
        or len(attestations) != len(set(attestations))
    ):
        _fail("release fixture metadata is malformed")
    assets_directory = path / "assets"
    try:
        assets = {
            asset.name: asset
            for asset in assets_directory.iterdir()
            if asset.is_file()
        }
    except OSError as error:
        _fail(f"could not read release fixture assets: {error}")
    fixture_attestations = frozenset(attestations)
    return ReleaseSnapshot(
        tag,
        draft,
        assets,
        fixture_attestations,
        fixture_attestations,
    )


def verify_release(
    version: str,
    repository: str,
    *,
    reviewed_sha: str | None = None,
    fixture: Path | None = None,
) -> VerificationResult:
    """Verify one public release, or an entirely local fixture snapshot."""
    if not version or version.startswith("v"):
        _fail("version must be nonempty and must not include the v tag prefix")
    if fixture is not None:
        return verify_snapshot(version, _fixture_snapshot(fixture))
    if not repository:
        _fail("--repository is required")
    if (
        reviewed_sha is None
        or len(reviewed_sha) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in reviewed_sha)
    ):
        _fail("--reviewed-sha must be the full lowercase reviewed commit SHA")
    with tempfile.TemporaryDirectory(prefix="betterborg-release-") as temporary:
        snapshot = _download_snapshot(
            repository,
            version,
            reviewed_sha,
            Path(temporary),
        )
        return verify_snapshot(version, snapshot)


def fixture_snapshot(path: Path) -> ReleaseSnapshot | None:
    """Load a local release fixture for cross-surface verification."""
    return _fixture_snapshot(path)


def download_snapshot(
    repository: str,
    version: str,
    reviewed_sha: str,
    directory: Path,
) -> ReleaseSnapshot | None:
    """Download a read-only public snapshot into an operator-owned temporary path."""
    if (
        len(reviewed_sha) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in reviewed_sha)
    ):
        _fail("--reviewed-sha must be the full lowercase reviewed commit SHA")
    return _download_snapshot(repository, version, reviewed_sha, directory)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--repository", default="betterborg/betterborg-cli")
    parser.add_argument(
        "--reviewed-sha",
        help="full commit SHA recorded when vVERSION was reviewed (required live)",
    )
    parser.add_argument("--fixture", type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        result = verify_release(
            arguments.version,
            arguments.repository,
            reviewed_sha=arguments.reviewed_sha,
            fixture=arguments.fixture,
        )
    except GitHubReleaseVerificationError as error:
        print(f"GitHub Release verification failed: {error}", file=sys.stderr)
        return 1
    if result.complete:
        print(
            f"Verified complete immutable GitHub Release v{arguments.version}: "
            f"{len(_expected_names())} assets and attestations"
        )
        return 0
    print(f"GitHub Release v{arguments.version} is partial. Remaining steps:")
    for step in result.remaining:
        print(f"- {step}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
