"""Create and validate checksums and the standalone binary release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Target:
    filename: str
    operating_system: str
    architecture: str


TARGETS = (
    Target("borg-darwin-arm64", "darwin", "arm64"),
    Target("borg-darwin-x86_64", "darwin", "x86_64"),
    Target("borg-linux-arm64", "linux", "arm64"),
    Target("borg-linux-x86_64", "linux", "x86_64"),
)


class ReleaseArtifactError(RuntimeError):
    """A release artifact set does not satisfy the public contract."""


def _fail(message: str) -> NoReturn:
    raise ReleaseArtifactError(message)


def sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        _fail(f"could not read release artifact {path}: {error}")
    return digest.hexdigest()


def write_checksum(path: Path) -> Path:
    """Write the conventional SHA-256 sidecar for one artifact."""
    if not path.is_file():
        _fail(f"release artifact is missing: {path}")
    checksum = path.with_name(f"{path.name}.sha256")
    checksum.write_text(f"{sha256(path)}  {path.name}\n", encoding="utf-8")
    return checksum


def _read_checksum(path: Path, artifact: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        _fail(f"could not read checksum {path}: {error}")
    expected = f"{sha256(artifact)}  {artifact.name}\n"
    if content != expected:
        _fail(f"checksum sidecar does not match {artifact.name}")
    return expected.split(" ", 1)[0]


def build_manifest(version: str, directory: Path) -> dict[str, object]:
    """Validate exactly four binaries and return the stable manifest shape."""
    if not version or version.startswith("v"):
        _fail("version must be nonempty and must not include the v tag prefix")

    expected_names = {
        name
        for target in TARGETS
        for name in (target.filename, f"{target.filename}.sha256")
    }
    actual_names = {path.name for path in directory.iterdir() if path.is_file()}
    unexpected = actual_names - expected_names - {"release-manifest.json"}
    missing = expected_names - actual_names
    if missing or unexpected:
        _fail(
            "binary artifact set differs: "
            f"missing {sorted(missing)}, unexpected {sorted(unexpected)}"
        )

    artifacts: list[dict[str, object]] = []
    for target in TARGETS:
        artifact = directory / target.filename
        digest = _read_checksum(
            directory / f"{target.filename}.sha256", artifact
        )
        artifacts.append(
            {
                "filename": target.filename,
                "os": target.operating_system,
                "arch": target.architecture,
                "sha256": digest,
                "size": artifact.stat().st_size,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "version": version,
        "artifacts": artifacts,
    }


def write_manifest(version: str, directory: Path, output: Path) -> None:
    manifest = build_manifest(version, directory)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    checksum = subparsers.add_parser("checksum")
    checksum.add_argument("artifact", type=Path)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--version", required=True)
    manifest.add_argument("--directory", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        if arguments.command == "checksum":
            write_checksum(arguments.artifact)
        else:
            write_manifest(arguments.version, arguments.directory, arguments.output)
    except ReleaseArtifactError as error:
        print(f"release artifact validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
