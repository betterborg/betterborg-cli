"""Shared command, marketplace, and bundle support for plugin installers."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

_OWNER_FILE = ".betterborg-owned.json"
_OWNER_SCHEMA = 1

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class PluginCommandError(RuntimeError):
    """A supported host plugin command could not be completed."""


class PluginCollisionError(RuntimeError):
    """An existing marketplace or bundle is not owned by BetterBorg."""


@dataclass(frozen=True, slots=True)
class BundleChange:
    """Materialized bundle state needed for host activation and rollback."""

    path: Path
    digest: str
    version: str
    changed: bool
    previous: Path | None = None
    previous_version: str | None = None
    created_parents: tuple[Path, ...] = ()


def run_plugin_command(
    command: tuple[str, ...],
    environment: Mapping[str, str],
    runner: CommandRunner,
) -> subprocess.CompletedProcess[str]:
    """Run one supported host command with consistent diagnostics."""

    try:
        completed = runner(
            list(command),
            capture_output=True,
            check=False,
            env=dict(environment),
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PluginCommandError(f"unable to run {command[0]!r}: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise PluginCommandError(
            f"`{' '.join(command)}` failed with exit code "
            f"{completed.returncode}: {detail}"
        )
    return completed


def json_plugin_command(
    command: tuple[str, ...],
    environment: Mapping[str, str],
    runner: CommandRunner,
) -> Any:
    """Run one supported host command and decode its JSON response."""

    completed = run_plugin_command(command, environment, runner)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise PluginCommandError(
            f"`{' '.join(command)}` returned invalid JSON: {error}"
        ) from error


def plugin_data_root(
    environment: Mapping[str, str], explicit: Path | None
) -> Path:
    """Resolve the stable per-user data root shared by host installers."""

    if explicit is not None:
        return Path(explicit).expanduser().resolve(strict=False)
    if environment.get("XDG_DATA_HOME"):
        return Path(environment["XDG_DATA_HOME"]).expanduser().resolve(strict=False)
    home = environment.get("HOME") or environment.get("USERPROFILE")
    if home:
        home_path = Path(home).expanduser().resolve(strict=False)
    else:
        try:
            home_path = Path.home().resolve(strict=False)
        except RuntimeError as error:
            raise ValueError("Unable to determine the user home for plugins") from error
    return home_path / ".local" / "share"


def records(value: Any, collection: str) -> list[dict[str, Any]]:
    """Normalize the list and keyed-object shapes returned by plugin hosts."""

    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []
    for key in dict.fromkeys((collection, "items", "installed")):
        nested = value.get(key)
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, dict)]
    normalized = []
    for key, item in value.items():
        if isinstance(item, dict):
            normalized.append({"name": key, **item})
    return normalized


def marketplace_entry(
    value: Any, marketplace_name: str
) -> dict[str, Any] | None:
    """Find a named marketplace in a normalized host response."""

    for entry in records(value, "marketplaces"):
        if entry.get("name") == marketplace_name:
            return entry
    return None


def owned_marketplace_source(entry: dict[str, Any], expected: Path) -> bool:
    """Return whether a host entry points at BetterBorg's stable bundle path."""

    candidates: list[str] = []
    source = entry.get("source")
    if isinstance(source, str):
        candidates.append(source)
    elif isinstance(source, dict):
        candidates.extend(
            str(source[key])
            for key in ("path", "directory", "url", "root")
            if isinstance(source.get(key), str)
        )
    marketplace_source = entry.get("marketplaceSource")
    if isinstance(marketplace_source, dict):
        candidates.extend(
            str(marketplace_source[key])
            for key in ("path", "directory", "url", "root", "source")
            if isinstance(marketplace_source.get(key), str)
        )
    candidates.extend(
        str(entry[key])
        for key in ("path", "directory", "root")
        if isinstance(entry.get(key), str)
    )
    expected = expected.resolve(strict=False)
    for candidate in candidates:
        if candidate.startswith("file://"):
            candidate = candidate.removeprefix("file://")
        if Path(candidate).expanduser().resolve(strict=False) == expected:
            return True
    return False


def bundle_digest(source: Any) -> str:
    """Hash all bundle files except BetterBorg's ownership marker."""

    digest = hashlib.sha256()
    for relative, body in _bundle_files(source):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(body)
        digest.update(b"\0")
    return digest.hexdigest()


def materialize_bundle(
    source: Any,
    destination: Path,
    digest: str,
    version: str,
    *,
    host_name: str,
    manifest_path: str,
    version_change_name: str,
) -> BundleChange:
    """Atomically materialize an owned bundle and retain its prior version."""

    created_parents = []
    parent = destination.parent
    while not parent.exists():
        created_parents.append(parent)
        parent = parent.parent
    destination.parent.mkdir(parents=True, exist_ok=True)
    previous_version = None
    ownership_before: dict[str, Any] | None = None
    if destination.exists():
        ownership_before = ownership(destination)
        if ownership_before is None:
            raise PluginCollisionError(
                f"{host_name} bundle path {destination} already exists without "
                "BetterBorg ownership metadata; it was left untouched."
            )
        materialized_digest = bundle_digest(destination)
        previous_version = ownership_before.get("version")
        if not isinstance(previous_version, str):
            previous_version = bundle_version(destination, manifest_path)
        if (
            ownership_before.get("digest") == digest
            and materialized_digest == digest
        ):
            return BundleChange(
                path=destination,
                digest=digest,
                version=version,
                changed=False,
                previous_version=previous_version,
            )
        if (
            ownership_before.get("digest") != digest
            and materialized_digest != digest
            and previous_version == version
        ):
            raise ValueError(
                f"{host_name} plugin bundle content changed without a "
                f"{version_change_name} from {version}."
            )

    staging = destination.parent / f".marketplace-staging-{uuid4().hex}"
    try:
        _copy_tree(source, staging)
        staging.joinpath(_OWNER_FILE).write_text(
            json.dumps(
                {
                    "schema": _OWNER_SCHEMA,
                    "owner": "betterborg-cli",
                    "digest": digest,
                    "version": version,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        previous = None
        if destination.exists():
            backups = destination.parent / "backups"
            backups.mkdir(exist_ok=True)
            ownership_before = ownership(destination)
            if ownership_before is None:
                raise OSError(
                    f"The owned {host_name} bundle changed before it could be "
                    "backed up."
                )
            old_digest = ownership_before["digest"][:12]
            previous = backups / f"marketplace-{old_digest}-{uuid4().hex[:8]}"
            destination.rename(previous)
        try:
            staging.rename(destination)
        except OSError as promotion_error:
            if previous is not None:
                try:
                    previous.rename(destination)
                except OSError as restoration_error:
                    raise OSError(
                        f"Could not promote the staged {host_name} marketplace "
                        f"bundle ({promotion_error}) or restore the previous "
                        f"bundle ({restoration_error})."
                    ) from promotion_error
            raise
        return BundleChange(
            path=destination,
            digest=digest,
            version=version,
            changed=True,
            previous=previous,
            previous_version=previous_version,
            created_parents=tuple(created_parents),
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def ownership(path: Path) -> dict[str, Any] | None:
    """Read valid BetterBorg ownership metadata from a materialized bundle."""

    marker = path / _OWNER_FILE
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        isinstance(value, dict)
        and value.get("schema") == _OWNER_SCHEMA
        and value.get("owner") == "betterborg-cli"
        and isinstance(value.get("digest"), str)
    ):
        return value
    return None


def bundle_version(source: Any, manifest_path: str) -> str | None:
    """Read a plugin version from a bundle manifest when possible."""

    manifest = source.joinpath(*manifest_path.split("/"))
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    version = value.get("version") if isinstance(value, dict) else None
    return version if isinstance(version, str) else None


def _bundle_files(source: Any, prefix: str = "") -> list[tuple[str, bytes]]:
    files: list[tuple[str, bytes]] = []
    for child in source.iterdir():
        relative = f"{prefix}/{child.name}" if prefix else child.name
        if child.is_dir():
            files.extend(_bundle_files(child, relative))
        elif child.is_file() and relative != _OWNER_FILE:
            files.append((relative, child.read_bytes()))
    return sorted(files)


def _copy_tree(source: Any, destination: Path) -> None:
    destination.mkdir()
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            _copy_tree(child, target)
        elif child.is_file():
            target.write_bytes(child.read_bytes())
