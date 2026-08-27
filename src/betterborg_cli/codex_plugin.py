"""Owned Codex plugin installation for BetterBorg."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from importlib import resources
from pathlib import Path
from typing import Any
from uuid import uuid4

from betterborg_cli.plugin_activation import (
    PluginActivationPreflight,
    PluginActivationVerificationError,
    preflight_plugin_activation,
    verify_borg_mcp,
)

MARKETPLACE_NAME = "betterborg"
PLUGIN_NAME = "borg"
PLUGIN_ID = f"{PLUGIN_NAME}@{MARKETPLACE_NAME}"
NEW_THREAD_GUIDANCE = (
    "Start a new Codex thread to load the BetterBorg plugin, its skill, and "
    "its MCP tools."
)

_OWNER_FILE = ".betterborg-owned.json"
_OWNER_SCHEMA = 1


class CodexPluginStatus(StrEnum):
    """Consumer-visible outcome of a Codex plugin installation."""

    INSTALLED = "installed"
    UNCHANGED = "unchanged"
    SETUP_REQUIRED = "setup_required"
    COLLISION = "collision"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CodexPluginInstallation:
    """Result of installing or reconciling the owned Codex plugin."""

    status: CodexPluginStatus
    preflight: PluginActivationPreflight | None = None
    bundle_path: Path | None = None
    previous_bundle: Path | None = None
    reason: str | None = None
    guidance: str | None = None
    new_thread_guidance: str | None = None

    @property
    def ready(self) -> bool:
        return self.status in {
            CodexPluginStatus.INSTALLED,
            CodexPluginStatus.UNCHANGED,
        }


@dataclass(frozen=True, slots=True)
class _BundleChange:
    path: Path
    digest: str
    version: str
    changed: bool
    previous: Path | None = None
    previous_version: str | None = None
    created_parents: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class _PluginState:
    installed: bool
    version: str | None


@dataclass(slots=True)
class _HostChanges:
    marketplace_removed: bool = False
    marketplace_added: bool = False
    plugin_removed: bool = False
    plugin_added: bool = False


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
ExecutableLookup = Callable[..., str | None]
McpVerifier = Callable[[PluginActivationPreflight, Mapping[str, str]], None]


class _CodexCommandError(RuntimeError):
    pass


class _CollisionError(RuntimeError):
    pass


def install_codex_plugin(
    *,
    launch_environment: Mapping[str, str] | None = None,
    data_home: Path | None = None,
    bundle_source: Any | None = None,
    executable_lookup: ExecutableLookup = shutil.which,
    command_runner: CommandRunner = subprocess.run,
    mcp_verifier: McpVerifier | None = None,
) -> CodexPluginInstallation:
    """Install and verify BetterBorg's user-scoped Codex plugin.

    Codex's supported plugin commands own marketplace and install state. This
    function owns only its stable materialized marketplace bundle and never
    edits Codex configuration or personal marketplace files directly.
    """

    environment = dict(os.environ if launch_environment is None else launch_environment)
    path = environment.get("PATH", os.defpath)
    codex = executable_lookup("codex", path=path)
    if codex is None:
        return CodexPluginInstallation(
            status=CodexPluginStatus.SETUP_REQUIRED,
            reason="Codex was not found on the host launch PATH.",
            guidance=(
                "Install Codex, ensure `codex` is on the host launch PATH, and "
                "confirm `codex --version` succeeds."
            ),
        )

    preflight = preflight_plugin_activation(
        launch_environment=environment,
        executable_lookup=executable_lookup,
    )
    if not preflight.ready:
        return CodexPluginInstallation(
            status=CodexPluginStatus.SETUP_REQUIRED,
            preflight=preflight,
            reason=preflight.reason,
            guidance=preflight.guidance,
        )

    source = (
        resources.files("betterborg_cli.codex_plugin_bundle") / "marketplace"
        if bundle_source is None
        else bundle_source
    )
    try:
        digest, version = _validate_and_digest_bundle(source)
    except (OSError, ValueError) as error:
        return CodexPluginInstallation(
            status=CodexPluginStatus.FAILED,
            preflight=preflight,
            reason=f"The packaged Codex plugin bundle is invalid: {error}",
        )

    try:
        root = _data_root(environment, data_home) / "betterborg" / "codex"
        marketplace_path = root / "marketplace"
        owned_bundle_before = _ownership(marketplace_path) is not None
        marketplaces = _json_command(
            (str(Path(codex)), "plugin", "marketplace", "list", "--json"),
            environment,
            command_runner,
        )
        registered = _marketplace_entry(marketplaces)
        if registered is not None and not _owned_marketplace_source(
            registered, marketplace_path
        ):
            raise _CollisionError(
                f"Codex marketplace {MARKETPLACE_NAME!r} is already registered "
                "from a source BetterBorg does not own; it was left untouched."
            )
        plugins = _json_command(
            (str(Path(codex)), "plugin", "list", "--available", "--json"),
            environment,
            command_runner,
        )
        before = _plugin_state(plugins)
        if registered is None and before.installed and not owned_bundle_before:
            raise _CollisionError(
                f"Codex reports an orphaned {PLUGIN_ID} installation without an "
                "owned BetterBorg marketplace bundle; it was left untouched."
            )
        change = _materialize_bundle(source, marketplace_path, digest, version)
    except _CollisionError as error:
        return CodexPluginInstallation(
            status=CodexPluginStatus.COLLISION,
            preflight=preflight,
            reason=str(error),
        )
    except (OSError, ValueError, _CodexCommandError) as error:
        return CodexPluginInstallation(
            status=CodexPluginStatus.FAILED,
            preflight=preflight,
            reason=str(error),
        )

    host_changes = _HostChanges()
    try:
        orphaned_install = registered is None and before.installed
        stale_install = before.installed and before.version != version
        bundle_version_changed = (
            change.changed
            and change.previous_version is not None
            and change.previous_version != version
        )
        refresh_marketplace = registered is not None and (
            stale_install or bundle_version_changed
        )

        if stale_install or orphaned_install:
            _run(
                (str(Path(codex)), "plugin", "remove", PLUGIN_ID, "--json"),
                environment,
                command_runner,
            )
            host_changes.plugin_removed = True
        if refresh_marketplace:
            _run(
                (
                    str(Path(codex)),
                    "plugin",
                    "marketplace",
                    "remove",
                    MARKETPLACE_NAME,
                    "--json",
                ),
                environment,
                command_runner,
            )
            host_changes.marketplace_removed = True

        if registered is None or refresh_marketplace:
            _run(
                (
                    str(Path(codex)),
                    "plugin",
                    "marketplace",
                    "add",
                    str(marketplace_path),
                    "--json",
                ),
                environment,
                command_runner,
            )
            host_changes.marketplace_added = True

        if not before.installed or host_changes.plugin_removed:
            _run(
                (str(Path(codex)), "plugin", "add", PLUGIN_ID, "--json"),
                environment,
                command_runner,
            )
            host_changes.plugin_added = True

        verified_marketplace = _marketplace_entry(
            _json_command(
                (str(Path(codex)), "plugin", "marketplace", "list", "--json"),
                environment,
                command_runner,
            )
        )
        if verified_marketplace is None or not _owned_marketplace_source(
            verified_marketplace, marketplace_path
        ):
            raise _CodexCommandError(
                f"Codex did not report marketplace {MARKETPLACE_NAME!r} from "
                "the owned BetterBorg bundle."
            )
        verified = _plugin_state(
            _json_command(
                (
                    str(Path(codex)),
                    "plugin",
                    "list",
                    "--available",
                    "--json",
                ),
                environment,
                command_runner,
            )
        )
        if not verified.installed:
            raise _CodexCommandError(
                f"Codex did not report {PLUGIN_ID} as installed."
            )
        if verified.version != version:
            reported = verified.version or "an unknown version"
            raise _CodexCommandError(
                f"Codex reported {PLUGIN_ID} at {reported}, expected {version}."
            )
        (mcp_verifier or verify_borg_mcp)(preflight, environment)
    except (
        OSError,
        ValueError,
        PluginActivationVerificationError,
        _CodexCommandError,
    ) as error:
        rollback_error = _rollback(
            change,
            codex=str(Path(codex)),
            environment=environment,
            command_runner=command_runner,
            marketplace_preexisting=registered is not None,
            plugin_before=before,
            host_changes=host_changes,
        )
        if rollback_error is None:
            reason = f"Codex plugin activation failed and was rolled back: {error}"
        else:
            reason = (
                f"Codex plugin activation failed: {error} "
                f"Rollback also failed: {rollback_error}"
            )
        return CodexPluginInstallation(
            status=CodexPluginStatus.FAILED,
            preflight=preflight,
            bundle_path=marketplace_path if marketplace_path.exists() else None,
            previous_bundle=(
                change.previous
                if change.previous is not None and change.previous.exists()
                else None
            ),
            reason=reason,
        )

    changed = change.changed or any(
        (
            host_changes.marketplace_removed,
            host_changes.marketplace_added,
            host_changes.plugin_removed,
            host_changes.plugin_added,
        )
    )
    return CodexPluginInstallation(
        status=(
            CodexPluginStatus.INSTALLED if changed else CodexPluginStatus.UNCHANGED
        ),
        preflight=preflight,
        bundle_path=marketplace_path,
        previous_bundle=change.previous,
        new_thread_guidance=NEW_THREAD_GUIDANCE,
    )


def _run(
    command: tuple[str, ...],
    environment: Mapping[str, str],
    runner: CommandRunner,
) -> subprocess.CompletedProcess[str]:
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
        raise _CodexCommandError(f"unable to run {command[0]!r}: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise _CodexCommandError(
            f"`{' '.join(command)}` failed with exit code "
            f"{completed.returncode}: {detail}"
        )
    return completed


def _json_command(
    command: tuple[str, ...],
    environment: Mapping[str, str],
    runner: CommandRunner,
) -> Any:
    completed = _run(command, environment, runner)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise _CodexCommandError(
            f"`{' '.join(command)}` returned invalid JSON: {error}"
        ) from error


def _data_root(environment: Mapping[str, str], explicit: Path | None) -> Path:
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
            raise ValueError(
                "Unable to determine the user home for the Codex plugin"
            ) from error
    return home_path / ".local" / "share"


def _records(value: Any, collection: str) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []
    for key in (collection, "items", "plugins", "installed"):
        nested = value.get(key)
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, dict)]
    records = []
    for key, item in value.items():
        if isinstance(item, dict):
            records.append({"name": key, **item})
    return records


def _marketplace_entry(value: Any) -> dict[str, Any] | None:
    for entry in _records(value, "marketplaces"):
        if entry.get("name") == MARKETPLACE_NAME:
            return entry
    return None


def _owned_marketplace_source(entry: dict[str, Any], expected: Path) -> bool:
    source = entry.get("source")
    candidates: list[str] = []
    if isinstance(source, str):
        candidates.append(source)
    elif isinstance(source, dict):
        candidates.extend(
            str(source[key])
            for key in ("path", "directory", "url", "root")
            if isinstance(source.get(key), str)
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


def _plugin_state(value: Any) -> _PluginState:
    installed = False
    version = None
    for entry in _records(value, "plugins"):
        identifier = (
            entry.get("pluginId")
            or entry.get("id")
            or entry.get("plugin")
            or entry.get("selector")
        )
        name = entry.get("name")
        marketplace = entry.get("marketplace") or entry.get("marketplaceName")
        if identifier != PLUGIN_ID and not (
            name == PLUGIN_NAME and marketplace == MARKETPLACE_NAME
        ):
            continue
        entry_installed = entry.get("installed")
        if entry_installed is None:
            status = entry.get("status")
            entry_installed = status == "installed" or entry.get("enabled") is True
        if not entry_installed:
            continue
        installed = True
        for key in ("installedVersion", "installed_version", "version"):
            if isinstance(entry.get(key), str):
                version = entry[key]
                break
    return _PluginState(installed=installed, version=version)


def _validate_and_digest_bundle(source: Any) -> tuple[str, str]:
    required = (
        ".agents/plugins/marketplace.json",
        "plugins/borg/.codex-plugin/plugin.json",
        "plugins/borg/.mcp.json",
        "plugins/borg/skills/orchestrate/SKILL.md",
    )
    content: dict[str, bytes] = {}
    for relative in required:
        item = source.joinpath(*relative.split("/"))
        if not item.is_file():
            raise ValueError(f"missing {relative}")
        content[relative] = item.read_bytes()
    marketplace = json.loads(content[required[0]])
    manifest = json.loads(content[required[1]])
    mcp = json.loads(content[required[2]])
    if not isinstance(marketplace, dict) or not isinstance(manifest, dict):
        raise ValueError("marketplace and plugin manifests must be JSON objects")
    if marketplace.get("name") != MARKETPLACE_NAME:
        raise ValueError("marketplace name is not stable")
    entries = marketplace.get("plugins")
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError("marketplace must expose exactly the Borg plugin")
    entry = entries[0]
    expected_entry = {
        "name": PLUGIN_NAME,
        "source": {"source": "local", "path": "./plugins/borg"},
        "policy": {
            "installation": "AVAILABLE",
            "authentication": "ON_INSTALL",
        },
        "category": "Productivity",
    }
    if entry != expected_entry:
        raise ValueError("marketplace Borg entry does not match the Codex contract")
    if manifest.get("name") != PLUGIN_NAME:
        raise ValueError("plugin name is not stable")
    version = manifest.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("plugin version must be a non-empty string")
    if manifest.get("skills") != "./skills/":
        raise ValueError("plugin manifest must register bundled skills")
    if manifest.get("mcpServers") != "./.mcp.json":
        raise ValueError("plugin manifest must register the MCP companion file")
    if mcp != {"borg": {"command": "borg", "args": ["mcp"]}}:
        raise ValueError("MCP registration must execute `borg mcp`")
    return _bundle_digest(source), version


def _bundle_digest(source: Any) -> str:
    digest = hashlib.sha256()
    for relative, body in _bundle_files(source):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(body)
        digest.update(b"\0")
    return digest.hexdigest()


def _bundle_files(source: Any, prefix: str = "") -> list[tuple[str, bytes]]:
    files: list[tuple[str, bytes]] = []
    for child in source.iterdir():
        relative = f"{prefix}/{child.name}" if prefix else child.name
        if child.is_dir():
            files.extend(_bundle_files(child, relative))
        elif child.is_file() and relative != _OWNER_FILE:
            files.append((relative, child.read_bytes()))
    return sorted(files)


def _materialize_bundle(
    source: Any,
    destination: Path,
    digest: str,
    version: str,
) -> _BundleChange:
    created_parents = []
    parent = destination.parent
    while not parent.exists():
        created_parents.append(parent)
        parent = parent.parent
    destination.parent.mkdir(parents=True, exist_ok=True)
    previous_version = None
    if destination.exists():
        ownership = _ownership(destination)
        if ownership is None:
            raise _CollisionError(
                f"Codex bundle path {destination} already exists without "
                "BetterBorg ownership metadata; it was left untouched."
            )
        materialized_digest = _bundle_digest(destination)
        previous_version = ownership.get("version")
        if not isinstance(previous_version, str):
            previous_version = _bundle_version(destination)
        if ownership.get("digest") == digest and materialized_digest == digest:
            return _BundleChange(
                path=destination,
                digest=digest,
                version=version,
                changed=False,
                previous_version=previous_version,
            )
        if (
            ownership.get("digest") != digest
            and materialized_digest != digest
            and previous_version == version
        ):
            raise ValueError(
                "Codex plugin bundle content changed without a cache-busting "
                f"version change from {version}."
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
            old_digest = _ownership(destination)["digest"][:12]
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
                        "Could not promote the staged Codex marketplace bundle "
                        f"({promotion_error}) or restore the previous bundle "
                        f"({restoration_error})."
                    ) from promotion_error
            raise
        return _BundleChange(
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


def _copy_tree(source: Any, destination: Path) -> None:
    destination.mkdir()
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            _copy_tree(child, target)
        elif child.is_file():
            target.write_bytes(child.read_bytes())


def _bundle_version(source: Any) -> str | None:
    manifest = source / "plugins/borg/.codex-plugin/plugin.json"
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    version = value.get("version") if isinstance(value, dict) else None
    return version if isinstance(version, str) else None


def _ownership(path: Path) -> dict[str, Any] | None:
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


def _rollback(
    change: _BundleChange,
    *,
    codex: str,
    environment: Mapping[str, str],
    command_runner: CommandRunner,
    marketplace_preexisting: bool,
    plugin_before: _PluginState,
    host_changes: _HostChanges,
) -> str | None:
    errors: list[str] = []

    if host_changes.plugin_added:
        try:
            _run(
                (codex, "plugin", "remove", PLUGIN_ID, "--json"),
                environment,
                command_runner,
            )
        except _CodexCommandError as error:
            errors.append(str(error))
    if host_changes.marketplace_added:
        try:
            _run(
                (
                    codex,
                    "plugin",
                    "marketplace",
                    "remove",
                    MARKETPLACE_NAME,
                    "--json",
                ),
                environment,
                command_runner,
            )
        except _CodexCommandError as error:
            errors.append(str(error))

    if change.changed and change.previous is not None:
        failed_name = f"failed-{change.digest[:12]}-{uuid4().hex[:8]}"
        failed = change.previous.parent / failed_name
        try:
            change.path.rename(failed)
            change.previous.rename(change.path)
        except OSError as error:
            errors.append(str(error))

    if host_changes.marketplace_removed:
        try:
            _run(
                (
                    codex,
                    "plugin",
                    "marketplace",
                    "add",
                    str(change.path),
                    "--json",
                ),
                environment,
                command_runner,
            )
        except _CodexCommandError as error:
            errors.append(str(error))
    if host_changes.plugin_removed:
        try:
            _run(
                (codex, "plugin", "add", PLUGIN_ID, "--json"),
                environment,
                command_runner,
            )
        except _CodexCommandError as error:
            errors.append(str(error))

    if change.changed and change.previous is None:
        try:
            shutil.rmtree(change.path)
        except OSError as error:
            errors.append(str(error))
        for parent in change.created_parents:
            try:
                parent.rmdir()
            except FileNotFoundError:
                pass
            except OSError as error:
                errors.append(str(error))

    try:
        marketplaces = _json_command(
            (codex, "plugin", "marketplace", "list", "--json"),
            environment,
            command_runner,
        )
    except _CodexCommandError as error:
        errors.append(f"could not verify the restored marketplace state: {error}")
    else:
        marketplace_after = _marketplace_entry(marketplaces)
        if marketplace_preexisting:
            if marketplace_after is None:
                errors.append(
                    f"Codex no longer reports marketplace {MARKETPLACE_NAME!r}"
                )
            elif not _owned_marketplace_source(marketplace_after, change.path):
                errors.append(
                    f"Codex reports marketplace {MARKETPLACE_NAME!r} from an "
                    "unexpected source after rollback"
                )
        elif marketplace_after is not None:
            errors.append(
                f"Codex still reports marketplace {MARKETPLACE_NAME!r} after rollback"
            )

    try:
        plugins = _json_command(
            (codex, "plugin", "list", "--available", "--json"),
            environment,
            command_runner,
        )
    except _CodexCommandError as error:
        errors.append(f"could not verify the restored plugin state: {error}")
    else:
        plugin_after = _plugin_state(plugins)
        if plugin_after != plugin_before:
            errors.append(
                f"Codex reports {PLUGIN_ID} after rollback as "
                f"installed={plugin_after.installed}, "
                f"version={plugin_after.version!r}; "
                f"expected installed={plugin_before.installed}, "
                f"version={plugin_before.version!r}"
            )
    return "; ".join(errors) or None
