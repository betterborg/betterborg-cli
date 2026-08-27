"""Owned Claude Code plugin installation for BetterBorg."""

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
    preflight_plugin_activation,
)

MARKETPLACE_NAME = "betterborg"
PLUGIN_NAME = "borg"
PLUGIN_ID = f"{PLUGIN_NAME}@{MARKETPLACE_NAME}"
RELOAD_GUIDANCE = (
    "Run `/reload-plugins` in any open Claude Code session (or start a new "
    "session) to load the BetterBorg plugin and its MCP tools."
)

_OWNER_FILE = ".betterborg-owned.json"
_OWNER_SCHEMA = 1


class ClaudePluginStatus(StrEnum):
    """Consumer-visible outcome of a Claude plugin installation."""

    INSTALLED = "installed"
    UNCHANGED = "unchanged"
    SETUP_REQUIRED = "setup_required"
    COLLISION = "collision"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ClaudePluginInstallation:
    """Result of installing or reconciling the owned Claude plugin."""

    status: ClaudePluginStatus
    preflight: PluginActivationPreflight | None = None
    bundle_path: Path | None = None
    previous_bundle: Path | None = None
    reason: str | None = None
    guidance: str | None = None
    reload_guidance: str | None = None

    @property
    def ready(self) -> bool:
        return self.status in {
            ClaudePluginStatus.INSTALLED,
            ClaudePluginStatus.UNCHANGED,
        }


@dataclass(frozen=True, slots=True)
class _BundleChange:
    path: Path
    digest: str
    changed: bool
    previous: Path | None = None
    created_parents: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class _PluginState:
    installed: bool
    enabled: bool
    version: str | None


@dataclass(slots=True)
class _HostChanges:
    marketplace_added: bool = False
    plugin_installed: bool = False
    plugin_enabled: bool = False


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
ExecutableLookup = Callable[..., str | None]
McpVerifier = Callable[[PluginActivationPreflight, Mapping[str, str]], None]


class _ClaudeCommandError(RuntimeError):
    pass


class _CollisionError(RuntimeError):
    pass


def install_claude_plugin(
    *,
    launch_environment: Mapping[str, str] | None = None,
    data_home: Path | None = None,
    bundle_source: Any | None = None,
    executable_lookup: ExecutableLookup = shutil.which,
    command_runner: CommandRunner = subprocess.run,
    mcp_verifier: McpVerifier | None = None,
) -> ClaudePluginInstallation:
    """Install and verify BetterBorg's user-scoped Claude Code plugin.

    Claude's supported CLI owns marketplace and plugin configuration. This
    function owns only its materialized marketplace bundle and never edits a
    Claude settings file directly.
    """

    environment = dict(os.environ if launch_environment is None else launch_environment)
    path = environment.get("PATH", os.defpath)
    claude = executable_lookup("claude", path=path)
    if claude is None:
        return ClaudePluginInstallation(
            status=ClaudePluginStatus.SETUP_REQUIRED,
            reason="Claude Code was not found on the host launch PATH.",
            guidance=(
                "Install Claude Code, ensure `claude` is on the host launch PATH, "
                "and confirm `claude --version` succeeds."
            ),
        )
    try:
        _run((str(Path(claude)), "--version"), environment, command_runner)
    except _ClaudeCommandError as error:
        return ClaudePluginInstallation(
            status=ClaudePluginStatus.SETUP_REQUIRED,
            reason=str(error),
            guidance=(
                "Repair or update Claude Code, then confirm `claude --version` "
                "succeeds."
            ),
        )

    preflight = preflight_plugin_activation(
        launch_environment=environment,
        executable_lookup=executable_lookup,
    )
    if not preflight.ready:
        return ClaudePluginInstallation(
            status=ClaudePluginStatus.SETUP_REQUIRED,
            preflight=preflight,
            reason=preflight.reason,
            guidance=preflight.guidance,
        )

    source = (
        resources.files("betterborg_cli.claude_plugin_bundle") / "marketplace"
        if bundle_source is None
        else bundle_source
    )
    try:
        digest, version = _validate_and_digest_bundle(source)
    except (OSError, ValueError) as error:
        return ClaudePluginInstallation(
            status=ClaudePluginStatus.FAILED,
            preflight=preflight,
            reason=f"The packaged Claude plugin bundle is invalid: {error}",
        )

    try:
        root = _data_root(environment, data_home) / "betterborg" / "claude"
        marketplace_path = root / "marketplace"
        marketplaces = _json_command(
            (str(Path(claude)), "plugin", "marketplace", "list", "--json"),
            environment,
            command_runner,
        )
        registered = _marketplace_entry(marketplaces)
        if registered is not None and not _owned_marketplace_source(
            registered, marketplace_path
        ):
            raise _CollisionError(
                f"Claude marketplace {MARKETPLACE_NAME!r} is already registered "
                "from a source BetterBorg does not own; it was left untouched."
            )
        plugins = _json_command(
            (str(Path(claude)), "plugin", "list", "--json"),
            environment,
            command_runner,
        )
        before = _plugin_state(plugins)
        change = _materialize_bundle(source, marketplace_path, digest, version)
    except _CollisionError as error:
        return ClaudePluginInstallation(
            status=ClaudePluginStatus.COLLISION,
            preflight=preflight,
            reason=str(error),
        )
    except (OSError, ValueError, _ClaudeCommandError) as error:
        return ClaudePluginInstallation(
            status=ClaudePluginStatus.FAILED,
            preflight=preflight,
            reason=str(error),
        )

    host_changes = _HostChanges()
    try:
        if registered is None:
            _run(
                (
                    str(Path(claude)),
                    "plugin",
                    "marketplace",
                    "add",
                    str(marketplace_path),
                    "--scope",
                    "user",
                ),
                environment,
                command_runner,
            )
            host_changes.marketplace_added = True
        elif change.changed:
            _run(
                (
                    str(Path(claude)),
                    "plugin",
                    "marketplace",
                    "update",
                    MARKETPLACE_NAME,
                ),
                environment,
                command_runner,
            )

        if not before.installed:
            _run(
                (
                    str(Path(claude)),
                    "plugin",
                    "install",
                    PLUGIN_ID,
                    "--scope",
                    "user",
                ),
                environment,
                command_runner,
            )
            host_changes.plugin_installed = True
        elif change.changed:
            _run(
                (
                    str(Path(claude)),
                    "plugin",
                    "update",
                    PLUGIN_ID,
                    "--scope",
                    "user",
                ),
                environment,
                command_runner,
            )

        if not before.installed or change.changed or not before.enabled:
            _run(
                (
                    str(Path(claude)),
                    "plugin",
                    "enable",
                    PLUGIN_ID,
                    "--scope",
                    "user",
                ),
                environment,
                command_runner,
            )
            host_changes.plugin_enabled = True

        verified = _plugin_state(
            _json_command(
                (str(Path(claude)), "plugin", "list", "--json"),
                environment,
                command_runner,
            )
        )
        if not verified.installed or not verified.enabled:
            raise _ClaudeCommandError(
                f"Claude did not report {PLUGIN_ID} as installed and enabled."
            )
        if verified.version != version:
            reported = verified.version or "an unknown version"
            raise _ClaudeCommandError(
                f"Claude reported {PLUGIN_ID} at {reported}, expected {version}."
            )
        (mcp_verifier or verify_borg_mcp)(preflight, environment)
    except (OSError, ValueError, _ClaudeCommandError) as error:
        rollback_error = _rollback(
            change,
            claude=str(Path(claude)),
            environment=environment,
            command_runner=command_runner,
            marketplace_preexisting=registered is not None,
            plugin_before=before,
            host_changes=host_changes,
        )
        reason = f"Claude plugin activation failed and was rolled back: {error}"
        if rollback_error is not None:
            reason += f" Rollback also failed: {rollback_error}"
        return ClaudePluginInstallation(
            status=ClaudePluginStatus.FAILED,
            preflight=preflight,
            bundle_path=marketplace_path if marketplace_path.exists() else None,
            previous_bundle=change.previous,
            reason=reason,
        )

    changed = (
        change.changed
        or registered is None
        or not before.installed
        or not before.enabled
    )
    return ClaudePluginInstallation(
        status=(
            ClaudePluginStatus.INSTALLED
            if changed
            else ClaudePluginStatus.UNCHANGED
        ),
        preflight=preflight,
        bundle_path=marketplace_path,
        previous_bundle=change.previous,
        reload_guidance=RELOAD_GUIDANCE,
    )


def verify_borg_mcp(
    preflight: PluginActivationPreflight,
    environment: Mapping[str, str],
) -> None:
    """Spawn the configured MCP command and require an initialize response."""

    if preflight.executable is None:
        raise ValueError("persistent borg executable is unavailable")
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "betterborg-plugin-installer", "version": "1"},
        },
    }
    try:
        completed = subprocess.run(
            [str(preflight.executable), "mcp"],
            input=json.dumps(request) + "\n",
            capture_output=True,
            check=False,
            env=dict(environment),
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise _ClaudeCommandError(f"unable to start `borg mcp`: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "no diagnostic output"
        raise _ClaudeCommandError(f"`borg mcp` failed: {detail}")
    for line in completed.stdout.splitlines():
        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            continue
        if response.get("id") == 1 and isinstance(response.get("result"), dict):
            return
    raise _ClaudeCommandError("`borg mcp` did not answer the initialize request")


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
        raise _ClaudeCommandError(f"unable to run {command[0]!r}: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise _ClaudeCommandError(
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
        raise _ClaudeCommandError(
            f"`{' '.join(command)}` returned invalid JSON: {error}"
        ) from error


def _data_root(environment: Mapping[str, str], explicit: Path | None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve(strict=False)
    if environment.get("XDG_DATA_HOME"):
        return Path(environment["XDG_DATA_HOME"]).expanduser().resolve(strict=False)
    home = environment.get("HOME")
    if not home:
        raise ValueError(
            "HOME or XDG_DATA_HOME is required to install the Claude plugin"
        )
    return Path(home).expanduser().resolve(strict=False) / ".local" / "share"


def _records(value: Any, collection: str) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []
    for key in (collection, "items", "installed"):
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
            for key in ("path", "directory", "url")
            if isinstance(source.get(key), str)
        )
    candidates.extend(
        str(entry[key])
        for key in ("path", "directory")
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
    enabled = False
    version = None
    for entry in _records(value, "plugins"):
        identifier = entry.get("id") or entry.get("plugin")
        name = entry.get("name")
        marketplace = entry.get("marketplace") or entry.get("marketplaceName")
        if identifier == PLUGIN_ID or (
            name == PLUGIN_NAME and marketplace == MARKETPLACE_NAME
        ):
            scope = entry.get("scope")
            if not isinstance(scope, str) or scope.casefold() != "user":
                continue
            installed = True
            entry_enabled = entry.get("enabled")
            if entry_enabled is None:
                entry_enabled = entry.get("status") == "enabled"
            enabled = enabled or bool(entry_enabled)
            if isinstance(entry.get("version"), str):
                version = entry["version"]
    return _PluginState(installed=installed, enabled=enabled, version=version)


def _validate_and_digest_bundle(source: Any) -> tuple[str, str]:
    required = (
        ".claude-plugin/marketplace.json",
        "plugins/borg/.claude-plugin/plugin.json",
        "plugins/borg/.mcp.json",
        "plugins/borg/commands/borg.md",
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
    if manifest.get("name") != PLUGIN_NAME:
        raise ValueError("plugin name is not stable")
    version = manifest.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("plugin version must be a non-empty string")
    if mcp != {"mcpServers": {"borg": {"command": "borg", "args": ["mcp"]}}}:
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
    if destination.exists():
        ownership = _ownership(destination)
        if ownership is None:
            raise _CollisionError(
                f"Claude bundle path {destination} already exists without "
                "BetterBorg ownership metadata; it was left untouched."
            )
        materialized_digest = _bundle_digest(destination)
        if ownership.get("digest") == digest and materialized_digest == digest:
            return _BundleChange(path=destination, digest=digest, changed=False)
        if ownership.get("digest") != digest and materialized_digest != digest:
            previous_version = ownership.get("version")
            if not isinstance(previous_version, str):
                previous_version = _bundle_version(destination)
            if previous_version == version:
                raise ValueError(
                    f"Claude plugin bundle content changed without a version bump "
                    f"from {version}."
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
        staging.rename(destination)
        return _BundleChange(
            path=destination,
            digest=digest,
            changed=True,
            previous=previous,
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
    manifest = source / "plugins/borg/.claude-plugin/plugin.json"
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
    claude: str,
    environment: Mapping[str, str],
    command_runner: CommandRunner,
    marketplace_preexisting: bool,
    plugin_before: _PluginState,
    host_changes: _HostChanges,
) -> str | None:
    errors: list[str] = []
    restored_previous = False
    if change.changed and change.previous is not None:
        failed_name = f"failed-{change.digest[:12]}-{uuid4().hex[:8]}"
        failed = change.previous.parent / failed_name
        try:
            change.path.rename(failed)
            change.previous.rename(change.path)
            restored_previous = True
        except OSError as error:
            errors.append(str(error))
    if marketplace_preexisting and restored_previous:
        try:
            _run(
                (claude, "plugin", "marketplace", "update", MARKETPLACE_NAME),
                environment,
                command_runner,
            )
        except _ClaudeCommandError as error:
            errors.append(str(error))
    if plugin_before.installed:
        commands = []
        if restored_previous:
            commands.append(
                (claude, "plugin", "update", PLUGIN_ID, "--scope", "user")
            )
        if restored_previous or host_changes.plugin_enabled:
            commands.append(
                (
                    claude,
                    "plugin",
                    "enable" if plugin_before.enabled else "disable",
                    PLUGIN_ID,
                    "--scope",
                    "user",
                )
            )
        for command in commands:
            try:
                _run(command, environment, command_runner)
            except _ClaudeCommandError as error:
                errors.append(str(error))
    elif host_changes.plugin_installed:
        try:
            _run(
                (claude, "plugin", "uninstall", PLUGIN_ID, "--scope", "user"),
                environment,
                command_runner,
            )
        except _ClaudeCommandError as error:
            errors.append(str(error))
    if host_changes.marketplace_added:
        try:
            _run(
                (
                    claude,
                    "plugin",
                    "marketplace",
                    "remove",
                    MARKETPLACE_NAME,
                    "--scope",
                    "user",
                ),
                environment,
                command_runner,
            )
        except _ClaudeCommandError as error:
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
    return "; ".join(errors) or None
