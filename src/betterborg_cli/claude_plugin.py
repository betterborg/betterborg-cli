"""Owned Claude Code plugin installation for BetterBorg."""

from __future__ import annotations

import json
import os
import re
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
from betterborg_cli.plugin_installation import (
    BundleChange,
    CommandRunner,
    PluginCollisionError,
    PluginCommandError,
    bundle_digest,
    json_plugin_command,
    marketplace_entry,
    materialize_bundle,
    owned_marketplace_source,
    plugin_data_root,
    records,
    run_plugin_command,
)

MARKETPLACE_NAME = "betterborg"
PLUGIN_NAME = "borg"
PLUGIN_ID = f"{PLUGIN_NAME}@{MARKETPLACE_NAME}"
RELOAD_GUIDANCE = (
    "Run `/reload-plugins` in any open Claude Code session (or start a new "
    "session) to load the BetterBorg plugin and its MCP tools."
)

_MINIMUM_SAFE_CLAUDE_VERSION = (2, 1, 212)
_MINIMUM_SAFE_CLAUDE_VERSION_TEXT = ".".join(
    str(part) for part in _MINIMUM_SAFE_CLAUDE_VERSION
)


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
class _PluginState:
    installed: bool
    enabled: bool
    version: str | None


@dataclass(slots=True)
class _HostChanges:
    marketplace_added: bool = False
    plugin_installed: bool = False
    plugin_enabled: bool = False


ExecutableLookup = Callable[..., str | None]
McpVerifier = Callable[[PluginActivationPreflight, Mapping[str, str]], None]


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
        version_result = run_plugin_command(
            (str(Path(claude)), "--version"), environment, command_runner
        )
    except PluginCommandError as error:
        return ClaudePluginInstallation(
            status=ClaudePluginStatus.SETUP_REQUIRED,
            reason=str(error),
            guidance=(
                "Repair or update Claude Code, then confirm `claude --version` "
                "succeeds."
            ),
        )
    claude_version = _parse_claude_version(version_result.stdout)
    if claude_version is None or claude_version < _MINIMUM_SAFE_CLAUDE_VERSION:
        reported = version_result.stdout.strip() or "an unknown version"
        return ClaudePluginInstallation(
            status=ClaudePluginStatus.SETUP_REQUIRED,
            reason=(
                f"Claude Code reported {reported!r}; version "
                f"{_MINIMUM_SAFE_CLAUDE_VERSION_TEXT} or newer is required for "
                "collision-safe plugin rollback."
            ),
            guidance=(
                "Update Claude Code, then confirm `claude --version` reports "
                f"{_MINIMUM_SAFE_CLAUDE_VERSION_TEXT} or newer."
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
        root = plugin_data_root(environment, data_home) / "betterborg" / "claude"
        marketplace_path = root / "marketplace"
        marketplaces = json_plugin_command(
            (str(Path(claude)), "plugin", "marketplace", "list", "--json"),
            environment,
            command_runner,
        )
        registered = marketplace_entry(marketplaces, MARKETPLACE_NAME)
        if registered is not None and not owned_marketplace_source(
            registered, marketplace_path
        ):
            raise PluginCollisionError(
                f"Claude marketplace {MARKETPLACE_NAME!r} is already registered "
                "from a source BetterBorg does not own; it was left untouched."
            )
        plugins = json_plugin_command(
            (str(Path(claude)), "plugin", "list", "--json"),
            environment,
            command_runner,
        )
        before = _plugin_state(plugins)
        change = materialize_bundle(
            source,
            marketplace_path,
            digest,
            version,
            host_name="Claude",
            manifest_path="plugins/borg/.claude-plugin/plugin.json",
            version_change_name="version bump",
        )
    except PluginCollisionError as error:
        return ClaudePluginInstallation(
            status=ClaudePluginStatus.COLLISION,
            preflight=preflight,
            reason=str(error),
        )
    except (
        OSError,
        ValueError,
        PluginActivationVerificationError,
        PluginCommandError,
    ) as error:
        return ClaudePluginInstallation(
            status=ClaudePluginStatus.FAILED,
            preflight=preflight,
            reason=str(error),
        )

    host_changes = _HostChanges()
    try:
        if registered is None:
            run_plugin_command(
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
            run_plugin_command(
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
            run_plugin_command(
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
            run_plugin_command(
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
            run_plugin_command(
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
            json_plugin_command(
                (str(Path(claude)), "plugin", "list", "--json"),
                environment,
                command_runner,
            )
        )
        if not verified.installed or not verified.enabled:
            raise PluginCommandError(
                f"Claude did not report {PLUGIN_ID} as installed and enabled."
            )
        if verified.version != version:
            reported = verified.version or "an unknown version"
            raise PluginCommandError(
                f"Claude reported {PLUGIN_ID} at {reported}, expected {version}."
            )
        (mcp_verifier or verify_borg_mcp)(preflight, environment)
    except (
        OSError,
        ValueError,
        PluginActivationVerificationError,
        PluginCommandError,
    ) as error:
        rollback_error = _rollback(
            change,
            claude=str(Path(claude)),
            environment=environment,
            command_runner=command_runner,
            marketplace_preexisting=registered is not None,
            plugin_before=before,
            host_changes=host_changes,
        )
        if rollback_error is None:
            reason = f"Claude plugin activation failed and was rolled back: {error}"
        else:
            reason = (
                f"Claude plugin activation failed: {error} "
                f"Rollback also failed: {rollback_error}"
            )
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


def _parse_claude_version(output: str) -> tuple[int, int, int] | None:
    match = re.search(
        r"(?<![\d.])(\d+)\.(\d+)\.(\d+)(?![-+\d.])",
        output,
    )
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def _plugin_state(value: Any) -> _PluginState:
    installed = False
    enabled = False
    version = None
    for entry in records(value, "plugins"):
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
    return bundle_digest(source), version


def _rollback(
    change: BundleChange,
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
            run_plugin_command(
                (claude, "plugin", "marketplace", "update", MARKETPLACE_NAME),
                environment,
                command_runner,
            )
        except PluginCommandError as error:
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
                run_plugin_command(command, environment, command_runner)
            except PluginCommandError as error:
                errors.append(str(error))
    elif host_changes.plugin_installed:
        try:
            run_plugin_command(
                (claude, "plugin", "uninstall", PLUGIN_ID, "--scope", "user"),
                environment,
                command_runner,
            )
        except PluginCommandError as error:
            errors.append(str(error))
    if host_changes.marketplace_added:
        try:
            run_plugin_command(
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
        except PluginCommandError as error:
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
        marketplaces = json_plugin_command(
            (claude, "plugin", "marketplace", "list", "--json"),
            environment,
            command_runner,
        )
    except PluginCommandError as error:
        errors.append(f"could not verify the restored marketplace state: {error}")
    else:
        marketplace_after = marketplace_entry(marketplaces, MARKETPLACE_NAME)
        if marketplace_preexisting:
            if marketplace_after is None:
                errors.append(
                    f"Claude no longer reports marketplace {MARKETPLACE_NAME!r}"
                )
            elif not owned_marketplace_source(marketplace_after, change.path):
                errors.append(
                    f"Claude reports marketplace {MARKETPLACE_NAME!r} from an "
                    "unexpected source after rollback"
                )
        elif marketplace_after is not None:
            errors.append(
                f"Claude still reports marketplace {MARKETPLACE_NAME!r} after "
                "rollback"
            )

    try:
        plugins = json_plugin_command(
            (claude, "plugin", "list", "--json"),
            environment,
            command_runner,
        )
    except PluginCommandError as error:
        errors.append(f"could not verify the restored plugin state: {error}")
    else:
        plugin_after = _plugin_state(plugins)
        if plugin_after != plugin_before:
            errors.append(
                f"Claude reports {PLUGIN_ID} after rollback as "
                f"installed={plugin_after.installed}, enabled={plugin_after.enabled}, "
                f"version={plugin_after.version!r}; expected "
                f"installed={plugin_before.installed}, "
                f"enabled={plugin_before.enabled}, "
                f"version={plugin_before.version!r}"
            )
    return "; ".join(errors) or None
