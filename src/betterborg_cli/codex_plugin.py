"""Owned Codex plugin installation for Betterborg."""

from __future__ import annotations

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
    ownership,
    plugin_data_root,
    records,
    run_plugin_command,
)

MARKETPLACE_NAME = "betterborg"
PLUGIN_NAME = "borg"
PLUGIN_ID = f"{PLUGIN_NAME}@{MARKETPLACE_NAME}"
NEW_THREAD_GUIDANCE = (
    "Start a new Codex thread to load the Betterborg plugin, its skill, and "
    "its MCP tools."
)

class CodexPluginStatus(StrEnum):
    """Consumer-visible outcome of a Codex plugin installation."""

    INSTALLED = "installed"
    UNCHANGED = "unchanged"
    DEFERRED = "deferred"
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
class _PluginState:
    installed: bool
    enabled: bool
    version: str | None


@dataclass(slots=True)
class _HostChanges:
    marketplace_removed: bool = False
    marketplace_added: bool = False
    plugin_removed: bool = False
    plugin_added: bool = False


ExecutableLookup = Callable[..., str | None]
McpVerifier = Callable[[PluginActivationPreflight, Mapping[str, str]], None]


def install_codex_plugin(
    *,
    launch_environment: Mapping[str, str] | None = None,
    data_home: Path | None = None,
    bundle_source: Any | None = None,
    executable_lookup: ExecutableLookup = shutil.which,
    command_runner: CommandRunner = subprocess.run,
    mcp_verifier: McpVerifier | None = None,
    preflight: PluginActivationPreflight | None = None,
) -> CodexPluginInstallation:
    """Install and verify Betterborg's user-scoped Codex plugin.

    Codex's supported plugin commands own marketplace and install state. This
    function owns only its stable materialized marketplace bundle and never
    edits Codex configuration or personal marketplace files directly.
    """

    environment = dict(os.environ if launch_environment is None else launch_environment)
    path = environment.get("PATH", os.defpath)
    codex = executable_lookup("codex", path=path)
    if codex is None:
        return CodexPluginInstallation(
            status=CodexPluginStatus.DEFERRED,
            reason="Codex was not found on the host launch PATH.",
            guidance=(
                "Install Codex, ensure `codex` is on the host launch PATH, and "
                "confirm `codex --version` succeeds."
            ),
        )

    if preflight is None:
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
        root = plugin_data_root(environment, data_home) / "betterborg" / "codex"
        marketplace_path = root / "marketplace"
        owned_bundle_before = ownership(marketplace_path) is not None
        marketplaces = json_plugin_command(
            (str(Path(codex)), "plugin", "marketplace", "list", "--json"),
            environment,
            command_runner,
        )
        registered = marketplace_entry(marketplaces, MARKETPLACE_NAME)
        if registered is not None and not owned_marketplace_source(
            registered, marketplace_path
        ):
            raise PluginCollisionError(
                f"Codex marketplace {MARKETPLACE_NAME!r} is already registered "
                "from a source Betterborg does not own; it was left untouched."
            )
        plugins = json_plugin_command(
            (str(Path(codex)), "plugin", "list", "--available", "--json"),
            environment,
            command_runner,
        )
        before = _plugin_state(plugins)
        if registered is None and before.installed and not owned_bundle_before:
            raise PluginCollisionError(
                f"Codex reports an orphaned {PLUGIN_ID} installation without an "
                "owned Betterborg marketplace bundle; it was left untouched."
            )
        change = materialize_bundle(
            source,
            marketplace_path,
            digest,
            version,
            host_name="Codex",
            manifest_path="plugins/borg/.codex-plugin/plugin.json",
            version_change_name="cache-busting version change",
        )
    except PluginCollisionError as error:
        return CodexPluginInstallation(
            status=CodexPluginStatus.COLLISION,
            preflight=preflight,
            reason=str(error),
        )
    except (OSError, ValueError, PluginCommandError) as error:
        return CodexPluginInstallation(
            status=CodexPluginStatus.FAILED,
            preflight=preflight,
            reason=str(error),
        )

    host_changes = _HostChanges()
    try:
        orphaned_install = registered is None and before.installed
        stale_version = before.installed and before.version != version
        disabled_install = before.installed and not before.enabled
        bundle_version_changed = (
            change.changed
            and change.previous_version is not None
            and change.previous_version != version
        )
        refresh_marketplace = registered is not None and (
            stale_version or bundle_version_changed
        )

        if stale_version or disabled_install or orphaned_install:
            run_plugin_command(
                (str(Path(codex)), "plugin", "remove", PLUGIN_ID, "--json"),
                environment,
                command_runner,
            )
            host_changes.plugin_removed = True
        if refresh_marketplace:
            run_plugin_command(
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
            run_plugin_command(
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
            run_plugin_command(
                (str(Path(codex)), "plugin", "add", PLUGIN_ID, "--json"),
                environment,
                command_runner,
            )
            host_changes.plugin_added = True

        verified_marketplace = marketplace_entry(
            json_plugin_command(
                (str(Path(codex)), "plugin", "marketplace", "list", "--json"),
                environment,
                command_runner,
            ),
            MARKETPLACE_NAME,
        )
        if verified_marketplace is None or not owned_marketplace_source(
            verified_marketplace, marketplace_path
        ):
            raise PluginCommandError(
                f"Codex did not report marketplace {MARKETPLACE_NAME!r} from "
                "the owned Betterborg bundle."
            )
        verified = _plugin_state(
            json_plugin_command(
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
        if not verified.installed or not verified.enabled:
            raise PluginCommandError(
                f"Codex did not report {PLUGIN_ID} as installed and enabled."
            )
        if verified.version != version:
            reported = verified.version or "an unknown version"
            raise PluginCommandError(
                f"Codex reported {PLUGIN_ID} at {reported}, expected {version}."
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


def _plugin_state(value: Any) -> _PluginState:
    installed = False
    enabled = False
    version = None
    for entry in records(value, "plugins"):
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
        entry_enabled = entry.get("enabled")
        if entry_enabled is None:
            entry_enabled = entry.get("status") == "enabled"
        enabled = enabled or bool(entry_enabled)
        for key in ("installedVersion", "installed_version", "version"):
            if isinstance(entry.get(key), str):
                version = entry[key]
                break
    return _PluginState(installed=installed, enabled=enabled, version=version)


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
    if manifest.get("name") != PLUGIN_NAME:
        raise ValueError("plugin name is not stable")
    version = manifest.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("plugin version must be a non-empty string")
    entry = entries[0]
    expected_entry = {
        "name": PLUGIN_NAME,
        "version": version,
        "source": {"source": "local", "path": "./plugins/borg"},
        "policy": {
            "installation": "AVAILABLE",
            "authentication": "ON_INSTALL",
        },
        "category": "Productivity",
    }
    if entry != expected_entry:
        raise ValueError("marketplace Borg entry does not match the Codex contract")
    if manifest.get("skills") != "./skills/":
        raise ValueError("plugin manifest must register bundled skills")
    if manifest.get("mcpServers") != "./.mcp.json":
        raise ValueError("plugin manifest must register the MCP companion file")
    if mcp != {"betterborg": {"command": "betterborg", "args": ["mcp"]}}:
        raise ValueError("MCP registration must execute `betterborg mcp`")
    return bundle_digest(source), version


def _rollback(
    change: BundleChange,
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
            run_plugin_command(
                (codex, "plugin", "remove", PLUGIN_ID, "--json"),
                environment,
                command_runner,
            )
        except PluginCommandError as error:
            errors.append(str(error))
    if host_changes.marketplace_added:
        try:
            run_plugin_command(
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
        except PluginCommandError as error:
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
            run_plugin_command(
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
        except PluginCommandError as error:
            errors.append(str(error))
    if host_changes.plugin_removed:
        try:
            run_plugin_command(
                (codex, "plugin", "add", PLUGIN_ID, "--json"),
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
            (codex, "plugin", "marketplace", "list", "--json"),
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
                    f"Codex no longer reports marketplace {MARKETPLACE_NAME!r}"
                )
            elif not owned_marketplace_source(marketplace_after, change.path):
                errors.append(
                    f"Codex reports marketplace {MARKETPLACE_NAME!r} from an "
                    "unexpected source after rollback"
                )
        elif marketplace_after is not None:
            errors.append(
                f"Codex still reports marketplace {MARKETPLACE_NAME!r} after rollback"
            )

    try:
        plugins = json_plugin_command(
            (codex, "plugin", "list", "--available", "--json"),
            environment,
            command_runner,
        )
    except PluginCommandError as error:
        errors.append(f"could not verify the restored plugin state: {error}")
    else:
        plugin_after = _plugin_state(plugins)
        if plugin_after != plugin_before:
            errors.append(
                f"Codex reports {PLUGIN_ID} after rollback as "
                f"installed={plugin_after.installed}, "
                f"enabled={plugin_after.enabled}, "
                f"version={plugin_after.version!r}; "
                f"expected installed={plugin_before.installed}, "
                f"enabled={plugin_before.enabled}, "
                f"version={plugin_before.version!r}"
            )
    return "; ".join(errors) or None
