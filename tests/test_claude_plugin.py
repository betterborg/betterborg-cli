"""Claude Code marketplace and plugin activation lifecycle."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib import resources
from pathlib import Path

import pytest
from plugin_test_support import copy_resource, executable

from betterborg_cli import __version__
from betterborg_cli.claude_plugin import (
    MARKETPLACE_NAME,
    PLUGIN_ID,
    ClaudePluginStatus,
    install_claude_plugin,
    verify_borg_mcp,
)
from betterborg_cli.plugin_activation import (
    PluginActivationPreflight,
    PluginActivationStatus,
    PluginActivationVerificationError,
)


class _FakeClaude:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.version = "2.1.212 (Claude Code)"
        self.marketplace_source: str | None = None
        self.installed = False
        self.enabled = False
        self.installed_version: str | None = None
        self.retain_version_on_update = False
        self.other_installations: list[dict[str, object]] = []
        self.fail_once: tuple[str, ...] | None = None

    def __call__(self, command, **_kwargs):
        call = tuple(command[1:])
        self.calls.append(call)
        if self.fail_once is not None and call == self.fail_once:
            self.fail_once = None
            return subprocess.CompletedProcess(command, 9, "", "injected failure")
        if call == ("--version",):
            return subprocess.CompletedProcess(command, 0, self.version + "\n", "")
        if call == ("plugin", "marketplace", "list", "--json"):
            entries = []
            if self.marketplace_source is not None:
                entries.append(
                    {"name": MARKETPLACE_NAME, "source": self.marketplace_source}
                )
            return self._json(command, {"marketplaces": entries})
        if call == ("plugin", "list", "--json"):
            entries = list(self.other_installations)
            if self.installed:
                entries.append(
                    {
                        "id": PLUGIN_ID,
                        "version": self.installed_version,
                        "scope": "user",
                        "enabled": self.enabled,
                    }
                )
            return self._json(command, {"plugins": entries})
        if call[:3] == ("plugin", "marketplace", "add"):
            self.marketplace_source = call[3]
        elif call == ("plugin", "install", PLUGIN_ID, "--scope", "user"):
            self.installed = True
            self.enabled = True
            self.installed_version = self._available_version()
        elif call == ("plugin", "enable", PLUGIN_ID, "--scope", "user"):
            self.enabled = True
        elif call == ("plugin", "disable", PLUGIN_ID, "--scope", "user"):
            self.enabled = False
        elif call == ("plugin", "uninstall", PLUGIN_ID, "--scope", "user"):
            self.installed = False
            self.enabled = False
            self.installed_version = None
        elif call == (
            "plugin",
            "marketplace",
            "remove",
            MARKETPLACE_NAME,
            "--scope",
            "user",
        ):
            self.marketplace_source = None
            self.installed = False
            self.enabled = False
            self.installed_version = None
        elif call == ("plugin", "marketplace", "update", MARKETPLACE_NAME):
            pass
        elif call == ("plugin", "update", PLUGIN_ID, "--scope", "user"):
            available = self._available_version()
            if (
                available != self.installed_version
                and not self.retain_version_on_update
            ):
                self.installed_version = available
        else:
            raise AssertionError(f"unexpected Claude command: {call}")
        return subprocess.CompletedProcess(command, 0, "ok\n", "")

    @staticmethod
    def _json(command, value):
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")

    def _available_version(self) -> str:
        assert self.marketplace_source is not None
        manifest = (
            Path(self.marketplace_source)
            / "plugins/borg/.claude-plugin/plugin.json"
        )
        return json.loads(manifest.read_text(encoding="utf-8"))["version"]


def _host(tmp_path: Path, fake: _FakeClaude):
    bin_dir = tmp_path / "host-bin"
    borg = executable(bin_dir / "borg", f"printf 'borg {__version__}\\n'")
    claude = executable(bin_dir / "claude", "exit 0")

    def lookup(name: str, *, path: str):
        assert path == str(bin_dir)
        return str({"borg": borg, "claude": claude}[name])

    environment = {"PATH": str(bin_dir), "HOME": str(tmp_path / "home")}
    spawns: list[Path] = []

    def verify(preflight, received_environment):
        assert received_environment == environment
        assert preflight.executable == borg.resolve()
        spawns.append(preflight.executable)

    return environment, lookup, verify, spawns


def _install(tmp_path: Path, fake: _FakeClaude, **kwargs):
    environment, lookup, verify, spawns = _host(tmp_path, fake)
    mcp_verifier = kwargs.pop("mcp_verifier", verify)
    result = install_claude_plugin(
        launch_environment=environment,
        data_home=tmp_path / "data",
        executable_lookup=lookup,
        command_runner=fake,
        mcp_verifier=mcp_verifier,
        **kwargs,
    )
    return result, spawns


def test_fresh_activation_materializes_registers_enables_and_spawns_mcp(
    tmp_path: Path,
) -> None:
    fake = _FakeClaude()

    result, spawns = _install(tmp_path, fake)

    assert result.status is ClaudePluginStatus.INSTALLED
    assert result.bundle_path == tmp_path / "data/betterborg/claude/marketplace"
    marketplace = result.bundle_path / ".claude-plugin/marketplace.json"
    assert json.loads(marketplace.read_text())["name"] == "betterborg"
    mcp_config = result.bundle_path / "plugins/borg/.mcp.json"
    assert json.loads(mcp_config.read_text()) == {
        "mcpServers": {"borg": {"command": "borg", "args": ["mcp"]}}
    }
    assert fake.installed is True
    assert fake.enabled is True
    assert (
        "plugin",
        "marketplace",
        "add",
        str(result.bundle_path),
        "--scope",
        "user",
    ) in fake.calls
    assert ("plugin", "install", PLUGIN_ID, "--scope", "user") in fake.calls
    assert ("plugin", "enable", PLUGIN_ID, "--scope", "user") in fake.calls
    assert len(spawns) == 1
    assert result.reload_guidance is not None
    assert "/reload-plugins" in result.reload_guidance


def test_native_windows_user_profile_selects_owned_data_root(tmp_path: Path) -> None:
    fake = _FakeClaude()
    environment, lookup, verify, spawns = _host(tmp_path, fake)
    profile = tmp_path / "windows-profile"
    environment.pop("HOME")
    environment["USERPROFILE"] = str(profile)

    result = install_claude_plugin(
        launch_environment=environment,
        executable_lookup=lookup,
        command_runner=fake,
        mcp_verifier=verify,
    )

    expected = profile / ".local/share/betterborg/claude/marketplace"
    assert result.status is ClaudePluginStatus.INSTALLED
    assert result.bundle_path == expected
    assert fake.marketplace_source == str(expected)
    assert len(spawns) == 1


def test_reinstall_is_a_verified_no_op(tmp_path: Path) -> None:
    fake = _FakeClaude()
    first, _ = _install(tmp_path, fake)
    fake.calls.clear()

    second, spawns = _install(tmp_path, fake)

    assert first.status is ClaudePluginStatus.INSTALLED
    assert second.status is ClaudePluginStatus.UNCHANGED
    assert fake.calls == [
        ("--version",),
        ("plugin", "marketplace", "list", "--json"),
        ("plugin", "list", "--json"),
        ("plugin", "list", "--json"),
    ]
    assert len(spawns) == 1


@pytest.mark.parametrize("scope", ["project", "local"])
def test_other_scope_installation_does_not_replace_user_installation(
    tmp_path: Path,
    scope: str,
) -> None:
    fake = _FakeClaude()
    fake.other_installations = [
        {"id": PLUGIN_ID, "scope": scope, "enabled": True}
    ]

    result, _ = _install(tmp_path, fake)

    assert result.status is ClaudePluginStatus.INSTALLED
    assert ("plugin", "install", PLUGIN_ID, "--scope", "user") in fake.calls
    assert fake.installed is True


def test_mixed_scope_installations_reconcile_user_enabled_state(
    tmp_path: Path,
) -> None:
    fake = _FakeClaude()
    first, _ = _install(tmp_path, fake)
    fake.other_installations = [
        {"id": PLUGIN_ID, "scope": "project", "enabled": True}
    ]
    fake.enabled = False
    fake.calls.clear()

    result, _ = _install(tmp_path, fake)

    assert first.status is ClaudePluginStatus.INSTALLED
    assert result.status is ClaudePluginStatus.INSTALLED
    assert ("plugin", "install", PLUGIN_ID, "--scope", "user") not in fake.calls
    assert ("plugin", "enable", PLUGIN_ID, "--scope", "user") in fake.calls
    assert fake.enabled is True


def test_verify_borg_mcp_spawns_installed_cli() -> None:
    executable = Path(sys.executable).with_name("borg").resolve(strict=True)
    preflight = PluginActivationPreflight(
        status=PluginActivationStatus.READY,
        executable=executable,
        version="borg test",
    )

    verify_borg_mcp(preflight, os.environ)


def test_owned_upgrade_updates_claude_and_retains_previous_bundle(
    tmp_path: Path,
) -> None:
    fake = _FakeClaude()
    first, _ = _install(tmp_path, fake)
    source = resources.files("betterborg_cli.claude_plugin_bundle") / "marketplace"
    upgraded = tmp_path / "upgraded-marketplace"
    copy_resource(source, upgraded)
    manifest = upgraded / "plugins/borg/.claude-plugin/plugin.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["version"] = "0.3.0"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    fake.calls.clear()

    result, _ = _install(tmp_path, fake, bundle_source=upgraded)

    assert first.bundle_path is not None
    assert result.status is ClaudePluginStatus.INSTALLED
    assert result.previous_bundle is not None
    assert result.previous_bundle.is_dir()
    old_manifest = result.previous_bundle / "plugins/borg/.claude-plugin/plugin.json"
    assert (
        json.loads(old_manifest.read_text(encoding="utf-8"))["version"]
        == __version__
    )
    current_manifest = result.bundle_path / "plugins/borg/.claude-plugin/plugin.json"
    assert json.loads(current_manifest.read_text(encoding="utf-8"))["version"] == (
        "0.3.0"
    )
    assert ("plugin", "marketplace", "update", MARKETPLACE_NAME) in fake.calls
    assert ("plugin", "update", PLUGIN_ID, "--scope", "user") in fake.calls


def test_owned_bundle_change_requires_a_new_plugin_version(tmp_path: Path) -> None:
    fake = _FakeClaude()
    first, _ = _install(tmp_path, fake)
    assert first.bundle_path is not None
    original_command = first.bundle_path.joinpath(
        "plugins/borg/commands/borg.md"
    ).read_text(encoding="utf-8")
    source = resources.files("betterborg_cli.claude_plugin_bundle") / "marketplace"
    changed = tmp_path / "same-version-marketplace"
    copy_resource(source, changed)
    command = changed / "plugins/borg/commands/borg.md"
    command.write_text(command.read_text() + "\nChanged content.\n")
    fake.calls.clear()

    result, spawns = _install(tmp_path, fake, bundle_source=changed)

    assert result.status is ClaudePluginStatus.FAILED
    assert f"without a version bump from {__version__}" in (result.reason or "")
    assert first.bundle_path.joinpath("plugins/borg/commands/borg.md").read_text(
        encoding="utf-8"
    ) == original_command
    assert ("plugin", "marketplace", "update", MARKETPLACE_NAME) not in fake.calls
    assert ("plugin", "update", PLUGIN_ID, "--scope", "user") not in fake.calls
    assert spawns == []


def test_owned_upgrade_requires_claude_to_report_the_new_version(
    tmp_path: Path,
) -> None:
    fake = _FakeClaude()
    first, _ = _install(tmp_path, fake)
    assert first.bundle_path is not None
    source = resources.files("betterborg_cli.claude_plugin_bundle") / "marketplace"
    upgraded = tmp_path / "uncached-upgrade-marketplace"
    copy_resource(source, upgraded)
    manifest = upgraded / "plugins/borg/.claude-plugin/plugin.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["version"] = "0.3.0"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    fake.retain_version_on_update = True
    fake.calls.clear()

    result, spawns = _install(tmp_path, fake, bundle_source=upgraded)

    assert result.status is ClaudePluginStatus.FAILED
    assert f"reported borg@betterborg at {__version__}, expected 0.3.0" in (
        result.reason or ""
    )
    restored_manifest = first.bundle_path / "plugins/borg/.claude-plugin/plugin.json"
    assert (
        json.loads(restored_manifest.read_text(encoding="utf-8"))["version"]
        == __version__
    )
    assert fake.installed_version == __version__
    assert spawns == []


@pytest.mark.parametrize("damage", ["changed", "deleted"])
def test_reinstall_repairs_materialized_bundle_contents(
    tmp_path: Path,
    damage: str,
) -> None:
    fake = _FakeClaude()
    first, _ = _install(tmp_path, fake)
    assert first.bundle_path is not None
    command = first.bundle_path / "plugins/borg/commands/borg.md"
    expected = command.read_text(encoding="utf-8")
    if damage == "changed":
        command.write_text("altered\n", encoding="utf-8")
    else:
        command.unlink()
    fake.calls.clear()

    result, spawns = _install(tmp_path, fake)

    assert result.status is ClaudePluginStatus.INSTALLED
    assert command.read_text(encoding="utf-8") == expected
    assert ("plugin", "marketplace", "update", MARKETPLACE_NAME) in fake.calls
    assert ("plugin", "update", PLUGIN_ID, "--scope", "user") in fake.calls
    assert fake.installed_version == __version__
    assert len(spawns) == 1


def test_foreign_marketplace_collision_is_left_untouched(tmp_path: Path) -> None:
    fake = _FakeClaude()
    fake.marketplace_source = str(tmp_path / "someone-elses-marketplace")

    result, spawns = _install(tmp_path, fake)

    assert result.status is ClaudePluginStatus.COLLISION
    assert "left untouched" in (result.reason or "")
    assert not tmp_path.joinpath("data").exists()
    assert fake.marketplace_source == str(tmp_path / "someone-elses-marketplace")
    assert spawns == []


def test_affected_claude_version_leaves_foreign_same_named_plugin_untouched(
    tmp_path: Path,
) -> None:
    fake = _FakeClaude()
    fake.version = "2.1.211 (Claude Code)"
    foreign = {
        "id": "borg@foreign-marketplace",
        "scope": "user",
        "enabled": True,
    }
    fake.other_installations = [foreign]

    result, spawns = _install(tmp_path, fake)

    assert result.status is ClaudePluginStatus.SETUP_REQUIRED
    assert "collision-safe plugin rollback" in (result.reason or "")
    assert "2.1.212 or newer" in (result.guidance or "")
    assert fake.calls == [("--version",)]
    assert fake.other_installations == [foreign]
    assert not tmp_path.joinpath("data").exists()
    assert spawns == []


def test_failed_owned_upgrade_restores_previous_bundle_and_plugin_state(
    tmp_path: Path,
) -> None:
    fake = _FakeClaude()
    first, _ = _install(tmp_path, fake)
    assert first.bundle_path is not None
    original_marker = first.bundle_path.joinpath(".betterborg-owned.json").read_text()
    source = resources.files("betterborg_cli.claude_plugin_bundle") / "marketplace"
    upgraded = tmp_path / "broken-upgrade"
    copy_resource(source, upgraded)
    command = upgraded / "plugins/borg/commands/borg.md"
    command.write_text(command.read_text() + "\nUpgrade content.\n")
    manifest = upgraded / "plugins/borg/.claude-plugin/plugin.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["version"] = "0.3.0"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    fake.enabled = False
    fake.fail_once = ("plugin", "update", PLUGIN_ID, "--scope", "user")

    result, spawns = _install(tmp_path, fake, bundle_source=upgraded)

    assert result.status is ClaudePluginStatus.FAILED
    assert "rolled back" in (result.reason or "")
    restored_marker = first.bundle_path / ".betterborg-owned.json"
    assert restored_marker.read_text() == original_marker
    assert fake.installed is True
    assert fake.enabled is False
    assert ("plugin", "disable", PLUGIN_ID, "--scope", "user") in fake.calls
    assert spawns == []
    assert list(first.bundle_path.parent.joinpath("backups").glob("failed-*"))


def test_failed_upgrade_reports_no_op_plugin_rollback(tmp_path: Path) -> None:
    fake = _FakeClaude()
    first, _ = _install(tmp_path, fake)
    assert first.bundle_path is not None
    source = resources.files("betterborg_cli.claude_plugin_bundle") / "marketplace"
    upgraded = tmp_path / "rollback-no-op"
    copy_resource(source, upgraded)
    manifest = upgraded / "plugins/borg/.claude-plugin/plugin.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["version"] = "0.3.0"
    manifest.write_text(json.dumps(value), encoding="utf-8")

    def fail_verification(*_args) -> None:
        fake.retain_version_on_update = True
        raise ValueError("injected MCP failure")

    result, spawns = _install(
        tmp_path,
        fake,
        bundle_source=upgraded,
        mcp_verifier=fail_verification,
    )

    assert result.status is ClaudePluginStatus.FAILED
    assert "Rollback also failed" in (result.reason or "")
    assert "version='0.3.0'; expected" in (result.reason or "")
    assert f"version='{__version__}'" in (result.reason or "")
    restored_manifest = first.bundle_path / "plugins/borg/.claude-plugin/plugin.json"
    assert (
        json.loads(restored_manifest.read_text(encoding="utf-8"))["version"]
        == __version__
    )
    assert fake.installed_version == "0.3.0"
    assert spawns == []


def test_failed_bundle_promotion_immediately_restores_previous_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeClaude()
    first, _ = _install(tmp_path, fake)
    assert first.bundle_path is not None
    original_marker = first.bundle_path.joinpath(".betterborg-owned.json").read_text()
    source = resources.files("betterborg_cli.claude_plugin_bundle") / "marketplace"
    upgraded = tmp_path / "promotion-failure"
    copy_resource(source, upgraded)
    manifest = upgraded / "plugins/borg/.claude-plugin/plugin.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["version"] = "0.3.0"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    original_rename = Path.rename

    def fail_staging_promotion(path: Path, target: Path) -> Path:
        if path.name.startswith(".marketplace-staging-"):
            raise OSError("injected staging promotion failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_staging_promotion)
    fake.calls.clear()

    result, spawns = _install(tmp_path, fake, bundle_source=upgraded)

    assert result.status is ClaudePluginStatus.FAILED
    assert "injected staging promotion failure" in (result.reason or "")
    assert first.bundle_path.joinpath(".betterborg-owned.json").read_text() == (
        original_marker
    )
    restored_manifest = first.bundle_path / "plugins/borg/.claude-plugin/plugin.json"
    assert (
        json.loads(restored_manifest.read_text(encoding="utf-8"))["version"]
        == __version__
    )
    assert fake.marketplace_source == str(first.bundle_path)
    assert fake.installed_version == __version__
    assert ("plugin", "marketplace", "update", MARKETPLACE_NAME) not in fake.calls
    assert ("plugin", "update", PLUGIN_ID, "--scope", "user") not in fake.calls
    assert spawns == []


def test_failed_fresh_activation_removes_host_and_bundle_state(
    tmp_path: Path,
) -> None:
    fake = _FakeClaude()

    def fail_verification(*_args) -> None:
        raise PluginActivationVerificationError("injected MCP failure")

    result, spawns = _install(
        tmp_path,
        fake,
        mcp_verifier=fail_verification,
    )

    bundle = tmp_path / "data/betterborg/claude/marketplace"
    assert result.status is ClaudePluginStatus.FAILED
    assert "rolled back" in (result.reason or "")
    assert result.bundle_path is None
    assert fake.marketplace_source is None
    assert fake.installed is False
    assert fake.enabled is False
    assert ("plugin", "uninstall", PLUGIN_ID, "--scope", "user") in fake.calls
    assert (
        "plugin",
        "marketplace",
        "remove",
        MARKETPLACE_NAME,
        "--scope",
        "user",
    ) in fake.calls
    assert not bundle.exists()
    assert not tmp_path.joinpath("data").exists()
    assert spawns == []


def test_failed_enable_repair_restores_disabled_state(tmp_path: Path) -> None:
    fake = _FakeClaude()
    first, _ = _install(tmp_path, fake)
    assert first.bundle_path is not None
    original_marker = first.bundle_path.joinpath(".betterborg-owned.json").read_text()
    fake.enabled = False
    fake.calls.clear()

    def fail_verification(*_args) -> None:
        raise ValueError("injected MCP failure")

    result, spawns = _install(
        tmp_path,
        fake,
        mcp_verifier=fail_verification,
    )

    assert result.status is ClaudePluginStatus.FAILED
    assert "rolled back" in (result.reason or "")
    assert fake.marketplace_source == str(first.bundle_path)
    assert fake.installed is True
    assert fake.enabled is False
    assert ("plugin", "disable", PLUGIN_ID, "--scope", "user") in fake.calls
    assert first.bundle_path.joinpath(".betterborg-owned.json").read_text() == (
        original_marker
    )
    assert spawns == []


def test_failed_missing_plugin_repair_uninstalls_only_new_user_plugin(
    tmp_path: Path,
) -> None:
    fake = _FakeClaude()
    first, _ = _install(tmp_path, fake)
    assert first.bundle_path is not None
    fake.installed = False
    fake.enabled = False
    fake.calls.clear()

    def fail_verification(*_args) -> None:
        raise ValueError("injected MCP failure")

    result, spawns = _install(
        tmp_path,
        fake,
        mcp_verifier=fail_verification,
    )

    assert result.status is ClaudePluginStatus.FAILED
    assert fake.marketplace_source == str(first.bundle_path)
    assert fake.installed is False
    assert fake.enabled is False
    assert ("plugin", "uninstall", PLUGIN_ID, "--scope", "user") in fake.calls
    assert (
        "plugin",
        "marketplace",
        "remove",
        MARKETPLACE_NAME,
        "--scope",
        "user",
    ) not in fake.calls
    assert first.bundle_path.is_dir()
    assert spawns == []


def test_absent_claude_returns_guidance_without_files_or_mcp_spawn(
    tmp_path: Path,
) -> None:
    commands: list[object] = []
    spawns: list[object] = []

    result = install_claude_plugin(
        launch_environment={"PATH": str(tmp_path / "empty"), "HOME": str(tmp_path)},
        data_home=tmp_path / "data",
        executable_lookup=lambda *_args, **_kwargs: None,
        command_runner=lambda *args, **kwargs: commands.append((args, kwargs)),
        mcp_verifier=lambda *args: spawns.append(args),
    )

    assert result.status is ClaudePluginStatus.DEFERRED
    assert "Install Claude Code" in (result.guidance or "")
    assert commands == []
    assert spawns == []
    assert not tmp_path.joinpath("data").exists()


def test_missing_persistent_borg_does_not_list_or_mutate_claude(
    tmp_path: Path,
) -> None:
    fake = _FakeClaude()
    claude = executable(tmp_path / "bin/claude", "exit 0")

    def lookup(name: str, **_kwargs):
        return str(claude) if name == "claude" else None

    result = install_claude_plugin(
        launch_environment={"PATH": str(claude.parent), "HOME": str(tmp_path)},
        data_home=tmp_path / "data",
        executable_lookup=lookup,
        command_runner=fake,
    )

    assert result.status is ClaudePluginStatus.SETUP_REQUIRED
    assert "`uv tool install betterborg`" in (result.guidance or "")
    assert fake.calls == [("--version",)]
    assert not tmp_path.joinpath("data").exists()
