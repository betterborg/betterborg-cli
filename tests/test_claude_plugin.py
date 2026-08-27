"""Claude Code marketplace and plugin activation lifecycle."""

from __future__ import annotations

import json
import stat
import subprocess
from importlib import resources
from pathlib import Path

from click.testing import CliRunner

from betterborg_cli import cli as cli_module
from betterborg_cli.claude_plugin import (
    MARKETPLACE_NAME,
    PLUGIN_ID,
    ClaudePluginInstallation,
    ClaudePluginStatus,
    install_claude_plugin,
)
from betterborg_cli.cli import cli


def _executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class _FakeClaude:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.marketplace_source: str | None = None
        self.installed = False
        self.enabled = False
        self.fail_once: tuple[str, ...] | None = None

    def __call__(self, command, **_kwargs):
        call = tuple(command[1:])
        self.calls.append(call)
        if self.fail_once is not None and call == self.fail_once:
            self.fail_once = None
            return subprocess.CompletedProcess(command, 9, "", "injected failure")
        if call == ("--version",):
            return subprocess.CompletedProcess(command, 0, "2.1.0\n", "")
        if call == ("plugin", "marketplace", "list", "--json"):
            entries = []
            if self.marketplace_source is not None:
                entries.append(
                    {"name": MARKETPLACE_NAME, "source": self.marketplace_source}
                )
            return self._json(command, {"marketplaces": entries})
        if call == ("plugin", "list", "--json"):
            entries = []
            if self.installed:
                entries.append({"id": PLUGIN_ID, "enabled": self.enabled})
            return self._json(command, {"plugins": entries})
        if call[:3] == ("plugin", "marketplace", "add"):
            self.marketplace_source = call[3]
        elif call == ("plugin", "install", PLUGIN_ID, "--scope", "user"):
            self.installed = True
            self.enabled = True
        elif call == ("plugin", "enable", PLUGIN_ID, "--scope", "user"):
            self.enabled = True
        elif call == ("plugin", "disable", PLUGIN_ID, "--scope", "user"):
            self.enabled = False
        elif call in {
            ("plugin", "marketplace", "update", MARKETPLACE_NAME),
            ("plugin", "update", PLUGIN_ID, "--scope", "user"),
        }:
            pass
        else:
            raise AssertionError(f"unexpected Claude command: {call}")
        return subprocess.CompletedProcess(command, 0, "ok\n", "")

    @staticmethod
    def _json(command, value):
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")


def _host(tmp_path: Path, fake: _FakeClaude):
    bin_dir = tmp_path / "host-bin"
    borg = _executable(bin_dir / "borg", "printf 'borg 0.1.0\\n'")
    claude = _executable(bin_dir / "claude", "exit 0")

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
    result = install_claude_plugin(
        launch_environment=environment,
        data_home=tmp_path / "data",
        executable_lookup=lookup,
        command_runner=fake,
        mcp_verifier=verify,
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


def test_owned_upgrade_updates_claude_and_retains_previous_bundle(
    tmp_path: Path,
) -> None:
    fake = _FakeClaude()
    first, _ = _install(tmp_path, fake)
    source = resources.files("betterborg_cli.claude_plugin_bundle") / "marketplace"
    upgraded = tmp_path / "upgraded-marketplace"
    _copy_resource(source, upgraded)
    manifest = upgraded / "plugins/borg/.claude-plugin/plugin.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["version"] = "0.2.0"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    fake.calls.clear()

    result, _ = _install(tmp_path, fake, bundle_source=upgraded)

    assert first.bundle_path is not None
    assert result.status is ClaudePluginStatus.INSTALLED
    assert result.previous_bundle is not None
    assert result.previous_bundle.is_dir()
    old_manifest = result.previous_bundle / "plugins/borg/.claude-plugin/plugin.json"
    assert json.loads(old_manifest.read_text(encoding="utf-8"))["version"] == "0.1.0"
    current_manifest = result.bundle_path / "plugins/borg/.claude-plugin/plugin.json"
    assert json.loads(current_manifest.read_text(encoding="utf-8"))["version"] == (
        "0.2.0"
    )
    assert ("plugin", "marketplace", "update", MARKETPLACE_NAME) in fake.calls
    assert ("plugin", "update", PLUGIN_ID, "--scope", "user") in fake.calls


def test_foreign_marketplace_collision_is_left_untouched(tmp_path: Path) -> None:
    fake = _FakeClaude()
    fake.marketplace_source = str(tmp_path / "someone-elses-marketplace")

    result, spawns = _install(tmp_path, fake)

    assert result.status is ClaudePluginStatus.COLLISION
    assert "left untouched" in (result.reason or "")
    assert not tmp_path.joinpath("data").exists()
    assert fake.marketplace_source == str(tmp_path / "someone-elses-marketplace")
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
    _copy_resource(source, upgraded)
    command = upgraded / "plugins/borg/commands/borg.md"
    command.write_text(command.read_text() + "\nUpgrade content.\n")
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

    assert result.status is ClaudePluginStatus.SETUP_REQUIRED
    assert "Install Claude Code" in (result.guidance or "")
    assert commands == []
    assert spawns == []
    assert not tmp_path.joinpath("data").exists()


def test_missing_persistent_borg_does_not_list_or_mutate_claude(
    tmp_path: Path,
) -> None:
    fake = _FakeClaude()
    claude = _executable(tmp_path / "bin/claude", "exit 0")

    def lookup(name: str, **_kwargs):
        return str(claude) if name == "claude" else None

    result = install_claude_plugin(
        launch_environment={"PATH": str(claude.parent), "HOME": str(tmp_path)},
        data_home=tmp_path / "data",
        executable_lookup=lookup,
        command_runner=fake,
    )

    assert result.status is ClaudePluginStatus.SETUP_REQUIRED
    assert "uv tool install betterborg-cli" in (result.guidance or "")
    assert fake.calls == [("--version",)]
    assert not tmp_path.joinpath("data").exists()


def test_cli_reports_success_and_reload_guidance(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_module,
        "install_claude_plugin",
        lambda: ClaudePluginInstallation(
            status=ClaudePluginStatus.INSTALLED,
            reload_guidance="Run /reload-plugins now.",
        ),
    )

    result = CliRunner().invoke(cli, ["plugin", "install", "claude"])

    assert result.exit_code == 0
    assert "Installed and enabled" in result.output
    assert "/reload-plugins" in result.output


def _copy_resource(source, destination: Path) -> None:
    destination.mkdir()
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            _copy_resource(child, target)
        else:
            target.write_bytes(child.read_bytes())
