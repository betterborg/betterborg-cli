"""Codex marketplace and plugin activation lifecycle."""

from __future__ import annotations

import json
import stat
import subprocess
from importlib import resources
from pathlib import Path

from click.testing import CliRunner

from betterborg_cli import cli as cli_module
from betterborg_cli.cli import cli
from betterborg_cli.codex_plugin import (
    MARKETPLACE_NAME,
    PLUGIN_ID,
    CodexPluginInstallation,
    CodexPluginStatus,
    install_codex_plugin,
)


def _executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class _FakeCodex:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.marketplace_source: str | None = None
        self.installed = False
        self.installed_version: str | None = None
        self.other_marketplaces: list[dict[str, object]] = []
        self.other_plugins: list[dict[str, object]] = []
        self.fail_once: tuple[str, ...] | None = None

    def __call__(self, command, **_kwargs):
        call = tuple(command[1:])
        self.calls.append(call)
        if self.fail_once is not None and call == self.fail_once:
            self.fail_once = None
            return subprocess.CompletedProcess(command, 9, "", "injected failure")
        if call == ("plugin", "marketplace", "list", "--json"):
            entries = list(self.other_marketplaces)
            if self.marketplace_source is not None:
                entries.append(
                    {
                        "name": MARKETPLACE_NAME,
                        "root": self.marketplace_source,
                        "marketplaceSource": {
                            "sourceType": "local",
                            "source": self.marketplace_source,
                        },
                    }
                )
            return self._json(command, {"marketplaces": entries})
        if call == ("plugin", "list", "--available", "--json"):
            installed = list(self.other_plugins)
            available = []
            if self.installed:
                installed.append(
                    {
                        "pluginId": PLUGIN_ID,
                        "name": "borg",
                        "marketplaceName": MARKETPLACE_NAME,
                        "version": self.installed_version,
                        "installed": True,
                        "enabled": True,
                    }
                )
            elif self.marketplace_source is not None:
                available.append(
                    {
                        "pluginId": PLUGIN_ID,
                        "name": "borg",
                        "marketplaceName": MARKETPLACE_NAME,
                        "version": self._available_version(),
                        "installed": False,
                        "enabled": False,
                    }
                )
            return self._json(
                command, {"installed": installed, "available": available}
            )
        if call[:3] == ("plugin", "marketplace", "add"):
            self.marketplace_source = call[3]
        elif call == ("plugin", "add", PLUGIN_ID, "--json"):
            self.installed = True
            self.installed_version = self._available_version()
        elif call == ("plugin", "remove", PLUGIN_ID, "--json"):
            self.installed = False
            self.installed_version = None
        elif call == (
            "plugin",
            "marketplace",
            "remove",
            MARKETPLACE_NAME,
            "--json",
        ):
            self.marketplace_source = None
        else:
            raise AssertionError(f"unexpected Codex command: {call}")
        return self._json(command, {"ok": True})

    @staticmethod
    def _json(command, value):
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")

    def _available_version(self) -> str:
        assert self.marketplace_source is not None
        manifest = (
            Path(self.marketplace_source)
            / "plugins/borg/.codex-plugin/plugin.json"
        )
        return json.loads(manifest.read_text(encoding="utf-8"))["version"]


def _host(tmp_path: Path):
    bin_dir = tmp_path / "host-bin"
    borg = _executable(bin_dir / "borg", "printf 'borg 0.1.0\\n'")
    codex = _executable(bin_dir / "codex", "exit 0")

    def lookup(name: str, *, path: str):
        assert path == str(bin_dir)
        return str({"borg": borg, "codex": codex}[name])

    environment = {"PATH": str(bin_dir), "HOME": str(tmp_path / "home")}
    spawns: list[Path] = []

    def verify(preflight, received_environment):
        assert received_environment == environment
        assert preflight.executable == borg.resolve()
        spawns.append(preflight.executable)

    return environment, lookup, verify, spawns


def _install(tmp_path: Path, fake: _FakeCodex, **kwargs):
    environment, lookup, verify, spawns = _host(tmp_path)
    mcp_verifier = kwargs.pop("mcp_verifier", verify)
    result = install_codex_plugin(
        launch_environment=environment,
        data_home=tmp_path / "data",
        executable_lookup=lookup,
        command_runner=fake,
        mcp_verifier=mcp_verifier,
        **kwargs,
    )
    return result, spawns


def test_fresh_activation_materializes_registers_installs_and_discovers_mcp(
    tmp_path: Path,
) -> None:
    fake = _FakeCodex()

    result, spawns = _install(tmp_path, fake)

    expected = tmp_path / "data/betterborg/codex/marketplace"
    assert result.status is CodexPluginStatus.INSTALLED
    assert result.bundle_path == expected
    marketplace = json.loads(
        expected.joinpath(".agents/plugins/marketplace.json").read_text()
    )
    assert marketplace["name"] == MARKETPLACE_NAME
    assert marketplace["plugins"][0]["source"] == {
        "source": "local",
        "path": "./plugins/borg",
    }
    manifest = json.loads(
        expected.joinpath(
            "plugins/borg/.codex-plugin/plugin.json"
        ).read_text()
    )
    assert manifest["mcpServers"] == "./.mcp.json"
    assert manifest["skills"] == "./skills/"
    mcp = json.loads(expected.joinpath("plugins/borg/.mcp.json").read_text())
    assert mcp == {"borg": {"command": "borg", "args": ["mcp"]}}
    assert (
        "plugin",
        "marketplace",
        "add",
        str(expected),
        "--json",
    ) in fake.calls
    assert ("plugin", "add", PLUGIN_ID, "--json") in fake.calls
    assert fake.installed is True
    assert fake.installed_version == "0.1.0"
    assert spawns == [Path(result.preflight.executable)]
    assert "new Codex thread" in (result.new_thread_guidance or "")


def test_reinstall_is_a_verified_no_op(tmp_path: Path) -> None:
    fake = _FakeCodex()
    first, _ = _install(tmp_path, fake)
    fake.calls.clear()

    second, spawns = _install(tmp_path, fake)

    assert first.status is CodexPluginStatus.INSTALLED
    assert second.status is CodexPluginStatus.UNCHANGED
    assert fake.calls == [
        ("plugin", "marketplace", "list", "--json"),
        ("plugin", "list", "--available", "--json"),
        ("plugin", "marketplace", "list", "--json"),
        ("plugin", "list", "--available", "--json"),
    ]
    assert len(spawns) == 1


def test_cachebuster_upgrade_remove_adds_and_retains_previous_bundle(
    tmp_path: Path,
) -> None:
    fake = _FakeCodex()
    first, _ = _install(tmp_path, fake)
    assert first.bundle_path is not None
    upgraded = _upgraded_bundle(tmp_path, "upgrade", "0.1.0+codex.next")
    unrelated_marketplace = {
        "name": "team-tools",
        "root": str(tmp_path / "team-tools"),
    }
    unrelated_plugin = {
        "pluginId": "helper@team-tools",
        "name": "helper",
        "marketplaceName": "team-tools",
        "version": "2.0.0",
        "installed": True,
        "enabled": True,
    }
    fake.other_marketplaces = [unrelated_marketplace]
    fake.other_plugins = [unrelated_plugin]
    fake.calls.clear()

    result, spawns = _install(tmp_path, fake, bundle_source=upgraded)

    assert result.status is CodexPluginStatus.INSTALLED
    assert result.previous_bundle is not None
    assert result.previous_bundle.is_dir()
    previous_manifest = result.previous_bundle / (
        "plugins/borg/.codex-plugin/plugin.json"
    )
    assert json.loads(previous_manifest.read_text())["version"] == "0.1.0"
    assert fake.installed_version == "0.1.0+codex.next"
    assert ("plugin", "remove", PLUGIN_ID, "--json") in fake.calls
    assert (
        "plugin",
        "marketplace",
        "remove",
        MARKETPLACE_NAME,
        "--json",
    ) in fake.calls
    assert ("plugin", "add", PLUGIN_ID, "--json") in fake.calls
    assert fake.other_marketplaces == [unrelated_marketplace]
    assert fake.other_plugins == [unrelated_plugin]
    assert len(spawns) == 1


def test_owned_stale_install_is_recovered_with_remove_add(tmp_path: Path) -> None:
    fake = _FakeCodex()
    first, _ = _install(tmp_path, fake)
    assert first.bundle_path is not None
    fake.installed_version = "0.0.9"
    fake.calls.clear()

    result, _ = _install(tmp_path, fake)

    assert result.status is CodexPluginStatus.INSTALLED
    assert fake.installed_version == "0.1.0"
    assert ("plugin", "remove", PLUGIN_ID, "--json") in fake.calls
    assert (
        "plugin",
        "marketplace",
        "remove",
        MARKETPLACE_NAME,
        "--json",
    ) in fake.calls
    assert ("plugin", "add", PLUGIN_ID, "--json") in fake.calls


def test_failed_upgrade_restores_prior_bundle_and_host_state(
    tmp_path: Path,
) -> None:
    fake = _FakeCodex()
    first, _ = _install(tmp_path, fake)
    assert first.bundle_path is not None
    upgraded = _upgraded_bundle(tmp_path, "broken", "0.1.0+codex.broken")

    def fail_verification(*_args) -> None:
        raise ValueError("injected MCP failure")

    result, spawns = _install(
        tmp_path,
        fake,
        bundle_source=upgraded,
        mcp_verifier=fail_verification,
    )

    assert result.status is CodexPluginStatus.FAILED
    assert "rolled back" in (result.reason or "")
    restored_manifest = first.bundle_path / (
        "plugins/borg/.codex-plugin/plugin.json"
    )
    assert json.loads(restored_manifest.read_text())["version"] == "0.1.0"
    assert fake.marketplace_source == str(first.bundle_path)
    assert fake.installed is True
    assert fake.installed_version == "0.1.0"
    assert list(first.bundle_path.parent.joinpath("backups").glob("failed-*"))
    assert spawns == []


def test_foreign_marketplace_collision_is_left_untouched(tmp_path: Path) -> None:
    fake = _FakeCodex()
    foreign = str(tmp_path / "someone-elses-marketplace")
    fake.marketplace_source = foreign

    result, spawns = _install(tmp_path, fake)

    assert result.status is CodexPluginStatus.COLLISION
    assert "left untouched" in (result.reason or "")
    assert fake.marketplace_source == foreign
    assert fake.calls == [("plugin", "marketplace", "list", "--json")]
    assert not tmp_path.joinpath("data").exists()
    assert spawns == []


def test_foreign_bundle_path_collision_is_left_untouched(tmp_path: Path) -> None:
    fake = _FakeCodex()
    bundle = tmp_path / "data/betterborg/codex/marketplace"
    bundle.mkdir(parents=True)
    marker = bundle / "foreign.txt"
    marker.write_text("not BetterBorg\n", encoding="utf-8")

    result, spawns = _install(tmp_path, fake)

    assert result.status is CodexPluginStatus.COLLISION
    assert "ownership metadata" in (result.reason or "")
    assert marker.read_text() == "not BetterBorg\n"
    assert ("plugin", "marketplace", "add", str(bundle), "--json") not in fake.calls
    assert spawns == []


def test_failed_fresh_activation_removes_host_and_bundle_state(
    tmp_path: Path,
) -> None:
    fake = _FakeCodex()

    def fail_verification(*_args) -> None:
        raise ValueError("injected MCP failure")

    result, spawns = _install(
        tmp_path,
        fake,
        mcp_verifier=fail_verification,
    )

    assert result.status is CodexPluginStatus.FAILED
    assert "rolled back" in (result.reason or "")
    assert result.bundle_path is None
    assert fake.marketplace_source is None
    assert fake.installed is False
    assert not tmp_path.joinpath("data").exists()
    assert spawns == []


def test_absent_codex_defers_without_files_commands_or_mcp_spawn(
    tmp_path: Path,
) -> None:
    commands: list[object] = []
    spawns: list[object] = []

    result = install_codex_plugin(
        launch_environment={"PATH": str(tmp_path / "empty"), "HOME": str(tmp_path)},
        data_home=tmp_path / "data",
        executable_lookup=lambda *_args, **_kwargs: None,
        command_runner=lambda *args, **kwargs: commands.append((args, kwargs)),
        mcp_verifier=lambda *args: spawns.append(args),
    )

    assert result.status is CodexPluginStatus.SETUP_REQUIRED
    assert "Install Codex" in (result.guidance or "")
    assert commands == []
    assert spawns == []
    assert not tmp_path.joinpath("data").exists()


def test_missing_persistent_borg_does_not_list_or_mutate_codex(
    tmp_path: Path,
) -> None:
    fake = _FakeCodex()
    codex = _executable(tmp_path / "bin/codex", "exit 0")

    def lookup(name: str, **_kwargs):
        return str(codex) if name == "codex" else None

    result = install_codex_plugin(
        launch_environment={"PATH": str(codex.parent), "HOME": str(tmp_path)},
        data_home=tmp_path / "data",
        executable_lookup=lookup,
        command_runner=fake,
    )

    assert result.status is CodexPluginStatus.SETUP_REQUIRED
    assert "uv tool install betterborg-cli" in (result.guidance or "")
    assert fake.calls == []
    assert not tmp_path.joinpath("data").exists()


def test_cli_reports_codex_success_and_new_thread_guidance(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_module,
        "install_codex_plugin",
        lambda: CodexPluginInstallation(
            status=CodexPluginStatus.INSTALLED,
            new_thread_guidance="Start a new Codex thread now.",
        ),
    )

    result = CliRunner().invoke(cli, ["plugin", "install", "codex"])

    assert result.exit_code == 0
    assert "Installed the BetterBorg plugin for Codex" in result.output
    assert "new Codex thread" in result.output


def _upgraded_bundle(tmp_path: Path, name: str, version: str) -> Path:
    source = resources.files("betterborg_cli.codex_plugin_bundle") / "marketplace"
    destination = tmp_path / name
    _copy_resource(source, destination)
    manifest = destination / "plugins/borg/.codex-plugin/plugin.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["version"] = version
    manifest.write_text(json.dumps(value), encoding="utf-8")
    return destination


def _copy_resource(source, destination: Path) -> None:
    destination.mkdir()
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            _copy_resource(child, target)
        else:
            target.write_bytes(child.read_bytes())
