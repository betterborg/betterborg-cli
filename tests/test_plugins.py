"""Plugin installation orchestration and public CLI behavior."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pytest
from click.testing import CliRunner

from betterborg_cli import cli as cli_module
from betterborg_cli.claude_plugin import (
    ClaudePluginInstallation,
    ClaudePluginStatus,
)
from betterborg_cli.cli import cli
from betterborg_cli.codex_plugin import CodexPluginInstallation, CodexPluginStatus
from betterborg_cli.plugin_activation import (
    PluginActivationPreflight,
    PluginActivationStatus,
)
from betterborg_cli.plugins import PluginInstaller


def _ready_preflight() -> PluginActivationPreflight:
    return PluginActivationPreflight(
        status=PluginActivationStatus.READY,
        executable=Path("/persistent/bin/borg"),
        version="borg test",
    )


def _configured_installer(
    calls: list[object],
    *,
    claude: ClaudePluginInstallation | None = None,
    codex: CodexPluginInstallation | None = None,
    preflight: PluginActivationPreflight | None = None,
) -> PluginInstaller:
    preflight_result = preflight or _ready_preflight()

    def verify() -> PluginActivationPreflight:
        calls.append("preflight")
        return preflight_result

    def install_claude(*, preflight):
        calls.append(("claude", preflight))
        return claude or ClaudePluginInstallation(
            status=ClaudePluginStatus.INSTALLED,
            reload_guidance="Run /reload-plugins now.",
        )

    def install_codex(*, preflight):
        calls.append(("codex", preflight))
        return codex or CodexPluginInstallation(
            status=CodexPluginStatus.UNCHANGED,
            new_thread_guidance="Start a new Codex thread now.",
        )

    return PluginInstaller(
        preflight=verify,
        host_installers={"claude": install_claude, "codex": install_codex},
    )


def _invoke(
    monkeypatch: pytest.MonkeyPatch,
    installer: PluginInstaller,
    *arguments: str,
):
    monkeypatch.setattr(cli_module, "PluginInstaller", lambda: installer)
    return CliRunner().invoke(cli, ["plugins", "install", *arguments])


@pytest.mark.parametrize("arguments", [(), ("--all",)])
def test_all_hosts_is_default_and_runs_both_after_one_preflight(
    monkeypatch: pytest.MonkeyPatch, arguments: tuple[str, ...]
) -> None:
    calls: list[object] = []
    installer = _configured_installer(calls)

    result = _invoke(monkeypatch, installer, *arguments)

    assert result.exit_code == 0, result.output
    assert calls == [
        "preflight",
        ("claude", _ready_preflight()),
        ("codex", _ready_preflight()),
    ]
    assert "Claude Code: completed" in result.output
    assert "Codex: completed" in result.output
    assert "/reload-plugins" in result.output
    assert "new Codex thread" in result.output


@pytest.mark.parametrize(
    ("host", "selected", "unselected"),
    (("claude", "claude", "codex"), ("codex", "codex", "claude")),
)
def test_explicit_host_invokes_only_the_selected_installer(
    monkeypatch: pytest.MonkeyPatch,
    host: str,
    selected: str,
    unselected: str,
) -> None:
    calls: list[object] = []

    result = _invoke(monkeypatch, _configured_installer(calls), "--host", host)

    assert result.exit_code == 0, result.output
    assert [call[0] for call in calls[1:]] == [selected]
    assert selected in result.output.casefold()
    assert unselected not in result.output.casefold()


@pytest.mark.parametrize(
    "arguments",
    (("--host", "other"), ("--all", "--host", "claude")),
)
def test_invalid_selection_is_rejected_before_installation(
    monkeypatch: pytest.MonkeyPatch, arguments: tuple[str, ...]
) -> None:
    monkeypatch.setattr(
        cli_module,
        "PluginInstaller",
        lambda: pytest.fail("installer must not be constructed"),
    )

    result = CliRunner().invoke(cli, ["plugins", "install", *arguments])

    assert result.exit_code == 2
    assert "Error" in result.output


def test_absent_hosts_are_reported_as_deferred(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    installer = _configured_installer(
        calls,
        claude=ClaudePluginInstallation(
            status=ClaudePluginStatus.DEFERRED,
            reason="Claude Code is absent.",
            guidance="Install Claude Code when needed.",
        ),
        codex=CodexPluginInstallation(
            status=CodexPluginStatus.DEFERRED,
            reason="Codex is absent.",
            guidance="Install Codex when needed.",
        ),
    )

    result = _invoke(monkeypatch, installer)

    assert result.exit_code == 0, result.output
    assert "Claude Code: deferred — Claude Code is absent." in result.output
    assert "Codex: deferred — Codex is absent." in result.output


def test_setup_required_preflight_blocks_every_host_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    preflight = PluginActivationPreflight(
        status=PluginActivationStatus.SETUP_REQUIRED,
        reason="Persistent borg is unavailable.",
        guidance="Install it with the persistent installer.",
    )

    result = _invoke(
        monkeypatch, _configured_installer(calls, preflight=preflight)
    )

    assert result.exit_code == 1
    assert calls == ["preflight"]
    assert result.output.count("setup_required") == 2
    assert "Persistent borg is unavailable." in result.output
    assert "persistent installer" in result.output


def test_partial_failure_keeps_completed_host_summary_and_returns_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    installer = _configured_installer(
        calls,
        codex=CodexPluginInstallation(
            status=CodexPluginStatus.FAILED,
            reason="Codex verification failed and was rolled back.",
        ),
    )

    result = _invoke(monkeypatch, installer)

    assert result.exit_code == 1
    assert [call[0] for call in calls[1:]] == ["claude", "codex"]
    assert "Claude Code: completed" in result.output
    assert "Codex: failed" in result.output
    assert "rolled back" in result.output


def test_selected_install_preserves_unrelated_and_unselected_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"unrelated": ["team-tools"], "claude": [], "codex": []}
    preflight = _ready_preflight()

    def install_claude(*, preflight: PluginActivationPreflight):
        state["claude"].append(preflight.version or "")
        return ClaudePluginInstallation(status=ClaudePluginStatus.INSTALLED)

    installer = PluginInstaller(
        preflight=lambda: preflight,
        host_installers={
            "claude": install_claude,
            "codex": lambda **_kwargs: pytest.fail("Codex must not be mutated"),
        },
    )

    result = _invoke(monkeypatch, installer, "--host", "claude")

    assert result.exit_code == 0, result.output
    assert state == {
        "unrelated": ["team-tools"],
        "claude": ["borg test"],
        "codex": [],
    }


def test_bundled_plugins_are_thin_registrations_of_the_same_mcp_server() -> None:
    claude = resources.files("betterborg_cli.claude_plugin_bundle") / "marketplace"
    codex = resources.files("betterborg_cli.codex_plugin_bundle") / "marketplace"
    claude_mcp = json.loads(
        (claude / "plugins/borg/.mcp.json").read_text(encoding="utf-8")
    )["mcpServers"]["borg"]
    codex_mcp = json.loads(
        (codex / "plugins/borg/.mcp.json").read_text(encoding="utf-8")
    )["borg"]

    assert claude_mcp == codex_mcp == {"command": "borg", "args": ["mcp"]}
    for bundle in (claude, codex):
        files = [entry for entry in bundle.rglob("*") if entry.is_file()]
        assert not any(entry.suffix == ".py" for entry in files)
        assert not any(
            "harness" in entry.read_text(encoding="utf-8").casefold()
            for entry in files
        )
