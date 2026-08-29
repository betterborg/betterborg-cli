"""Persistent executable preflight for host plugin activation."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

from betterborg_cli.plugin_activation import (
    PERSISTENT_INSTALL_COMMAND,
    PluginActivationStatus,
    preflight_plugin_activation,
    prepare_plugin_activation,
)


def _borg(directory: Path, body: str = "printf 'borg 0.1.0\\n'") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    executable = directory / "borg"
    executable.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable


def test_missing_borg_returns_guidance_without_mutation_or_spawn(
    tmp_path: Path,
) -> None:
    empty_path = tmp_path / "host-home" / "empty-bin"
    empty_path.mkdir(parents=True)
    materializations: list[object] = []
    version_spawns: list[object] = []

    result = prepare_plugin_activation(
        lambda preflight: materializations.append(preflight),
        launch_environment={"PATH": str(empty_path)},
        version_runner=lambda *args, **kwargs: version_spawns.append((args, kwargs)),
    )

    assert result.preflight.status is PluginActivationStatus.SETUP_REQUIRED
    assert "host launch PATH" in (result.preflight.reason or "")
    assert PERSISTENT_INSTALL_COMMAND in (result.preflight.guidance or "")
    assert "uvx" in (result.preflight.guidance or "")
    assert result.bundle is None
    assert materializations == []
    assert version_spawns == []


def test_persistent_borg_is_verified_before_bundle_materialization(
    tmp_path: Path,
) -> None:
    host_home = tmp_path / "host-home"
    executable = _borg(host_home / ".local" / "bin")
    events: list[object] = []

    def materialize(preflight):
        events.append(("materialize", preflight.executable))
        bundle = host_home / ".local" / "share" / "betterborg" / "plugin"
        bundle.mkdir(parents=True)
        return bundle

    def run_version(*args, **kwargs):
        events.append(("version", tuple(args[0])))
        return subprocess.run(*args, **kwargs)

    result = prepare_plugin_activation(
        materialize,
        launch_environment={"PATH": str(executable.parent)},
        version_runner=run_version,
    )

    assert result.preflight.status is PluginActivationStatus.READY
    assert result.preflight.executable == executable.resolve()
    assert result.preflight.version == "borg 0.1.0"
    assert result.bundle == host_home / ".local" / "share" / "betterborg" / "plugin"
    assert events == [
        ("version", (str(executable.resolve()), "version")),
        ("materialize", executable.resolve()),
    ]


def test_transient_extraction_borg_is_rejected_before_version_or_materialization(
    tmp_path: Path,
) -> None:
    extraction = tmp_path / "host-home" / "plugin-extraction"
    executable = _borg(extraction / "bin")
    spawns: list[object] = []
    materializations: list[object] = []

    result = prepare_plugin_activation(
        lambda preflight: materializations.append(preflight),
        launch_environment={"PATH": str(executable.parent)},
        transient_roots=(extraction,),
        version_runner=lambda *args, **kwargs: spawns.append((args, kwargs)),
    )

    assert result.preflight.status is PluginActivationStatus.SETUP_REQUIRED
    assert "transient" in (result.preflight.reason or "")
    assert result.bundle is None
    assert spawns == []
    assert materializations == []


def test_uvx_archive_executable_is_treated_as_transient(tmp_path: Path) -> None:
    executable = _borg(tmp_path / ".cache" / "uv" / "archive-v0" / "bin")

    result = preflight_plugin_activation(
        launch_environment={"PATH": str(executable.parent)}
    )

    assert result.status is PluginActivationStatus.SETUP_REQUIRED
    assert "transient" in (result.reason or "")


def test_failed_version_check_does_not_materialize_bundle(tmp_path: Path) -> None:
    executable = _borg(tmp_path / "host-home" / "bin", "exit 23")
    materializations: list[object] = []

    result = prepare_plugin_activation(
        lambda preflight: materializations.append(preflight),
        launch_environment={"PATH": str(executable.parent)},
    )

    assert result.preflight.status is PluginActivationStatus.SETUP_REQUIRED
    assert "exit code 23" in (result.preflight.reason or "")
    assert result.bundle is None
    assert materializations == []


def test_resolution_uses_only_the_host_launch_path(
    tmp_path: Path, monkeypatch
) -> None:
    interactive_bin = tmp_path / "interactive-bin"
    _borg(interactive_bin)
    host_bin = tmp_path / "host-bin"
    host_bin.mkdir()
    monkeypatch.setenv(
        "PATH", os.pathsep.join((str(interactive_bin), os.environ["PATH"]))
    )

    result = preflight_plugin_activation(
        launch_environment={"PATH": str(host_bin)}
    )

    assert result.status is PluginActivationStatus.SETUP_REQUIRED
    assert result.executable is None
