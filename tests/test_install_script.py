"""Checksum-verifying persistent curl installer behavior."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
from release_test_support import REPOSITORY_ROOT, release_artifacts

INSTALLER = REPOSITORY_ROOT / "scripts/install.sh"


def _write_tool(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\nset -eu\n{body}", encoding="utf-8")
    path.chmod(0o755)


def _fixture_release(directory: Path, version: str = "1.2.3") -> None:
    directory.mkdir()
    binary = """\
printf '%s|%s|%s\\n' "$0" "$*" "$PATH" >> "$INSTALL_LOG"
case "${1:-}" in
    version)
        printf 'betterborg %s\\n' "${REPORTED_VERSION:-$RELEASE_VERSION}"
        ;;
    plugins)
        [ "${2:-}" = install ]
        [ "${3:-}" = --all ]
        exit "${PLUGIN_EXIT:-0}"
        ;;
    *) exit 9 ;;
esac
"""
    for target in release_artifacts.TARGETS:
        _write_tool(directory / target.filename, binary)
        release_artifacts.write_checksum(directory / target.filename)
    shutil.copyfile(INSTALLER, directory / release_artifacts.INSTALLER_FILENAME)
    release_artifacts.write_manifest(
        version, directory, directory / "release-manifest.json"
    )


def _run_installer(
    tmp_path: Path,
    *,
    system: str = "Linux",
    architecture: str = "x86_64",
    kernel: str = "6.8.0",
    version: str = "1.2.3",
    extra_environment: dict[str, str] | None = None,
    tamper_target: bool = False,
    existing_install: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
    fixture = tmp_path / "release"
    _fixture_release(fixture, version)
    if tamper_target:
        (fixture / "betterborg-linux-x86_64").write_text(
            "tampered\n", encoding="utf-8"
        )
    tools = tmp_path / "tools"
    tools.mkdir()
    home = tmp_path / "home with spaces"
    home.mkdir()
    if existing_install is not None:
        installed = home / ".local/bin/betterborg"
        installed.parent.mkdir(parents=True)
        installed.write_text(existing_install, encoding="utf-8")
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    curl_log = tmp_path / "curl.log"
    install_log = tmp_path / "install.log"

    _write_tool(
        tools / "uname",
        """\
case "${1:-}" in
    -s) printf '%s\\n' "$TEST_UNAME_SYSTEM" ;;
    -m) printf '%s\\n' "$TEST_UNAME_ARCH" ;;
    -r) printf '%s\\n' "$TEST_UNAME_KERNEL" ;;
    *) exit 2 ;;
esac
""",
    )
    _write_tool(
        tools / "curl",
        """\
destination=
url=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --output)
            shift
            destination=$1
            ;;
        http://*|https://*) url=$1 ;;
    esac
    shift
done
[ -n "$destination" ]
[ -n "$url" ]
printf '%s\\n' "$url" >> "$CURL_LOG"
cp "$RELEASE_FIXTURE/${url##*/}" "$destination"
""",
    )
    environment = {
        **os.environ,
        "CURL_LOG": str(curl_log),
        "HOME": str(home),
        "INSTALL_LOG": str(install_log),
        "PATH": f"{tools}:/usr/bin:/bin",
        "RELEASE_FIXTURE": str(fixture),
        "RELEASE_VERSION": version,
        "TEST_UNAME_ARCH": architecture,
        "TEST_UNAME_KERNEL": kernel,
        "TEST_UNAME_SYSTEM": system,
        "TMPDIR": str(temporary),
        "_BETTERBORG_RELEASES_URL": "https://releases.example.test",
    }
    environment.update(extra_environment or {})
    completed = subprocess.run(
        ["/bin/sh", str(INSTALLER)],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    return completed, home, curl_log, install_log


@pytest.mark.parametrize(
    ("system", "architecture", "kernel", "target"),
    (
        ("Linux", "x86_64", "6.8.0", "betterborg-linux-x86_64"),
        ("Linux", "aarch64", "6.8.0", "betterborg-linux-arm64"),
        ("Darwin", "amd64", "23.4.0", "betterborg-darwin-x86_64"),
        ("Darwin", "arm64", "23.4.0", "betterborg-darwin-arm64"),
        (
            "Linux",
            "x86_64",
            "5.15.153.1-microsoft-standard-WSL2",
            "betterborg-linux-x86_64",
        ),
    ),
)
def test_supported_target_mapping_uses_manifest_and_exact_versioned_asset(
    tmp_path: Path,
    system: str,
    architecture: str,
    kernel: str,
    target: str,
) -> None:
    result, _home, curl_log, _install_log = _run_installer(
        tmp_path, system=system, architecture=architecture, kernel=kernel
    )

    assert result.returncode == 0, result.stderr
    assert curl_log.read_text(encoding="utf-8").splitlines() == [
        "https://releases.example.test/latest/download/release-manifest.json",
        f"https://releases.example.test/download/v1.2.3/{target}",
    ]


def test_selected_version_uses_only_versioned_manifest_and_assets(
    tmp_path: Path,
) -> None:
    result, _home, curl_log, _install_log = _run_installer(
        tmp_path, extra_environment={"BETTERBORG_VERSION": "1.2.3"}
    )

    assert result.returncode == 0, result.stderr
    assert curl_log.read_text(encoding="utf-8").splitlines() == [
        "https://releases.example.test/download/v1.2.3/release-manifest.json",
        "https://releases.example.test/download/v1.2.3/betterborg-linux-x86_64",
    ]


def test_install_is_atomic_verified_and_activates_plugins_last(
    tmp_path: Path,
) -> None:
    result, home, _curl_log, install_log = _run_installer(tmp_path)

    installed = home / ".local/bin/betterborg"
    assert result.returncode == 0, result.stderr
    assert installed.read_bytes() == (
        tmp_path / "release/betterborg-linux-x86_64"
    ).read_bytes()
    assert installed.stat().st_mode & stat.S_IXUSR
    calls = [line.split("|", 2) for line in install_log.read_text().splitlines()]
    assert calls[0][0].startswith(str(home / ".local/bin/.betterborg.install."))
    assert calls[0][1] == "version"
    assert calls[1][0:2] == [str(installed), "version"]
    assert calls[2][0:2] == [str(installed), "plugins install --all"]
    assert calls[2][2].split(":", 1)[0] == str(installed.parent)
    assert not list(installed.parent.glob(".betterborg.install.*"))
    assert "Add " + str(installed.parent) + " to PATH" in result.stdout
    assert f'export PATH="{installed.parent}:$PATH"' in result.stdout


def test_checksum_failure_preserves_existing_install_and_skips_plugins(
    tmp_path: Path,
) -> None:
    result, home, _curl_log, install_log = _run_installer(
        tmp_path,
        tamper_target=True,
        existing_install="existing installation\n",
    )

    existing = home / ".local/bin/betterborg"
    assert result.returncode == 1
    assert "failed SHA-256 verification" in result.stderr
    assert existing.read_text(encoding="utf-8") == "existing installation\n"
    assert not install_log.exists()


def test_version_failure_does_not_replace_existing_install(
    tmp_path: Path,
) -> None:
    result, home, _curl_log, install_log = _run_installer(
        tmp_path,
        extra_environment={"REPORTED_VERSION": "9.9.9"},
        existing_install="existing installation\n",
    )

    installed = home / ".local/bin/betterborg"
    assert result.returncode == 1
    assert "expected 'betterborg 1.2.3'" in result.stderr
    assert installed.read_text(encoding="utf-8") == "existing installation\n"
    assert [
        line.split("|", 2)[1] for line in install_log.read_text().splitlines()
    ] == ["version"]
    assert not list(installed.parent.glob(".betterborg.install.*"))


def test_plugin_failure_occurs_after_persistent_install_verification(
    tmp_path: Path,
) -> None:
    result, home, _curl_log, install_log = _run_installer(
        tmp_path, extra_environment={"PLUGIN_EXIT": "7"}
    )

    installed = home / ".local/bin/betterborg"
    assert result.returncode == 1
    assert installed.is_file()
    assert (
        "CLI was installed, but host plugin activation did not complete"
        in result.stderr
    )
    assert [
        line.split("|", 2)[1] for line in install_log.read_text().splitlines()
    ] == [
        "version",
        "version",
        "plugins install --all",
    ]


@pytest.mark.parametrize(
    ("system", "architecture", "kernel", "guidance"),
    (
        ("FreeBSD", "x86_64", "14.0", "No standalone"),
        ("MINGW64_NT", "x86_64", "10.0", "WSL2"),
        ("Linux", "x86_64", "4.4.0-Microsoft", "WSL2"),
        ("Linux", "riscv64", "6.8.0", "No standalone"),
    ),
)
def test_unsupported_targets_fail_with_one_uvx_fallback_and_wsl_guidance(
    tmp_path: Path,
    system: str,
    architecture: str,
    kernel: str,
    guidance: str,
) -> None:
    result, home, curl_log, install_log = _run_installer(
        tmp_path, system=system, architecture=architecture, kernel=kernel
    )

    assert result.returncode == 1
    assert guidance in result.stderr
    assert result.stderr.count("uvx") == 1
    assert not curl_log.exists()
    assert not install_log.exists()
    assert not (home / ".local/bin/betterborg").exists()
