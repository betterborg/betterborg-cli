"""Clean build, installation, bundled-data, and binary release contracts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import venv
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from betterborg_cli import __version__

REPOSITORY_ROOT = Path(__file__).parents[1]
CHECK_VERSIONS = REPOSITORY_ROOT / "scripts/check_versions.py"
VERSION_SOURCES = (
    ("src/betterborg_cli/__init__.py", "python"),
    ("npm/package.json", "top-level"),
    (
        "src/betterborg_cli/claude_plugin_bundle/marketplace/plugins/borg/"
        ".claude-plugin/plugin.json",
        "top-level",
    ),
    (
        "src/betterborg_cli/codex_plugin_bundle/marketplace/plugins/borg/"
        ".codex-plugin/plugin.json",
        "top-level",
    ),
    (
        "src/betterborg_cli/claude_plugin_bundle/marketplace/.claude-plugin/"
        "marketplace.json",
        "marketplace-entry",
    ),
    (
        "src/betterborg_cli/codex_plugin_bundle/marketplace/.agents/plugins/"
        "marketplace.json",
        "marketplace-entry",
    ),
)
PACKAGE_ASSETS = (
    "betterborg_cli/agent_runtime/pricing.py",
    "betterborg_cli/execution_estimate.py",
    "betterborg_cli/planning/architect.py",
    "betterborg_cli/repo_analysis/analyzer.py",
    "betterborg_cli/store/migrations/001_initial.sql",
    (
        "betterborg_cli/claude_plugin_bundle/marketplace/.claude-plugin/"
        "marketplace.json"
    ),
    (
        "betterborg_cli/claude_plugin_bundle/marketplace/plugins/borg/"
        ".claude-plugin/plugin.json"
    ),
    (
        "betterborg_cli/codex_plugin_bundle/marketplace/.agents/plugins/"
        "marketplace.json"
    ),
    (
        "betterborg_cli/codex_plugin_bundle/marketplace/plugins/borg/"
        ".codex-plugin/plugin.json"
    ),
)


@dataclass(frozen=True)
class BuiltDistributions:
    wheel: Path
    sdist: Path


def _run(
    *command: str | os.PathLike[str],
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _copy_release_source(destination: Path) -> Path:
    source = destination / "source"
    shutil.copytree(
        REPOSITORY_ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".betterborg-analysis",
            ".betterborg-task",
            ".git",
            ".orchestry",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            "__pycache__",
            "build",
            "dist",
            "*.egg-info",
            "*.pyc",
        ),
    )
    return source


@pytest.fixture(scope="module")
def built_distributions(tmp_path_factory: pytest.TempPathFactory) -> BuiltDistributions:
    root = tmp_path_factory.mktemp("python-distributions")
    source = _copy_release_source(root)
    output = root / "dist"
    _run(sys.executable, "-m", "build", "--outdir", output, cwd=source)
    wheel = next(output.glob("betterborg-*.whl"))
    sdist = next(output.glob("betterborg-*.tar.gz"))
    return BuiltDistributions(wheel=wheel, sdist=sdist)


def _wheel_members(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return set(archive.namelist())


def _sdist_members(path: Path) -> set[str]:
    with tarfile.open(path, "r:gz") as archive:
        members = {member.name.split("/", 1)[-1] for member in archive.getmembers()}
    members.update(
        member.removeprefix("src/")
        for member in tuple(members)
        if member.startswith("src/")
    )
    return members


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_python_distributions_contain_release_assets(
    built_distributions: BuiltDistributions,
    kind: str,
) -> None:
    artifact = getattr(built_distributions, kind)
    members = _wheel_members(artifact) if kind == "wheel" else _sdist_members(artifact)

    assert all(asset in members for asset in PACKAGE_ASSETS)
    if kind == "wheel":
        assert any(name.endswith(".dist-info/entry_points.txt") for name in members)
        assert any(name.endswith(".dist-info/METADATA") for name in members)
    else:
        assert any(name.endswith(".egg-info/entry_points.txt") for name in members)
        assert "PKG-INFO" in members


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_clean_install_exposes_borg_and_matching_metadata(
    tmp_path: Path,
    built_distributions: BuiltDistributions,
    kind: str,
) -> None:
    artifact = getattr(built_distributions, kind)
    environment = tmp_path / f"{kind}-environment"
    venv.EnvBuilder(with_pip=True).create(environment)
    scripts = environment / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    borg = scripts / ("borg.exe" if os.name == "nt" else "borg")
    _run(python, "-m", "pip", "install", artifact)

    completed = _run(borg, "version")
    metadata = _run(
        python,
        "-c",
        "from importlib.metadata import version; print(version('betterborg'))",
    )

    assert completed.stdout.strip() == f"borg {__version__}"
    assert metadata.stdout.strip() == __version__


def test_one_file_binary_reports_version_and_contains_assets(tmp_path: Path) -> None:
    pyinstaller = Path(sys.executable).with_name(
        "pyinstaller.exe" if os.name == "nt" else "pyinstaller"
    )
    if not pyinstaller.is_file():
        pytest.fail("PyInstaller is required by the locked development environment")
    output = tmp_path / "dist"
    work = tmp_path / "build"
    _run(
        pyinstaller,
        "--clean",
        "--noconfirm",
        "--distpath",
        output,
        "--workpath",
        work,
        REPOSITORY_ROOT / "betterborg.spec",
        cwd=REPOSITORY_ROOT,
    )
    binary = output / ("borg.exe" if os.name == "nt" else "borg")

    assert _run(binary, "version").stdout.strip() == f"borg {__version__}"
    from PyInstaller.archive.readers import CArchiveReader

    bundled = set(CArchiveReader(str(binary)).toc)
    assert all(
        asset in bundled for asset in PACKAGE_ASSETS if not asset.endswith(".py")
    )
    assert not any(
        name.startswith("betterborg-") and ".dist-info/" in name
        for name in bundled
    )
    assert not any(
        name.endswith("direct_url.json") or "__editable__" in name
        for name in bundled
    )


def _minimal_version_tree(tmp_path: Path) -> Path:
    for relative in (
        Path("pyproject.toml"),
        *(Path(path) for path, _ in VERSION_SOURCES),
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY_ROOT / relative, destination)
    return tmp_path


@pytest.mark.parametrize(
    ("source", "version_location"),
    VERSION_SOURCES,
)
def test_version_check_rejects_each_source_mismatch(
    tmp_path: Path,
    source: str,
    version_location: str,
) -> None:
    root = _minimal_version_tree(tmp_path)
    path = root / source
    mismatched_version = "0.0.0" if __version__ != "0.0.0" else "0.0.1"
    if version_location == "python":
        content = path.read_text(encoding="utf-8")
        path.write_text(
            content.replace(
                f'__version__ = "{__version__}"',
                f'__version__ = "{mismatched_version}"',
            ),
            encoding="utf-8",
        )
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if version_location == "marketplace-entry":
            value["plugins"][0]["version"] = mismatched_version
        else:
            value["version"] = mismatched_version
        path.write_text(json.dumps(value), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(CHECK_VERSIONS),
            "--root",
            str(root),
            "--tag",
            f"v{__version__}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert source in completed.stderr


def test_version_check_accepts_all_sources_and_prospective_tag(tmp_path: Path) -> None:
    root = _minimal_version_tree(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(CHECK_VERSIONS),
            "--root",
            str(root),
            "--tag",
            f"v{__version__}",
            "--greater-than",
            "0.1.0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == f"release versions match {__version__}"


def test_version_check_rejects_phase_10_version_as_final(tmp_path: Path) -> None:
    root = _minimal_version_tree(tmp_path)
    for source, version_location in VERSION_SOURCES:
        path = root / source
        content = path.read_text(encoding="utf-8")
        if version_location == "python":
            content = content.replace(
                f'__version__ = "{__version__}"', '__version__ = "0.1.0"'
            )
        else:
            value = json.loads(content)
            if version_location == "marketplace-entry":
                value["plugins"][0]["version"] = "0.1.0"
            else:
                value["version"] = "0.1.0"
            content = json.dumps(value)
        path.write_text(content, encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(CHECK_VERSIONS),
            "--root",
            str(root),
            "--tag",
            "v0.1.0",
            "--greater-than",
            "0.1.0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "must be greater than '0.1.0'" in completed.stderr


def test_version_check_rejects_mismatched_prospective_tag() -> None:
    prospective_tag = "v0.0.0" if __version__ != "0.0.0" else "v0.0.1"

    completed = subprocess.run(
        [
            sys.executable,
            str(CHECK_VERSIONS),
            "--root",
            str(REPOSITORY_ROOT),
            "--tag",
            prospective_tag,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert f"prospective tag {prospective_tag!r}" in completed.stderr
    assert f"expected 'v{__version__}'" in completed.stderr


def test_version_check_rejects_unreviewed_source_version() -> None:
    reviewed = "0.0.0" if __version__ != "0.0.0" else "0.0.1"

    completed = subprocess.run(
        [
            sys.executable,
            str(CHECK_VERSIONS),
            "--root",
            str(REPOSITORY_ROOT),
            "--expected",
            reviewed,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert f"reviewed version {reviewed!r}" in completed.stderr
    assert f"source version {__version__!r}" in completed.stderr
