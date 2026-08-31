"""Clean build, installation, bundled-data, and binary release contracts."""

from __future__ import annotations

import configparser
import json
import os
import shutil
import subprocess
import sys
import tarfile
import venv
import zipfile
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version
from test_adapter_harness import LocalHttpServer

from betterborg_cli import __version__

REPOSITORY_ROOT = Path(__file__).parents[1]
CHECK_VERSIONS = REPOSITORY_ROOT / "scripts/check_versions.py"
VERSION_SOURCES = (
    ("src/betterborg_cli/__init__.py", "python"),
    ("npm/package.json", "top-level"),
    (
        "src/betterborg_cli/claude_plugin_bundle/marketplace/plugins/betterborg/"
        ".claude-plugin/plugin.json",
        "top-level",
    ),
    (
        "src/betterborg_cli/codex_plugin_bundle/marketplace/plugins/betterborg/"
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
        "betterborg_cli/claude_plugin_bundle/marketplace/plugins/betterborg/"
        ".claude-plugin/plugin.json"
    ),
    (
        "betterborg_cli/codex_plugin_bundle/marketplace/.agents/plugins/"
        "marketplace.json"
    ),
    (
        "betterborg_cli/codex_plugin_bundle/marketplace/plugins/betterborg/"
        ".codex-plugin/plugin.json"
    ),
)

SPAWNED_URL_REQUEST = """
import multiprocessing
import sys

from betterborg_cli.agent_runtime import (
    CancellationToken,
    MultiprocessUrlRequest,
    UrlRequestSpec,
)

if len(sys.argv) == 3:
    multiprocessing.set_executable(sys.argv[2])
    sys.frozen = True

response = MultiprocessUrlRequest(
    UrlRequestSpec(sys.argv[1], "GET", {}, None),
    CancellationToken(),
).run()
print(response.status_code, response.body.decode("utf-8"))
"""


@dataclass(frozen=True)
class BuiltDistributions:
    wheel: Path
    sdist: Path


def _run(
    *command: str | os.PathLike[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        env=env,
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


def test_wheel_console_entry_uses_root_run_lifecycle(
    built_distributions: BuiltDistributions,
) -> None:
    with zipfile.ZipFile(built_distributions.wheel) as archive:
        entry_points = next(
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/entry_points.txt")
        )

        parsed = configparser.ConfigParser()
        parsed.read_string(archive.read(entry_points).decode("utf-8"))

        assert dict(parsed["console_scripts"]) == {
            "betterborg": "betterborg_cli.cli:main"
        }


def test_wheel_declares_python_compatible_rich_dependency(
    built_distributions: BuiltDistributions,
) -> None:
    with zipfile.ZipFile(built_distributions.wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(
            archive.read(metadata_name).decode("utf-8"), headersonly=True
        )

    requirements = [
        Requirement(value) for value in metadata.get_all("Requires-Dist", [])
    ]
    rich = next(
        requirement for requirement in requirements if requirement.name == "rich"
    )
    assert Version("15.0.0") in rich.specifier
    assert Version("16.0.0") not in rich.specifier


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
    borg = scripts / ("betterborg.exe" if os.name == "nt" else "betterborg")
    _run(python, "-m", "pip", "install", artifact)

    startup_hook = tmp_path / "startup-hook"
    startup_hook.mkdir()
    freeze_marker = tmp_path / "freeze-support"
    startup_hook.joinpath("sitecustomize.py").write_text(
        """import multiprocessing
import os
from pathlib import Path

original_freeze_support = multiprocessing.freeze_support


def freeze_support():
    Path(os.environ["BETTERBORG_FREEZE_MARKER"]).write_text("called")
    original_freeze_support()


multiprocessing.freeze_support = freeze_support
""",
        encoding="utf-8",
    )
    environment_variables = os.environ.copy()
    environment_variables["PYTHONPATH"] = str(startup_hook)
    environment_variables["BETTERBORG_FREEZE_MARKER"] = str(freeze_marker)

    completed = _run(borg, "version", env=environment_variables)
    with LocalHttpServer(
        lambda _request: (200, {}, b"installed-worker-response")
    ) as server:
        worker = _run(
            python,
            "-c",
            SPAWNED_URL_REQUEST,
            server.url("/installed-worker"),
        )
    metadata = _run(
        python,
        "-c",
        "from importlib.metadata import version; print(version('betterborg'))",
    )
    rich_metadata = _run(
        python,
        "-c",
        "from importlib.metadata import version; import rich; print(version('rich'))",
    )

    assert completed.stdout.strip() == f"betterborg {__version__}"
    assert freeze_marker.read_text(encoding="utf-8") == "called"
    assert worker.stdout.strip() == "200 installed-worker-response"
    assert [request.path for request in server.requests] == ["/installed-worker"]
    assert metadata.stdout.strip() == __version__
    assert Version(rich_metadata.stdout.strip()).major == 15


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
    binary = output / ("betterborg.exe" if os.name == "nt" else "betterborg")

    assert _run(binary, "version").stdout.strip() == f"betterborg {__version__}"
    with LocalHttpServer(
        lambda _request: (200, {}, b"frozen-worker-response")
    ) as server:
        worker = _run(
            sys.executable,
            "-c",
            SPAWNED_URL_REQUEST,
            server.url("/frozen-worker"),
            binary,
        )
    assert worker.stdout.strip() == "200 frozen-worker-response"
    assert [request.path for request in server.requests] == ["/frozen-worker"]
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
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == f"release versions match {__version__}"


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
