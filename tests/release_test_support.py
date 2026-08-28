"""Helpers shared by standalone binary release tests."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

REPOSITORY_ROOT = Path(__file__).parents[1]


def load_script(name: str) -> ModuleType:
    """Load a repository script as an importable test module."""
    specification = importlib.util.spec_from_file_location(
        name, REPOSITORY_ROOT / "scripts" / f"{name}.py"
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


release_artifacts = load_script("release_artifacts")


def write_binary_artifact_set(directory: Path, version: str = "1.2.3") -> None:
    """Write the binaries, checksums, installer, and release manifest."""
    directory.mkdir(parents=True)
    for index, target in enumerate(release_artifacts.TARGETS, start=1):
        binary = directory / target.filename
        binary.write_bytes(f"binary fixture {index}\n".encode())
        release_artifacts.write_checksum(binary)
    shutil.copyfile(
        REPOSITORY_ROOT / "scripts/install.sh",
        directory / release_artifacts.INSTALLER_FILENAME,
    )
    release_artifacts.write_manifest(
        version, directory, directory / "release-manifest.json"
    )
