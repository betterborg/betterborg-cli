"""Check every release-facing version against the Python package version."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from pathlib import Path

VERSION_ATTRIBUTE = "betterborg_cli.__version__"
VERSION_PATTERN = re.compile(
    r"^[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?"
    r"(?:\.dev[0-9]+)?(?:\+[a-z0-9]+(?:[._-][a-z0-9]+)*)?$",
    re.IGNORECASE,
)
PLUGIN_MANIFESTS = (
    Path(
        "src/betterborg_cli/claude_plugin_bundle/marketplace/plugins/borg/"
        ".claude-plugin/plugin.json"
    ),
    Path(
        "src/betterborg_cli/codex_plugin_bundle/marketplace/plugins/borg/"
        ".codex-plugin/plugin.json"
    ),
)


def _python_version(root: Path) -> str:
    source = root.joinpath("src/betterborg_cli/__init__.py").read_text(
        encoding="utf-8"
    )
    module = ast.parse(source)
    for statement in module.body:
        if not isinstance(statement, ast.Assign | ast.AnnAssign):
            continue
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        if any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in targets
        ):
            value = ast.literal_eval(statement.value)
            if isinstance(value, str) and VERSION_PATTERN.fullmatch(value):
                return value
            raise ValueError(
                "betterborg_cli.__version__ must be a valid version string"
            )
    raise ValueError("betterborg_cli.__version__ is missing")


def version_errors(root: Path) -> list[str]:
    """Return release-version inconsistencies below *root*."""
    errors: list[str] = []
    try:
        version = _python_version(root)
    except (OSError, SyntaxError, ValueError) as error:
        return [str(error)]

    try:
        project = tomllib.loads(root.joinpath("pyproject.toml").read_text("utf-8"))
        metadata = project["project"]
        dynamic = metadata.get("dynamic", [])
        version_source = project["tool"]["setuptools"]["dynamic"]["version"]
        if metadata.get("name") != "betterborg":
            errors.append("pyproject.toml project.name must be 'betterborg'")
        if "version" not in dynamic or "version" in metadata:
            errors.append("pyproject.toml must declare only a dynamic project version")
        if version_source != {"attr": VERSION_ATTRIBUTE}:
            errors.append(
                "pyproject.toml dynamic version must use " + VERSION_ATTRIBUTE
            )
    except (KeyError, OSError, tomllib.TOMLDecodeError) as error:
        errors.append(f"cannot read pyproject.toml version metadata: {error}")

    for relative_path in PLUGIN_MANIFESTS:
        try:
            manifest = json.loads(root.joinpath(relative_path).read_text("utf-8"))
            plugin_version = manifest.get("version")
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"cannot read {relative_path}: {error}")
            continue
        if plugin_version != version:
            errors.append(
                f"{relative_path} has version {plugin_version!r}; expected {version!r}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root to check",
    )
    parser.add_argument(
        "--expected",
        help="also require the reviewed release version to match the source",
    )
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    errors = version_errors(root)
    if not errors and arguments.expected is not None:
        source_version = _python_version(root)
        if arguments.expected != source_version:
            errors.append(
                f"reviewed version {arguments.expected!r} does not match "
                f"source version {source_version!r}"
            )
    if errors:
        for error in errors:
            print(f"version check failed: {error}", file=sys.stderr)
        return 1
    print(f"release versions match {_python_version(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
