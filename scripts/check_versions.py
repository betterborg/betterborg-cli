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
STABLE_VERSION_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")
JSON_VERSION_SOURCES = (
    Path("npm/package.json"),
    Path(
        "src/betterborg_cli/claude_plugin_bundle/marketplace/plugins/borg/"
        ".claude-plugin/plugin.json"
    ),
    Path(
        "src/betterborg_cli/codex_plugin_bundle/marketplace/plugins/borg/"
        ".codex-plugin/plugin.json"
    ),
)
MARKETPLACE_SOURCES = (
    Path(
        "src/betterborg_cli/claude_plugin_bundle/marketplace/.claude-plugin/"
        "marketplace.json"
    ),
    Path(
        "src/betterborg_cli/codex_plugin_bundle/marketplace/.agents/plugins/"
        "marketplace.json"
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


def _marketplace_version(root: Path, relative_path: Path) -> str | None:
    marketplace = json.loads(root.joinpath(relative_path).read_text("utf-8"))
    plugins = marketplace.get("plugins", [])
    if not isinstance(plugins, list):
        raise ValueError("plugins must be a list")
    matches = [
        plugin
        for plugin in plugins
        if isinstance(plugin, dict) and plugin.get("name") == "borg"
    ]
    if len(matches) != 1:
        raise ValueError("must contain exactly one borg plugin entry")
    return matches[0].get("version")


def version_errors(root: Path, tag: str | None = None) -> list[str]:
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

    for relative_path in JSON_VERSION_SOURCES:
        try:
            manifest = json.loads(root.joinpath(relative_path).read_text("utf-8"))
            source_version = manifest.get("version")
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"cannot read {relative_path}: {error}")
            continue
        if source_version != version:
            errors.append(
                f"{relative_path} has version {source_version!r}; expected {version!r}"
            )

    for relative_path in MARKETPLACE_SOURCES:
        try:
            source_version = _marketplace_version(root, relative_path)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            errors.append(f"cannot read {relative_path}: {error}")
            continue
        if source_version != version:
            errors.append(
                f"{relative_path} borg entry has version {source_version!r}; "
                f"expected {version!r}"
            )

    if tag is not None:
        expected_tag = f"v{version}"
        if tag != expected_tag:
            errors.append(
                "src/betterborg_cli/__init__.py has version "
                f"{version!r}; prospective tag {tag!r} does not match; "
                f"expected {expected_tag!r}"
            )
    return errors


def _stable_version_parts(version: str) -> tuple[int, ...]:
    if not STABLE_VERSION_PATTERN.fullmatch(version):
        raise ValueError(
            "greater-version validation requires stable numeric versions"
        )
    return tuple(int(part) for part in version.split("."))


def is_greater_version(version: str, previous: str) -> bool:
    """Return whether one stable numeric version is strictly greater."""
    left = _stable_version_parts(version)
    right = _stable_version_parts(previous)
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)) > right + (0,) * (width - len(right))


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
    parser.add_argument(
        "--tag",
        help="also require the prospective release tag (in vVERSION form) to match",
    )
    parser.add_argument(
        "--greater-than",
        help="also require a stable source version greater than this prior release",
    )
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    errors = version_errors(root, arguments.tag)
    if not errors and arguments.expected is not None:
        source_version = _python_version(root)
        if arguments.expected != source_version:
            errors.append(
                f"reviewed version {arguments.expected!r} does not match "
                f"source version {source_version!r}"
            )
    if not errors and arguments.greater_than is not None:
        source_version = _python_version(root)
        try:
            greater = is_greater_version(source_version, arguments.greater_than)
        except ValueError as error:
            errors.append(str(error))
        else:
            if not greater:
                errors.append(
                    f"release version {source_version!r} must be greater than "
                    f"{arguments.greater_than!r}"
                )
    if errors:
        for error in errors:
            print(f"version check failed: {error}", file=sys.stderr)
        return 1
    print(f"release versions match {_python_version(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
