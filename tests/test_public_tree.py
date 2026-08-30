"""Safety scans for material extracted into the public repository."""

import ast
import re
from collections.abc import Iterable
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]

EXCLUDED_DIRECTORIES = {
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
}
PRIVATE_PATH_PARTS = {"private", "proprietary"}
CREDENTIAL_SUFFIXES = {".key", ".kubeconfig", ".p12", ".pem", ".pfx"}
UNAUTHORIZED_SUFFIXES = {
    ".class",
    ".dll",
    ".dylib",
    ".exe",
    ".jar",
    ".pyc",
    ".so",
    ".sqlite",
    ".sqlite3",
}
CLOUD_ONLY_MODULES = (
    "azure",
    "boto3",
    "botocore",
    "google.cloud",
    "pulumi",
)
SECRET_PATTERNS = (
    re.compile("AK" + r"IA[0-9A-Z]{16}"),
    re.compile("gh" + r"p_[A-Za-z0-9]{36}"),
    re.compile("-----BEGIN " + "PRIVATE KEY-----"),
)


def _public_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        relative_parts = path.relative_to(root).parts
        if any(part in EXCLUDED_DIRECTORIES for part in relative_parts):
            continue
        if path.is_file():
            yield path


def _imported_modules(source: str, filename: Path) -> Iterable[str]:
    tree = ast.parse(source, filename=str(filename))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def _scan_public_tree(root: Path) -> list[str]:
    violations: list[str] = []

    for path in _public_files(root):
        relative_path = path.relative_to(root)
        normalized_parts = {part.casefold() for part in relative_path.parts}

        if normalized_parts & PRIVATE_PATH_PARTS:
            violations.append(f"private path: {relative_path}")

        name = path.name.casefold()
        if (
            path.suffix.casefold() in CREDENTIAL_SUFFIXES
            or name == ".env"
            or name.startswith(".env.")
        ):
            violations.append(f"credential file: {relative_path}")

        if path.suffix.casefold() in UNAUTHORIZED_SUFFIXES:
            violations.append(f"unauthorized generated or binary file: {relative_path}")

        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            violations.append(f"unauthorized non-text file: {relative_path}")
            continue

        for pattern in SECRET_PATTERNS:
            if pattern.search(source):
                violations.append(f"credential content: {relative_path}")

        if path.suffix == ".py":
            for module in _imported_modules(source, relative_path):
                if any(
                    module == cloud_module or module.startswith(f"{cloud_module}.")
                    for cloud_module in CLOUD_ONLY_MODULES
                ):
                    violations.append(
                        f"cloud-only import {module!r}: {relative_path}"
                    )

    return sorted(violations)


def test_public_tree_passes_extraction_scan() -> None:
    assert _scan_public_tree(REPOSITORY_ROOT) == []


@pytest.mark.parametrize(
    ("relative_path", "content", "expected_violation"),
    [
        ("credentials/id.pem", "placeholder", "credential file"),
        ("private/worker.py", "", "private path"),
        ("src/cloud.py", "import boto3\n", "cloud-only import"),
        (
            "notes.txt",
            "-----BEGIN " + "PRIVATE KEY-----",
            "credential content",
        ),
        ("artifact.pyc", "compiled", "unauthorized generated or binary file"),
    ],
)
def test_extraction_scan_rejects_forbidden_material(
    tmp_path: Path,
    relative_path: str,
    content: str,
    expected_violation: str,
) -> None:
    candidate = tmp_path / relative_path
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(content, encoding="utf-8")

    violations = _scan_public_tree(tmp_path)

    assert any(expected_violation in violation for violation in violations)


def test_license_attribution_and_reference_guidance_are_present() -> None:
    license_text = (REPOSITORY_ROOT / "LICENSE").read_text(encoding="utf-8")
    notice_text = (REPOSITORY_ROOT / "NOTICE").read_text(encoding="utf-8")
    guidance = (REPOSITORY_ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "MIT License" in license_text
    assert "Copyright 2026 BetterBorg" in notice_text
    assert "read-only reference" in guidance
    assert "Never edit" in guidance
