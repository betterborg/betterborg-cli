"""Build a bounded, allowlisted workspace for repository analysis."""

from __future__ import annotations

import fnmatch
import json
import os
import shutil
import stat
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from betterborg_cli.agent_runtime.base import CancellationToken

Clock = Callable[[], float]

#: Index of the evidence workspace, written beside the copied files. Analysis is
#: told to open it first, so a model naming it as evidence is describing where it
#: looked rather than what it found; it is never itself a manifest file.
ANALYSIS_INPUT_FILENAME = "analysis_input.json"


@dataclass(frozen=True)
class DiscoveryLimits:
    """Resource caps for one discovery run."""

    per_file_bytes: int = 128 * 1024
    total_bytes: int = 2 * 1024 * 1024
    deadline_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.per_file_bytes <= 0:
            raise ValueError("per_file_bytes must be positive")
        if self.total_bytes <= 0:
            raise ValueError("total_bytes must be positive")
        if self.deadline_seconds < 0:
            raise ValueError("deadline_seconds must be non-negative")


def discovery_limits_from_mapping(raw: object) -> DiscoveryLimits:
    """Parse known limit fields, falling back to the safe defaults."""
    if not isinstance(raw, Mapping):
        return DiscoveryLimits()
    allowed = {
        key: raw[key]
        for key in ("per_file_bytes", "total_bytes", "deadline_seconds")
        if key in raw
    }
    return DiscoveryLimits(**allowed)


@dataclass(frozen=True)
class DiscoveryFile:
    """One copied file excerpt in the sanitized workspace."""

    path: str
    workspace_path: str
    category: str
    size_bytes: int
    copied_bytes: int
    truncated: bool = False
    truncation_reason: str | None = None


@dataclass(frozen=True)
class DiscoveryOmission:
    """Evidence intentionally omitted from the sanitized workspace."""

    path: str
    reason: str
    category: str | None = None
    size_bytes: int | None = None


@dataclass(frozen=True)
class DiscoveryManifest:
    """Authoritative description of analyzer-visible evidence."""

    repo_name: str
    files_root: str = "files"
    files: list[DiscoveryFile] = field(default_factory=list)
    omitted: list[DiscoveryOmission] = field(default_factory=list)
    total_copied_bytes: int = 0
    limits: DiscoveryLimits = field(default_factory=DiscoveryLimits)
    deadline_exceeded: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible manifest representation."""
        return asdict(self)


@dataclass(frozen=True)
class _Candidate:
    path: Path
    rel_path: str
    category: str
    size_bytes: int
    device: int
    inode: int


_REPOSITORY_WRAPPER_BASENAMES = frozenset(
    {"composer.phar", "gradlew", "gradlew.bat", "mvnw", "mvnw.cmd", "pants"}
)

_SKIPPED_DIR_NAMES = {
    ".betterborg-analysis",
    ".betterborg-task",
    ".betterborg",
    ".git",
    ".hg",
    ".mypy_cache",
    ".orchestry",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "bower_components",
    "dist",
    "node_modules",
    "target",
    "vendor",
    "venv",
}

_ROOT_DOC_PREFIXES = (
    "agents",
    "architecture",
    "changelog",
    "claude",
    "code_of_conduct",
    "contributing",
    "license",
    "readme",
    "security",
)
_DOC_SUFFIXES = {"", ".md", ".mdx", ".rst", ".txt"}

_CONFIG_BASENAMES = {
    ".editorconfig",
    ".env.example",
    ".env.sample",
    ".flake8",
    ".gitlab-ci.yml",
    ".gitlab-ci.yaml",
    ".mise.toml",
    ".node-version",
    ".npmrc",
    ".nvmrc",
    ".pre-commit-config.yaml",
    ".prettierrc",
    ".prettierrc.json",
    ".prettierrc.toml",
    ".prettierrc.yml",
    ".prettierrc.yaml",
    ".python-version",
    ".ruby-version",
    ".tool-versions",
    ".yarnrc",
    ".yarnrc.yml",
    "biome.json",
    "bun.lockb",
    "cargo.lock",
    "cargo.toml",
    "composer.json",
    "composer.lock",
    "compose.yaml",
    "compose.yml",
    "deno.json",
    "deno.jsonc",
    "docker-compose.yaml",
    "docker-compose.yml",
    "dockerfile",
    "gemfile",
    "gemfile.lock",
    "go.mod",
    "go.sum",
    "go.work",
    "go.work.sum",
    "gradle.properties",
    "gradle-wrapper.properties",
    "justfile",
    "lerna.json",
    "makefile",
    "maven-wrapper.properties",
    "mypy.ini",
    "nx.json",
    "package-lock.json",
    "package.json",
    "pipfile",
    "pipfile.lock",
    "pnpm-lock.yaml",
    "pnpm-workspace.yaml",
    "poetry.lock",
    "pom.xml",
    "pyproject.toml",
    "requirements.txt",
    "ruff.toml",
    "setup.cfg",
    "setup.py",
    "taskfile.yml",
    "taskfile.yaml",
    "turbo.json",
    "uv.lock",
    "yarn.lock",
}

_CONFIG_PATTERNS = (
    "*.code-workspace",
    "*.config.js",
    "*.config.cjs",
    "*.config.mjs",
    "*.config.ts",
    "build.gradle",
    "build.gradle.kts",
    "compose.*.yaml",
    "compose.*.yml",
    "docker-compose.*.yaml",
    "docker-compose.*.yml",
    "eslint.config.*",
    "jest.config.*",
    "next.config.*",
    "playwright.config.*",
    "requirements-*.txt",
    "requirements.*.txt",
    "settings.gradle",
    "settings.gradle.kts",
    "tsconfig*.json",
    "vite.config.*",
    "vitest.config.*",
)

_SCRIPT_DIRS = {"bin", "hack", "script", "scripts", "tools"}
_SCRIPT_SUFFIXES = {
    ".bash",
    ".cjs",
    ".js",
    ".mjs",
    ".pl",
    ".ps1",
    ".py",
    ".rb",
    ".sh",
    ".ts",
    ".zsh",
}
_CATEGORY_PRIORITY = {
    "manifest": 0,
    "ci": 1,
    "config": 2,
    "script": 3,
    "documentation": 4,
}


def build_discovery_workspace(
    repo_root: Path | str,
    workspace_dir: Path | str,
    *,
    limits: DiscoveryLimits | None = None,
    per_file_bytes: int | None = None,
    total_bytes: int | None = None,
    deadline_seconds: float | None = None,
    deadline_monotonic: float | None = None,
    clock: Clock = time.monotonic,
    cancel: CancellationToken | None = None,
) -> DiscoveryManifest:
    """Copy bounded, allowlisted evidence into a sanitized workspace."""
    _cancellation_checkpoint(cancel)
    repo = Path(repo_root).resolve()
    if not repo.is_dir():
        raise ValueError(f"repository root is not a directory: {repo}")

    effective_limits = _resolve_limits(
        limits,
        per_file_bytes=per_file_bytes,
        total_bytes=total_bytes,
        deadline_seconds=deadline_seconds,
    )
    workspace = Path(workspace_dir).resolve()
    _prepare_workspace(repo, workspace)
    _cancellation_checkpoint(cancel)

    deadline = (
        deadline_monotonic
        if deadline_monotonic is not None
        else clock() + effective_limits.deadline_seconds
    )
    candidates, omitted, deadline_exceeded = _collect_candidates(
        repo,
        workspace=workspace,
        deadline=deadline,
        clock=clock,
        cancel=cancel,
    )

    copied_files: list[DiscoveryFile] = []
    total_copied = 0
    for candidate in sorted(candidates, key=_candidate_sort_key):
        _cancellation_checkpoint(cancel)
        if _deadline_expired(clock, deadline):
            deadline_exceeded = True
            omitted.append(
                DiscoveryOmission(
                    path=candidate.rel_path,
                    reason="deadline_exceeded",
                    category=candidate.category,
                    size_bytes=candidate.size_bytes,
                )
            )
            break

        remaining = effective_limits.total_bytes - total_copied
        if remaining <= 0:
            omitted.append(
                DiscoveryOmission(
                    path=candidate.rel_path,
                    reason="total_byte_cap",
                    category=candidate.category,
                    size_bytes=candidate.size_bytes,
                )
            )
            continue

        copied = min(candidate.size_bytes, effective_limits.per_file_bytes, remaining)
        if copied <= 0 and candidate.size_bytes > 0:
            omitted.append(
                DiscoveryOmission(
                    path=candidate.rel_path,
                    reason="total_byte_cap",
                    category=candidate.category,
                    size_bytes=candidate.size_bytes,
                )
            )
            continue

        workspace_rel = _workspace_file_path(candidate.rel_path)
        output_path = workspace / workspace_rel
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _cancellation_checkpoint(cancel)
        data, read_error = _read_candidate(candidate, copied)
        if read_error is not None:
            omitted.append(
                DiscoveryOmission(
                    path=candidate.rel_path,
                    reason=read_error,
                    category=candidate.category,
                    size_bytes=candidate.size_bytes,
                )
            )
            continue
        _cancellation_checkpoint(cancel)
        output_path.write_bytes(data)
        copied_bytes = len(data)
        total_copied += copied_bytes
        truncated = copied_bytes < candidate.size_bytes
        truncation_reason: str | None = None
        if truncated:
            if copied_bytes >= remaining:
                truncation_reason = "total_byte_cap"
            elif copied_bytes >= effective_limits.per_file_bytes:
                truncation_reason = "per_file_byte_cap"
            else:
                truncation_reason = "read_short"
        copied_files.append(
            DiscoveryFile(
                path=candidate.rel_path,
                workspace_path=workspace_rel,
                category=candidate.category,
                size_bytes=candidate.size_bytes,
                copied_bytes=copied_bytes,
                truncated=truncated,
                truncation_reason=truncation_reason,
            )
        )

    _cancellation_checkpoint(cancel)
    manifest = DiscoveryManifest(
        repo_name=repo.name,
        files=copied_files,
        omitted=omitted,
        total_copied_bytes=total_copied,
        limits=effective_limits,
        deadline_exceeded=deadline_exceeded,
    )
    (workspace / ANALYSIS_INPUT_FILENAME).write_text(
        json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _resolve_limits(
    limits: DiscoveryLimits | None,
    *,
    per_file_bytes: int | None,
    total_bytes: int | None,
    deadline_seconds: float | None,
) -> DiscoveryLimits:
    base = limits or DiscoveryLimits()
    return DiscoveryLimits(
        per_file_bytes=(
            per_file_bytes if per_file_bytes is not None else base.per_file_bytes
        ),
        total_bytes=total_bytes if total_bytes is not None else base.total_bytes,
        deadline_seconds=(
            deadline_seconds
            if deadline_seconds is not None
            else base.deadline_seconds
        ),
    )


def _prepare_workspace(repo: Path, workspace: Path) -> None:
    if workspace == repo:
        raise ValueError("analysis workspace must not be the repository root")
    if workspace == Path(workspace.anchor):
        raise ValueError("refusing to use filesystem root as analysis workspace")
    if workspace in repo.parents:
        raise ValueError("analysis workspace must not contain the repository root")
    if repo in workspace.parents:
        raise ValueError("analysis workspace must not be inside the repository root")
    if workspace.exists():
        if workspace.is_dir():
            shutil.rmtree(workspace)
        else:
            workspace.unlink()
    workspace.mkdir(parents=True, exist_ok=False)


def _collect_candidates(
    repo: Path,
    *,
    workspace: Path,
    deadline: float,
    clock: Clock,
    cancel: CancellationToken | None,
) -> tuple[list[_Candidate], list[DiscoveryOmission], bool]:
    candidates: list[_Candidate] = []
    omitted: list[DiscoveryOmission] = []
    deadline_exceeded = False

    for current_root_raw, dirnames, filenames in os.walk(repo, topdown=True):
        _cancellation_checkpoint(cancel)
        current_root = Path(current_root_raw)
        if _deadline_expired(clock, deadline):
            deadline_exceeded = True
            omitted.append(
                DiscoveryOmission(
                    path=_relative_posix(repo, current_root),
                    reason="deadline_exceeded",
                )
            )
            break

        kept_dirs: list[str] = []
        for dirname in sorted(dirnames):
            _cancellation_checkpoint(cancel)
            path = current_root / dirname
            rel = _relative_posix(repo, path)
            if path.is_symlink():
                omitted.append(DiscoveryOmission(path=rel, reason="symlink"))
                continue
            if path.resolve() == workspace:
                continue
            if dirname.lower() in _SKIPPED_DIR_NAMES:
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in sorted(filenames):
            _cancellation_checkpoint(cancel)
            if _deadline_expired(clock, deadline):
                deadline_exceeded = True
                omitted.append(
                    DiscoveryOmission(
                        path=_relative_posix(repo, current_root),
                        reason="deadline_exceeded",
                    )
                )
                break
            path = current_root / filename
            rel = _relative_posix(repo, path)
            try:
                file_stat = path.lstat()
            except OSError:
                omitted.append(DiscoveryOmission(path=rel, reason="stat_error"))
                continue
            if stat.S_ISLNK(file_stat.st_mode):
                omitted.append(DiscoveryOmission(path=rel, reason="symlink"))
                continue
            category = _allowed_category(rel)
            if category is None:
                continue
            if not stat.S_ISREG(file_stat.st_mode):
                omitted.append(
                    DiscoveryOmission(path=rel, reason="not_regular_file")
                )
                continue
            candidates.append(
                _Candidate(
                    path=path,
                    rel_path=rel,
                    category=category,
                    size_bytes=file_stat.st_size,
                    device=file_stat.st_dev,
                    inode=file_stat.st_ino,
                )
            )

    return candidates, omitted, deadline_exceeded


def _read_candidate(candidate: _Candidate, byte_limit: int) -> tuple[bytes, str | None]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate.path, flags)
    except OSError:
        return b"", "symlink" if _is_symlink(candidate.path) else "read_error"

    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            return b"", "not_regular_file"
        if (opened_stat.st_dev, opened_stat.st_ino) != (
            candidate.device,
            candidate.inode,
        ):
            return b"", "changed_during_discovery"
        if _is_symlink(candidate.path):
            return b"", "symlink"
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            return source.read(byte_limit), None
    except OSError:
        return b"", "read_error"
    finally:
        os.close(descriptor)


def _is_symlink(path: Path) -> bool:
    try:
        return stat.S_ISLNK(path.lstat().st_mode)
    except OSError:
        return False


def _allowed_category(rel_path: str) -> str | None:
    parts = rel_path.split("/")
    lower_name = parts[-1].lower()
    lower_rel = rel_path.lower()

    if lower_name in _REPOSITORY_WRAPPER_BASENAMES:
        return "script"
    if len(parts) == 1 and _is_doc_basename(lower_name):
        return "documentation"
    if _is_doc_path(parts, lower_name):
        return "documentation"
    if _is_devcontainer_config_path(parts, lower_name):
        return "config"
    if lower_name in _CONFIG_BASENAMES:
        if lower_name in {
            "cargo.toml",
            "composer.json",
            "gemfile",
            "go.mod",
            "go.work",
            "package.json",
            "pipfile",
            "pom.xml",
            "pyproject.toml",
            "setup.cfg",
            "setup.py",
        }:
            return "manifest"
        return "config"
    if any(fnmatch.fnmatchcase(lower_name, pattern) for pattern in _CONFIG_PATTERNS):
        return "config"
    if lower_rel.startswith(".github/workflows/") and lower_name.endswith(
        (".yaml", ".yml")
    ):
        return "ci"
    if lower_rel.startswith(".circleci/") and lower_name.endswith((".yaml", ".yml")):
        return "ci"
    if _is_script_path(parts, lower_name):
        return "script"
    return None


def _is_doc_basename(lower_name: str) -> bool:
    suffix = Path(lower_name).suffix
    if suffix not in _DOC_SUFFIXES:
        return False
    stem = lower_name[: -len(suffix)] if suffix else lower_name
    return stem in _ROOT_DOC_PREFIXES


def _is_doc_path(parts: list[str], lower_name: str) -> bool:
    if len(parts) < 2 or parts[0].lower() not in {
        "adr",
        "adrs",
        "doc",
        "docs",
        "documentation",
    }:
        return False
    return Path(lower_name).suffix in _DOC_SUFFIXES - {""}


def _is_devcontainer_config_path(parts: list[str], lower_name: str) -> bool:
    if len(parts) == 1:
        return lower_name == ".devcontainer.json"
    return parts[0].lower() == ".devcontainer" and lower_name == "devcontainer.json"


def _is_script_path(parts: list[str], lower_name: str) -> bool:
    if len(parts) < 2 or parts[0].lower() not in _SCRIPT_DIRS:
        return False
    suffix = Path(lower_name).suffix
    return suffix == "" or suffix in _SCRIPT_SUFFIXES


def _candidate_sort_key(candidate: _Candidate) -> tuple[int, int, str]:
    depth = candidate.rel_path.count("/")
    return (
        _CATEGORY_PRIORITY.get(candidate.category, 99),
        depth,
        candidate.rel_path,
    )


def _workspace_file_path(rel_path: str) -> str:
    parts = [part for part in rel_path.split("/") if part not in {"", ".", ".."}]
    if not parts:
        raise ValueError("empty repository-relative file path")
    return "/".join(["files", *parts])


def _relative_posix(root: Path, path: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return "." if rel == "" else rel


def _deadline_expired(clock: Clock, deadline: float) -> bool:
    return clock() >= deadline


def _cancellation_checkpoint(cancel: CancellationToken | None) -> None:
    if cancel is not None and cancel.is_set():
        raise KeyboardInterrupt
