"""Allow-listed Git execution for BetterBorg-owned host operations."""

from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


class UnsafeGitError(RuntimeError):
    """Raised when a host Git operation falls outside the safe contract."""


_SAFE_SUBCOMMANDS = frozenset(
    {
        "add",
        "branch",
        "check-ref-format",
        "checkout",
        "commit",
        "diff",
        "fetch",
        "log",
        "ls-files",
        "merge",
        "merge-base",
        "rev-list",
        "rev-parse",
        "show",
        "show-ref",
        "status",
        "switch",
        "worktree",
    }
)
_BLOCKED_FLAGS = frozenset(
    {
        "--delete",
        "--force",
        "--force-with-lease",
        "--hard",
        "--no-verify",
        "--amend",
        "-B",
        "-D",
        "-f",
    }
)
_BLOCKED_COMBINATIONS = (
    ("checkout", "--"),
    ("checkout", "."),
    ("worktree", "remove", "--force"),
)

_FETCH_LOCKS: dict[str, threading.Lock] = {}
_FETCH_LOCKS_GUARD = threading.Lock()

# Git treats these variables as an alternative to repository discovery from
# ``cwd``.  They must not cross the SafeGit boundary: otherwise validation can
# inspect the bound worktree while the subsequent command targets a caller-
# selected repository, index, or object database.
_GIT_REPOSITORY_ENVIRONMENT = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_CONFIG_PARAMETERS",
        "GIT_DIR",
        "GIT_GRAFT_FILE",
        "GIT_IMPLICIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_INTERNAL_SUPER_PREFIX",
        "GIT_NAMESPACE",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PREFIX",
        "GIT_QUARANTINE_PATH",
        "GIT_REPLACE_REF_BASE",
        "GIT_SHALLOW_FILE",
        "GIT_WORK_TREE",
    }
)


def _has_long_option(argument: str, option: str) -> bool:
    return argument == option or argument.startswith(f"{option}=")


def _has_short_option(argument: str, option: str) -> bool:
    return (
        argument.startswith("-")
        and not argument.startswith("--")
        and option in argument[1:]
    )


def _assert_safe_branch_args(arguments: Sequence[str]) -> None:
    destructive_long = {"--copy", "--delete", "--force", "--move"}
    destructive_short = frozenset("CDFMcdm")
    if any(
        any(_has_long_option(argument, option) for option in destructive_long)
        or any(_has_short_option(argument, option) for option in destructive_short)
        for argument in arguments[1:]
    ):
        raise UnsafeGitError("branch deletion, copying, or movement is blocked")


def _assert_safe_fetch_args(arguments: Sequence[str]) -> None:
    for argument in arguments[1:]:
        if argument.startswith("+"):
            raise UnsafeGitError("force-update fetch refspecs are blocked")
        if _has_long_option(argument, "--refmap"):
            refmap = argument.partition("=")[2]
            if refmap.startswith("+"):
                raise UnsafeGitError("force-update fetch refspecs are blocked")
        if _has_long_option(argument, "--update-head-ok") or _has_short_option(
            argument, "u"
        ):
            raise UnsafeGitError("fetch may not update the checked-out branch")
        if any(
            _has_long_option(argument, option)
            for option in ("--prune", "--prune-tags")
        ) or _has_short_option(argument, "p"):
            raise UnsafeGitError("fetch ref deletion is blocked")


def _assert_safe_worktree_args(arguments: Sequence[str]) -> None:
    if len(arguments) < 2 or arguments[1] not in {"add", "list", "remove"}:
        raise UnsafeGitError("only worktree add, list, and remove are allowed")
    operation = arguments[1]
    operation_args = arguments[2:]
    if operation == "add" and any(
        _has_long_option(argument, "--force")
        or _has_short_option(argument, "B")
        or _has_short_option(argument, "f")
        for argument in operation_args
    ):
        raise UnsafeGitError("forced worktree branch movement is blocked")
    if operation == "remove" and (
        len(operation_args) != 1 or operation_args[0].startswith("-")
    ):
        raise UnsafeGitError("worktree removal requires exactly one path")
    if operation == "list" and any(
        argument not in {"--porcelain", "--verbose", "-v", "-z"}
        for argument in operation_args
    ):
        raise UnsafeGitError("unsupported worktree list arguments")


def assert_safe_git_args(arguments: Sequence[str]) -> None:
    """Reject destructive flags, discard commands, and unknown subcommands."""
    if not arguments:
        raise UnsafeGitError("empty Git argument list")
    if arguments[0] not in _SAFE_SUBCOMMANDS:
        raise UnsafeGitError(
            f"Git subcommand not in allow-list: {arguments[0]!r}"
        )
    blocked = next(
        (
            flag
            for flag in _BLOCKED_FLAGS
            if any(_has_long_option(argument, flag) for argument in arguments[1:])
        ),
        None,
    )
    if blocked is not None:
        raise UnsafeGitError(f"blocked Git flag: {blocked!r}")
    if any(_has_short_option(argument, "f") for argument in arguments[1:]):
        raise UnsafeGitError("blocked Git flag: '-f'")
    for combination in _BLOCKED_COMBINATIONS:
        if tuple(arguments[: len(combination)]) == combination:
            raise UnsafeGitError(
                f"blocked Git operation: {' '.join(combination)}"
            )
    subcommand = arguments[0]
    if subcommand in {"checkout", "switch"} and (
        len(arguments) != 2 or arguments[1].startswith("-")
    ):
        raise UnsafeGitError(
            f"only a single branch name is allowed for Git {subcommand}"
        )
    if subcommand == "branch":
        _assert_safe_branch_args(arguments)
    if subcommand == "fetch":
        _assert_safe_fetch_args(arguments)
    if subcommand == "worktree":
        _assert_safe_worktree_args(arguments)
    if subcommand == "merge" and any(
        argument in {"--abort", "--quit"} for argument in arguments[1:]
    ):
        raise UnsafeGitError("merge cleanup operations are blocked")


def _hardened_git_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment that cannot prompt or run repository hooks."""
    result = dict(environment) if environment is not None else dict(os.environ)
    for variable in _GIT_REPOSITORY_ENVIRONMENT:
        result.pop(variable, None)
    for variable in tuple(result):
        if variable == "GIT_CONFIG_COUNT" or variable.startswith(
            ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")
        ):
            result.pop(variable, None)
    result.update(
        {
            "GIT_ASKPASS": "true",
            "GIT_CONFIG_COUNT": "4",
            "GIT_CONFIG_KEY_0": "core.hooksPath",
            "GIT_CONFIG_KEY_1": "commit.gpgSign",
            "GIT_CONFIG_KEY_2": "tag.gpgSign",
            "GIT_CONFIG_KEY_3": "core.fsmonitor",
            "GIT_CONFIG_VALUE_0": os.devnull,
            "GIT_CONFIG_VALUE_1": "false",
            "GIT_CONFIG_VALUE_2": "false",
            "GIT_CONFIG_VALUE_3": "false",
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "never",
            "SSH_ASKPASS": "true",
        }
    )
    return result


def _common_git_directory(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--git-common-dir"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env=_hardened_git_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return f"fallback::{root.resolve()}"
    value = result.stdout.strip()
    if result.returncode != 0 or not value:
        return f"fallback::{root.resolve()}"
    path = Path(value)
    return str(path.resolve() if path.is_absolute() else (root / path).resolve())


def _fetch_lock(root: Path) -> threading.Lock:
    key = _common_git_directory(root)
    with _FETCH_LOCKS_GUARD:
        return _FETCH_LOCKS.setdefault(key, threading.Lock())


class SafeGit:
    """A Git runner permanently bound to one exact worktree root."""

    def __init__(self, cwd: Path) -> None:
        self._cwd = Path(cwd).resolve()

    @property
    def cwd(self) -> Path:
        return self._cwd

    def run(
        self,
        arguments: Sequence[str],
        *,
        check: bool = True,
        capture: bool = True,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run one validated command without prompts or repository hooks."""
        assert_safe_git_args(arguments)
        run_environment = _hardened_git_environment(env)
        self.assert_exact_worktree_root(env=run_environment)
        command = ["git", *arguments]
        options = {
            "cwd": str(self._cwd),
            "check": check,
            "capture_output": capture,
            "text": True,
            "env": run_environment,
            "timeout": timeout,
        }
        if arguments[0] == "fetch":
            with _fetch_lock(self._cwd):
                return subprocess.run(command, **options)
        return subprocess.run(command, **options)

    def assert_exact_worktree_root(
        self, *, env: Mapping[str, str] | None = None
    ) -> None:
        """Refuse parent-repository discovery from a nested directory."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=str(self._cwd),
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
                env=_hardened_git_environment(env),
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise UnsafeGitError(
                f"cannot validate Git worktree root {self._cwd}: {error}"
            ) from error
        if result.returncode != 0:
            raise UnsafeGitError(f"not a Git worktree root: {self._cwd}")
        top_level = Path(result.stdout.strip()).resolve()
        if top_level != self._cwd:
            raise UnsafeGitError(
                "refusing Git operation from a nested path: "
                f"cwd={self._cwd}, worktree_root={top_level}"
            )

    def current_branch(self) -> str:
        return self.run(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()

    def head_sha(self) -> str:
        return self.run(["rev-parse", "HEAD"]).stdout.strip()

    def branch_exists(self, branch: str) -> bool:
        result = self.run(
            ["show-ref", "--verify", f"refs/heads/{branch}"], check=False
        )
        return result.returncode == 0

    def is_valid_branch_name(self, branch: str) -> bool:
        return (
            self.run(["check-ref-format", "--branch", branch], check=False).returncode
            == 0
        )

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return (
            self.run(
                ["merge-base", "--is-ancestor", ancestor, descendant],
                check=False,
            ).returncode
            == 0
        )

    def is_clean(self, *, ignore_path: Callable[[str], bool] | None = None) -> bool:
        output = self.run(["status", "--porcelain", "-uall"]).stdout
        for line in output.splitlines():
            path = _status_path(line)
            if path and (ignore_path is None or not ignore_path(path)):
                return False
        return True

    def create_branch(self, branch: str, start_point: str) -> None:
        """Create a branch without checking it out or moving another ref."""
        self.run(["branch", branch, start_point])

    def add_worktree(self, path: Path, branch: str, *, base: str) -> None:
        if self.branch_exists(branch):
            self.run(["worktree", "add", str(path), branch])
        else:
            self.run(["worktree", "add", "-b", branch, str(path), base])

    def remove_worktree(self, path: Path) -> None:
        """Remove only a clean worktree; dirty task work is never discarded."""
        self.run(["worktree", "remove", str(path)])

    def fast_forward_branch(self, branch: str, target: str) -> bool:
        """Compare-and-swap a non-checked-out branch to a descendant only."""
        self.assert_exact_worktree_root()
        if not self.branch_exists(branch):
            return False
        reference = f"refs/heads/{branch}"
        if any(item.get("branch") == reference for item in self.worktree_list()):
            return False
        current_result = self.run(["rev-parse", "--verify", branch], check=False)
        target_result = self.run(["rev-parse", "--verify", target], check=False)
        if current_result.returncode or target_result.returncode:
            return False
        current = current_result.stdout.strip()
        destination = target_result.stdout.strip()
        if not current or current == destination:
            return False
        if not self.is_ancestor(current, destination):
            return False
        result = subprocess.run(
            ["git", "update-ref", reference, destination, current],
            cwd=str(self._cwd),
            check=False,
            capture_output=True,
            text=True,
            env=_hardened_git_environment(),
        )
        return result.returncode == 0

    def worktree_list(self) -> list[dict[str, str]]:
        output = self.run(["worktree", "list", "--porcelain"]).stdout
        entries: list[dict[str, str]] = []
        current: dict[str, str] = {}
        for line in output.splitlines():
            if not line:
                if current:
                    entries.append(current)
                    current = {}
            elif line.startswith("worktree "):
                current["path"] = line.removeprefix("worktree ")
            elif line.startswith("HEAD "):
                current["head"] = line.removeprefix("HEAD ")
            elif line.startswith("branch "):
                current["branch"] = line.removeprefix("branch ")
        if current:
            entries.append(current)
        return entries


def _status_path(line: str) -> str:
    path = line[3:] if len(line) > 3 else ""
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.strip().strip('"')
