"""Contained file tools and role-gated commands for provider API agents.

The file operations in this module resolve every caller-supplied path beneath
one run directory.  ``run_command`` deliberately has a different security
contract: it avoids a shell, but the invoked program remains host-capable and
is therefore available only to trusted execution roles.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path, PureWindowsPath
from typing import Any

from betterborg_cli.agent_runtime.base import CancellationToken
from betterborg_cli.agent_runtime.process import run_captured


class ApiAgentRole(StrEnum):
    """Roles supported by provider API adapters."""

    ANALYSIS = "analysis"
    PLANNING = "planning"
    CODING = "coding"
    REVIEW = "review"
    MERGE = "merge"


class ApiToolError(RuntimeError):
    """Base error raised for a rejected or failed API tool operation."""


class PathContainmentError(ApiToolError):
    """Raised when a file-tool path is outside its run directory."""


class ToolGrantError(ApiToolError):
    """Raised when a role attempts to invoke a tool it was not granted."""


@dataclass(frozen=True, slots=True)
class SearchMatch:
    """One plain-text search result."""

    path: str
    line: int
    text: str


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Captured result from a shell-free host command."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class ApiToolDefinition:
    """Provider-neutral description and argument schema for one API tool."""

    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _PatchAction:
    operation: str
    relative_path: str
    body: tuple[str, ...]


_COMMAND_ROLES = frozenset(
    {ApiAgentRole.CODING, ApiAgentRole.REVIEW, ApiAgentRole.MERGE}
)
READ_ONLY_API_TOOLS = ("list_files", "read_file", "search_text")


def is_read_only_tool_set(allowed_tools: Sequence[str]) -> bool:
    """Return whether a run is confined to an explicit read-only tool set.

    Native CLI adapters translate this into a provider read-only sandbox, so
    an empty tool set is not read-only: it grants the adapter's default, which
    is unrestricted host access.
    """
    return bool(allowed_tools) and set(allowed_tools) <= set(READ_ONLY_API_TOOLS)


def _object_schema(
    properties: Mapping[str, Any], *, required: Sequence[str] = ()
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": dict(properties),
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    return schema


_TOOL_DEFINITIONS = {
    "list_files": ApiToolDefinition(
        name="list_files",
        description=(
            "List files recursively beneath a relative path in the run directory."
        ),
        parameters=_object_schema({"path": {"type": "string"}}),
    ),
    "search_text": ApiToolDefinition(
        name="search_text",
        description="Search UTF-8 files for literal text beneath a relative path.",
        parameters=_object_schema(
            {"query": {"type": "string"}, "path": {"type": "string"}},
            required=("query",),
        ),
    ),
    "read_file": ApiToolDefinition(
        name="read_file",
        description="Read one UTF-8 file at a relative path in the run directory.",
        parameters=_object_schema(
            {"path": {"type": "string"}}, required=("path",)
        ),
    ),
    "apply_patch": ApiToolDefinition(
        name="apply_patch",
        description="Apply a BetterBorg patch to files inside the run directory.",
        parameters=_object_schema(
            {"patch": {"type": "string"}}, required=("patch",)
        ),
    ),
    "run_command": ApiToolDefinition(
        name="run_command",
        description=(
            "Run a shell-free argv on the host from the trusted run directory."
        ),
        parameters=_object_schema(
            {
                "argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                }
            },
            required=("argv",),
        ),
    ),
}
_FILE_TOOL_NAMES = frozenset(_TOOL_DEFINITIONS) - {"run_command"}


def api_tool_definition(name: str) -> ApiToolDefinition:
    """Return the shared provider-neutral definition for a contained tool."""
    try:
        return _TOOL_DEFINITIONS[name]
    except KeyError as error:
        raise ValueError(f"unknown API tool: {name}") from error


def select_api_tool_names(
    tools: ContainedApiTools, requested: Sequence[str]
) -> frozenset[str]:
    """Apply a run allowlist to the tools granted by role and trust."""
    if not requested:
        return tools.available_tools
    return tools.available_tools.intersection(requested)


class ContainedApiTools:
    """Provider API tools bound to a run directory, role, and trust decision.

    File paths must be relative, must not contain ``..``, and must resolve
    beneath ``cwd`` after following symlinks.  This boundary applies only to
    the four file tools.  An argv passed to :meth:`run_command` does not use a
    shell, but the invoked program can still access the host with the current
    process's authority.
    """

    def __init__(
        self,
        cwd: Path,
        role: ApiAgentRole | str,
        *,
        workspace_trusted: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> None:
        try:
            root = Path(cwd).resolve(strict=True)
        except OSError as error:
            raise ValueError(f"API tool cwd does not exist: {cwd}") from error
        if not root.is_dir():
            raise ValueError(f"API tool cwd is not a directory: {cwd}")
        self._cwd = root
        self._role = ApiAgentRole(role)
        self._workspace_trusted = workspace_trusted
        self._environment = {**os.environ, **(env or {})}

    @property
    def cwd(self) -> Path:
        """Return the resolved run directory containing file-tool access."""
        return self._cwd

    @property
    def role(self) -> ApiAgentRole:
        """Return the API agent role used for tool grants."""
        return self._role

    @property
    def available_tools(self) -> frozenset[str]:
        """Return the tools that a provider may advertise for this run."""
        if self._workspace_trusted and self._role in _COMMAND_ROLES:
            return _FILE_TOOL_NAMES | {"run_command"}
        return _FILE_TOOL_NAMES

    def list_files(self, path: str = ".") -> tuple[str, ...]:
        """List files recursively beneath a contained file or directory."""
        candidate = self._contained_path(path, must_exist=True)
        return tuple(
            self._relative(item)
            for item in self._files_under(candidate)
        )

    def search_text(self, query: str, path: str = ".") -> tuple[SearchMatch, ...]:
        """Find literal ``query`` occurrences in UTF-8 text files."""
        if not query:
            raise ValueError("search query must not be empty")
        candidate = self._contained_path(path, must_exist=True)
        matches: list[SearchMatch] = []
        for file_path in self._files_under(candidate):
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, start=1):
                if query in line:
                    matches.append(
                        SearchMatch(
                            path=self._relative(file_path),
                            line=line_number,
                            text=line,
                        )
                    )
        return tuple(matches)

    def read_file(self, path: str) -> str:
        """Read one contained UTF-8 file."""
        candidate = self._contained_path(path, must_exist=True)
        if not candidate.is_file():
            raise ApiToolError(f"not a file: {path}")
        return candidate.read_text(encoding="utf-8")

    def apply_patch(self, patch: str) -> tuple[str, ...]:
        """Apply a contained ``*** Begin Patch`` file patch.

        Add, update, and delete actions are supported.  Every target is
        validated, including nonexistent add targets, before any change is
        written.
        """
        actions = _parse_patch(patch)
        prepared: list[tuple[_PatchAction, Path, str | None]] = []
        seen: set[Path] = set()
        for action in actions:
            must_exist = action.operation != "add"
            target = self._contained_path(
                action.relative_path,
                must_exist=must_exist,
            )
            if target in seen:
                raise ApiToolError(
                    f"patch contains duplicate target: {action.relative_path}"
                )
            seen.add(target)
            if action.operation == "add":
                if target.exists() or target.is_symlink():
                    raise ApiToolError(
                        f"patch add target exists: {action.relative_path}"
                    )
                content = "\n".join(action.body)
                if action.body:
                    content += "\n"
            elif action.operation == "update":
                if not target.is_file():
                    raise ApiToolError(
                        "patch update target is not a file: "
                        f"{action.relative_path}"
                    )
                content = _apply_update(
                    target.read_text(encoding="utf-8"),
                    action.body,
                    action.relative_path,
                )
            else:
                if not target.is_file() and not target.is_symlink():
                    raise ApiToolError(
                        "patch delete target is not a file: "
                        f"{action.relative_path}"
                    )
                content = None
            prepared.append((action, target, content))

        changed: list[str] = []
        for action, target, content in prepared:
            if action.operation == "delete":
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content or "", encoding="utf-8")
            changed.append(action.relative_path)
        return tuple(changed)

    def run_command(
        self,
        argv: Sequence[str],
        *,
        cancel: CancellationToken | None = None,
    ) -> CommandResult:
        """Run an argv directly on the host for a trusted execution role.

        No shell parses ``argv``.  That prevents shell metacharacters from
        becoming syntax, but it does not contain the selected program: the
        program remains host-capable. Cancellation terminates its complete
        process group and reports return code ``-1``.
        """
        if self._role not in _COMMAND_ROLES:
            raise ToolGrantError(
                f"role {self._role.value!r} is never granted run_command"
            )
        if not self._workspace_trusted:
            raise ToolGrantError("run_command requires a trusted workspace")
        if isinstance(argv, str | bytes) or not argv:
            raise ValueError("command must be a non-empty argv sequence")
        command = tuple(argv)
        if any(
            not isinstance(argument, str) or "\x00" in argument
            for argument in command
        ):
            raise ValueError("command arguments must be strings without NUL bytes")
        result = run_captured(
            command,
            cwd=self._cwd,
            env=self._environment,
            cancel=cancel,
        )
        return CommandResult(
            argv=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        cancel: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """Execute a granted tool and return a JSON-serializable value."""
        if name not in self.available_tools:
            raise ToolGrantError(f"tool is not granted: {name}")
        if name == "list_files":
            return {"files": list(self.list_files(**arguments))}
        if name == "search_text":
            return {
                "matches": [asdict(match) for match in self.search_text(**arguments)]
            }
        if name == "read_file":
            return {"content": self.read_file(**arguments)}
        if name == "apply_patch":
            return {"changed_files": list(self.apply_patch(**arguments))}
        return asdict(self.run_command(**arguments, cancel=cancel))

    def _contained_path(self, path: str, *, must_exist: bool) -> Path:
        if not isinstance(path, str) or not path:
            raise PathContainmentError("file-tool path must be a non-empty string")
        requested = Path(path)
        windows_path = PureWindowsPath(path)
        if requested.is_absolute() or windows_path.is_absolute() or windows_path.drive:
            raise PathContainmentError(f"absolute file-tool path is forbidden: {path}")
        if ".." in requested.parts:
            raise PathContainmentError(f"file-tool traversal is forbidden: {path}")

        candidate = self._cwd / requested
        try:
            resolved = candidate.resolve(strict=must_exist)
        except (OSError, RuntimeError) as error:
            raise PathContainmentError(
                f"cannot resolve file-tool path: {path}"
            ) from error
        if not resolved.is_relative_to(self._cwd):
            raise PathContainmentError(
                f"file-tool path escapes the run directory: {path}"
            )
        return candidate

    def _files_under(self, candidate: Path) -> list[Path]:
        if candidate.is_file():
            return [candidate]
        if not candidate.is_dir():
            raise ApiToolError(f"not a file or directory: {self._relative(candidate)}")

        files: list[Path] = []
        for directory, directory_names, file_names in os.walk(
            candidate,
            followlinks=False,
        ):
            directory_path = Path(directory)
            for name in directory_names:
                self._contained_path(
                    self._relative(directory_path / name),
                    must_exist=True,
                )
            for name in file_names:
                file_path = self._contained_path(
                    self._relative(directory_path / name),
                    must_exist=True,
                )
                if file_path.is_file():
                    files.append(file_path)
        return sorted(files, key=self._relative)

    def _relative(self, path: Path) -> str:
        return path.relative_to(self._cwd).as_posix()


def _parse_patch(patch: str) -> tuple[_PatchAction, ...]:
    lines = patch.splitlines()
    if len(lines) < 3 or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
        raise ApiToolError("patch must be wrapped in *** Begin Patch and *** End Patch")

    actions: list[_PatchAction] = []
    index = 1
    prefixes = {
        "*** Add File: ": "add",
        "*** Update File: ": "update",
        "*** Delete File: ": "delete",
    }
    while index < len(lines) - 1:
        header = lines[index]
        matched = next(
            (
                (prefix, operation)
                for prefix, operation in prefixes.items()
                if header.startswith(prefix)
            ),
            None,
        )
        if matched is None:
            raise ApiToolError(f"invalid patch action header: {header}")
        prefix, operation = matched
        relative_path = header.removeprefix(prefix)
        if not relative_path:
            raise ApiToolError("patch action path must not be empty")
        index += 1
        body: list[str] = []
        while index < len(lines) - 1 and not any(
            lines[index].startswith(candidate) for candidate in prefixes
        ):
            body.append(lines[index])
            index += 1
        if operation == "add":
            if any(not line.startswith("+") for line in body):
                raise ApiToolError(
                    f"added file lines must start with '+': {relative_path}"
                )
            body = [line[1:] for line in body]
        elif operation == "delete" and body:
            raise ApiToolError(f"delete action must not have a body: {relative_path}")
        actions.append(_PatchAction(operation, relative_path, tuple(body)))

    if not actions:
        raise ApiToolError("patch must contain at least one file action")
    return tuple(actions)


def _apply_update(original: str, body: tuple[str, ...], path: str) -> str:
    hunks: list[list[str]] = []
    current: list[str] | None = None
    for line in body:
        if line.startswith("@@"):
            current = []
            hunks.append(current)
        elif current is None:
            raise ApiToolError(f"update patch requires a hunk header: {path}")
        else:
            current.append(line)
    if not hunks:
        raise ApiToolError(f"update patch requires at least one hunk: {path}")

    original_lines = original.splitlines()
    cursor = 0
    for hunk in hunks:
        before: list[str] = []
        after: list[str] = []
        for line in hunk:
            if line == "\\ No newline at end of file":
                continue
            if not line or line[0] not in {" ", "+", "-"}:
                raise ApiToolError(f"invalid update hunk line for {path}: {line}")
            if line[0] in {" ", "-"}:
                before.append(line[1:])
            if line[0] in {" ", "+"}:
                after.append(line[1:])

        match = _find_lines(original_lines, before, cursor)
        if match is None:
            raise ApiToolError(f"update hunk does not match file: {path}")
        original_lines[match : match + len(before)] = after
        cursor = match + len(after)

    result = "\n".join(original_lines)
    if original.endswith("\n") and original_lines:
        result += "\n"
    return result


def _find_lines(lines: list[str], expected: list[str], start: int) -> int | None:
    if not expected:
        return start
    final_start = len(lines) - len(expected)
    for index in range(start, final_start + 1):
        if lines[index : index + len(expected)] == expected:
            return index
    return None
