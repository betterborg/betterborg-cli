"""Logged-in Claude Code CLI adapter with native host tool access."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from betterborg_cli.agent_runtime.api_tools import ApiAgentRole, is_read_only_tool_set
from betterborg_cli.agent_runtime.base import (
    AgentArtifact,
    AgentCapabilities,
    AgentRunSpec,
    AgentUsage,
    BillingMode,
)
from betterborg_cli.agent_runtime.native_cli import (
    NativeCliAdapter,
    NativeInvocation,
    NativePayload,
)
from betterborg_cli.agent_runtime.process import ProcessRunner, run_streamed
from betterborg_cli.agent_runtime.retry import (
    DEFAULT_TRANSIENT_BACKOFF_SECONDS,
    DEFAULT_TRANSIENT_MAX_ATTEMPTS,
)
from betterborg_cli.agent_runtime.structured import (
    StructuredResultError,
    extract_json,
)

_PROVIDER = "claude"
_CLAUDE_TOOL_NAMES = {
    "list_files": "Glob",
    "read_file": "Read",
    "search_text": "Grep",
    "apply_patch": "Edit",
    "run_command": "Bash",
}
_TRANSIENT_API_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
_SCHEMA_INSTRUCTIONS = """
## Output format requirement

Your response MUST be a single JSON object matching this JSON Schema:

```json
{schema}
```

Do not wrap the JSON in prose. Do not include any text before or after the
JSON object.
""".strip()


@dataclass(slots=True)
class ClaudeAdapter(NativeCliAdapter):
    """Run Claude Code in autonomous print mode for a Betterborg role."""

    role: ApiAgentRole | str
    binary: str = "claude"
    proc_runner: ProcessRunner = run_streamed
    artifacts: tuple[AgentArtifact, ...] = ()
    transient_backoff_seconds: float = DEFAULT_TRANSIENT_BACKOFF_SECONDS
    transient_max_attempts: int = DEFAULT_TRANSIENT_MAX_ATTEMPTS
    name: str = field(default=_PROVIDER, init=False)
    provider_label: str = field(default="Claude", init=False)
    capabilities: AgentCapabilities = field(
        default_factory=lambda: AgentCapabilities(
            billing_modes=frozenset({BillingMode.SUBSCRIPTION}),
            structured_output=True,
            streaming=True,
            tool_allowlist=True,
            resumable=True,
            host_capable=True,
        ),
        init=False,
    )

    def __post_init__(self) -> None:
        self._validate_native_configuration()

    @contextmanager
    def _prepare_invocation(
        self, spec: AgentRunSpec
    ) -> Iterator[NativeInvocation]:
        system_prompt = (
            spec.system_prompt.rstrip()
            + "\n\n"
            + _SCHEMA_INSTRUCTIONS.format(
                schema=json.dumps(spec.schema, indent=2, sort_keys=True)
            )
        )
        claude_command = [
            self.binary,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            spec.model,
        ]
        if spec.effort:
            claude_command.extend(("--effort", spec.effort))
        if is_read_only_tool_set(spec.allowed_tools):
            claude_command.extend(("--permission-mode", "plan"))
        else:
            claude_command.append("--dangerously-skip-permissions")
        if spec.allowed_tools:
            allowed_tools = tuple(
                dict.fromkeys(
                    _CLAUDE_TOOL_NAMES.get(tool, tool)
                    for tool in spec.allowed_tools
                )
            )
            claude_command.extend(("--allowed-tools", ",".join(allowed_tools)))
        system_prompt_path = _write_system_prompt(spec.log_path.parent, system_prompt)
        claude_command.extend(("--system-prompt-file", str(system_prompt_path)))
        try:
            yield NativeInvocation(
                command=claude_command,
                stdin_text=spec.user_prompt,
                load_payload=lambda: self._load_payload(spec.log_path),
            )
        finally:
            system_prompt_path.unlink(missing_ok=True)

    def _load_payload(self, log_path: Path) -> NativePayload:
        envelope = _result_envelope(log_path)
        if envelope is None:
            return NativePayload(error="unable to extract Claude result envelope")
        if envelope.get("is_error") or str(
            envelope.get("stop_reason") or ""
        ).lower() == "refusal":
            return NativePayload(error=_terminal_error(log_path, 0))

        result_text = envelope.get("result")
        if not isinstance(result_text, str):
            return NativePayload(error="Claude result envelope has no text result")
        try:
            payload = extract_json(result_text)
            if not isinstance(payload, dict):
                raise StructuredResultError("result must be a JSON object")
        except StructuredResultError as error:
            return NativePayload(
                error=f"structured result validation failed: {error}"
            )
        return NativePayload(payload=payload)

    def _extract_usage(self, log_path: Path) -> AgentUsage | None:
        return _extract_usage(log_path)

    def _classify_transient_error(
        self, log_path: Path, exit_code: int
    ) -> str | None:
        return _classify_transient_error(
            log_path,
            allow_text_fallback=exit_code != 0,
        )

    def _terminal_error(self, log_path: Path, exit_code: int) -> str:
        return _terminal_error(log_path, exit_code)


def _write_system_prompt(directory: Path, system_prompt: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=directory,
        prefix=".betterborg-claude-system-",
        suffix=".txt",
        text=True,
    )
    path = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as prompt_file:
            prompt_file.write(system_prompt)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _result_envelope(log_path: Path) -> dict[str, Any] | None:
    if not log_path.exists() or log_path.stat().st_size == 0:
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    parsed_objects: list[dict[str, Any]] = []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        for line in text.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                parsed_objects.append(event)
    else:
        if isinstance(parsed, dict):
            parsed_objects.append(parsed)
        elif isinstance(parsed, list):
            parsed_objects.extend(item for item in parsed if isinstance(item, dict))

    for item in reversed(parsed_objects):
        if item.get("type") == "result":
            return item
    return parsed_objects[-1] if parsed_objects else None


def _classify_transient_error(
    log_path: Path,
    *,
    allow_text_fallback: bool,
) -> str | None:
    if not log_path.exists() or log_path.stat().st_size == 0:
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    envelope = _result_envelope(log_path)
    if isinstance(envelope, Mapping) and envelope.get("is_error"):
        status = envelope.get("api_error_status")
        result = str(envelope.get("result") or "")
        if isinstance(status, int) and status in _TRANSIENT_API_STATUSES:
            return f"transient Claude API {status}: {result[:200]}"
        if any(
            marker in result.lower()
            for marker in (
                "out of extra usage",
                "usage limit",
                "rate limit",
                "overload",
                "connection error",
                "connection reset",
                "fetch failed",
            )
        ):
            return f"transient Claude error: {result[:200]}"
        return None
    if isinstance(envelope, Mapping) and (
        envelope.get("is_error") is False
        or (
            "is_error" not in envelope
            and envelope.get("type") == "result"
            and envelope.get("subtype") == "success"
        )
    ):
        return None
    if allow_text_fallback and any(
        marker in text.lower()
        for marker in ("out of extra usage", "usage limit", "rate limit")
    ):
        return f"transient Claude CLI error: {text.strip()[-200:]}"
    return None


def _terminal_error(log_path: Path, exit_code: int) -> str:
    envelope = _result_envelope(log_path)
    if envelope is not None:
        result = str(envelope.get("result") or "").strip()
        if result:
            prefix = (
                f"Claude exited {exit_code}"
                if exit_code != 0
                else "Claude reported an error"
            )
            return f"{prefix}: {result[:1000]}"
    if log_path.exists():
        lines = [
            line.strip()
            for line in log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if line.strip()
        ]
        if lines:
            return f"Claude exited {exit_code}: {lines[-1][-1000:]}"
    return f"Claude exited {exit_code}"


def _extract_usage(log_path: Path) -> AgentUsage | None:
    envelope = _result_envelope(log_path)
    if envelope is None:
        return None
    usage = envelope.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}

    def number(value: Any) -> int | float | None:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return value
        return None

    values = {
        "cost_usd": number(envelope.get("total_cost_usd")),
        "tokens_input": number(usage.get("input_tokens")),
        "tokens_output": number(usage.get("output_tokens")),
        "tokens_cache_read": number(usage.get("cache_read_input_tokens")),
        "tokens_cache_write": number(usage.get("cache_creation_input_tokens")),
        "num_turns": number(envelope.get("num_turns")),
    }
    return AgentUsage(**values)
