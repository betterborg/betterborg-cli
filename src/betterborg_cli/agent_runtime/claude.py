"""Logged-in Claude Code CLI adapter with native host tool access."""

from __future__ import annotations

import base64
import json
import os
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from betterborg_cli.agent_runtime.api_tools import ApiAgentRole
from betterborg_cli.agent_runtime.base import (
    AgentArtifact,
    AgentCapabilities,
    AgentResult,
    AgentRunSpec,
    AgentStatus,
    AgentUsage,
    BillingMode,
    CancellationToken,
    combine_agent_usage,
)
from betterborg_cli.agent_runtime.process import ProcessRunner, run_streamed
from betterborg_cli.agent_runtime.retry import (
    DEFAULT_TRANSIENT_BACKOFF_SECONDS,
    DEFAULT_TRANSIENT_MAX_ATTEMPTS,
    run_with_transient_retry,
)
from betterborg_cli.agent_runtime.structured import (
    StructuredResultError,
    extract_json,
    validate_structured_result,
)

_PROVIDER = "claude"
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
_SYSTEM_PROMPT_WRAPPER_SENTINEL = "betterborg-claude-system-prompt"
_SYSTEM_PROMPT_WRAPPER_SCRIPT = """set -eu
encoded_size=$1
shift
prompt_file=$(mktemp "${TMPDIR:-/tmp}/betterborg-claude-system.XXXXXX")
cleanup() { rm -f -- "$prompt_file"; }
trap cleanup EXIT HUP INT TERM
dd iflag=fullblock bs="$encoded_size" count=1 status=none | base64 -d > "$prompt_file"
"$@" --system-prompt-file "$prompt_file"
"""


@dataclass(slots=True)
class ClaudeAdapter:
    """Run Claude Code in autonomous print mode for a BetterBorg role."""

    role: ApiAgentRole | str
    binary: str = "claude"
    proc_runner: ProcessRunner = run_streamed
    artifacts: tuple[AgentArtifact, ...] = ()
    transient_backoff_seconds: float = DEFAULT_TRANSIENT_BACKOFF_SECONDS
    transient_max_attempts: int = DEFAULT_TRANSIENT_MAX_ATTEMPTS
    name: str = field(default=_PROVIDER, init=False)
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
        self.role = ApiAgentRole(self.role)
        self.artifacts = tuple(self.artifacts)
        if not self.binary:
            raise ValueError("Claude binary must not be empty")
        if self.transient_backoff_seconds < 0:
            raise ValueError("transient backoff must not be negative")
        if self.transient_max_attempts < 1:
            raise ValueError("transient max attempts must be at least one")

    def run(
        self,
        spec: AgentRunSpec,
        *,
        cancel: CancellationToken | None = None,
    ) -> AgentResult:
        start = time.monotonic()
        if spec.billing_mode != BillingMode.SUBSCRIPTION:
            return self._result(
                spec,
                start,
                AgentStatus.FAILED,
                error="Claude CLI adapter requires subscription billing mode",
            )
        if cancel is not None and cancel.is_set():
            return self._cancelled(spec, start, attempts=0)
        if self.proc_runner is run_streamed and shutil.which(self.binary) is None:
            return self._result(
                spec,
                start,
                AgentStatus.FAILED,
                error=f"Claude binary not found on PATH: {self.binary!r}",
            )

        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        spec.result_path.parent.mkdir(parents=True, exist_ok=True)
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
            "json",
            "--model",
            spec.model,
        ]
        if spec.effort:
            claude_command.extend(("--effort", spec.effort))
        claude_command.append("--dangerously-skip-permissions")
        if spec.allowed_tools:
            claude_command.extend(("--allowed-tools", ",".join(spec.allowed_tools)))
        command, stdin_text = _command_and_stdin(
            claude_command,
            system_prompt=system_prompt,
            user_prompt=spec.user_prompt,
        )
        environment = {**os.environ, **spec.env}
        attempt_usage: list[AgentUsage | None] = []

        def run_once() -> int:
            exit_code = self.proc_runner(
                command,
                spec.cwd,
                stdin_text,
                spec.log_path,
                cancel,
                environment,
            )
            attempt_usage.append(_extract_usage(spec.log_path))
            return exit_code

        outcome = run_with_transient_retry(
            run_once,
            lambda exit_code: _classify_transient_error(
                spec.log_path,
                allow_text_fallback=exit_code != 0,
            ),
            cancel=cancel,
            backoff_seconds=self.transient_backoff_seconds,
            max_attempts=self.transient_max_attempts,
        )
        usage = combine_agent_usage(attempt_usage)
        if outcome.cancelled or (cancel is not None and cancel.is_set()):
            return self._cancelled(
                spec,
                start,
                attempts=outcome.attempts,
                usage=usage,
            )
        if outcome.exhausted:
            return self._cancelled(
                spec,
                start,
                attempts=outcome.attempts,
                usage=usage,
                exit_code=outcome.exit_code,
                error=f"transient retry exhausted: {outcome.transient_reason}",
            )
        if outcome.exit_code != 0:
            return self._result(
                spec,
                start,
                AgentStatus.FAILED,
                exit_code=outcome.exit_code,
                error=_terminal_error(spec.log_path, outcome.exit_code),
                usage=usage,
                attempts=outcome.attempts,
            )

        envelope = _result_envelope(spec.log_path)
        if envelope is None:
            return self._result(
                spec,
                start,
                AgentStatus.FAILED,
                exit_code=0,
                error="unable to extract Claude result envelope",
                usage=usage,
                attempts=outcome.attempts,
            )
        if envelope.get("is_error") or str(
            envelope.get("stop_reason") or ""
        ).lower() == "refusal":
            return self._result(
                spec,
                start,
                AgentStatus.FAILED,
                exit_code=0,
                error=_terminal_error(spec.log_path, 0),
                usage=usage,
                attempts=outcome.attempts,
            )

        result_text = envelope.get("result")
        if not isinstance(result_text, str):
            return self._result(
                spec,
                start,
                AgentStatus.FAILED,
                exit_code=0,
                error="Claude result envelope has no text result",
                usage=usage,
                attempts=outcome.attempts,
            )
        try:
            payload = extract_json(result_text)
            if not isinstance(payload, dict):
                raise StructuredResultError("result must be a JSON object")
            validate_structured_result(payload, spec.schema)
        except StructuredResultError as error:
            return self._result(
                spec,
                start,
                AgentStatus.FAILED,
                exit_code=0,
                error=f"structured result validation failed: {error}",
                usage=usage,
                attempts=outcome.attempts,
            )

        spec.result_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return self._result(
            spec,
            start,
            AgentStatus.COMPLETED,
            exit_code=0,
            payload=payload,
            result_path=spec.result_path,
            usage=usage,
            attempts=outcome.attempts,
        )

    def _cancelled(
        self,
        spec: AgentRunSpec,
        start: float,
        *,
        attempts: int,
        usage: AgentUsage | None = None,
        exit_code: int = -1,
        error: str = "agent run cancelled",
    ) -> AgentResult:
        return self._result(
            spec,
            start,
            AgentStatus.CANCELLED,
            exit_code=exit_code,
            error=error,
            usage=usage,
            attempts=attempts,
            retryable=True,
        )

    def _result(
        self,
        spec: AgentRunSpec,
        start: float,
        status: AgentStatus,
        *,
        exit_code: int | None = None,
        payload: dict[str, Any] | None = None,
        error: str | None = None,
        result_path: Path | None = None,
        usage: AgentUsage | None = None,
        attempts: int = 1,
        retryable: bool = False,
    ) -> AgentResult:
        return AgentResult(
            status=status,
            exit_code=exit_code,
            payload=payload,
            error=error,
            log_path=spec.log_path,
            result_path=result_path,
            duration_seconds=time.monotonic() - start,
            usage=usage,
            billing_mode=spec.billing_mode,
            provider=_PROVIDER,
            model=spec.model,
            artifacts=self.artifacts,
            attempts=attempts,
            retryable=retryable,
            resume_token=spec.resume_token,
        )


def _command_and_stdin(
    claude_command: list[str],
    *,
    system_prompt: str,
    user_prompt: str,
) -> tuple[list[str], str]:
    """Pass an arbitrarily large system prompt through a transient file."""
    encoded_prompt = base64.b64encode(system_prompt.encode()).decode("ascii")
    return (
        [
            "sh",
            "-c",
            _SYSTEM_PROMPT_WRAPPER_SCRIPT,
            _SYSTEM_PROMPT_WRAPPER_SENTINEL,
            str(len(encoded_prompt)),
            *claude_command,
        ],
        encoded_prompt + user_prompt,
    )


def _result_envelope(log_path: Path) -> dict[str, Any] | None:
    if not log_path.exists() or log_path.stat().st_size == 0:
        return None
    try:
        parsed = extract_json(log_path.read_text(encoding="utf-8", errors="replace"))
    except StructuredResultError:
        return None
    if isinstance(parsed, dict):
        return parsed
    for item in reversed(parsed):
        if isinstance(item, dict) and item.get("type") == "result":
            return item
    return next((item for item in reversed(parsed) if isinstance(item, dict)), None)


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
