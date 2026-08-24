"""Logged-in Codex CLI adapter with native host tool access."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from betterborg_cli.agent_runtime.api_tools import ApiAgentRole
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

_PROVIDER = "codex"
_SANDBOX = "danger-full-access"


@dataclass(slots=True)
class CodexAdapter(NativeCliAdapter):
    """Run Codex non-interactively for a BetterBorg role."""

    role: ApiAgentRole | str
    binary: str = "codex"
    proc_runner: ProcessRunner = run_streamed
    artifacts: tuple[AgentArtifact, ...] = ()
    transient_backoff_seconds: float = DEFAULT_TRANSIENT_BACKOFF_SECONDS
    transient_max_attempts: int = DEFAULT_TRANSIENT_MAX_ATTEMPTS
    name: str = field(default=_PROVIDER, init=False)
    provider_label: str = field(default="Codex", init=False)
    capabilities: AgentCapabilities = field(
        default_factory=lambda: AgentCapabilities(
            billing_modes=frozenset({BillingMode.SUBSCRIPTION}),
            structured_output=True,
            streaming=True,
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
        with tempfile.TemporaryDirectory(prefix="betterborg-codex-") as directory:
            temp_directory = Path(directory)
            schema_path = temp_directory / "schema.json"
            invocation_result_path = temp_directory / "result.json"
            schema_path.write_text(
                json.dumps(spec.schema, sort_keys=True), encoding="utf-8"
            )
            yield NativeInvocation(
                command=self._command(
                    spec,
                    schema_path,
                    invocation_result_path,
                ),
                stdin_text=_stdin_prompt(spec),
                load_payload=lambda: self._load_payload(
                    invocation_result_path, spec.log_path
                ),
                before_attempt=lambda: invocation_result_path.unlink(
                    missing_ok=True
                ),
            )

    def _command(
        self,
        spec: AgentRunSpec,
        schema_path: Path,
        invocation_result_path: Path,
    ) -> list[str]:
        command = [
            self.binary,
            "exec",
            "--json",
            "-C",
            str(spec.cwd),
            "-m",
            spec.model,
            "-s",
            _SANDBOX,
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--ephemeral",
            "--output-schema",
            str(schema_path),
            "-o",
            str(invocation_result_path),
        ]
        if spec.effort:
            command.extend(("-c", f"model_reasoning_effort={spec.effort}"))
        command.append("-")
        return command

    def _load_payload(
        self, invocation_result_path: Path, log_path: Path
    ) -> NativePayload:
        payload = _load_payload(invocation_result_path)
        if payload is None:
            return NativePayload(error=_terminal_error(log_path, 0))
        return NativePayload(payload=payload)

    def _extract_usage(self, log_path: Path) -> AgentUsage | None:
        return _extract_usage(log_path)

    def _classify_transient_error(
        self, log_path: Path, exit_code: int
    ) -> str | None:
        if exit_code == 0:
            return None
        return _classify_transient_error(log_path)

    def _terminal_error(self, log_path: Path, exit_code: int) -> str:
        return _terminal_error(log_path, exit_code)


def _stdin_prompt(spec: AgentRunSpec) -> str:
    system_prompt = spec.system_prompt.rstrip()
    user_prompt = spec.user_prompt.rstrip()
    return f"{system_prompt}\n\n<stdin>\n{user_prompt}\n</stdin>\n"


def _load_payload(result_path: Path) -> dict[str, Any] | None:
    if not result_path.exists() or result_path.stat().st_size == 0:
        return None
    try:
        payload = extract_json(
            result_path.read_text(encoding="utf-8", errors="replace")
        )
    except (OSError, StructuredResultError):
        return None
    return payload if isinstance(payload, dict) else None


def _classify_transient_error(log_path: Path) -> str | None:
    if not log_path.exists() or log_path.stat().st_size == 0:
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    patterns = (
        ("usage limit", "Codex usage limit hit"),
        ("status 429", "Codex rate limit"),
        ("rate limit", "Codex rate limit"),
        ("at capacity", "Codex model at capacity"),
        ("is overloaded", "Codex model overloaded"),
        ("model is overloaded", "Codex model overloaded"),
        ("try a different model", "Codex model at capacity"),
        ("status 503", "Codex service unavailable"),
        ("failed to refresh token", "Codex authentication refresh failed"),
        ("could not be refreshed", "Codex authentication refresh failed"),
        ("401 unauthorized", "Codex authentication failed"),
        ("stream disconnected", "Codex network transport failed"),
        ("error sending request", "Codex network transport failed"),
        ("connection reset by peer", "Codex network transport failed"),
    )
    for marker, description in patterns:
        if marker in lowered:
            return f"{description}: {_matching_log_line(text, marker)}"
    return None


def _matching_log_line(text: str, marker: str) -> str:
    for line in reversed(text.splitlines()):
        if marker in line.lower():
            return line.strip()[:300]
    return "see log"


def _terminal_error(log_path: Path, exit_code: int) -> str:
    message = (
        "Codex exited 0 and produced no result"
        if exit_code == 0
        else f"Codex exited {exit_code} without completing successfully"
    )
    if not log_path.exists():
        return message
    lines = [
        line.strip()
        for line in log_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if line.strip()
    ]
    if not lines:
        return message
    for line in reversed(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, Mapping):
            error = event.get("error")
            if isinstance(error, Mapping):
                error = error.get("message")
            detail = event.get("message") or error
            if isinstance(detail, str) and detail.strip():
                return f"{message}: {detail.strip()[:1000]}"
    return f"{message}: {lines[-1][-1000:]}"


def _extract_usage(log_path: Path) -> AgentUsage | None:
    if not log_path.exists() or log_path.stat().st_size == 0:
        return None
    total_input = 0
    cached_input = 0
    cache_write_input = 0
    output = 0
    turns = 0
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping) or event.get("type") != "turn.completed":
            continue
        usage = event.get("usage")
        if not isinstance(usage, Mapping):
            continue
        total_input += _token(usage.get("input_tokens"))
        cached_input += _token(usage.get("cached_input_tokens"))
        cache_write_input += _token(usage.get("cache_write_input_tokens"))
        output += _token(usage.get("output_tokens"))
        turns += 1
    if turns == 0:
        return None
    return AgentUsage(
        tokens_input=total_input,
        tokens_output=output,
        tokens_cache_read=cached_input,
        tokens_cache_write=cache_write_input,
        num_turns=turns,
    )


def _token(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0
