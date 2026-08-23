"""Logged-in Codex CLI adapter with native host tool access."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
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

_PROVIDER = "codex"
_SANDBOX = "danger-full-access"


@dataclass(slots=True)
class CodexAdapter:
    """Run Codex non-interactively for a BetterBorg role."""

    role: ApiAgentRole | str
    binary: str = "codex"
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
            resumable=True,
            host_capable=True,
        ),
        init=False,
    )

    def __post_init__(self) -> None:
        self.role = ApiAgentRole(self.role)
        self.artifacts = tuple(self.artifacts)
        if not self.binary:
            raise ValueError("Codex binary must not be empty")
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
                error="Codex CLI adapter requires subscription billing mode",
            )
        if cancel is not None and cancel.is_set():
            return self._cancelled(spec, start, attempts=0)
        if self.proc_runner is run_streamed and shutil.which(self.binary) is None:
            return self._result(
                spec,
                start,
                AgentStatus.FAILED,
                error=f"Codex binary not found on PATH: {self.binary!r}",
            )

        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        spec.result_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix="betterborg-codex-") as directory:
                temp_directory = Path(directory)
                schema_path = temp_directory / "schema.json"
                invocation_result_path = temp_directory / "result.json"
                schema_path.write_text(
                    json.dumps(spec.schema, sort_keys=True), encoding="utf-8"
                )
                command = self._command(spec, schema_path, invocation_result_path)
                return self._run_native(
                    spec,
                    start,
                    command,
                    invocation_result_path,
                    cancel,
                )
        except OSError as error:
            return self._result(
                spec,
                start,
                AgentStatus.FAILED,
                error=f"unable to prepare Codex process: {error}",
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

    def _run_native(
        self,
        spec: AgentRunSpec,
        start: float,
        command: list[str],
        invocation_result_path: Path,
        cancel: CancellationToken | None,
    ) -> AgentResult:
        environment = {**os.environ, **spec.env}
        attempt_usage: list[AgentUsage | None] = []
        attempts = 0

        def run_once() -> int:
            nonlocal attempts
            attempts += 1
            invocation_result_path.unlink(missing_ok=True)
            exit_code = self.proc_runner(
                command,
                spec.cwd,
                _stdin_prompt(spec),
                spec.log_path,
                cancel,
                environment,
            )
            attempt_usage.append(_extract_usage(spec.log_path))
            return exit_code

        def classify(exit_code: int) -> str | None:
            if exit_code == 0:
                return None
            return _classify_transient_error(spec.log_path)

        try:
            outcome = run_with_transient_retry(
                run_once,
                classify,
                cancel=cancel,
                backoff_seconds=self.transient_backoff_seconds,
                max_attempts=self.transient_max_attempts,
            )
        except OSError as error:
            return self._result(
                spec,
                start,
                AgentStatus.FAILED,
                error=f"unable to start Codex process: {error}",
                attempts=attempts,
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

        payload = _load_payload(invocation_result_path)
        if payload is None:
            return self._result(
                spec,
                start,
                AgentStatus.FAILED,
                exit_code=outcome.exit_code,
                error=_terminal_error(spec.log_path, outcome.exit_code),
                usage=usage,
                attempts=outcome.attempts,
            )
        try:
            validate_structured_result(payload, spec.schema)
        except StructuredResultError as error:
            return self._result(
                spec,
                start,
                AgentStatus.FAILED,
                exit_code=outcome.exit_code,
                error=f"structured result validation failed: {error}",
                usage=usage,
                attempts=outcome.attempts,
            )

        try:
            spec.result_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            return self._result(
                spec,
                start,
                AgentStatus.FAILED,
                exit_code=outcome.exit_code,
                payload=payload,
                error=f"unable to persist Codex result: {error}",
                usage=usage,
                attempts=outcome.attempts,
            )
        return self._result(
            spec,
            start,
            AgentStatus.COMPLETED,
            exit_code=outcome.exit_code,
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
