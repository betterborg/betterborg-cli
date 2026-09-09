"""Provider-neutral execution support for logged-in native CLI adapters."""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from betterborg_cli.agent_runtime.api_tools import ApiAgentRole
from betterborg_cli.agent_runtime.base import (
    AgentArtifact,
    AgentResult,
    AgentRunSpec,
    AgentStatus,
    AgentUsage,
    BillingMode,
    CancellationToken,
    combine_agent_usage,
)
from betterborg_cli.agent_runtime.process import ProcessRunner, run_streamed
from betterborg_cli.agent_runtime.retry import SchemaRetry, run_with_transient_retry
from betterborg_cli.agent_runtime.structured import (
    StructuredResultError,
    validate_structured_result,
)
from betterborg_cli.progress import AgentActivity, AgentActivityKind


@dataclass(frozen=True, slots=True)
class NativePayload:
    """A provider-parsed payload or its provider-specific extraction error."""

    payload: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class NativeInvocation:
    """Provider-specific inputs and callbacks for one native CLI invocation."""

    command: Sequence[str]
    stdin_text: str
    load_payload: Callable[[], NativePayload]
    before_attempt: Callable[[], None] | None = None
    accept_payload_on_nonzero_exit: bool = False
    translate_event: (
        Callable[[Mapping[str, Any]], AgentActivity | None] | None
    ) = None


class NativeCliAdapter:
    """Shared lifecycle for subscription-backed native CLI adapters."""

    __slots__ = ()

    role: ApiAgentRole | str
    binary: str
    proc_runner: ProcessRunner
    artifacts: tuple[AgentArtifact, ...]
    transient_backoff_seconds: float
    transient_max_attempts: int
    name: str
    provider_label: str

    def _validate_native_configuration(self) -> None:
        self.role = ApiAgentRole(self.role)
        self.artifacts = tuple(self.artifacts)
        if not self.binary:
            raise ValueError(f"{self.provider_label} binary must not be empty")
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
        """Prepare and execute one provider-specific native invocation."""
        start = time.monotonic()
        if spec.billing_mode != BillingMode.SUBSCRIPTION:
            return self._result(
                spec,
                start,
                AgentStatus.FAILED,
                error=(
                    f"{self.provider_label} CLI adapter requires subscription "
                    "billing mode"
                ),
            )
        if cancel is not None and cancel.is_set():
            return self._cancelled(spec, start, attempts=0)
        if self.proc_runner is run_streamed and shutil.which(self.binary) is None:
            return self._result(
                spec,
                start,
                AgentStatus.FAILED,
                error=(
                    f"{self.provider_label} binary not found on PATH: {self.binary!r}"
                ),
            )

        try:
            spec.log_path.parent.mkdir(parents=True, exist_ok=True)
            spec.result_path.parent.mkdir(parents=True, exist_ok=True)
            with self._prepare_invocation(spec) as invocation:
                return self._run_native(spec, start, invocation, cancel)
        except OSError as error:
            return self._result(
                spec,
                start,
                AgentStatus.FAILED,
                error=f"unable to prepare {self.provider_label} process: {error}",
            )

    def _run_native(
        self,
        spec: AgentRunSpec,
        start: float,
        invocation: NativeInvocation,
        cancel: CancellationToken | None,
    ) -> AgentResult:
        environment = {**os.environ, **spec.env}
        attempt_usage: list[AgentUsage | None] = []
        attempts = 0
        prompt = invocation.stdin_text
        schema_retry = SchemaRetry()

        def run_once() -> int:
            nonlocal attempts
            attempts += 1
            _emit_activity(
                spec.activity_sink,
                AgentActivity(AgentActivityKind.THINKING),
            )
            if invocation.before_attempt is not None:
                invocation.before_attempt()
            line_observer = _line_observer(
                spec.activity_sink, invocation.translate_event
            )
            try:
                exit_code = self.proc_runner(
                    invocation.command,
                    spec.cwd,
                    invocation.stdin_text,
                    spec.log_path,
                    cancel,
                    environment,
                    line_observer,
                )
            finally:
                if line_observer is not None:
                    line_observer.finish()
            attempt_usage.append(self._extract_usage(spec.log_path))
            return exit_code

        def classify_transient_error(exit_code: int) -> str | None:
            if (
                exit_code != 0
                and invocation.accept_payload_on_nonzero_exit
                and self._has_valid_payload(invocation, spec)
            ):
                return None
            return self._classify_transient_error(spec, exit_code)

        # A missed schema restarts the invocation with the validating error and
        # the result it rejected appended to the prompt, so the next process
        # repairs what the last one sent rather than answering the same
        # question again. The correction reaches the process over stdin, which
        # this adapter never writes to the run log.
        while True:
            try:
                outcome = run_with_transient_retry(
                    run_once,
                    classify_transient_error,
                    cancel=cancel,
                    backoff_seconds=self.transient_backoff_seconds,
                    max_attempts=self.transient_max_attempts,
                )
            except OSError as error:
                return self._result(
                    spec,
                    start,
                    AgentStatus.FAILED,
                    error=(
                        f"unable to start {self.provider_label} process: {error}"
                    ),
                    attempts=attempts,
                )

            usage = combine_agent_usage(attempt_usage)
            if outcome.cancelled or (cancel is not None and cancel.is_set()):
                return self._cancelled(
                    spec,
                    start,
                    attempts=attempts,
                    usage=usage,
                )
            if outcome.exhausted:
                return self._cancelled(
                    spec,
                    start,
                    attempts=attempts,
                    usage=usage,
                    exit_code=outcome.exit_code,
                    error=f"transient retry exhausted: {outcome.transient_reason}",
                )
            extracted = (
                invocation.load_payload()
                if outcome.exit_code == 0
                or invocation.accept_payload_on_nonzero_exit
                else NativePayload()
            )
            if outcome.exit_code != 0 and extracted.payload is None:
                return self._result(
                    spec,
                    start,
                    AgentStatus.FAILED,
                    exit_code=outcome.exit_code,
                    error=self._terminal_error(spec, outcome.exit_code),
                    usage=usage,
                    attempts=attempts,
                )

            if extracted.payload is None:
                return self._result(
                    spec,
                    start,
                    AgentStatus.FAILED,
                    exit_code=outcome.exit_code,
                    error=(
                        extracted.error
                        or self._terminal_error(spec, outcome.exit_code)
                    ),
                    usage=usage,
                    attempts=attempts,
                )
            payload = extracted.payload
            try:
                validate_structured_result(payload, spec.schema)
            except StructuredResultError as error:
                correction = schema_retry.correction(error, payload)
                if correction is not None:
                    invocation = replace(
                        invocation,
                        stdin_text=f"{prompt}\n\n{correction}",
                    )
                    continue
                return self._result(
                    spec,
                    start,
                    AgentStatus.FAILED,
                    exit_code=outcome.exit_code,
                    error=f"structured result validation failed: {error}",
                    usage=usage,
                    attempts=attempts,
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
                    error=(
                        f"unable to persist {self.provider_label} result: {error}"
                    ),
                    usage=usage,
                    attempts=attempts,
                )
            return self._result(
                spec,
                start,
                AgentStatus.COMPLETED,
                exit_code=outcome.exit_code,
                payload=payload,
                result_path=spec.result_path,
                usage=usage,
                attempts=attempts,
            )

    @staticmethod
    def _has_valid_payload(
        invocation: NativeInvocation,
        spec: AgentRunSpec,
    ) -> bool:
        """Return whether a nonzero invocation left a usable result artifact."""
        payload = invocation.load_payload().payload
        if payload is None:
            return False
        try:
            validate_structured_result(payload, spec.schema)
        except StructuredResultError:
            return False
        return True

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
            provider=self.name,
            model=spec.model,
            artifacts=self.artifacts,
            attempts=attempts,
            retryable=retryable,
            resume_token=spec.resume_token,
        )

    def _prepare_invocation(
        self, spec: AgentRunSpec
    ) -> AbstractContextManager[NativeInvocation]:
        raise NotImplementedError

    def _extract_usage(self, log_path: Path) -> AgentUsage | None:
        raise NotImplementedError

    def _classify_transient_error(
        self, spec: AgentRunSpec, exit_code: int
    ) -> str | None:
        raise NotImplementedError

    def _terminal_error(self, spec: AgentRunSpec, exit_code: int) -> str:
        raise NotImplementedError


class _JsonlActivityObserver:
    """Translate complete JSONL records from bounded process fragments."""

    def __init__(
        self,
        activity_sink: Callable[[AgentActivity], None],
        translate_event: Callable[[Mapping[str, Any]], AgentActivity | None],
    ) -> None:
        self._activity_sink = activity_sink
        self._translate_event = translate_event
        self._pending = ""
        self._thinking = AgentActivity(AgentActivityKind.THINKING)

    def __call__(self, fragment: str) -> None:
        self._pending += fragment
        while "\n" in self._pending:
            record, _, self._pending = self._pending.partition("\n")
            self._observe_record(record)

    def finish(self) -> None:
        """Observe a final record when the stream omits its trailing newline."""
        if not self._pending:
            return
        record = self._pending
        self._pending = ""
        self._observe_record(record)

    def _observe_record(self, record: str) -> None:
        activity = self._thinking
        try:
            event = json.loads(record)
            if isinstance(event, Mapping):
                activity = self._translate_event(event) or self._thinking
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        except Exception:
            # Provider translation is observational and cannot affect a run.
            pass
        _emit_activity(self._activity_sink, activity)


def _line_observer(
    activity_sink: Callable[[AgentActivity], None] | None,
    translate_event: Callable[[Mapping[str, Any]], AgentActivity | None] | None,
) -> _JsonlActivityObserver | None:
    if activity_sink is None or translate_event is None:
        return None
    return _JsonlActivityObserver(activity_sink, translate_event)


def _emit_activity(
    activity_sink: Callable[[AgentActivity], None] | None,
    activity: AgentActivity,
) -> None:
    if activity_sink is None:
        return
    try:
        activity_sink(activity)
    except Exception:
        # Rendering and reporting callbacks are observational only.
        return
