"""Queued and dynamic agent adapter for hermetic consumers and tests."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from betterborg_cli.agent_runtime.base import (
    AgentAdapter,
    AgentArtifact,
    AgentCapabilities,
    AgentResult,
    AgentRunSpec,
    AgentStatus,
    AgentUsage,
    BillingMode,
    CancellationToken,
)
from betterborg_cli.agent_runtime.structured import (
    StructuredResultError,
    validate_structured_result,
)
from betterborg_cli.progress import AgentActivity

DynamicResponse = Callable[[AgentRunSpec], Any]


@dataclass(frozen=True, slots=True)
class MockResponse:
    """One queued mock invocation result."""

    payload: dict[str, Any] | None = None
    exit_code: int = 0
    delay_seconds: float = 0.0
    error: str | None = None
    dynamic: DynamicResponse | None = None
    usage: AgentUsage | None = None
    billing_mode: BillingMode | None = None
    artifacts: tuple[AgentArtifact, ...] = ()
    resume_token: str | None = None
    retryable: bool = False
    raise_error: Exception | None = None
    activities: tuple[AgentActivity, ...] = ()

    def __post_init__(self) -> None:
        if self.delay_seconds < 0:
            raise ValueError("mock delay must not be negative")


@dataclass(slots=True)
class MockAdapter(AgentAdapter):
    """Adapter whose responses are supplied in FIFO or dynamic order."""

    name: str = "mock"
    capabilities: AgentCapabilities = field(
        default_factory=lambda: AgentCapabilities(
            billing_modes=frozenset(BillingMode),
            structured_output=True,
            streaming=False,
            tool_allowlist=True,
            resumable=True,
        )
    )
    responses: list[MockResponse] = field(default_factory=list)
    calls: list[AgentRunSpec] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _response_consumed: threading.Event = field(
        default_factory=threading.Event,
        repr=False,
    )

    def queue(self, response: MockResponse) -> MockAdapter:
        """Append a response and return this adapter for fluent setup."""
        with self._lock:
            self.responses.append(response)
            self._response_consumed.clear()
        return self

    def wait_for_response_consumption(self, timeout: float | None = None) -> bool:
        """Wait until a run has claimed one queued response."""
        return self._response_consumed.wait(timeout)

    def run(
        self,
        spec: AgentRunSpec,
        *,
        cancel: CancellationToken | None = None,
    ) -> AgentResult:
        start = time.monotonic()
        with self._lock:
            self.calls.append(spec)
            call_number = len(self.calls)
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        spec.log_path.write_text(
            f"mock adapter invocation #{call_number}\n", encoding="utf-8"
        )
        if cancel is not None and cancel.is_set():
            return self._cancelled(spec, start)

        with self._lock:
            response = self.responses.pop(0) if self.responses else None
            if response is not None:
                self._response_consumed.set()
        if response is None:
            return AgentResult(
                status=AgentStatus.FAILED,
                exit_code=1,
                error="mock adapter has no queued responses",
                log_path=spec.log_path,
                duration_seconds=time.monotonic() - start,
                billing_mode=spec.billing_mode,
            )

        if response.delay_seconds and cancel is not None:
            if cancel.wait(response.delay_seconds):
                return self._cancelled(spec, start, response)
        elif response.delay_seconds:
            time.sleep(response.delay_seconds)

        if response.raise_error is not None:
            _emit_activities(spec, response.activities)
            raise response.raise_error
        if response.dynamic is not None:
            generated = response.dynamic(spec)
            response = (
                generated
                if isinstance(generated, MockResponse)
                else MockResponse(
                    payload=generated,
                    exit_code=response.exit_code,
                    error=response.error,
                    usage=response.usage,
                    billing_mode=response.billing_mode,
                    artifacts=response.artifacts,
                    activities=response.activities,
                    resume_token=response.resume_token,
                    retryable=response.retryable,
                )
            )

        _emit_activities(spec, response.activities)

        billing_mode = response.billing_mode or spec.billing_mode
        if response.exit_code != 0 or response.payload is None:
            return AgentResult(
                status=AgentStatus.FAILED,
                exit_code=response.exit_code,
                payload=response.payload,
                error=response.error or "mock agent failed",
                log_path=spec.log_path,
                duration_seconds=time.monotonic() - start,
                usage=response.usage,
                billing_mode=billing_mode,
                artifacts=response.artifacts,
                retryable=response.retryable,
                resume_token=response.resume_token,
            )

        try:
            validate_structured_result(response.payload, spec.schema)
        except StructuredResultError as error:
            return AgentResult(
                status=AgentStatus.FAILED,
                exit_code=0,
                error=f"structured result validation failed: {error}",
                log_path=spec.log_path,
                duration_seconds=time.monotonic() - start,
                usage=response.usage,
                billing_mode=billing_mode,
                artifacts=response.artifacts,
                retryable=response.retryable,
                resume_token=response.resume_token,
            )

        spec.result_path.parent.mkdir(parents=True, exist_ok=True)
        spec.result_path.write_text(
            json.dumps(response.payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return AgentResult(
            status=AgentStatus.COMPLETED,
            exit_code=0,
            payload=response.payload,
            log_path=spec.log_path,
            result_path=spec.result_path,
            duration_seconds=time.monotonic() - start,
            usage=response.usage,
            billing_mode=billing_mode,
            artifacts=response.artifacts,
            resume_token=response.resume_token,
        )

    @staticmethod
    def _cancelled(
        spec: AgentRunSpec,
        start: float,
        response: MockResponse | None = None,
    ) -> AgentResult:
        return AgentResult(
            status=AgentStatus.CANCELLED,
            exit_code=-1,
            error="agent run cancelled",
            log_path=spec.log_path,
            duration_seconds=time.monotonic() - start,
            usage=response.usage if response is not None else None,
            billing_mode=(response.billing_mode if response is not None else None)
            or spec.billing_mode,
            artifacts=response.artifacts if response is not None else (),
            resume_token=(response.resume_token if response is not None else None)
            or spec.resume_token,
            retryable=True,
        )


def _emit_activities(
    spec: AgentRunSpec,
    activities: tuple[AgentActivity, ...],
) -> None:
    sink = spec.activity_sink
    if sink is None:
        return
    for activity in activities:
        try:
            sink(activity)
        except Exception:
            # Rendering and reporting callbacks are observational only.
            continue
