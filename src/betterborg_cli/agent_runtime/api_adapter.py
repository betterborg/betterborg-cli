"""Provider-neutral execution support for HTTP API agent adapters."""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from betterborg_cli.agent_runtime.base import (
    AgentResult,
    AgentRunSpec,
    AgentStatus,
    AgentUsage,
    BillingMode,
    CancellationToken,
    combine_agent_usage,
)
from betterborg_cli.agent_runtime.retry import run_with_transient_retry
from betterborg_cli.agent_runtime.structured import (
    StructuredResultError,
    validate_structured_result,
)


@dataclass(frozen=True, slots=True)
class ApiCredentialRedactor:
    """Remove one active credential and provider credential-shaped text."""

    api_key: str | None
    credential_pattern: re.Pattern[str]

    def text(self, value: str) -> str:
        """Return redacted text."""
        if self.api_key:
            value = value.replace(self.api_key, "[REDACTED]")
        return self.credential_pattern.sub("[REDACTED]", value)

    def value(self, value: Any) -> Any:
        """Recursively redact strings in JSON-shaped values."""
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, Mapping):
            return {
                self.text(str(key)): self.value(item) for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.value(item) for item in value]
        return value


@dataclass(slots=True)
class _RequestState:
    response: Mapping[str, Any] | None = None
    error: Exception | None = None


@dataclass(slots=True)
class ApiRunContext:
    """Shared result, request, retry, logging, and redaction state for one run."""

    spec: AgentRunSpec
    provider: str
    redactor: ApiCredentialRedactor
    started_at: float = field(default_factory=time.monotonic)
    attempts: int = 0
    model: str = field(init=False)
    usage: list[AgentUsage] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.model = self.spec.model
        self.spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.spec.log_path.write_text("", encoding="utf-8")

    def result(
        self,
        status: AgentStatus,
        *,
        error: str | None = None,
        payload: dict[str, Any] | None = None,
        result_path: Path | None = None,
        retryable: bool = False,
    ) -> AgentResult:
        """Build a consistently attributed, credential-safe API result."""
        return AgentResult(
            status=status,
            exit_code=(
                0
                if status == AgentStatus.COMPLETED
                else -1
                if status == AgentStatus.CANCELLED
                else 1
            ),
            payload=payload,
            error=self.redactor.text(error or "") or None,
            log_path=self.spec.log_path,
            result_path=result_path,
            duration_seconds=time.monotonic() - self.started_at,
            usage=combine_agent_usage(self.usage),
            billing_mode=BillingMode.API,
            provider=self.provider,
            model=self.model,
            attempts=self.attempts,
            retryable=retryable,
            resume_token=self.spec.resume_token,
        )

    def cancelled(self) -> AgentResult:
        """Build the shared resumable cancellation result."""
        return self.result(
            AgentStatus.CANCELLED,
            error="agent run cancelled",
            retryable=True,
        )

    def append_log(self, event: Any) -> None:
        """Append one redacted JSON event to the run log."""
        with self.spec.log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(self.redactor.value(event), sort_keys=True) + "\n")

    def record_response(self, response: Mapping[str, Any], usage: AgentUsage) -> None:
        """Record provider-parsed response accounting and model metadata."""
        self.usage.append(usage)
        response_model = response.get("model")
        if isinstance(response_model, str) and response_model:
            self.model = response_model

    def complete(self, payload: Mapping[str, Any]) -> AgentResult:
        """Validate, redact, and persist a provider-submitted structured result."""
        redacted = self.redactor.value(dict(payload))
        if not isinstance(redacted, dict):
            raise AssertionError("redacting an object must preserve its shape")
        try:
            validate_structured_result(redacted, self.spec.schema)
        except StructuredResultError as error:
            return self.result(
                AgentStatus.FAILED,
                error=f"structured result validation failed: {error}",
            )
        self.spec.result_path.parent.mkdir(parents=True, exist_ok=True)
        self.spec.result_path.write_text(
            json.dumps(redacted, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return self.result(
            AgentStatus.COMPLETED,
            payload=redacted,
            result_path=self.spec.result_path,
        )

    def request(
        self,
        send: Callable[[CancellationToken | None], Mapping[str, Any]],
        *,
        api_error_type: type[Exception],
        cancel: CancellationToken | None,
        thread_name: str,
        backoff_seconds: float,
        max_attempts: int,
    ) -> Mapping[str, Any] | AgentResult:
        """Send one cancellable request with bounded transient retries."""
        state = _RequestState()

        def request_once() -> int:
            state.response = None
            state.error = None
            if cancel is None:
                try:
                    state.response = send(None)
                except Exception as error:
                    state.error = error
            else:
                if cancel.is_set():
                    return -1
                completed = threading.Event()

                def send_request() -> None:
                    try:
                        state.response = send(cancel)
                    except Exception as error:
                        state.error = error
                    finally:
                        completed.set()

                threading.Thread(
                    target=send_request,
                    name=thread_name,
                    daemon=True,
                ).start()
                while not completed.wait(0.05):
                    if cancel.is_set():
                        return -1
                if cancel.is_set():
                    return -1
            if state.error is not None:
                self.append_log({"error": str(state.error)})
                if isinstance(state.error, api_error_type):
                    status_code = getattr(state.error, "status_code", None)
                    return status_code if isinstance(status_code, int) else 1
                return 1
            return 0

        def classify_request(_exit_code: int) -> str | None:
            if isinstance(state.error, api_error_type) and getattr(
                state.error, "transient", False
            ):
                return str(state.error)
            return None

        retry = run_with_transient_retry(
            request_once,
            classify_request,
            cancel=cancel,
            backoff_seconds=backoff_seconds,
            max_attempts=max_attempts,
        )
        self.attempts += retry.attempts
        if (cancel is not None and cancel.is_set()) or retry.cancelled:
            return self.cancelled()
        if retry.exhausted:
            return self.result(
                AgentStatus.CANCELLED,
                error=f"transient retry exhausted: {retry.transient_reason}",
                retryable=True,
            )
        if retry.exit_code != 0 or state.response is None:
            return self.result(AgentStatus.FAILED, error=str(state.error))
        return state.response
