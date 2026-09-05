"""Provider-neutral execution support for HTTP API agent adapters."""

from __future__ import annotations

import json
import re
import shlex
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from betterborg_cli.agent_runtime.api_tools import ApiToolError, api_patch_paths
from betterborg_cli.agent_runtime.base import (
    AgentResult,
    AgentRunSpec,
    AgentStatus,
    AgentUsage,
    BillingMode,
    CancellationDeliveryError,
    CancellationRegistration,
    CancellationToken,
    ForceTarget,
    combine_agent_usage,
)
from betterborg_cli.agent_runtime.retry import SchemaRetry, run_with_transient_retry
from betterborg_cli.agent_runtime.structured import (
    StructuredResultError,
    validate_structured_result,
)
from betterborg_cli.progress import AgentActivity, AgentActivityKind


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


class AbortableApiRequest(Protocol):
    """One attempt-local API request owned until execution is cleaned up."""

    def execute(self) -> Mapping[str, Any]:
        """Execute the request and return its decoded provider response."""

    def abort(self) -> None:
        """Request nonblocking graceful cancellation; repeated calls are safe."""

    def force(self) -> None:
        """Request nonblocking forced cancellation; repeated calls are safe."""


ApiRequestFactory = Callable[
    [CancellationToken | None],
    AbortableApiRequest,
]


@dataclass(frozen=True, slots=True)
class SchemaCorrection:
    """A rejected submission handed back to the turn loop that produced it."""

    message: str


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
    schema_retry: SchemaRetry = field(default_factory=SchemaRetry)

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

    def emit_activity(self, activity: AgentActivity | None) -> None:
        """Emit one credential-safe observational activity when configured."""
        sink = self.spec.activity_sink
        if sink is None or activity is None:
            return
        detail = activity.detail
        if detail is not None:
            detail = self.redactor.text(detail)
        try:
            sink(AgentActivity(activity.kind, detail))
        except Exception:
            # Rendering and reporting callbacks are observational only.
            return

    def emit_tool_activity(self, name: str, arguments: Mapping[str, Any]) -> None:
        """Translate and emit one provider-neutral contained-tool activity."""
        self.emit_activity(_tool_activity(name, arguments))

    def record_response(self, response: Mapping[str, Any], usage: AgentUsage) -> None:
        """Record provider-parsed response accounting and model metadata."""
        self.usage.append(usage)
        response_model = response.get("model")
        if isinstance(response_model, str) and response_model:
            self.model = response_model

    def complete(
        self, payload: Mapping[str, Any]
    ) -> AgentResult | SchemaCorrection:
        """Validate, redact, and persist a provider-submitted structured result.

        A submission that misses the schema comes back as a correction while
        the retry budget lasts, so the turn loop can return the validating
        error to the agent and let it submit again in the same conversation.
        The correction describes and quotes the redacted payload, so it is safe
        to place wherever the redacted payload is safe. It is not what keeps a
        credential from the provider: the provider produced the submission, and
        a transport that replays its own turn sends it back regardless.
        """
        redacted = self.redactor.value(dict(payload))
        if not isinstance(redacted, dict):
            raise AssertionError("redacting an object must preserve its shape")
        try:
            validate_structured_result(redacted, self.spec.schema)
        except StructuredResultError as error:
            correction = self.schema_retry.correction(error, redacted)
            if correction is not None:
                return SchemaCorrection(correction)
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
        factory: ApiRequestFactory,
        *,
        api_error_type: type[Exception],
        cancel: CancellationToken | None,
        backoff_seconds: float,
        max_attempts: int,
    ) -> Mapping[str, Any] | AgentResult:
        """Send one cancellable request with bounded transient retries."""
        state: _RequestState | None = None

        def request_once() -> int:
            nonlocal state
            attempt = _RequestState()
            state = attempt
            if cancel is not None and cancel.is_set():
                return -1

            self.emit_activity(AgentActivity(AgentActivityKind.THINKING))

            request: AbortableApiRequest | None = None
            registration: CancellationRegistration | None = None
            try:
                request = factory(cancel)
                if cancel is not None:
                    try:
                        registration = cancel.register(
                            request.abort,
                            request.force,
                            force_target=ForceTarget(
                                ("api-request", id(request))
                            ),
                        )
                    except CancellationDeliveryError as error:
                        registration = error.registration
                        raise
                attempt.response = request.execute()
            except Exception as error:
                attempt.error = error
            finally:
                try:
                    if (
                        request is not None
                        and cancel is not None
                        and cancel.is_set()
                    ):
                        request.abort()
                finally:
                    if registration is not None:
                        registration.unregister()

            if cancel is not None and cancel.is_set():
                return -1
            if attempt.error is not None:
                self.append_log({"error": str(attempt.error)})
                if isinstance(attempt.error, api_error_type):
                    status_code = getattr(attempt.error, "status_code", None)
                    return status_code if isinstance(status_code, int) else 1
                return 1
            return 0

        def classify_request(_exit_code: int) -> str | None:
            if state is not None and isinstance(
                state.error, api_error_type
            ) and getattr(
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
        if retry.exit_code != 0 or state is None or state.response is None:
            error = state.error if state is not None else None
            return self.result(AgentStatus.FAILED, error=str(error))
        return state.response


def _tool_activity(
    name: str,
    arguments: Mapping[str, Any],
) -> AgentActivity | None:
    if name == "read_file":
        path = arguments.get("path")
        if isinstance(path, str) and path:
            return AgentActivity(AgentActivityKind.READING, path)
        return None
    if name == "search_text":
        query = arguments.get("query")
        if isinstance(query, str) and query:
            return AgentActivity(AgentActivityKind.SEARCHING, query)
        return None
    if name == "list_files":
        path = arguments.get("path", ".")
        if isinstance(path, str):
            return AgentActivity(AgentActivityKind.SEARCHING, path or ".")
        return None
    if name == "run_command":
        argv = arguments.get("argv")
        if (
            isinstance(argv, list)
            and argv
            and all(isinstance(item, str) for item in argv)
        ):
            return AgentActivity(AgentActivityKind.COMMAND, shlex.join(argv))
        return None
    if name == "apply_patch":
        patch = arguments.get("patch")
        if not isinstance(patch, str):
            return None
        try:
            paths = api_patch_paths(patch)
        except (ApiToolError, TypeError, ValueError):
            return None
        return AgentActivity(AgentActivityKind.WRITING, ", ".join(paths))
    return None
