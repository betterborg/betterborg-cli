"""OpenAI Responses API adapter over the contained API tool registry."""

from __future__ import annotations

import http.client
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from betterborg_cli.agent_runtime.api_tools import (
    ApiAgentRole,
    ContainedApiTools,
    api_tool_definition,
    select_api_tool_names,
)
from betterborg_cli.agent_runtime.base import (
    AgentCapabilities,
    AgentResult,
    AgentRunSpec,
    AgentStatus,
    AgentUsage,
    BillingMode,
    CancellationToken,
    combine_agent_usage,
)
from betterborg_cli.agent_runtime.retry import (
    DEFAULT_TRANSIENT_BACKOFF_SECONDS,
    DEFAULT_TRANSIENT_MAX_ATTEMPTS,
    run_with_transient_retry,
)
from betterborg_cli.agent_runtime.structured import (
    StructuredResultError,
    validate_structured_result,
)

OPENAI_API_URL = "https://api.openai.com/v1/responses"
_PROVIDER = "openai"
_SUBMIT_TOOL = "submit_result"
_HTTP_TIMEOUT_SECONDS = 60.0
_TRANSIENT_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
_TRANSIENT_ERROR_TYPES = frozenset(
    {"api_error", "rate_limit_error", "server_error"}
)
_CREDENTIAL_PATTERN = re.compile(
    r"(?i)(?:sk-(?:proj-|svcacct-)?[a-z0-9_-]+|"
    r"(?:openai_api_key|authorization)\s*[:=]\s*['\"]?"
    r"(?:bearer\s+)?[^\s,'\"}]+)"
)


class OpenAITransport(Protocol):
    """Transport boundary used by the adapter and hermetic test doubles."""

    def create_response(
        self,
        payload: Mapping[str, Any],
        *,
        api_key: str,
        cancel: CancellationToken | None = None,
    ) -> Mapping[str, Any]: ...


class OpenAIApiError(RuntimeError):
    """A failed Responses API request with retry classification metadata."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type

    @property
    def transient(self) -> bool:
        """Return whether repeating this exact request can safely recover."""
        return (
            self.status_code in _TRANSIENT_STATUS_CODES
            or self.error_type in _TRANSIENT_ERROR_TYPES
        )


@dataclass(slots=True)
class _RequestState:
    response: Mapping[str, Any] | None = None
    error: Exception | None = None


@dataclass(frozen=True, slots=True)
class UrllibOpenAITransport:
    """Small standard-library transport for OpenAI's Responses endpoint."""

    url: str = OPENAI_API_URL

    def create_response(
        self,
        payload: Mapping[str, Any],
        *,
        api_key: str,
        cancel: CancellationToken | None = None,
    ) -> Mapping[str, Any]:
        if cancel is not None and cancel.is_set():
            raise OpenAIApiError("agent run cancelled")
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            chunks: list[bytes] = []
            with urllib.request.urlopen(
                request,
                timeout=_HTTP_TIMEOUT_SECONDS,
            ) as response:
                while chunk := response.read(64 * 1024):
                    chunks.append(chunk)
                    if cancel is not None and cancel.is_set():
                        raise OpenAIApiError("agent run cancelled")
            body = b"".join(chunks).decode("utf-8", errors="replace")
        except urllib.error.HTTPError as error:
            try:
                body = error.read().decode("utf-8", errors="replace")
            except (OSError, http.client.HTTPException) as body_error:
                raise OpenAIApiError(
                    f"OpenAI HTTP {error.code} response body failed: {body_error}",
                    status_code=error.code,
                ) from body_error
            error_type, message = _api_error_details(body, str(error.reason))
            raise OpenAIApiError(
                message,
                status_code=error.code,
                error_type=error_type,
            ) from error
        except urllib.error.URLError as error:
            raise OpenAIApiError(
                f"OpenAI network error: {error.reason}",
                error_type="api_error",
            ) from error
        except TimeoutError as error:
            raise OpenAIApiError(
                "OpenAI network request timed out",
                error_type="api_error",
            ) from error
        except (OSError, http.client.HTTPException) as error:
            raise OpenAIApiError(
                f"OpenAI network response failed: {error}",
                error_type="api_error",
            ) from error
        if cancel is not None and cancel.is_set():
            raise OpenAIApiError("agent run cancelled")
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as error:
            raise OpenAIApiError("OpenAI returned malformed JSON") from error
        if not isinstance(decoded, Mapping):
            raise OpenAIApiError("OpenAI returned a non-object response")
        return decoded


@dataclass(slots=True)
class OpenAIAdapter:
    """Run a schema-shaped, contained multi-turn OpenAI agent."""

    role: ApiAgentRole | str
    api_key: str | None = None
    workspace_trusted: bool = False
    transport: OpenAITransport = field(default_factory=UrllibOpenAITransport)
    max_output_tokens: int = 8192
    max_turns: int = 64
    transient_backoff_seconds: float = DEFAULT_TRANSIENT_BACKOFF_SECONDS
    transient_max_attempts: int = DEFAULT_TRANSIENT_MAX_ATTEMPTS
    name: str = field(default=_PROVIDER, init=False)
    capabilities: AgentCapabilities = field(
        default_factory=lambda: AgentCapabilities(
            billing_modes=frozenset({BillingMode.API}),
            structured_output=True,
            streaming=False,
            tool_allowlist=True,
            resumable=True,
        ),
        init=False,
    )

    def __post_init__(self) -> None:
        self.role = ApiAgentRole(self.role)
        if self.max_output_tokens < 1:
            raise ValueError("max output tokens must be at least one")
        if self.max_turns < 1:
            raise ValueError("max turns must be at least one")
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
        attempts = 0
        usage: list[AgentUsage] = []
        model = spec.model
        key = self.api_key or spec.env.get("OPENAI_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )
        _prepare_log(spec.log_path)

        def result(
            status: AgentStatus,
            *,
            error: str | None = None,
            payload: dict[str, Any] | None = None,
            result_path: Path | None = None,
            retryable: bool = False,
        ) -> AgentResult:
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
                error=_redact(error or "", key) or None,
                log_path=spec.log_path,
                result_path=result_path,
                duration_seconds=time.monotonic() - start,
                usage=combine_agent_usage(usage),
                billing_mode=BillingMode.API,
                provider=_PROVIDER,
                model=model,
                attempts=attempts,
                retryable=retryable,
                resume_token=spec.resume_token,
            )

        if spec.billing_mode != BillingMode.API:
            return result(
                AgentStatus.FAILED,
                error="OpenAI API adapter requires API billing mode",
            )
        if cancel is not None and cancel.is_set():
            return result(
                AgentStatus.CANCELLED,
                error="agent run cancelled",
                retryable=True,
            )
        if not key:
            return result(
                AgentStatus.FAILED,
                error="OpenAI API credential is not configured",
            )

        try:
            tools = ContainedApiTools(
                spec.cwd,
                cast(ApiAgentRole, self.role),
                workspace_trusted=self.workspace_trusted,
            )
        except (OSError, ValueError) as error:
            return result(AgentStatus.FAILED, error=str(error))
        allowed = select_api_tool_names(tools, spec.allowed_tools)
        tool_definitions = [
            _tool_definition(name) for name in sorted(allowed)
        ] + [_submit_definition(spec.schema)]
        input_items: str | list[dict[str, Any]] = spec.user_prompt
        previous_response_id: str | None = None

        for _turn in range(self.max_turns):
            if cancel is not None and cancel.is_set():
                return result(
                    AgentStatus.CANCELLED,
                    error="agent run cancelled",
                    retryable=True,
                )
            request_payload: dict[str, Any] = {
                "model": spec.model,
                "instructions": spec.system_prompt,
                "input": input_items,
                "tools": tool_definitions,
                "parallel_tool_calls": False,
                "max_output_tokens": self.max_output_tokens,
            }
            if previous_response_id is not None:
                request_payload["previous_response_id"] = previous_response_id
            if spec.effort is not None:
                request_payload["reasoning"] = {"effort": spec.effort}
            request_state = _RequestState()

            def request_once(
                payload: Mapping[str, Any] = request_payload,
                state: _RequestState = request_state,
            ) -> int:
                state.response = None
                state.error = None
                if cancel is None:
                    try:
                        state.response = self.transport.create_response(
                            payload,
                            api_key=key,
                            cancel=None,
                        )
                    except Exception as error:
                        state.error = error
                else:
                    if cancel.is_set():
                        return -1
                    completed = threading.Event()

                    def send_request() -> None:
                        try:
                            state.response = self.transport.create_response(
                                payload,
                                api_key=key,
                                cancel=cancel,
                            )
                        except Exception as error:
                            state.error = error
                        finally:
                            completed.set()

                    threading.Thread(
                        target=send_request,
                        name="betterborg-openai-request",
                        daemon=True,
                    ).start()
                    while not completed.wait(0.05):
                        if cancel.is_set():
                            return -1
                    if cancel.is_set():
                        return -1
                if state.error is not None:
                    _append_log(
                        spec.log_path,
                        {"error": _redact(str(state.error), key)},
                    )
                    if isinstance(state.error, OpenAIApiError):
                        return state.error.status_code or 1
                    return 1
                return 0

            def classify_request(
                _exit_code: int,
                state: _RequestState = request_state,
            ) -> str | None:
                if isinstance(state.error, OpenAIApiError):
                    return str(state.error) if state.error.transient else None
                return None

            retry = run_with_transient_retry(
                request_once,
                classify_request,
                cancel=cancel,
                backoff_seconds=self.transient_backoff_seconds,
                max_attempts=self.transient_max_attempts,
            )
            attempts += retry.attempts
            if (cancel is not None and cancel.is_set()) or retry.cancelled:
                return result(
                    AgentStatus.CANCELLED,
                    error="agent run cancelled",
                    retryable=True,
                )
            if retry.exhausted:
                return result(
                    AgentStatus.CANCELLED,
                    error=f"transient retry exhausted: {retry.transient_reason}",
                    retryable=True,
                )
            if retry.exit_code != 0 or request_state.response is None:
                return result(AgentStatus.FAILED, error=str(request_state.error))
            response = request_state.response

            _append_log(spec.log_path, _redact_value(response, key))
            try:
                output = _response_output(response)
                response_usage = _response_usage(response)
            except OpenAIApiError as error:
                return result(AgentStatus.FAILED, error=str(error))
            usage.append(response_usage)
            response_model = response.get("model")
            if isinstance(response_model, str) and response_model:
                model = response_model
            response_id = response.get("id")
            if not isinstance(response_id, str) or not response_id:
                return result(
                    AgentStatus.FAILED,
                    error="OpenAI response is missing an id",
                )
            response_status = response.get("status")
            if response_status != "completed":
                return result(
                    AgentStatus.FAILED,
                    error=_incomplete_response_error(response, response_status),
                )

            calls = [item for item in output if item.get("type") == "function_call"]
            submissions = [
                item for item in calls if item.get("name") == _SUBMIT_TOOL
            ]
            if submissions:
                if len(calls) != 1 or len(submissions) != 1:
                    return result(
                        AgentStatus.FAILED,
                        error="submit_result must be the only tool call in its turn",
                    )
                try:
                    payload = _tool_arguments(submissions[0])
                except OpenAIApiError as error:
                    return result(AgentStatus.FAILED, error=str(error))
                payload = cast(dict[str, Any], _redact_value(payload, key))
                try:
                    validate_structured_result(payload, spec.schema)
                except StructuredResultError as error:
                    return result(
                        AgentStatus.FAILED,
                        error=f"structured result validation failed: {error}",
                    )
                spec.result_path.parent.mkdir(parents=True, exist_ok=True)
                spec.result_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                return result(
                    AgentStatus.COMPLETED,
                    payload=payload,
                    result_path=spec.result_path,
                )

            if not calls:
                return result(
                    AgentStatus.FAILED,
                    error="OpenAI ended without submit_result",
                )
            function_outputs: list[dict[str, Any]] = []
            for call in calls:
                if cancel is not None and cancel.is_set():
                    return result(
                        AgentStatus.CANCELLED,
                        error="agent run cancelled",
                        retryable=True,
                    )
                try:
                    function_output = _execute_tool_call(
                        call,
                        tools,
                        allowed,
                        api_key=key,
                        cancel=cancel,
                    )
                except OpenAIApiError as error:
                    return result(AgentStatus.FAILED, error=str(error))
                if cancel is not None and cancel.is_set():
                    return result(
                        AgentStatus.CANCELLED,
                        error="agent run cancelled",
                        retryable=True,
                    )
                function_outputs.append(function_output)
            input_items = function_outputs
            previous_response_id = response_id

        return result(
            AgentStatus.FAILED,
            error=f"OpenAI exceeded the {self.max_turns}-turn limit",
        )


def _tool_definition(name: str) -> dict[str, Any]:
    definition = api_tool_definition(name)
    return {
        "type": "function",
        "name": definition.name,
        "description": definition.description,
        "parameters": definition.parameters,
        "strict": False,
    }


def _submit_definition(schema: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": _SUBMIT_TOOL,
        "description": "Submit the final result after all work is complete.",
        "parameters": dict(schema),
        "strict": False,
    }


def _tool_arguments(call: Mapping[str, Any]) -> dict[str, Any]:
    arguments = call.get("arguments")
    if not isinstance(arguments, str):
        raise OpenAIApiError("OpenAI tool call arguments must be JSON text")
    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError as error:
        raise OpenAIApiError("OpenAI tool call arguments are malformed JSON") from error
    if not isinstance(decoded, dict):
        raise OpenAIApiError("OpenAI tool call arguments must be an object")
    return decoded


def _execute_tool_call(
    call: Mapping[str, Any],
    tools: ContainedApiTools,
    allowed: frozenset[str],
    *,
    api_key: str | None,
    cancel: CancellationToken | None = None,
) -> dict[str, Any]:
    call_id = call.get("call_id")
    name = call.get("name")
    if not isinstance(call_id, str) or not call_id:
        raise OpenAIApiError("OpenAI tool call is missing a call_id")
    result: dict[str, Any] = {
        "type": "function_call_output",
        "call_id": call_id,
    }
    if not isinstance(name, str) or name not in allowed:
        value: Any = {"error": f"tool is not allowed: {name}"}
    else:
        try:
            arguments = _tool_arguments(call)
            value = tools.execute(name, arguments, cancel=cancel)
        except Exception as error:
            value = {"error": str(error)}
    result["output"] = json.dumps(
        _redact_value(value, api_key),
        sort_keys=True,
    )
    return result


def _response_output(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = response.get("output")
    if not isinstance(output, list) or not all(
        isinstance(item, Mapping) for item in output
    ):
        raise OpenAIApiError("OpenAI response has malformed output")
    return [dict(item) for item in output]


def _response_usage(response: Mapping[str, Any]) -> AgentUsage:
    raw = response.get("usage")
    if not isinstance(raw, Mapping):
        raise OpenAIApiError("OpenAI response has malformed usage")

    def token(container: Mapping[str, Any], name: str) -> int | None:
        value = container.get(name)
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise OpenAIApiError(f"OpenAI usage {name} is invalid")
        return value

    details = raw.get("input_tokens_details", {})
    if not isinstance(details, Mapping):
        raise OpenAIApiError("OpenAI input token details are malformed")
    return AgentUsage(
        tokens_input=token(raw, "input_tokens"),
        tokens_output=token(raw, "output_tokens"),
        tokens_cache_read=token(details, "cached_tokens"),
        tokens_cache_write=token(details, "cache_write_tokens"),
        num_turns=1,
    )


def _incomplete_response_error(
    response: Mapping[str, Any], status: Any
) -> str:
    detail = response.get("incomplete_details") or response.get("error")
    if isinstance(detail, Mapping):
        reason = detail.get("reason") or detail.get("message") or detail.get("code")
        if isinstance(reason, str) and reason:
            return f"OpenAI response is {status or 'not completed'}: {reason}"
    return f"OpenAI response is {status or 'not completed'}"


def _api_error_details(body: str, fallback: str) -> tuple[str | None, str]:
    try:
        decoded = json.loads(body)
        error = decoded.get("error", {})
    except (json.JSONDecodeError, AttributeError):
        return None, fallback
    if not isinstance(error, Mapping):
        return None, fallback
    error_type = error.get("type") or error.get("code")
    message = error.get("message")
    return (
        error_type if isinstance(error_type, str) else None,
        message if isinstance(message, str) else fallback,
    )


def _prepare_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _append_log(path: Path, event: Any) -> None:
    with path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(event, sort_keys=True) + "\n")


def _redact(text: str, api_key: str | None) -> str:
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    return _CREDENTIAL_PATTERN.sub("[REDACTED]", text)


def _redact_value(value: Any, api_key: str | None) -> Any:
    if isinstance(value, str):
        return _redact(value, api_key)
    if isinstance(value, Mapping):
        return {
            _redact(str(key), api_key): _redact_value(item, api_key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item, api_key) for item in value]
    return value
