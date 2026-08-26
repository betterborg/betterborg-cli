"""OpenAI Responses API adapter over the contained API tool registry."""

from __future__ import annotations

import http.client
import json
import re
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from betterborg_cli.agent_runtime.api_adapter import (
    ApiCredentialRedactor,
    ApiRunContext,
)
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
)
from betterborg_cli.agent_runtime.retry import (
    DEFAULT_TRANSIENT_BACKOFF_SECONDS,
    DEFAULT_TRANSIENT_MAX_ATTEMPTS,
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
    api_key: str | None = field(default=None, repr=False)
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
        key = self.api_key
        runtime = ApiRunContext(
            spec,
            _PROVIDER,
            ApiCredentialRedactor(key, _CREDENTIAL_PATTERN),
        )

        if spec.billing_mode != BillingMode.API:
            return runtime.result(
                AgentStatus.FAILED,
                error="OpenAI API adapter requires API billing mode",
            )
        if cancel is not None and cancel.is_set():
            return runtime.cancelled()
        if not key:
            return runtime.result(
                AgentStatus.FAILED,
                error="OpenAI API credential is not configured",
            )

        try:
            tools = ContainedApiTools(
                spec.cwd,
                cast(ApiAgentRole, self.role),
                workspace_trusted=self.workspace_trusted,
                env=spec.env,
            )
        except (OSError, ValueError) as error:
            return runtime.result(AgentStatus.FAILED, error=str(error))
        allowed = select_api_tool_names(tools, spec.allowed_tools)
        tool_definitions = [
            _tool_definition(name) for name in sorted(allowed)
        ] + [_submit_definition(spec.schema)]
        input_items: str | list[dict[str, Any]] = spec.user_prompt
        previous_response_id: str | None = None

        for _turn in range(self.max_turns):
            if cancel is not None and cancel.is_set():
                return runtime.cancelled()
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
            response = runtime.request(
                lambda request_cancel, payload=request_payload: (
                    _create_response(
                        self.transport,
                        payload,
                        api_key=key,
                        cancel=request_cancel,
                    )
                ),
                api_error_type=OpenAIApiError,
                cancel=cancel,
                thread_name="betterborg-openai-request",
                backoff_seconds=self.transient_backoff_seconds,
                max_attempts=self.transient_max_attempts,
            )
            if isinstance(response, AgentResult):
                return response

            runtime.append_log(response)
            response_status = response.get("status")
            if response_status == "failed":
                return runtime.result(
                    AgentStatus.FAILED,
                    error=_incomplete_response_error(response, response_status),
                )
            try:
                output = _response_output(response)
                response_usage = _response_usage(response)
            except OpenAIApiError as error:
                return runtime.result(AgentStatus.FAILED, error=str(error))
            runtime.record_response(response, response_usage)
            response_id = response.get("id")
            if not isinstance(response_id, str) or not response_id:
                return runtime.result(
                    AgentStatus.FAILED,
                    error="OpenAI response is missing an id",
                )
            if response_status != "completed":
                return runtime.result(
                    AgentStatus.FAILED,
                    error=_incomplete_response_error(response, response_status),
                )

            calls = [item for item in output if item.get("type") == "function_call"]
            submissions = [
                item for item in calls if item.get("name") == _SUBMIT_TOOL
            ]
            if submissions:
                if len(calls) != 1 or len(submissions) != 1:
                    return runtime.result(
                        AgentStatus.FAILED,
                        error="submit_result must be the only tool call in its turn",
                    )
                try:
                    payload = _tool_arguments(submissions[0])
                except OpenAIApiError as error:
                    return runtime.result(AgentStatus.FAILED, error=str(error))
                return runtime.complete(payload)

            if not calls:
                return runtime.result(
                    AgentStatus.FAILED,
                    error="OpenAI ended without submit_result",
                )
            function_outputs: list[dict[str, Any]] = []
            for call in calls:
                if cancel is not None and cancel.is_set():
                    return runtime.cancelled()
                try:
                    function_output = _execute_tool_call(
                        call,
                        tools,
                        allowed,
                        redactor=runtime.redactor,
                        cancel=cancel,
                    )
                except OpenAIApiError as error:
                    return runtime.result(AgentStatus.FAILED, error=str(error))
                if cancel is not None and cancel.is_set():
                    return runtime.cancelled()
                function_outputs.append(function_output)
            input_items = function_outputs
            previous_response_id = response_id

        return runtime.result(
            AgentStatus.FAILED,
            error=f"OpenAI exceeded the {self.max_turns}-turn limit",
        )


def _create_response(
    transport: OpenAITransport,
    payload: Mapping[str, Any],
    *,
    api_key: str,
    cancel: CancellationToken | None,
) -> Mapping[str, Any]:
    response = transport.create_response(
        payload,
        api_key=api_key,
        cancel=cancel,
    )
    if response.get("status") != "failed":
        return response
    detail = response.get("error")
    if not isinstance(detail, Mapping):
        return response
    error_type = detail.get("type") or detail.get("code")
    error = OpenAIApiError(
        _incomplete_response_error(response, "failed"),
        error_type=error_type if isinstance(error_type, str) else None,
    )
    if error.transient:
        raise error
    return response


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
    redactor: ApiCredentialRedactor,
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
        redactor.value(value),
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
    total_input = token(raw, "input_tokens")
    cache_read = token(details, "cached_tokens")
    cache_write = token(details, "cache_write_tokens")
    cached_input = (cache_read or 0) + (cache_write or 0)
    if total_input is not None and cached_input > total_input:
        raise OpenAIApiError("OpenAI cached input exceeds total input")
    return AgentUsage(
        tokens_input=(
            total_input - cached_input if total_input is not None else None
        ),
        tokens_output=token(raw, "output_tokens"),
        tokens_cache_read=cache_read,
        tokens_cache_write=cache_write,
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
