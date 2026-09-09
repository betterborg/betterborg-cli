"""Anthropic Messages API adapter over the contained API tool registry."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from betterborg_cli.agent_runtime.api_adapter import (
    AbortableApiRequest,
    ApiCredentialRedactor,
    ApiRunContext,
    SchemaCorrection,
)
from betterborg_cli.agent_runtime.api_http import (
    MultiprocessUrlRequest,
    UrlRequestSpec,
    UrlResponse,
    UrlTransportError,
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

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"
_PROVIDER = "anthropic"
_SUBMIT_TOOL = "submit_result"
_HTTP_TIMEOUT_SECONDS = 60.0
_TRANSIENT_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
_TRANSIENT_ERROR_TYPES = frozenset(
    {"api_error", "overloaded_error", "rate_limit_error"}
)
_CREDENTIAL_PATTERN = re.compile(
    r"(?i)(?:sk-ant-[a-z0-9_-]+|"
    r"(?:anthropic_api_key|x-api-key)\s*[:=]\s*['\"]?[^\s,'\"}]+)"
)


class AnthropicTransport(Protocol):
    """Transport boundary used by the adapter and hermetic test doubles."""

    def create_message(
        self,
        payload: Mapping[str, Any],
        *,
        api_key: str,
        cancel: CancellationToken | None = None,
    ) -> AbortableApiRequest: ...


class AnthropicApiError(RuntimeError):
    """A failed Messages API request with retry classification metadata."""

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
class UrllibAnthropicTransport:
    """Small standard-library transport for Anthropic's Messages endpoint."""

    url: str = ANTHROPIC_API_URL

    def create_message(
        self,
        payload: Mapping[str, Any],
        *,
        api_key: str,
        cancel: CancellationToken | None = None,
    ) -> AbortableApiRequest:
        return _AnthropicRequest(
            MultiprocessUrlRequest(
                UrlRequestSpec(
                    self.url,
                    "POST",
                    {
                        "anthropic-version": ANTHROPIC_API_VERSION,
                        "content-type": "application/json",
                        "x-api-key": api_key,
                    },
                    json.dumps(payload).encode("utf-8"),
                    timeout_seconds=_HTTP_TIMEOUT_SECONDS,
                ),
                cancel,
            )
        )


@dataclass(frozen=True, slots=True)
class _AnthropicRequest:
    request: MultiprocessUrlRequest

    def execute(self) -> Mapping[str, Any]:
        try:
            response = self.request.execute()
        except UrlTransportError as error:
            raise _transport_error(error) from error
        return _decode_response(response)

    def abort(self) -> None:
        self.request.abort()

    def force(self) -> None:
        self.request.force()


def _decode_response(response: UrlResponse) -> Mapping[str, Any]:
    body = response.body.decode("utf-8", errors="replace")
    if not 200 <= response.status_code < 300:
        error_type, message = _api_error_details(body, response.reason)
        raise AnthropicApiError(
            message,
            status_code=response.status_code,
            error_type=error_type,
        )
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as error:
        raise AnthropicApiError("Anthropic returned malformed JSON") from error
    if not isinstance(decoded, Mapping):
        raise AnthropicApiError("Anthropic returned a non-object response")
    return decoded


def _transport_error(error: UrlTransportError) -> AnthropicApiError:
    if error.kind == "cancelled":
        return AnthropicApiError("agent run cancelled")
    if error.status_code is not None:
        return AnthropicApiError(
            (
                f"Anthropic HTTP {error.status_code} response body failed: "
                f"{error.message}"
            ),
            status_code=error.status_code,
        )
    if error.kind == "timeout":
        return AnthropicApiError(
            "Anthropic network request timed out",
            error_type="api_error",
        )
    if error.kind == "network":
        return AnthropicApiError(
            f"Anthropic network error: {error.message}",
            error_type="api_error",
        )
    return AnthropicApiError(
        f"Anthropic network response failed: {error.message}",
        error_type="api_error",
    )


@dataclass(slots=True)
class AnthropicAdapter:
    """Run a schema-shaped, contained multi-turn Anthropic agent."""

    role: ApiAgentRole | str
    api_key: str | None = field(default=None, repr=False)
    workspace_trusted: bool = False
    transport: AnthropicTransport = field(default_factory=UrllibAnthropicTransport)
    max_tokens: int = 8192
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
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least one")
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least one")
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
                error="Anthropic API adapter requires API billing mode",
            )
        if cancel is not None and cancel.is_set():
            return runtime.cancelled()
        if not key:
            return runtime.result(
                AgentStatus.FAILED,
                error="Anthropic API credential is not configured",
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
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": spec.user_prompt}
        ]

        for _turn in range(self.max_turns):
            if cancel is not None and cancel.is_set():
                return runtime.cancelled()
            request_payload: dict[str, Any] = {
                "model": spec.model,
                "max_tokens": self.max_tokens,
                "system": spec.system_prompt,
                "messages": messages,
                "tools": tool_definitions,
                # Every turn must end in a tool call: a run concludes by
                # calling submit_result, so a prose answer is never a valid
                # turn and the model would otherwise be free to give one.
                "tool_choice": {"type": "any"},
            }
            if spec.effort is not None:
                request_payload["output_config"] = {"effort": spec.effort}
            response = runtime.request(
                lambda request_cancel, payload=request_payload: (
                    self.transport.create_message(
                        payload,
                        api_key=key,
                        cancel=request_cancel,
                    )
                ),
                api_error_type=AnthropicApiError,
                cancel=cancel,
                backoff_seconds=self.transient_backoff_seconds,
                max_attempts=self.transient_max_attempts,
            )
            if isinstance(response, AgentResult):
                return response

            runtime.append_log(response)
            try:
                content = _response_content(response)
                response_usage = _response_usage(response)
            except AnthropicApiError as error:
                return runtime.result(AgentStatus.FAILED, error=str(error))
            runtime.record_response(response, response_usage)

            tool_uses = [block for block in content if block.get("type") == "tool_use"]
            stop_reason = response.get("stop_reason")
            if tool_uses and stop_reason != "tool_use":
                return runtime.result(
                    AgentStatus.FAILED,
                    error=(
                        "Anthropic returned tool calls without a tool_use "
                        f"stop reason ({stop_reason or 'missing'})"
                    ),
                )
            submissions = [
                block for block in tool_uses if block.get("name") == _SUBMIT_TOOL
            ]
            if submissions:
                if len(tool_uses) != 1 or len(submissions) != 1:
                    return runtime.result(
                        AgentStatus.FAILED,
                        error="submit_result must be the only tool call in its turn",
                    )
                candidate = submissions[0].get("input")
                if not isinstance(candidate, Mapping):
                    return runtime.result(
                        AgentStatus.FAILED,
                        error="submit_result input must be an object",
                    )
                submitted = runtime.complete(candidate)
                if not isinstance(submitted, SchemaCorrection):
                    return submitted
                submission_id = submissions[0].get("id")
                if not isinstance(submission_id, str) or not submission_id:
                    return runtime.result(
                        AgentStatus.FAILED,
                        error="Anthropic tool call is missing an id",
                    )
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": submission_id,
                                "content": submitted.message,
                                "is_error": True,
                            }
                        ],
                    }
                )
                continue

            if not tool_uses:
                return runtime.result(
                    AgentStatus.FAILED,
                    error=(
                        "Anthropic ended without submit_result "
                        f"({stop_reason or 'unknown'})"
                    ),
                )
            messages.append({"role": "assistant", "content": content})
            tool_results: list[dict[str, Any]] = []
            for block in tool_uses:
                if cancel is not None and cancel.is_set():
                    return runtime.cancelled()
                try:
                    tool_result = _execute_tool_use(
                        block,
                        tools,
                        allowed,
                        runtime=runtime,
                        redactor=runtime.redactor,
                        cancel=cancel,
                    )
                except AnthropicApiError as error:
                    return runtime.result(AgentStatus.FAILED, error=str(error))
                if cancel is not None and cancel.is_set():
                    return runtime.cancelled()
                tool_results.append(tool_result)
            messages.append({"role": "user", "content": tool_results})

        return runtime.result(
            AgentStatus.FAILED,
            error=f"Anthropic exceeded the {self.max_turns}-turn limit",
        )


def _tool_definition(name: str) -> dict[str, Any]:
    definition = api_tool_definition(name)
    return {
        "name": definition.name,
        "description": definition.description,
        "input_schema": definition.parameters,
    }


def _submit_definition(schema: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": _SUBMIT_TOOL,
        "description": "Submit the final result after all work is complete.",
        "input_schema": dict(schema),
    }


def _execute_tool_use(
    block: Mapping[str, Any],
    tools: ContainedApiTools,
    allowed: frozenset[str],
    *,
    runtime: ApiRunContext,
    redactor: ApiCredentialRedactor,
    cancel: CancellationToken | None = None,
) -> dict[str, Any]:
    tool_id = block.get("id")
    name = block.get("name")
    arguments = block.get("input")
    if not isinstance(tool_id, str) or not tool_id:
        raise AnthropicApiError("Anthropic tool call is missing an id")
    result: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_id}

    def redacted_result() -> dict[str, Any]:
        return cast(dict[str, Any], redactor.value(result))

    if not isinstance(name, str) or name not in allowed:
        result.update(content=f"tool is not allowed: {name}", is_error=True)
        return redacted_result()
    if not isinstance(arguments, Mapping):
        result.update(content="tool input must be an object", is_error=True)
        return redacted_result()
    try:
        value = tools.execute(name, arguments, cancel=cancel)
        runtime.emit_tool_activity(name, arguments)
        result["content"] = json.dumps(value, sort_keys=True)
    except Exception as error:
        result.update(content=str(error), is_error=True)
    return redacted_result()


def _response_content(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    content = response.get("content")
    if not isinstance(content, list) or not all(
        isinstance(block, Mapping) for block in content
    ):
        raise AnthropicApiError("Anthropic response has malformed content")
    return [dict(block) for block in content]


def _response_usage(response: Mapping[str, Any]) -> AgentUsage:
    raw = response.get("usage")
    if not isinstance(raw, Mapping):
        raise AnthropicApiError("Anthropic response has malformed usage")

    def token(name: str) -> int | None:
        value = raw.get(name)
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise AnthropicApiError(f"Anthropic usage {name} is invalid")
        return value

    return AgentUsage(
        tokens_input=token("input_tokens"),
        tokens_output=token("output_tokens"),
        tokens_cache_read=token("cache_read_input_tokens"),
        tokens_cache_write=token("cache_creation_input_tokens"),
        num_turns=1,
    )


def _api_error_details(body: str, fallback: str) -> tuple[str | None, str]:
    try:
        decoded = json.loads(body)
        error = decoded.get("error", {})
    except (json.JSONDecodeError, AttributeError):
        return None, fallback
    if not isinstance(error, Mapping):
        return None, fallback
    error_type = error.get("type")
    message = error.get("message")
    return (
        error_type if isinstance(error_type, str) else None,
        message if isinstance(message, str) else fallback,
    )
