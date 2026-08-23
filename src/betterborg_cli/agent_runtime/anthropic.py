"""Anthropic Messages API adapter over the contained API tool registry."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from betterborg_cli.agent_runtime.api_tools import (
    ApiAgentRole,
    ContainedApiTools,
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
    ) -> Mapping[str, Any]: ...


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


@dataclass(slots=True)
class _RequestState:
    response: Mapping[str, Any] | None = None
    error: Exception | None = None


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
    ) -> Mapping[str, Any]:
        if cancel is not None and cancel.is_set():
            raise AnthropicApiError("agent run cancelled")
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "anthropic-version": ANTHROPIC_API_VERSION,
                "content-type": "application/json",
                "x-api-key": api_key,
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
                        raise AnthropicApiError("agent run cancelled")
            body = b"".join(chunks).decode("utf-8", errors="replace")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            error_type, message = _api_error_details(body, str(error.reason))
            raise AnthropicApiError(
                message,
                status_code=error.code,
                error_type=error_type,
            ) from error
        except urllib.error.URLError as error:
            raise AnthropicApiError(
                f"Anthropic network error: {error.reason}",
                error_type="api_error",
            ) from error
        except TimeoutError as error:
            raise AnthropicApiError(
                "Anthropic network request timed out",
                error_type="api_error",
            ) from error
        if cancel is not None and cancel.is_set():
            raise AnthropicApiError("agent run cancelled")
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as error:
            raise AnthropicApiError("Anthropic returned malformed JSON") from error
        if not isinstance(decoded, Mapping):
            raise AnthropicApiError("Anthropic returned a non-object response")
        return decoded


@dataclass(slots=True)
class AnthropicAdapter:
    """Run a schema-shaped, contained multi-turn Anthropic agent."""

    role: ApiAgentRole | str
    api_key: str | None = None
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
        start = time.monotonic()
        attempts = 0
        usage: list[AgentUsage] = []
        model = spec.model
        key = self.api_key or spec.env.get("ANTHROPIC_API_KEY") or os.environ.get(
            "ANTHROPIC_API_KEY"
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
                exit_code=0 if status == AgentStatus.COMPLETED else -1
                if status == AgentStatus.CANCELLED
                else 1,
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
                error="Anthropic API adapter requires API billing mode",
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
                error="Anthropic API credential is not configured",
            )

        try:
            tools = ContainedApiTools(
                spec.cwd,
                cast(ApiAgentRole, self.role),
                workspace_trusted=self.workspace_trusted,
            )
        except (OSError, ValueError) as error:
            return result(AgentStatus.FAILED, error=str(error))
        allowed = _allowed_tool_names(tools, spec.allowed_tools)
        tool_definitions = [
            _tool_definition(name) for name in sorted(allowed)
        ] + [_submit_definition(spec.schema)]
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": spec.user_prompt}
        ]

        for _turn in range(self.max_turns):
            if cancel is not None and cancel.is_set():
                return result(
                    AgentStatus.CANCELLED,
                    error="agent run cancelled",
                    retryable=True,
                )
            request_payload = {
                "model": spec.model,
                "max_tokens": self.max_tokens,
                "system": spec.system_prompt,
                "messages": messages,
                "tools": tool_definitions,
            }
            request_state = _RequestState()

            def request_once(
                payload: Mapping[str, Any] = request_payload,
                state: _RequestState = request_state,
            ) -> int:
                state.response = None
                state.error = None
                if cancel is None:
                    try:
                        state.response = self.transport.create_message(
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
                            state.response = self.transport.create_message(
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
                        name="betterborg-anthropic-request",
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
                    if isinstance(state.error, AnthropicApiError):
                        return state.error.status_code or 1
                    return 1
                return 0

            def classify_request(
                _exit_code: int,
                state: _RequestState = request_state,
            ) -> str | None:
                if isinstance(state.error, AnthropicApiError):
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
                content = _response_content(response)
                response_usage = _response_usage(response)
            except AnthropicApiError as error:
                return result(AgentStatus.FAILED, error=str(error))
            usage.append(response_usage)
            response_model = response.get("model")
            if isinstance(response_model, str) and response_model:
                model = response_model

            tool_uses = [block for block in content if block.get("type") == "tool_use"]
            stop_reason = response.get("stop_reason")
            if tool_uses and stop_reason != "tool_use":
                return result(
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
                    return result(
                        AgentStatus.FAILED,
                        error="submit_result must be the only tool call in its turn",
                    )
                candidate = submissions[0].get("input")
                if not isinstance(candidate, Mapping):
                    return result(
                        AgentStatus.FAILED,
                        error="submit_result input must be an object",
                    )
                payload = cast(
                    dict[str, Any],
                    _redact_value(dict(candidate), key),
                )
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

            if not tool_uses:
                return result(
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
                    return result(
                        AgentStatus.CANCELLED,
                        error="agent run cancelled",
                        retryable=True,
                    )
                try:
                    tool_result = _execute_tool_use(
                        block,
                        tools,
                        allowed,
                        api_key=key,
                        cancel=cancel,
                    )
                except AnthropicApiError as error:
                    return result(AgentStatus.FAILED, error=str(error))
                if cancel is not None and cancel.is_set():
                    return result(
                        AgentStatus.CANCELLED,
                        error="agent run cancelled",
                        retryable=True,
                    )
                tool_results.append(tool_result)
            messages.append({"role": "user", "content": tool_results})

        return result(
            AgentStatus.FAILED,
            error=f"Anthropic exceeded the {self.max_turns}-turn limit",
        )


def _allowed_tool_names(
    tools: ContainedApiTools, requested: Sequence[str]
) -> frozenset[str]:
    if not requested:
        return tools.available_tools
    return tools.available_tools.intersection(requested)


def _tool_definition(name: str) -> dict[str, Any]:
    definitions: dict[str, tuple[str, dict[str, Any]]] = {
        "list_files": (
            "List files recursively beneath a relative path in the run directory.",
            _object_schema({"path": {"type": "string"}}),
        ),
        "search_text": (
            "Search UTF-8 files for literal text beneath a relative path.",
            _object_schema(
                {"query": {"type": "string"}, "path": {"type": "string"}},
                required=("query",),
            ),
        ),
        "read_file": (
            "Read one UTF-8 file at a relative path in the run directory.",
            _object_schema(
                {"path": {"type": "string"}}, required=("path",)
            ),
        ),
        "apply_patch": (
            "Apply a BetterBorg patch to files inside the run directory.",
            _object_schema(
                {"patch": {"type": "string"}}, required=("patch",)
            ),
        ),
        "run_command": (
            "Run a shell-free argv on the host from the trusted run directory.",
            _object_schema(
                {
                    "argv": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    }
                },
                required=("argv",),
            ),
        ),
    }
    description, schema = definitions[name]
    return {"name": name, "description": description, "input_schema": schema}


def _submit_definition(schema: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": _SUBMIT_TOOL,
        "description": "Submit the final result after all work is complete.",
        "input_schema": dict(schema),
    }


def _object_schema(
    properties: Mapping[str, Any], *, required: Sequence[str] = ()
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": dict(properties),
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    return schema


def _execute_tool_use(
    block: Mapping[str, Any],
    tools: ContainedApiTools,
    allowed: frozenset[str],
    *,
    api_key: str | None,
    cancel: CancellationToken | None = None,
) -> dict[str, Any]:
    tool_id = block.get("id")
    name = block.get("name")
    arguments = block.get("input")
    if not isinstance(tool_id, str) or not tool_id:
        raise AnthropicApiError("Anthropic tool call is missing an id")
    result: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_id}

    def redacted_result() -> dict[str, Any]:
        return cast(dict[str, Any], _redact_value(result, api_key))

    if not isinstance(name, str) or name not in allowed:
        result.update(content=f"tool is not allowed: {name}", is_error=True)
        return redacted_result()
    if not isinstance(arguments, Mapping):
        result.update(content="tool input must be an object", is_error=True)
        return redacted_result()
    try:
        if name == "list_files":
            value: Any = {"files": list(tools.list_files(**arguments))}
        elif name == "search_text":
            value = {
                "matches": [asdict(match) for match in tools.search_text(**arguments)]
            }
        elif name == "read_file":
            value = {"content": tools.read_file(**arguments)}
        elif name == "apply_patch":
            value = {"changed_files": list(tools.apply_patch(**arguments))}
        else:
            value = asdict(tools.run_command(**arguments, cancel=cancel))
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
