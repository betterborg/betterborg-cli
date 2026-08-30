"""Shared provider API adapter contract harness."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from betterborg_cli.agent_runtime import (
    AgentAdapter,
    AgentRunSpec,
    AnthropicAdapter,
    AnthropicApiError,
    ApiAgentRole,
    CancellationToken,
    OpenAIAdapter,
    OpenAIApiError,
)

Response = Mapping[str, Any]
QueuedResponse = Response | Exception | Callable[[CancellationToken | None], Response]
Provider = Literal["anthropic", "openai"]
RunProvider = Literal["anthropic", "openai", "claude", "codex"]


def write_native_output(
    log_path: Path,
    text: str,
    on_line: Callable[[str], None] | None,
) -> None:
    """Persist native output and present the same text to its line observer."""
    log_path.write_text(text, encoding="utf-8")
    if on_line is not None:
        for line in text.splitlines(keepends=True):
            on_line(line)


@dataclass
class FakeApiTransport:
    """Queue responses while recording JSON-safe copies of provider requests."""

    responses: list[QueuedResponse]
    payloads: list[dict[str, Any]] = field(default_factory=list)
    api_keys: list[str] = field(default_factory=list, repr=False)

    def create_message(
        self,
        payload: Mapping[str, Any],
        *,
        api_key: str,
        cancel: CancellationToken | None = None,
    ) -> Mapping[str, Any]:
        return self._dequeue(payload, api_key=api_key, cancel=cancel)

    def create_response(
        self,
        payload: Mapping[str, Any],
        *,
        api_key: str,
        cancel: CancellationToken | None = None,
    ) -> Mapping[str, Any]:
        return self._dequeue(payload, api_key=api_key, cancel=cancel)

    def _dequeue(
        self,
        payload: Mapping[str, Any],
        *,
        api_key: str,
        cancel: CancellationToken | None,
    ) -> Mapping[str, Any]:
        self.payloads.append(json.loads(json.dumps(payload)))
        self.api_keys.append(api_key)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(cancel)
        return response


class ChunkedHttpResponse:
    """Small urllib response double that can fail between body chunks."""

    def __init__(self, chunks: list[bytes | Exception]) -> None:
        self.chunks = iter(chunks)

    def __enter__(self) -> ChunkedHttpResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _size: int = -1) -> bytes:
        chunk = next(self.chunks)
        if isinstance(chunk, Exception):
            raise chunk
        return chunk

    def close(self) -> None:
        return None


def make_run_spec(
    tmp_path: Path,
    provider: RunProvider,
    **changes: Any,
) -> AgentRunSpec:
    """Build the shared structured-result run contract for one provider."""
    model = "gpt-test" if provider in {"openai", "codex"} else "claude-test"
    values: dict[str, Any] = {
        "system_prompt": "Inspect the repository and complete the task.",
        "user_prompt": "Read the version, then submit the result.",
        "schema": {
            "type": "object",
            "required": ["status", "version"],
            "properties": {
                "status": {"const": "completed"},
                "version": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "cwd": tmp_path,
        "model": model,
        "log_path": tmp_path / f"{provider}.jsonl",
        "result_path": tmp_path / "result.json",
    }
    if provider in {"claude", "codex"}:
        values["billing_mode"] = "subscription"
    values.update(changes)
    return AgentRunSpec(**values)


def anthropic_spec(tmp_path: Path, **changes: Any) -> AgentRunSpec:
    return make_run_spec(tmp_path, "anthropic", **changes)


def openai_spec(tmp_path: Path, **changes: Any) -> AgentRunSpec:
    return make_run_spec(tmp_path, "openai", **changes)


def claude_spec(tmp_path: Path, **changes: Any) -> AgentRunSpec:
    return make_run_spec(tmp_path, "claude", **changes)


def codex_spec(tmp_path: Path, **changes: Any) -> AgentRunSpec:
    return make_run_spec(tmp_path, "codex", **changes)


def anthropic_message(
    content: list[dict[str, Any]],
    *,
    model: str = "claude-test-20260801",
    input_tokens: int = 10,
    output_tokens: int = 4,
    cache_read: int = 0,
    cache_write: int = 0,
    stop_reason: str = "tool_use",
) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_write,
        },
    }


def openai_function_call(
    name: str,
    arguments: Mapping[str, Any],
    *,
    call_id: str,
) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
        "status": "completed",
    }


def openai_response(
    output: list[dict[str, Any]],
    *,
    response_id: str = "resp_test",
    model: str = "gpt-test-2026-08-01",
    input_tokens: int = 10,
    output_tokens: int = 4,
    cache_read: int = 0,
    cache_write: int = 0,
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "status": status,
        "model": model,
        "output": output,
        "usage": {
            # Responses reports an inclusive input total; the harness arguments
            # describe the provider-neutral, mutually exclusive buckets.
            "input_tokens": input_tokens + cache_read + cache_write,
            "output_tokens": output_tokens,
            "input_tokens_details": {
                "cached_tokens": cache_read,
                "cache_write_tokens": cache_write,
            },
        },
    }


@dataclass(frozen=True, slots=True)
class ApiAdapterHarness:
    """Translate provider-neutral contract cases to each provider wire shape."""

    provider: Provider

    @property
    def resolved_model(self) -> str:
        if self.provider == "anthropic":
            return "claude-test-20260801"
        return "gpt-test-2026-08-01"

    @property
    def credential(self) -> str:
        if self.provider == "anthropic":
            return "sk-ant-api03-contract-secret"
        return "sk-proj-contract-secret"

    def spec(self, tmp_path: Path, **changes: Any) -> AgentRunSpec:
        return make_run_spec(tmp_path, self.provider, **changes)

    def tool_call(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        call_id: str,
    ) -> dict[str, Any]:
        if self.provider == "anthropic":
            return {
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": dict(arguments),
            }
        return openai_function_call(name, arguments, call_id=call_id)

    def response(
        self,
        calls: list[dict[str, Any]],
        *,
        response_id: str = "response",
        input_tokens: int = 10,
        output_tokens: int = 4,
        cache_read: int = 0,
        cache_write: int = 0,
    ) -> dict[str, Any]:
        if self.provider == "anthropic":
            return anthropic_message(
                calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read=cache_read,
                cache_write=cache_write,
            )
        return openai_response(
            calls,
            response_id=response_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read=cache_read,
            cache_write=cache_write,
        )

    def adapter(
        self,
        role: ApiAgentRole,
        *,
        transport: FakeApiTransport,
        api_key: str | None = "key",
        workspace_trusted: bool = False,
        **options: Any,
    ) -> AgentAdapter:
        adapter_type = (
            AnthropicAdapter if self.provider == "anthropic" else OpenAIAdapter
        )
        return adapter_type(
            role,
            api_key=api_key,
            workspace_trusted=workspace_trusted,
            transport=transport,
            **options,
        )

    def transient_error(self, message: str = "temporarily overloaded") -> Exception:
        if self.provider == "anthropic":
            return AnthropicApiError(
                message,
                status_code=529,
                error_type="overloaded_error",
            )
        return OpenAIApiError(
            message,
            status_code=503,
            error_type="server_error",
        )

    def api_error(self, message: str) -> Exception:
        if self.provider == "anthropic":
            return AnthropicApiError(message, status_code=401)
        return OpenAIApiError(message, status_code=401)

    def extract_tool_output(self, payload: Mapping[str, Any]) -> str:
        if self.provider == "anthropic":
            return payload["messages"][-1]["content"][0]["content"]
        return payload["input"][0]["output"]


API_ADAPTER_HARNESSES = (
    ApiAdapterHarness("anthropic"),
    ApiAdapterHarness("openai"),
)
