"""OpenAI Responses API adapter behavior with hermetic transports."""

from __future__ import annotations

import http.client
import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from betterborg_cli.agent_runtime import (
    AgentRunSpec,
    AgentStatus,
    AgentUsage,
    ApiAgentRole,
    BillingMode,
    CancellationToken,
    OpenAIAdapter,
    OpenAIApiError,
    UrllibOpenAITransport,
)

Response = Mapping[str, Any]
QueuedResponse = Response | Exception | Callable[[CancellationToken | None], Response]


@dataclass
class FakeTransport:
    responses: list[QueuedResponse]
    payloads: list[dict[str, Any]] = field(default_factory=list)
    api_keys: list[str] = field(default_factory=list)

    def create_response(
        self,
        payload: Mapping[str, Any],
        *,
        api_key: str,
        cancel: CancellationToken | None = None,
    ) -> Mapping[str, Any]:
        self.payloads.append(json.loads(json.dumps(payload)))
        self.api_keys.append(api_key)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(cancel)
        return response


class _ChunkedHttpResponse:
    def __init__(self, chunks: list[bytes | Exception]) -> None:
        self.chunks = iter(chunks)

    def __enter__(self) -> _ChunkedHttpResponse:
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


def _spec(tmp_path: Path, **changes: Any) -> AgentRunSpec:
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
        "model": "gpt-test",
        "log_path": tmp_path / "openai.jsonl",
        "result_path": tmp_path / "result.json",
    }
    values.update(changes)
    return AgentRunSpec(**values)


def _response(
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
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_tokens_details": {
                "cached_tokens": cache_read,
                "cache_write_tokens": cache_write,
            },
        },
    }


def _call(
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


def test_multi_turn_tools_submit_schema_and_persist_metadata(tmp_path: Path) -> None:
    (tmp_path / "VERSION").write_text("1.2.3\n", encoding="utf-8")
    transport = FakeTransport(
        [
            _response(
                [_call("read_file", {"path": "VERSION"}, call_id="read")],
                response_id="resp_read",
                input_tokens=12,
                output_tokens=3,
                cache_write=2,
            ),
            _response(
                [
                    _call(
                        "submit_result",
                        {"status": "completed", "version": "1.2.3"},
                        call_id="submit",
                    )
                ],
                response_id="resp_submit",
                input_tokens=20,
                output_tokens=5,
                cache_read=7,
            ),
        ]
    )
    adapter = OpenAIAdapter(
        ApiAgentRole.ANALYSIS,
        api_key="sk-proj-secret",
        transport=transport,
    )

    result = adapter.run(_spec(tmp_path, effort="high"))

    assert result.status == AgentStatus.COMPLETED
    assert result.payload == {"status": "completed", "version": "1.2.3"}
    assert result.provider == "openai"
    assert result.model == "gpt-test-2026-08-01"
    assert result.billing_mode == BillingMode.API
    assert result.duration_seconds >= 0
    assert result.attempts == 2
    assert result.usage == AgentUsage(
        tokens_input=32,
        tokens_output=8,
        tokens_cache_read=7,
        tokens_cache_write=2,
        num_turns=2,
    )
    assert json.loads(result.result_path.read_text(encoding="utf-8")) == result.payload

    first_tools = {tool["name"]: tool for tool in transport.payloads[0]["tools"]}
    assert "run_command" not in first_tools
    assert first_tools["submit_result"]["parameters"] == _spec(tmp_path).schema
    assert transport.payloads[0]["input"] == _spec(tmp_path).user_prompt
    assert transport.payloads[0]["reasoning"] == {"effort": "high"}
    assert transport.payloads[1]["previous_response_id"] == "resp_read"
    tool_output = transport.payloads[1]["input"][0]
    assert tool_output["type"] == "function_call_output"
    assert tool_output["call_id"] == "read"
    assert json.loads(tool_output["output"]) == {"content": "1.2.3\n"}


def test_execution_role_advertises_command_only_after_trust(tmp_path: Path) -> None:
    response = _response(
        [
            _call(
                "submit_result",
                {"status": "completed", "version": "one"},
                call_id="submit",
            )
        ]
    )
    untrusted_transport = FakeTransport([response])
    trusted_transport = FakeTransport([response])

    OpenAIAdapter(
        ApiAgentRole.CODING,
        api_key="key",
        transport=untrusted_transport,
    ).run(_spec(tmp_path))
    OpenAIAdapter(
        ApiAgentRole.CODING,
        api_key="key",
        workspace_trusted=True,
        transport=trusted_transport,
    ).run(_spec(tmp_path))

    untrusted_names = {
        tool["name"] for tool in untrusted_transport.payloads[0]["tools"]
    }
    trusted_names = {
        tool["name"] for tool in trusted_transport.payloads[0]["tools"]
    }
    assert "run_command" not in untrusted_names
    assert "run_command" in trusted_names


def test_cancellation_after_in_flight_response_wins(tmp_path: Path) -> None:
    def cancel_during_request(cancel: CancellationToken | None) -> Response:
        assert cancel is not None
        cancel.cancel()
        return _response(
            [
                _call(
                    "submit_result",
                    {"status": "completed", "version": "ignored"},
                    call_id="submit",
                )
            ]
        )

    cancel = CancellationToken()
    adapter = OpenAIAdapter(
        ApiAgentRole.ANALYSIS,
        api_key="key",
        transport=FakeTransport([cancel_during_request]),
    )

    result = adapter.run(_spec(tmp_path), cancel=cancel)

    assert result.status == AgentStatus.CANCELLED
    assert result.resumable
    assert result.attempts == 1
    assert not (tmp_path / "result.json").exists()


def test_cancellation_interrupts_blocked_transport(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    def block_request(_cancel: CancellationToken | None) -> Response:
        started.set()
        release.wait()
        return _response([])

    cancel = CancellationToken()
    adapter = OpenAIAdapter(
        ApiAgentRole.ANALYSIS,
        api_key="key",
        transport=FakeTransport([block_request]),
    )

    def cancel_when_started() -> None:
        assert started.wait(1)
        cancel.cancel()

    canceller = threading.Thread(target=cancel_when_started)
    canceller.start()
    before = time.monotonic()
    try:
        result = adapter.run(_spec(tmp_path), cancel=cancel)
    finally:
        release.set()
        canceller.join()

    assert time.monotonic() - before < 1
    assert result.status == AgentStatus.CANCELLED
    assert result.resumable


def test_incomplete_response_does_not_dispatch_tool(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("unchanged\n", encoding="utf-8")
    response = _response(
        [
            _call(
                "apply_patch",
                {
                    "patch": (
                        "*** Begin Patch\n"
                        "*** Delete File: target.txt\n"
                        "*** End Patch"
                    )
                },
                call_id="patch",
            )
        ],
        status="incomplete",
    )
    response["incomplete_details"] = {"reason": "max_output_tokens"}

    result = OpenAIAdapter(
        ApiAgentRole.CODING,
        api_key="key",
        transport=FakeTransport([response]),
    ).run(_spec(tmp_path))

    assert result.status == AgentStatus.FAILED
    assert "max_output_tokens" in (result.error or "")
    assert target.read_text(encoding="utf-8") == "unchanged\n"


def test_transient_failure_retries_same_turn_and_then_completes(
    tmp_path: Path,
) -> None:
    transient = OpenAIApiError(
        "temporarily overloaded",
        status_code=503,
        error_type="server_error",
    )
    transport = FakeTransport(
        [
            transient,
            _response(
                [
                    _call(
                        "submit_result",
                        {"status": "completed", "version": "retry"},
                        call_id="submit",
                    )
                ]
            ),
        ]
    )
    adapter = OpenAIAdapter(
        ApiAgentRole.ANALYSIS,
        api_key="key",
        transport=transport,
        transient_backoff_seconds=0,
    )

    result = adapter.run(_spec(tmp_path))

    assert result.status == AgentStatus.COMPLETED
    assert result.attempts == 2
    assert transport.payloads[0] == transport.payloads[1]


def test_mid_response_disconnect_retries_through_urllib_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_body = json.dumps(
        _response(
            [
                _call(
                    "submit_result",
                    {"status": "completed", "version": "retry"},
                    call_id="submit",
                )
            ]
        )
    ).encode()
    response_chunks: list[list[bytes | Exception]] = [
        [http.client.IncompleteRead(b'{"object":"response"')],
        [response_body, b""],
    ]
    requests: list[Any] = []

    def urlopen(request: Any, *, timeout: float) -> _ChunkedHttpResponse:
        assert timeout > 0
        requests.append(request)
        return _ChunkedHttpResponse(response_chunks.pop(0))

    monkeypatch.setattr(
        "betterborg_cli.agent_runtime.openai.urllib.request.urlopen",
        urlopen,
    )
    adapter = OpenAIAdapter(
        ApiAgentRole.ANALYSIS,
        api_key="key",
        transport=UrllibOpenAITransport(),
        transient_backoff_seconds=0,
    )

    result = adapter.run(_spec(tmp_path))

    assert result.status == AgentStatus.COMPLETED
    assert result.attempts == 2
    assert len(requests) == 2
    assert requests[0].data == requests[1].data
    assert requests[0].get_header("Authorization") == "Bearer key"


def test_transient_exhaustion_is_cancelled_and_resumable(tmp_path: Path) -> None:
    transport = FakeTransport(
        [
            OpenAIApiError("rate limited", status_code=429),
            OpenAIApiError("rate limited", status_code=429),
        ]
    )
    adapter = OpenAIAdapter(
        ApiAgentRole.ANALYSIS,
        api_key="key",
        transport=transport,
        transient_backoff_seconds=0,
        transient_max_attempts=2,
    )

    result = adapter.run(_spec(tmp_path, resume_token="operation-123"))

    assert result.status == AgentStatus.CANCELLED
    assert result.retryable and result.resumable
    assert result.resume_token == "operation-123"
    assert result.attempts == 2
    assert "transient retry exhausted" in (result.error or "")


@pytest.mark.parametrize(
    ("arguments", "expected_error"),
    [
        (json.dumps({"status": "completed"}), "missing required property 'version'"),
        ("{not-json", "arguments are malformed JSON"),
    ],
)
def test_malformed_submission_fails_without_persisting_result(
    tmp_path: Path,
    arguments: str,
    expected_error: str,
) -> None:
    call = _call(
        "submit_result",
        {"status": "completed", "version": "placeholder"},
        call_id="submit",
    )
    call["arguments"] = arguments
    adapter = OpenAIAdapter(
        ApiAgentRole.ANALYSIS,
        api_key="key",
        transport=FakeTransport([_response([call])]),
    )

    result = adapter.run(_spec(tmp_path))

    assert result.status == AgentStatus.FAILED
    assert expected_error in (result.error or "")
    assert not (tmp_path / "result.json").exists()


def test_credentials_are_redacted_from_errors_and_logs(tmp_path: Path) -> None:
    credential = "sk-proj-api-super-secret"
    transport = FakeTransport(
        [
            OpenAIApiError(
                f"Authorization: Bearer {credential} rejected",
                status_code=401,
            )
        ]
    )
    adapter = OpenAIAdapter(
        ApiAgentRole.ANALYSIS,
        api_key=credential,
        transport=transport,
    )

    result = adapter.run(_spec(tmp_path))

    log = (tmp_path / "openai.jsonl").read_text(encoding="utf-8")
    assert result.status == AgentStatus.FAILED
    assert credential not in (result.error or "")
    assert credential not in log
    assert "[REDACTED]" in (result.error or "")
    assert "[REDACTED]" in log


def test_credentials_are_redacted_from_tool_output_and_payload(tmp_path: Path) -> None:
    credential = "sk-proj-tool-output-secret"
    (tmp_path / "credential.txt").write_text(credential, encoding="utf-8")
    transport = FakeTransport(
        [
            _response(
                [
                    _call(
                        "read_file",
                        {"path": "credential.txt"},
                        call_id="read",
                    )
                ],
                response_id="resp_read",
            ),
            _response(
                [
                    _call(
                        "submit_result",
                        {"status": "completed", "version": credential},
                        call_id="submit",
                    )
                ]
            ),
        ]
    )
    adapter = OpenAIAdapter(
        ApiAgentRole.ANALYSIS,
        api_key=credential,
        transport=transport,
    )

    result = adapter.run(_spec(tmp_path))

    tool_output = transport.payloads[1]["input"][0]["output"]
    persisted = (tmp_path / "result.json").read_text(encoding="utf-8")
    assert result.status == AgentStatus.COMPLETED
    assert json.loads(tool_output) == {"content": "[REDACTED]"}
    assert result.payload == {"status": "completed", "version": "[REDACTED]"}
    assert credential not in persisted
