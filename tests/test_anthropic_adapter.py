"""Anthropic Messages API adapter behavior with hermetic transports."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from betterborg_cli.agent_runtime import (
    AgentRunSpec,
    AgentStatus,
    AgentUsage,
    AnthropicAdapter,
    AnthropicApiError,
    ApiAgentRole,
    BillingMode,
    CancellationToken,
)

Response = Mapping[str, Any]
QueuedResponse = Response | Exception | Callable[[CancellationToken | None], Response]


@dataclass
class FakeTransport:
    responses: list[QueuedResponse]
    payloads: list[dict[str, Any]] = field(default_factory=list)
    api_keys: list[str] = field(default_factory=list)

    def create_message(
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
        "model": "claude-test",
        "log_path": tmp_path / "anthropic.jsonl",
        "result_path": tmp_path / "result.json",
    }
    values.update(changes)
    return AgentRunSpec(**values)


def _message(
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


def test_multi_turn_tools_submit_schema_and_persist_metadata(tmp_path: Path) -> None:
    (tmp_path / "VERSION").write_text("1.2.3\n", encoding="utf-8")
    transport = FakeTransport(
        [
            _message(
                [
                    {
                        "type": "tool_use",
                        "id": "tool_read",
                        "name": "read_file",
                        "input": {"path": "VERSION"},
                    }
                ],
                input_tokens=12,
                output_tokens=3,
                cache_write=2,
            ),
            _message(
                [
                    {
                        "type": "tool_use",
                        "id": "tool_submit",
                        "name": "submit_result",
                        "input": {"status": "completed", "version": "1.2.3"},
                    }
                ],
                input_tokens=20,
                output_tokens=5,
                cache_read=7,
            ),
        ]
    )
    adapter = AnthropicAdapter(
        ApiAgentRole.ANALYSIS,
        api_key="sk-ant-secret",
        transport=transport,
    )

    result = adapter.run(_spec(tmp_path))

    assert result.status == AgentStatus.COMPLETED
    assert result.payload == {"status": "completed", "version": "1.2.3"}
    assert result.provider == "anthropic"
    assert result.model == "claude-test-20260801"
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
    assert first_tools["submit_result"]["input_schema"] == _spec(tmp_path).schema
    assert transport.payloads[1]["messages"][-2] == {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "tool_read",
                "name": "read_file",
                "input": {"path": "VERSION"},
            }
        ],
    }
    tool_result = transport.payloads[1]["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "tool_read"
    assert json.loads(tool_result["content"]) == {"content": "1.2.3\n"}


def test_execution_role_advertises_command_only_after_trust(tmp_path: Path) -> None:
    response = _message(
        [
            {
                "type": "tool_use",
                "id": "submit",
                "name": "submit_result",
                "input": {"status": "completed", "version": "one"},
            }
        ]
    )
    untrusted_transport = FakeTransport([response])
    trusted_transport = FakeTransport([response])

    AnthropicAdapter(
        ApiAgentRole.CODING,
        api_key="key",
        transport=untrusted_transport,
    ).run(_spec(tmp_path))
    AnthropicAdapter(
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
        return _message(
            [
                {
                    "type": "tool_use",
                    "id": "submit",
                    "name": "submit_result",
                    "input": {"status": "completed", "version": "ignored"},
                }
            ]
        )

    cancel = CancellationToken()
    transport = FakeTransport([cancel_during_request])
    adapter = AnthropicAdapter(
        ApiAgentRole.ANALYSIS,
        api_key="key",
        transport=transport,
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
        return _message([])

    cancel = CancellationToken()
    adapter = AnthropicAdapter(
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


def test_cancellation_terminates_in_flight_command(tmp_path: Path) -> None:
    started_path = tmp_path / "command-started"
    cancel = CancellationToken()
    transport = FakeTransport(
        [
            _message(
                [
                    {
                        "type": "tool_use",
                        "id": "command",
                        "name": "run_command",
                        "input": {
                            "argv": [
                                sys.executable,
                                "-c",
                                (
                                    "from pathlib import Path; import time; "
                                    "Path('command-started').write_text('yes'); "
                                    "time.sleep(3)"
                                ),
                            ]
                        },
                    }
                ]
            )
        ]
    )
    adapter = AnthropicAdapter(
        ApiAgentRole.CODING,
        api_key="key",
        workspace_trusted=True,
        transport=transport,
    )

    def cancel_when_started() -> None:
        deadline = time.monotonic() + 1
        while not started_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert started_path.exists()
        cancel.cancel()

    canceller = threading.Thread(target=cancel_when_started)
    canceller.start()
    before = time.monotonic()
    result = adapter.run(_spec(tmp_path), cancel=cancel)
    canceller.join()

    assert time.monotonic() - before < 2
    assert result.status == AgentStatus.CANCELLED
    assert result.resumable
    assert len(transport.payloads) == 1


def test_truncated_tool_response_does_not_dispatch_tool(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("unchanged\n", encoding="utf-8")
    transport = FakeTransport(
        [
            _message(
                [
                    {
                        "type": "tool_use",
                        "id": "patch",
                        "name": "apply_patch",
                        "input": {
                            "patch": (
                                "*** Begin Patch\n"
                                "*** Delete File: target.txt\n"
                                "*** End Patch"
                            )
                        },
                    }
                ],
                stop_reason="max_tokens",
            )
        ]
    )
    adapter = AnthropicAdapter(
        ApiAgentRole.CODING,
        api_key="key",
        transport=transport,
    )

    result = adapter.run(_spec(tmp_path))

    assert result.status == AgentStatus.FAILED
    assert "without a tool_use stop reason (max_tokens)" in (result.error or "")
    assert target.read_text(encoding="utf-8") == "unchanged\n"


def test_transient_failure_retries_same_turn_and_then_completes(
    tmp_path: Path,
) -> None:
    transient = AnthropicApiError(
        "temporarily overloaded",
        status_code=529,
        error_type="overloaded_error",
    )
    transport = FakeTransport(
        [
            transient,
            _message(
                [
                    {
                        "type": "tool_use",
                        "id": "submit",
                        "name": "submit_result",
                        "input": {"status": "completed", "version": "retry"},
                    }
                ]
            ),
        ]
    )
    adapter = AnthropicAdapter(
        ApiAgentRole.ANALYSIS,
        api_key="key",
        transport=transport,
        transient_backoff_seconds=0,
    )

    result = adapter.run(_spec(tmp_path))

    assert result.status == AgentStatus.COMPLETED
    assert result.attempts == 2
    assert transport.payloads[0] == transport.payloads[1]


def test_transient_exhaustion_is_cancelled_and_resumable(tmp_path: Path) -> None:
    errors = [
        AnthropicApiError("rate limited", status_code=429),
        AnthropicApiError("rate limited", status_code=429),
    ]
    transport = FakeTransport(errors)
    adapter = AnthropicAdapter(
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


def test_schema_invalid_submission_fails_without_persisting_result(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(
        [
            _message(
                [
                    {
                        "type": "tool_use",
                        "id": "submit",
                        "name": "submit_result",
                        "input": {"status": "completed"},
                    }
                ]
            )
        ]
    )
    adapter = AnthropicAdapter(
        ApiAgentRole.ANALYSIS,
        api_key="key",
        transport=transport,
    )

    result = adapter.run(_spec(tmp_path))

    assert result.status == AgentStatus.FAILED
    assert "missing required property 'version'" in (result.error or "")
    assert not (tmp_path / "result.json").exists()


def test_credentials_are_redacted_from_errors_and_logs(tmp_path: Path) -> None:
    credential = "sk-ant-api03-super-secret"
    transport = FakeTransport(
        [AnthropicApiError(f"x-api-key: {credential} rejected", status_code=401)]
    )
    adapter = AnthropicAdapter(
        ApiAgentRole.ANALYSIS,
        api_key=credential,
        transport=transport,
    )

    result = adapter.run(_spec(tmp_path))

    log = (tmp_path / "anthropic.jsonl").read_text(encoding="utf-8")
    assert result.status == AgentStatus.FAILED
    assert credential not in (result.error or "")
    assert credential not in log
    assert "[REDACTED]" in (result.error or "")
    assert "[REDACTED]" in log
