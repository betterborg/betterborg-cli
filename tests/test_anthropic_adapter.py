"""Anthropic Messages API adapter behavior with hermetic transports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_adapter_harness import (
    FakeApiTransport as FakeTransport,
)
from test_adapter_harness import (
    FakeUrlRequestFactory,
)
from test_adapter_harness import (
    anthropic_message as _message,
)
from test_adapter_harness import (
    anthropic_spec as _spec,
)

from betterborg_cli.agent_runtime import (
    ANTHROPIC_API_VERSION,
    AgentStatus,
    AnthropicAdapter,
    ApiAgentRole,
    UrllibAnthropicTransport,
    UrlResponse,
    UrlTransportError,
)


def test_multi_turn_messages_use_anthropic_wire_format(tmp_path: Path) -> None:
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
                ]
            ),
            _message(
                [
                    {
                        "type": "tool_use",
                        "id": "tool_submit",
                        "name": "submit_result",
                        "input": {"status": "completed", "version": "1.2.3"},
                    }
                ]
            ),
        ]
    )
    adapter = AnthropicAdapter(
        ApiAgentRole.ANALYSIS,
        api_key="sk-ant-secret",
        transport=transport,
    )

    result = adapter.run(_spec(tmp_path, effort="high"))

    assert result.status == AgentStatus.COMPLETED

    assert all(
        payload["tool_choice"] == {"type": "any"}
        for payload in transport.payloads
    )
    first_tools = {tool["name"]: tool for tool in transport.payloads[0]["tools"]}
    assert "run_command" not in first_tools
    assert first_tools["submit_result"]["input_schema"] == _spec(tmp_path).schema
    assert transport.payloads[0]["output_config"] == {"effort": "high"}
    assert transport.payloads[1]["output_config"] == {"effort": "high"}
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


def test_message_without_effort_omits_output_config(tmp_path: Path) -> None:
    transport = FakeTransport(
        [
            _message(
                [
                    {
                        "type": "tool_use",
                        "id": "tool_submit",
                        "name": "submit_result",
                        "input": {"status": "completed", "version": "1.2.3"},
                    }
                ]
            )
        ]
    )
    adapter = AnthropicAdapter(
        ApiAgentRole.ANALYSIS,
        api_key="sk-ant-secret",
        transport=transport,
    )

    result = adapter.run(_spec(tmp_path))

    assert result.status == AgentStatus.COMPLETED
    assert "output_config" not in transport.payloads[0]


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


@pytest.mark.parametrize(
    "disconnect",
    [
        ConnectionResetError("connection reset during response"),
        OSError("incomplete response body"),
    ],
)
def test_mid_response_disconnect_retries_through_urllib_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    disconnect: Exception,
) -> None:
    response_body = json.dumps(
        _message(
            [
                {
                    "type": "tool_use",
                    "id": "submit",
                    "name": "submit_result",
                    "input": {"status": "completed", "version": "retry"},
                }
            ]
        )
    ).encode()
    requests = FakeUrlRequestFactory(
        [
            UrlTransportError("response", str(disconnect)),
            UrlResponse(200, "OK", response_body),
        ]
    )
    monkeypatch.setattr(
        "betterborg_cli.agent_runtime.anthropic.MultiprocessUrlRequest",
        requests,
    )
    adapter = AnthropicAdapter(
        ApiAgentRole.ANALYSIS,
        api_key="key",
        transport=UrllibAnthropicTransport("https://provider.invalid/messages"),
        transient_backoff_seconds=0,
    )

    result = adapter.run(_spec(tmp_path))

    assert result.status == AgentStatus.COMPLETED
    assert result.attempts == 2
    assert len(requests.specs) == 2
    assert requests.specs[0] == requests.specs[1]
    assert requests.specs[0].url == "https://provider.invalid/messages"
    assert requests.specs[0].timeout_seconds == 60.0
    assert requests.specs[0].headers == {
        "anthropic-version": ANTHROPIC_API_VERSION,
        "content-type": "application/json",
        "x-api-key": "key",
    }


@pytest.mark.parametrize("status_code", [429, 529])
def test_transient_http_error_retries_when_error_body_disconnects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    response_body = json.dumps(
        _message(
            [
                {
                    "type": "tool_use",
                    "id": "submit",
                    "name": "submit_result",
                    "input": {"status": "completed", "version": "retry"},
                }
            ]
        )
    ).encode()
    requests = FakeUrlRequestFactory(
        [
            UrlTransportError(
                "response",
                "incomplete response body",
                status_code=status_code,
                reason="transient error",
            ),
            UrlResponse(200, "OK", response_body),
        ]
    )
    monkeypatch.setattr(
        "betterborg_cli.agent_runtime.anthropic.MultiprocessUrlRequest",
        requests,
    )
    adapter = AnthropicAdapter(
        ApiAgentRole.ANALYSIS,
        api_key="key",
        transport=UrllibAnthropicTransport(),
        transient_backoff_seconds=0,
    )

    result = adapter.run(_spec(tmp_path))

    assert result.status == AgentStatus.COMPLETED
    assert result.attempts == 2
    assert len(requests.specs) == 2
    assert requests.specs[0] == requests.specs[1]
