"""Anthropic Messages API adapter behavior with hermetic transports."""

from __future__ import annotations

import http.client
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest
from test_adapter_harness import (
    ChunkedHttpResponse as _ChunkedHttpResponse,
)
from test_adapter_harness import (
    FakeApiTransport as FakeTransport,
)
from test_adapter_harness import (
    anthropic_message as _message,
)
from test_adapter_harness import (
    anthropic_spec as _spec,
)

from betterborg_cli.agent_runtime import (
    AgentStatus,
    AnthropicAdapter,
    ApiAgentRole,
    UrllibAnthropicTransport,
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

    result = adapter.run(_spec(tmp_path))

    assert result.status == AgentStatus.COMPLETED

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
        http.client.IncompleteRead(b'{"type":"message"'),
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
    response_chunks: list[list[bytes | Exception]] = [
        [disconnect],
        [response_body, b""],
    ]
    request_bodies: list[bytes | None] = []

    def urlopen(request: Any, *, timeout: float) -> _ChunkedHttpResponse:
        assert timeout > 0
        request_bodies.append(request.data)
        return _ChunkedHttpResponse(response_chunks.pop(0))

    monkeypatch.setattr(
        "betterborg_cli.agent_runtime.anthropic.urllib.request.urlopen",
        urlopen,
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
    assert len(request_bodies) == 2
    assert request_bodies[0] == request_bodies[1]


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
    request_bodies: list[bytes | None] = []

    def urlopen(request: Any, *, timeout: float) -> _ChunkedHttpResponse:
        assert timeout > 0
        request_bodies.append(request.data)
        if len(request_bodies) == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                status_code,
                "transient error",
                {},
                _ChunkedHttpResponse(
                    [http.client.IncompleteRead(b'{"type":"error"')]
                ),
            )
        return _ChunkedHttpResponse([response_body, b""])

    monkeypatch.setattr(
        "betterborg_cli.agent_runtime.anthropic.urllib.request.urlopen",
        urlopen,
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
    assert len(request_bodies) == 2
    assert request_bodies[0] == request_bodies[1]
