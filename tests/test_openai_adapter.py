"""OpenAI Responses API adapter behavior with hermetic transports."""

from __future__ import annotations

import http.client
import json
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
    openai_function_call as _call,
)
from test_adapter_harness import (
    openai_response as _response,
)
from test_adapter_harness import (
    openai_spec as _spec,
)

from betterborg_cli.agent_runtime import (
    AgentStatus,
    ApiAgentRole,
    OpenAIAdapter,
    UrllibOpenAITransport,
)


def test_multi_turn_responses_use_openai_wire_format(tmp_path: Path) -> None:
    (tmp_path / "VERSION").write_text("1.2.3\n", encoding="utf-8")
    transport = FakeTransport(
        [
            _response(
                [_call("read_file", {"path": "VERSION"}, call_id="read")],
                response_id="resp_read",
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


def test_malformed_json_submission_fails_without_persisting_result(
    tmp_path: Path,
) -> None:
    call = _call(
        "submit_result",
        {"status": "completed", "version": "placeholder"},
        call_id="submit",
    )
    call["arguments"] = "{not-json"
    adapter = OpenAIAdapter(
        ApiAgentRole.ANALYSIS,
        api_key="key",
        transport=FakeTransport([_response([call])]),
    )

    result = adapter.run(_spec(tmp_path))

    assert result.status == AgentStatus.FAILED
    assert "arguments are malformed JSON" in (result.error or "")
    assert not (tmp_path / "result.json").exists()
