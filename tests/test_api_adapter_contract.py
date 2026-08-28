"""Provider-neutral behavioral contract for contained API adapters."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest
from test_adapter_harness import (
    API_ADAPTER_HARNESSES,
    ApiAdapterHarness,
    FakeApiTransport,
)

from betterborg_cli.agent_runtime import (
    AgentStatus,
    AgentUsage,
    ApiAgentRole,
    BillingMode,
    CancellationToken,
)


@pytest.fixture(params=API_ADAPTER_HARNESSES, ids=lambda harness: harness.provider)
def harness(request: pytest.FixtureRequest) -> ApiAdapterHarness:
    return request.param


def test_structured_result_persists_usage_and_metadata(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    transport = FakeApiTransport(
        [
            harness.response(
                [
                    harness.tool_call(
                        "submit_result",
                        {"status": "completed", "version": "1.2.3"},
                        call_id="submit",
                    )
                ],
                input_tokens=20,
                output_tokens=5,
                cache_read=7,
                cache_write=2,
            )
        ]
    )

    result = harness.adapter(
        ApiAgentRole.ANALYSIS,
        transport=transport,
    ).run(harness.spec(tmp_path))

    assert result.status == AgentStatus.COMPLETED
    assert result.payload == {"status": "completed", "version": "1.2.3"}
    assert result.provider == harness.provider
    assert result.model == harness.resolved_model
    assert result.billing_mode == BillingMode.API
    assert result.duration_seconds >= 0
    assert result.attempts == 1
    assert result.usage == AgentUsage(
        tokens_input=20,
        tokens_output=5,
        tokens_cache_read=7,
        tokens_cache_write=2,
        num_turns=1,
    )
    assert json.loads(result.result_path.read_text(encoding="utf-8")) == result.payload


def test_execution_role_advertises_command_only_after_trust(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    response = harness.response(
        [
            harness.tool_call(
                "submit_result",
                {"status": "completed", "version": "one"},
                call_id="submit",
            )
        ]
    )
    untrusted_transport = FakeApiTransport([response])
    trusted_transport = FakeApiTransport([response])

    harness.adapter(
        ApiAgentRole.CODING,
        transport=untrusted_transport,
    ).run(harness.spec(tmp_path))
    harness.adapter(
        ApiAgentRole.CODING,
        transport=trusted_transport,
        workspace_trusted=True,
    ).run(harness.spec(tmp_path))

    untrusted_names = {
        tool["name"] for tool in untrusted_transport.payloads[0]["tools"]
    }
    trusted_names = {
        tool["name"] for tool in trusted_transport.payloads[0]["tools"]
    }
    assert "run_command" not in untrusted_names
    assert "run_command" in trusted_names


def test_cancellation_after_in_flight_response_wins(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    def cancel_during_request(cancel: CancellationToken | None):
        assert cancel is not None
        cancel.cancel()
        return harness.response(
            [
                harness.tool_call(
                    "submit_result",
                    {"status": "completed", "version": "ignored"},
                    call_id="submit",
                )
            ]
        )

    cancel = CancellationToken()
    result = harness.adapter(
        ApiAgentRole.ANALYSIS,
        transport=FakeApiTransport([cancel_during_request]),
    ).run(harness.spec(tmp_path), cancel=cancel)

    assert result.status == AgentStatus.CANCELLED
    assert result.resumable
    assert result.attempts == 1
    assert not (tmp_path / "result.json").exists()


def test_cancellation_interrupts_blocked_transport(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def block_request(_cancel: CancellationToken | None):
        started.set()
        release.wait()
        return harness.response([])

    cancel = CancellationToken()
    adapter = harness.adapter(
        ApiAgentRole.ANALYSIS,
        transport=FakeApiTransport([block_request]),
    )

    def cancel_when_started() -> None:
        assert started.wait(1)
        cancel.cancel()

    canceller = threading.Thread(target=cancel_when_started)
    canceller.start()
    before = time.monotonic()
    try:
        result = adapter.run(harness.spec(tmp_path), cancel=cancel)
    finally:
        release.set()
        canceller.join()

    assert time.monotonic() - before < 1
    assert result.status == AgentStatus.CANCELLED
    assert result.resumable


def test_cancellation_terminates_in_flight_command(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    started_path = tmp_path / "command-started"
    cancel = CancellationToken()
    transport = FakeApiTransport(
        [
            harness.response(
                [
                    harness.tool_call(
                        "run_command",
                        {
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
                        call_id="command",
                    )
                ]
            )
        ]
    )
    adapter = harness.adapter(
        ApiAgentRole.CODING,
        transport=transport,
        workspace_trusted=True,
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
    result = adapter.run(harness.spec(tmp_path), cancel=cancel)
    canceller.join()

    assert time.monotonic() - before < 2
    assert result.status == AgentStatus.CANCELLED
    assert result.resumable
    assert len(transport.payloads) == 1


def test_transient_failure_retries_same_turn_and_then_completes(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    transport = FakeApiTransport(
        [
            harness.transient_error(),
            harness.response(
                [
                    harness.tool_call(
                        "submit_result",
                        {"status": "completed", "version": "retry"},
                        call_id="submit",
                    )
                ]
            ),
        ]
    )
    adapter = harness.adapter(
        ApiAgentRole.ANALYSIS,
        transport=transport,
        transient_backoff_seconds=0,
    )

    result = adapter.run(harness.spec(tmp_path))

    assert result.status == AgentStatus.COMPLETED
    assert result.attempts == 2
    assert transport.payloads[0] == transport.payloads[1]


def test_transient_exhaustion_is_cancelled_and_resumable(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    transport = FakeApiTransport(
        [harness.transient_error("rate limited") for _attempt in range(2)]
    )
    adapter = harness.adapter(
        ApiAgentRole.ANALYSIS,
        transport=transport,
        transient_backoff_seconds=0,
        transient_max_attempts=2,
    )

    result = adapter.run(harness.spec(tmp_path, resume_token="operation-123"))

    assert result.status == AgentStatus.CANCELLED
    assert result.retryable and result.resumable
    assert result.resume_token == "operation-123"
    assert result.attempts == 2
    assert "transient retry exhausted" in (result.error or "")


def test_schema_invalid_submission_fails_without_persisting_result(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    transport = FakeApiTransport(
        [
            harness.response(
                [
                    harness.tool_call(
                        "submit_result",
                        {"status": "completed"},
                        call_id="submit",
                    )
                ]
            )
        ]
    )

    result = harness.adapter(
        ApiAgentRole.ANALYSIS,
        transport=transport,
    ).run(harness.spec(tmp_path))

    assert result.status == AgentStatus.FAILED
    assert "missing required property 'version'" in (result.error or "")
    assert not (tmp_path / "result.json").exists()


def test_credentials_are_redacted_from_errors_and_logs(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    credential = harness.credential
    transport = FakeApiTransport(
        [harness.api_error(f"Authorization: {credential} rejected")]
    )
    adapter = harness.adapter(
        ApiAgentRole.ANALYSIS,
        api_key=credential,
        transport=transport,
    )

    result = adapter.run(harness.spec(tmp_path))

    log = (tmp_path / f"{harness.provider}.jsonl").read_text(encoding="utf-8")
    assert result.status == AgentStatus.FAILED
    assert credential not in (result.error or "")
    assert credential not in log
    assert "[REDACTED]" in (result.error or "")
    assert "[REDACTED]" in log


def test_adapter_requires_an_injected_credential(
    tmp_path: Path,
    harness: ApiAdapterHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variable = (
        "ANTHROPIC_API_KEY"
        if harness.provider == "anthropic"
        else "OPENAI_API_KEY"
    )
    monkeypatch.setenv(variable, "ambient-secret")
    transport = FakeApiTransport(
        [
            harness.response(
                [
                    harness.tool_call(
                        "submit_result",
                        {"status": "completed", "version": "unexpected"},
                        call_id="submit",
                    )
                ]
            )
        ]
    )

    result = harness.adapter(
        ApiAgentRole.ANALYSIS,
        api_key=None,
        transport=transport,
    ).run(harness.spec(tmp_path, env={variable: "spec-secret"}))

    assert result.status == AgentStatus.FAILED
    provider_name = "Anthropic" if harness.provider == "anthropic" else "OpenAI"
    assert result.error == f"{provider_name} API credential is not configured"
    assert transport.payloads == []


def test_credentials_are_redacted_from_tools_and_completed_payload(
    tmp_path: Path,
    harness: ApiAdapterHarness,
) -> None:
    credential = harness.credential
    (tmp_path / "credential.txt").write_text(credential, encoding="utf-8")
    transport = FakeApiTransport(
        [
            harness.response(
                [
                    harness.tool_call(
                        "read_file",
                        {"path": "credential.txt"},
                        call_id="read",
                    )
                ],
                response_id="read_response",
            ),
            harness.response(
                [
                    harness.tool_call(
                        "submit_result",
                        {"status": "completed", "version": credential},
                        call_id="submit",
                    )
                ]
            ),
        ]
    )

    result = harness.adapter(
        ApiAgentRole.ANALYSIS,
        api_key=credential,
        transport=transport,
    ).run(harness.spec(tmp_path))

    tool_output = harness.extract_tool_output(transport.payloads[1])
    persisted = (tmp_path / "result.json").read_text(encoding="utf-8")
    assert result.status == AgentStatus.COMPLETED
    assert json.loads(tool_output) == {"content": "[REDACTED]"}
    assert result.payload == {"status": "completed", "version": "[REDACTED]"}
    assert credential not in persisted
    assert json.loads(persisted) == result.payload
