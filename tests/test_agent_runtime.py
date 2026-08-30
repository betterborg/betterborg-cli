"""Provider-neutral agent runtime contracts and mock behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from betterborg_cli.agent_runtime import (
    AgentActivity,
    AgentActivityKind,
    AgentArtifact,
    AgentRunSpec,
    AgentStatus,
    AgentUsage,
    BillingMode,
    CancellationToken,
    MockAdapter,
    MockResponse,
    StructuredResultError,
    combine_agent_usage,
    extract_structured_result,
    retry_outcome_to_result,
    run_with_transient_retry,
    validate_structured_result,
)


def _spec(tmp_path: Path, **changes) -> AgentRunSpec:
    values = {
        "system_prompt": "Act as a coding agent.",
        "user_prompt": "Complete this task.",
        "schema": {
            "type": "object",
            "required": ["status"],
            "properties": {
                "status": {"type": "string", "enum": ["completed"]},
                "count": {"type": "integer", "minimum": 0},
            },
            "additionalProperties": False,
        },
        "cwd": tmp_path,
        "model": "test-model",
        "log_path": tmp_path / "agent.log",
        "result_path": tmp_path / "result.json",
    }
    values.update(changes)
    return AgentRunSpec(**values)


def test_agent_run_spec_activity_sink_is_optional_and_provider_neutral(
    tmp_path: Path,
) -> None:
    assert _spec(tmp_path).activity_sink is None
    received: list[AgentActivity] = []
    spec = _spec(tmp_path, activity_sink=received.append)
    activity = AgentActivity(AgentActivityKind.SEARCHING, "AgentRunSpec")

    assert spec.activity_sink is not None
    spec.activity_sink(activity)

    assert received == [activity]
    assert activity.kind is AgentActivityKind.SEARCHING


def test_mock_completed_run_preserves_metadata_and_writes_result(
    tmp_path: Path,
) -> None:
    usage = AgentUsage(cost_usd=0.25, tokens_input=10, tokens_output=4)
    artifact = AgentArtifact(tmp_path / "patch.diff", kind="patch")
    adapter = MockAdapter().queue(
        MockResponse(
            payload={"status": "completed", "count": 2},
            usage=usage,
            billing_mode=BillingMode.SUBSCRIPTION,
            artifacts=(artifact,),
        )
    )

    result = adapter.run(_spec(tmp_path))

    assert result.status == AgentStatus.COMPLETED
    assert result.usage == usage
    assert result.billing_mode == BillingMode.SUBSCRIPTION
    assert result.artifacts == (artifact,)
    assert result.result_path == tmp_path / "result.json"
    assert json.loads(result.result_path.read_text(encoding="utf-8")) == {
        "count": 2,
        "status": "completed",
    }
    assert adapter.calls[0].model == "test-model"


def test_mock_failed_run_preserves_failure_details(tmp_path: Path) -> None:
    adapter = MockAdapter().queue(MockResponse(exit_code=17, error="provider failed"))

    result = adapter.run(_spec(tmp_path))

    assert result.status == AgentStatus.FAILED
    assert result.exit_code == 17
    assert result.error == "provider failed"
    assert result.result_path is None


def test_mock_rejects_schema_invalid_payload(tmp_path: Path) -> None:
    adapter = MockAdapter().queue(MockResponse(payload={"count": -1}))

    result = adapter.run(_spec(tmp_path))

    assert result.status == AgentStatus.FAILED
    assert result.exit_code == 0
    assert "missing required property 'status'" in (result.error or "")
    assert not (tmp_path / "result.json").exists()


def test_mock_honors_preexisting_cancellation_without_consuming_response(
    tmp_path: Path,
) -> None:
    cancel = CancellationToken()
    cancel.cancel()
    adapter = MockAdapter().queue(MockResponse(payload={"status": "completed"}))

    result = adapter.run(_spec(tmp_path), cancel=cancel)

    assert result.status == AgentStatus.CANCELLED
    assert result.resumable
    assert len(adapter.responses) == 1


def test_mock_dynamic_response_uses_run_spec(tmp_path: Path) -> None:
    adapter = MockAdapter().queue(
        MockResponse(
            dynamic=lambda spec: {"status": "completed", "count": len(spec.model)}
        )
    )

    result = adapter.run(_spec(tmp_path, model="abcd"))

    assert result.payload == {"status": "completed", "count": 4}


def test_structured_extraction_skips_invalid_event_before_valid_result() -> None:
    schema = {
        "type": "object",
        "required": ["status"],
        "properties": {"status": {"const": "completed"}},
        "additionalProperties": False,
    }
    output = 'event: {"progress": 50}\n```json\n{"status":"completed"}\n```'

    assert extract_structured_result(output, schema) == {"status": "completed"}


def test_structured_validation_supports_local_references() -> None:
    schema = {
        "$defs": {
            "item": {
                "type": "string",
                "minLength": 1,
                "pattern": "^[a-z]+$",
            }
        },
        "type": "object",
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"$ref": "#/$defs/item"},
            }
        },
        "additionalProperties": False,
    }

    validate_structured_result({"items": ["one", "two"]}, schema)
    with pytest.raises(StructuredResultError, match="must be unique"):
        validate_structured_result({"items": ["one", "one"]}, schema)


def test_usage_combines_reported_fields_without_inventing_unknowns() -> None:
    combined = combine_agent_usage(
        [
            AgentUsage(cost_usd=1.5, tokens_input=10),
            None,
            AgentUsage(cost_usd=0.5, tokens_output=4),
        ]
    )

    assert combined == AgentUsage(cost_usd=2.0, tokens_input=10, tokens_output=4)


def test_transient_retry_exhaustion_is_bounded_and_resumable(tmp_path: Path) -> None:
    calls: list[int] = []

    def fake_process_runner() -> int:
        calls.append(1)
        return 75

    outcome = run_with_transient_retry(
        fake_process_runner,
        lambda exit_code: "provider overloaded" if exit_code == 75 else None,
        backoff_seconds=0,
        max_attempts=3,
    )
    usage = AgentUsage(tokens_input=30)
    artifact = AgentArtifact("artifact://partial/transcript", kind="transcript")
    spec = _spec(
        tmp_path,
        billing_mode=BillingMode.SUBSCRIPTION,
        resume_token="session-123",
    )

    result = retry_outcome_to_result(
        outcome,
        spec,
        duration_seconds=1.25,
        usage=usage,
        artifacts=(artifact,),
    )

    assert len(calls) == 3
    assert outcome.exhausted and outcome.resumable
    assert result.status == AgentStatus.CANCELLED
    assert result.attempts == 3
    assert result.resumable
    assert result.resume_token == "session-123"
    assert result.usage == usage
    assert result.billing_mode == BillingMode.SUBSCRIPTION
    assert result.artifacts == (artifact,)
    assert "provider overloaded" in (result.error or "")


def test_transient_retry_cancels_during_backoff() -> None:
    cancel = CancellationToken()

    def classify(_exit_code: int) -> str:
        cancel.cancel()
        return "temporary"

    outcome = run_with_transient_retry(
        lambda: 75,
        classify,
        cancel=cancel,
        backoff_seconds=60,
        max_attempts=3,
    )

    assert outcome.cancelled
    assert outcome.attempts == 1
