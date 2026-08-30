"""Provider-neutral agent runtime contracts and mock behavior."""

from __future__ import annotations

import json
import threading
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
    CancellationRegistrationWindow,
    CancellationState,
    CancellationToken,
    ForceTarget,
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


def test_cancellation_token_broadcasts_with_one_absolute_deadline() -> None:
    clock_values = iter([10.0, 99.0])
    cancel = CancellationToken(grace_seconds=0.75, clock=lambda: next(clock_values))
    release = threading.Event()
    entered = [threading.Event() for _ in range(3)]
    deadlines: list[float | None] = []

    def blocking_callback(index: int) -> None:
        deadlines.append(cancel.force_deadline)
        entered[index].set()
        release.wait()

    registrations = [
        cancel.register(lambda index=index: blocking_callback(index))
        for index in range(3)
    ]

    cancel.cancel()

    assert all(event.wait(1) for event in entered)
    assert cancel.state is CancellationState.CANCELLED
    assert cancel.force_deadline == 10.75
    assert deadlines == [10.75, 10.75, 10.75]
    cancel.cancel()
    assert cancel.force_deadline == 10.75
    release.set()
    assert all(registration.unregister() for registration in registrations)


def test_cancellation_token_late_delivery_is_synchronous_and_exactly_once() -> None:
    cancel = CancellationToken(clock=lambda: 5.0)
    cancel.cancel()
    delivered: list[str] = []
    entered = threading.Event()
    release = threading.Event()
    returned = threading.Event()
    registrations = []

    def on_cancel() -> None:
        delivered.append("cancel")
        entered.set()
        release.wait()

    def register_late() -> None:
        registrations.append(cancel.register(on_cancel))
        returned.set()

    thread = threading.Thread(target=register_late)
    thread.start()
    assert entered.wait(1)
    assert not returned.is_set()
    release.set()
    thread.join(timeout=1)

    assert returned.is_set()
    assert delivered == ["cancel"]
    cancel.cancel()
    assert delivered == ["cancel"]
    assert registrations[0].unregister()
    assert not registrations[0].unregister()


def test_cancellation_token_late_force_delivery_preserves_order() -> None:
    cancel = CancellationToken(clock=lambda: 7.0)
    cancel.force()
    delivered: list[str] = []
    cancel_entered = threading.Event()
    release_cancel = threading.Event()
    registration_returned = threading.Event()
    registrations = []
    target = ForceTarget("worker-1")

    def on_cancel() -> None:
        delivered.append("cancel")
        cancel_entered.set()
        release_cancel.wait()

    def register_late() -> None:
        registrations.append(
            cancel.register(
                on_cancel,
                lambda: delivered.append("force"),
                force_target=target,
            )
        )
        registration_returned.set()

    thread = threading.Thread(target=register_late)
    thread.start()
    assert cancel_entered.wait(1)
    assert delivered == ["cancel"]
    assert not registration_returned.is_set()
    release_cancel.set()
    thread.join(timeout=1)

    assert registration_returned.is_set()
    assert delivered == ["cancel", "force"]
    assert cancel.state is CancellationState.FORCED
    assert cancel.force_deadline == 7.0
    assert cancel.force_targets == (target,)
    cancel.force()
    assert delivered == ["cancel", "force"]
    assert registrations[0].unregister()
    assert cancel.force_targets == ()


def test_force_delivers_before_return_without_waiting_for_cancel_callback() -> None:
    cancel = CancellationToken()
    cancel_entered = threading.Event()
    release_cancel = threading.Event()
    force_delivered = threading.Event()

    def on_cancel() -> None:
        cancel_entered.set()
        release_cancel.wait()

    registration = cancel.register(
        on_cancel,
        force_delivered.set,
        force_target=ForceTarget("worker"),
    )

    cancel.cancel()
    assert cancel_entered.wait(1)

    cancel.force()

    assert force_delivered.is_set()
    assert not release_cancel.is_set()
    release_cancel.set()
    assert registration.unregister()


def test_registration_window_retains_created_resource_until_late_force() -> None:
    cancel = CancellationToken()
    window = cancel.registration_window()
    snapshot = cancel.active_windows
    created = threading.Event()
    publish = threading.Event()
    forced = threading.Event()
    registrations = []

    def create_and_publish() -> None:
        created.set()
        publish.wait()
        registrations.append(
            window.register(
                lambda: None,
                forced.set,
                force_target=ForceTarget("created-worker"),
            )
        )

    worker = threading.Thread(target=create_and_publish)
    worker.start()
    assert created.wait(1)
    assert snapshot == (window,)
    assert isinstance(snapshot, tuple)

    cancel.force()

    assert not window.is_settled
    assert cancel.active_windows == (window,)
    with pytest.raises(RuntimeError, match="after force"):
        cancel.registration_window()

    publish.set()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert forced.is_set()
    assert window.wait(0)
    assert cancel.active_windows == ()
    assert registrations[0].force_target == ForceTarget("created-worker")
    assert registrations[0].unregister()


def test_registration_window_settles_conclusive_precreation_failure() -> None:
    cancel = CancellationToken()
    window = cancel.registration_window()
    failure_confirmed = threading.Event()
    release = threading.Event()

    def fail_before_creation() -> None:
        release.wait()
        failure_confirmed.set()
        window.no_resource()

    worker = threading.Thread(target=fail_before_creation)
    worker.start()
    snapshot = cancel.active_windows
    cancel.force()

    assert snapshot == (window,)
    assert not window.wait(0)
    release.set()
    worker.join(timeout=1)

    assert failure_confirmed.is_set()
    assert window.is_settled
    assert snapshot[0].wait(0)
    assert cancel.active_windows == ()
    with pytest.raises(RuntimeError, match="already settled"):
        window.no_resource()


def test_registration_window_requires_force_target_and_single_settlement() -> None:
    cancel = CancellationToken()
    window = cancel.registration_window()

    with pytest.raises(ValueError, match="validated force target"):
        window.register(lambda: None)

    registration = window.register(
        lambda: None,
        lambda: None,
        force_target=ForceTarget("worker"),
    )

    assert isinstance(window, CancellationRegistrationWindow)
    assert window.is_settled
    with pytest.raises(RuntimeError, match="already settled"):
        window.register(
            lambda: None,
            lambda: None,
            force_target=ForceTarget("other-worker"),
        )
    assert registration.unregister()


def test_cancellation_token_unregister_before_cancel_prevents_delivery() -> None:
    cancel = CancellationToken()
    delivered: list[str] = []
    registration = cancel.register(lambda: delivered.append("cancel"))

    assert registration.unregister()
    cancel.cancel()

    assert delivered == []


def test_cleanup_registration_skips_cancel_but_remains_force_deliverable() -> None:
    cancel = CancellationToken(grace_seconds=0.5, clock=lambda: 3.0)
    delivered: list[str] = []
    forced = threading.Event()
    target = ForceTarget(123)

    def on_force() -> None:
        delivered.append("force")
        forced.set()

    registration = cancel.register(
        lambda: delivered.append("cancel"),
        on_force,
        terminate_on_cancel=False,
        force_target=target,
    )

    cancel.cancel()
    assert delivered == []
    assert cancel.force_deadline == 3.5
    assert cancel.force_targets == (target,)

    cancel.force()
    assert forced.wait(1)
    assert delivered == ["force"]
    assert cancel.force_targets == (target,)
    assert registration.unregister()
    assert cancel.force_targets == ()


@pytest.mark.parametrize("identity", [None, True, 0, -1, ""])
def test_force_target_rejects_unvalidated_identity(identity: object) -> None:
    with pytest.raises(ValueError):
        ForceTarget(identity)


def test_cancellation_token_requires_force_callback_and_target_together() -> None:
    cancel = CancellationToken()

    with pytest.raises(ValueError, match="provided together"):
        cancel.register(lambda: None, lambda: None)
    with pytest.raises(ValueError, match="provided together"):
        cancel.register(lambda: None, force_target=ForceTarget("worker"))
    with pytest.raises(ValueError, match="force-deliverable"):
        cancel.register(lambda: None, terminate_on_cancel=False)


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
