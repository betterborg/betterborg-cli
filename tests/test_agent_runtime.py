"""Provider-neutral agent runtime contracts and mock behavior."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from betterborg_cli.agent_runtime import (
    DEFAULT_SCHEMA_MAX_ATTEMPTS,
    AgentActivity,
    AgentActivityKind,
    AgentArtifact,
    AgentRunSpec,
    AgentStatus,
    AgentUsage,
    BillingMode,
    CancellationDeliveryError,
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


def test_mock_response_emits_scripted_provider_neutral_activity(
    tmp_path: Path,
) -> None:
    activities = (
        AgentActivity(AgentActivityKind.THINKING),
        AgentActivity(AgentActivityKind.READING, "pyproject.toml"),
    )
    received: list[AgentActivity] = []
    adapter = MockAdapter().queue(
        MockResponse(payload={"status": "completed"}, activities=activities)
    )

    result = adapter.run(_spec(tmp_path, activity_sink=received.append))

    assert result.status == AgentStatus.COMPLETED
    assert received == list(activities)


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


def test_mock_retries_schema_miss_with_the_validating_error(
    tmp_path: Path,
) -> None:
    adapter = MockAdapter()
    adapter.queue(MockResponse(payload={"count": -1}))
    adapter.queue(MockResponse(payload={"status": "completed"}))
    spec = _spec(tmp_path)

    result = adapter.run(spec)

    assert result.status == AgentStatus.COMPLETED
    assert result.payload == {"status": "completed"}
    assert len(adapter.calls) == 2
    assert adapter.calls[0].user_prompt == spec.user_prompt
    assert adapter.calls[1].user_prompt.startswith(spec.user_prompt)
    assert "missing required property 'status'" in adapter.calls[1].user_prompt


def test_mock_bills_every_attempt_of_a_schema_retried_turn(tmp_path: Path) -> None:
    adapter = MockAdapter()
    adapter.queue(
        MockResponse(
            payload={"count": -1},
            usage=AgentUsage(tokens_input=2, tokens_output=3),
        )
    )
    adapter.queue(
        MockResponse(
            payload={"status": "completed"},
            usage=AgentUsage(tokens_input=5, tokens_output=7),
        )
    )

    result = adapter.run(_spec(tmp_path))

    # The real adapters combine usage across attempts, so a turn that was
    # retried costs both attempts here too.
    assert result.status == AgentStatus.COMPLETED
    assert result.usage is not None
    assert result.usage.tokens_input == 7
    assert result.usage.tokens_output == 10


def test_mock_schema_miss_fails_after_bounded_attempts(tmp_path: Path) -> None:
    adapter = MockAdapter()
    for attempt in range(DEFAULT_SCHEMA_MAX_ATTEMPTS + 1):
        adapter.queue(
            MockResponse(
                payload={"count": -1},
                usage=AgentUsage(tokens_input=attempt + 1),
            )
        )

    result = adapter.run(_spec(tmp_path))

    assert result.status == AgentStatus.FAILED
    assert "missing required property 'status'" in (result.error or "")
    assert len(adapter.calls) == DEFAULT_SCHEMA_MAX_ATTEMPTS
    assert len(adapter.responses) == 1
    # A turn that never conforms is still billed for every attempt it made.
    assert result.usage is not None
    assert result.usage.tokens_input == sum(
        range(1, DEFAULT_SCHEMA_MAX_ATTEMPTS + 1)
    )


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


def test_cancellation_token_serializes_work_start_against_cancellation() -> None:
    cancel = CancellationToken()
    transition_started = threading.Event()
    release_transition = threading.Event()
    transition_finished = threading.Event()
    cancellation_started = threading.Event()
    cancellation_finished = threading.Event()

    def transition() -> None:
        transition_started.set()
        assert release_transition.wait(1)

    def start_work() -> None:
        assert cancel.start_if_active(transition)
        transition_finished.set()

    def request_cancellation() -> None:
        cancellation_started.set()
        cancel.cancel()
        cancellation_finished.set()

    starter = threading.Thread(target=start_work)
    starter.start()
    assert transition_started.wait(1)
    canceller = threading.Thread(target=request_cancellation)
    canceller.start()
    assert cancellation_started.wait(1)
    assert not cancellation_finished.wait(0.05)

    release_transition.set()
    starter.join(timeout=1)
    canceller.join(timeout=1)

    assert transition_finished.is_set()
    assert cancellation_finished.is_set()
    assert not cancel.start_if_active(lambda: pytest.fail("work started after cancel"))


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


def test_cancelled_late_registration_dispatches_cancel_before_force(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancel = CancellationToken()
    cancel.cancel()
    cancel_claimed = threading.Event()
    release_cancel = threading.Event()
    force_returned = threading.Event()
    delivered: list[str] = []
    registrations = []
    registration_errors: list[Exception] = []
    force_errors: list[Exception] = []
    original_claim_cancel = cancel._claim_cancel

    def claim_cancel_with_gate(entry: Any) -> Callable[[], None] | None:
        callback = original_claim_cancel(entry)
        if callback is None:
            return None

        def gated_callback() -> None:
            cancel_claimed.set()
            release_cancel.wait()
            callback()

        return gated_callback

    monkeypatch.setattr(cancel, "_claim_cancel", claim_cancel_with_gate)

    def register_late() -> None:
        try:
            registrations.append(
                cancel.register(
                    lambda: delivered.append("cancel"),
                    lambda: delivered.append("force"),
                    force_target=ForceTarget("late-worker"),
                )
            )
        except Exception as error:
            registration_errors.append(error)

    def force_late() -> None:
        try:
            cancel.force()
        except Exception as error:
            force_errors.append(error)
        finally:
            force_returned.set()

    registration_thread = threading.Thread(target=register_late, daemon=True)
    registration_thread.start()
    assert cancel_claimed.wait(1)

    force_thread = threading.Thread(target=force_late, daemon=True)
    force_thread.start()
    try:
        assert cancel.wait_for_force(1)
        assert not force_returned.wait(0.05)
        assert delivered == []
    finally:
        release_cancel.set()
    registration_thread.join(timeout=1)
    force_thread.join(timeout=1)

    assert not registration_thread.is_alive()
    assert not force_thread.is_alive()
    assert force_returned.is_set()
    assert registration_errors == []
    assert force_errors == []
    assert delivered == ["cancel", "force"]
    assert registrations[0].unregister()


def test_late_cancel_callback_failure_propagates() -> None:
    cancel = CancellationToken()
    cancel.cancel()

    def fail_cancel() -> None:
        raise RuntimeError("cancel delivery failed")

    with pytest.raises(
        CancellationDeliveryError, match="cancel delivery failed"
    ) as raised:
        cancel.register(fail_cancel)

    assert raised.value.errors == (raised.value.__cause__,)
    assert raised.value.registration.unregister()


def test_late_forced_registration_attempts_force_after_cancel_failure() -> None:
    cancel = CancellationToken()
    cancel.force()
    force_delivered = threading.Event()

    def fail_cancel() -> None:
        raise RuntimeError("cancel delivery failed")

    with pytest.raises(
        CancellationDeliveryError, match="cancel delivery failed"
    ) as raised:
        cancel.register(
            fail_cancel,
            force_delivered.set,
            force_target=ForceTarget("late-worker"),
        )

    assert force_delivered.is_set()
    assert cancel.force_targets == (ForceTarget("late-worker"),)
    assert raised.value.registration.unregister()
    assert cancel.force_targets == ()


def test_force_callback_failure_propagates_and_remains_failed() -> None:
    cancel = CancellationToken()

    def fail_force() -> None:
        raise RuntimeError("force delivery failed")

    registration = cancel.register(
        lambda: None,
        fail_force,
        force_target=ForceTarget("worker"),
    )
    cancel.cancel()

    with pytest.raises(RuntimeError, match="force delivery failed"):
        cancel.force()
    with pytest.raises(RuntimeError, match="force delivery failed"):
        cancel.force()

    assert cancel.force_targets == (ForceTarget("worker"),)
    assert registration.unregister()


def test_failed_force_delivery_keeps_registration_window_active() -> None:
    cancel = CancellationToken()
    window = cancel.registration_window()
    window.resource_created()
    cancel.force()

    def fail_force() -> None:
        raise RuntimeError("created resource was not forced")

    with pytest.raises(
        CancellationDeliveryError, match="created resource was not forced"
    ) as raised:
        window.register(
            lambda: None,
            fail_force,
            force_target=ForceTarget("created-worker"),
        )

    assert not window.is_settled
    assert cancel.active_windows == (window,)
    assert cancel.force_targets == (ForceTarget("created-worker"),)

    assert raised.value.registration.unregister()
    assert window.is_settled
    assert cancel.active_windows == ()
    assert cancel.force_targets == ()


def test_failed_late_cancel_settles_window_after_successful_force() -> None:
    cancel = CancellationToken()
    window = cancel.registration_window()
    window.resource_created()
    cancel.force()
    force_delivered = threading.Event()

    def fail_cancel() -> None:
        raise RuntimeError("cancel delivery failed")

    with pytest.raises(
        CancellationDeliveryError, match="cancel delivery failed"
    ) as raised:
        window.register(
            fail_cancel,
            force_delivered.set,
            force_target=ForceTarget("created-worker"),
        )

    assert force_delivered.is_set()
    assert window.is_settled
    assert cancel.active_windows == ()
    assert cancel.force_targets == (ForceTarget("created-worker"),)
    assert raised.value.registration.unregister()
    assert cancel.force_targets == ()


def test_registration_window_retains_created_resource_until_late_force() -> None:
    cancel = CancellationToken()
    window = cancel.registration_window()
    snapshot = cancel.active_windows
    created = threading.Event()
    publish = threading.Event()
    forced = threading.Event()
    registrations = []

    def create_and_publish() -> None:
        window.resource_created()
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
    window.resource_created()

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


def test_registration_window_rejects_no_resource_after_creation() -> None:
    cancel = CancellationToken()
    window = cancel.registration_window()

    with pytest.raises(RuntimeError, match="record resource creation"):
        window.register(
            lambda: None,
            lambda: None,
            force_target=ForceTarget("created-worker"),
        )
    window.resource_created()

    with pytest.raises(RuntimeError, match="created resource must be published"):
        window.no_resource()
    with pytest.raises(RuntimeError, match="already recorded"):
        window.resource_created()
    assert cancel.active_windows == (window,)

    registration = window.register(
        lambda: None,
        lambda: None,
        force_target=ForceTarget("created-worker"),
    )
    assert window.is_settled
    assert registration.unregister()


def test_registration_window_publishes_cleaned_resource_after_failure() -> None:
    cancel = CancellationToken()
    window = cancel.registration_window()
    window.resource_created()
    cancel.force()
    delivered: list[str] = []

    window.publish_cleaned_resource(
        lambda: delivered.append("cancel"),
        lambda: delivered.append("force"),
        force_target=ForceTarget("cleaned-worker"),
    )

    assert delivered == ["cancel", "force"]
    assert window.is_settled
    assert cancel.active_windows == ()
    assert cancel.force_targets == ()


def test_cancellation_token_unregister_before_cancel_prevents_delivery() -> None:
    cancel = CancellationToken()
    delivered: list[str] = []
    registration = cancel.register(lambda: delivered.append("cancel"))

    assert registration.unregister()
    cancel.cancel()

    assert delivered == []


def test_unregister_waits_for_claimed_force_delivery() -> None:
    cancel = CancellationToken()
    force_entered = threading.Event()
    release_force = threading.Event()
    unregister_returned = threading.Event()
    target = ForceTarget("worker")

    def on_force() -> None:
        force_entered.set()
        release_force.wait()

    registration = cancel.register(
        lambda: None,
        on_force,
        force_target=target,
    )
    force_thread = threading.Thread(target=cancel.force)
    force_thread.start()
    assert force_entered.wait(1)

    unregister_result: list[bool] = []

    def unregister() -> None:
        unregister_result.append(registration.unregister())
        unregister_returned.set()

    unregister_thread = threading.Thread(target=unregister)
    unregister_thread.start()
    try:
        assert not unregister_returned.wait(0.05)
        assert cancel.force_targets == (target,)
    finally:
        release_force.set()
    force_thread.join(timeout=1)
    unregister_thread.join(timeout=1)

    assert not force_thread.is_alive()
    assert not unregister_thread.is_alive()
    assert unregister_result == [True]
    assert cancel.force_targets == ()


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


@pytest.mark.parametrize("process_group_id", [True, 1.5, "1"])
def test_force_target_rejects_noninteger_process_group(
    process_group_id: object,
) -> None:
    with pytest.raises(TypeError, match="process group.*integer"):
        ForceTarget("worker", process_group_id=process_group_id)


@pytest.mark.parametrize("process_group_id", [0, -1])
def test_force_target_rejects_nonpositive_process_group(
    process_group_id: int,
) -> None:
    with pytest.raises(ValueError, match="process group"):
        ForceTarget("worker", process_group_id=process_group_id)


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
