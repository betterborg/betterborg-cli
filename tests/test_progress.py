"""Lifecycle tests for the shared command progress reporter."""

from __future__ import annotations

from io import StringIO

import pytest

from betterborg_cli.progress import (
    AgentActivity,
    ChildSpec,
    ProgressError,
    RunProgress,
    StageSpec,
    StageState,
)


class FakeClock:
    """A manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.mark.parametrize(
    ("method_name", "state", "result", "duration", "rendered_duration"),
    [
        ("seed_completed", StageState.COMPLETED, "cached", 12.5, "12.5s"),
        (
            "seed_completed",
            StageState.COMPLETED,
            "cached",
            None,
            "duration unknown",
        ),
        ("seed_failed", StageState.FAILED, "durable error", 4.0, "4.0s"),
        (
            "seed_failed",
            StageState.FAILED,
            "durable error",
            None,
            "duration unknown",
        ),
    ],
)
def test_retained_parent_seeding_preserves_authoritative_outcome(
    method_name: str,
    state: StageState,
    result: str,
    duration: float | None,
    rendered_duration: str,
) -> None:
    stream = StringIO()
    clock = FakeClock()
    progress = RunProgress(
        [StageSpec("analysis", "Repository analysis")],
        stream=stream,
        clock=clock,
    )

    record = getattr(progress, method_name)("analysis", result, duration)
    clock.advance(100)

    assert record.state is state
    assert record.result == result
    assert record.retained is True
    assert record.started_at is None
    assert record.finished_at is None
    assert record.duration_seconds == duration
    assert progress.elapsed("analysis") == duration
    assert progress.counts[state] == 1
    assert stream.getvalue().splitlines() == [
        f"{state.value} Repository analysis — {result} "
        f"({rendered_duration}) [retained]"
    ]


def test_fresh_parent_and_child_have_independent_frozen_timing() -> None:
    clock = FakeClock()
    progress = RunProgress(
        [StageSpec("plan", "Plan", (ChildSpec("prompt", "Prompt"),))],
        stream=StringIO(),
        clock=clock,
    )

    parent = progress.start("plan")
    clock.advance(2)
    child = progress.start_child("plan", "prompt")
    clock.advance(3)
    progress.update("plan", "checking")
    progress.update_child("plan", "prompt", "writing")
    progress.activity("plan", AgentActivity("review", "contracts"))
    progress.child_activity("plan", "prompt", "agent", "drafting")

    assert progress.elapsed("plan") == 5
    assert progress.child_elapsed("plan", "prompt") == 3
    assert parent.detail == "checking"
    assert child.detail == "writing"
    assert parent.activity == AgentActivity("review", "contracts")
    assert child.activity == AgentActivity("agent", "drafting")

    progress.complete_child("plan", "prompt", "accepted")
    clock.advance(4)
    progress.complete("plan", "approved")
    clock.advance(20)

    assert child.duration_seconds == 3
    assert parent.duration_seconds == 9
    assert progress.child_elapsed("plan", "prompt") == 3
    assert progress.elapsed("plan") == 9
    assert child.result == "accepted"
    assert parent.result == "approved"
    with pytest.raises(ProgressError, match="must be running"):
        progress.fail("plan", "replacement")
    assert parent.result == "approved"
    assert parent.duration_seconds == 9


def test_parent_seeding_rejects_invalid_or_touched_records() -> None:
    progress = RunProgress(
        [StageSpec("stage", "Stage", (ChildSpec("fixed", "Fixed"),))],
        stream=StringIO(),
    )
    with pytest.raises(ValueError, match="non-negative"):
        progress.seed_completed("stage", "bad", -0.1)

    progress.start("stage")
    progress.activity("stage", AgentActivity("agent"))
    with pytest.raises(ProgressError, match="must be pending"):
        progress.seed_completed("stage", "retained")

    terminal = RunProgress([StageSpec("stage", "Stage")], stream=StringIO())
    terminal.start("stage")
    terminal.complete("stage", "fresh")
    with pytest.raises(ProgressError, match="must be pending"):
        terminal.seed_failed("stage", "replacement")

    cancelled = RunProgress([StageSpec("stage", "Stage")], stream=StringIO())
    cancelled.begin_cancellation()
    with pytest.raises(ProgressError, match="after cancellation"):
        cancelled.seed_completed("stage", "retained")


@pytest.mark.parametrize("seed_method", ["seed_completed", "seed_failed"])
def test_seeded_parent_rejects_reseed_and_every_terminal_restart(
    seed_method: str,
) -> None:
    progress = RunProgress([StageSpec("stage", "Stage")], stream=StringIO())
    getattr(progress, seed_method)("stage", "durable", 1)

    for operation in (
        lambda: progress.start("stage"),
        lambda: progress.seed_completed("stage", "again"),
        lambda: progress.seed_failed("stage", "again"),
        lambda: progress.complete("stage", "again"),
        lambda: progress.fail("stage", "again"),
        lambda: progress.stop("stage", "again"),
    ):
        with pytest.raises(ProgressError):
            operation()


def test_parent_seed_and_finish_reject_running_child() -> None:
    fresh = RunProgress(
        [StageSpec("stage", "Stage", (ChildSpec("child", "Child"),))],
        stream=StringIO(),
    )
    fresh.start("stage")
    fresh.start_child("stage", "child")
    with pytest.raises(ProgressError, match="unresolved children"):
        fresh.complete("stage")

    retained = RunProgress(
        [StageSpec("stage", "Stage", (ChildSpec("child", "Child"),))],
        stream=StringIO(),
    )
    with pytest.raises(ProgressError, match="unresolved children"):
        retained.seed_completed("stage", "durable")


def test_fixed_and_dynamic_children_share_lifecycle_and_bound_rendering() -> None:
    clock = FakeClock()
    stream = StringIO()
    progress = RunProgress(
        [StageSpec("plan", "Plan", (ChildSpec("prompt", "Prompt"),))],
        stream=stream,
        clock=clock,
        attempt_history_limit=1,
    )
    progress.seed_child_completed("plan", "prompt", "reused", None)

    first_key = "revision:batch-7:attempt-1"
    second_key = "revision:batch-7:attempt-2"
    first = progress.declare_child("plan", ChildSpec(first_key, "Revision 1"))
    progress.seed_child_completed("plan", first_key, "rejected", 8)
    second = progress.declare_child("plan", ChildSpec(second_key, "Revision 2"))

    render_state = progress.child_render_state("plan")
    assert list(progress.stages["plan"].children) == [
        "prompt",
        first_key,
        second_key,
    ]
    assert [child.key for child in render_state.children] == ["prompt", second_key]
    assert render_state.earlier_attempt_count == 1
    assert first.retained is True
    assert first.started_at is None
    with pytest.raises(ProgressError, match="already declared"):
        progress.declare_child("plan", ChildSpec(first_key, "Duplicate"))

    progress.start("plan")
    with pytest.raises(ProgressError, match="must be pending"):
        progress.start_child("plan", first_key)
    progress.start_child("plan", second_key)
    clock.advance(2)
    progress.fail_child("plan", second_key, "needs changes")
    progress.fail("plan", "revision failed")

    assert second.state is StageState.FAILED
    assert second.duration_seconds == 2
    with pytest.raises(ProgressError, match="already terminal"):
        progress.declare_child("plan", ChildSpec("revision:3", "Revision 3"))
    with pytest.raises(ProgressError, match="already terminal"):
        progress.seed_child_completed("plan", second_key, "replacement")


@pytest.mark.parametrize("resolution", ["complete", "stop"])
def test_cancellation_is_nonterminal_until_authoritative_reconciliation(
    resolution: str,
) -> None:
    stream = StringIO()
    progress = RunProgress([StageSpec("work", "Work")], stream=stream)
    record = progress.start("work")

    assert progress.begin_cancellation() is True
    assert progress.begin_cancellation() is False
    assert progress.cancelling is True
    assert record.state is StageState.RUNNING
    assert stream.getvalue().splitlines() == ["stopping..."]
    with pytest.raises(ProgressError, match="after cancellation"):
        progress.declare(StageSpec("late", "Late"))
    with pytest.raises(ProgressError, match="after cancellation"):
        progress.start("work")

    getattr(progress, resolution)("work", "durable result")
    expected = (
        StageState.COMPLETED if resolution == "complete" else StageState.STOPPED
    )
    assert record.state is expected
    progress.close()
    assert progress.closed is True


def test_close_rejects_unresolved_started_parent_and_child() -> None:
    progress = RunProgress(
        [StageSpec("work", "Work", (ChildSpec("child", "Child"),))],
        stream=StringIO(),
    )
    progress.start("work")
    progress.start_child("work", "child")
    progress.begin_cancellation()

    with pytest.raises(ProgressError, match="unresolved started records") as error:
        progress.close()
    assert "stage 'work'" in str(error.value)
    assert "child 'work'/'child'" in str(error.value)
    assert progress.stages["work"].state is StageState.RUNNING

    progress.stop_child("work", "child", "cancelled")
    progress.stop("work", "cancelled")
    progress.close()


def test_nested_suspension_queues_one_time_lines_in_transition_order() -> None:
    stream = StringIO()
    progress = RunProgress(
        [StageSpec("one", "One"), StageSpec("two", "Two")], stream=stream
    )

    with progress.suspend():
        progress.seed_completed("one", "cached", 1)
        with progress.suspend():
            progress.seed_failed("two", "cached failure", None)
            progress.begin_cancellation()
        assert stream.getvalue() == ""
    assert stream.getvalue().splitlines() == [
        "completed One — cached (1.0s) [retained]",
        "failed Two — cached failure (duration unknown) [retained]",
        "stopping...",
    ]


def test_machine_readable_progress_emits_no_output() -> None:
    stream = StringIO()
    progress = RunProgress(
        [StageSpec("stage", "Stage")], stream=stream, machine_readable=True
    )
    progress.seed_completed("stage", "cached")
    progress.begin_cancellation()

    assert stream.getvalue() == ""
