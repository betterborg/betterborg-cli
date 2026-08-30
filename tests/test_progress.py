"""Lifecycle tests for the shared command progress reporter."""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
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


class TTYStringIO(StringIO):
    """An in-memory stream that exercises Rich's interactive renderer."""

    def isatty(self) -> bool:
        return True


def _terminal_text(value: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value).replace("\r", "")


@pytest.mark.parametrize(
    ("method_name", "state", "result", "duration", "rendered_duration"),
    [
        ("seed_completed", StageState.COMPLETED, "cached", 12.5, "12.5s"),
        (
            "seed_completed",
            StageState.COMPLETED,
            "cached",
            None,
            "—",
        ),
        ("seed_failed", StageState.FAILED, "durable error", 4.0, "4.0s"),
        (
            "seed_failed",
            StageState.FAILED,
            "durable error",
            None,
            "—",
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
    progress.refresh()

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
    assert stream.getvalue().splitlines() == [
        "running Work (0.0s)",
        "stopping...",
    ]
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
        "failed Two — cached failure (—) [retained]",
        "stopping...",
    ]


def test_machine_readable_progress_emits_no_output() -> None:
    stream = StringIO()
    progress = RunProgress(
        [StageSpec("stage", "Stage")], stream=stream, machine_readable=True
    )
    progress.seed_completed("stage", "cached")
    progress.begin_cancellation()
    progress.close()

    assert stream.getvalue() == ""


def test_plain_heartbeats_cover_only_fresh_running_work() -> None:
    stream = StringIO()
    clock = FakeClock()
    progress = RunProgress(
        [StageSpec("fresh", "Fresh"), StageSpec("retained", "Retained")],
        stream=stream,
        clock=clock,
        heartbeat_interval=5,
    )

    progress.start("fresh")
    progress.seed_completed("retained", "cached", 9)
    clock.advance(4)
    progress.refresh()
    clock.advance(1)
    progress.refresh()
    progress.complete("fresh", "built")

    assert stream.getvalue().splitlines() == [
        "running Fresh (0.0s)",
        "completed Retained — cached (9.0s) [retained]",
        "running Fresh (5.0s)",
        "completed Fresh — built (5.0s)",
    ]


def test_plain_starts_emit_only_new_row_without_postponing_heartbeat() -> None:
    stream = StringIO()
    clock = FakeClock()
    progress = RunProgress(
        [
            StageSpec("one", "One", children=(ChildSpec("child", "Child"),)),
            StageSpec("two", "Two"),
        ],
        stream=stream,
        clock=clock,
        heartbeat_interval=5,
    )

    progress.start("one")
    clock.advance(1)
    progress.start("two")
    clock.advance(1)
    progress.start_child("one", "child")

    assert stream.getvalue().splitlines() == [
        "running One (0.0s)",
        "running Two (0.0s)",
        "running One: Child (0.0s)",
    ]

    clock.advance(2.9)
    progress.refresh()
    assert len(stream.getvalue().splitlines()) == 3
    clock.advance(0.1)
    progress.refresh()

    assert stream.getvalue().splitlines()[3:] == [
        "running One (5.0s)",
        "running One: Child (3.0s)",
        "running Two (4.0s)",
    ]


def test_plain_suspension_skips_stale_heartbeat_and_flushes_permanent_lines() -> None:
    stream = StringIO()
    clock = FakeClock()
    progress = RunProgress(
        [StageSpec("fresh", "Fresh"), StageSpec("retained", "Retained")],
        stream=stream,
        clock=clock,
        heartbeat_interval=5,
    )
    progress.start("fresh")
    stream.seek(0)
    stream.truncate()

    with progress.suspend():
        clock.advance(10)
        progress.refresh()
        progress.seed_failed("retained", "cached failure")
        assert stream.getvalue() == ""

    assert stream.getvalue().splitlines() == [
        "failed Retained — cached failure (—) [retained]"
    ]
    clock.advance(4)
    progress.refresh()
    assert len(stream.getvalue().splitlines()) == 1
    clock.advance(1)
    progress.refresh()
    assert stream.getvalue().splitlines()[-1] == "running Fresh (15.0s)"


def test_permanent_lines_match_in_plain_and_rich_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TERM", raising=False)
    plain = StringIO()
    rich = TTYStringIO()

    for stream in (plain, rich):
        progress = RunProgress(
            [StageSpec("stage", "Stage")], stream=stream, width=120
        )
        progress.seed_completed("stage", "cached", 2.5)

    expected = "completed Stage — cached (2.5s) [retained]"
    assert plain.getvalue().strip() == expected
    assert _terminal_text(rich.getvalue()).strip() == expected


@pytest.mark.parametrize(
    ("environment", "interactive"),
    [({}, True), ({"NO_COLOR": "1"}, True), ({"TERM": "dumb"}, False)],
)
def test_nested_suspension_crosses_heartbeat_and_orders_concurrent_lines(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    interactive: bool,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    stream = TTYStringIO()
    clock = FakeClock()
    progress = RunProgress(
        [
            StageSpec("active", "Active"),
            StageSpec("one", "One"),
            StageSpec("two", "Two"),
            StageSpec("cached-one", "Cached one"),
            StageSpec("cached-two", "Cached two"),
        ],
        stream=stream,
        clock=clock,
        width=120,
        heartbeat_interval=5,
    )
    progress.start("active")
    progress.start("one")
    progress.start("two")
    release_one = threading.Event()
    release_two = threading.Event()
    first_done = threading.Event()

    def finish_one() -> None:
        release_one.wait()
        progress.complete("one", "first")
        first_done.set()

    def finish_two() -> None:
        release_two.wait()
        progress.fail("two", "second")

    with progress.suspend():
        stream.seek(0)
        stream.truncate()
        progress.seed_completed("cached-one", "reused", 3)
        clock.advance(10)
        progress.refresh()
        with progress.suspend():
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(finish_one), executor.submit(finish_two)]
                release_one.set()
                assert first_done.wait(timeout=2)
                release_two.set()
                for future in futures:
                    future.result(timeout=2)
            progress.seed_failed("cached-two", "cached failure")
            assert stream.getvalue() == ""
        assert stream.getvalue() == ""

    permanent_lines = [
        "completed Cached one — reused (3.0s) [retained]",
        "completed One — first (10.0s)",
        "failed Two — second (10.0s)",
        "failed Cached two — cached failure (—) [retained]",
    ]
    resumed = _terminal_text(stream.getvalue()) if interactive else stream.getvalue()
    assert all(resumed.count(line) == 1 for line in permanent_lines)
    assert [resumed.index(line) for line in permanent_lines] == sorted(
        resumed.index(line) for line in permanent_lines
    )

    clock.advance(4)
    progress.refresh()
    if not interactive:
        assert stream.getvalue() == resumed
        clock.advance(1)
        progress.refresh()
        assert stream.getvalue().splitlines()[-1] == "running Active (15.0s)"
    else:
        refreshed = _terminal_text(stream.getvalue())
        assert all(refreshed.count(line) == 1 for line in permanent_lines)

    progress.complete("active", "third")


def test_rich_live_output_uses_bounded_dynamic_child_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    stream = TTYStringIO()
    progress = RunProgress(
        [StageSpec("plan", "Plan")],
        stream=stream,
        width=120,
        attempt_history_limit=2,
    )
    for number in range(1, 4):
        key = f"attempt-{number}"
        progress.declare_child("plan", ChildSpec(key, f"Attempt {number}"))
        progress.seed_child_completed("plan", key, "retry")
    progress.declare_child("plan", ChildSpec("attempt-4", "Attempt 4"))
    stream.seek(0)
    stream.truncate()

    progress.start("plan")
    output = _terminal_text(stream.getvalue())

    assert "completed Plan: Attempt 1" not in output
    assert "completed Plan: Attempt 2" not in output
    assert "completed Plan: Attempt 3 — retry (—) [retained]" in output
    assert "pending Plan: Attempt 4" in output
    assert "Plan: … 2 earlier attempts" in output


def test_rich_live_output_is_bounded_for_many_running_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TERM", raising=False)
    stream = TTYStringIO()
    progress = RunProgress(
        [StageSpec(f"stage-{number}", f"Stage {number}") for number in range(12)],
        stream=stream,
        width=120,
    )

    with progress.suspend():
        for number in range(12):
            progress.start(f"stage-{number}")
    initial_frame = _terminal_text(stream.getvalue())

    assert all(f"running Stage {number}" in initial_frame for number in range(7))
    assert "running Stage 7" not in initial_frame
    assert "… 5 more running" in initial_frame


def test_width_truncation_and_run_summary_are_canonical() -> None:
    stream = StringIO()
    progress = RunProgress(
        [StageSpec("cached", "A very long retained stage")],
        stream=stream,
        width=24,
    )
    progress.seed_completed("cached", "a long durable result", 1)
    progress.close()

    lines = stream.getvalue().splitlines()
    assert all(len(line) <= 24 for line in lines)
    assert all(line.endswith("…") for line in lines)

    summary_stream = StringIO()
    summary = RunProgress(
        [StageSpec("done", "Done"), StageSpec("failed", "Failed")],
        stream=summary_stream,
    )
    summary.seed_completed("done", "cached")
    summary.seed_failed("failed", "cached failure")
    summary.close()
    assert summary_stream.getvalue().splitlines()[-1] == (
        "summary: 1 completed, 1 failed, 0 stopped — 2 retained"
    )


def test_disabled_mode_suppresses_transient_permanent_and_summary_output() -> None:
    stream = TTYStringIO()
    clock = FakeClock()
    progress = RunProgress(
        [StageSpec("stage", "Stage")],
        stream=stream,
        clock=clock,
        enabled=False,
        heartbeat_interval=1,
    )

    progress.start("stage")
    clock.advance(2)
    progress.refresh()
    progress.complete("stage", "done")
    progress.close()

    assert stream.getvalue() == ""


def test_concurrent_permanent_lines_follow_lock_transition_order() -> None:
    stream = StringIO()
    progress = RunProgress(
        [StageSpec("one", "One"), StageSpec("two", "Two")], stream=stream
    )
    progress.start("one")
    progress.start("two")
    stream.seek(0)
    stream.truncate()
    release_one = threading.Event()
    release_two = threading.Event()
    first_done = threading.Event()

    def finish_one() -> None:
        release_one.wait()
        progress.complete("one", "first")
        first_done.set()

    def finish_two() -> None:
        release_two.wait()
        progress.fail("two", "second")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(finish_one), executor.submit(finish_two)]
        release_one.set()
        assert first_done.wait(timeout=2)
        release_two.set()
        for future in futures:
            future.result(timeout=2)

    assert stream.getvalue().splitlines() == [
        "completed One — first (0.0s)",
        "failed Two — second (0.0s)",
    ]
