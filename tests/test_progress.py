"""Lifecycle tests for the shared command progress reporter."""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from typing import Any

import pytest
from progress_test_support import FakeClock, TTYStringIO
from rich.cells import cell_len

from betterborg_cli.progress import (
    AgentActivity,
    AgentActivityKind,
    ChildSpec,
    ProgressError,
    RunProgress,
    StageSpec,
    StageState,
    _format_duration,
)


def _terminal_text(value: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value).replace("\r", "")


@pytest.mark.parametrize(
    ("method_name", "state", "glyph", "result", "duration", "rendered_duration"),
    [
        ("seed_completed", StageState.COMPLETED, "✔", "cached", 12.5, "0:12"),
        (
            "seed_completed",
            StageState.COMPLETED,
            "✔",
            "cached",
            None,
            "—",
        ),
        ("seed_failed", StageState.FAILED, "✖", "durable error", 4.0, "0:04"),
        (
            "seed_failed",
            StageState.FAILED,
            "✖",
            "durable error",
            None,
            "—",
        ),
    ],
)
def test_retained_parent_seeding_preserves_authoritative_outcome(
    method_name: str,
    state: StageState,
    glyph: str,
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
        f"{glyph} Repository analysis    {rendered_duration}  {result} "
        "· reused from earlier run"
    ]
    assert "[retained]" not in stream.getvalue()


@pytest.mark.parametrize(
    ("seconds", "rendered"),
    [
        (0.0, "0:00"),
        (59.0, "0:59"),
        (60.0, "1:00"),
        (3599.0, "59:59"),
        (3600.0, "1:00:00"),
        (27_845.75, "7:44:05"),
    ],
)
def test_duration_boundaries_preserve_source_precision(
    seconds: float, rendered: str
) -> None:
    stream = StringIO()
    progress = RunProgress([StageSpec("stage", "Stage")], stream=stream)

    record = progress.seed_completed("stage", "result", seconds)

    assert _format_duration(seconds) == rendered
    assert f"  {rendered}  result" in stream.getvalue()
    assert record.duration_seconds == seconds


def test_completed_row_uses_exact_canonical_alignment() -> None:
    stream = StringIO()
    clock = FakeClock()
    progress = RunProgress(
        [StageSpec("analysis", "Analyze repository")], stream=stream, clock=clock
    )
    progress.start("analysis")
    stream.seek(0)
    stream.truncate()
    clock.advance(134)

    progress.complete("analysis", "claude · opus-4-8")

    assert stream.getvalue().strip() == (
        "✔ Analyze repository     2:14  claude · opus-4-8"
    )
    assert "completed" not in stream.getvalue()


def test_terminal_state_glyphs_and_rich_styles_remain_distinct_without_colour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    raw_outputs: dict[str, str] = {}

    for resolution in ("complete", "fail", "stop"):
        stream = TTYStringIO()
        progress = RunProgress([StageSpec("stage", "Stage")], stream=stream)
        progress.start("stage")
        getattr(progress, resolution)("stage", "result")
        raw_outputs[resolution] = stream.getvalue()

    assert "\x1b[32m✔" in raw_outputs["complete"]
    assert "\x1b[31m✖" in raw_outputs["fail"]
    assert "\x1b[2;36m■" in raw_outputs["stop"]
    assert "\x1b[31m■" not in raw_outputs["stop"]
    assert "\x1b[2;31m0:00" in raw_outputs["fail"]

    pending_stream = TTYStringIO()
    pending = RunProgress(
        [StageSpec("stage", "Stage", (ChildSpec("child", "Child"),))],
        stream=pending_stream,
    )
    pending.start("stage")
    assert "\x1b[2m◦ Stage: Child" in pending_stream.getvalue()

    stripped = {
        resolution: _terminal_text(output)
        for resolution, output in raw_outputs.items()
    }
    assert "✔ Stage" in stripped["complete"]
    assert "✖ Stage" in stripped["fail"]
    assert "■ Stage" in stripped["stop"]


@pytest.mark.parametrize(
    ("kind", "detail", "expected"),
    [
        (AgentActivityKind.THINKING, None, "thinking"),
        (
            AgentActivityKind.READING,
            "src/webhook/retry.go",
            "reading src/webhook/retry.go",
        ),
        (
            AgentActivityKind.SEARCHING,
            "docker-compose",
            'searching "docker-compose"',
        ),
        (AgentActivityKind.COMMAND, "make test", "running make test"),
        (AgentActivityKind.WRITING, "result.json", "writing result.json"),
    ],
)
def test_activity_kinds_use_product_language(
    kind: AgentActivityKind, detail: str | None, expected: str
) -> None:
    stream = StringIO()
    clock = FakeClock()
    progress = RunProgress(
        [StageSpec("stage", "Stage")],
        stream=stream,
        clock=clock,
        heartbeat_interval=1,
    )
    progress.start("stage")
    stream.seek(0)
    stream.truncate()
    progress.activity("stage", AgentActivity(kind, detail))
    clock.advance(1)

    progress.refresh()

    assert stream.getvalue().splitlines() == [
        f"⠋ Stage                  0:01  {expected}"
    ]
    assert f"{kind.value}:" not in stream.getvalue()


def test_empty_parent_and_child_activity_falls_back_to_thinking() -> None:
    stream = StringIO()
    progress = RunProgress(
        [StageSpec("stage", "Parent", (ChildSpec("child", "Child"),))],
        stream=stream,
    )

    progress.start("stage")
    progress.start_child("stage", "child")

    assert stream.getvalue().splitlines() == [
        "⠋ Parent                 0:00  thinking",
        "⠋ Parent: Child          0:00  thinking",
    ]


def test_dynamic_cells_normalize_controls_before_layout_and_truncation() -> None:
    unsafe = "left\r\nright\rmid\nnext\ttab\x1b[31mred\x00nul\x85c1"
    normalized = "left right mid next tab [31mred nul c1"
    stream = StringIO()
    clock = FakeClock()
    progress = RunProgress(
        [StageSpec("stage", unsafe, (ChildSpec("child", unsafe),))],
        stream=stream,
        clock=clock,
        heartbeat_interval=1,
    )
    progress.start("stage")
    progress.start_child("stage", "child")
    stream.seek(0)
    stream.truncate()

    progress.update("stage", unsafe)
    progress.update_child("stage", "child", unsafe)
    clock.advance(1)
    progress.refresh()

    update_lines = stream.getvalue().splitlines()
    assert len(update_lines) == 2
    assert all(normalized in line for line in update_lines)

    stream.seek(0)
    stream.truncate()
    progress.activity("stage", AgentActivity(AgentActivityKind.READING, unsafe))
    progress.child_activity(
        "stage", "child", AgentActivity(AgentActivityKind.COMMAND, unsafe)
    )
    clock.advance(1)
    progress.refresh()

    activity_lines = stream.getvalue().splitlines()
    assert len(activity_lines) == 2
    assert all(normalized in line for line in activity_lines)

    class UnsafeResult:
        def __str__(self) -> str:
            return unsafe

    stream.seek(0)
    stream.truncate()
    progress.complete_child("stage", "child", UnsafeResult())
    progress.complete("stage", UnsafeResult())

    result_lines = stream.getvalue().splitlines()
    assert len(result_lines) == 2
    assert all(normalized in line for line in result_lines)
    assert all(not re.search(r"[\x00-\x1f\x7f-\x9f]", line) for line in result_lines)


def test_control_normalization_cannot_expand_the_live_region() -> None:
    unsafe = "stage\r\nchild\t\x1b\x85"
    progress = RunProgress(
        [StageSpec(f"stage-{number}", f"{unsafe}{number}") for number in range(12)],
        stream=TTYStringIO(),
        width=80,
    )

    with progress.suspend():
        for number in range(12):
            progress.start(f"stage-{number}")

    live_lines = progress._live_lines()
    assert len(live_lines) == 8
    assert all(len(line.plain.splitlines()) == 1 for line in live_lines)
    assert all(
        not re.search(r"[\x00-\x1f\x7f-\x9f]", line.plain) for line in live_lines
    )


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

    assert progress.elapsed("plan") == 5
    assert progress.child_elapsed("plan", "prompt") == 3
    assert parent.detail == "checking"
    assert child.detail == "writing"

    parent_activity = AgentActivity(AgentActivityKind.SEARCHING, "contracts")
    child_activity = AgentActivity(AgentActivityKind.WRITING, "drafting")
    progress.activity("plan", parent_activity)
    progress.child_activity("plan", "prompt", child_activity)

    assert parent.detail == "checking"
    assert child.detail == "writing"
    assert parent.activity == parent_activity
    assert child.activity == child_activity

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
    progress.activity("stage", AgentActivity(AgentActivityKind.THINKING))
    with pytest.raises(ProgressError, match="must be pending"):
        progress.seed_completed("stage", "retained")

    terminal = RunProgress([StageSpec("stage", "Stage")], stream=StringIO())
    terminal.start("stage")
    terminal.complete("stage", "fresh")
    with pytest.raises(ProgressError, match="must be pending"):
        terminal.seed_failed("stage", "replacement")



def test_parent_seeding_allows_durable_reconciliation_during_cancellation() -> None:
    progress = RunProgress([StageSpec("stage", "Stage")], stream=StringIO())
    progress.begin_cancellation()

    retained = progress.seed_completed("stage", "retained")

    assert retained.state is StageState.COMPLETED
    assert retained.retained is True


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
        "⠋ Work                   0:00  thinking",
        "stopping…",
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


def test_stopped_parent_preserves_children_not_started_before_cancellation() -> None:
    progress = RunProgress(
        [StageSpec("work", "Work", (ChildSpec("child", "Child"),))],
        stream=StringIO(),
    )
    progress.start("work")
    progress.begin_cancellation()

    progress.stop("work", "cancelled")
    progress.close()

    assert progress.stages["work"].state is StageState.STOPPED
    assert progress.stages["work"].children["child"].state is StageState.PENDING


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
        "✔ One                    0:01  cached · reused from earlier run",
        "✖ Two                    —  cached failure · reused from earlier run",
        "stopping…",
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
        "⠋ Fresh                  0:00  thinking",
        "✔ Retained               0:09  cached · reused from earlier run",
        "⠋ Fresh                  0:05  thinking",
        "✔ Fresh                  0:05  built",
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
        "⠋ One                    0:00  thinking",
        "⠋ Two                    0:00  thinking",
        "⠋ One: Child             0:00  thinking",
    ]

    clock.advance(2.9)
    progress.refresh()
    assert len(stream.getvalue().splitlines()) == 3
    clock.advance(0.1)
    progress.refresh()

    assert stream.getvalue().splitlines()[3:] == [
        "⠋ One                    0:05  thinking",
        "⠋ One: Child             0:03  thinking",
        "⠋ Two                    0:04  thinking",
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
        "✖ Retained               —  cached failure · reused from earlier run"
    ]
    clock.advance(4)
    progress.refresh()
    assert len(stream.getvalue().splitlines()) == 1
    clock.advance(1)
    progress.refresh()
    assert stream.getvalue().splitlines()[-1] == (
        "⠋ Fresh                  0:15  thinking"
    )


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

    expected = "✔ Stage                  0:02  cached · reused from earlier run"
    assert plain.getvalue().strip() == expected
    assert _terminal_text(rich.getvalue()).strip() == expected


@pytest.mark.parametrize(
    ("label", "width", "expected"),
    [
        ("A" * 100, None, None),
        ("界" * 10, 12, "✔ 界界界界…"),
    ],
)
def test_long_permanent_lines_remain_one_canonical_terminal_line(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    width: int | None,
    expected: str | None,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    plain = StringIO()
    rich = TTYStringIO()

    for stream in (plain, rich):
        progress = RunProgress(
            [StageSpec("stage", label)], stream=stream, width=width
        )
        progress.seed_completed("stage", "cached", 2.5)

    plain_lines = plain.getvalue().splitlines()
    rich_lines = _terminal_text(rich.getvalue()).splitlines()
    assert rich_lines == plain_lines
    assert len(rich_lines) == 1
    if expected is not None:
        assert rich_lines == [expected]
    if width is not None:
        assert cell_len(rich_lines[0]) <= width


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
        "✔ Cached one             0:03  reused · reused from earlier run",
        "✔ One                    0:10  first",
        "✖ Two                    0:10  second",
        "✖ Cached two             —  cached failure · reused from earlier run",
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
        assert stream.getvalue().splitlines()[-1] == (
            "⠋ Active                 0:15  thinking"
        )
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

    assert "✔ Plan: Attempt 1" not in output
    assert "✔ Plan: Attempt 2" not in output
    assert "✔ Plan: Attempt 3        —  retry · reused from earlier run" in output
    assert "◦ Plan: Attempt 4" in output
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

    assert all(f"Stage {number}" in initial_frame for number in range(7))
    assert "Stage 7" not in initial_frame
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
        "✔ One                    0:00  first",
        "✖ Two                    0:00  second",
    ]


@pytest.mark.parametrize(
    ("kind", "detail", "work", "truncated"),
    [
        (AgentActivityKind.THINKING, None, "thinking", False),
        (
            AgentActivityKind.READING,
            "src/betterborg_cli/a/deeply/nested/provider_adapter.py",
            "reading src/betterborg_cli/a/deeply/nested/provider_adapter.py",
            True,
        ),
        (
            AgentActivityKind.SEARCHING,
            "every occurrence of the long-lived\nexecution attempt identifier",
            "searching \"every occurrence of the long-lived "
            "execution attempt identifier\"",
            True,
        ),
        (
            AgentActivityKind.COMMAND,
            "pytest tests/test_progress.py",
            "running pytest tests/test_progress.py",
            False,
        ),
        (
            AgentActivityKind.WRITING,
            "src/betterborg_cli/progress.py",
            "writing src/betterborg_cli/progress.py",
            False,
        ),
    ],
)
def test_parent_and_child_neutral_activity_render_as_one_capped_line(
    kind: AgentActivityKind,
    detail: str | None,
    work: str,
    truncated: bool,
) -> None:
    stream = StringIO()
    clock = FakeClock()
    width = 72
    progress = RunProgress(
        [StageSpec("stage", "Parent", (ChildSpec("child", "Child"),))],
        stream=stream,
        clock=clock,
        width=width,
        heartbeat_interval=1,
    )
    progress.start("stage")
    progress.start_child("stage", "child")
    stream.seek(0)
    stream.truncate()

    activity = AgentActivity(kind, detail)
    progress.activity("stage", activity)
    progress.child_activity("stage", "child", activity)
    clock.advance(1)
    progress.refresh()

    lines = stream.getvalue().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("⠋ Parent                 0:01  ")
    assert lines[1].startswith("⠋ Parent: Child          0:01  ")
    if truncated:
        assert all(line.endswith("…") for line in lines)
    else:
        assert all(line.endswith(work) for line in lines)
    assert all(cell_len(line) <= width for line in lines)


def test_latest_detail_or_activity_replaces_the_rendered_parent_and_child_state(
) -> None:
    stream = StringIO()
    clock = FakeClock()
    progress = RunProgress(
        [StageSpec("stage", "Stage", (ChildSpec("child", "Child"),))],
        stream=stream,
        clock=clock,
        heartbeat_interval=1,
    )
    parent = progress.start("stage")
    child = progress.start_child("stage", "child")
    stream.seek(0)
    stream.truncate()

    progress.activity(
        "stage", AgentActivity(AgentActivityKind.READING, "old-parent.py")
    )
    progress.child_activity(
        "stage",
        "child",
        AgentActivity(AgentActivityKind.SEARCHING, "old child detail"),
    )
    progress.update("stage", "new parent detail")
    progress.update_child("stage", "child", "new child detail")
    clock.advance(1)
    progress.refresh()

    assert parent.detail == "new parent detail"
    assert parent.activity == AgentActivity(
        AgentActivityKind.READING, "old-parent.py"
    )
    assert child.detail == "new child detail"
    assert child.activity == AgentActivity(
        AgentActivityKind.SEARCHING, "old child detail"
    )
    detail_lines = stream.getvalue().splitlines()
    assert detail_lines[-2].endswith("  new parent detail")
    assert detail_lines[-1].endswith("  new child detail")

    progress.activity("stage", AgentActivity(AgentActivityKind.THINKING))
    progress.child_activity(
        "stage", "child", AgentActivity(AgentActivityKind.WRITING, "answer.json")
    )

    clock.advance(1)
    progress.refresh()

    assert parent.detail == "new parent detail"
    assert parent.activity == AgentActivity(AgentActivityKind.THINKING)
    assert child.detail == "new child detail"
    assert child.activity == AgentActivity(
        AgentActivityKind.WRITING, "answer.json"
    )
    activity_lines = stream.getvalue().splitlines()
    assert activity_lines[-2].endswith("  thinking")
    assert activity_lines[-1].endswith("  writing answer.json")


def test_activity_api_rejects_values_outside_the_neutral_contract() -> None:
    with pytest.raises(ValueError, match="unknown"):
        AgentActivity("unknown")

    progress = RunProgress(
        [StageSpec("stage", "Stage")],
        stream=StringIO(),
    )
    progress.start("stage")
    with pytest.raises(TypeError, match="AgentActivity"):
        progress.activity("stage", "thinking")


def test_concurrent_parent_and_child_activity_updates_are_isolated() -> None:
    clock = FakeClock()
    progress = RunProgress(
        [StageSpec("stage", "Stage", (ChildSpec("child", "Child"),))],
        stream=StringIO(),
        clock=clock,
    )
    parent = progress.start("stage")
    child = progress.start_child("stage", "child")
    clock.advance(4)
    parent_activity = AgentActivity(AgentActivityKind.COMMAND, "make test")
    child_activity = AgentActivity(AgentActivityKind.WRITING, "result.json")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(progress.activity, "stage", parent_activity),
            executor.submit(
                progress.child_activity,
                "stage",
                "child",
                child_activity,
            ),
        ]
        for future in futures:
            future.result(timeout=2)

    assert parent.activity == parent_activity
    assert child.activity == child_activity
    assert parent.started_at == 0
    assert child.started_at == 0
    assert progress.elapsed("stage") == 4
    assert progress.child_elapsed("stage", "child") == 4


def test_recording_progress_records_only_neutral_parent_and_child_values(
    recording_progress: Any,
) -> None:
    parent_activity = AgentActivity(AgentActivityKind.READING, "README.md")
    child_activity = AgentActivity(AgentActivityKind.COMMAND, "make lint")

    recording_progress.update("plan", "preparing")
    recording_progress.activity("plan", parent_activity)
    recording_progress.update_child("plan", "prompt", "drafting")
    recording_progress.child_activity("plan", "prompt", child_activity)

    assert recording_progress.updates == [("plan", "preparing")]
    assert recording_progress.activities == [("plan", parent_activity)]
    assert recording_progress.child_updates == [("plan", "prompt", "drafting")]
    assert recording_progress.child_activities == [
        ("plan", "prompt", child_activity)
    ]
