"""Authoritative lifecycle and shared rendering for command progress.

Workflow code owns durable outcomes and must explicitly reconcile every started
record before closing the reporter.  This module owns the corresponding
human-readable account so interactive and plain invocations use the same
permanent lines.
"""

from __future__ import annotations

import math
import os
import re
import sys
import threading
import time
import weakref
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TextIO, TypeVar

from rich.cells import cell_len, chop_cells
from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Span, Text


class StageState(StrEnum):
    """Legal parent and child lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class AgentActivityKind(StrEnum):
    """Provider-neutral kinds of agent work shown by the reporter."""

    THINKING = "thinking"
    READING = "reading"
    SEARCHING = "searching"
    COMMAND = "command"
    WRITING = "writing"


TERMINAL_STATES = frozenset(
    {StageState.COMPLETED, StageState.FAILED, StageState.STOPPED}
)


@dataclass(frozen=True, slots=True)
class ChildSpec:
    """A stable child key and its user-facing label."""

    key: str
    label: str

    def __post_init__(self) -> None:
        _require_text(self.key, "child key")
        _require_text(self.label, "child label")


@dataclass(frozen=True, slots=True)
class StageSpec:
    """A stable parent key, label, and optional fixed children."""

    key: str
    label: str
    children: tuple[ChildSpec, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.key, "stage key")
        _require_text(self.label, "stage label")
        children = tuple(self.children)
        if len({child.key for child in children}) != len(children):
            raise ValueError(f"stage {self.key!r} has duplicate child keys")
        object.__setattr__(self, "children", children)


@dataclass(frozen=True, slots=True)
class AgentActivity:
    """The current kind of agent work and an optional detail string."""

    kind: AgentActivityKind
    detail: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", AgentActivityKind(self.kind))
        if self.detail is not None and not isinstance(self.detail, str):
            raise TypeError("activity detail must be a string or None")


@dataclass(slots=True)
class ChildRecord:
    """Lifecycle and timing retained for one fixed or dynamic child."""

    key: str
    label: str
    dynamic: bool
    state: StageState = StageState.PENDING
    detail: str | None = None
    activity: AgentActivity | None = None
    result: object | None = None
    started_at: float | None = None
    finished_at: float | None = None
    duration_seconds: float | None = None
    retained: bool = False
    _activity_is_latest: bool = field(default=False, repr=False)


@dataclass(slots=True)
class StageRecord:
    """Lifecycle and timing retained for a parent and all of its children."""

    key: str
    label: str
    state: StageState = StageState.PENDING
    detail: str | None = None
    activity: AgentActivity | None = None
    result: object | None = None
    started_at: float | None = None
    finished_at: float | None = None
    duration_seconds: float | None = None
    retained: bool = False
    _activity_is_latest: bool = field(default=False, repr=False)
    _children: dict[str, ChildRecord] = field(default_factory=dict, repr=False)

    @property
    def children(self) -> Mapping[str, ChildRecord]:
        """Return all children, including attempts omitted from rendering."""

        return MappingProxyType(self._children)


@dataclass(frozen=True, slots=True)
class ChildRenderState:
    """Bounded children to render and the number of older dynamic attempts."""

    children: tuple[ChildRecord, ...]
    earlier_attempt_count: int


@dataclass(frozen=True, slots=True)
class _RecordSnapshot:
    """Immutable rendering state copied from one authoritative record."""

    key: str
    label: str
    state: StageState
    detail: str | None
    activity: AgentActivity | None
    result: str | None
    duration_seconds: float | None
    retained: bool
    _activity_is_latest: bool
    dynamic: bool = False


@dataclass(frozen=True, slots=True)
class _StageSnapshot:
    """Immutable rendering state for a parent and its declared children."""

    record: _RecordSnapshot
    children: tuple[_RecordSnapshot, ...]


@dataclass(frozen=True, slots=True)
class _ProjectionSnapshot:
    """One atomic view of authoritative records and display-only previews."""

    stages: tuple[_StageSnapshot, ...]
    previews: tuple[StageSpec, ...]
    cohort_keys: frozenset[str]
    startup_label: str | None
    elapsed_seconds: float
    cancelling: bool


@dataclass(frozen=True, slots=True)
class _DetachedDisplay:
    """Renderer resources detached while the reporter lock is held."""

    live: Live | None
    worker: threading.Thread | None
    stopped: threading.Event | None


_ChildRenderRecord = TypeVar(
    "_ChildRenderRecord", ChildRecord, _RecordSnapshot
)


class ProgressError(ValueError):
    """Raised when a requested progress lifecycle operation is illegal."""


Clock = Callable[[], float]

_MAX_LIVE_ROWS = 8
_LABEL_COLUMN_WIDTH = 21
_DOTS_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_DOTS_FRAME = _DOTS_FRAMES[0]
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]+")
_TOKEN_SUBSTITUTIONS = {
    "✔": "[OK]",
    "✖": "[X]",
    "■": "[-]",
    "◦": "[.]",
    **dict.fromkeys(_DOTS_FRAMES, "*"),
    "│": "|",
    "├": "+-",
    "└": r"\-",
    "·": "-",
    "…": "...",
    "—": "-",
}
_STATE_GLYPHS = {
    StageState.PENDING: "◦",
    StageState.COMPLETED: "✔",
    StageState.FAILED: "✖",
    StageState.STOPPED: "■",
}
_STATE_STYLES = {
    StageState.PENDING: "dim",
    StageState.COMPLETED: "green",
    StageState.FAILED: "red",
    StageState.STOPPED: "dim cyan",
}


class RunProgress:
    """Own parent and child progress state for one command invocation."""

    def __init__(
        self,
        stages: Iterable[StageSpec] = (),
        *,
        stream: TextIO | None = None,
        clock: Clock = time.monotonic,
        width: int | None = None,
        enabled: bool = True,
        machine_readable: bool = False,
        attempt_history_limit: int = 2,
        heartbeat_interval: float = 30.0,
        startup_label: str | None = None,
        startup_pending: tuple[str, ...] = (),
    ) -> None:
        if width is not None and width <= 0:
            raise ValueError("width must be positive")
        if attempt_history_limit < 1:
            raise ValueError("attempt_history_limit must be at least one")
        if heartbeat_interval <= 0 or not math.isfinite(heartbeat_interval):
            raise ValueError("heartbeat_interval must be a finite positive value")
        if startup_label is not None:
            _require_text(startup_label, "startup label")
        self._stream = sys.stderr if stream is None else stream
        self._clock = clock
        self._started_at = self._clock()
        self._spinner_started_at = time.monotonic()
        self._width = width
        self._enabled = enabled and not machine_readable
        self._attempt_history_limit = attempt_history_limit
        self._heartbeat_interval = float(heartbeat_interval)
        self._interactive = self._enabled and _is_interactive(self._stream)
        self._console: Console | None = None
        self._live: Live | None = None
        self._live_empty = False
        self._next_heartbeat_at: float | None = None
        self._cadence_worker: threading.Thread | None = None
        self._cadence_stop: threading.Event | None = None
        self._render_failure: BaseException | None = None
        self._render_failed = False
        self._display_stopped = False
        self._closing = False
        self._stages: dict[str, StageRecord] = {}
        self._preview_specs: tuple[StageSpec, ...] = ()
        self._cohort_keys: frozenset[str] = frozenset()
        self._startup_label = startup_label
        self._cancelling = False
        self._closed = False
        self._suspension_depth = 0
        self._queued_lines: list[Text] = []
        self._lock = threading.RLock()
        self._initializing = True
        for spec in stages:
            self.declare(spec)
        startup_specs = tuple(StageSpec(label, label) for label in startup_pending)
        previews, cohort_keys = self._validated_preview(startup_specs, None)
        self._preview_specs = previews
        self._cohort_keys = cohort_keys
        self._initializing = False
        if self._interactive:
            self._refresh_safely()

    @property
    def stages(self) -> Mapping[str, StageRecord]:
        """Return every declared stage in declaration order."""

        return MappingProxyType(self._stages)

    @property
    def records(self) -> Mapping[str, StageRecord]:
        """Alias for the complete stage state exposed to renderers."""

        return self.stages

    @property
    def cancelling(self) -> bool:
        """Whether cancellation has been acknowledged for this run."""

        return self._cancelling

    @property
    def closed(self) -> bool:
        """Whether strict close validation has succeeded."""

        return self._closed

    @property
    def counts(self) -> Mapping[StageState, int]:
        """Count top-level stages by their current authoritative state."""

        counts = Counter(record.state for record in self._stages.values())
        return MappingProxyType({state: counts[state] for state in StageState})

    def declare(self, spec: StageSpec) -> StageRecord:
        """Declare a pending parent and its fixed children."""

        with self._lock:
            self._require_open()
            self._require_not_cancelling("declare a stage")
            if spec.key in self._stages:
                raise ProgressError(f"stage {spec.key!r} is already declared")
            record = StageRecord(spec.key, spec.label)
            record._children.update(
                {
                    child.key: ChildRecord(child.key, child.label, dynamic=False)
                    for child in spec.children
                }
            )
            self._stages[spec.key] = record
            self._preview_specs = tuple(
                preview
                for preview in self._preview_specs
                if preview.key != spec.key
            )
            if not self._initializing:
                self._refresh_safely()
            return record

    def preview_pending(
        self,
        specs: tuple[StageSpec, ...],
        *,
        cohort_keys: tuple[str, ...] | None = None,
    ) -> None:
        """Atomically replace display-only pending stages and count membership."""

        with self._lock:
            self._require_open()
            previews, selected_keys = self._validated_preview(specs, cohort_keys)
            self._preview_specs = previews
            self._cohort_keys = selected_keys
            self._refresh_safely()

    def start(self, stage_key: str) -> StageRecord:
        """Start a fresh parent timer exactly once."""

        with self._lock:
            record = self._stage(stage_key)
            self._require_new_work_allowed()
            self._start_record(record, f"stage {stage_key!r}")
            self._refresh_safely(started=record)
            return record

    def update(self, stage_key: str, detail: str | None) -> StageRecord:
        """Replace a running parent's current detail without adding history."""

        with self._lock:
            record = self._stage(stage_key)
            self._require_running(record, f"stage {stage_key!r}")
            record.detail = detail
            record._activity_is_latest = False
            self._refresh_safely()
            return record

    def activity(self, stage_key: str, activity: AgentActivity) -> StageRecord:
        """Replace a running parent's current agent activity."""

        with self._lock:
            record = self._stage(stage_key)
            self._require_running(record, f"stage {stage_key!r}")
            record.activity = _require_activity(activity)
            record._activity_is_latest = True
            self._refresh_safely()
            return record

    def seed_completed(
        self,
        stage_key: str,
        result: object,
        duration_seconds: float | None = None,
    ) -> StageRecord:
        """Seed one authoritative retained completed parent."""

        return self._seed_stage(
            stage_key, StageState.COMPLETED, result, duration_seconds
        )

    def seed_failed(
        self,
        stage_key: str,
        result: object,
        duration_seconds: float | None = None,
    ) -> StageRecord:
        """Seed one authoritative retained failed parent."""

        return self._seed_stage(stage_key, StageState.FAILED, result, duration_seconds)

    def complete(self, stage_key: str, result: object | None = None) -> StageRecord:
        """Complete a running parent with its authoritative result."""

        return self._finish_stage(stage_key, StageState.COMPLETED, result)

    def fail(self, stage_key: str, result: object | None = None) -> StageRecord:
        """Fail a running parent with its authoritative result."""

        return self._finish_stage(stage_key, StageState.FAILED, result)

    def stop(self, stage_key: str, result: object | None = None) -> StageRecord:
        """Stop a running parent after running children are reconciled.

        Pending children remain pending because cancellation can prevent their
        work from ever starting.
        """

        return self._finish_stage(stage_key, StageState.STOPPED, result)

    def declare_child(self, stage_key: str, child_spec: ChildSpec) -> ChildRecord:
        """Declare one uniquely keyed dynamic child attempt."""

        with self._lock:
            parent = self._stage(stage_key)
            self._require_new_work_allowed()
            self._require_parent_nonterminal(parent)
            if child_spec.key in parent._children:
                raise ProgressError(
                    f"child {child_spec.key!r} is already declared for stage "
                    f"{stage_key!r}"
                )
            child = ChildRecord(child_spec.key, child_spec.label, dynamic=True)
            parent._children[child.key] = child
            return child

    def start_child(self, stage_key: str, child_key: str) -> ChildRecord:
        """Start a fresh child timer independently of its parent timer."""

        with self._lock:
            parent, child = self._child(stage_key, child_key)
            self._require_new_work_allowed()
            self._require_running(parent, f"stage {stage_key!r}")
            self._start_record(child, f"child {child_key!r}")
            self._refresh_safely(started=child, parent_label=parent.label)
            return child

    def update_child(
        self, stage_key: str, child_key: str, detail: str | None
    ) -> ChildRecord:
        """Replace a running child's current detail without adding history."""

        with self._lock:
            _parent, child = self._child(stage_key, child_key)
            self._require_running(child, f"child {child_key!r}")
            child.detail = detail
            child._activity_is_latest = False
            self._refresh_safely()
            return child

    def child_activity(
        self,
        stage_key: str,
        child_key: str,
        activity: AgentActivity,
    ) -> ChildRecord:
        """Replace a running child's current agent activity."""

        with self._lock:
            _parent, child = self._child(stage_key, child_key)
            self._require_running(child, f"child {child_key!r}")
            child.activity = _require_activity(activity)
            child._activity_is_latest = True
            self._refresh_safely()
            return child

    def seed_child_completed(
        self,
        stage_key: str,
        child_key: str,
        result: object,
        duration_seconds: float | None = None,
    ) -> ChildRecord:
        """Seed one authoritative retained completed child attempt."""

        with self._lock:
            parent, child = self._child(stage_key, child_key)
            self._require_open()
            self._require_not_cancelling("seed a retained child")
            self._require_parent_nonterminal(parent)
            self._require_pending(child, f"child {child_key!r}")
            duration = _validate_duration(duration_seconds)
            self._seed_record(child, StageState.COMPLETED, result, duration)
            self._refresh_safely()
            return child

    def complete_child(
        self, stage_key: str, child_key: str, result: object | None = None
    ) -> ChildRecord:
        """Complete a running child with its authoritative result."""

        return self._finish_child(stage_key, child_key, StageState.COMPLETED, result)

    def fail_child(
        self, stage_key: str, child_key: str, result: object | None = None
    ) -> ChildRecord:
        """Fail a running child with its authoritative result."""

        return self._finish_child(stage_key, child_key, StageState.FAILED, result)

    def stop_child(
        self, stage_key: str, child_key: str, result: object | None = None
    ) -> ChildRecord:
        """Stop a running child after its owner reconciles it."""

        return self._finish_child(stage_key, child_key, StageState.STOPPED, result)

    def elapsed(self, stage_key: str) -> float | None:
        """Return a parent's frozen or current elapsed seconds."""

        with self._lock:
            return self._elapsed(self._stage(stage_key))

    def child_elapsed(self, stage_key: str, child_key: str) -> float | None:
        """Return a child's frozen or current elapsed seconds."""

        with self._lock:
            _parent, child = self._child(stage_key, child_key)
            return self._elapsed(child)

    def child_render_state(self, stage_key: str) -> ChildRenderState:
        """Return fixed children plus a bounded dynamic-attempt window."""

        with self._lock:
            parent = self._stage(stage_key)
            children, earlier_attempt_count = _select_render_children(
                parent._children.values(), self._attempt_history_limit
            )
            return ChildRenderState(
                children=children,
                earlier_attempt_count=earlier_attempt_count,
            )

    def refresh(self) -> None:
        """Refresh live elapsed time or emit a due plain-mode heartbeat."""

        with self._lock:
            self._require_open()
            self._refresh_safely()

    def raise_if_render_failed(self) -> None:
        """Consume and raise the first autonomous renderer failure, if any."""

        with self._lock:
            failure = self._render_failure
            self._render_failure = None
        if failure is not None:
            raise failure

    def begin_cancellation(self) -> bool:
        """Acknowledge cancellation once without changing any record state."""

        with self._lock:
            self._require_open()
            if self._cancelling:
                return False
            self._cancelling = True
            self._startup_label = None
            self._emit(Text("stopping…", style="dim cyan"))
            self._refresh_safely()
            return True

    @contextmanager
    def suspend(self) -> Iterator[RunProgress]:
        """Queue permanent output until the outermost suspension exits."""

        self.raise_if_render_failed()
        detached = _DetachedDisplay(None, None, None)
        with self._lock:
            self._require_open()
            if self._suspension_depth == 0:
                self._suspension_depth = 1
                detached = self._detach_display()
            else:
                self._suspension_depth += 1
        try:
            self._teardown_display(detached)
            self.raise_if_render_failed()
            yield self
        finally:
            with self._lock:
                self._suspension_depth -= 1
                if self._suspension_depth == 0:
                    queued, self._queued_lines = self._queued_lines, []
                    for line in queued:
                        if self._render_failed or self._display_stopped:
                            break
                        try:
                            self._write_permanent(line)
                        except BaseException as error:
                            self._latch_render_failure(error)
                    if not self._render_failed and not self._display_stopped:
                        self._reset_heartbeat()
                        self._refresh_safely()
        self.raise_if_render_failed()

    def close(self) -> None:
        """Close only after every started parent and child is terminal."""

        detached = _DetachedDisplay(None, None, None)
        with self._lock:
            if self._closed:
                return
            if self._closing:
                raise ProgressError("progress is already closing")
            running = [
                f"stage {stage.key!r}"
                for stage in self._stages.values()
                if stage.state is StageState.RUNNING
            ]
            running.extend(
                f"child {stage.key!r}/{child.key!r}"
                for stage in self._stages.values()
                for child in stage._children.values()
                if child.state is StageState.RUNNING
            )
            if running:
                raise ProgressError(
                    "cannot close progress with unresolved started records: "
                    + ", ".join(running)
                )
            if self._suspension_depth:
                raise ProgressError("cannot close progress while output is suspended")
            self._preview_specs = ()
            self._cohort_keys = frozenset()
            self._startup_label = None
            self._closing = True
            detached = self._detach_display()
        self._teardown_display(detached)
        with self._lock:
            elapsed_seconds = max(0.0, self._clock() - self._started_at)
            if self._enabled and not self._render_failed and not self._display_stopped:
                try:
                    self._write_permanent(
                        _format_summary_line(self._stages.values(), elapsed_seconds)
                    )
                except BaseException as error:
                    self._latch_render_failure(error)
            self._closed = True
            self._closing = False

    def stop_display(self) -> None:
        """Permanently dispose renderer resources without changing lifecycle."""

        with self._lock:
            if self._display_stopped:
                return
            self._display_stopped = True
            self._preview_specs = ()
            self._cohort_keys = frozenset()
            self._startup_label = None
            self._queued_lines = []
            detached = self._detach_display()
        self._teardown_display(detached)

    def _seed_stage(
        self,
        stage_key: str,
        state: StageState,
        result: object,
        duration_seconds: float | None,
    ) -> StageRecord:
        with self._lock:
            record = self._stage(stage_key)
            self._require_open()
            self._require_pending(record, f"stage {stage_key!r}")
            duration = _validate_duration(duration_seconds)
            self._require_terminal_children(record)
            self._seed_record(record, state, result, duration)
            self._emit_terminal_tree(record)
            return record

    def _finish_stage(
        self, stage_key: str, state: StageState, result: object | None
    ) -> StageRecord:
        with self._lock:
            record = self._stage(stage_key)
            self._require_open()
            self._require_running(record, f"stage {stage_key!r}")
            if state is StageState.STOPPED:
                self._require_no_running_children(record)
            else:
                self._require_terminal_children(record)
            self._finish_record(record, state, result)
            self._emit_terminal_tree(record)
            self._refresh_safely()
            return record

    def _finish_child(
        self,
        stage_key: str,
        child_key: str,
        state: StageState,
        result: object | None,
    ) -> ChildRecord:
        with self._lock:
            parent, child = self._child(stage_key, child_key)
            self._require_open()
            self._require_parent_nonterminal(parent)
            self._require_running(child, f"child {child_key!r}")
            self._finish_record(child, state, result)
            self._refresh_safely()
            return child

    def _stage(self, stage_key: str) -> StageRecord:
        self._require_open()
        try:
            return self._stages[stage_key]
        except KeyError as exc:
            raise ProgressError(f"unknown stage {stage_key!r}") from exc

    def _validated_preview(
        self,
        specs: tuple[StageSpec, ...],
        cohort_keys: tuple[str, ...] | None,
    ) -> tuple[tuple[StageSpec, ...], frozenset[str]]:
        candidates = tuple(specs)
        if any(not isinstance(spec, StageSpec) for spec in candidates):
            raise TypeError("pending previews must be StageSpec instances")
        spec_keys = tuple(spec.key for spec in candidates)
        available_keys = set(spec_keys)
        if len(available_keys) != len(spec_keys):
            raise ValueError("pending previews have duplicate stage keys")

        selected = spec_keys if cohort_keys is None else tuple(cohort_keys)
        if any(not isinstance(key, str) or not key for key in selected):
            raise ValueError("cohort keys must be non-empty strings")
        if len(set(selected)) != len(selected):
            raise ValueError("cohort keys must be unique")
        unknown = [key for key in selected if key not in available_keys]
        if unknown:
            raise ValueError(
                "cohort keys must be members of pending previews: "
                + ", ".join(repr(key) for key in unknown)
            )

        previews = tuple(
            spec for spec in candidates if spec.key not in self._stages
        )
        return previews, frozenset(selected)

    def _child(self, stage_key: str, child_key: str) -> tuple[StageRecord, ChildRecord]:
        parent = self._stage(stage_key)
        try:
            return parent, parent._children[child_key]
        except KeyError as exc:
            raise ProgressError(
                f"unknown child {child_key!r} for stage {stage_key!r}"
            ) from exc

    def _start_record(self, record: StageRecord | ChildRecord, name: str) -> None:
        self._require_pending(record, name)
        record.state = StageState.RUNNING
        record.started_at = self._clock()
        self._startup_label = None

    def _seed_record(
        self,
        record: StageRecord | ChildRecord,
        state: StageState,
        result: object,
        duration_seconds: float | None,
    ) -> None:
        record.state = state
        record.result = result
        record.duration_seconds = duration_seconds
        record.retained = True
        self._startup_label = None

    def _finish_record(
        self,
        record: StageRecord | ChildRecord,
        state: StageState,
        result: object | None,
    ) -> None:
        finished_at = self._clock()
        if record.started_at is None:  # guarded by _require_running
            raise AssertionError("running record has no start time")
        record.state = state
        record.result = result
        record.finished_at = finished_at
        record.duration_seconds = max(0.0, finished_at - record.started_at)

    def _elapsed(
        self, record: StageRecord | ChildRecord | _RecordSnapshot
    ) -> float | None:
        if isinstance(record, _RecordSnapshot):
            return record.duration_seconds
        if record.duration_seconds is not None or record.state in TERMINAL_STATES:
            return record.duration_seconds
        if record.started_at is None:
            return None
        return max(0.0, self._clock() - record.started_at)

    def _require_open(self) -> None:
        if self._closed or self._closing:
            raise ProgressError("progress is already closed")

    def _require_not_cancelling(self, action: str) -> None:
        if self._cancelling:
            raise ProgressError(f"cannot {action} after cancellation begins")

    def _require_new_work_allowed(self) -> None:
        self._require_open()
        self._require_not_cancelling("start new work")

    @staticmethod
    def _require_pending(record: StageRecord | ChildRecord, name: str) -> None:
        if record.state is not StageState.PENDING:
            raise ProgressError(
                f"{name} must be pending, not {record.state.value}"
            )

    @staticmethod
    def _require_running(record: StageRecord | ChildRecord, name: str) -> None:
        if record.state is not StageState.RUNNING:
            raise ProgressError(
                f"{name} must be running, not {record.state.value}"
            )

    @staticmethod
    def _require_parent_nonterminal(parent: StageRecord) -> None:
        if parent.state in TERMINAL_STATES:
            raise ProgressError(f"stage {parent.key!r} is already terminal")

    @staticmethod
    def _require_terminal_children(parent: StageRecord) -> None:
        unresolved = [
            child.key
            for child in parent._children.values()
            if child.state not in TERMINAL_STATES
        ]
        if unresolved:
            raise ProgressError(
                f"stage {parent.key!r} has unresolved children: "
                + ", ".join(unresolved)
            )

    @staticmethod
    def _require_no_running_children(parent: StageRecord) -> None:
        running = [
            child.key
            for child in parent._children.values()
            if child.state is StageState.RUNNING
        ]
        if running:
            raise ProgressError(
                f"stage {parent.key!r} has running children: " + ", ".join(running)
            )

    def _emit_terminal(
        self, record: StageRecord | ChildRecord, *, parent_label: str | None = None
    ) -> None:
        self._emit(_format_terminal_line(record, parent_label=parent_label))

    def _emit_terminal_tree(self, parent: StageRecord) -> None:
        self._emit_terminal(parent)
        children = self.child_render_state(parent.key).children
        for index, child in enumerate(children):
            branch = "└" if index == len(children) - 1 else "├"
            self._emit(_format_child_terminal_line(child, branch=branch))

    def _emit(self, line: Text) -> None:
        if not self._enabled or self._render_failed or self._display_stopped:
            return
        if self._suspension_depth:
            self._queued_lines.append(line)
            return
        try:
            if self._interactive:
                snapshot = self._projection_snapshot()
                if self._project_live_lines(snapshot):
                    self._refresh_transient()
            self._write_permanent(line)
            if self._interactive:
                self._refresh_transient()
        except BaseException as error:
            self._latch_render_failure(error)
            self._render_failure = None
            raise

    def _refresh_safely(
        self,
        *,
        started: StageRecord | ChildRecord | None = None,
        parent_label: str | None = None,
    ) -> None:
        try:
            self._refresh_transient(started=started, parent_label=parent_label)
        except BaseException as error:
            self._latch_render_failure(error)
            self._render_failure = None
            raise

    def _refresh_transient(
        self,
        *,
        started: StageRecord | ChildRecord | None = None,
        parent_label: str | None = None,
    ) -> None:
        if (
            not self._enabled
            or self._suspension_depth
            or self._render_failed
            or self._display_stopped
            or self._closing
            or self._closed
        ):
            return
        snapshot = self._projection_snapshot()
        if self._interactive:
            lines = self._project_live_lines(snapshot)
            if not lines:
                if self._live is not None and not self._live_empty:
                    self._live.update(Text(""), refresh=True)
                    self._live_empty = True
                return
            renderable = Group(*(self._live_renderable(line) for line in lines))
            if self._live is None:
                self._live = Live(
                    renderable,
                    console=self._output_console(),
                    auto_refresh=False,
                    transient=True,
                    redirect_stdout=False,
                    redirect_stderr=False,
                )
                self._live.start(refresh=True)
            else:
                self._live.update(renderable, refresh=True)
            self._live_empty = False
            self._ensure_cadence_worker()
            return

        lines = self._plain_active_lines(snapshot)
        if not lines:
            self._next_heartbeat_at = None
            return
        now = self._clock()
        if started is not None:
            started_snapshot = self._record_snapshot(started, now)
            self._write_plain(
                self._format_running_line(
                    started_snapshot, parent_label=parent_label
                )
            )
            if self._next_heartbeat_at is None:
                self._next_heartbeat_at = now + self._heartbeat_interval
        elif self._next_heartbeat_at is None or now >= self._next_heartbeat_at:
            for line in lines:
                self._write_plain(line)
            self._next_heartbeat_at = now + self._heartbeat_interval
        self._ensure_cadence_worker()

    def _live_lines(self) -> list[Text]:
        return self._project_live_lines(self._projection_snapshot())

    def _projection_snapshot(self) -> _ProjectionSnapshot:
        with self._lock:
            now = self._clock()
            stages = tuple(
                _StageSnapshot(
                    record=self._record_snapshot(stage, now),
                    children=tuple(
                        self._record_snapshot(child, now)
                        for child in stage._children.values()
                    ),
                )
                for stage in self._stages.values()
            )
            return _ProjectionSnapshot(
                stages=stages,
                previews=self._preview_specs,
                cohort_keys=self._cohort_keys,
                startup_label=self._startup_label,
                elapsed_seconds=max(0.0, now - self._started_at),
                cancelling=self._cancelling,
            )

    def _record_snapshot(
        self, record: StageRecord | ChildRecord, now: float
    ) -> _RecordSnapshot:
        duration = record.duration_seconds
        if duration is None and record.state is StageState.RUNNING:
            if record.started_at is None:
                raise AssertionError("running record has no start time")
            duration = max(0.0, now - record.started_at)
        return _RecordSnapshot(
            key=record.key,
            label=record.label,
            state=record.state,
            detail=record.detail,
            activity=record.activity,
            result=None if record.result is None else str(record.result),
            duration_seconds=duration,
            retained=record.retained,
            _activity_is_latest=record._activity_is_latest,
            dynamic=isinstance(record, ChildRecord) and record.dynamic,
        )

    def _project_live_lines(self, snapshot: _ProjectionSnapshot) -> list[Text]:
        projected_lines: list[tuple[Text, StageState | None]] = []
        stages_by_key = {stage.record.key: stage for stage in snapshot.stages}
        aggregate_cohort = len(snapshot.cohort_keys) > _MAX_LIVE_ROWS

        if snapshot.startup_label is not None:
            projected_lines.append(
                (
                    self._format_startup_line(
                        snapshot.startup_label, snapshot.elapsed_seconds
                    ),
                    StageState.RUNNING,
                )
            )

        for stage in snapshot.stages:
            record = stage.record
            if record.state is StageState.RUNNING:
                projected_lines.append(
                    (
                        self._format_running_line(
                            record, cancelling=snapshot.cancelling
                        ),
                        StageState.RUNNING,
                    )
                )
                children, earlier_attempt_count = _select_render_children(
                    stage.children, self._attempt_history_limit
                )
                projected_lines.extend(
                    (
                        self._format_child_live_line(
                            child,
                            parent_label=record.label,
                            cancelling=snapshot.cancelling,
                        ),
                        child.state,
                    )
                    for child in children
                )
                if earlier_attempt_count:
                    noun = "attempt" if earlier_attempt_count == 1 else "attempts"
                    projected_lines.append(
                        (
                            Text(
                                f"… {earlier_attempt_count} earlier {noun}",
                                style="dim",
                            ),
                            None,
                        )
                    )
            elif record.state is StageState.PENDING and not (
                aggregate_cohort and record.key in snapshot.cohort_keys
            ):
                projected_lines.append(
                    (_format_pending_stage_line(record.label), StageState.PENDING)
                )

        projected_lines.extend(
            (_format_pending_stage_line(spec.label), StageState.PENDING)
            for spec in snapshot.previews
            if not (aggregate_cohort and spec.key in snapshot.cohort_keys)
        )

        work_lines = self._bound_live_lines(projected_lines)
        if not work_lines and not aggregate_cohort and not snapshot.cancelling:
            return []
        if snapshot.cancelling:
            footer = Text("stopping…", style="dim cyan")
        elif aggregate_cohort:
            footer = self._format_cohort_counts(snapshot, stages_by_key)
            footer.append("  ·  ctrl-c to stop")
        else:
            footer = Text("ctrl-c to stop", style="dim")
        return [*work_lines, Text(""), footer]

    def _format_cohort_counts(
        self,
        snapshot: _ProjectionSnapshot,
        stages_by_key: Mapping[str, _StageSnapshot],
    ) -> Text:
        preview_keys = {spec.key for spec in snapshot.previews}
        states = [
            stages_by_key[key].record.state
            if key in stages_by_key
            else StageState.PENDING
            for key in snapshot.cohort_keys
            if key in stages_by_key or key in preview_keys
        ]
        counts = Counter(states)
        done = sum(counts[state] for state in TERMINAL_STATES)
        return Text(
            f"{done} done · {counts[StageState.RUNNING]} running · "
            f"{counts[StageState.PENDING]} pending",
            style="dim",
        )

    def _format_startup_line(self, label: str, elapsed: float) -> Text:
        normalized = _normalize_cell(label)
        line = Text(_DOTS_FRAME, style="cyan")
        line.append(" ")
        line.append(normalized)
        line.append(" " * _column_gap(normalized))
        line.append(_format_duration(elapsed), style="dim")
        line.append("  thinking")
        return line

    def _plain_active_lines(self, snapshot: _ProjectionSnapshot) -> list[Text]:
        lines: list[Text] = []
        for stage in snapshot.stages:
            if stage.record.state is StageState.RUNNING:
                lines.append(
                    self._format_running_line(
                        stage.record, cancelling=snapshot.cancelling
                    )
                )
            lines.extend(
                self._format_running_line(
                    child,
                    parent_label=stage.record.label,
                    cancelling=snapshot.cancelling,
                )
                for child in stage.children
                if child.state is StageState.RUNNING
            )
        return lines

    @staticmethod
    def _bound_live_lines(
        lines: list[tuple[Text, StageState | None]],
    ) -> list[Text]:
        prioritized = sorted(
            lines, key=lambda line: line[1] is not StageState.RUNNING
        )
        if len(prioritized) <= _MAX_LIVE_ROWS:
            return [line for line, _state in prioritized]
        visible = prioritized[: _MAX_LIVE_ROWS - 1]
        hidden = prioritized[_MAX_LIVE_ROWS - 1 :]
        hidden_states = {state for _line, state in hidden}
        suffix = ""
        if hidden_states == {StageState.RUNNING}:
            suffix = " running"
        elif hidden_states == {StageState.PENDING}:
            suffix = " pending"
        return [
            *(line for line, _state in visible),
            Text(f"… {len(hidden)} more{suffix}", style="dim"),
        ]

    def _format_child_live_line(
        self,
        child: ChildRecord | _RecordSnapshot,
        *,
        parent_label: str,
        cancelling: bool | None = None,
    ) -> Text:
        if child.state is StageState.RUNNING:
            return self._format_running_line(
                child,
                parent_label=parent_label,
                cancelling=cancelling,
            )
        if child.state is StageState.PENDING:
            return _format_pending_line(child, parent_label=parent_label)
        return _format_terminal_line(child, parent_label=parent_label)

    def _format_running_line(
        self,
        record: StageRecord | ChildRecord | _RecordSnapshot,
        *,
        parent_label: str | None = None,
        cancelling: bool | None = None,
    ) -> Text:
        label = _format_label(record, parent_label=parent_label)
        elapsed = self._elapsed(record)
        work = _format_current_work(record)
        if self._cancelling if cancelling is None else cancelling:
            work = "stopping…"
        line = Text(_DOTS_FRAME, style="cyan")
        line.append(" ")
        line.append(label)
        line.append(" " * _column_gap(label))
        line.append(_format_duration(elapsed), style="dim")
        line.append("  ")
        line.append(work)
        return line

    def _live_renderable(self, line: Text) -> Text | Spinner:
        line = self._truncate(line)
        is_spinner = line.plain.startswith(_DOTS_FRAME)
        line = self._stream_safe(line)
        if is_spinner and _can_encode(_DOTS_FRAMES, self._stream_encoding()):
            spinner_text = line[2:]
            spinner_text.no_wrap = True
            spinner = Spinner("dots", text=spinner_text, style="cyan")
            spinner.start_time = self._spinner_started_at
            return spinner
        line.no_wrap = True
        return line

    def _reset_heartbeat(self) -> None:
        self._next_heartbeat_at = (
            self._clock() + self._heartbeat_interval
            if self._plain_active_lines(self._projection_snapshot())
            else None
        )

    def _ensure_cadence_worker(self) -> None:
        if self._cadence_worker is not None and self._cadence_worker.is_alive():
            return
        if not self._cadence_work_pending():
            return
        stopped = threading.Event()
        progress_ref = weakref.ref(self, lambda _reference: stopped.set())
        interval = 0.1 if self._interactive else self._heartbeat_interval
        worker = threading.Thread(
            target=_run_cadence_worker,
            args=(progress_ref, stopped, interval),
            name="betterborg-progress-renderer",
            daemon=True,
        )
        self._cadence_stop = stopped
        self._cadence_worker = worker
        worker.start()

    def _cadence_work_pending(self) -> bool:
        snapshot = self._projection_snapshot()
        if self._interactive:
            return snapshot.startup_label is not None or any(
                stage.record.state is StageState.RUNNING
                or any(
                    child.state is StageState.RUNNING
                    for child in stage.children
                )
                for stage in snapshot.stages
            )
        return bool(self._plain_active_lines(snapshot))

    def _render_cadence_frame(self, stopped: threading.Event) -> bool:
        try:
            with self._lock:
                if (
                    stopped is not self._cadence_stop
                    or stopped.is_set()
                    or self._suspension_depth
                    or self._render_failed
                    or self._display_stopped
                    or self._closing
                    or self._closed
                ):
                    return False
                if not self._cadence_work_pending():
                    return False
                snapshot = self._projection_snapshot()
                if self._interactive:
                    lines = self._project_live_lines(snapshot)
                    renderable = Group(
                        *(self._live_renderable(line) for line in lines)
                    )
                    if self._live is None:
                        self._live = Live(
                            renderable,
                            console=self._output_console(),
                            auto_refresh=False,
                            transient=True,
                            redirect_stdout=False,
                            redirect_stderr=False,
                        )
                        self._live.start(refresh=True)
                    else:
                        self._live.update(renderable, refresh=True)
                    self._live_empty = False
                else:
                    for line in self._plain_active_lines(snapshot):
                        self._write_plain(line)
                    self._next_heartbeat_at = (
                        self._clock() + self._heartbeat_interval
                    )
                return True
        except BaseException as error:
            with self._lock:
                self._latch_render_failure(error)
                detached = _DetachedDisplay(self._detach_live(), None, None)
                self._next_heartbeat_at = None
            self._teardown_display(detached)
            return False

    def _cadence_worker_finished(
        self, worker: threading.Thread, stopped: threading.Event
    ) -> None:
        with self._lock:
            if self._cadence_worker is worker and self._cadence_stop is stopped:
                self._cadence_worker = None
                self._cadence_stop = None
                if (
                    not self._suspension_depth
                    and not self._render_failed
                    and not self._display_stopped
                    and not self._closing
                    and not self._closed
                ):
                    self._ensure_cadence_worker()

    def _latch_render_failure(self, error: BaseException) -> None:
        if self._render_failure is None:
            self._render_failure = error
        self._render_failed = True
        if self._cadence_stop is not None:
            self._cadence_stop.set()

    def _detach_display(self) -> _DetachedDisplay:
        live = self._detach_live()
        worker, self._cadence_worker = self._cadence_worker, None
        stopped, self._cadence_stop = self._cadence_stop, None
        if stopped is not None:
            stopped.set()
        self._next_heartbeat_at = None
        return _DetachedDisplay(live, worker, stopped)

    def _detach_live(self) -> Live | None:
        live, self._live = self._live, None
        self._live_empty = False
        return live

    def _teardown_display(self, detached: _DetachedDisplay) -> None:
        if detached.live is not None:
            try:
                detached.live.stop()
            except BaseException as error:
                with self._lock:
                    self._latch_render_failure(error)
        if (
            detached.worker is not None
            and detached.worker is not threading.current_thread()
        ):
            detached.worker.join()

    def _output_console(self) -> Console:
        if self._console is None:
            options: dict[str, object] = {
                "file": self._stream,
                "force_terminal": True,
                "highlight": False,
                "no_color": "NO_COLOR" in os.environ,
            }
            if self._width is not None:
                options["width"] = self._width
            self._console = Console(**options)
        return self._console

    def _write_permanent(self, line: Text) -> None:
        if self._interactive:
            line = self._stream_safe(self._truncate(line))
            self._output_console().print(line, soft_wrap=True)
        else:
            self._write_plain(line)

    def _write_plain(self, line: Text) -> None:
        line = self._stream_safe(self._truncate(line))
        self._stream.write(line.plain + "\n")
        self._stream.flush()

    def _stream_safe(self, line: Text) -> Text:
        return _degrade_text(line, self._stream_encoding())

    def _stream_encoding(self) -> str:
        encoding = getattr(self._stream, "encoding", None)
        return encoding if isinstance(encoding, str) and encoding else "utf-8"

    def _truncate(self, line: Text) -> Text:
        if self._width is None or cell_len(line.plain) <= self._width:
            return line
        if self._width == 1:
            return Text("…", style=line.style)
        prefix = chop_cells(line.plain, self._width - 1)[0]
        truncated = line[: len(prefix)]
        truncated.append("…")
        return truncated


def _run_cadence_worker(
    progress_ref: weakref.ReferenceType[RunProgress],
    stopped: threading.Event,
    interval: float,
) -> None:
    """Refresh one reporter without retaining it after its owner releases it."""

    worker = threading.current_thread()
    try:
        while not stopped.wait(interval):
            progress = progress_ref()
            if progress is None:
                return
            keep_running = progress._render_cadence_frame(stopped)
            del progress
            if not keep_running:
                return
    finally:
        progress = progress_ref()
        if progress is not None:
            progress._cadence_worker_finished(worker, stopped)


def _format_terminal_line(
    record: StageRecord | ChildRecord | _RecordSnapshot,
    *,
    parent_label: str | None = None,
) -> Text:
    label = _format_label(record, parent_label=parent_label)
    glyph = _STATE_GLYPHS[record.state]
    line = Text(glyph, style=_STATE_STYLES[record.state])
    line.append(" ")
    line.append(label)
    line.append(" " * _column_gap(label))
    line.append(_format_duration(record.duration_seconds), style="dim")
    result_parts = []
    if record.result is not None:
        result_parts.append(_normalize_cell(str(record.result)))
    if record.retained:
        result_parts.append("reused from earlier run")
    if result_parts:
        line.append("  ")
        line.append(" · ".join(result_parts))
    return line


def _format_child_terminal_line(child: ChildRecord, *, branch: str) -> Text:
    return Text.assemble((branch, "dim"), " ", _format_terminal_line(child))


def _format_pending_line(
    record: ChildRecord | _RecordSnapshot, *, parent_label: str | None = None
) -> Text:
    label = _format_label(record, parent_label=parent_label)
    return Text(
        f"{_STATE_GLYPHS[StageState.PENDING]} {label}",
        style=_STATE_STYLES[StageState.PENDING],
    )


def _format_pending_stage_line(label: str) -> Text:
    return Text(
        f"  {_STATE_GLYPHS[StageState.PENDING]} {_normalize_cell(label)}",
        style=_STATE_STYLES[StageState.PENDING],
    )


def _format_label(
    record: StageRecord | ChildRecord | _RecordSnapshot,
    *,
    parent_label: str | None = None,
) -> str:
    label = _normalize_cell(record.label)
    if parent_label is not None:
        label = f"{_normalize_cell(parent_label)}: {label}"
    return label


def _column_gap(label: str) -> int:
    return max(2, _LABEL_COLUMN_WIDTH - cell_len(label) + 2)


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    whole_seconds = int(seconds)
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _format_current_work(
    record: StageRecord | ChildRecord | _RecordSnapshot,
) -> str:
    if record._activity_is_latest and record.activity is not None:
        return _format_activity(record.activity)
    if record.detail is not None:
        detail = _normalize_cell(record.detail)
        if detail.strip():
            return detail
    return "thinking"


def _format_activity(activity: AgentActivity) -> str:
    if activity.kind is AgentActivityKind.THINKING:
        return "thinking"
    detail = "" if activity.detail is None else _normalize_cell(activity.detail)
    if not detail.strip():
        return "thinking"
    if activity.kind is AgentActivityKind.READING:
        return f"reading {detail}"
    if activity.kind is AgentActivityKind.SEARCHING:
        return f'searching "{detail}"'
    if activity.kind is AgentActivityKind.COMMAND:
        return f"running {detail}"
    return f"writing {detail}"


def _normalize_cell(value: str) -> str:
    return _CONTROL_RE.sub(" ", value)


def _degrade_text(text: Text, encoding: str) -> Text:
    if _can_encode(text.plain, encoding):
        return text

    parts: list[str] = []
    offsets = [0]
    for character in text.plain:
        replacement = character
        if not _can_encode(character, encoding):
            replacement = _TOKEN_SUBSTITUTIONS.get(character)
            if replacement is None:
                replacement = character.encode(
                    encoding, errors="backslashreplace"
                ).decode(encoding)
        parts.append(replacement)
        offsets.append(offsets[-1] + len(replacement))

    spans = [
        Span(offsets[span.start], offsets[span.end], span.style)
        for span in text.spans
    ]
    return Text(
        "".join(parts),
        style=text.style,
        justify=text.justify,
        overflow=text.overflow,
        no_wrap=text.no_wrap,
        end=text.end,
        tab_size=text.tab_size,
        spans=spans,
    )


def _can_encode(value: str, encoding: str) -> bool:
    try:
        value.encode(encoding)
    except UnicodeEncodeError:
        return False
    return True


def _format_summary_line(
    records: Iterable[StageRecord], elapsed_seconds: float
) -> Text:
    participating = tuple(
        record for record in records if record.state is not StageState.PENDING
    )
    counts = Counter(record.state for record in participating)
    stage_word = "stage" if len(participating) == 1 else "stages"
    outcome = "none failed or stopped"
    if counts[StageState.FAILED] or counts[StageState.STOPPED]:
        outcome = (
            f"{counts[StageState.FAILED]} failed and "
            f"{counts[StageState.STOPPED]} stopped"
        )
    return Text(
        f"{counts[StageState.COMPLETED]} of {len(participating)} {stage_word} "
        f"finished in {_format_duration(elapsed_seconds)}; {outcome}.",
        style="dim",
    )


def _is_interactive(stream: TextIO) -> bool:
    if (
        "CI" in os.environ
        or "NO_COLOR" in os.environ
        or os.environ.get("TERM", "").casefold() == "dumb"
    ):
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, OSError):
        return False


def _select_render_children(
    children: Iterable[_ChildRenderRecord], history_limit: int
) -> tuple[tuple[_ChildRenderRecord, ...], int]:
    """Select fixed children and one bounded window of dynamic attempts."""

    records = tuple(children)
    fixed = [child for child in records if not child.dynamic]
    attempts = [child for child in records if child.dynamic]
    active = [child for child in attempts if child.state not in TERMINAL_STATES]
    active_keys = {child.key for child in active}
    remaining_slots = max(history_limit - len(active), 0)
    terminal = [child for child in attempts if child.state in TERMINAL_STATES]
    latest_terminal = terminal[-remaining_slots:] if remaining_slots else []
    visible_keys = active_keys | {child.key for child in latest_terminal}
    visible_attempts = [child for child in attempts if child.key in visible_keys]
    return tuple(fixed + visible_attempts), len(attempts) - len(visible_attempts)


def _require_activity(activity: AgentActivity) -> AgentActivity:
    if not isinstance(activity, AgentActivity):
        raise TypeError("activity must be an AgentActivity")
    return activity


def _validate_duration(duration_seconds: float | None) -> float | None:
    if duration_seconds is None:
        return None
    duration = float(duration_seconds)
    if duration < 0 or not math.isfinite(duration):
        raise ValueError("duration_seconds must be a finite non-negative value")
    return duration


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


__all__ = [
    "AgentActivity",
    "AgentActivityKind",
    "ChildRecord",
    "ChildRenderState",
    "ChildSpec",
    "ProgressError",
    "RunProgress",
    "StageRecord",
    "StageSpec",
    "StageState",
    "TERMINAL_STATES",
]
