"""Authoritative lifecycle state for command progress reporting.

Rendering is deliberately kept behind the small ``_emit_terminal`` boundary in
this module.  Workflow code owns durable outcomes and must explicitly reconcile
every started record before closing the reporter.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TextIO


class StageState(StrEnum):
    """Legal parent and child lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


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

    kind: str
    detail: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.kind, "activity kind")


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


class ProgressError(ValueError):
    """Raised when a requested progress lifecycle operation is illegal."""


Clock = Callable[[], float]


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
    ) -> None:
        if width is not None and width <= 0:
            raise ValueError("width must be positive")
        if attempt_history_limit < 1:
            raise ValueError("attempt_history_limit must be at least one")
        self._stream = sys.stderr if stream is None else stream
        self._clock = clock
        self._width = width
        self._enabled = enabled and not machine_readable
        self._attempt_history_limit = attempt_history_limit
        self._stages: dict[str, StageRecord] = {}
        self._cancelling = False
        self._closed = False
        self._suspension_depth = 0
        self._queued_lines: list[str] = []
        self._lock = threading.RLock()
        for spec in stages:
            self.declare(spec)

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
            return record

    def start(self, stage_key: str) -> StageRecord:
        """Start a fresh parent timer exactly once."""

        with self._lock:
            record = self._stage(stage_key)
            self._require_new_work_allowed()
            self._start_record(record, f"stage {stage_key!r}")
            return record

    def update(self, stage_key: str, detail: str | None) -> StageRecord:
        """Replace a running parent's current detail without adding history."""

        with self._lock:
            record = self._stage(stage_key)
            self._require_running(record, f"stage {stage_key!r}")
            record.detail = detail
            return record

    def activity(
        self, stage_key: str, activity: AgentActivity | str, detail: str | None = None
    ) -> StageRecord:
        """Replace a running parent's current agent activity."""

        with self._lock:
            record = self._stage(stage_key)
            self._require_running(record, f"stage {stage_key!r}")
            record.activity = _coerce_activity(activity, detail)
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
        """Stop a running parent after its workflow owner reconciles it."""

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
            return child

    def update_child(
        self, stage_key: str, child_key: str, detail: str | None
    ) -> ChildRecord:
        """Replace a running child's current detail without adding history."""

        with self._lock:
            _parent, child = self._child(stage_key, child_key)
            self._require_running(child, f"child {child_key!r}")
            child.detail = detail
            return child

    def child_activity(
        self,
        stage_key: str,
        child_key: str,
        activity: AgentActivity | str,
        detail: str | None = None,
    ) -> ChildRecord:
        """Replace a running child's current agent activity."""

        with self._lock:
            _parent, child = self._child(stage_key, child_key)
            self._require_running(child, f"child {child_key!r}")
            child.activity = _coerce_activity(activity, detail)
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
            self._emit_terminal(child, parent_label=parent.label)
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
            fixed = [child for child in parent._children.values() if not child.dynamic]
            attempts = [child for child in parent._children.values() if child.dynamic]
            active = [child for child in attempts if child.state not in TERMINAL_STATES]
            active_keys = {child.key for child in active}
            remaining_slots = max(self._attempt_history_limit - len(active), 0)
            latest_terminal = [
                child for child in attempts if child.state in TERMINAL_STATES
            ][-remaining_slots:]
            if remaining_slots == 0:
                latest_terminal = []
            visible_keys = active_keys | {child.key for child in latest_terminal}
            visible_attempts = [
                child for child in attempts if child.key in visible_keys
            ]
            return ChildRenderState(
                children=tuple(fixed + visible_attempts),
                earlier_attempt_count=len(attempts) - len(visible_attempts),
            )

    def begin_cancellation(self) -> bool:
        """Acknowledge cancellation once without changing any record state."""

        with self._lock:
            self._require_open()
            if self._cancelling:
                return False
            self._cancelling = True
            self._emit("stopping...")
            return True

    @contextmanager
    def suspend(self) -> Iterator[RunProgress]:
        """Queue permanent output until the outermost suspension exits."""

        with self._lock:
            self._require_open()
            self._suspension_depth += 1
        try:
            yield self
        finally:
            with self._lock:
                self._suspension_depth -= 1
                if self._suspension_depth == 0:
                    queued, self._queued_lines = self._queued_lines, []
                    for line in queued:
                        self._write(line)

    def close(self) -> None:
        """Close only after every started parent and child is terminal."""

        with self._lock:
            if self._closed:
                return
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
            self._closed = True

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
            self._require_not_cancelling("seed a retained stage")
            self._require_pending(record, f"stage {stage_key!r}")
            duration = _validate_duration(duration_seconds)
            self._require_terminal_children(record)
            self._seed_record(record, state, result, duration)
            self._emit_terminal(record)
            return record

    def _finish_stage(
        self, stage_key: str, state: StageState, result: object | None
    ) -> StageRecord:
        with self._lock:
            record = self._stage(stage_key)
            self._require_open()
            self._require_running(record, f"stage {stage_key!r}")
            self._require_terminal_children(record)
            self._finish_record(record, state, result)
            self._emit_terminal(record)
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
            self._emit_terminal(child, parent_label=parent.label)
            return child

    def _stage(self, stage_key: str) -> StageRecord:
        self._require_open()
        try:
            return self._stages[stage_key]
        except KeyError as exc:
            raise ProgressError(f"unknown stage {stage_key!r}") from exc

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

    def _elapsed(self, record: StageRecord | ChildRecord) -> float | None:
        if record.duration_seconds is not None or record.state in TERMINAL_STATES:
            return record.duration_seconds
        if record.started_at is None:
            return None
        return max(0.0, self._clock() - record.started_at)

    def _require_open(self) -> None:
        if self._closed:
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

    def _emit_terminal(
        self, record: StageRecord | ChildRecord, *, parent_label: str | None = None
    ) -> None:
        self._emit(_format_terminal_line(record, parent_label=parent_label))

    def _emit(self, line: str) -> None:
        if not self._enabled:
            return
        if self._suspension_depth:
            self._queued_lines.append(line)
            return
        self._write(line)

    def _write(self, line: str) -> None:
        if self._width is not None and len(line) > self._width:
            line = line[: max(self._width - 1, 0)] + "…"
        self._stream.write(line + "\n")
        self._stream.flush()


def _format_terminal_line(
    record: StageRecord | ChildRecord, *, parent_label: str | None = None
) -> str:
    label = record.label
    if parent_label is not None:
        label = f"{parent_label}: {label}"
    duration = (
        "duration unknown"
        if record.duration_seconds is None
        else f"{record.duration_seconds:.1f}s"
    )
    result = "" if record.result is None else f" — {record.result}"
    retained = " [retained]" if record.retained else ""
    return f"{record.state.value} {label}{result} ({duration}){retained}"


def _coerce_activity(
    activity: AgentActivity | str, detail: str | None
) -> AgentActivity:
    if isinstance(activity, AgentActivity):
        if detail is not None:
            raise ValueError("detail cannot accompany an AgentActivity")
        return activity
    return AgentActivity(activity, detail)


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
