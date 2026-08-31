"""Shared durable execution machinery for planning-agent turns."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from betterborg_cli.agent_runtime.api_tools import READ_ONLY_API_TOOLS
from betterborg_cli.agent_runtime.base import (
    AgentAdapter,
    AgentRunSpec,
    AgentStatus,
    CancellationToken,
)
from betterborg_cli.agent_runtime.selection import SelectedAgent
from betterborg_cli.agent_runtime.structured import (
    StructuredResultError,
    validate_structured_result,
)
from betterborg_cli.planning.plan_contracts import validate_plan
from betterborg_cli.planning.worktree import materialize_planning_worktree
from betterborg_cli.progress import AgentActivity
from betterborg_cli.store import (
    Borg,
    BorgState,
    PlanningAttempt,
    PlanningAttemptStatus,
    Repository,
    SqliteStore,
)

ErrorFactory = Callable[[str], Exception]


def current_planning_cycle_attempts(
    store: SqliteStore, borg_id: UUID
) -> list[PlanningAttempt]:
    """Return attempts belonging to the latest human planning cycle."""

    attempts = store.list_planning_attempts(borg_id)
    change_requests = store.list_plan_change_requests(borg_id)
    if not change_requests:
        return attempts
    cycle_started_at = change_requests[-1].created_at
    return [item for item in attempts if item.started_at >= cycle_started_at]


def completed_planning_phase_attempts(
    attempts: Sequence[PlanningAttempt], phase: str
) -> list[PlanningAttempt]:
    """Project one phase's completed attempts from durable cycle history."""

    return [
        item
        for item in attempts
        if item.phase == phase and item.status is PlanningAttemptStatus.COMPLETED
    ]


def planning_request_change_attempts(
    attempts: Sequence[PlanningAttempt], phase: str, *, round_cap: int
) -> list[PlanningAttempt]:
    """Return review rejections which are eligible to create revision work."""

    return [
        item
        for index, item in enumerate(
            completed_planning_phase_attempts(attempts, phase), start=1
        )
        if index < round_cap
        and (item.result or {}).get("decision") == "request_changes"
    ]


def latest_planning_review_requests_changes(
    attempts: Sequence[PlanningAttempt], phase: str
) -> bool:
    """Return whether the latest completed review requests another revision."""

    completed = completed_planning_phase_attempts(attempts, phase)
    return bool(
        completed
        and (completed[-1].result or {}).get("decision") == "request_changes"
    )


def planning_attempt_result(attempt: PlanningAttempt, *, default: str) -> str:
    """Return the durable summary used for retained progress history."""

    return attempt.summary or str((attempt.result or {}).get("title") or default)


def planning_attempt_duration(attempt: PlanningAttempt) -> float | None:
    """Return a completed attempt's non-negative authoritative duration."""

    if attempt.finished_at is None:
        return None
    return max((attempt.finished_at - attempt.started_at).total_seconds(), 0.0)


class PlanningProgress(Protocol):
    """Provider-neutral activity operations used by a planning turn."""

    def activity(self, stage_key: str, activity: AgentActivity) -> object: ...

    def child_activity(
        self, stage_key: str, child_key: str, activity: AgentActivity
    ) -> object: ...


class DurablePlanningTurns:
    """Create, recover, execute, and validate durable planning turns."""

    def __init__(
        self,
        repository: Repository,
        borg: Borg,
        store: SqliteStore,
        agent: AgentAdapter | SelectedAgent,
        *,
        role: str,
        model: str,
        artifact_dir: Path,
        error_factory: ErrorFactory,
        cancelled_error_factory: ErrorFactory,
        cancel: CancellationToken | None = None,
        progress: PlanningProgress | None = None,
        stage_key: str | None = None,
        child_key: str | None = None,
        dirty_borg_documents: Sequence[Path] = (),
        worktrees_root: Path | None = None,
    ) -> None:
        if store.get_repository(repository.id) != repository:
            raise ValueError("repository must already be present in the supplied store")
        if borg.repository_id != repository.id or store.get_borg(borg.id) != borg:
            raise ValueError(
                "Borg must already belong to the supplied repository and store"
            )
        if progress is None and (stage_key is not None or child_key is not None):
            raise ValueError("planning progress keys require a progress reporter")
        if progress is not None and stage_key is None:
            raise ValueError("planning progress requires a stage key")
        self.repository = repository
        self.borg_id = borg.id
        self.store = store
        self.agent = agent
        self.role = role
        self.model = model
        self.artifact_dir = Path(artifact_dir).resolve()
        self.error_factory = error_factory
        self.cancelled_error_factory = cancelled_error_factory
        self.cancel = cancel
        self.progress = progress
        self.stage_key = stage_key
        self.child_key = child_key
        self.dirty_borg_documents = tuple(dirty_borg_documents)
        self.worktrees_root = worktrees_root

    def run(
        self,
        *,
        phase: str,
        round_number: int,
        schema: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        current_plan: str | None = None,
        turn_name: str | None = None,
        request_context: Mapping[str, Any] | None = None,
    ) -> tuple[PlanningAttempt, dict[str, Any]]:
        """Resume or invoke one provider turn while preserving its durable record."""
        label = turn_name or phase
        while True:
            running = next(
                (
                    item
                    for item in reversed(self.attempts(phase))
                    if item.status is PlanningAttemptStatus.RUNNING
                ),
                None,
            )
            if running is None:
                attempt = self._start_attempt(
                    phase, round_number, request_context=request_context
                )
                break

            stale_context = {
                key: value
                for key, value in (request_context or {}).items()
                if running.request.get(key) != value
            }
            if stale_context:
                self.store.complete_planning_attempt(
                    running.id,
                    status=PlanningAttemptStatus.FAILED,
                    summary="interrupted attempt belongs to stale request context",
                )
                round_number = max(round_number, running.round + 1)
                continue

            recovered = self._recover_payload(running, schema)
            if recovered is not None:
                return running, recovered
            refreshed = next(
                item for item in self.attempts(phase) if item.id == running.id
            )
            if refreshed.status is PlanningAttemptStatus.RUNNING:
                attempt = running
                break
            round_number = max(round_number, refreshed.round + 1)

        result_path = self.result_path(attempt)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        try:
            with self.materialized_worktree(current_plan=current_plan) as worktree:
                result = self.agent.run(
                    AgentRunSpec(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        schema=schema,
                        cwd=worktree,
                        model=self.model,
                        allowed_tools=READ_ONLY_API_TOOLS,
                        log_path=result_path.with_suffix(".log"),
                        result_path=result_path,
                        activity_sink=(
                            self._record_activity if self.progress is not None else None
                        ),
                    ),
                    cancel=self.cancel,
                )
        except KeyboardInterrupt:
            self.store.complete_planning_attempt(
                attempt.id,
                status=PlanningAttemptStatus.CANCELLED,
                summary=f"{self.role} run cancelled",
            )
            raise
        except Exception as error:
            raise self.error_factory(
                f"{self.role} {label} turn crashed: {error}"
            ) from error

        if result.status is AgentStatus.CANCELLED:
            self.store.complete_planning_attempt(
                attempt.id,
                status=PlanningAttemptStatus.CANCELLED,
                summary=result.error,
            )
            raise self.cancelled_error_factory(
                result.error or f"{self.role} run cancelled"
            )
        if result.status is not AgentStatus.COMPLETED or result.payload is None:
            self.store.complete_planning_attempt(
                attempt.id,
                status=PlanningAttemptStatus.FAILED,
                result=result.payload,
                summary=result.error,
            )
            raise self.error_factory(
                result.error
                or f"{self.role} {label} returned {result.status.value}"
            )
        try:
            validate_structured_result(result.payload, schema)
        except StructuredResultError as error:
            self.store.complete_planning_attempt(
                attempt.id,
                status=PlanningAttemptStatus.FAILED,
                result=result.payload,
                summary=f"invalid structured result: {error}",
            )
            raise self.error_factory(
                f"{self.role} {label} returned an invalid structured result: {error}"
            ) from error
        return attempt, result.payload

    def attempts(self, phase: str) -> list[PlanningAttempt]:
        """Return the durable attempt history for one planning phase."""
        return [
            item
            for item in self.store.list_planning_attempts(self.borg_id)
            if item.phase == phase
        ]

    def next_round(self, phase: str) -> int:
        """Return the next monotonically increasing attempt round."""
        return max((item.round for item in self.attempts(phase)), default=0) + 1

    def current_borg(self) -> Borg:
        """Load the current durable Borg state for this planning lifecycle."""
        borg = self.store.get_borg(self.borg_id)
        if borg is None:
            raise self.error_factory(f"Borg {self.borg_id} no longer exists")
        return borg

    def transition(self, borg: Borg, state: BorgState) -> Borg:
        """Move an unchanged Borg snapshot to its next durable state."""
        if borg.state is state:
            return borg
        return self.store.compare_and_set_borg_state(
            borg.id,
            expected_state=borg.state,
            expected_version=borg.state_version,
            new_state=state,
        )

    def result_path(self, attempt: PlanningAttempt) -> Path:
        """Resolve the stable result artifact path recorded by an attempt."""
        stored = attempt.request.get("result_path")
        if isinstance(stored, str) and stored != "pending":
            return Path(stored)
        return self.artifact_dir / f"{attempt.id}.json"

    @contextmanager
    def materialized_worktree(
        self, *, current_plan: str | None = None
    ) -> Iterator[Path]:
        """Expose the same committed planning snapshot to turns and validators."""
        with materialize_planning_worktree(
            self.repository,
            self.current_borg(),
            self.store,
            current_plan=current_plan,
            dirty_borg_documents=self.dirty_borg_documents,
            worktrees_root=self.worktrees_root,
            cancel=self.cancel,
        ) as worktree:
            yield worktree

    def validate_plan(self, plan: dict[str, Any]) -> None:
        """Validate a plan against the exact repository snapshot agents inspect."""
        with self.materialized_worktree() as worktree:
            validate_plan(plan, worktree)

    def _record_activity(self, activity: AgentActivity) -> None:
        if self.progress is None or self.stage_key is None:
            return
        if self.child_key is None:
            self.progress.activity(self.stage_key, activity)
            return
        self.progress.child_activity(self.stage_key, self.child_key, activity)

    def _start_attempt(
        self,
        phase: str,
        round_number: int,
        *,
        request_context: Mapping[str, Any] | None = None,
    ) -> PlanningAttempt:
        request = dict(request_context or {})
        if "result_path" in request:
            raise ValueError("planning request context cannot override result_path")
        attempt = PlanningAttempt(
            borg_id=self.borg_id,
            phase=phase,
            round=round_number,
            adapter=self.agent.name,
            model=self.model,
            request={**request, "result_path": "pending"},
        )
        attempt = replace(
            attempt,
            request={
                **request,
                "result_path": str(self.result_path(attempt)),
            },
        )
        self.store.append_planning_attempt(attempt)
        return attempt

    def _recover_payload(
        self, attempt: PlanningAttempt, schema: dict[str, Any]
    ) -> dict[str, Any] | None:
        try:
            payload = json.loads(
                self.result_path(attempt).read_text(encoding="utf-8")
            )
            validate_structured_result(payload, schema)
        except FileNotFoundError:
            return None
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            StructuredResultError,
        ) as error:
            self.store.complete_planning_attempt(
                attempt.id,
                status=PlanningAttemptStatus.FAILED,
                summary=f"unusable interrupted result: {error}",
            )
            return None
        if not isinstance(payload, dict):
            return None
        return payload


__all__ = [
    "DurablePlanningTurns",
    "completed_planning_phase_attempts",
    "current_planning_cycle_attempts",
    "latest_planning_review_requests_changes",
    "planning_attempt_duration",
    "planning_attempt_result",
    "planning_request_change_attempts",
]
