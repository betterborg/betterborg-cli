"""Durable Tech Lead review and bounded Architect revision lifecycle."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from betterborg_cli.agent_runtime.base import (
    AgentAdapter,
    CancellationToken,
)
from betterborg_cli.agent_runtime.selection import (
    AgentSelectionError,
    SelectedAgent,
    resolve_agent_model,
)
from betterborg_cli.planning.architect import ArchitectLoop
from betterborg_cli.planning.plan_contracts import PlanValidationError
from betterborg_cli.planning.turns import (
    DurablePlanningTurns,
    require_read_only_agent,
)
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.store import (
    Borg,
    BorgState,
    PlanningAttempt,
    PlanningAttemptStatus,
    PlanningFinding,
    Repository,
    SqliteStore,
)

TECH_REVIEW_ROUND_CAP = 3
_TECH_REVIEW_PHASE = "tech_review"
_ARCHITECT_PLAN_PHASE = "architect_plan"

TECH_LEAD_REVIEW_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "summary", "findings"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approve", "request_changes"],
        },
        "summary": {"type": "string", "minLength": 1},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["severity", "message"],
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["blocker", "major", "minor"],
                    },
                    "message": {"type": "string", "minLength": 1},
                    "suggestion": {"type": "string"},
                },
            },
        },
    },
}

_TECH_LEAD_SYSTEM_PROMPT = """You are the Tech Lead reviewing an Architect plan.
Inspect the materialized repository, confirmed PRD, current plan, and complete
finding history. Verify the plan against the actual code and return approve only
when it is ready for human approval. Otherwise return concise, actionable
findings for the Architect. Do not modify files. Return only the required JSON
object.
"""


class TechLeadError(RuntimeError):
    """Raised when the review lifecycle cannot reach a durable outcome."""


class TechLeadCancelled(TechLeadError):
    """Raised after preserving enough review state to resume later."""


@dataclass(frozen=True, slots=True)
class TechLeadResult:
    """The latest reviewed plan and its durable lifecycle state."""

    borg: Borg
    plan: dict[str, Any]
    attempt: PlanningAttempt


class TechLeadLoop:
    """Review a validated plan, revising it at most twice before blocking."""

    def __init__(
        self,
        repository: Repository,
        borg: Borg,
        store: SqliteStore,
        agent: AgentAdapter | SelectedAgent,
        *,
        architect_agent: AgentAdapter | SelectedAgent | None = None,
        io: InteractiveIO,
        artifact_dir: Path | None = None,
        model: str | None = None,
        architect_model: str | None = None,
        cancel: CancellationToken | None = None,
        dirty_borg_documents: Sequence[Path] = (),
        worktrees_root: Path | None = None,
    ) -> None:
        architect = architect_agent or agent
        require_read_only_agent(
            agent, role="Tech Lead", error_factory=TechLeadError
        )
        require_read_only_agent(
            architect, role="Architect", error_factory=TechLeadError
        )
        try:
            resolved_model = resolve_agent_model(agent, model)
            resolved_architect_model = resolve_agent_model(
                architect, architect_model
            )
        except AgentSelectionError as error:
            raise TechLeadError(str(error)) from error

        paths = RepoPaths.discover(repository.root)
        if paths.root != repository.root:
            raise ValueError("repository root does not match its discovered Git root")
        self.repository = repository
        self.borg_id = borg.id
        self.store = store
        self.agent = agent
        self.architect_agent = architect
        self.io = io
        self.artifact_dir = Path(
            artifact_dir or paths.artifacts_dir / "planning" / str(borg.id)
        ).resolve()
        self.model = resolved_model
        self.architect_model = resolved_architect_model
        self.cancel = cancel
        self.dirty_borg_documents = tuple(dirty_borg_documents)
        self.worktrees_root = worktrees_root
        self._turns = DurablePlanningTurns(
            repository,
            borg,
            store,
            agent,
            role="Tech Lead",
            model=resolved_model,
            artifact_dir=self.artifact_dir,
            error_factory=TechLeadError,
            cancelled_error_factory=TechLeadCancelled,
            cancel=cancel,
            dirty_borg_documents=dirty_borg_documents,
            worktrees_root=worktrees_root,
        )

    def run(self) -> TechLeadResult:
        """Continue reviewing and revising until approved or review-capped."""
        terminal = self._terminal_result()
        if terminal is not None:
            return terminal

        while True:
            borg = self._turns.current_borg()
            if borg.state in {
                BorgState.ARCHITECT_WORKING,
                BorgState.ARCHITECT_AWAITING_ANSWERS,
            }:
                revised = ArchitectLoop(
                    self.repository,
                    borg,
                    self.store,
                    self.architect_agent,
                    io=self.io,
                    artifact_dir=self.artifact_dir,
                    model=self.architect_model,
                    cancel=self.cancel,
                    dirty_borg_documents=self.dirty_borg_documents,
                    worktrees_root=self.worktrees_root,
                ).run()
                if revised.borg.state is not BorgState.TECH_REVIEW_WORKING:
                    raise TechLeadError(
                        "Architect revision did not return to Tech Lead"
                    )
                continue
            if borg.state is not BorgState.TECH_REVIEW_WORKING:
                raise TechLeadError(
                    f"Borg {borg.name!r} cannot run Tech Lead from state "
                    f"{borg.state.value!r}"
                )
            if self.cancel is not None and self.cancel.is_set():
                raise TechLeadCancelled("Tech Lead run cancelled")

            plan = self._validated_handoff()
            review_round = len(self._completed_reviews()) + 1
            attempt, payload = self._run_turn(
                round_number=self._turns.next_round(_TECH_REVIEW_PHASE),
                review_round=review_round,
                plan=plan,
            )
            try:
                findings = self._findings(payload, attempt, review_round)
            except TechLeadError as error:
                self.store.complete_planning_attempt(
                    attempt.id,
                    status=PlanningAttemptStatus.FAILED,
                    result=payload,
                    summary=str(error),
                )
                raise

            decision = payload["decision"]
            if decision == "approve":
                next_state = BorgState.PLAN_APPROVAL_PENDING
            elif review_round < TECH_REVIEW_ROUND_CAP:
                next_state = BorgState.ARCHITECT_WORKING
            else:
                next_state = BorgState.BLOCKED

            with self.store.transaction():
                completed = self.store.complete_planning_attempt(
                    attempt.id,
                    status=PlanningAttemptStatus.COMPLETED,
                    result=payload,
                    summary=str(payload["summary"]).strip(),
                )
                for finding in findings:
                    self.store.append_planning_finding(finding)
                borg = self._turns.transition(borg, next_state)

            if next_state is not BorgState.ARCHITECT_WORKING:
                return TechLeadResult(borg=borg, plan=plan, attempt=completed)

    def _run_turn(
        self,
        *,
        round_number: int,
        review_round: int,
        plan: dict[str, Any],
    ) -> tuple[PlanningAttempt, dict[str, Any]]:
        return self._turns.run(
            phase=_TECH_REVIEW_PHASE,
            round_number=round_number,
            schema=TECH_LEAD_REVIEW_SCHEMA,
            system_prompt=_TECH_LEAD_SYSTEM_PROMPT,
            user_prompt=(
                "Read .borg/state/planning/context/manifest.json and all "
                "referenced evidence. Review the complete current plan. "
                f"This is Tech Lead review round {review_round} of "
                f"{TECH_REVIEW_ROUND_CAP}."
            ),
            current_plan=json.dumps(plan, indent=2, sort_keys=True),
            turn_name="review",
        )

    def _validated_handoff(self) -> dict[str, Any]:
        attempts = self.store.list_planning_attempts(self.borg_id)
        plan_index = next(
            (
                index
                for index in range(len(attempts) - 1, -1, -1)
                if attempts[index].phase == _ARCHITECT_PLAN_PHASE
                and attempts[index].status is PlanningAttemptStatus.COMPLETED
                and attempts[index].result is not None
            ),
            None,
        )
        if plan_index is None:
            raise TechLeadError("Tech Lead review requires a completed Architect plan")
        review_index = next(
            (
                index
                for index in range(len(attempts) - 1, -1, -1)
                if attempts[index].phase == _TECH_REVIEW_PHASE
                and attempts[index].status is PlanningAttemptStatus.COMPLETED
            ),
            None,
        )
        if review_index is not None and review_index > plan_index:
            raise TechLeadError(
                "Tech Lead review requires an Architect revision after the latest "
                "completed review"
            )
        plan_attempt = attempts[plan_index]
        plan = plan_attempt.result or {}
        try:
            self._turns.validate_plan(plan)
        except PlanValidationError as error:
            raise TechLeadError(
                f"Architect handoff failed deterministic validation: {error}"
            ) from error
        return plan

    def _findings(
        self,
        payload: dict[str, Any],
        attempt: PlanningAttempt,
        review_round: int,
    ) -> list[PlanningFinding]:
        summary = str(payload["summary"])
        if not summary.strip():
            raise TechLeadError("Tech Lead summary must not be blank")
        raw_findings = list(payload["findings"])
        if payload["decision"] == "request_changes" and not raw_findings:
            raise TechLeadError("Tech Lead request_changes must include findings")
        findings: list[PlanningFinding] = []
        for item in raw_findings:
            message = str(item["message"])
            if not message.strip():
                raise TechLeadError("Tech Lead finding messages must not be blank")
            suggestion = item.get("suggestion")
            findings.append(
                PlanningFinding(
                    borg_id=self.borg_id,
                    attempt_id=attempt.id,
                    round=review_round,
                    severity=str(item["severity"]),
                    message=message.strip(),
                    suggestion=(
                        str(suggestion).strip() if suggestion is not None else None
                    ),
                )
            )
        if payload["decision"] == "approve" and any(
            finding.severity in {"blocker", "major"} for finding in findings
        ):
            raise TechLeadError(
                "Tech Lead cannot approve while blocker or major findings remain"
            )
        return findings

    def _terminal_result(self) -> TechLeadResult | None:
        borg = self._turns.current_borg()
        if borg.state not in {
            BorgState.PLAN_APPROVAL_PENDING,
            BorgState.BLOCKED,
        }:
            return None
        attempt = next(
            (
                item
                for item in reversed(self._review_attempts())
                if item.status is PlanningAttemptStatus.COMPLETED
            ),
            None,
        )
        if attempt is None:
            return None
        decision = (attempt.result or {}).get("decision")
        if (
            borg.state is BorgState.PLAN_APPROVAL_PENDING
            and decision != "approve"
        ) or (
            borg.state is BorgState.BLOCKED
            and (
                decision != "request_changes"
                or len(self._completed_reviews()) < TECH_REVIEW_ROUND_CAP
            )
        ):
            return None
        plan_attempt = next(
            (
                item
                for item in reversed(
                    self.store.list_planning_attempts(self.borg_id)
                )
                if item.phase == _ARCHITECT_PLAN_PHASE
                and item.status is PlanningAttemptStatus.COMPLETED
                and item.result is not None
            ),
            None,
        )
        if plan_attempt is None:
            return None
        return TechLeadResult(
            borg=borg,
            plan=plan_attempt.result or {},
            attempt=attempt,
        )

    def _completed_reviews(self) -> list[PlanningAttempt]:
        return [
            item
            for item in self._review_attempts()
            if item.status is PlanningAttemptStatus.COMPLETED
        ]

    def _review_attempts(self) -> list[PlanningAttempt]:
        attempts = self._turns.attempts(_TECH_REVIEW_PHASE)
        change_requests = self.store.list_plan_change_requests(self.borg_id)
        if not change_requests:
            return attempts
        cycle_started_at = change_requests[-1].created_at
        return [
            attempt for attempt in attempts if attempt.started_at >= cycle_started_at
        ]
