"""Durable Tech Lead review and bounded Architect revision lifecycle."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from betterborg_cli.agent_runtime.base import (
    AgentAdapter,
    CancellationToken,
)
from betterborg_cli.agent_runtime.selection import (
    AgentSelectionError,
    SelectedAgent,
    require_read_only_agent,
    resolve_agent_model,
)
from betterborg_cli.planning.architect import ArchitectCancelled, ArchitectLoop
from betterborg_cli.planning.convergence import assess_convergence, drain_evidence
from betterborg_cli.planning.cycles import current_planning_cycle_id
from betterborg_cli.planning.findings_ledger import (
    REPEATS_SCHEMA,
    RESOLVED_SCHEMA,
    open_findings,
    reconcile_planning_ledger,
)
from betterborg_cli.planning.grants import (
    PLANNING_GRANT_BUDGET,
    GrantAccount,
    assess_grant,
    planning_grant_account,
)
from betterborg_cli.planning.plan_contracts import PlanValidationError
from betterborg_cli.planning.steering import (
    PlanningSteeringTurn,
    SteeringScope,
    SteeringSubject,
    recorded_steering_note,
    steering_verdict,
)
from betterborg_cli.planning.turns import (
    DurablePlanningTurns,
    completed_planning_phase_attempts,
    current_planning_cycle_attempts,
    planning_attempt_duration,
    planning_attempt_result,
    planning_request_change_attempts,
)
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.progress import ChildSpec, RunProgress, StageSpec, StageState
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.store import (
    Borg,
    BorgState,
    PlanningAttempt,
    PlanningAttemptStatus,
    PlanningFinding,
    Repository,
    ReviewAssessment,
    SqliteStore,
)

#: Review rounds a plan gets before every further round is a grant, where its
#: repository configures no minimum of its own.
TECH_REVIEW_ROUND_MINIMUM = 3
_TECH_REVIEW_PHASE = "tech_review"
_ARCHITECT_PLAN_PHASE = "architect_plan"

#: Who a steering note in this loop is written for, and what the argument it
#: reads is about.
_STEERING_SUBJECT = SteeringSubject(
    answerer="The Architect", work="the implementation plan"
)

TECH_LEAD_REVIEW_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "summary", "findings", "resolved"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approve", "request_changes"],
        },
        "summary": {"type": "string", "minLength": 1},
        "resolved": RESOLVED_SCHEMA,
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["severity", "message", "repeats"],
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["blocker", "major", "minor"],
                    },
                    "message": {"type": "string", "minLength": 1},
                    "suggestion": {"type": "string"},
                    "repeats": REPEATS_SCHEMA,
                },
            },
        },
    },
}

_TECH_LEAD_SYSTEM_PROMPT = """You are the Tech Lead reviewing an Architect plan.
Inspect the materialized repository, confirmed PRD, current plan, the open
findings the plan still has to answer, and the complete finding history. Verify
the plan against the actual code and return approve only when it is ready for
human approval. Otherwise return concise, actionable findings for the
Architect. Account for the open findings you were given: list in resolved the
id of every one the current plan closes, and on each finding of your own set
repeats to the id of the open finding it raises again, or null when the
objection is new. Both are always required, so a review that closes nothing
sends an empty list and a new finding sends null. An open finding you neither
resolve nor repeat stays open. Betterborg performs the delivery around the
plan: branching, worktrees, commits, review, merge, and the repository's own
checks. The plan covers the product change only, so never hold it to work
Betterborg already does. A phase name is two digits then lowercase words of
letters and digits, all joined by single hyphens and at most 32 characters, as
in 01-schema-migration, so never ask for a name the Architect cannot use. Do
not modify files. Return only the required JSON object.
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


def tech_lead_grant_account(
    store: SqliteStore, borg_id: UUID
) -> GrantAccount:
    """Account for what the current cycle's review rounds cost this plan."""

    return planning_grant_account(
        store,
        borg_id,
        loop=_TECH_REVIEW_PHASE,
        cycle_id=current_planning_cycle_id(store, borg_id),
    )


def _review_round_sentence(review_round: int) -> str:
    """Name the round and nothing else.

    No loop knows whether a round is its last: that depends on findings the
    round has not produced yet. And a budget number in a reviewer's prompt
    reads as a deadline it is being held to, so where the loop stands belongs
    to the operator's surfaces.
    """

    return f"This is Tech Lead review round {review_round}."


class TechLeadLoop:
    """Review a validated plan, revising it while its grant budget holds."""

    def __init__(
        self,
        repository: Repository,
        borg: Borg,
        store: SqliteStore,
        agent: AgentAdapter | SelectedAgent,
        *,
        architect_agent: AgentAdapter | SelectedAgent | None = None,
        steering_agent: AgentAdapter | SelectedAgent | None = None,
        io: InteractiveIO,
        unattended: bool = False,
        review_rounds: int = TECH_REVIEW_ROUND_MINIMUM,
        grant_budget: int = PLANNING_GRANT_BUDGET,
        artifact_dir: Path | None = None,
        model: str | None = None,
        architect_model: str | None = None,
        steering_model: str | None = None,
        cancel: CancellationToken | None = None,
        progress: RunProgress | None = None,
        dirty_borg_documents: Sequence[Path] = (),
        worktrees_root: Path | None = None,
    ) -> None:
        if cancel is not None and cancel.is_set():
            raise TechLeadCancelled("Tech Lead run cancelled")
        # Zero buys nothing and is legal; below zero would stop the loop short
        # of the minimum it was told to run.
        if not isinstance(grant_budget, int) or grant_budget < 0:
            raise TechLeadError(
                "Tech Lead grant budget must be a whole number of at least 0"
            )
        architect = architect_agent or agent
        steering = steering_agent or agent
        require_read_only_agent(
            agent, role="Tech Lead", error_factory=TechLeadError
        )
        require_read_only_agent(
            architect, role="Architect", error_factory=TechLeadError
        )
        require_read_only_agent(
            steering, role="Steering", error_factory=TechLeadError
        )
        try:
            resolved_model = resolve_agent_model(agent, model)
            resolved_architect_model = resolve_agent_model(
                architect, architect_model
            )
            resolved_steering_model = resolve_agent_model(
                steering, steering_model
            )
        except AgentSelectionError as error:
            raise TechLeadError(str(error)) from error

        paths = RepoPaths.discover(repository.root, cancel=cancel)
        if paths.root != repository.root:
            raise ValueError("repository root does not match its discovered Git root")
        self.repository = repository
        self.borg_id = borg.id
        self.store = store
        self.agent = agent
        self.architect_agent = architect
        self.steering_agent = steering
        self.io = io
        self.unattended = unattended
        self.review_rounds = review_rounds
        self.grant_budget = grant_budget
        self.artifact_dir = Path(
            artifact_dir or paths.artifacts_dir / "planning" / str(borg.id)
        ).resolve()
        self.model = resolved_model
        self.architect_model = resolved_architect_model
        self.steering_model = resolved_steering_model
        self.cancel = cancel
        self.progress = progress
        self.dirty_borg_documents = tuple(dirty_borg_documents)
        self.worktrees_root = worktrees_root
        if progress is not None:
            if "architect" not in progress.stages:
                progress.declare(StageSpec("architect", "Architect"))
            if "tech-lead" not in progress.stages:
                progress.declare(StageSpec("tech-lead", "Tech Lead"))
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
            progress=progress,
            stage_key="tech-lead" if progress is not None else None,
            dirty_borg_documents=dirty_borg_documents,
            worktrees_root=worktrees_root,
        )

    def run(self) -> TechLeadResult:
        """Continue reviewing and revising until approved or out of budget."""
        try:
            self._seed_architect_progress()
            self._declare_revision_progress()
            terminal = self._terminal_result()
            if terminal is not None:
                self._seed_revision_progress()
                self._seed_tech_lead_progress(terminal.attempt)
                return terminal

            if self.progress is not None:
                self._seed_revision_progress()
                self.progress.start("tech-lead")
            result = self._run()
            if self.progress is not None:
                self.progress.complete(
                    "tech-lead", result.attempt.summary or "review complete"
                )
            return result
        except (ArchitectCancelled, TechLeadCancelled, KeyboardInterrupt) as error:
            self._reconcile_progress(str(error), stopped=True)
            raise
        except Exception as error:
            self._reconcile_progress(
                str(error),
                stopped=self.cancel is not None and self.cancel.is_set(),
            )
            raise

    def _run(self) -> TechLeadResult:
        """Execute the active Tech Lead parent through all revision cycles."""

        while True:
            borg = self._turns.current_borg()
            if borg.state in {
                BorgState.ARCHITECT_WORKING,
                BorgState.ARCHITECT_AWAITING_ANSWERS,
            }:
                child_key = self._active_revision_key()
                if child_key is None:
                    raise TechLeadError(
                        "Architect revision requires a rejected Tech Lead attempt"
                    )
                # Written before the revision starts, because the note has to
                # reach the Architect's constructor before that loop builds the
                # prompt it joins.
                note = self._steering_note(borg)
                self._start_revision_progress(child_key)
                revised = ArchitectLoop(
                    self.repository,
                    borg,
                    self.store,
                    self.architect_agent,
                    io=self.io,
                    unattended=self.unattended,
                    artifact_dir=self.artifact_dir,
                    model=self.architect_model,
                    cancel=self.cancel,
                    progress=self.progress,
                    stage_key="tech-lead",
                    child_key=child_key,
                    steering_note=note,
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
                declared = self._findings(payload, attempt, review_round)
            except TechLeadError as error:
                self.store.complete_planning_attempt(
                    attempt.id,
                    status=PlanningAttemptStatus.FAILED,
                    result=payload,
                    summary=str(error),
                )
                raise

            approved = payload["decision"] == "approve"
            # Reconciled before the decision is taken, and written with it, so
            # no reader is a round behind the objection it is judging.
            findings = [finding for finding, _ in declared]
            cycle_id = current_planning_cycle_id(self.store, self.borg_id)
            ledger = reconcile_planning_ledger(
                self.store.list_planning_ledger_findings(
                    self.borg_id, cycle_id=cycle_id
                ),
                findings=declared,
                resolved=payload["resolved"],
                attempt_id=attempt.id,
                cycle_id=cycle_id,
                review_round=review_round,
                approved=approved,
            )

            # Two questions of the same round, and they are not the same
            # question: what the round cost decides whether the loop gets
            # another, and whether its argument is closing in is a judgement on
            # the shape of it that the record keeps.
            convergence = assess_convergence(ledger)
            snapshot = len(open_findings(ledger))
            grant = assess_grant(
                review_round=review_round,
                minimum=self.review_rounds,
                budget=self.grant_budget,
                snapshot=snapshot,
                recorded=self.store.list_review_assessments(
                    self.borg_id, loop=_TECH_REVIEW_PHASE, cycle_id=cycle_id
                ),
            )
            assessment = ReviewAssessment(
                borg_id=self.borg_id,
                loop=_TECH_REVIEW_PHASE,
                cycle_id=cycle_id,
                attempt_id=attempt.id,
                round=review_round,
                minimum=self.review_rounds,
                converging=convergence.converging,
                open_findings=snapshot,
                refunded=grant.refunded,
                evidence=drain_evidence(convergence),
            )

            if approved:
                next_state = BorgState.PLAN_APPROVAL_PENDING
            elif grant.continues:
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
                self.store.record_planning_ledger_findings(ledger)
                self.store.record_review_assessment(assessment)
                borg = self._turns.transition(borg, next_state)

            if next_state is not BorgState.ARCHITECT_WORKING:
                return TechLeadResult(borg=borg, plan=plan, attempt=completed)

            self._declare_revision_progress()

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
                "Read .betterborg/state/planning/context/manifest.json and all "
                "referenced evidence. Review the complete current plan. "
                + _review_round_sentence(review_round)
            ),
            current_plan=json.dumps(plan, indent=2, sort_keys=True),
            turn_name="review",
        )

    def _steering_note(self, borg: Borg) -> str | None:
        """Return the note this granted revision runs on, or nothing.

        A revision that follows a converging verdict, and one leading into a
        round still inside the minimum, get the prompt they would have had: the
        findings they have to answer are already in front of them.
        """

        cycle_id = current_planning_cycle_id(self.store, self.borg_id)
        scope = SteeringScope(loop=_TECH_REVIEW_PHASE, cycle_id=cycle_id)
        verdict = steering_verdict(
            self.store.list_review_assessments(
                self.borg_id, loop=_TECH_REVIEW_PHASE, cycle_id=cycle_id
            )
        )
        if verdict is None:
            return None
        recorded = recorded_steering_note(
            self.store, self.borg_id, scope=scope, round_number=verdict.round
        )
        if recorded is not None:
            return recorded.note
        child_key = f"steering:{verdict.round}"
        self._start_steering_progress(child_key, verdict.round)
        note = PlanningSteeringTurn(
            self.repository,
            borg,
            self.store,
            self.steering_agent,
            scope=scope,
            subject=_STEERING_SUBJECT,
            model=self.steering_model,
            artifact_dir=self.artifact_dir,
            error_factory=TechLeadError,
            cancelled_error_factory=TechLeadCancelled,
            cancel=self.cancel,
            progress=self.progress,
            stage_key="tech-lead" if self.progress is not None else None,
            child_key=child_key if self.progress is not None else None,
            dirty_borg_documents=self.dirty_borg_documents,
            worktrees_root=self.worktrees_root,
        ).note(
            verdict=verdict,
            ledger=self.store.list_planning_ledger_findings(
                self.borg_id, cycle_id=cycle_id
            ),
            summaries=self._round_summaries(),
        )
        self._complete_steering_progress(child_key)
        return note

    def _round_summaries(self) -> list[tuple[int, str]]:
        """Pair each completed review with its own account of what it decided."""

        return [
            (number, summary)
            for number, attempt in enumerate(self._completed_reviews(), start=1)
            if (summary := (attempt.summary or "").strip())
        ]

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
    ) -> list[tuple[PlanningFinding, str | None]]:
        """Build this round's findings, each beside the objection it restates."""

        summary = str(payload["summary"])
        if not summary.strip():
            raise TechLeadError("Tech Lead summary must not be blank")
        raw_findings = list(payload["findings"])
        if payload["decision"] == "request_changes" and not raw_findings:
            raise TechLeadError("Tech Lead request_changes must include findings")
        declared: list[tuple[PlanningFinding, str | None]] = []
        for item in raw_findings:
            message = str(item["message"])
            if not message.strip():
                raise TechLeadError("Tech Lead finding messages must not be blank")
            suggestion = item.get("suggestion")
            finding = PlanningFinding(
                borg_id=self.borg_id,
                attempt_id=attempt.id,
                round=review_round,
                severity=str(item["severity"]),
                message=message.strip(),
                suggestion=(
                    str(suggestion).strip() if suggestion is not None else None
                ),
            )
            declared.append((finding, item["repeats"]))
        if payload["decision"] == "approve" and any(
            finding.severity in {"blocker", "major"} for finding, _ in declared
        ):
            raise TechLeadError(
                "Tech Lead cannot approve while blocker or major findings remain"
            )
        return declared

    def _terminal_result(self) -> TechLeadResult | None:
        borg = self._turns.current_borg()
        if borg.state not in {
            BorgState.PLAN_APPROVAL_PENDING,
            BorgState.BLOCKED,
        }:
            return None
        completed_reviews = self._completed_reviews()
        attempt = completed_reviews[-1] if completed_reviews else None
        if attempt is None:
            return None
        decision = (attempt.result or {}).get("decision")
        if (
            borg.state is BorgState.PLAN_APPROVAL_PENDING
            and decision != "approve"
        ) or (borg.state is BorgState.BLOCKED and decision != "request_changes"):
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
        return completed_planning_phase_attempts(
            self._cycle_attempts(), _TECH_REVIEW_PHASE
        )

    def _cycle_attempts(self) -> list[PlanningAttempt]:
        return current_planning_cycle_attempts(self.store, self.borg_id)

    def _initial_architect_plan(self) -> PlanningAttempt | None:
        return next(
            (
                item
                for item in self._cycle_attempts()
                if item.phase == _ARCHITECT_PLAN_PHASE
                and item.status is PlanningAttemptStatus.COMPLETED
                and item.result is not None
                and not item.result.get("open_questions")
            ),
            None,
        )

    def _revision_reviews(self) -> list[PlanningAttempt]:
        return planning_request_change_attempts(
            self._cycle_attempts(), _TECH_REVIEW_PHASE
        )

    @staticmethod
    def _revision_key(review: PlanningAttempt) -> str:
        return f"architect-revision:{review.id}"

    def _revision_plan(self, review: PlanningAttempt) -> PlanningAttempt | None:
        attempts = self._cycle_attempts()
        review_index = next(
            index for index, item in enumerate(attempts) if item.id == review.id
        )
        for item in attempts[review_index + 1 :]:
            if item.phase == _TECH_REVIEW_PHASE:
                return None
            if (
                item.phase == _ARCHITECT_PLAN_PHASE
                and item.status is PlanningAttemptStatus.COMPLETED
                and item.result is not None
                and not item.result.get("open_questions")
            ):
                return item
        return None

    def _active_revision_key(self) -> str | None:
        for review in reversed(self._revision_reviews()):
            if self._revision_plan(review) is None:
                return self._revision_key(review)
        return None

    def _revisions_with_work(self) -> set[str]:
        """Identify the rejections a revision belongs to, run or under way.

        The rejection that blocked is followed by neither, and only the Borg's
        state tells it apart from one whose revision has not finished. A child
        declared for it would stay pending forever, and a pending child refuses
        to let its parent be seeded, so reconstructing a blocked plan would
        raise instead of reporting what the record holds.
        """
        reviews = self._revision_reviews()
        revising = self._turns.current_borg().state in {
            BorgState.ARCHITECT_WORKING,
            BorgState.ARCHITECT_AWAITING_ANSWERS,
        }
        return {
            review.id
            for index, review in enumerate(reviews)
            if self._revision_plan(review) is not None
            or (revising and index == len(reviews) - 1)
        }

    def _declare_revision_progress(self) -> None:
        if self.progress is None:
            return
        with_work = self._revisions_with_work()
        for number, review in enumerate(self._revision_reviews(), start=1):
            if review.id not in with_work:
                continue
            key = self._revision_key(review)
            if key not in self.progress.stages["tech-lead"].children:
                self.progress.declare_child(
                    "tech-lead", ChildSpec(key, f"Architect revision {number}")
                )

    def _start_revision_progress(self, child_key: str) -> None:
        if self.progress is None:
            return
        child = self.progress.stages["tech-lead"].children[child_key]
        if child.state is StageState.PENDING:
            self.progress.start_child("tech-lead", child_key)

    def _start_steering_progress(self, child_key: str, review_round: int) -> None:
        """Show a steering turn as itself rather than as the reviewer's.

        A child of its own, so a long note does not read as a stalled review.
        """

        if self.progress is None:
            return
        if child_key not in self.progress.stages["tech-lead"].children:
            self.progress.declare_child(
                "tech-lead",
                ChildSpec(child_key, f"Steering note {review_round}"),
            )
        child = self.progress.stages["tech-lead"].children[child_key]
        if child.state is StageState.PENDING:
            self.progress.start_child("tech-lead", child_key)

    def _complete_steering_progress(self, child_key: str) -> None:
        """Reconcile the steering child on every path, not only the fallbacks.

        The turn machinery emits into a child and never touches its lifecycle,
        so a note that succeeded leaves its child running exactly as a note
        that failed does — and a stage refuses to complete while any child of
        it is still running.
        """

        if self.progress is None:
            return
        child = self.progress.stages["tech-lead"].children[child_key]
        if child.state is StageState.RUNNING:
            self.progress.complete_child(
                "tech-lead", child_key, "steering note ready"
            )

    def _seed_revision_progress(self) -> None:
        if self.progress is None:
            return
        for review in self._revision_reviews():
            plan = self._revision_plan(review)
            if plan is None:
                continue
            key = self._revision_key(review)
            child = self.progress.stages["tech-lead"].children[key]
            if child.state is StageState.PENDING:
                self.progress.seed_child_completed(
                    "tech-lead",
                    key,
                    planning_attempt_result(plan, default="plan ready"),
                    planning_attempt_duration(plan),
                )

    def _seed_architect_progress(self) -> None:
        if self.progress is None:
            return
        plan = self._initial_architect_plan()
        record = self.progress.stages["architect"]
        if plan is not None and record.state is StageState.PENDING:
            self.progress.seed_completed(
                "architect",
                planning_attempt_result(plan, default="plan ready"),
                planning_attempt_duration(plan),
            )

    def _seed_tech_lead_progress(self, attempt: PlanningAttempt) -> None:
        if self.progress is None:
            return
        record = self.progress.stages["tech-lead"]
        if record.state is StageState.PENDING:
            self.progress.seed_completed(
                "tech-lead",
                planning_attempt_result(attempt, default="review complete"),
                planning_attempt_duration(attempt),
            )

    def _finish_progress(self, result: str, *, stopped: bool) -> None:
        if self.progress is None:
            return
        for child in self.progress.stages["tech-lead"].children.values():
            if child.state is not StageState.RUNNING:
                continue
            if stopped:
                self.progress.stop_child("tech-lead", child.key, result)
            else:
                self.progress.fail_child("tech-lead", child.key, result)
        if self.progress.stages["tech-lead"].state is StageState.RUNNING:
            if stopped:
                self.progress.stop("tech-lead", result)
            else:
                self.progress.fail("tech-lead", result)

    def _reconcile_progress(self, result: str, *, stopped: bool) -> None:
        if self.progress is None:
            return
        record = self.progress.stages["tech-lead"]
        if record.state is not StageState.RUNNING:
            return
        terminal = self._terminal_result()
        if terminal is not None:
            self.progress.complete(
                "tech-lead", terminal.attempt.summary or "review complete"
            )
        else:
            self._finish_progress(result, stopped=stopped)
