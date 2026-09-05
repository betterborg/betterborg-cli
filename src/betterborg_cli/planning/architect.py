"""Resumable, terminal-native Architect question and planning loop."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
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
    require_read_only_agent,
    resolve_agent_model,
)
from betterborg_cli.planning.plan_contracts import PlanValidationError
from betterborg_cli.planning.turns import (
    DurablePlanningTurns,
    planning_attempt_duration,
    planning_attempt_result,
)
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.progress import ChildSpec, RunProgress, StageSpec, StageState
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.store import (
    Borg,
    BorgState,
    PlanningAttempt,
    PlanningAttemptStatus,
    PlanningQuestion,
    Repository,
    SqliteStore,
)

ARCHITECT_QUESTION_ROUND_CAP = 3
#: Plan turns spent proving one plan against the deterministic checks that
#: follow its schema, the rejected turn included. A failed contract is a
#: property of one sampled plan exactly as a missed schema is, so it carries
#: the same budget as the schema retry it sits one layer above.
ARCHITECT_PLAN_CONTRACT_ROUND_CAP = 3
_QUESTIONS_PHASE = "architect_questions"
_ANSWERS_PHASE = "architect_answers"
_PLAN_PHASE = "architect_plan"

ARCHITECT_QUESTIONS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["decision"],
    "properties": {
        "decision": {"type": "string", "enum": ["ask_more", "ready_to_plan"]},
        "questions": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "question"],
                "properties": {
                    "id": {"type": "string", "pattern": "^q[1-9][0-9]?$"},
                    "question": {"type": "string", "minLength": 1},
                    "why": {"type": "string"},
                    "hint": {"type": "string"},
                },
            },
        },
    },
}

ARCHITECT_ANSWERS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["answers"],
    "properties": {
        "answers": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["q_id", "answer"],
                "properties": {
                    "q_id": {"type": "string", "minLength": 1},
                    "answer": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}

_FILE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["path", "role"],
    "properties": {
        "path": {"type": "string", "minLength": 1},
        "role": {"type": "string", "enum": ["new", "modified", "deleted", "read"]},
        "repo": {"type": "string", "minLength": 1},
        "description": {"type": "string"},
    },
}
_CONTRACT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "spec"],
    "properties": {
        "kind": {
            "type": "string",
            "enum": [
                "db_migration",
                "api_endpoint",
                "type",
                "function_signature",
                "config",
                "event",
                "other",
            ],
        },
        "spec": {"type": "string", "minLength": 1},
        "repo": {"type": "string", "minLength": 1},
    },
}
_NONEMPTY_STRINGS = {"type": "array", "items": {"type": "string", "minLength": 1}}

ARCHITECT_PLAN_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "summary", "overall_approach", "phases"],
    "properties": {
        "title": {"type": "string", "minLength": 1, "maxLength": 120},
        "repositories": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "role": {"type": "string", "enum": ["primary", "secondary"]},
                },
            },
        },
        "summary": {"type": "string", "minLength": 1},
        "overall_approach": {"type": "string", "minLength": 1},
        "phases": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "name",
                    "title",
                    "goal",
                    "technical_approach",
                    "files_touched",
                    "test_strategy",
                    "acceptance_criteria",
                    "deliverables",
                ],
                "properties": {
                    "name": {
                        "type": "string",
                        "pattern": "^[0-9]{2}-[a-z0-9-]+$",
                        "minLength": 4,
                        "maxLength": 32,
                    },
                    "title": {"type": "string", "minLength": 1, "maxLength": 120},
                    "goal": {"type": "string", "minLength": 1},
                    "technical_approach": {"type": "string", "minLength": 1},
                    "repositories": _NONEMPTY_STRINGS,
                    "files_touched": {
                        "type": "array",
                        "minItems": 1,
                        "items": _FILE_SCHEMA,
                    },
                    "contracts": {"type": "array", "items": _CONTRACT_SCHEMA},
                    "test_strategy": {"type": "string", "minLength": 1},
                    "acceptance_criteria": {**_NONEMPTY_STRINGS, "minItems": 1},
                    "dependencies_on": {
                        "type": "array",
                        "items": {"type": "string", "pattern": "^[0-9]{2}-[a-z0-9-]+$"},
                    },
                    "deliverables": {**_NONEMPTY_STRINGS, "minItems": 1},
                    "constraints": _NONEMPTY_STRINGS,
                    "risks": _NONEMPTY_STRINGS,
                },
            },
        },
        "code_pointers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "why"],
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "why": {"type": "string", "minLength": 1},
                },
            },
        },
        "risks": _NONEMPTY_STRINGS,
        "open_questions": _NONEMPTY_STRINGS,
        # Derived from the durable question rounds the Architect answered
        # itself, so a reader of the plan meets every assumption it rests on.
        "assumptions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["question", "assumption"],
                "properties": {
                    "question": {"type": "string", "minLength": 1},
                    "assumption": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}

_QUESTIONS_SYSTEM_PROMPT = """You are the Architect for this project. Inspect the
materialized repository and planning context before deciding. Ask only genuine
product questions that the PRD and code cannot answer. Return ask_more with at
most eight concise questions, or ready_to_plan when no material uncertainty
remains. Do not modify files. Return only the required JSON object.
"""

_ANSWERS_SYSTEM_PROMPT = """You are the Architect for this project, and nobody
is available to answer the questions you asked. Decide each one yourself:
inspect the materialized repository and planning context, and pick the reading
that evidence best supports. Answer every question exactly once, concretely
enough to plan against, and state the decision rather than the uncertainty.
Every answer is recorded as an assumption the plan rests on. Do not modify
files. Return only the required JSON object.
"""

#: Appended to the questions prompt when nobody can be asked. A question put to
#: an empty room stops the run, so the same judgement has to be spent deciding
#: instead of asking.
_UNATTENDED_QUESTIONS_DIRECTIVE = """
Nobody is available to answer questions on this run. Asking one ends the run
without a plan, so it buys nothing. Decide every uncertainty yourself: read the
evidence, take the reading it best supports, and return ready_to_plan. Prefer a
defensible assumption to a question. Ask only where the repository, the PRD and
the analysis together leave you guessing rather than inferring, and the plan
would be built on the guess.
"""

#: Appended to the plan prompt on the same runs. Deciding a requirement is
#: sound; presenting the decision as a given is what costs the reader.
_UNATTENDED_PLAN_DIRECTIVE = """
Nobody confirmed the requirements you settled yourself on this run. List each
one under assumptions, paired with the question it answers, so a reader meets
every decision the plan rests on. A requirement the PRD, the analysis or the
answered Q&A already settles is not an assumption.
"""

_PLAN_SYSTEM_PROMPT = """You are the Architect for this project. Inspect the
materialized repository, confirmed PRD, analysis, and answered Q&A. Return a
detailed phased implementation plan matching the supplied schema. Ground file
paths and contracts in the repository, include concrete test strategies and
acceptance criteria, and do not modify files. Return only the required JSON
object.
"""

_PLAN_CONTRACT_CORRECTION = """
## Rejected plan

Your previous plan matched the schema but failed the deterministic checks that
follow it:

{error}

That rejected plan is the current plan in your context. Fix what the failure
names, then re-check the whole plan for anything else these checks would
reject before returning it: they stop at the first value they reject, so a
rejection usually means more remain. Return the whole plan again.
""".strip()


class ArchitectError(RuntimeError):
    """Raised when the Architect lifecycle cannot produce a durable plan."""


class ArchitectCancelled(ArchitectError):
    """Raised after preserving enough durable state to resume later."""


@dataclass(frozen=True, slots=True)
class ArchitectResult:
    """A durable Architect plan and its post-Architect Borg state."""

    borg: Borg
    plan: dict[str, Any]
    attempt: PlanningAttempt


class ArchitectLoop:
    """Run inline Architect Q&A and plan emission with durable resume points."""

    def __init__(
        self,
        repository: Repository,
        borg: Borg,
        store: SqliteStore,
        agent: AgentAdapter | SelectedAgent,
        *,
        io: InteractiveIO,
        unattended: bool = False,
        artifact_dir: Path | None = None,
        model: str | None = None,
        cancel: CancellationToken | None = None,
        progress: RunProgress | None = None,
        stage_key: str = "architect",
        child_key: str | None = None,
        dirty_borg_documents: Sequence[Path] = (),
        worktrees_root: Path | None = None,
    ) -> None:
        if cancel is not None and cancel.is_set():
            raise ArchitectCancelled("Architect run cancelled")
        require_read_only_agent(
            agent, role="Architect", error_factory=ArchitectError
        )
        try:
            resolved_model = resolve_agent_model(agent, model)
        except AgentSelectionError as error:
            raise ArchitectError(str(error)) from error

        paths = RepoPaths.discover(repository.root, cancel=cancel)
        if paths.root != repository.root:
            raise ValueError("repository root does not match its discovered Git root")
        self.repository = repository
        self.borg_id = borg.id
        self.store = store
        self.agent = agent
        self.io = io
        self.unattended = unattended
        self.artifact_dir = Path(
            artifact_dir or paths.artifacts_dir / "planning" / str(borg.id)
        ).resolve()
        self.model = resolved_model
        self.cancel = cancel
        self.progress = progress
        self.stage_key = stage_key
        self.child_key = child_key
        self.dirty_borg_documents = tuple(dirty_borg_documents)
        self.worktrees_root = worktrees_root
        if progress is not None:
            if child_key is None and stage_key not in progress.stages:
                progress.declare(StageSpec(stage_key, "Architect"))
            elif child_key is not None:
                if stage_key not in progress.stages:
                    raise ValueError(
                        "Architect revision parent must already be declared"
                    )
                if child_key not in progress.stages[stage_key].children:
                    progress.declare_child(
                        stage_key,
                        ChildSpec(child_key, "Architect revision"),
                    )
        self._turns = DurablePlanningTurns(
            repository,
            borg,
            store,
            agent,
            role="Architect",
            model=resolved_model,
            artifact_dir=self.artifact_dir,
            error_factory=ArchitectError,
            cancelled_error_factory=ArchitectCancelled,
            cancel=cancel,
            progress=progress,
            stage_key=stage_key if progress is not None else None,
            child_key=child_key if progress is not None else None,
            dirty_borg_documents=dirty_borg_documents,
            worktrees_root=worktrees_root,
        )

    def run(self) -> ArchitectResult:
        """Continue from durable history until a plan is ready for Tech Lead review."""
        try:
            completed_plan = self._completed_plan()
            if completed_plan is not None:
                self._seed_progress(completed_plan)
                return ArchitectResult(
                    borg=self._turns.current_borg(),
                    plan=completed_plan.result or {},
                    attempt=completed_plan,
                )

            self._start_progress()
            result = self._run()
            self._complete_progress(result.attempt)
            return result
        except (ArchitectCancelled, KeyboardInterrupt) as error:
            self._reconcile_progress(str(error), stopped=True)
            raise
        except Exception as error:
            self._reconcile_progress(
                str(error),
                stopped=self.cancel is not None and self.cancel.is_set(),
            )
            raise

    def _run(self) -> ArchitectResult:
        """Execute fresh Architect work after its progress record starts."""
        borg = self._turns.current_borg()

        if borg.state is BorgState.DRAFT:
            borg = self._turns.transition(borg, BorgState.ARCHITECT_WORKING)
        elif borg.state not in {
            BorgState.ARCHITECT_WORKING,
            BorgState.ARCHITECT_AWAITING_ANSWERS,
        }:
            raise ArchitectError(
                f"Borg {borg.name!r} cannot run Architect from state "
                f"{borg.state.value!r}"
            )

        pending = self._pending_question()
        if pending is not None:
            borg = self._answer_question_round(borg, pending)

        while True:
            if self.cancel is not None and self.cancel.is_set():
                raise ArchitectCancelled("Architect run cancelled")

            question_attempts = self._turns.attempts(_QUESTIONS_PHASE)
            completed_questions = [
                attempt
                for attempt in question_attempts
                if attempt.status is PlanningAttemptStatus.COMPLETED
            ]
            latest = completed_questions[-1] if completed_questions else None
            ready = (
                latest is not None
                and (latest.result or {}).get("decision") == "ready_to_plan"
            )
            forced = len(completed_questions) >= ARCHITECT_QUESTION_ROUND_CAP
            if ready or forced:
                return self._run_plan()

            question_round = len(completed_questions) + 1
            attempt, payload = self._run_turn(
                phase=_QUESTIONS_PHASE,
                round_number=self._turns.next_round(_QUESTIONS_PHASE),
                schema=ARCHITECT_QUESTIONS_SCHEMA,
                system_prompt=self._questions_system_prompt(),
                user_prompt=(
                    "Inspect .betterborg/state/planning/context/manifest.json and its "
                    "referenced evidence. This is Architect question round "
                    f"{question_round} "
                    f"of {ARCHITECT_QUESTION_ROUND_CAP}."
                ),
            )
            questions = list(payload.get("questions") or [])
            try:
                self._validate_question_payload(payload, questions)
            except ArchitectError as error:
                self.store.complete_planning_attempt(
                    attempt.id,
                    status=PlanningAttemptStatus.FAILED,
                    result=payload,
                    summary=str(error),
                )
                raise
            if payload["decision"] == "ready_to_plan":
                self._complete_attempt(attempt, payload, "ready to plan")
                continue

            question = PlanningQuestion(
                borg_id=self.borg_id,
                attempt_id=attempt.id,
                round=self._next_question_round(),
                questions=questions,
            )
            borg = self._store_question_turn(borg, attempt, payload, question)
            borg = self._answer_question_round(borg, question)

    def _run_plan(self) -> ArchitectResult:
        rejected_plan: dict[str, Any] | None = None
        correction: str | None = None
        contract_rounds = 0
        while True:
            completed = self._completed_plan()
            if completed is not None:
                return ArchitectResult(
                    borg=self._turns.current_borg(),
                    plan=completed.result or {},
                    attempt=completed,
                )

            current_plan = self._latest_plan()
            revision = current_plan is not None and not self._plan_open_questions(
                current_plan.result
            )
            user_prompt = (
                "Read .betterborg/state/planning/context/manifest.json and every "
                "relevant referenced context file, then emit the implementation "
                "plan. Resolve every answered product question and leave "
                "open_questions empty unless a genuine uncertainty remains."
            )
            if revision:
                user_prompt += (
                    " Revise the current plan in place, addressing every "
                    "persisted Tech Lead finding without regressing earlier "
                    "corrections."
                )
            if correction is not None:
                user_prompt += f"\n\n{correction}"
            # A rejected plan supersedes the last completed one as the plan to
            # revise, because the turn that follows it exists to correct it.
            plan_to_revise = (
                rejected_plan
                if rejected_plan is not None
                else current_plan.result
                if current_plan is not None
                else None
            )
            attempt, payload = self._run_turn(
                phase=_PLAN_PHASE,
                round_number=self._turns.next_round(_PLAN_PHASE),
                schema=ARCHITECT_PLAN_SCHEMA,
                system_prompt=self._plan_system_prompt(),
                user_prompt=user_prompt,
                current_plan=(
                    json.dumps(plan_to_revise, indent=2, sort_keys=True)
                    if plan_to_revise is not None
                    else None
                ),
            )
            correction = None
            rejected_plan = None
            # Everything below this point materializes a worktree, and a token
            # set during the turn would otherwise surface from git discovery as
            # a missing repository rather than as the cancellation it is.
            if self.cancel is not None and self.cancel.is_set():
                raise ArchitectCancelled("Architect run cancelled")
            payload = self._with_assumptions(payload, plan_to_revise)
            open_questions = self._plan_open_questions(payload)
            borg = self._turns.current_borg()
            if open_questions:
                question = PlanningQuestion(
                    borg_id=self.borg_id,
                    attempt_id=attempt.id,
                    round=self._next_question_round(),
                    questions=[
                        {"id": f"q{index}", "question": text}
                        for index, text in enumerate(open_questions, start=1)
                    ],
                )
                borg = self._store_plan_question_turn(
                    borg, attempt, payload, question
                )
                self._answer_question_round(borg, question)
                continue

            try:
                self._validate_plan_in_snapshot(payload)
            except PlanValidationError as error:
                self.store.complete_planning_attempt(
                    attempt.id,
                    status=PlanningAttemptStatus.FAILED,
                    result=payload,
                    summary=f"invalid plan contract: {error}",
                )
                contract_rounds += 1
                if contract_rounds >= ARCHITECT_PLAN_CONTRACT_ROUND_CAP:
                    raise ArchitectError(
                        f"Architect plan failed deterministic validation: {error}"
                    ) from error
                correction = _PLAN_CONTRACT_CORRECTION.format(error=error)
                rejected_plan = payload
                continue

            with self.store.transaction():
                completed = self.store.complete_planning_attempt(
                    attempt.id,
                    status=PlanningAttemptStatus.COMPLETED,
                    result=payload,
                    summary=str(payload["title"]),
                )
                borg = self._turns.transition(borg, BorgState.TECH_REVIEW_WORKING)
            return ArchitectResult(borg=borg, plan=payload, attempt=completed)

    def _run_turn(
        self,
        *,
        phase: str,
        round_number: int,
        schema: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        current_plan: str | None = None,
    ) -> tuple[PlanningAttempt, dict[str, Any]]:
        return self._turns.run(
            phase=phase,
            round_number=round_number,
            schema=schema,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            current_plan=current_plan,
        )

    def _store_question_turn(
        self,
        borg: Borg,
        attempt: PlanningAttempt,
        payload: dict[str, Any],
        question: PlanningQuestion,
    ) -> Borg:
        with self.store.transaction():
            self.store.complete_planning_attempt(
                attempt.id,
                status=PlanningAttemptStatus.COMPLETED,
                result=payload,
                summary=f"asked {len(question.questions)} product question(s)",
            )
            self.store.append_planning_question(question)
            return self._turns.transition(
                borg, BorgState.ARCHITECT_AWAITING_ANSWERS
            )

    def _store_plan_question_turn(
        self,
        borg: Borg,
        attempt: PlanningAttempt,
        payload: dict[str, Any],
        question: PlanningQuestion,
    ) -> Borg:
        with self.store.transaction():
            self.store.complete_planning_attempt(
                attempt.id,
                status=PlanningAttemptStatus.COMPLETED,
                result=payload,
                summary=f"plan has {len(question.questions)} open question(s)",
            )
            self.store.append_planning_question(question)
            return self._turns.transition(
                borg, BorgState.ARCHITECT_AWAITING_ANSWERS
            )

    def _answer_question_round(self, borg: Borg, question: PlanningQuestion) -> Borg:
        answers = (
            self._assume_question_round(question)
            if self.unattended
            else self._prompt_question_round(question)
        )
        with self.store.transaction():
            self.store.answer_planning_question(question.id, answers)
            return self._turns.transition(borg, BorgState.ARCHITECT_WORKING)

    def _prompt_question_round(
        self, question: PlanningQuestion
    ) -> list[dict[str, object]]:
        """Ask the operator at the terminal for one round of answers."""
        answers: list[dict[str, object]] = []
        for item in question.questions:
            suspension = self.progress.suspend() if self.progress else nullcontext()
            with suspension:
                why = str(item.get("why") or "").strip()
                hint = str(item.get("hint") or "").strip()
                if why:
                    self.io.write(f"Why this matters: {why}")
                if hint:
                    self.io.write(f"Answer guidance: {hint}")
                answer = self.io.prompt(str(item["question"]))
            if answer is None:
                raise ArchitectCancelled("Architect questions are awaiting answers")
            answer = answer.strip()
            if not answer:
                raise ArchitectError("Architect question answers must not be empty")
            answers.append({"q_id": item["id"], "answer": answer})
        return answers

    def _assume_question_round(
        self, question: PlanningQuestion
    ) -> list[dict[str, object]]:
        """Decide one round from the evidence, because nobody else can."""
        if question.round > ARCHITECT_QUESTION_ROUND_CAP:
            raise ArchitectError(
                "Architect asked past question round "
                f"{ARCHITECT_QUESTION_ROUND_CAP}; an unattended run cannot "
                "assume further answers"
            )
        # A round raised by a plan is a question about that plan, and the turn
        # answering it is a fresh agent with none of the reasoning that raised
        # it. Without the plan in its workspace the manifest tells it no plan
        # exists, and it decides the question from the PRD alone.
        plan = self._latest_plan()
        attempt, payload = self._turns.run(
            phase=_ANSWERS_PHASE,
            round_number=self._turns.next_round(_ANSWERS_PHASE),
            schema=ARCHITECT_ANSWERS_SCHEMA,
            system_prompt=_ANSWERS_SYSTEM_PROMPT,
            user_prompt=self._assumed_answers_prompt(question, plan is not None),
            current_plan=(
                json.dumps(plan.result, indent=2, sort_keys=True)
                if plan is not None
                else None
            ),
            turn_name="assumed answers",
            request_context={"question_id": str(question.id)},
        )
        try:
            answers = self._assumed_answers(question, payload)
        except ArchitectError as error:
            self.store.complete_planning_attempt(
                attempt.id,
                status=PlanningAttemptStatus.FAILED,
                result=payload,
                summary=str(error),
            )
            raise
        self._complete_attempt(
            attempt, payload, f"assumed {len(answers)} answer(s)"
        )
        return answers

    def _start_progress(self) -> None:
        if self.progress is None:
            return
        if self.child_key is None:
            self.progress.start(self.stage_key)
        else:
            child = self.progress.stages[self.stage_key].children[self.child_key]
            if child.state is StageState.PENDING:
                self.progress.start_child(self.stage_key, self.child_key)
            elif child.state is not StageState.RUNNING:
                raise ArchitectError(
                    "Architect revision progress "
                    f"{self.child_key!r} is already terminal"
                )

    def _seed_progress(self, attempt: PlanningAttempt) -> None:
        if self.progress is None:
            return
        result = planning_attempt_result(attempt, default="plan ready")
        duration = planning_attempt_duration(attempt)
        if self.child_key is None:
            record = self.progress.stages[self.stage_key]
            if record.state is StageState.PENDING:
                self.progress.seed_completed(self.stage_key, result, duration)
        else:
            child = self.progress.stages[self.stage_key].children[self.child_key]
            if child.state is StageState.PENDING:
                self.progress.seed_child_completed(
                    self.stage_key, self.child_key, result, duration
                )

    def _complete_progress(self, attempt: PlanningAttempt) -> None:
        if self.progress is None:
            return
        if self.child_key is None:
            self.progress.complete(
                self.stage_key,
                planning_attempt_result(attempt, default="plan ready"),
            )
        else:
            self.progress.complete_child(
                self.stage_key,
                self.child_key,
                planning_attempt_result(attempt, default="plan ready"),
            )

    def _reconcile_progress(self, result: str, *, stopped: bool) -> None:
        if self.progress is None:
            return
        record = (
            self.progress.stages[self.stage_key]
            if self.child_key is None
            else self.progress.stages[self.stage_key].children[self.child_key]
        )
        if record.state is not StageState.RUNNING:
            return
        completed_plan = self._completed_plan()
        if completed_plan is not None:
            self._complete_progress(completed_plan)
        elif stopped:
            self._stop_progress(result)
        else:
            self._fail_progress(result)

    def _fail_progress(self, result: str) -> None:
        if self.progress is None:
            return
        if self.child_key is None:
            self.progress.fail(self.stage_key, result)
        else:
            self.progress.fail_child(self.stage_key, self.child_key, result)

    def _stop_progress(self, result: str) -> None:
        if self.progress is None:
            return
        if self.child_key is None:
            self.progress.stop(self.stage_key, result)
        else:
            self.progress.stop_child(self.stage_key, self.child_key, result)

    def _complete_attempt(
        self, attempt: PlanningAttempt, payload: dict[str, Any], summary: str
    ) -> PlanningAttempt:
        return self.store.complete_planning_attempt(
            attempt.id,
            status=PlanningAttemptStatus.COMPLETED,
            result=payload,
            summary=summary,
        )

    def _completed_plan(self) -> PlanningAttempt | None:
        attempts = self.store.list_planning_attempts(self.borg_id)
        completed_index = next(
            (
                index
                for index in range(len(attempts) - 1, -1, -1)
                if attempts[index].phase == _PLAN_PHASE
                and attempts[index].status is PlanningAttemptStatus.COMPLETED
                and not self._plan_open_questions(attempts[index].result)
            ),
            None,
        )
        latest_review_index = next(
            (
                index
                for index in range(len(attempts) - 1, -1, -1)
                if attempts[index].phase == "tech_review"
                and attempts[index].status is PlanningAttemptStatus.COMPLETED
            ),
            None,
        )
        if completed_index is None or (
            latest_review_index is not None and latest_review_index > completed_index
        ):
            return None
        completed = attempts[completed_index]
        try:
            self._validate_plan_in_snapshot(completed.result or {})
        except PlanValidationError as error:
            raise ArchitectError(
                "Stored Architect plan failed deterministic validation: "
                f"{error}"
            ) from error
        return completed

    def _validate_plan_in_snapshot(self, plan: dict[str, Any]) -> None:
        """Validate against the same committed-only view exposed to Architect."""
        self._turns.validate_plan(plan)

    def _latest_plan(self) -> PlanningAttempt | None:
        return next(
            (
                item
                for item in reversed(self._turns.attempts(_PLAN_PHASE))
                if item.status is PlanningAttemptStatus.COMPLETED
                and item.result is not None
            ),
            None,
        )

    def _pending_question(self) -> PlanningQuestion | None:
        return next(
            (
                item
                for item in reversed(self.store.list_planning_questions(self.borg_id))
                if item.answers is None
            ),
            None,
        )

    def _next_question_round(self) -> int:
        questions = self.store.list_planning_questions(self.borg_id)
        return max((question.round for question in questions), default=0) + 1

    def _questions_system_prompt(self) -> str:
        """Instruct the Architect for the room it is actually speaking to."""
        if not self.unattended:
            return _QUESTIONS_SYSTEM_PROMPT
        return _QUESTIONS_SYSTEM_PROMPT + _UNATTENDED_QUESTIONS_DIRECTIVE

    def _plan_system_prompt(self) -> str:
        if not self.unattended:
            return _PLAN_SYSTEM_PROMPT
        return _PLAN_SYSTEM_PROMPT + _UNATTENDED_PLAN_DIRECTIVE

    def _with_assumptions(
        self, plan: dict[str, Any], superseded: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Settle which assumptions the plan carries, rather than the plan.

        Three sources reach here. Betterborg contributes the rounds the
        Architect asked and then answered itself, which are recorded and so
        cannot be dropped by leaving them out of a plan. The Architect
        declares the requirements it decided in place of asking, which only it
        knows it decided. And the plan this one supersedes contributes what it
        already carried, because a revision that rewrites a plan to address a
        finding has no reason to restate an assumption it is not revisiting,
        and losing one there costs the reader the same warning as never making
        it.

        Carrying forward is unconditional, so an assumption outlives the mode
        of the run that revises it. A stale one sends a reader to look at
        settled ground; a dropped one leaves a decision nobody took reading
        like a requirement somebody gave. The first is the cheaper mistake.

        What an attended run cannot do is originate one. A question there was
        answered by a person, so a plan claiming an assumption is describing a
        conversation that did not happen.
        """
        declared = self._declared_assumptions(plan) if self.unattended else []
        carried = self._declared_assumptions(superseded or {})
        stored = self._stored_assumptions()
        assumptions: list[dict[str, str]] = []
        seen: set[str] = set()
        # Stored first: an assumption Betterborg recorded is the one whose
        # wording came from the question as asked.
        for assumption in [*stored, *carried, *declared]:
            key = assumption["question"].casefold()
            if key in seen:
                continue
            seen.add(key)
            assumptions.append(assumption)
        plan = {key: value for key, value in plan.items() if key != "assumptions"}
        return {**plan, "assumptions": assumptions} if assumptions else plan

    @staticmethod
    def _declared_assumptions(plan: Mapping[str, Any]) -> list[dict[str, str]]:
        """Read the assumptions the Architect named in its own plan."""
        declared = plan.get("assumptions")
        if not isinstance(declared, list):
            return []
        assumptions: list[dict[str, str]] = []
        for assumption in declared:
            if not isinstance(assumption, Mapping):
                continue
            question = str(assumption.get("question") or "").strip()
            decision = str(assumption.get("assumption") or "").strip()
            if question and decision:
                assumptions.append({"question": question, "assumption": decision})
        return assumptions

    def _stored_assumptions(self) -> list[dict[str, str]]:
        """Pair every assumed answer with the question that prompted it."""
        assumptions: list[dict[str, str]] = []
        for stored in self.store.list_planning_questions(self.borg_id):
            asked = {
                str(item.get("id")): str(item.get("question") or "").strip()
                for item in stored.questions
            }
            for answer in stored.answers or []:
                if not answer.get("assumed"):
                    continue
                question = asked.get(str(answer.get("q_id")), "")
                assumption = str(answer.get("answer") or "").strip()
                if question and assumption:
                    assumptions.append(
                        {"question": question, "assumption": assumption}
                    )
        return assumptions

    @staticmethod
    def _assumed_answers_prompt(
        question: PlanningQuestion, has_plan: bool = False
    ) -> str:
        lines = [
            "Nobody is available to answer these Architect questions, so "
            "decide them yourself. Read "
            ".betterborg/state/planning/context/manifest.json and its "
            "referenced evidence, then answer every question below exactly "
            "once, by id.",
            "",
        ]
        if has_plan:
            lines[0] += (
                " The plan these questions were raised against is supplied "
                "with this turn; read it first, because a question raised by "
                "a plan is a question about that plan."
            )
        for item in question.questions:
            lines.append(f"- {item['id']}: {str(item['question']).strip()}")
            why = str(item.get("why") or "").strip()
            hint = str(item.get("hint") or "").strip()
            if why:
                lines.append(f"  Why this matters: {why}")
            if hint:
                lines.append(f"  Answer guidance: {hint}")
        return "\n".join(lines)

    @staticmethod
    def _assumed_answers(
        question: PlanningQuestion, payload: dict[str, Any]
    ) -> list[dict[str, object]]:
        answered = {
            str(item["q_id"]): str(item["answer"]).strip()
            for item in payload["answers"]
        }
        if len(answered) != len(payload["answers"]):
            raise ArchitectError("Architect assumed answer IDs must be unique")
        asked = [str(item["id"]) for item in question.questions]
        if set(answered) != set(asked):
            raise ArchitectError(
                "Architect must assume exactly one answer for each question"
            )
        if not all(answered.values()):
            raise ArchitectError("Architect assumed answers must not be empty")
        return [
            {"q_id": q_id, "answer": answered[q_id], "assumed": True}
            for q_id in asked
        ]

    @staticmethod
    def _plan_open_questions(result: dict[str, Any] | None) -> list[str]:
        return [
            question.strip()
            for question in (result or {}).get("open_questions", [])
            if isinstance(question, str) and question.strip()
        ]

    @staticmethod
    def _validate_question_payload(
        payload: dict[str, Any], questions: list[dict[str, Any]]
    ) -> None:
        if payload["decision"] == "ask_more" and not questions:
            raise ArchitectError("Architect ask_more result must contain questions")
        if payload["decision"] == "ready_to_plan" and questions:
            raise ArchitectError(
                "Architect ready_to_plan result must not contain questions"
            )
        identifiers = [item["id"] for item in questions]
        if len(identifiers) != len(set(identifiers)):
            raise ArchitectError("Architect question IDs must be unique within a round")


__all__ = [
    "ARCHITECT_ANSWERS_SCHEMA",
    "ARCHITECT_PLAN_CONTRACT_ROUND_CAP",
    "ARCHITECT_PLAN_SCHEMA",
    "ARCHITECT_QUESTION_ROUND_CAP",
    "ARCHITECT_QUESTIONS_SCHEMA",
    "ArchitectCancelled",
    "ArchitectError",
    "ArchitectLoop",
    "ArchitectResult",
]
