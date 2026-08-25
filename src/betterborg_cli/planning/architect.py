"""Resumable, terminal-native Architect question and planning loop."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from betterborg_cli.agent_runtime.api_tools import READ_ONLY_API_TOOLS
from betterborg_cli.agent_runtime.base import (
    AgentAdapter,
    AgentRunSpec,
    AgentStatus,
    CancellationToken,
)
from betterborg_cli.agent_runtime.selection import (
    AgentSelectionError,
    SelectedAgent,
    resolve_agent_model,
)
from betterborg_cli.agent_runtime.structured import (
    StructuredResultError,
    validate_structured_result,
)
from betterborg_cli.planning.worktree import materialize_planning_worktree
from betterborg_cli.prd_session import InteractiveIO
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
_QUESTIONS_PHASE = "architect_questions"
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
    },
}

_QUESTIONS_SYSTEM_PROMPT = """You are the Architect for this project. Inspect the
materialized repository and planning context before deciding. Ask only genuine
product questions that the PRD and code cannot answer. Return ask_more with at
most eight concise questions, or ready_to_plan when no material uncertainty
remains. Do not modify files. Return only the required JSON object.
"""

_PLAN_SYSTEM_PROMPT = """You are the Architect for this project. Inspect the
materialized repository, confirmed PRD, analysis, and answered Q&A. Return a
detailed phased implementation plan matching the supplied schema. Ground file
paths and contracts in the repository, include concrete test strategies and
acceptance criteria, and do not modify files. Return only the required JSON
object.
"""


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
        artifact_dir: Path | None = None,
        model: str | None = None,
        cancel: CancellationToken | None = None,
        dirty_borg_documents: Sequence[Path] = (),
        worktrees_root: Path | None = None,
    ) -> None:
        if store.get_repository(repository.id) != repository:
            raise ValueError("repository must already be present in the supplied store")
        if borg.repository_id != repository.id or store.get_borg(borg.id) != borg:
            raise ValueError(
                "Borg must already belong to the supplied repository and store"
            )
        if not (
            agent.capabilities.tool_allowlist
            or agent.capabilities.read_only_sandbox
        ):
            raise ArchitectError(
                f"adapter {agent.name!r} cannot enforce the Architect "
                "read-only execution boundary"
            )
        if agent.capabilities.host_capable and not isinstance(agent, SelectedAgent):
            raise ArchitectError(
                f"host-capable adapter {agent.name!r} must be wrapped by SelectedAgent"
            )
        try:
            resolved_model = resolve_agent_model(agent, model)
        except AgentSelectionError as error:
            raise ArchitectError(str(error)) from error

        paths = RepoPaths.discover(repository.root)
        if paths.root != repository.root:
            raise ValueError("repository root does not match its discovered Git root")
        self.repository = repository
        self.borg_id = borg.id
        self.store = store
        self.agent = agent
        self.io = io
        self.artifact_dir = Path(
            artifact_dir or paths.artifacts_dir / "planning" / str(borg.id)
        ).resolve()
        self.model = resolved_model
        self.cancel = cancel
        self.dirty_borg_documents = tuple(dirty_borg_documents)
        self.worktrees_root = worktrees_root

    def run(self) -> ArchitectResult:
        """Continue from durable history until a plan is ready for Tech Lead review."""
        borg = self._current_borg()
        completed_plan = self._completed_plan()
        if completed_plan is not None:
            return ArchitectResult(
                borg=borg, plan=completed_plan.result or {}, attempt=completed_plan
            )

        if borg.state is BorgState.DRAFT:
            borg = self._transition(borg, BorgState.ARCHITECT_WORKING)
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

            question_attempts = self._phase_attempts(_QUESTIONS_PHASE)
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
                round_number=self._next_attempt_round(_QUESTIONS_PHASE),
                schema=ARCHITECT_QUESTIONS_SCHEMA,
                system_prompt=_QUESTIONS_SYSTEM_PROMPT,
                user_prompt=(
                    "Inspect .borg/state/planning/context/manifest.json and its "
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
        while True:
            completed = self._completed_plan()
            if completed is not None:
                return ArchitectResult(
                    borg=self._current_borg(),
                    plan=completed.result or {},
                    attempt=completed,
                )

            current_plan = self._latest_ambiguous_plan()
            attempt, payload = self._run_turn(
                phase=_PLAN_PHASE,
                round_number=self._next_attempt_round(_PLAN_PHASE),
                schema=ARCHITECT_PLAN_SCHEMA,
                system_prompt=_PLAN_SYSTEM_PROMPT,
                user_prompt=(
                    "Read .borg/state/planning/context/manifest.json and every "
                    "relevant referenced context file, then emit the implementation "
                    "plan. Resolve every answered product question and leave "
                    "open_questions empty unless a genuine uncertainty remains."
                ),
                current_plan=(
                    json.dumps(current_plan.result, indent=2, sort_keys=True)
                    if current_plan is not None
                    else None
                ),
            )
            open_questions = self._plan_open_questions(payload)
            borg = self._current_borg()
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

            with self.store.transaction():
                completed = self.store.complete_planning_attempt(
                    attempt.id,
                    status=PlanningAttemptStatus.COMPLETED,
                    result=payload,
                    summary=str(payload["title"]),
                )
                borg = self._transition(borg, BorgState.TECH_REVIEW_WORKING)
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
        running = next(
            (
                item
                for item in reversed(self._phase_attempts(phase))
                if item.status is PlanningAttemptStatus.RUNNING
            ),
            None,
        )
        if running is not None:
            recovered = self._recover_payload(running, schema)
            if recovered is not None:
                return running, recovered
            refreshed = next(
                item
                for item in self._phase_attempts(phase)
                if item.id == running.id
            )
            if refreshed.status is not PlanningAttemptStatus.RUNNING:
                return self._run_turn(
                    phase=phase,
                    round_number=refreshed.round + 1,
                    schema=schema,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    current_plan=current_plan,
                )
            attempt = running
        else:
            attempt = PlanningAttempt(
                borg_id=self.borg_id,
                phase=phase,
                round=round_number,
                adapter=self.agent.name,
                model=self.model,
                request={"result_path": "pending"},
            )
            result_path = self._result_path(attempt)
            attempt = PlanningAttempt(
                id=attempt.id,
                borg_id=attempt.borg_id,
                phase=attempt.phase,
                round=attempt.round,
                adapter=attempt.adapter,
                model=attempt.model,
                request={"result_path": str(result_path)},
                started_at=attempt.started_at,
            )
            self.store.append_planning_attempt(attempt)

        result_path = self._result_path(attempt)
        log_path = result_path.with_suffix(".log")
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        try:
            with materialize_planning_worktree(
                self.repository,
                self._current_borg(),
                self.store,
                current_plan=current_plan,
                dirty_borg_documents=self.dirty_borg_documents,
                worktrees_root=self.worktrees_root,
            ) as worktree:
                spec = AgentRunSpec(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema=schema,
                    cwd=worktree,
                    model=self.model,
                    allowed_tools=READ_ONLY_API_TOOLS,
                    log_path=log_path,
                    result_path=result_path,
                )
                result = self.agent.run(spec, cancel=self.cancel)
        except Exception as error:
            raise ArchitectError(f"Architect {phase} turn crashed: {error}") from error

        if result.status is AgentStatus.CANCELLED:
            self.store.complete_planning_attempt(
                attempt.id,
                status=PlanningAttemptStatus.CANCELLED,
                summary=result.error,
            )
            raise ArchitectCancelled(result.error or "Architect run cancelled")
        if result.status is not AgentStatus.COMPLETED or result.payload is None:
            self.store.complete_planning_attempt(
                attempt.id,
                status=PlanningAttemptStatus.FAILED,
                result=result.payload,
                summary=result.error,
            )
            raise ArchitectError(
                result.error or f"Architect {phase} returned {result.status.value}"
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
            raise ArchitectError(
                f"Architect {phase} returned an invalid structured result: {error}"
            ) from error
        return attempt, result.payload

    def _recover_payload(
        self, attempt: PlanningAttempt, schema: dict[str, Any]
    ) -> dict[str, Any] | None:
        result_path = self._result_path(attempt)
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
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
            return self._transition(borg, BorgState.ARCHITECT_AWAITING_ANSWERS)

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
            return self._transition(borg, BorgState.ARCHITECT_AWAITING_ANSWERS)

    def _answer_question_round(self, borg: Borg, question: PlanningQuestion) -> Borg:
        answers: list[dict[str, object]] = []
        for item in question.questions:
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

        with self.store.transaction():
            self.store.answer_planning_question(question.id, answers)
            return self._transition(borg, BorgState.ARCHITECT_WORKING)

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
        return next(
            (
                item
                for item in reversed(self._phase_attempts(_PLAN_PHASE))
                if item.status is PlanningAttemptStatus.COMPLETED
                and not self._plan_open_questions(item.result)
            ),
            None,
        )

    def _latest_ambiguous_plan(self) -> PlanningAttempt | None:
        return next(
            (
                item
                for item in reversed(self._phase_attempts(_PLAN_PHASE))
                if item.status is PlanningAttemptStatus.COMPLETED
                and self._plan_open_questions(item.result)
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

    def _phase_attempts(self, phase: str) -> list[PlanningAttempt]:
        return [
            item
            for item in self.store.list_planning_attempts(self.borg_id)
            if item.phase == phase
        ]

    def _next_attempt_round(self, phase: str) -> int:
        attempts = self._phase_attempts(phase)
        return max((attempt.round for attempt in attempts), default=0) + 1

    def _next_question_round(self) -> int:
        questions = self.store.list_planning_questions(self.borg_id)
        return max((question.round for question in questions), default=0) + 1

    @staticmethod
    def _plan_open_questions(result: dict[str, Any] | None) -> list[str]:
        return [
            question.strip()
            for question in (result or {}).get("open_questions", [])
            if isinstance(question, str) and question.strip()
        ]

    def _current_borg(self) -> Borg:
        borg = self.store.get_borg(self.borg_id)
        if borg is None:
            raise ArchitectError(f"Borg {self.borg_id} no longer exists")
        return borg

    def _transition(self, borg: Borg, state: BorgState) -> Borg:
        if borg.state is state:
            return borg
        return self.store.compare_and_set_borg_state(
            borg.id,
            expected_state=borg.state,
            expected_version=borg.state_version,
            new_state=state,
        )

    def _result_path(self, attempt: PlanningAttempt) -> Path:
        stored = attempt.request.get("result_path")
        if isinstance(stored, str) and stored != "pending":
            return Path(stored)
        return self.artifact_dir / f"{attempt.id}.json"

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
    "ARCHITECT_PLAN_SCHEMA",
    "ARCHITECT_QUESTION_ROUND_CAP",
    "ARCHITECT_QUESTIONS_SCHEMA",
    "ArchitectCancelled",
    "ArchitectError",
    "ArchitectLoop",
    "ArchitectResult",
]
