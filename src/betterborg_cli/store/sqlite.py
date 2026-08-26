"""Serialized, transactional SQLite storage with forward-only migrations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from importlib import resources
from pathlib import Path
from uuid import UUID

from betterborg_cli.agent_runtime.base import AgentStatus, AgentUsage, BillingMode
from betterborg_cli.store.models import (
    AgentAttempt,
    Borg,
    BorgState,
    ComposeResource,
    EnvironmentAttempt,
    ExecutionAttemptStatus,
    ExecutionEvent,
    ExecutionRun,
    ExecutionRunAcquisition,
    ExecutionRunStatus,
    GeneratedPrompt,
    Operation,
    PlanApproval,
    PlanChangeRequest,
    PlanningAttempt,
    PlanningAttemptStatus,
    PlanningFinding,
    PlanningQuestion,
    PrdSession,
    PrdTurn,
    Repository,
    RepositoryAnalysis,
    RepositoryPackage,
    TaskBatch,
    TaskClaim,
    TaskComplexity,
    TaskDependency,
    TaskFinding,
    TaskGeneration,
    TaskGenerationStatus,
    TaskRecord,
    TaskRuntime,
    TaskRuntimeCost,
    TaskRuntimeRow,
    TaskRuntimeStatus,
    utcnow,
)

_MIGRATION_NAME = re.compile(r"^(?P<version>[0-9]{3})_[a-z0-9_]+\.sql$")
_TERMINAL_ATTEMPT_EVENT_KINDS = frozenset(
    {
        "environment.attempt_finished",
        "environment.attempt_interrupted",
        "agent.attempt_finished",
        "agent.attempt_interrupted",
    }
)


class StaleBorgStateError(RuntimeError):
    """Raised when a Borg state compare-and-set loses a concurrent race."""


class ExecutionOwnershipError(RuntimeError):
    """Raised when a caller no longer owns a live execution lease."""


class StaleTaskRuntimeError(RuntimeError):
    """Raised when a guarded task phase transition loses its race."""


_ACTIVE_TASK_STATUSES = frozenset(
    {
        TaskRuntimeStatus.CLAIMED,
        TaskRuntimeStatus.ENVIRONMENT,
        TaskRuntimeStatus.CODING,
        TaskRuntimeStatus.REVIEW,
        TaskRuntimeStatus.FIX,
        TaskRuntimeStatus.MERGING,
    }
)
_TERMINAL_TASK_STATUSES = frozenset(
    {
        TaskRuntimeStatus.DONE,
        TaskRuntimeStatus.BLOCKED,
        TaskRuntimeStatus.FAILED,
    }
)
_TASK_TRANSITIONS = {
    TaskRuntimeStatus.CLAIMED: (
        _ACTIVE_TASK_STATUSES | _TERMINAL_TASK_STATUSES
    )
    - {TaskRuntimeStatus.CLAIMED},
    TaskRuntimeStatus.ENVIRONMENT: {
        TaskRuntimeStatus.CODING,
        *_TERMINAL_TASK_STATUSES,
    },
    TaskRuntimeStatus.CODING: {
        TaskRuntimeStatus.REVIEW,
        *_TERMINAL_TASK_STATUSES,
    },
    TaskRuntimeStatus.REVIEW: {
        TaskRuntimeStatus.FIX,
        TaskRuntimeStatus.MERGING,
        *_TERMINAL_TASK_STATUSES,
    },
    TaskRuntimeStatus.FIX: {
        TaskRuntimeStatus.REVIEW,
        *_TERMINAL_TASK_STATUSES,
    },
    TaskRuntimeStatus.MERGING: _TERMINAL_TASK_STATUSES,
}


class SqliteStore:
    """A single SQLite connection serialized by a re-entrant lock."""

    def __init__(self, connection: sqlite3.Connection):
        self._connection = connection
        self._lock = threading.RLock()
        self._transaction_depth = 0
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA recursive_triggers = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        self._ensure_schema()

    @classmethod
    def open(cls, path: Path) -> SqliteStore:
        """Open or create an on-disk store at ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            path,
            check_same_thread=False,
            isolation_level=None,
            timeout=30.0,
        )
        try:
            return cls(connection)
        except BaseException:
            connection.close()
            raise

    @contextmanager
    def locked_connection(self) -> Iterator[sqlite3.Connection]:
        """Yield the connection while excluding access from other threads."""
        with self._lock:
            yield self._connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run an atomic transaction, using savepoints when called recursively."""
        with self._lock:
            depth = self._transaction_depth
            savepoint = f"betterborg_{depth}"
            if depth == 0:
                self._connection.execute("BEGIN IMMEDIATE")
            else:
                self._connection.execute(f"SAVEPOINT {savepoint}")
            self._transaction_depth += 1
            try:
                yield self._connection
                if depth == 0:
                    self._connection.execute("COMMIT")
                else:
                    self._connection.execute(f"RELEASE {savepoint}")
            except BaseException:
                if depth == 0:
                    if self._connection.in_transaction:
                        self._connection.execute("ROLLBACK")
                else:
                    self._connection.execute(f"ROLLBACK TO {savepoint}")
                    self._connection.execute(f"RELEASE {savepoint}")
                raise
            finally:
                self._transaction_depth -= 1

    def close(self) -> None:
        """Close the store after all current access has completed."""
        with self._lock:
            if self._transaction_depth:
                raise RuntimeError("cannot close the store during a transaction")
            self._connection.close()

    def __enter__(self) -> SqliteStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def add_repository(self, repository: Repository) -> None:
        """Persist a repository record."""
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO repositories(id, root, created_at) VALUES (?, ?, ?)",
                (
                    str(repository.id),
                    str(repository.root),
                    repository.created_at.isoformat(),
                ),
            )

    def get_repository(self, repository_id: UUID) -> Repository | None:
        """Return one repository by ID, if it exists."""
        with self.locked_connection() as connection:
            row = connection.execute(
                "SELECT id, root, created_at FROM repositories WHERE id = ?",
                (str(repository_id),),
            ).fetchone()
        if row is None:
            return None
        return Repository(
            id=UUID(row["id"]),
            root=Path(row["root"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def append_operation(self, operation: Operation) -> None:
        """Append one immutable operation to the repository ledger."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO operations(id, repository_id, kind, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(operation.id),
                    str(operation.repository_id),
                    operation.kind,
                    json.dumps(
                        operation.payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    operation.created_at.isoformat(),
                ),
            )

    def list_operations(self, repository_id: UUID) -> list[Operation]:
        """Return ledger entries in stable append order."""
        with self.locked_connection() as connection:
            rows = connection.execute(
                """
                SELECT id, repository_id, kind, payload, created_at
                FROM operations
                WHERE repository_id = ?
                ORDER BY created_at, id
                """,
                (str(repository_id),),
            ).fetchall()
        return [
            Operation(
                id=UUID(row["id"]),
                repository_id=UUID(row["repository_id"]),
                kind=row["kind"],
                payload=json.loads(row["payload"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def append_analysis(
        self,
        analysis: RepositoryAnalysis,
        packages: list[RepositoryPackage],
    ) -> None:
        """Atomically append one successful analysis and all package rows."""
        if not packages:
            raise ValueError("an analysis must contain at least one package")
        if any(
            package.repository_id != analysis.repository_id
            or package.analysis_id != analysis.id
            for package in packages
        ):
            raise ValueError("analysis packages must belong to the appended analysis")
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO repository_analyses(
                    id, repository_id, head_sha, summary, primary_language,
                    is_monorepo, overall_score, analysis_json,
                    prior_analysis_id, score_delta, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(analysis.id),
                    str(analysis.repository_id),
                    analysis.head_sha,
                    analysis.summary,
                    analysis.primary_language,
                    int(analysis.is_monorepo),
                    analysis.overall_score,
                    _encode_json(analysis.analysis_json),
                    str(analysis.prior_analysis_id)
                    if analysis.prior_analysis_id is not None
                    else None,
                    analysis.score_delta,
                    analysis.created_at.isoformat(),
                ),
            )
            connection.executemany(
                """
                INSERT INTO repository_packages(
                    id, repository_id, analysis_id, package_path,
                    package_name, primary_language, rubric_json, overall_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(package.id),
                        str(package.repository_id),
                        str(package.analysis_id),
                        package.package_path,
                        package.package_name,
                        package.primary_language,
                        _encode_json(package.rubric),
                        package.overall_score,
                    )
                    for package in packages
                ],
            )

    def get_analysis(self, analysis_id: UUID) -> RepositoryAnalysis | None:
        """Return one immutable analysis by ID."""
        with self.locked_connection() as connection:
            row = connection.execute(
                "SELECT * FROM repository_analyses WHERE id = ?",
                (str(analysis_id),),
            ).fetchone()
        return _row_to_analysis(row) if row is not None else None

    def list_analyses(self, repository_id: UUID) -> list[RepositoryAnalysis]:
        """Return a repository's successful analyses in append order."""
        with self.locked_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM repository_analyses
                WHERE repository_id = ?
                ORDER BY created_at, id
                """,
                (str(repository_id),),
            ).fetchall()
        return [_row_to_analysis(row) for row in rows]

    def get_prior_ready_analysis(
        self,
        repository_id: UUID,
        *,
        before_analysis_id: UUID | None = None,
    ) -> RepositoryAnalysis | None:
        """Return the latest successful analysis, optionally before one run."""
        if before_analysis_id is not None:
            with self.locked_connection() as connection:
                row = connection.execute(
                    """
                    SELECT prior.*
                    FROM repository_analyses AS current
                    JOIN repository_analyses AS prior
                      ON prior.id = current.prior_analysis_id
                     AND prior.repository_id = current.repository_id
                    WHERE current.id = ? AND current.repository_id = ?
                    """,
                    (str(before_analysis_id), str(repository_id)),
                ).fetchone()
            return _row_to_analysis(row) if row is not None else None

        query = """
            SELECT candidate.*
            FROM repository_analyses AS candidate
            WHERE candidate.repository_id = ?
        """
        parameters: list[str] = [str(repository_id)]
        query += " ORDER BY candidate.created_at DESC, candidate.id DESC LIMIT 1"
        with self.locked_connection() as connection:
            row = connection.execute(query, parameters).fetchone()
        return _row_to_analysis(row) if row is not None else None

    def list_packages(self, analysis_id: UUID) -> list[RepositoryPackage]:
        """Return package rows for one analysis in stable path order."""
        with self.locked_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM repository_packages
                WHERE analysis_id = ?
                ORDER BY package_path
                """,
                (str(analysis_id),),
            ).fetchall()
        return [_row_to_package(row) for row in rows]

    def append_generated_prompt(
        self,
        *,
        repository_id: UUID,
        analysis_id: UUID,
        role: str,
        body_md: str,
    ) -> GeneratedPrompt:
        """Append and return the next prompt version for a repository role."""
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(MAX(version), 0) AS version
                FROM generated_prompts
                WHERE repository_id = ? AND role = ?
                """,
                (str(repository_id), role),
            ).fetchone()
            prompt = GeneratedPrompt(
                repository_id=repository_id,
                analysis_id=analysis_id,
                role=role,
                version=row["version"] + 1,
                body_md=body_md,
            )
            connection.execute(
                """
                INSERT INTO generated_prompts(
                    id, repository_id, analysis_id, role, version,
                    body_md, generated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(prompt.id),
                    str(prompt.repository_id),
                    str(prompt.analysis_id),
                    prompt.role,
                    prompt.version,
                    prompt.body_md,
                    prompt.generated_at.isoformat(),
                ),
            )
        return prompt

    def get_latest_generated_prompts(
        self, repository_id: UUID
    ) -> dict[str, GeneratedPrompt]:
        """Return the highest prompt version present for each role."""
        with self.locked_connection() as connection:
            rows = connection.execute(
                """
                SELECT prompt.*
                FROM generated_prompts AS prompt
                WHERE prompt.repository_id = ?
                  AND prompt.version = (
                      SELECT MAX(candidate.version)
                      FROM generated_prompts AS candidate
                      WHERE candidate.repository_id = prompt.repository_id
                        AND candidate.role = prompt.role
                  )
                ORDER BY prompt.role
                """,
                (str(repository_id),),
            ).fetchall()
        return {row["role"]: _row_to_generated_prompt(row) for row in rows}

    def add_borg(self, borg: Borg) -> None:
        """Persist one repository-scoped named Borg identity."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO borgs(
                    id, repository_id, name, state, state_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(borg.id),
                    str(borg.repository_id),
                    borg.name,
                    borg.state.value,
                    borg.state_version,
                    borg.created_at.isoformat(),
                ),
            )

    def get_borg(self, borg_id: UUID) -> Borg | None:
        """Return one Borg identity by ID, if it exists."""
        with self.locked_connection() as connection:
            row = connection.execute(
                "SELECT * FROM borgs WHERE id = ?", (str(borg_id),)
            ).fetchone()
        return _row_to_borg(row) if row is not None else None

    def get_borg_by_name(self, repository_id: UUID, name: str) -> Borg | None:
        """Return one repository's Borg identity by its unique name."""
        with self.locked_connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM borgs
                WHERE repository_id = ? AND name = ?
                """,
                (str(repository_id), name),
            ).fetchone()
        return _row_to_borg(row) if row is not None else None

    def compare_and_set_borg_state(
        self,
        borg_id: UUID,
        *,
        expected_state: BorgState,
        expected_version: int,
        new_state: BorgState,
    ) -> Borg:
        """Atomically transition a Borg when its state snapshot is still current."""
        if not isinstance(expected_state, BorgState) or not isinstance(
            new_state, BorgState
        ):
            raise TypeError("Borg states must be BorgState values")
        if expected_version < 0:
            raise ValueError("expected Borg state version must not be negative")
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE borgs
                SET state = ?, state_version = state_version + 1
                WHERE id = ? AND state = ? AND state_version = ?
                """,
                (
                    new_state.value,
                    str(borg_id),
                    expected_state.value,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                exists = connection.execute(
                    "SELECT 1 FROM borgs WHERE id = ?", (str(borg_id),)
                ).fetchone()
                if exists is None:
                    raise KeyError(f"Borg {borg_id} not found")
                raise StaleBorgStateError(
                    "Borg state changed before compare-and-set transition"
                )
            row = connection.execute(
                "SELECT * FROM borgs WHERE id = ?", (str(borg_id),)
            ).fetchone()
        return _row_to_borg(row)

    def append_planning_attempt(self, attempt: PlanningAttempt) -> None:
        """Append one planning attempt, whether running or already completed."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO planning_attempts(
                    id, borg_id, phase, round, adapter, model, request_json,
                    status, result_json, summary, started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(attempt.id),
                    str(attempt.borg_id),
                    attempt.phase,
                    attempt.round,
                    attempt.adapter,
                    attempt.model,
                    _encode_json(attempt.request),
                    attempt.status.value,
                    _encode_json(attempt.result)
                    if attempt.result is not None
                    else None,
                    attempt.summary,
                    attempt.started_at.isoformat(),
                    attempt.finished_at.isoformat()
                    if attempt.finished_at is not None
                    else None,
                ),
            )

    def complete_planning_attempt(
        self,
        attempt_id: UUID,
        *,
        status: PlanningAttemptStatus,
        result: dict[str, object] | None = None,
        summary: str | None = None,
    ) -> PlanningAttempt:
        """Finish a running attempt exactly once and return its durable record."""
        if status is PlanningAttemptStatus.RUNNING:
            raise ValueError("a completed planning attempt cannot remain running")
        if not isinstance(status, PlanningAttemptStatus):
            raise TypeError("planning attempt status must be a PlanningAttemptStatus")
        finished_at = utcnow()
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE planning_attempts
                SET status = ?, result_json = ?, summary = ?, finished_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (
                    status.value,
                    _encode_json(result) if result is not None else None,
                    summary,
                    finished_at.isoformat(),
                    str(attempt_id),
                ),
            )
            if cursor.rowcount != 1:
                exists = connection.execute(
                    "SELECT 1 FROM planning_attempts WHERE id = ?",
                    (str(attempt_id),),
                ).fetchone()
                if exists is None:
                    raise KeyError(f"planning attempt {attempt_id} not found")
                raise ValueError("planning attempt has already completed")
            row = connection.execute(
                "SELECT * FROM planning_attempts WHERE id = ?",
                (str(attempt_id),),
            ).fetchone()
        return _row_to_planning_attempt(row)

    def list_planning_attempts(self, borg_id: UUID) -> list[PlanningAttempt]:
        """Return a Borg's attempts in stable invocation order."""
        with self.locked_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM planning_attempts
                WHERE borg_id = ?
                ORDER BY started_at, id
                """,
                (str(borg_id),),
            ).fetchall()
        return [_row_to_planning_attempt(row) for row in rows]

    def append_planning_question(self, question: PlanningQuestion) -> None:
        """Append one architect Q&A round."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO planning_questions(
                    id, borg_id, attempt_id, round, questions_json,
                    answers_json, asked_at, answered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(question.id),
                    str(question.borg_id),
                    str(question.attempt_id)
                    if question.attempt_id is not None
                    else None,
                    question.round,
                    _encode_json(question.questions),
                    _encode_json(question.answers)
                    if question.answers is not None
                    else None,
                    question.asked_at.isoformat(),
                    question.answered_at.isoformat()
                    if question.answered_at is not None
                    else None,
                ),
            )

    def answer_planning_question(
        self, question_id: UUID, answers: list[dict[str, object]]
    ) -> PlanningQuestion:
        """Record answers for a pending question round exactly once."""
        answered_at = utcnow()
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE planning_questions
                SET answers_json = ?, answered_at = ?
                WHERE id = ? AND answers_json IS NULL
                """,
                (_encode_json(answers), answered_at.isoformat(), str(question_id)),
            )
            if cursor.rowcount != 1:
                exists = connection.execute(
                    "SELECT 1 FROM planning_questions WHERE id = ?",
                    (str(question_id),),
                ).fetchone()
                if exists is None:
                    raise KeyError(f"planning question {question_id} not found")
                raise ValueError("planning question has already been answered")
            row = connection.execute(
                "SELECT * FROM planning_questions WHERE id = ?",
                (str(question_id),),
            ).fetchone()
        return _row_to_planning_question(row)

    def list_planning_questions(self, borg_id: UUID) -> list[PlanningQuestion]:
        """Return a Borg's complete Q&A history in round order."""
        with self.locked_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM planning_questions
                WHERE borg_id = ?
                ORDER BY round
                """,
                (str(borg_id),),
            ).fetchall()
        return [_row_to_planning_question(row) for row in rows]

    def append_planning_finding(self, finding: PlanningFinding) -> None:
        """Append one immutable tech-lead finding."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO planning_findings(
                    id, borg_id, attempt_id, round, severity, message,
                    suggestion, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(finding.id),
                    str(finding.borg_id),
                    str(finding.attempt_id),
                    finding.round,
                    finding.severity,
                    finding.message,
                    finding.suggestion,
                    finding.created_at.isoformat(),
                ),
            )

    def list_planning_findings(self, borg_id: UUID) -> list[PlanningFinding]:
        """Return all review findings in stable append order."""
        with self.locked_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM planning_findings
                WHERE borg_id = ?
                ORDER BY created_at, id
                """,
                (str(borg_id),),
            ).fetchall()
        return [_row_to_planning_finding(row) for row in rows]

    def append_plan_change_request(self, request: PlanChangeRequest) -> None:
        """Append one immutable human plan-revision request."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO plan_change_requests(
                    id, borg_id, round, note, decided_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(request.id),
                    str(request.borg_id),
                    request.round,
                    request.note,
                    request.decided_by,
                    request.created_at.isoformat(),
                ),
            )

    def list_plan_change_requests(self, borg_id: UUID) -> list[PlanChangeRequest]:
        """Return the full plan change-request thread in append order."""
        with self.locked_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM plan_change_requests
                WHERE borg_id = ?
                ORDER BY created_at, id
                """,
                (str(borg_id),),
            ).fetchall()
        return [_row_to_plan_change_request(row) for row in rows]

    def append_plan_approval(self, approval: PlanApproval) -> None:
        """Append one immutable, digest-bound plan approval."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO plan_approvals(
                    id, borg_id, attempt_id, plan_digest, manifest_json,
                    approved_by, approved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(approval.id),
                    str(approval.borg_id),
                    str(approval.attempt_id)
                    if approval.attempt_id is not None
                    else None,
                    approval.plan_digest,
                    _encode_json(approval.manifest),
                    approval.approved_by,
                    approval.approved_at.isoformat(),
                ),
            )

    def list_plan_approvals(self, borg_id: UUID) -> list[PlanApproval]:
        """Return a Borg's plan approvals in stable approval order."""
        with self.locked_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM plan_approvals
                WHERE borg_id = ?
                ORDER BY approved_at, id
                """,
                (str(borg_id),),
            ).fetchall()
        return [_row_to_plan_approval(row) for row in rows]

    def append_task_batch(self, batch: TaskBatch) -> None:
        """Append one immutable task-decomposition batch."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO task_batches(
                    id, borg_id, plan_approval_id, attempt_id, round, summary,
                    manifest_json, digest, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(batch.id),
                    str(batch.borg_id),
                    str(batch.plan_approval_id),
                    str(batch.attempt_id) if batch.attempt_id is not None else None,
                    batch.round,
                    batch.summary,
                    _encode_json(batch.manifest),
                    batch.digest,
                    batch.created_at.isoformat(),
                ),
            )

    def list_task_batches(self, borg_id: UUID) -> list[TaskBatch]:
        """Return a Borg's immutable task batches in append order."""
        with self.locked_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM task_batches
                WHERE borg_id = ?
                ORDER BY created_at, id
                """,
                (str(borg_id),),
            ).fetchall()
        return [_row_to_task_batch(row) for row in rows]

    def append_task_finding(self, finding: TaskFinding) -> None:
        """Append one immutable supervisor finding for a task batch."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO task_findings(
                    id, borg_id, batch_id, attempt_id, round, severity,
                    message, suggestion, task_ref, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(finding.id),
                    str(finding.borg_id),
                    str(finding.batch_id),
                    str(finding.attempt_id)
                    if finding.attempt_id is not None
                    else None,
                    finding.round,
                    finding.severity,
                    finding.message,
                    finding.suggestion,
                    finding.task_ref,
                    finding.created_at.isoformat(),
                ),
            )

    def list_task_findings(
        self, borg_id: UUID, *, batch_id: UUID | None = None
    ) -> list[TaskFinding]:
        """Return immutable task findings, optionally limited to one batch."""
        query = "SELECT * FROM task_findings WHERE borg_id = ?"
        parameters = [str(borg_id)]
        if batch_id is not None:
            query += " AND batch_id = ?"
            parameters.append(str(batch_id))
        query += " ORDER BY created_at, id"
        with self.locked_connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [_row_to_task_finding(row) for row in rows]

    def add_task_generation(
        self,
        generation: TaskGeneration,
        tasks: Iterable[TaskRecord] = (),
        dependencies: Iterable[TaskDependency] = (),
    ) -> None:
        """Atomically persist a generation and its immutable task graph rows."""
        if generation.status is not TaskGenerationStatus.PREPARING:
            raise ValueError("new task generations must start in preparing status")
        task_rows = list(tasks)
        dependency_rows = list(dependencies)
        if any(
            task.generation_id != generation.id or task.borg_id != generation.borg_id
            for task in task_rows
        ):
            raise ValueError("task records must belong to the added generation")
        task_ids = {task.id for task in task_rows}
        if len(task_ids) != len(task_rows):
            raise ValueError("task record IDs must be unique within a generation")
        if any(
            dependency.generation_id != generation.id
            or dependency.task_id not in task_ids
            or dependency.depends_on_task_id not in task_ids
            for dependency in dependency_rows
        ):
            raise ValueError(
                "task dependencies must connect records in the added generation"
            )

        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO task_generations(
                    id, borg_id, plan_approval_id, batch_id, status,
                    manifest_json, digest, created_at, current_at, superseded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(generation.id),
                    str(generation.borg_id),
                    str(generation.plan_approval_id),
                    str(generation.batch_id),
                    generation.status.value,
                    _encode_json(generation.manifest),
                    generation.digest,
                    generation.created_at.isoformat(),
                    generation.current_at.isoformat()
                    if generation.current_at is not None
                    else None,
                    generation.superseded_at.isoformat()
                    if generation.superseded_at is not None
                    else None,
                ),
            )
            connection.executemany(
                """
                INSERT INTO task_records(
                    id, generation_id, borg_id, task_ref, stage, stem,
                    position, title, complexity, task_json, manifest_json,
                    digest, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(task.id),
                        str(task.generation_id),
                        str(task.borg_id),
                        task.task_ref,
                        task.stage,
                        task.stem,
                        task.position,
                        task.title,
                        task.complexity.value,
                        _encode_json(task.task),
                        _encode_json(task.manifest),
                        task.digest,
                        task.created_at.isoformat(),
                    )
                    for task in task_rows
                ],
            )
            connection.executemany(
                """
                INSERT INTO task_dependencies(
                    id, generation_id, task_id, depends_on_task_id, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(dependency.id),
                        str(dependency.generation_id),
                        str(dependency.task_id),
                        str(dependency.depends_on_task_id),
                        dependency.created_at.isoformat(),
                    )
                    for dependency in dependency_rows
                ],
            )

    def get_task_generation(
        self, generation_id: UUID
    ) -> TaskGeneration | None:
        """Return one task generation by ID, if it exists."""
        with self.locked_connection() as connection:
            row = connection.execute(
                "SELECT * FROM task_generations WHERE id = ?",
                (str(generation_id),),
            ).fetchone()
        return _row_to_task_generation(row) if row is not None else None

    def get_current_task_generation(self, borg_id: UUID) -> TaskGeneration | None:
        """Return the sole SQLite-current generation for a Borg, if present."""
        with self.locked_connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM task_generations
                WHERE borg_id = ? AND status = 'current'
                """,
                (str(borg_id),),
            ).fetchone()
        return _row_to_task_generation(row) if row is not None else None

    def list_task_generations(self, borg_id: UUID) -> list[TaskGeneration]:
        """Return every generation without hiding superseded history."""
        with self.locked_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM task_generations
                WHERE borg_id = ?
                ORDER BY created_at, id
                """,
                (str(borg_id),),
            ).fetchall()
        return [_row_to_task_generation(row) for row in rows]

    def _promote_published_task_generation(
        self, generation_id: UUID, *, durable_root: Path
    ) -> TaskGeneration:
        """Commit publication after ``TaskPublisher`` crosses its durable seam.

        This is deliberately not a public store operation: a digest already held
        in SQLite cannot prove publication. The store independently verifies and
        fsyncs the final tree before opening the current-generation transaction;
        ``TaskPublisher`` remains the sole production caller and has already
        crossed the stricter stage-and-rename boundaries when it invokes this.
        """
        generation = self.get_task_generation(generation_id)
        if generation is None:
            raise KeyError(f"task generation {generation_id} not found")
        borg = self.get_borg(generation.borg_id)
        if borg is None:
            raise ValueError("task generation Borg not found")
        repository = self.get_repository(borg.repository_id)
        if repository is None:
            raise ValueError("task generation repository not found")
        expected_root = (
            repository.root / ".borg" / "tasks" / borg.name / str(generation.id)
        )
        if durable_root != expected_root:
            raise ValueError("durable task generation path does not match SQLite")
        records = self.list_task_records(generation.id)
        if not records:
            raise ValueError("durable task generation has no task records")
        expected_files = {
            Path(record.stage) / f"{record.stem}.md": record.digest
            for record in records
        }
        if len(expected_files) != len(records) or any(
            relative.is_absolute() or ".." in relative.parts
            for relative in expected_files
        ):
            raise ValueError("durable task generation contains an unsafe path")
        _verify_and_fsync_task_tree(
            repository.root, durable_root, expected_files
        )

        promoted_at = utcnow()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM task_generations WHERE id = ?",
                (str(generation_id),),
            ).fetchone()
            if row is None:
                raise KeyError(f"task generation {generation_id} not found")
            if row["status"] != TaskGenerationStatus.PREPARING.value:
                raise ValueError("only a preparing task generation can become current")
            connection.execute(
                """
                UPDATE task_generations
                SET status = 'superseded', superseded_at = ?
                WHERE borg_id = ? AND status = 'current'
                """,
                (promoted_at.isoformat(), row["borg_id"]),
            )
            connection.execute(
                """
                UPDATE task_generations
                SET status = 'current', current_at = ?
                WHERE id = ? AND status = 'preparing'
                """,
                (promoted_at.isoformat(), str(generation_id)),
            )
            promoted = connection.execute(
                "SELECT * FROM task_generations WHERE id = ?",
                (str(generation_id),),
            ).fetchone()
        return _row_to_task_generation(promoted)

    def list_task_records(self, generation_id: UUID) -> list[TaskRecord]:
        """Return immutable task metadata in generation position order."""
        with self.locked_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM task_records
                WHERE generation_id = ?
                ORDER BY position, id
                """,
                (str(generation_id),),
            ).fetchall()
        return [_row_to_task_record(row) for row in rows]

    def list_task_dependencies(
        self, generation_id: UUID
    ) -> list[TaskDependency]:
        """Return immutable dependency edges for one generation."""
        with self.locked_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM task_dependencies
                WHERE generation_id = ?
                ORDER BY created_at, id
                """,
                (str(generation_id),),
            ).fetchall()
        return [_row_to_task_dependency(row) for row in rows]

    def acquire_execution_run(
        self,
        borg_id: UUID,
        generation_id: UUID,
        *,
        lease_duration: timedelta,
        now: datetime | None = None,
    ) -> ExecutionRunAcquisition:
        """Atomically acquire a run or return the live operation already owning it."""
        acquired_at = _execution_time(now)
        lease_expires_at = _lease_expiry(acquired_at, lease_duration)
        with self.transaction() as connection:
            generation = connection.execute(
                """
                SELECT 1 FROM task_generations
                WHERE id = ? AND borg_id = ? AND status = 'current'
                """,
                (str(generation_id), str(borg_id)),
            ).fetchone()
            if generation is None:
                raise ValueError(
                    "execution runs require the Borg's current task generation"
                )

            live = connection.execute(
                "SELECT * FROM execution_runs WHERE borg_id = ? AND status = 'running'",
                (str(borg_id),),
            ).fetchone()
            if live is not None and datetime.fromisoformat(
                live["lease_expires_at"]
            ) <= acquired_at:
                self._interrupt_run(
                    connection,
                    live,
                    now=acquired_at,
                    reason="execution lease expired",
                    event_kind="run.expired",
                )
                live = None
            if live is not None:
                return ExecutionRunAcquisition(
                    operation_id=UUID(live["id"]),
                    owner_token=None,
                    acquired=False,
                )

            run = ExecutionRun(
                borg_id=borg_id,
                generation_id=generation_id,
                started_at=acquired_at,
                heartbeat_at=acquired_at,
                lease_expires_at=lease_expires_at,
            )
            self._insert_execution_run(connection, run)
            task_rows = connection.execute(
                "SELECT id FROM task_records WHERE generation_id = ?",
                (str(generation_id),),
            ).fetchall()
            for task_row in task_rows:
                runtime = TaskRuntime(
                    generation_id=generation_id,
                    task_id=UUID(task_row["id"]),
                    created_at=acquired_at,
                    updated_at=acquired_at,
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO task_runtimes(
                        id, generation_id, task_id, status, resume_phase,
                        review_round, state_reason, branch, worktree_path,
                        last_run_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(runtime.id),
                        str(runtime.generation_id),
                        str(runtime.task_id),
                        runtime.status.value,
                        runtime.resume_phase,
                        runtime.review_round,
                        runtime.state_reason,
                        runtime.branch,
                        runtime.worktree_path,
                        None,
                        acquired_at.isoformat(),
                        acquired_at.isoformat(),
                    ),
                )
            self._insert_execution_event(
                connection,
                run_id=run.id,
                kind="run.acquired",
                payload={"generation_id": str(generation_id)},
                created_at=acquired_at,
            )
        return ExecutionRunAcquisition(
            operation_id=run.id,
            owner_token=run.owner_token,
            acquired=True,
        )

    def renew_execution_run(
        self,
        run_id: UUID,
        owner_token: str,
        *,
        lease_duration: timedelta,
        now: datetime | None = None,
    ) -> ExecutionRun:
        """Renew a live run and all of its open task claims under one write lock."""
        heartbeat_at = _execution_time(now)
        lease_expires_at = _lease_expiry(heartbeat_at, lease_duration)
        expired = False
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM execution_runs WHERE id = ?",
                (str(run_id),),
            ).fetchone()
            if row is None:
                raise KeyError(f"execution run {run_id} not found")
            if row["owner_token"] != owner_token:
                raise ExecutionOwnershipError("execution run ownership changed")
            if row["status"] != ExecutionRunStatus.RUNNING.value:
                raise ExecutionOwnershipError("execution run is no longer running")
            if datetime.fromisoformat(row["lease_expires_at"]) <= heartbeat_at:
                self._interrupt_run(
                    connection,
                    row,
                    now=heartbeat_at,
                    reason="execution lease expired",
                    event_kind="run.expired",
                )
                expired = True
            else:
                self._reconcile_expired_claims(
                    connection, run_id, now=heartbeat_at
                )
                connection.execute(
                    """
                    UPDATE execution_runs
                    SET heartbeat_at = ?, lease_expires_at = ?
                    WHERE id = ? AND owner_token = ? AND status = 'running'
                    """,
                    (
                        heartbeat_at.isoformat(),
                        lease_expires_at.isoformat(),
                        str(run_id),
                        owner_token,
                    ),
                )
                connection.execute(
                    """
                    UPDATE task_claims
                    SET lease_expires_at = ?
                    WHERE run_id = ? AND released_at IS NULL
                      AND lease_expires_at > ?
                    """,
                    (
                        lease_expires_at.isoformat(),
                        str(run_id),
                        heartbeat_at.isoformat(),
                    ),
                )
                self._insert_execution_event(
                    connection,
                    run_id=run_id,
                    kind="run.lease_renewed",
                    payload={"lease_expires_at": lease_expires_at.isoformat()},
                    created_at=heartbeat_at,
                )
            updated = connection.execute(
                "SELECT * FROM execution_runs WHERE id = ?", (str(run_id),)
            ).fetchone()
        if expired:
            raise ExecutionOwnershipError("execution run lease expired")
        return _row_to_execution_run(updated)

    def finish_execution_run(
        self,
        run_id: UUID,
        owner_token: str,
        *,
        status: ExecutionRunStatus,
        now: datetime | None = None,
    ) -> ExecutionRun:
        """Finish an owned run after its scheduler has no in-flight work."""
        finished_at = _execution_time(now)
        if status not in {
            ExecutionRunStatus.COMPLETED,
            ExecutionRunStatus.FAILED,
        }:
            raise ValueError("scheduler may only finish a run completed or failed")
        with self.transaction() as connection:
            run = self._require_live_run(
                connection, run_id, owner_token, now=finished_at
            )
            runtime_rows = connection.execute(
                """
                SELECT status FROM task_runtimes
                WHERE generation_id = ?
                """,
                (run["generation_id"],),
            ).fetchall()
            runtime_statuses = {
                TaskRuntimeStatus(row["status"]) for row in runtime_rows
            }
            if runtime_statuses & _ACTIVE_TASK_STATUSES:
                raise ValueError("execution run still has active task runtimes")
            all_done = all(
                TaskRuntimeStatus(row["status"]) is TaskRuntimeStatus.DONE
                for row in runtime_rows
            )
            if status is ExecutionRunStatus.COMPLETED and not all_done:
                raise ValueError("completed execution run still has unfinished tasks")
            if status is ExecutionRunStatus.FAILED and all_done:
                raise ValueError("failed execution run has no unfinished tasks")
            cursor = connection.execute(
                """
                UPDATE execution_runs
                SET status = ?, finished_at = ?
                WHERE id = ? AND owner_token = ? AND status = 'running'
                """,
                (
                    status.value,
                    finished_at.isoformat(),
                    str(run_id),
                    owner_token,
                ),
            )
            if cursor.rowcount != 1:
                raise ExecutionOwnershipError("execution run ownership changed")
            self._insert_execution_event(
                connection,
                run_id=run_id,
                kind=f"run.{status.value}",
                payload={},
                created_at=finished_at,
            )
            updated = connection.execute(
                "SELECT * FROM execution_runs WHERE id = ?", (str(run_id),)
            ).fetchone()
        return _row_to_execution_run(updated)

    def claim_dependency_ready_task(
        self,
        run_id: UUID,
        owner_token: str,
        *,
        lease_duration: timedelta,
        now: datetime | None = None,
    ) -> TaskClaim | None:
        """Claim the first unowned task whose dependencies are all complete."""
        claimed_at = _execution_time(now)
        requested_expiry = _lease_expiry(claimed_at, lease_duration)
        with self.transaction() as connection:
            run = self._require_live_run(
                connection, run_id, owner_token, now=claimed_at
            )
            self._reconcile_expired_claims(connection, run_id, now=claimed_at)
            run_expiry = datetime.fromisoformat(run["lease_expires_at"])
            lease_expires_at = min(requested_expiry, run_expiry)
            task = connection.execute(
                """
                SELECT task.id, runtime.resume_phase
                FROM task_records AS task
                JOIN task_runtimes AS runtime ON runtime.task_id = task.id
                WHERE task.generation_id = ?
                  AND runtime.status = 'pending'
                  AND NOT EXISTS (
                      SELECT 1 FROM task_claims AS claim
                      WHERE claim.task_id = task.id AND claim.released_at IS NULL
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM task_dependencies AS dependency
                      LEFT JOIN task_runtimes AS prerequisite
                        ON prerequisite.task_id = dependency.depends_on_task_id
                      WHERE dependency.task_id = task.id
                        AND (
                            prerequisite.id IS NULL
                            OR prerequisite.status != 'done'
                        )
                  )
                ORDER BY task.position, task.id
                LIMIT 1
                """,
                (run["generation_id"],),
            ).fetchone()
            if task is None:
                return None

            claim = TaskClaim(
                run_id=run_id,
                task_id=UUID(task["id"]),
                resume_phase=task["resume_phase"],
                claimed_at=claimed_at,
                lease_expires_at=lease_expires_at,
            )
            connection.execute(
                """
                INSERT INTO task_claims(
                    id, run_id, generation_id, task_id, claim_token,
                    resume_phase, claimed_at, lease_expires_at, released_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    str(claim.id),
                    str(run_id),
                    run["generation_id"],
                    str(claim.task_id),
                    claim.claim_token,
                    claim.resume_phase,
                    claimed_at.isoformat(),
                    lease_expires_at.isoformat(),
                ),
            )
            connection.execute(
                """
                UPDATE task_runtimes
                SET status = 'claimed', last_run_id = ?, state_reason = NULL,
                    updated_at = ?
                WHERE task_id = ? AND status = 'pending'
                """,
                (str(run_id), claimed_at.isoformat(), str(claim.task_id)),
            )
            self._insert_execution_event(
                connection,
                run_id=run_id,
                task_id=claim.task_id,
                kind="task.claimed",
                payload={
                    "claim_id": str(claim.id),
                    "resume_phase": claim.resume_phase,
                },
                created_at=claimed_at,
            )
        return claim

    def transition_task_runtime(
        self,
        run_id: UUID,
        owner_token: str,
        claim_id: UUID,
        claim_token: str,
        *,
        expected_status: TaskRuntimeStatus,
        new_status: TaskRuntimeStatus,
        resume_phase: str | None = None,
        review_round: int | None = None,
        state_reason: str | None = None,
        branch: str | None = None,
        worktree_path: str | None = None,
        now: datetime | None = None,
    ) -> TaskRuntime:
        """Compare-and-set one task phase while both run leases are still owned."""
        transitioned_at = _execution_time(now)
        if not isinstance(expected_status, TaskRuntimeStatus) or not isinstance(
            new_status, TaskRuntimeStatus
        ):
            raise TypeError("task runtime states must be TaskRuntimeStatus values")
        next_phase = resume_phase or new_status.value
        if not next_phase.strip():
            raise ValueError("task resume phase must not be empty")
        if review_round is not None and review_round < 0:
            raise ValueError("task review round must not be negative")
        if (branch is None) != (worktree_path is None):
            raise ValueError(
                "task branch and worktree path must be transitioned together"
            )
        if new_status not in _TASK_TRANSITIONS.get(expected_status, frozenset()):
            raise ValueError(
                f"illegal task phase transition: {expected_status.value} -> "
                f"{new_status.value}"
            )

        with self.transaction() as connection:
            self._require_live_run(connection, run_id, owner_token, now=transitioned_at)
            claim = connection.execute(
                """
                SELECT * FROM task_claims
                WHERE id = ? AND run_id = ? AND claim_token = ?
                  AND released_at IS NULL AND lease_expires_at > ?
                """,
                (
                    str(claim_id),
                    str(run_id),
                    claim_token,
                    transitioned_at.isoformat(),
                ),
            ).fetchone()
            if claim is None:
                raise ExecutionOwnershipError("task claim is no longer owned")
            runtime_identity = connection.execute(
                """
                SELECT branch, worktree_path FROM task_runtimes
                WHERE task_id = ?
                """,
                (claim["task_id"],),
            ).fetchone()
            if branch is not None and (
                runtime_identity["branch"], runtime_identity["worktree_path"]
            ) not in {(None, None), (branch, worktree_path)}:
                raise StaleTaskRuntimeError(
                    "task worktree identity cannot be replaced"
                )
            assignments = [
                "status = ?",
                "resume_phase = ?",
                "state_reason = ?",
                "updated_at = ?",
            ]
            parameters: list[object] = [
                new_status.value,
                next_phase,
                state_reason,
                transitioned_at.isoformat(),
            ]
            if review_round is not None:
                assignments.append("review_round = ?")
                parameters.append(review_round)
            if branch is not None:
                assignments.append("branch = ?")
                parameters.append(branch)
            if worktree_path is not None:
                assignments.append("worktree_path = ?")
                parameters.append(worktree_path)
            parameters.extend((claim["task_id"], expected_status.value))
            cursor = connection.execute(
                f"""
                UPDATE task_runtimes SET {", ".join(assignments)}
                WHERE task_id = ? AND status = ?
                """,
                parameters,
            )
            if cursor.rowcount != 1:
                raise StaleTaskRuntimeError(
                    "task phase changed before compare-and-set transition"
                )
            task_id = UUID(claim["task_id"])
            self._insert_execution_event(
                connection,
                run_id=run_id,
                task_id=task_id,
                kind="task.phase_transitioned",
                payload={
                    "claim_id": str(claim_id),
                    "from": expected_status.value,
                    "to": new_status.value,
                    "resume_phase": next_phase,
                },
                created_at=transitioned_at,
            )
            terminal_without_cleanup = (
                new_status in _TERMINAL_TASK_STATUSES
                and not self._claim_has_pending_compose(connection, claim_id)
            )
            if terminal_without_cleanup:
                connection.execute(
                    "UPDATE task_claims SET released_at = ? WHERE id = ?",
                    (transitioned_at.isoformat(), str(claim_id)),
                )
            updated = connection.execute(
                "SELECT * FROM task_runtimes WHERE task_id = ?", (str(task_id),)
            ).fetchone()
        return _row_to_task_runtime(updated)

    def interrupt_execution_run(
        self,
        run_id: UUID,
        owner_token: str,
        *,
        reason: str = "execution interrupted",
        now: datetime | None = None,
    ) -> list[ComposeResource]:
        """Cooperatively stop an owned run and return cleanup still required."""
        interrupted_at = _execution_time(now)
        if not reason.strip():
            raise ValueError("execution interruption reason must not be empty")
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM execution_runs WHERE id = ?", (str(run_id),)
            ).fetchone()
            if row is None:
                raise KeyError(f"execution run {run_id} not found")
            if row["owner_token"] != owner_token:
                raise ExecutionOwnershipError("execution run ownership changed")
            if row["status"] == ExecutionRunStatus.RUNNING.value:
                self._interrupt_run(
                    connection,
                    row,
                    now=interrupted_at,
                    reason=reason,
                    event_kind="run.interrupted",
                )
            rows = self._stale_compose_rows(connection, run_id=run_id)
        return [_row_to_compose_resource(row) for row in rows]

    def reconcile_expired_execution_runs(
        self, *, now: datetime | None = None
    ) -> list[ComposeResource]:
        """Interrupt every expired run and identify exact pending Compose resources."""
        reconciled_at = _execution_time(now)
        with self.transaction() as connection:
            expired = connection.execute(
                """
                SELECT * FROM execution_runs
                WHERE status = 'running' AND lease_expires_at <= ?
                ORDER BY lease_expires_at, id
                """,
                (reconciled_at.isoformat(),),
            ).fetchall()
            for row in expired:
                self._interrupt_run(
                    connection,
                    row,
                    now=reconciled_at,
                    reason="execution lease expired",
                    event_kind="run.expired",
                )
            rows = self._stale_compose_rows(connection)
        return [_row_to_compose_resource(row) for row in rows]

    def list_stale_compose_resources(
        self, run_id: UUID | None = None
    ) -> list[ComposeResource]:
        """Return stale resources whose exact project awaits teardown."""
        with self.locked_connection() as connection:
            rows = self._stale_compose_rows(connection, run_id=run_id)
        return [_row_to_compose_resource(row) for row in rows]

    def confirm_compose_project_cleanup(
        self,
        run_id: UUID,
        task_id: UUID,
        project_name: str,
        *,
        command: Iterable[str] | None = None,
        now: datetime | None = None,
    ) -> list[ComposeResource]:
        """Record successful external teardown and release its blocked claim."""
        cleaned_at = _execution_time(now)
        if not project_name.strip():
            raise ValueError("Compose project name must not be empty")
        with self.transaction() as connection:
            resources = connection.execute(
                """
                SELECT * FROM compose_resources
                WHERE run_id = ? AND task_id = ? AND project_name = ?
                ORDER BY created_at, id
                """,
                (str(run_id), str(task_id), project_name),
            ).fetchall()
            if not resources:
                raise KeyError("Compose project identity not found")
            self._reconcile_expired_claims(connection, run_id, now=cleaned_at)
            pending = [
                row
                for row in resources
                if not self._compose_resource_cleanup_confirmed(
                    connection,
                    run_id=run_id,
                    task_id=task_id,
                    resource_id=UUID(row["id"]),
                )
            ]
            if pending:
                command_payload = list(command or ())
                self._insert_execution_event(
                    connection,
                    run_id=run_id,
                    task_id=task_id,
                    kind="compose.stopped",
                    payload={
                        "project_name": project_name,
                        "resource_ids": [row["id"] for row in pending],
                        "command": command_payload,
                    },
                    created_at=cleaned_at,
                )
                self._insert_execution_event(
                    connection,
                    run_id=run_id,
                    task_id=task_id,
                    kind="compose.cleanup_completed",
                    payload={
                        "project_name": project_name,
                        "resource_ids": [row["id"] for row in pending],
                        "command": command_payload,
                    },
                    created_at=cleaned_at,
                )
            claim_ids = {UUID(row["claim_id"]) for row in resources}
            for claim_id in claim_ids:
                if not self._claim_has_pending_compose(connection, claim_id):
                    claim = connection.execute(
                        "SELECT lease_expires_at FROM task_claims WHERE id = ?",
                        (str(claim_id),),
                    ).fetchone()
                    cleanup_failures = connection.execute(
                        """
                        SELECT payload_json FROM execution_events
                        WHERE run_id = ? AND task_id = ?
                          AND kind = 'compose.cleanup_failed'
                          AND json_extract(payload_json, '$.project_name') = ?
                        """,
                        (str(run_id), str(task_id), project_name),
                    ).fetchall()
                    reclaimable_statuses = {
                        status.value for status in _ACTIVE_TASK_STATUSES
                    } | {TaskRuntimeStatus.PENDING.value}
                    reset_cleanup_block = any(
                        json.loads(row["payload_json"]).get("previous_status")
                        in reclaimable_statuses
                        for row in cleanup_failures
                    )
                    self._release_reconciled_claim(
                        connection,
                        claim_id,
                        now=cleaned_at,
                        reason="Compose cleanup completed",
                        force=(
                            datetime.fromisoformat(claim["lease_expires_at"])
                            <= cleaned_at
                        ),
                        reset_cleanup_block=reset_cleanup_block,
                    )
        return [_row_to_compose_resource(row) for row in resources]

    def record_compose_cleanup_failure(
        self,
        run_id: UUID,
        task_id: UUID,
        project_name: str,
        *,
        command: Iterable[str],
        error: str,
        now: datetime | None = None,
    ) -> bool:
        """Persist a failed teardown only while its project remains pending."""
        failed_at = _execution_time(now)
        command_payload = list(command)
        if not project_name.strip() or not command_payload or not error.strip():
            raise ValueError("Compose cleanup failure details must not be empty")
        with self.transaction() as connection:
            resource = connection.execute(
                """
                SELECT resource.id
                FROM compose_resources AS resource
                WHERE resource.run_id = ?
                  AND resource.task_id = ?
                  AND resource.project_name = ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM execution_events AS event,
                           json_each(event.payload_json, '$.resource_ids') AS cleaned
                      WHERE event.run_id = resource.run_id
                        AND event.task_id = resource.task_id
                        AND event.kind = 'compose.cleanup_completed'
                        AND cleaned.value = resource.id
                  )
                LIMIT 1
                """,
                (str(run_id), str(task_id), project_name),
            ).fetchone()
            if resource is None:
                known = connection.execute(
                    """
                    SELECT 1 FROM compose_resources
                    WHERE run_id = ? AND task_id = ? AND project_name = ?
                    LIMIT 1
                    """,
                    (str(run_id), str(task_id), project_name),
                ).fetchone()
                if known is None:
                    raise KeyError("Compose project identity not found")
                return False
            runtime = connection.execute(
                "SELECT status FROM task_runtimes WHERE task_id = ?",
                (str(task_id),),
            ).fetchone()
            if runtime is None:
                raise KeyError("task runtime not found")
            self._insert_execution_event(
                connection,
                run_id=run_id,
                task_id=task_id,
                kind="compose.cleanup_failed",
                payload={
                    "project_name": project_name,
                    "command": command_payload,
                    "error": error,
                    "previous_status": runtime["status"],
                },
                created_at=failed_at,
            )
            reason = (
                f"Compose cleanup failed for project {project_name!r}; command: "
                f"{shlex.join(command_payload)}; {error}"
            )
            connection.execute(
                """
                UPDATE task_runtimes
                SET status = 'blocked', state_reason = ?, updated_at = ?
                WHERE task_id = ? AND status NOT IN ('done', 'failed')
                """,
                (reason, failed_at.isoformat(), str(task_id)),
            )
            return True

    @staticmethod
    def _insert_execution_run(
        connection: sqlite3.Connection, run: ExecutionRun
    ) -> None:
        connection.execute(
            """
            INSERT INTO execution_runs(
                id, borg_id, generation_id, owner_token, status, started_at,
                heartbeat_at, lease_expires_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(run.id),
                str(run.borg_id),
                str(run.generation_id),
                run.owner_token,
                run.status.value,
                run.started_at.isoformat(),
                run.heartbeat_at.isoformat() if run.heartbeat_at else None,
                run.lease_expires_at.isoformat(),
                run.finished_at.isoformat() if run.finished_at else None,
            ),
        )

    @staticmethod
    def _insert_execution_event(
        connection: sqlite3.Connection,
        *,
        run_id: UUID,
        kind: str,
        payload: dict[str, object],
        created_at: datetime,
        task_id: UUID | None = None,
        attempt_id: UUID | None = None,
    ) -> None:
        event = ExecutionEvent(
            run_id=run_id,
            task_id=task_id,
            attempt_id=attempt_id,
            kind=kind,
            payload=payload,
            created_at=created_at,
        )
        connection.execute(
            """
            INSERT INTO execution_events(
                id, run_id, task_id, attempt_id, kind, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(event.id),
                str(run_id),
                str(task_id) if task_id else None,
                str(attempt_id) if attempt_id else None,
                kind,
                _encode_json(payload),
                created_at.isoformat(),
            ),
        )

    def _require_live_run(
        self,
        connection: sqlite3.Connection,
        run_id: UUID,
        owner_token: str,
        *,
        now: datetime,
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM execution_runs WHERE id = ?", (str(run_id),)
        ).fetchone()
        if row is None:
            raise KeyError(f"execution run {run_id} not found")
        if row["owner_token"] != owner_token:
            raise ExecutionOwnershipError("execution run ownership changed")
        if row["status"] != ExecutionRunStatus.RUNNING.value:
            raise ExecutionOwnershipError("execution run is no longer running")
        if datetime.fromisoformat(row["lease_expires_at"]) <= now:
            raise ExecutionOwnershipError("execution run lease expired")
        return row

    def _require_live_claim(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: UUID,
        owner_token: str,
        claim_id: UUID,
        claim_token: str,
        task_id: UUID,
        now: datetime,
    ) -> sqlite3.Row:
        self._require_live_run(connection, run_id, owner_token, now=now)
        claim = connection.execute(
            """
            SELECT * FROM task_claims
            WHERE id = ? AND run_id = ? AND task_id = ? AND claim_token = ?
              AND released_at IS NULL AND lease_expires_at > ?
            """,
            (
                str(claim_id),
                str(run_id),
                str(task_id),
                claim_token,
                now.isoformat(),
            ),
        ).fetchone()
        if claim is None:
            raise ExecutionOwnershipError("task claim is no longer owned")
        return claim

    def _interrupt_run(
        self,
        connection: sqlite3.Connection,
        run: sqlite3.Row,
        *,
        now: datetime,
        reason: str,
        event_kind: str,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE execution_runs
            SET status = 'cancelled', finished_at = ?
            WHERE id = ? AND status = 'running'
            """,
            (now.isoformat(), run["id"]),
        )
        if cursor.rowcount != 1:
            return
        self._interrupt_open_attempts(
            connection,
            run_id=UUID(run["id"]),
            now=now,
            reason=reason,
        )
        claims = connection.execute(
            """
            SELECT id, task_id FROM task_claims
            WHERE run_id = ? AND released_at IS NULL
            """,
            (run["id"],),
        ).fetchall()
        for claim in claims:
            claim_id = UUID(claim["id"])
            task_id = UUID(claim["task_id"])
            self._insert_execution_event(
                connection,
                run_id=UUID(run["id"]),
                task_id=task_id,
                kind="task.interrupted",
                payload={"claim_id": claim["id"], "reason": reason},
                created_at=now,
            )
            if self._claim_has_pending_compose(connection, claim_id):
                connection.execute(
                    """
                    UPDATE task_runtimes SET state_reason = ?, updated_at = ?
                    WHERE task_id = ? AND status NOT IN ('done', 'blocked', 'failed')
                    """,
                    (
                        f"{reason}; awaiting Compose cleanup",
                        now.isoformat(),
                        claim["task_id"],
                    ),
                )
            else:
                self._release_reconciled_claim(
                    connection, claim_id, now=now, reason=reason
                )
        self._insert_execution_event(
            connection,
            run_id=UUID(run["id"]),
            kind=event_kind,
            payload={"reason": reason},
            created_at=now,
        )

    def _interrupt_open_attempts(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: UUID,
        now: datetime,
        reason: str,
        claim_id: UUID | None = None,
    ) -> None:
        for table, event_kind in (
            ("environment_attempts", "environment.attempt_interrupted"),
            ("agent_attempts", "agent.attempt_interrupted"),
        ):
            claim_parameter = str(claim_id) if claim_id is not None else None
            attempts = connection.execute(
                f"""
                SELECT id, task_id FROM {table} AS attempt
                WHERE run_id = ? AND status = 'running'
                  AND (? IS NULL OR claim_id = ?)
                  AND NOT EXISTS (
                      SELECT 1 FROM execution_events AS event
                      WHERE event.attempt_id = attempt.id
                        AND event.kind IN (
                            'environment.attempt_finished',
                            'environment.attempt_interrupted',
                            'agent.attempt_finished',
                            'agent.attempt_interrupted'
                        )
                  )
                ORDER BY started_at, id
                """,
                (str(run_id), claim_parameter, claim_parameter),
            ).fetchall()
            for attempt in attempts:
                self._insert_execution_event(
                    connection,
                    run_id=run_id,
                    task_id=UUID(attempt["task_id"]),
                    attempt_id=UUID(attempt["id"]),
                    kind=event_kind,
                    payload={"reason": reason},
                    created_at=now,
                )

    def _release_reconciled_claim(
        self,
        connection: sqlite3.Connection,
        claim_id: UUID,
        *,
        now: datetime,
        reason: str,
        force: bool = False,
        reset_cleanup_block: bool = False,
    ) -> None:
        claim = connection.execute(
            "SELECT * FROM task_claims WHERE id = ?", (str(claim_id),)
        ).fetchone()
        if claim is None or claim["released_at"] is not None:
            return
        runtime = connection.execute(
            "SELECT status FROM task_runtimes WHERE task_id = ?",
            (claim["task_id"],),
        ).fetchone()
        run = connection.execute(
            "SELECT status FROM execution_runs WHERE id = ?", (claim["run_id"],)
        ).fetchone()
        runtime_status = TaskRuntimeStatus(runtime["status"])
        may_release = force or (
            run["status"] != ExecutionRunStatus.RUNNING.value
            or runtime_status in _TERMINAL_TASK_STATUSES
        )
        if not may_release:
            return
        connection.execute(
            "UPDATE task_claims SET released_at = ? WHERE id = ?",
            (now.isoformat(), str(claim_id)),
        )
        if runtime_status in _ACTIVE_TASK_STATUSES or (
            runtime_status is TaskRuntimeStatus.BLOCKED and reset_cleanup_block
        ):
            connection.execute(
                """
                UPDATE task_runtimes
                SET status = 'pending', state_reason = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (reason, now.isoformat(), claim["task_id"]),
            )

    def _reconcile_expired_claims(
        self,
        connection: sqlite3.Connection,
        run_id: UUID,
        *,
        now: datetime,
    ) -> None:
        expired = connection.execute(
            """
            SELECT id, task_id FROM task_claims
            WHERE run_id = ? AND released_at IS NULL AND lease_expires_at <= ?
              AND NOT EXISTS (
                  SELECT 1 FROM execution_events AS event
                  WHERE event.run_id = task_claims.run_id
                    AND event.task_id = task_claims.task_id
                    AND event.kind = 'task.claim_expired'
                    AND json_extract(event.payload_json, '$.claim_id') =
                        task_claims.id
              )
            """,
            (str(run_id), now.isoformat()),
        ).fetchall()
        for claim in expired:
            claim_id = UUID(claim["id"])
            self._interrupt_open_attempts(
                connection,
                run_id=run_id,
                claim_id=claim_id,
                now=now,
                reason="task claim expired",
            )
            self._insert_execution_event(
                connection,
                run_id=run_id,
                task_id=UUID(claim["task_id"]),
                kind="task.claim_expired",
                payload={"claim_id": claim["id"]},
                created_at=now,
            )
            if self._claim_has_pending_compose(connection, claim_id):
                connection.execute(
                    """
                    UPDATE task_runtimes
                    SET state_reason = ?, updated_at = ?
                    WHERE task_id = ? AND status NOT IN ('done', 'blocked', 'failed')
                    """,
                    (
                        "task claim expired; awaiting Compose cleanup",
                        now.isoformat(),
                        claim["task_id"],
                    ),
                )
            else:
                self._release_reconciled_claim(
                    connection,
                    claim_id,
                    now=now,
                    reason="task claim expired",
                    force=True,
                )

    @staticmethod
    def _claim_has_pending_compose(
        connection: sqlite3.Connection, claim_id: UUID
    ) -> bool:
        row = connection.execute(
            """
            SELECT 1
            FROM compose_resources AS resource
            WHERE resource.claim_id = ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM execution_events AS event,
                       json_each(event.payload_json, '$.resource_ids') AS cleaned
                  WHERE event.run_id = resource.run_id
                    AND event.task_id = resource.task_id
                    AND event.kind = 'compose.cleanup_completed'
                    AND cleaned.value = resource.id
              )
            LIMIT 1
            """,
            (str(claim_id),),
        ).fetchone()
        return row is not None

    @staticmethod
    def _compose_resource_cleanup_confirmed(
        connection: sqlite3.Connection,
        *,
        run_id: UUID,
        task_id: UUID,
        resource_id: UUID,
    ) -> bool:
        row = connection.execute(
            """
            SELECT 1
            FROM execution_events AS event,
                 json_each(event.payload_json, '$.resource_ids') AS cleaned
            WHERE event.run_id = ? AND event.task_id = ?
              AND event.kind = 'compose.cleanup_completed'
              AND cleaned.value = ?
            LIMIT 1
            """,
            (str(run_id), str(task_id), str(resource_id)),
        ).fetchone()
        return row is not None

    @staticmethod
    def _stale_compose_rows(
        connection: sqlite3.Connection, *, run_id: UUID | None = None
    ) -> list[sqlite3.Row]:
        filters = "AND resource.run_id = ?" if run_id is not None else ""
        parameters = (str(run_id),) if run_id is not None else ()
        return connection.execute(
            f"""
            SELECT resource.*
            FROM compose_resources AS resource
            JOIN execution_runs AS run ON run.id = resource.run_id
            WHERE (
                run.status != 'running'
                OR EXISTS (
                    SELECT 1 FROM execution_events AS expired
                    WHERE expired.run_id = resource.run_id
                      AND expired.task_id = resource.task_id
                      AND expired.kind = 'task.claim_expired'
                      AND json_extract(expired.payload_json, '$.claim_id') =
                          resource.claim_id
                )
              )
              {filters}
              AND NOT EXISTS (
                  SELECT 1
                  FROM execution_events AS event,
                       json_each(event.payload_json, '$.resource_ids') AS cleaned
                  WHERE event.run_id = resource.run_id
                    AND event.task_id = resource.task_id
                    AND event.kind = 'compose.cleanup_completed'
                    AND cleaned.value = resource.id
              )
            ORDER BY resource.project_name, resource.created_at, resource.id
            """,
            parameters,
        ).fetchall()

    def add_execution_run(self, run: ExecutionRun) -> None:
        """Persist a newly leased execution run."""
        with self.transaction() as connection:
            self._insert_execution_run(connection, run)

    def get_execution_run(self, run_id: UUID) -> ExecutionRun | None:
        """Return an execution run by ID."""
        with self.locked_connection() as connection:
            row = connection.execute(
                "SELECT * FROM execution_runs WHERE id = ?", (str(run_id),)
            ).fetchone()
        return _row_to_execution_run(row) if row is not None else None

    def list_execution_runs(self, borg_id: UUID) -> list[ExecutionRun]:
        """Return a Borg's execution runs in stable creation order."""
        with self.locked_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM execution_runs
                WHERE borg_id = ?
                ORDER BY started_at, id
                """,
                (str(borg_id),),
            ).fetchall()
        return [_row_to_execution_run(row) for row in rows]

    def execution_run_owned_by(self, run_id: UUID, owner_token: str) -> bool:
        """Return whether ``owner_token`` owns ``run_id`` without exposing it."""
        with self.locked_connection() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM execution_runs
                WHERE id = ? AND owner_token = ?
                """,
                (str(run_id), owner_token),
            ).fetchone()
        return row is not None

    def add_task_runtime(self, runtime: TaskRuntime) -> None:
        """Create the durable runtime projection for one generated task."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO task_runtimes(
                    id, generation_id, task_id, status, resume_phase,
                    review_round, state_reason, branch, worktree_path,
                    last_run_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(runtime.id),
                    str(runtime.generation_id),
                    str(runtime.task_id),
                    runtime.status.value,
                    runtime.resume_phase,
                    runtime.review_round,
                    runtime.state_reason,
                    runtime.branch,
                    runtime.worktree_path,
                    str(runtime.last_run_id) if runtime.last_run_id else None,
                    runtime.created_at.isoformat(),
                    runtime.updated_at.isoformat(),
                ),
            )

    def get_task_runtime(self, task_id: UUID) -> TaskRuntime | None:
        """Return the durable runtime projection for one task."""
        with self.locked_connection() as connection:
            row = connection.execute(
                "SELECT * FROM task_runtimes WHERE task_id = ?", (str(task_id),)
            ).fetchone()
        return _row_to_task_runtime(row) if row is not None else None

    def assign_task_worktree(
        self,
        run_id: UUID,
        owner_token: str,
        task_id: UUID,
        *,
        branch: str,
        worktree_path: str,
        now: datetime | None = None,
    ) -> TaskRuntime:
        """Persist one immutable task Git identity under the live run lease."""
        assigned_at = _execution_time(now)
        if not branch.strip() or not worktree_path.strip():
            raise ValueError("task branch and worktree path must not be empty")
        with self.transaction() as connection:
            run = self._require_live_run(
                connection, run_id, owner_token, now=assigned_at
            )
            row = connection.execute(
                """
                SELECT * FROM task_runtimes
                WHERE task_id = ? AND generation_id = ?
                """,
                (str(task_id), run["generation_id"]),
            ).fetchone()
            if row is None:
                raise ValueError("task does not belong to the execution run")
            existing = (row["branch"], row["worktree_path"])
            requested = (branch, worktree_path)
            if existing == requested:
                return _row_to_task_runtime(row)
            if existing != (None, None):
                raise StaleTaskRuntimeError(
                    "task worktree identity is already assigned"
                )
            cursor = connection.execute(
                """
                UPDATE task_runtimes
                SET branch = ?, worktree_path = ?, updated_at = ?
                WHERE task_id = ? AND branch IS NULL AND worktree_path IS NULL
                """,
                (
                    branch,
                    worktree_path,
                    assigned_at.isoformat(),
                    str(task_id),
                ),
            )
            if cursor.rowcount != 1:
                raise StaleTaskRuntimeError(
                    "task worktree identity changed before assignment"
                )
            self._insert_execution_event(
                connection,
                run_id=run_id,
                task_id=task_id,
                kind="task.worktree_assigned",
                payload={"branch": branch, "worktree_path": worktree_path},
                created_at=assigned_at,
            )
            updated = connection.execute(
                "SELECT * FROM task_runtimes WHERE task_id = ?", (str(task_id),)
            ).fetchone()
        return _row_to_task_runtime(updated)

    def list_task_runtimes(self, generation_id: UUID) -> list[TaskRuntime]:
        """Return runtime projections in generated-task order."""
        with self.locked_connection() as connection:
            rows = connection.execute(
                """
                SELECT runtime.*
                FROM task_runtimes AS runtime
                JOIN task_records AS task ON task.id = runtime.task_id
                WHERE runtime.generation_id = ?
                ORDER BY task.position, runtime.id
                """,
                (str(generation_id),),
            ).fetchall()
        return [_row_to_task_runtime(row) for row in rows]

    def list_task_runtime(self, borg_id: UUID) -> list[TaskRuntimeRow]:
        """Return the shared runtime projection for a Borg's current tasks.

        Agent-reported subscription estimates are deliberately excluded from
        API spend. If any API attempt has no reported price, the API component
        is unknown instead of becoming a partial or zero-valued total.
        """
        with self.locked_connection() as connection:
            task_rows = connection.execute(
                """
                SELECT task.*, runtime.status AS runtime_status,
                       runtime.state_reason AS runtime_state_reason,
                       runtime.review_round AS runtime_review_round
                FROM task_generations AS generation
                JOIN task_records AS task
                  ON task.generation_id = generation.id
                LEFT JOIN task_runtimes AS runtime
                  ON runtime.task_id = task.id
                 AND runtime.generation_id = generation.id
                WHERE generation.borg_id = ? AND generation.status = 'current'
                ORDER BY task.position, task.id
                """,
                (str(borg_id),),
            ).fetchall()
            if not task_rows:
                return []

            task_ids = [row["id"] for row in task_rows]
            attempt_rows = connection.execute(
                """
                SELECT attempt.*,
                       terminal.kind AS terminal_kind,
                       terminal.payload_json AS terminal_payload_json,
                       terminal.created_at AS terminal_at
                FROM agent_attempts AS attempt
                JOIN task_records AS task ON task.id = attempt.task_id
                JOIN task_generations AS generation
                  ON generation.id = task.generation_id
                LEFT JOIN execution_events AS terminal
                  ON terminal.attempt_id = attempt.id
                 AND terminal.kind IN (
                    'agent.attempt_finished', 'agent.attempt_interrupted'
                 )
                WHERE generation.borg_id = ? AND generation.status = 'current'
                ORDER BY attempt.started_at, attempt.id
                """,
                (str(borg_id),),
            ).fetchall()

        attempts_by_task: dict[str, list[AgentAttempt]] = {
            task_id: [] for task_id in task_ids
        }
        for attempt_row in attempt_rows:
            attempts_by_task[attempt_row["task_id"]].append(
                _row_to_agent_attempt(attempt_row)
            )

        rows = []
        for task_row in task_rows:
            attempts = attempts_by_task[task_row["id"]]
            durations = [
                attempt.duration_seconds
                for attempt in attempts
                if attempt.duration_seconds is not None
            ]
            api_attempts = [
                attempt
                for attempt in attempts
                if attempt.billing_mode is BillingMode.API
            ]
            api_costs = [
                attempt.usage.cost_usd
                for attempt in api_attempts
                if attempt.usage is not None
                and attempt.usage.cost_usd is not None
            ]
            api_spend_unknown = (not attempts) or any(
                attempt.usage is None or attempt.usage.cost_usd is None
                for attempt in api_attempts
            )
            api_spend_usd = (
                None
                if api_spend_unknown or not api_costs
                else float(sum(api_costs))
            )
            rows.append(
                TaskRuntimeRow(
                    generation_id=UUID(task_row["generation_id"]),
                    task_id=UUID(task_row["id"]),
                    task_ref=task_row["task_ref"],
                    stage=task_row["stage"],
                    stem=task_row["stem"],
                    position=task_row["position"],
                    title=task_row["title"],
                    complexity=TaskComplexity(task_row["complexity"]),
                    status=(
                        TaskRuntimeStatus(task_row["runtime_status"])
                        if task_row["runtime_status"] is not None
                        else TaskRuntimeStatus.PENDING
                    ),
                    state_reason=task_row["runtime_state_reason"],
                    review_round=task_row["runtime_review_round"] or 0,
                    attempt_count=len(attempts),
                    duration_seconds=(
                        float(sum(durations)) if durations else None
                    ),
                    cost=TaskRuntimeCost(
                        api_spend_usd=api_spend_usd,
                        api_spend_unknown=api_spend_unknown,
                        subscription_included=any(
                            attempt.billing_mode is BillingMode.SUBSCRIPTION
                            for attempt in attempts
                        ),
                    ),
                )
            )
        return rows

    def append_task_claim(self, claim: TaskClaim) -> None:
        """Persist a run's lease-backed claim on a generated task."""
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO task_claims(
                    id, run_id, generation_id, task_id, claim_token,
                    resume_phase, claimed_at, lease_expires_at, released_at
                )
                SELECT ?, run.id, run.generation_id, task.id, ?, ?, ?, ?, ?
                FROM execution_runs AS run
                JOIN task_records AS task
                  ON task.generation_id = run.generation_id AND task.id = ?
                WHERE run.id = ?
                """,
                (
                    str(claim.id),
                    claim.claim_token,
                    claim.resume_phase,
                    claim.claimed_at.isoformat(),
                    claim.lease_expires_at.isoformat(),
                    claim.released_at.isoformat() if claim.released_at else None,
                    str(claim.task_id),
                    str(claim.run_id),
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("task claim run and task must share a generation")

    def list_task_claims(self, run_id: UUID) -> list[TaskClaim]:
        """Return a run's durable task claims in claim order."""
        with self.locked_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM task_claims
                WHERE run_id = ?
                ORDER BY claimed_at, id
                """,
                (str(run_id),),
            ).fetchall()
        return [_row_to_task_claim(row) for row in rows]

    def task_claim_owned_by(self, claim_id: UUID, claim_token: str) -> bool:
        """Return whether ``claim_token`` owns ``claim_id`` without exposing it."""
        with self.locked_connection() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM task_claims
                WHERE id = ? AND claim_token = ?
                """,
                (str(claim_id), claim_token),
            ).fetchone()
        return row is not None

    def append_environment_attempt(
        self,
        attempt: EnvironmentAttempt,
        owner_token: str,
        claim_token: str | None,
        *,
        now: datetime | None = None,
    ) -> None:
        """Append an environment attempt while its execution lease is owned."""
        persisted_at = _execution_time(now)
        with self.transaction() as connection:
            self._require_attempt_ownership(
                connection,
                attempt,
                owner_token=owner_token,
                claim_token=claim_token,
                now=persisted_at,
            )
            connection.execute(
                """
                INSERT INTO environment_attempts(
                    id, run_id, claim_id, task_id, kind, attempt_number,
                    fingerprint, status, commands_json, result_json, error,
                    duration_seconds, started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(attempt.id),
                    str(attempt.run_id),
                    str(attempt.claim_id) if attempt.claim_id is not None else None,
                    str(attempt.task_id),
                    attempt.kind,
                    attempt.attempt_number,
                    attempt.fingerprint,
                    attempt.status.value,
                    _encode_json(attempt.commands),
                    (
                        _encode_json(attempt.result)
                        if attempt.result is not None
                        else None
                    ),
                    attempt.error,
                    attempt.duration_seconds,
                    attempt.started_at.isoformat(),
                    (
                        attempt.finished_at.isoformat()
                        if attempt.finished_at is not None
                        else None
                    ),
                ),
            )

    def list_environment_attempts(self, task_id: UUID) -> list[EnvironmentAttempt]:
        """Return immutable environment attempts for one task."""
        with self.locked_connection() as connection:
            rows = connection.execute(
                """
                SELECT environment_attempts.*,
                       terminal.kind AS terminal_kind,
                       terminal.payload_json AS terminal_payload_json,
                       terminal.created_at AS terminal_at
                FROM environment_attempts
                LEFT JOIN execution_events AS terminal
                  ON terminal.attempt_id = environment_attempts.id
                 AND terminal.kind IN (
                    'environment.attempt_finished',
                    'environment.attempt_interrupted'
                 )
                WHERE environment_attempts.task_id = ?
                ORDER BY environment_attempts.started_at, environment_attempts.id
                """,
                (str(task_id),),
            ).fetchall()
        return [_row_to_environment_attempt(row) for row in rows]

    def find_completed_environment_attempt(
        self,
        fingerprint: str,
        *,
        kind: str,
        task_id: UUID | None = None,
    ) -> EnvironmentAttempt | None:
        """Return the newest successful attempt matching an exact descriptor.

        Preparation caches are reusable across tasks and execution runs, while
        checkout materialization is task-local.  The optional task filter lets
        callers enforce that distinction without treating failed, cancelled,
        or interrupted attempts as cache hits.
        """
        if not fingerprint.strip() or not kind.strip():
            raise ValueError("environment fingerprint and kind must not be empty")
        task_filter = (
            "AND environment_attempts.task_id = ?" if task_id is not None else ""
        )
        parameters: list[str] = [fingerprint, kind]
        if task_id is not None:
            parameters.append(str(task_id))
        with self.locked_connection() as connection:
            row = connection.execute(
                f"""
                SELECT environment_attempts.*,
                       terminal.kind AS terminal_kind,
                       terminal.payload_json AS terminal_payload_json,
                       terminal.created_at AS terminal_at
                FROM environment_attempts
                LEFT JOIN execution_events AS terminal
                  ON terminal.attempt_id = environment_attempts.id
                 AND terminal.kind IN (
                    'environment.attempt_finished',
                    'environment.attempt_interrupted'
                 )
                WHERE environment_attempts.fingerprint = ?
                  AND environment_attempts.kind = ?
                  {task_filter}
                  AND (
                    (
                      terminal.kind = 'environment.attempt_finished'
                      AND json_extract(terminal.payload_json, '$.status') = 'completed'
                    )
                    OR (
                      terminal.kind IS NULL
                      AND environment_attempts.status = 'completed'
                    )
                  )
                ORDER BY COALESCE(terminal.created_at,
                                  environment_attempts.finished_at) DESC,
                         environment_attempts.id DESC
                LIMIT 1
                """,
                parameters,
            ).fetchone()
        return _row_to_environment_attempt(row) if row is not None else None

    def complete_environment_attempt(
        self,
        attempt_id: UUID,
        owner_token: str,
        claim_token: str | None,
        *,
        status: ExecutionAttemptStatus | AgentStatus,
        result: dict[str, object] | None = None,
        error: str | None = None,
        duration_seconds: float | None = None,
        now: datetime | None = None,
    ) -> EnvironmentAttempt:
        """Durably finish one owned environment attempt exactly once."""
        finished_at = _execution_time(now)
        terminal_status = _terminal_attempt_status(status)
        if duration_seconds is not None and duration_seconds < 0:
            raise ValueError("environment attempt duration must not be negative")
        with self.transaction() as connection:
            attempt = self._require_owned_open_attempt(
                connection,
                table="environment_attempts",
                attempt_id=attempt_id,
                owner_token=owner_token,
                claim_token=claim_token,
                now=finished_at,
            )
            duration = _completed_attempt_duration(
                attempt, finished_at=finished_at, duration_seconds=duration_seconds
            )
            self._insert_execution_event(
                connection,
                run_id=UUID(attempt["run_id"]),
                task_id=UUID(attempt["task_id"]),
                attempt_id=attempt_id,
                kind="environment.attempt_finished",
                payload={
                    "status": terminal_status.value,
                    "result": result,
                    "error": error,
                    "duration_seconds": duration,
                },
                created_at=finished_at,
            )
            row = self._environment_attempt_projection(connection, attempt_id)
        return _row_to_environment_attempt(row)

    def append_agent_attempt(
        self,
        attempt: AgentAttempt,
        owner_token: str,
        claim_token: str,
        *,
        now: datetime | None = None,
    ) -> None:
        """Append a billing-aware agent attempt while its lease is owned."""
        persisted_at = _execution_time(now)
        with self.transaction() as connection:
            self._require_attempt_ownership(
                connection,
                attempt,
                owner_token=owner_token,
                claim_token=claim_token,
                now=persisted_at,
            )
            connection.execute(
                """
                INSERT INTO agent_attempts(
                    id, run_id, claim_id, task_id, phase, review_round,
                    attempt_number, adapter, model, billing_mode, status,
                    log_path, result_path, result_json, summary,
                    duration_seconds, cost_usd, tokens_input, tokens_output,
                    tokens_cache_read, tokens_cache_write, num_turns,
                    started_at, finished_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    str(attempt.id),
                    str(attempt.run_id),
                    str(attempt.claim_id),
                    str(attempt.task_id),
                    attempt.phase,
                    attempt.review_round,
                    attempt.attempt_number,
                    attempt.adapter,
                    attempt.model,
                    attempt.billing_mode.value,
                    attempt.status.value,
                    attempt.log_path,
                    attempt.result_path,
                    (
                        _encode_json(attempt.result)
                        if attempt.result is not None
                        else None
                    ),
                    attempt.summary,
                    attempt.duration_seconds,
                    attempt.usage.cost_usd if attempt.usage else None,
                    attempt.usage.tokens_input if attempt.usage else None,
                    attempt.usage.tokens_output if attempt.usage else None,
                    attempt.usage.tokens_cache_read if attempt.usage else None,
                    attempt.usage.tokens_cache_write if attempt.usage else None,
                    attempt.usage.num_turns if attempt.usage else None,
                    attempt.started_at.isoformat(),
                    (
                        attempt.finished_at.isoformat()
                        if attempt.finished_at is not None
                        else None
                    ),
                ),
            )

    def list_agent_attempts(self, task_id: UUID) -> list[AgentAttempt]:
        """Return immutable agent attempts for one task."""
        with self.locked_connection() as connection:
            rows = connection.execute(
                """
                SELECT agent_attempts.*,
                       terminal.kind AS terminal_kind,
                       terminal.payload_json AS terminal_payload_json,
                       terminal.created_at AS terminal_at
                FROM agent_attempts
                LEFT JOIN execution_events AS terminal
                  ON terminal.attempt_id = agent_attempts.id
                 AND terminal.kind IN (
                    'agent.attempt_finished',
                    'agent.attempt_interrupted'
                 )
                WHERE agent_attempts.task_id = ?
                ORDER BY agent_attempts.started_at, agent_attempts.id
                """,
                (str(task_id),),
            ).fetchall()
        return [_row_to_agent_attempt(row) for row in rows]

    def complete_agent_attempt(
        self,
        attempt_id: UUID,
        owner_token: str,
        claim_token: str,
        *,
        status: ExecutionAttemptStatus | AgentStatus,
        result_path: str | None = None,
        result: dict[str, object] | None = None,
        summary: str | None = None,
        duration_seconds: float | None = None,
        usage: AgentUsage | None = None,
        now: datetime | None = None,
    ) -> AgentAttempt:
        """Durably finish one owned billing-aware agent attempt exactly once."""
        finished_at = _execution_time(now)
        terminal_status = _terminal_attempt_status(status)
        if duration_seconds is not None and duration_seconds < 0:
            raise ValueError("agent attempt duration must not be negative")
        if usage is not None and not isinstance(usage, AgentUsage):
            raise TypeError("agent attempt usage must be an AgentUsage")
        with self.transaction() as connection:
            attempt = self._require_owned_open_attempt(
                connection,
                table="agent_attempts",
                attempt_id=attempt_id,
                owner_token=owner_token,
                claim_token=claim_token,
                now=finished_at,
            )
            duration = _completed_attempt_duration(
                attempt, finished_at=finished_at, duration_seconds=duration_seconds
            )
            self._insert_execution_event(
                connection,
                run_id=UUID(attempt["run_id"]),
                task_id=UUID(attempt["task_id"]),
                attempt_id=attempt_id,
                kind="agent.attempt_finished",
                payload={
                    "status": terminal_status.value,
                    "result_path": result_path,
                    "result": result,
                    "summary": summary,
                    "duration_seconds": duration,
                    "usage": _agent_usage_payload(usage),
                },
                created_at=finished_at,
            )
            row = self._agent_attempt_projection(connection, attempt_id)
        return _row_to_agent_attempt(row)

    def _require_owned_open_attempt(
        self,
        connection: sqlite3.Connection,
        *,
        table: str,
        attempt_id: UUID,
        owner_token: str,
        claim_token: str | None,
        now: datetime,
    ) -> sqlite3.Row:
        if table not in {"environment_attempts", "agent_attempts"}:
            raise ValueError("unsupported execution attempt table")
        attempt = connection.execute(
            f"SELECT * FROM {table} WHERE id = ?", (str(attempt_id),)
        ).fetchone()
        if attempt is None:
            raise KeyError(f"execution attempt {attempt_id} not found")
        if attempt["status"] != ExecutionAttemptStatus.RUNNING.value:
            raise ValueError("execution attempt has already finished")
        terminal = connection.execute(
            """
            SELECT 1 FROM execution_events
            WHERE attempt_id = ? AND kind IN (
                'environment.attempt_finished',
                'environment.attempt_interrupted',
                'agent.attempt_finished',
                'agent.attempt_interrupted'
            )
            """,
            (str(attempt_id),),
        ).fetchone()
        if terminal is not None:
            raise ValueError("execution attempt has already finished")
        if attempt["claim_id"] is None:
            if claim_token is not None:
                raise ExecutionOwnershipError(
                    "run-owned environment attempt cannot use a claim token"
                )
            self._require_live_run(
                connection,
                UUID(attempt["run_id"]),
                owner_token,
                now=now,
            )
        else:
            if claim_token is None:
                raise ExecutionOwnershipError(
                    "claim-owned execution attempt requires a claim token"
                )
            self._require_live_claim(
                connection,
                run_id=UUID(attempt["run_id"]),
                owner_token=owner_token,
                claim_id=UUID(attempt["claim_id"]),
                claim_token=claim_token,
                task_id=UUID(attempt["task_id"]),
                now=now,
            )
        return attempt

    @staticmethod
    def _environment_attempt_projection(
        connection: sqlite3.Connection, attempt_id: UUID
    ) -> sqlite3.Row:
        return connection.execute(
            """
            SELECT environment_attempts.*,
                   terminal.kind AS terminal_kind,
                   terminal.payload_json AS terminal_payload_json,
                   terminal.created_at AS terminal_at
            FROM environment_attempts
            LEFT JOIN execution_events AS terminal
              ON terminal.attempt_id = environment_attempts.id
             AND terminal.kind IN (
                'environment.attempt_finished',
                'environment.attempt_interrupted'
             )
            WHERE environment_attempts.id = ?
            """,
            (str(attempt_id),),
        ).fetchone()

    @staticmethod
    def _agent_attempt_projection(
        connection: sqlite3.Connection, attempt_id: UUID
    ) -> sqlite3.Row:
        return connection.execute(
            """
            SELECT agent_attempts.*,
                   terminal.kind AS terminal_kind,
                   terminal.payload_json AS terminal_payload_json,
                   terminal.created_at AS terminal_at
            FROM agent_attempts
            LEFT JOIN execution_events AS terminal
              ON terminal.attempt_id = agent_attempts.id
             AND terminal.kind IN (
                'agent.attempt_finished',
                'agent.attempt_interrupted'
             )
            WHERE agent_attempts.id = ?
            """,
            (str(attempt_id),),
        ).fetchone()

    def _require_attempt_ownership(
        self,
        connection: sqlite3.Connection,
        attempt: EnvironmentAttempt | AgentAttempt,
        *,
        owner_token: str,
        claim_token: str | None,
        now: datetime,
    ) -> None:
        if isinstance(attempt, EnvironmentAttempt) and attempt.claim_id is None:
            if claim_token is not None:
                raise ExecutionOwnershipError(
                    "run-owned environment attempt cannot use a claim token"
                )
            run = self._require_live_run(
                connection, attempt.run_id, owner_token, now=now
            )
            task = connection.execute(
                """
                SELECT 1 FROM task_records
                WHERE id = ? AND generation_id = ?
                """,
                (str(attempt.task_id), run["generation_id"]),
            ).fetchone()
            if task is None:
                raise ExecutionOwnershipError(
                    "environment attempt task does not belong to the execution run"
                )
            return
        if claim_token is None:
            raise ExecutionOwnershipError(
                "claim-owned execution attempt requires a claim token"
            )
        assert attempt.claim_id is not None
        self._require_live_claim(
            connection,
            run_id=attempt.run_id,
            owner_token=owner_token,
            claim_id=attempt.claim_id,
            claim_token=claim_token,
            task_id=attempt.task_id,
            now=now,
        )

    def append_execution_event(self, event: ExecutionEvent) -> None:
        """Append one durable execution event."""
        if event.kind in _TERMINAL_ATTEMPT_EVENT_KINDS:
            raise ValueError(
                "terminal attempt lifecycle events require guarded completion"
            )
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO execution_events(
                    id, run_id, task_id, attempt_id, kind, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event.id),
                    str(event.run_id),
                    str(event.task_id) if event.task_id else None,
                    str(event.attempt_id) if event.attempt_id else None,
                    event.kind,
                    _encode_json(event.payload),
                    event.created_at.isoformat(),
                ),
            )

    def append_claim_execution_event(
        self,
        event: ExecutionEvent,
        owner_token: str,
        claim_token: str,
        *,
        now: datetime | None = None,
    ) -> None:
        """Append an event only while its run and task claim are owned."""
        recorded_at = _execution_time(now)
        if event.task_id is None:
            raise ValueError("claim-owned execution events require a task ID")
        claim_id = event.payload.get("claim_id")
        if not isinstance(claim_id, str):
            raise ValueError("claim-owned execution events require a claim ID")
        with self.transaction() as connection:
            self._require_live_claim(
                connection,
                run_id=event.run_id,
                owner_token=owner_token,
                claim_id=UUID(claim_id),
                claim_token=claim_token,
                task_id=event.task_id,
                now=recorded_at,
            )
            connection.execute(
                """
                INSERT INTO execution_events(
                    id, run_id, task_id, attempt_id, kind, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event.id),
                    str(event.run_id),
                    str(event.task_id),
                    str(event.attempt_id) if event.attempt_id else None,
                    event.kind,
                    _encode_json(event.payload),
                    event.created_at.isoformat(),
                ),
            )

    def list_execution_events(self, run_id: UUID) -> list[ExecutionEvent]:
        """Return one run's events in stable append order."""
        with self.locked_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM execution_events
                WHERE run_id = ?
                ORDER BY created_at, id
                """,
                (str(run_id),),
            ).fetchall()
        return [_row_to_execution_event(row) for row in rows]

    def list_task_execution_events(
        self, task_id: UUID, *, kind: str | None = None
    ) -> list[ExecutionEvent]:
        """Return durable events for one task across execution runs."""
        if kind is not None and not kind.strip():
            raise ValueError("execution event kind must not be empty")
        with self.locked_connection() as connection:
            if kind is None:
                rows = connection.execute(
                    """
                    SELECT * FROM execution_events
                    WHERE task_id = ?
                    ORDER BY created_at, id
                    """,
                    (str(task_id),),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM execution_events
                    WHERE task_id = ? AND kind = ?
                    ORDER BY created_at, id
                    """,
                    (str(task_id), kind),
                ).fetchall()
        return [_row_to_execution_event(row) for row in rows]

    def add_compose_resource(
        self,
        resource: ComposeResource,
        owner_token: str,
        claim_token: str,
        *,
        now: datetime | None = None,
    ) -> None:
        """Persist Compose identity while its run and task claim are owned."""
        persisted_at = _execution_time(now)
        with self.transaction() as connection:
            self._require_live_claim(
                connection,
                run_id=resource.run_id,
                owner_token=owner_token,
                claim_id=resource.claim_id,
                claim_token=claim_token,
                task_id=resource.task_id,
                now=persisted_at,
            )
            connection.execute(
                """
                INSERT INTO compose_resources(
                    id, run_id, claim_id, task_id, project_name,
                    resource_type, resource_name, labels_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(resource.id),
                    str(resource.run_id),
                    str(resource.claim_id),
                    str(resource.task_id),
                    resource.project_name,
                    resource.resource_type,
                    resource.resource_name,
                    _encode_json(resource.labels),
                    resource.created_at.isoformat(),
                ),
            )

    def list_compose_resources(self, task_id: UUID) -> list[ComposeResource]:
        """Return durable Compose resources owned by one task."""
        with self.locked_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM compose_resources
                WHERE task_id = ?
                ORDER BY created_at, id
                """,
                (str(task_id),),
            ).fetchall()
        return [_row_to_compose_resource(row) for row in rows]

    def add_prd_session(self, session: PrdSession) -> None:
        """Persist a PRD session that points to tracked Markdown."""
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO prd_sessions(
                    id, repository_id, borg_id, prd_path, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(session.id),
                    str(session.repository_id),
                    str(session.borg_id),
                    session.prd_path.as_posix(),
                    session.created_at.isoformat(),
                ),
            )

    def get_prd_session(self, session_id: UUID) -> PrdSession | None:
        """Return one PRD session by ID, if it exists."""
        with self.locked_connection() as connection:
            row = connection.execute(
                "SELECT * FROM prd_sessions WHERE id = ?", (str(session_id),)
            ).fetchone()
        return _row_to_prd_session(row) if row is not None else None

    def get_prd_session_for_borg(self, borg_id: UUID) -> PrdSession | None:
        """Return the latest PRD session belonging to one Borg, if present."""
        with self.locked_connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM prd_sessions
                WHERE borg_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (str(borg_id),),
            ).fetchone()
        return _row_to_prd_session(row) if row is not None else None

    def append_prd_turn(
        self, *, session_id: UUID, role: str, content: str
    ) -> PrdTurn:
        """Append and return the next ordered turn for a PRD session."""
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(MAX(position), 0) AS position
                FROM prd_turns
                WHERE session_id = ?
                """,
                (str(session_id),),
            ).fetchone()
            turn = PrdTurn(
                session_id=session_id,
                position=row["position"] + 1,
                role=role,
                content=content,
            )
            connection.execute(
                """
                INSERT INTO prd_turns(
                    id, session_id, position, role, content, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(turn.id),
                    str(turn.session_id),
                    turn.position,
                    turn.role,
                    turn.content,
                    turn.created_at.isoformat(),
                ),
            )
        return turn

    def list_prd_turns(self, session_id: UUID) -> list[PrdTurn]:
        """Return a PRD session's immutable turns in conversation order."""
        with self.locked_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM prd_turns
                WHERE session_id = ?
                ORDER BY position
                """,
                (str(session_id),),
            ).fetchall()
        return [_row_to_prd_turn(row) for row in rows]

    def applied_migrations(self) -> tuple[int, ...]:
        """Return applied migration versions in ascending order."""
        with self.locked_connection() as connection:
            rows = connection.execute(
                "SELECT version FROM schema_version ORDER BY version"
            ).fetchall()
        return tuple(row["version"] for row in rows)

    def _ensure_schema(self) -> None:
        with self._lock:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            migrations = self._load_migrations()
            known_versions = tuple(version for version, _ in migrations)
            applied_versions = self.applied_migrations()
            if applied_versions != known_versions[: len(applied_versions)]:
                raise RuntimeError(
                    "database migration history is newer than or incompatible "
                    "with this BetterBorg CLI"
                )
            for version, sql in migrations[len(applied_versions) :]:
                self._apply_migration(version, sql)

    @staticmethod
    def _load_migrations() -> list[tuple[int, str]]:
        migration_root = resources.files("betterborg_cli.store.migrations")
        migrations: list[tuple[int, str]] = []
        for candidate in migration_root.iterdir():
            match = _MIGRATION_NAME.fullmatch(candidate.name)
            if match:
                migrations.append(
                    (int(match.group("version")), candidate.read_text(encoding="utf-8"))
                )
        migrations.sort(key=lambda migration: migration[0])
        versions = [version for version, _ in migrations]
        if versions != list(range(1, len(versions) + 1)):
            raise RuntimeError(
                "store migrations must be a contiguous sequence from 001"
            )
        return migrations

    def _apply_migration(self, version: int, sql: str) -> None:
        applied_at = utcnow().isoformat()
        quoted_applied_at = self._connection.execute(
            "SELECT quote(?)", (applied_at,)
        ).fetchone()[0]
        script = (
            "BEGIN IMMEDIATE;\n"
            f"{sql.rstrip()}\n"
            "INSERT INTO schema_version(version, applied_at) "
            f"VALUES ({version}, {quoted_applied_at});\n"
            "COMMIT;\n"
        )
        try:
            self._connection.executescript(script)
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise


def _encode_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _execution_time(value: datetime | None) -> datetime:
    timestamp = value or utcnow()
    if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(timestamp):
        raise ValueError("execution timestamps must be timezone-aware UTC values")
    return timestamp


def _lease_expiry(started_at: datetime, duration: timedelta) -> datetime:
    if not isinstance(duration, timedelta):
        raise TypeError("execution lease duration must be a timedelta")
    if duration <= timedelta(0):
        raise ValueError("execution lease duration must be positive")
    return started_at + duration


def _row_to_analysis(row: sqlite3.Row) -> RepositoryAnalysis:
    return RepositoryAnalysis(
        id=UUID(row["id"]),
        repository_id=UUID(row["repository_id"]),
        head_sha=row["head_sha"],
        summary=row["summary"],
        primary_language=row["primary_language"],
        is_monorepo=bool(row["is_monorepo"]),
        overall_score=row["overall_score"],
        analysis_json=json.loads(row["analysis_json"]),
        prior_analysis_id=(
            UUID(row["prior_analysis_id"])
            if row["prior_analysis_id"] is not None
            else None
        ),
        score_delta=row["score_delta"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_package(row: sqlite3.Row) -> RepositoryPackage:
    return RepositoryPackage(
        id=UUID(row["id"]),
        repository_id=UUID(row["repository_id"]),
        analysis_id=UUID(row["analysis_id"]),
        package_path=row["package_path"],
        package_name=row["package_name"],
        primary_language=row["primary_language"],
        rubric=json.loads(row["rubric_json"]),
        overall_score=row["overall_score"],
    )


def _row_to_generated_prompt(row: sqlite3.Row) -> GeneratedPrompt:
    return GeneratedPrompt(
        id=UUID(row["id"]),
        repository_id=UUID(row["repository_id"]),
        analysis_id=UUID(row["analysis_id"]),
        role=row["role"],
        version=row["version"],
        body_md=row["body_md"],
        generated_at=datetime.fromisoformat(row["generated_at"]),
    )


def _row_to_borg(row: sqlite3.Row) -> Borg:
    return Borg(
        id=UUID(row["id"]),
        repository_id=UUID(row["repository_id"]),
        name=row["name"],
        state=BorgState(row["state"]),
        state_version=row["state_version"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_prd_session(row: sqlite3.Row) -> PrdSession:
    return PrdSession(
        id=UUID(row["id"]),
        repository_id=UUID(row["repository_id"]),
        borg_id=UUID(row["borg_id"]),
        prd_path=Path(row["prd_path"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_prd_turn(row: sqlite3.Row) -> PrdTurn:
    return PrdTurn(
        id=UUID(row["id"]),
        session_id=UUID(row["session_id"]),
        position=row["position"],
        role=row["role"],
        content=row["content"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_planning_attempt(row: sqlite3.Row) -> PlanningAttempt:
    return PlanningAttempt(
        id=UUID(row["id"]),
        borg_id=UUID(row["borg_id"]),
        phase=row["phase"],
        round=row["round"],
        adapter=row["adapter"],
        model=row["model"],
        request=json.loads(row["request_json"]),
        status=PlanningAttemptStatus(row["status"]),
        result=(
            json.loads(row["result_json"])
            if row["result_json"] is not None
            else None
        ),
        summary=row["summary"],
        started_at=datetime.fromisoformat(row["started_at"]),
        finished_at=(
            datetime.fromisoformat(row["finished_at"])
            if row["finished_at"] is not None
            else None
        ),
    )


def _row_to_planning_question(row: sqlite3.Row) -> PlanningQuestion:
    return PlanningQuestion(
        id=UUID(row["id"]),
        borg_id=UUID(row["borg_id"]),
        attempt_id=(
            UUID(row["attempt_id"]) if row["attempt_id"] is not None else None
        ),
        round=row["round"],
        questions=json.loads(row["questions_json"]),
        answers=(
            json.loads(row["answers_json"])
            if row["answers_json"] is not None
            else None
        ),
        asked_at=datetime.fromisoformat(row["asked_at"]),
        answered_at=(
            datetime.fromisoformat(row["answered_at"])
            if row["answered_at"] is not None
            else None
        ),
    )


def _row_to_planning_finding(row: sqlite3.Row) -> PlanningFinding:
    return PlanningFinding(
        id=UUID(row["id"]),
        borg_id=UUID(row["borg_id"]),
        attempt_id=UUID(row["attempt_id"]),
        round=row["round"],
        severity=row["severity"],
        message=row["message"],
        suggestion=row["suggestion"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_plan_change_request(row: sqlite3.Row) -> PlanChangeRequest:
    return PlanChangeRequest(
        id=UUID(row["id"]),
        borg_id=UUID(row["borg_id"]),
        round=row["round"],
        note=row["note"],
        decided_by=row["decided_by"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_plan_approval(row: sqlite3.Row) -> PlanApproval:
    return PlanApproval(
        id=UUID(row["id"]),
        borg_id=UUID(row["borg_id"]),
        attempt_id=(
            UUID(row["attempt_id"]) if row["attempt_id"] is not None else None
        ),
        plan_digest=row["plan_digest"],
        manifest=json.loads(row["manifest_json"]),
        approved_by=row["approved_by"],
        approved_at=datetime.fromisoformat(row["approved_at"]),
    )


def _row_to_task_batch(row: sqlite3.Row) -> TaskBatch:
    return TaskBatch(
        id=UUID(row["id"]),
        borg_id=UUID(row["borg_id"]),
        plan_approval_id=UUID(row["plan_approval_id"]),
        attempt_id=(
            UUID(row["attempt_id"]) if row["attempt_id"] is not None else None
        ),
        round=row["round"],
        summary=row["summary"],
        manifest=json.loads(row["manifest_json"]),
        digest=row["digest"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_task_finding(row: sqlite3.Row) -> TaskFinding:
    return TaskFinding(
        id=UUID(row["id"]),
        borg_id=UUID(row["borg_id"]),
        batch_id=UUID(row["batch_id"]),
        attempt_id=(
            UUID(row["attempt_id"]) if row["attempt_id"] is not None else None
        ),
        round=row["round"],
        severity=row["severity"],
        message=row["message"],
        suggestion=row["suggestion"],
        task_ref=row["task_ref"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _verify_and_fsync_task_tree(
    repository_root: Path,
    durable_root: Path,
    expected_files: dict[Path, str],
) -> None:
    candidate = repository_root
    for component in durable_root.relative_to(repository_root).parts:
        candidate /= component
        if candidate.is_symlink():
            raise ValueError(f"durable task publication path is a symlink: {candidate}")
    if not durable_root.is_dir():
        raise ValueError("durable task generation tree is missing")

    actual_files: set[Path] = set()
    actual_directories: set[Path] = set()
    for path in durable_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"durable task generation contains a symlink: {path}")
        relative = path.relative_to(durable_root)
        if path.is_file():
            actual_files.add(relative)
        elif path.is_dir():
            actual_directories.add(relative)
        else:
            raise ValueError(f"durable task generation contains a non-file: {path}")
    expected_directories = {
        parent
        for relative in expected_files
        for parent in relative.parents
        if parent != Path(".")
    }
    if (
        actual_files != set(expected_files)
        or actual_directories != expected_directories
    ):
        raise ValueError("durable task generation layout does not match SQLite")

    for relative, expected_digest in expected_files.items():
        path = durable_root / relative
        with path.open("rb") as task_file:
            body = task_file.read()
            actual_digest = f"sha256:{hashlib.sha256(body).hexdigest()}"
            if actual_digest != expected_digest:
                raise ValueError(
                    f"durable task file {relative.as_posix()} digest does not "
                    "match SQLite"
                )
            os.fsync(task_file.fileno())
    for directory in sorted(
        (durable_root / relative for relative in expected_directories),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        _fsync_store_directory(directory)
    _fsync_store_directory(durable_root)
    _fsync_store_directory(durable_root.parent)


def _fsync_store_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _row_to_task_generation(row: sqlite3.Row) -> TaskGeneration:
    return TaskGeneration(
        id=UUID(row["id"]),
        borg_id=UUID(row["borg_id"]),
        plan_approval_id=UUID(row["plan_approval_id"]),
        batch_id=UUID(row["batch_id"]),
        status=TaskGenerationStatus(row["status"]),
        manifest=json.loads(row["manifest_json"]),
        digest=row["digest"],
        created_at=datetime.fromisoformat(row["created_at"]),
        current_at=(
            datetime.fromisoformat(row["current_at"])
            if row["current_at"] is not None
            else None
        ),
        superseded_at=(
            datetime.fromisoformat(row["superseded_at"])
            if row["superseded_at"] is not None
            else None
        ),
    )


def _row_to_task_record(row: sqlite3.Row) -> TaskRecord:
    return TaskRecord(
        id=UUID(row["id"]),
        generation_id=UUID(row["generation_id"]),
        borg_id=UUID(row["borg_id"]),
        task_ref=row["task_ref"],
        stage=row["stage"],
        stem=row["stem"],
        position=row["position"],
        title=row["title"],
        complexity=TaskComplexity(row["complexity"]),
        task=json.loads(row["task_json"]),
        manifest=json.loads(row["manifest_json"]),
        digest=row["digest"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_task_dependency(row: sqlite3.Row) -> TaskDependency:
    return TaskDependency(
        id=UUID(row["id"]),
        generation_id=UUID(row["generation_id"]),
        task_id=UUID(row["task_id"]),
        depends_on_task_id=UUID(row["depends_on_task_id"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_execution_run(row: sqlite3.Row) -> ExecutionRun:
    return ExecutionRun(
        id=UUID(row["id"]),
        borg_id=UUID(row["borg_id"]),
        generation_id=UUID(row["generation_id"]),
        owner_token=row["owner_token"],
        status=ExecutionRunStatus(row["status"]),
        started_at=datetime.fromisoformat(row["started_at"]),
        heartbeat_at=(
            datetime.fromisoformat(row["heartbeat_at"])
            if row["heartbeat_at"] is not None
            else None
        ),
        lease_expires_at=datetime.fromisoformat(row["lease_expires_at"]),
        finished_at=(
            datetime.fromisoformat(row["finished_at"])
            if row["finished_at"] is not None
            else None
        ),
    )


def _row_to_task_runtime(row: sqlite3.Row) -> TaskRuntime:
    return TaskRuntime(
        id=UUID(row["id"]),
        generation_id=UUID(row["generation_id"]),
        task_id=UUID(row["task_id"]),
        status=TaskRuntimeStatus(row["status"]),
        resume_phase=row["resume_phase"],
        review_round=row["review_round"],
        state_reason=row["state_reason"],
        branch=row["branch"],
        worktree_path=row["worktree_path"],
        last_run_id=(
            UUID(row["last_run_id"]) if row["last_run_id"] is not None else None
        ),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_task_claim(row: sqlite3.Row) -> TaskClaim:
    return TaskClaim(
        id=UUID(row["id"]),
        run_id=UUID(row["run_id"]),
        task_id=UUID(row["task_id"]),
        claim_token=row["claim_token"],
        resume_phase=row["resume_phase"],
        claimed_at=datetime.fromisoformat(row["claimed_at"]),
        lease_expires_at=datetime.fromisoformat(row["lease_expires_at"]),
        released_at=(
            datetime.fromisoformat(row["released_at"])
            if row["released_at"] is not None
            else None
        ),
    )


def _row_to_environment_attempt(row: sqlite3.Row) -> EnvironmentAttempt:
    status, finished_at, duration_seconds, terminal = _project_attempt_lifecycle(row)
    return EnvironmentAttempt(
        id=UUID(row["id"]),
        run_id=UUID(row["run_id"]),
        claim_id=UUID(row["claim_id"]) if row["claim_id"] is not None else None,
        task_id=UUID(row["task_id"]),
        kind=row["kind"],
        attempt_number=row["attempt_number"],
        fingerprint=row["fingerprint"],
        status=status,
        commands=json.loads(row["commands_json"]),
        result=(
            terminal["result"]
            if terminal is not None
            else (
                json.loads(row["result_json"])
                if row["result_json"] is not None
                else None
            )
        ),
        error=terminal["error"] if terminal is not None else row["error"],
        duration_seconds=duration_seconds,
        started_at=datetime.fromisoformat(row["started_at"]),
        finished_at=finished_at,
    )


def _row_to_agent_attempt(row: sqlite3.Row) -> AgentAttempt:
    status, finished_at, duration_seconds, terminal = _project_attempt_lifecycle(row)
    return AgentAttempt(
        id=UUID(row["id"]),
        run_id=UUID(row["run_id"]),
        claim_id=UUID(row["claim_id"]),
        task_id=UUID(row["task_id"]),
        phase=row["phase"],
        review_round=row["review_round"],
        attempt_number=row["attempt_number"],
        adapter=row["adapter"],
        model=row["model"],
        billing_mode=BillingMode(row["billing_mode"]),
        status=status,
        log_path=row["log_path"],
        result_path=(
            terminal["result_path"] if terminal is not None else row["result_path"]
        ),
        result=(
            terminal["result"]
            if terminal is not None
            else (
                json.loads(row["result_json"])
                if row["result_json"] is not None
                else None
            )
        ),
        summary=terminal["summary"] if terminal is not None else row["summary"],
        duration_seconds=duration_seconds,
        usage=(
            _agent_usage_from_payload(terminal["usage"])
            if terminal is not None
            else _row_to_agent_usage(row)
        ),
        started_at=datetime.fromisoformat(row["started_at"]),
        finished_at=finished_at,
    )


def _project_attempt_lifecycle(
    row: sqlite3.Row,
) -> tuple[
    ExecutionAttemptStatus,
    datetime | None,
    float | None,
    dict[str, object] | None,
]:
    status = ExecutionAttemptStatus(row["status"])
    finished_at = (
        datetime.fromisoformat(row["finished_at"])
        if row["finished_at"] is not None
        else None
    )
    duration_seconds = row["duration_seconds"]
    terminal_kind = row["terminal_kind"] if "terminal_kind" in row.keys() else None
    terminal = None
    if status is ExecutionAttemptStatus.RUNNING and terminal_kind is not None:
        finished_at = datetime.fromisoformat(row["terminal_at"])
        if terminal_kind.endswith("attempt_finished"):
            terminal = json.loads(row["terminal_payload_json"])
            status = ExecutionAttemptStatus(terminal["status"])
            duration_seconds = terminal["duration_seconds"]
        else:
            status = ExecutionAttemptStatus.CANCELLED
            duration_seconds = (
                finished_at - datetime.fromisoformat(row["started_at"])
            ).total_seconds()
    return status, finished_at, duration_seconds, terminal


def _terminal_attempt_status(
    status: ExecutionAttemptStatus | AgentStatus,
) -> ExecutionAttemptStatus:
    if not isinstance(status, ExecutionAttemptStatus | AgentStatus):
        raise TypeError("attempt status must be an ExecutionAttemptStatus")
    terminal_status = ExecutionAttemptStatus(status.value)
    if terminal_status is ExecutionAttemptStatus.RUNNING:
        raise ValueError("a completed execution attempt cannot remain running")
    return terminal_status


def _completed_attempt_duration(
    row: sqlite3.Row,
    *,
    finished_at: datetime,
    duration_seconds: float | None,
) -> float:
    started_at = datetime.fromisoformat(row["started_at"])
    if finished_at < started_at:
        raise ValueError("execution attempt cannot finish before it starts")
    if duration_seconds is not None:
        return duration_seconds
    return (finished_at - started_at).total_seconds()


def _agent_usage_payload(usage: AgentUsage | None) -> dict[str, object] | None:
    if usage is None:
        return None
    return {
        "cost_usd": usage.cost_usd,
        "tokens_input": usage.tokens_input,
        "tokens_output": usage.tokens_output,
        "tokens_cache_read": usage.tokens_cache_read,
        "tokens_cache_write": usage.tokens_cache_write,
        "num_turns": usage.num_turns,
    }


def _agent_usage_from_payload(payload: object) -> AgentUsage | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise TypeError("agent attempt usage payload must be a dictionary")
    return AgentUsage(**payload)


def _row_to_agent_usage(row: sqlite3.Row) -> AgentUsage | None:
    values = {
        "cost_usd": row["cost_usd"],
        "tokens_input": row["tokens_input"],
        "tokens_output": row["tokens_output"],
        "tokens_cache_read": row["tokens_cache_read"],
        "tokens_cache_write": row["tokens_cache_write"],
        "num_turns": row["num_turns"],
    }
    if all(value is None for value in values.values()):
        return None
    return AgentUsage(**values)


def _row_to_execution_event(row: sqlite3.Row) -> ExecutionEvent:
    return ExecutionEvent(
        id=UUID(row["id"]),
        run_id=UUID(row["run_id"]),
        task_id=UUID(row["task_id"]) if row["task_id"] is not None else None,
        attempt_id=(
            UUID(row["attempt_id"]) if row["attempt_id"] is not None else None
        ),
        kind=row["kind"],
        payload=json.loads(row["payload_json"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_compose_resource(row: sqlite3.Row) -> ComposeResource:
    return ComposeResource(
        id=UUID(row["id"]),
        run_id=UUID(row["run_id"]),
        claim_id=UUID(row["claim_id"]),
        task_id=UUID(row["task_id"]),
        project_name=row["project_name"],
        resource_type=row["resource_type"],
        resource_name=row["resource_name"],
        labels=json.loads(row["labels_json"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )
