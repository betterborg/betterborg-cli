"""Serialized, transactional SQLite storage with forward-only migrations."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from importlib import resources
from pathlib import Path
from uuid import UUID

from betterborg_cli.store.models import (
    Borg,
    BorgState,
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
    TaskComplexity,
    TaskDependency,
    TaskFinding,
    TaskGeneration,
    TaskGenerationStatus,
    TaskRecord,
    utcnow,
)

_MIGRATION_NAME = re.compile(r"^(?P<version>[0-9]{3})_[a-z0-9_]+\.sql$")


class StaleBorgStateError(RuntimeError):
    """Raised when a Borg state compare-and-set loses a concurrent race."""


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

    def promote_task_generation(self, generation_id: UUID) -> TaskGeneration:
        """Atomically make a preparing generation current and supersede its peer."""
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
