"""Serialized, transactional SQLite storage with forward-only migrations."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from importlib import resources
from pathlib import Path
from uuid import UUID

from betterborg_cli.store.models import (
    GeneratedPrompt,
    Operation,
    Repository,
    RepositoryAnalysis,
    RepositoryPackage,
    utcnow,
)

_MIGRATION_NAME = re.compile(r"^(?P<version>[0-9]{3})_[a-z0-9_]+\.sql$")


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
