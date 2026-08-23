"""Contracts for the forward-only local SQLite store."""

import sqlite3
from pathlib import Path
from uuid import UUID

import pytest

from betterborg_cli.store import Operation, Repository, SqliteStore


def test_store_reopens_without_reapplying_migration_and_preserves_rows(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state" / "borg.sqlite3"
    repository = Repository(root=tmp_path / "repo")
    operation = Operation(
        repository_id=repository.id,
        kind="repository.discovered",
        payload={"source": "test"},
    )

    assert isinstance(repository.id, UUID)
    assert repository.id.version == 4
    assert isinstance(operation.id, UUID)
    assert operation.id.version == 4
    assert isinstance(operation.repository_id, UUID)
    assert operation.repository_id == repository.id

    with SqliteStore.open(database) as store:
        with store.transaction():
            store.add_repository(repository)
            store.append_operation(operation)
        assert store.applied_migrations() == (1,)
        with store.locked_connection() as connection:
            applied_at = connection.execute(
                "SELECT applied_at FROM schema_version WHERE version = 1"
            ).fetchone()[0]
            stored_repository_id = connection.execute(
                "SELECT id FROM repositories"
            ).fetchone()[0]
            stored_operation_ids = connection.execute(
                "SELECT id, repository_id FROM operations"
            ).fetchone()

        assert stored_repository_id == str(repository.id)
        assert tuple(stored_operation_ids) == (
            str(operation.id),
            str(repository.id),
        )

    with SqliteStore.open(database) as reopened:
        assert reopened.applied_migrations() == (1,)
        reopened_repository = reopened.get_repository(repository.id)
        reopened_operations = reopened.list_operations(repository.id)
        assert reopened_repository == repository
        assert isinstance(reopened_repository.id, UUID)
        assert reopened_operations == [operation]
        assert isinstance(reopened_operations[0].id, UUID)
        assert isinstance(reopened_operations[0].repository_id, UUID)
        with reopened.locked_connection() as connection:
            reopened_applied_at = connection.execute(
                "SELECT applied_at FROM schema_version WHERE version = 1"
            ).fetchone()[0]

    assert reopened_applied_at == applied_at


def test_reentrant_transaction_rolls_back_as_one_unit(tmp_path: Path) -> None:
    database = tmp_path / "borg.sqlite3"
    repository = Repository(root=tmp_path / "repo")

    with SqliteStore.open(database) as store:
        with pytest.raises(RuntimeError, match="stop"):
            with store.transaction():
                store.add_repository(repository)
                store.append_operation(
                    Operation(repository_id=repository.id, kind="test.started")
                )
                raise RuntimeError("stop")

        assert store.get_repository(repository.id) is None
        assert store.list_operations(repository.id) == []


def test_commit_failure_rolls_back_and_leaves_store_usable(tmp_path: Path) -> None:
    repository = Repository(root=tmp_path / "repo")
    orphaned_operation = Operation(
        repository_id=repository.id,
        kind="test.orphaned",
    )

    with SqliteStore.open(tmp_path / "borg.sqlite3") as store:
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            with store.transaction() as connection:
                connection.execute("PRAGMA defer_foreign_keys = ON")
                store.append_operation(orphaned_operation)

        store.add_repository(repository)

        assert store.get_repository(repository.id) == repository
        assert store.list_operations(repository.id) == []


def test_operation_ledger_rejects_mutation_deletion_and_replacement(
    tmp_path: Path,
) -> None:
    repository = Repository(root=tmp_path / "repo")
    operation = Operation(repository_id=repository.id, kind="test.completed")

    with SqliteStore.open(tmp_path / "borg.sqlite3") as store:
        store.add_repository(repository)
        store.append_operation(operation)

        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            with store.transaction() as connection:
                connection.execute(
                    "UPDATE operations SET kind = ? WHERE id = ?",
                    ("changed", str(operation.id)),
                )

        assert store.list_operations(repository.id) == [operation]

        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            with store.transaction() as connection:
                connection.execute(
                    "DELETE FROM operations WHERE id = ?",
                    (str(operation.id),),
                )

        assert store.list_operations(repository.id) == [operation]

        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            with store.transaction() as connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO operations(
                        id, repository_id, kind, payload, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(operation.id),
                        str(repository.id),
                        "changed",
                        "{}",
                        operation.created_at.isoformat(),
                    ),
                )

        assert store.list_operations(repository.id) == [operation]
