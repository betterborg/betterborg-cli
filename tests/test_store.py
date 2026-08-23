"""Contracts for the forward-only local SQLite store."""

import sqlite3
from pathlib import Path

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

    with SqliteStore.open(database) as store:
        with store.transaction():
            store.add_repository(repository)
            store.append_operation(operation)
        assert store.applied_migrations() == (1,)
        with store.locked_connection() as connection:
            applied_at = connection.execute(
                "SELECT applied_at FROM schema_version WHERE version = 1"
            ).fetchone()[0]

    with SqliteStore.open(database) as reopened:
        assert reopened.applied_migrations() == (1,)
        assert reopened.get_repository(repository.id) == repository
        assert reopened.list_operations(repository.id) == [operation]
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


def test_operation_ledger_rejects_mutation(tmp_path: Path) -> None:
    repository = Repository(root=tmp_path / "repo")
    operation = Operation(repository_id=repository.id, kind="test.completed")

    with SqliteStore.open(tmp_path / "borg.sqlite3") as store:
        store.add_repository(repository)
        store.append_operation(operation)

        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            with store.transaction() as connection:
                connection.execute(
                    "UPDATE operations SET kind = ? WHERE id = ?",
                    ("changed", operation.id),
                )

        assert store.list_operations(repository.id) == [operation]
