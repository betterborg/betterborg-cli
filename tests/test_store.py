"""Contracts for the forward-only local SQLite store."""

import sqlite3
from pathlib import Path
from uuid import UUID

import pytest

from betterborg_cli.store import Borg, Operation, PrdSession, Repository, SqliteStore


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
        assert store.applied_migrations() == (1, 2, 3, 4)
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
        assert reopened.applied_migrations() == (1, 2, 3, 4)
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


def test_borg_and_prd_session_history_survive_reopen(tmp_path: Path) -> None:
    database = tmp_path / "state" / "borg.sqlite3"
    repository = Repository(root=tmp_path / "repo")
    borg = Borg(repository_id=repository.id, name="Ada")
    session = PrdSession(
        repository_id=repository.id,
        borg_id=borg.id,
        prd_path=Path(".borg/prds/first-product.md"),
    )

    with SqliteStore.open(database) as store:
        store.add_repository(repository)
        store.add_borg(borg)
        store.add_prd_session(session)
        user_turn = store.append_prd_turn(
            session_id=session.id,
            role="user",
            content="The CLI should make the first run approachable.",
        )
        assistant_turn = store.append_prd_turn(
            session_id=session.id,
            role="assistant",
            content="Which workflow should the first run prioritize?",
        )
        with store.locked_connection() as connection:
            applied_at = connection.execute(
                "SELECT applied_at FROM schema_version WHERE version = 3"
            ).fetchone()[0]
            session_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(prd_sessions)")
            }

        assert user_turn.position == 1
        assert assistant_turn.position == 2
        assert "prd_path" in session_columns
        assert "content" not in session_columns
        assert "body_md" not in session_columns

    with SqliteStore.open(database) as reopened:
        assert reopened.applied_migrations() == (1, 2, 3, 4)
        assert reopened.get_borg(borg.id) == borg
        assert reopened.get_borg_by_name(repository.id, "Ada") == borg
        assert reopened.get_prd_session(session.id) == session
        assert reopened.list_prd_turns(session.id) == [user_turn, assistant_turn]
        with reopened.locked_connection() as connection:
            reopened_applied_at = connection.execute(
                "SELECT applied_at FROM schema_version WHERE version = 3"
            ).fetchone()[0]

    assert reopened_applied_at == applied_at


def test_borg_names_are_unique_within_a_repository(tmp_path: Path) -> None:
    repository = Repository(root=tmp_path / "repo")

    with SqliteStore.open(tmp_path / "borg.sqlite3") as store:
        store.add_repository(repository)
        store.add_borg(Borg(repository_id=repository.id, name="Ada"))

        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            store.add_borg(Borg(repository_id=repository.id, name="Ada"))


@pytest.mark.parametrize("statement", ["UPDATE", "DELETE", "REPLACE"])
def test_prd_turns_reject_mutation_deletion_and_replacement(
    tmp_path: Path, statement: str
) -> None:
    repository = Repository(root=tmp_path / "repo")
    borg = Borg(repository_id=repository.id, name="Ada")
    session = PrdSession(
        repository_id=repository.id,
        borg_id=borg.id,
        prd_path=Path(".borg/prds/product.md"),
    )

    with SqliteStore.open(tmp_path / "borg.sqlite3") as store:
        store.add_repository(repository)
        store.add_borg(borg)
        store.add_prd_session(session)
        turn = store.append_prd_turn(
            session_id=session.id, role="user", content="Keep this turn."
        )

        sql, parameters = {
            "UPDATE": (
                "UPDATE prd_turns SET content = ? WHERE id = ?",
                ("changed", str(turn.id)),
            ),
            "DELETE": (
                "DELETE FROM prd_turns WHERE id = ?",
                (str(turn.id),),
            ),
            "REPLACE": (
                """
                INSERT OR REPLACE INTO prd_turns(
                    id, session_id, position, role, content, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(turn.id),
                    str(turn.session_id),
                    turn.position,
                    turn.role,
                    "changed",
                    turn.created_at.isoformat(),
                ),
            ),
        }[statement]

        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            with store.transaction() as connection:
                connection.execute(sql, parameters)

        assert store.list_prd_turns(session.id) == [turn]
