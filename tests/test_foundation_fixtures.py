"""Contract tests for fixtures shared by later foundation tasks."""

import sqlite3
import subprocess
from pathlib import Path


def test_git_repo_is_initialized(git_repo: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(git_repo), "rev-parse", "--is-inside-work-tree"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "true"


def test_sqlite_db_is_writable(sqlite_db: sqlite3.Connection) -> None:
    sqlite_db.execute("CREATE TABLE example (value TEXT NOT NULL)")
    sqlite_db.execute("INSERT INTO example VALUES (?)", ("ready",))

    row = sqlite_db.execute("SELECT value FROM example").fetchone()

    assert row == ("ready",)
