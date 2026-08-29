-- Schema version 1. Local repository identity and append-only operations.

CREATE TABLE repositories (
    id TEXT PRIMARY KEY,
    root TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE operations (
    id TEXT PRIMARY KEY,
    repository_id TEXT NOT NULL REFERENCES repositories(id) ON DELETE RESTRICT,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_operations_repository_created
    ON operations(repository_id, created_at, id);

CREATE TRIGGER operations_are_append_only_on_update
BEFORE UPDATE ON operations
BEGIN
    SELECT RAISE(ABORT, 'operations are append-only');
END;

CREATE TRIGGER operations_are_append_only_on_delete
BEFORE DELETE ON operations
BEGIN
    SELECT RAISE(ABORT, 'operations are append-only');
END;
