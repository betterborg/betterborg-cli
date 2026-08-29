-- Schema version 3. Named Borg identities and append-only PRD session turns.

CREATE TABLE borgs (
    id TEXT PRIMARY KEY,
    repository_id TEXT NOT NULL REFERENCES repositories(id) ON DELETE RESTRICT,
    name TEXT NOT NULL CHECK (length(trim(name)) > 0),
    created_at TEXT NOT NULL,
    UNIQUE (repository_id, name),
    UNIQUE (id, repository_id)
);

CREATE TABLE prd_sessions (
    id TEXT PRIMARY KEY,
    repository_id TEXT NOT NULL,
    borg_id TEXT NOT NULL,
    prd_path TEXT NOT NULL CHECK (length(trim(prd_path)) > 0),
    created_at TEXT NOT NULL,
    FOREIGN KEY (borg_id, repository_id)
        REFERENCES borgs(id, repository_id) ON DELETE RESTRICT
);

CREATE INDEX idx_prd_sessions_repository_created
    ON prd_sessions(repository_id, created_at, id);

CREATE TABLE prd_turns (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES prd_sessions(id) ON DELETE RESTRICT,
    position INTEGER NOT NULL CHECK (position > 0),
    role TEXT NOT NULL CHECK (length(trim(role)) > 0),
    content TEXT NOT NULL CHECK (length(content) > 0),
    created_at TEXT NOT NULL,
    UNIQUE (session_id, position)
);

CREATE INDEX idx_prd_turns_session_position
    ON prd_turns(session_id, position);

CREATE TRIGGER prd_turns_are_append_only_on_update
BEFORE UPDATE ON prd_turns
BEGIN
    SELECT RAISE(ABORT, 'PRD turns are append-only');
END;

CREATE TRIGGER prd_turns_are_append_only_on_delete
BEFORE DELETE ON prd_turns
BEGIN
    SELECT RAISE(ABORT, 'PRD turns are append-only');
END;
