-- Schema version 10. Run-owned environment preparation before task claims.

DROP TRIGGER execution_events_match_attempt;
DROP TRIGGER execution_events_match_terminal_attempt_kind;
DROP TRIGGER environment_attempts_are_append_only_on_update;
DROP TRIGGER environment_attempts_are_append_only_on_delete;

ALTER TABLE environment_attempts RENAME TO environment_attempts_v9;

CREATE TABLE environment_attempts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    claim_id TEXT,
    task_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (length(trim(kind)) > 0),
    attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
    fingerprint TEXT NOT NULL CHECK (length(trim(fingerprint)) > 0),
    status TEXT NOT NULL
        CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    commands_json TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    duration_seconds REAL CHECK (
        duration_seconds IS NULL OR duration_seconds >= 0
    ),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    UNIQUE (run_id, task_id, kind, attempt_number),
    FOREIGN KEY (claim_id, run_id, task_id)
        REFERENCES task_claims(id, run_id, task_id) ON DELETE RESTRICT,
    CHECK (claim_id IS NOT NULL OR kind = 'prepare'),
    CHECK ((status = 'running') = (finished_at IS NULL)),
    CHECK (finished_at IS NULL OR finished_at >= started_at)
);

INSERT INTO environment_attempts
SELECT * FROM environment_attempts_v9;

DROP TABLE environment_attempts_v9;

CREATE INDEX idx_environment_attempts_task_started
    ON environment_attempts(task_id, started_at, id);

CREATE TRIGGER environment_attempts_are_append_only_on_update
BEFORE UPDATE ON environment_attempts
BEGIN
    SELECT RAISE(ABORT, 'environment attempts are immutable');
END;

CREATE TRIGGER environment_attempts_are_append_only_on_delete
BEFORE DELETE ON environment_attempts
BEGIN
    SELECT RAISE(ABORT, 'environment attempts are immutable');
END;

CREATE TRIGGER execution_events_match_attempt
BEFORE INSERT ON execution_events
WHEN NEW.attempt_id IS NOT NULL
 AND (
    NEW.task_id IS NULL
    OR (
        NOT EXISTS (
            SELECT 1 FROM agent_attempts
            WHERE id = NEW.attempt_id
              AND run_id = NEW.run_id
              AND task_id = NEW.task_id
        )
        AND NOT EXISTS (
            SELECT 1 FROM environment_attempts
            WHERE id = NEW.attempt_id
              AND run_id = NEW.run_id
              AND task_id = NEW.task_id
        )
    )
 )
BEGIN
    SELECT RAISE(ABORT, 'execution event attempt is not owned by its task');
END;

CREATE TRIGGER execution_events_match_terminal_attempt_kind
BEFORE INSERT ON execution_events
WHEN (
    (
        NEW.kind IN (
            'environment.attempt_finished',
            'environment.attempt_interrupted'
        )
        AND (
            NEW.attempt_id IS NULL
            OR NOT EXISTS (
                SELECT 1 FROM environment_attempts
                WHERE id = NEW.attempt_id
                  AND run_id = NEW.run_id
                  AND task_id = NEW.task_id
            )
        )
    )
    OR (
        NEW.kind IN (
            'agent.attempt_finished',
            'agent.attempt_interrupted'
        )
        AND (
            NEW.attempt_id IS NULL
            OR NOT EXISTS (
                SELECT 1 FROM agent_attempts
                WHERE id = NEW.attempt_id
                  AND run_id = NEW.run_id
                  AND task_id = NEW.task_id
            )
        )
    )
)
BEGIN
    SELECT RAISE(ABORT, 'terminal execution event kind does not match attempt');
END;
