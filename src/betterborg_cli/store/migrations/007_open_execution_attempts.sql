-- Schema version 7. Durable open execution attempts closed by lifecycle events.

DROP TRIGGER execution_events_match_attempt;

DROP TRIGGER environment_attempts_are_append_only_on_update;
DROP TRIGGER environment_attempts_are_append_only_on_delete;

ALTER TABLE environment_attempts RENAME TO environment_attempts_v6;

CREATE TABLE environment_attempts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
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
    CHECK ((status = 'running') = (finished_at IS NULL)),
    CHECK (finished_at IS NULL OR finished_at >= started_at)
);

INSERT INTO environment_attempts
SELECT * FROM environment_attempts_v6;

DROP TABLE environment_attempts_v6;

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

DROP TRIGGER agent_attempts_are_append_only_on_update;
DROP TRIGGER agent_attempts_are_append_only_on_delete;

ALTER TABLE agent_attempts RENAME TO agent_attempts_v6;

CREATE TABLE agent_attempts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    phase TEXT NOT NULL CHECK (length(trim(phase)) > 0),
    review_round INTEGER NOT NULL DEFAULT 0 CHECK (review_round >= 0),
    attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
    adapter TEXT NOT NULL CHECK (length(trim(adapter)) > 0),
    model TEXT NOT NULL CHECK (length(trim(model)) > 0),
    billing_mode TEXT NOT NULL CHECK (billing_mode IN ('api', 'subscription')),
    status TEXT NOT NULL
        CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    log_path TEXT NOT NULL CHECK (length(trim(log_path)) > 0),
    result_path TEXT,
    result_json TEXT,
    summary TEXT,
    duration_seconds REAL CHECK (
        duration_seconds IS NULL OR duration_seconds >= 0
    ),
    cost_usd REAL CHECK (cost_usd IS NULL OR cost_usd >= 0),
    tokens_input INTEGER CHECK (tokens_input IS NULL OR tokens_input >= 0),
    tokens_output INTEGER CHECK (tokens_output IS NULL OR tokens_output >= 0),
    tokens_cache_read INTEGER CHECK (
        tokens_cache_read IS NULL OR tokens_cache_read >= 0
    ),
    tokens_cache_write INTEGER CHECK (
        tokens_cache_write IS NULL OR tokens_cache_write >= 0
    ),
    num_turns INTEGER CHECK (num_turns IS NULL OR num_turns >= 0),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    UNIQUE (run_id, task_id, phase, review_round, attempt_number),
    FOREIGN KEY (claim_id, run_id, task_id)
        REFERENCES task_claims(id, run_id, task_id) ON DELETE RESTRICT,
    CHECK ((status = 'running') = (finished_at IS NULL)),
    CHECK (finished_at IS NULL OR finished_at >= started_at)
);

INSERT INTO agent_attempts
SELECT * FROM agent_attempts_v6;

DROP TABLE agent_attempts_v6;

CREATE INDEX idx_agent_attempts_task_started
    ON agent_attempts(task_id, started_at, id);

CREATE TRIGGER agent_attempts_are_append_only_on_update
BEFORE UPDATE ON agent_attempts
BEGIN
    SELECT RAISE(ABORT, 'agent attempts are immutable');
END;

CREATE TRIGGER agent_attempts_are_append_only_on_delete
BEFORE DELETE ON agent_attempts
BEGIN
    SELECT RAISE(ABORT, 'agent attempts are immutable');
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
