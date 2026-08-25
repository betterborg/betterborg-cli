-- Schema version 6. Lease-owned, resumable host-execution state.

CREATE TABLE execution_runs (
    id TEXT PRIMARY KEY,
    borg_id TEXT NOT NULL REFERENCES borgs(id) ON DELETE RESTRICT,
    generation_id TEXT NOT NULL,
    owner_token TEXT NOT NULL UNIQUE CHECK (length(owner_token) >= 32),
    status TEXT NOT NULL
        CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    started_at TEXT NOT NULL,
    heartbeat_at TEXT,
    lease_expires_at TEXT NOT NULL,
    finished_at TEXT,
    UNIQUE (id, generation_id),
    FOREIGN KEY (generation_id, borg_id)
        REFERENCES task_generations(id, borg_id) ON DELETE RESTRICT,
    CHECK (lease_expires_at > started_at),
    CHECK ((status = 'running') = (finished_at IS NULL))
);

CREATE INDEX idx_execution_runs_borg_started
    ON execution_runs(borg_id, started_at, id);
CREATE UNIQUE INDEX idx_execution_runs_one_live_per_borg
    ON execution_runs(borg_id) WHERE status = 'running';

CREATE TABLE task_runtimes (
    id TEXT PRIMARY KEY,
    generation_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN (
        'pending', 'claimed', 'environment', 'coding', 'review', 'fix',
        'merging', 'done', 'blocked', 'failed'
    )),
    resume_phase TEXT NOT NULL CHECK (length(trim(resume_phase)) > 0),
    review_round INTEGER NOT NULL DEFAULT 0 CHECK (review_round >= 0),
    state_reason TEXT,
    branch TEXT,
    worktree_path TEXT,
    last_run_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (task_id),
    UNIQUE (id, task_id),
    FOREIGN KEY (task_id, generation_id)
        REFERENCES task_records(id, generation_id) ON DELETE RESTRICT,
    FOREIGN KEY (last_run_id, generation_id)
        REFERENCES execution_runs(id, generation_id) ON DELETE RESTRICT,
    CHECK (updated_at >= created_at)
);

CREATE INDEX idx_task_runtimes_generation_status
    ON task_runtimes(generation_id, status, task_id);

CREATE TABLE task_claims (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    claim_token TEXT NOT NULL UNIQUE CHECK (length(claim_token) >= 32),
    resume_phase TEXT NOT NULL CHECK (length(trim(resume_phase)) > 0),
    claimed_at TEXT NOT NULL,
    lease_expires_at TEXT NOT NULL,
    released_at TEXT,
    UNIQUE (id, run_id, task_id),
    FOREIGN KEY (run_id, generation_id)
        REFERENCES execution_runs(id, generation_id) ON DELETE RESTRICT,
    FOREIGN KEY (task_id, generation_id)
        REFERENCES task_records(id, generation_id) ON DELETE RESTRICT,
    CHECK (lease_expires_at > claimed_at),
    CHECK (released_at IS NULL OR released_at >= claimed_at)
);

CREATE INDEX idx_task_claims_run_claimed
    ON task_claims(run_id, claimed_at, id);
CREATE UNIQUE INDEX idx_task_claims_one_live_per_task
    ON task_claims(task_id) WHERE released_at IS NULL;

CREATE TABLE environment_attempts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (length(trim(kind)) > 0),
    attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
    fingerprint TEXT NOT NULL CHECK (length(trim(fingerprint)) > 0),
    status TEXT NOT NULL CHECK (status IN ('completed', 'failed', 'cancelled')),
    commands_json TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    duration_seconds REAL CHECK (
        duration_seconds IS NULL OR duration_seconds >= 0
    ),
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    UNIQUE (run_id, task_id, kind, attempt_number),
    FOREIGN KEY (claim_id, run_id, task_id)
        REFERENCES task_claims(id, run_id, task_id) ON DELETE RESTRICT,
    CHECK (finished_at >= started_at)
);

CREATE INDEX idx_environment_attempts_task_started
    ON environment_attempts(task_id, started_at, id);

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
    status TEXT NOT NULL CHECK (status IN ('completed', 'failed', 'cancelled')),
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
    finished_at TEXT NOT NULL,
    UNIQUE (run_id, task_id, phase, review_round, attempt_number),
    FOREIGN KEY (claim_id, run_id, task_id)
        REFERENCES task_claims(id, run_id, task_id) ON DELETE RESTRICT,
    CHECK (finished_at >= started_at)
);

CREATE INDEX idx_agent_attempts_task_started
    ON agent_attempts(task_id, started_at, id);

CREATE TABLE execution_events (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES execution_runs(id) ON DELETE RESTRICT,
    task_id TEXT REFERENCES task_records(id) ON DELETE RESTRICT,
    attempt_id TEXT,
    kind TEXT NOT NULL CHECK (length(trim(kind)) > 0),
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_execution_events_run_created
    ON execution_events(run_id, created_at, id);
CREATE INDEX idx_execution_events_task_created
    ON execution_events(task_id, created_at, id);

CREATE TRIGGER execution_events_match_run_task
BEFORE INSERT ON execution_events
WHEN NEW.task_id IS NOT NULL
 AND NOT EXISTS (
    SELECT 1
    FROM execution_runs AS run
    JOIN task_records AS task ON task.generation_id = run.generation_id
    WHERE run.id = NEW.run_id AND task.id = NEW.task_id
 )
BEGIN
    SELECT RAISE(ABORT, 'execution event task is not owned by its run');
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

CREATE TABLE compose_resources (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    project_name TEXT NOT NULL CHECK (length(trim(project_name)) > 0),
    resource_type TEXT NOT NULL CHECK (length(trim(resource_type)) > 0),
    resource_name TEXT NOT NULL CHECK (length(trim(resource_name)) > 0),
    labels_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (task_id, project_name, resource_type, resource_name),
    FOREIGN KEY (claim_id, run_id, task_id)
        REFERENCES task_claims(id, run_id, task_id) ON DELETE RESTRICT
);

CREATE INDEX idx_compose_resources_task_project
    ON compose_resources(task_id, project_name, id);

CREATE TRIGGER execution_runs_ownership_is_immutable
BEFORE UPDATE ON execution_runs
WHEN NEW.id != OLD.id
  OR NEW.borg_id != OLD.borg_id
  OR NEW.generation_id != OLD.generation_id
  OR NEW.owner_token != OLD.owner_token
  OR NEW.started_at != OLD.started_at
BEGIN
    SELECT RAISE(ABORT, 'execution run ownership is immutable');
END;

CREATE TRIGGER execution_runs_are_not_deleted
BEFORE DELETE ON execution_runs
BEGIN
    SELECT RAISE(ABORT, 'execution runs are durable');
END;

CREATE TRIGGER task_runtimes_identity_is_immutable
BEFORE UPDATE ON task_runtimes
WHEN NEW.id != OLD.id
  OR NEW.generation_id != OLD.generation_id
  OR NEW.task_id != OLD.task_id
  OR NEW.created_at != OLD.created_at
BEGIN
    SELECT RAISE(ABORT, 'task runtime identity is immutable');
END;

CREATE TRIGGER task_runtimes_are_not_deleted
BEFORE DELETE ON task_runtimes
BEGIN
    SELECT RAISE(ABORT, 'task runtimes are durable');
END;

CREATE TRIGGER task_claims_ownership_is_immutable
BEFORE UPDATE ON task_claims
WHEN NEW.id != OLD.id
  OR NEW.run_id != OLD.run_id
  OR NEW.generation_id != OLD.generation_id
  OR NEW.task_id != OLD.task_id
  OR NEW.claim_token != OLD.claim_token
  OR NEW.resume_phase != OLD.resume_phase
  OR NEW.claimed_at != OLD.claimed_at
BEGIN
    SELECT RAISE(ABORT, 'task claim ownership is immutable');
END;

CREATE TRIGGER task_claims_are_not_deleted
BEFORE DELETE ON task_claims
BEGIN
    SELECT RAISE(ABORT, 'task claims are durable');
END;

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

CREATE TRIGGER execution_events_are_append_only_on_update
BEFORE UPDATE ON execution_events
BEGIN
    SELECT RAISE(ABORT, 'execution events are append-only');
END;

CREATE TRIGGER execution_events_are_append_only_on_delete
BEFORE DELETE ON execution_events
BEGIN
    SELECT RAISE(ABORT, 'execution events are append-only');
END;

CREATE TRIGGER compose_resources_are_append_only_on_update
BEFORE UPDATE ON compose_resources
BEGIN
    SELECT RAISE(ABORT, 'Compose resource ownership is immutable');
END;

CREATE TRIGGER compose_resources_are_append_only_on_delete
BEFORE DELETE ON compose_resources
BEGIN
    SELECT RAISE(ABORT, 'Compose resource ownership is durable');
END;
