-- Schema version 4. Durable Borg planning state and append-only history.

ALTER TABLE borgs ADD COLUMN state TEXT NOT NULL DEFAULT 'draft'
    CHECK (state IN (
        'draft',
        'architect_working',
        'architect_awaiting_answers',
        'tech_review_working',
        'plan_approval_pending',
        'pm_working',
        'supervisor_working',
        'tasks_approval_pending',
        'executing',
        'done',
        'blocked'
    ));
ALTER TABLE borgs ADD COLUMN state_version INTEGER NOT NULL DEFAULT 0
    CHECK (state_version >= 0);

CREATE INDEX idx_borgs_state ON borgs(state);

CREATE TABLE planning_attempts (
    id TEXT PRIMARY KEY,
    borg_id TEXT NOT NULL REFERENCES borgs(id) ON DELETE RESTRICT,
    phase TEXT NOT NULL CHECK (length(trim(phase)) > 0),
    round INTEGER NOT NULL CHECK (round > 0),
    adapter TEXT NOT NULL CHECK (length(trim(adapter)) > 0),
    model TEXT NOT NULL CHECK (length(trim(model)) > 0),
    request_json TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    result_json TEXT,
    summary TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    UNIQUE (id, borg_id),
    UNIQUE (borg_id, phase, round),
    CHECK ((status = 'running') = (finished_at IS NULL))
);

CREATE INDEX idx_planning_attempts_borg_started
    ON planning_attempts(borg_id, started_at, id);

CREATE TABLE planning_questions (
    id TEXT PRIMARY KEY,
    borg_id TEXT NOT NULL REFERENCES borgs(id) ON DELETE RESTRICT,
    attempt_id TEXT,
    round INTEGER NOT NULL CHECK (round > 0),
    questions_json TEXT NOT NULL,
    answers_json TEXT,
    asked_at TEXT NOT NULL,
    answered_at TEXT,
    UNIQUE (borg_id, round),
    CHECK ((answers_json IS NULL) = (answered_at IS NULL)),
    FOREIGN KEY (attempt_id, borg_id)
        REFERENCES planning_attempts(id, borg_id) ON DELETE RESTRICT
);

CREATE INDEX idx_planning_questions_borg_round
    ON planning_questions(borg_id, round);

CREATE TABLE planning_findings (
    id TEXT PRIMARY KEY,
    borg_id TEXT NOT NULL REFERENCES borgs(id) ON DELETE RESTRICT,
    attempt_id TEXT NOT NULL,
    round INTEGER NOT NULL CHECK (round > 0),
    severity TEXT NOT NULL CHECK (length(trim(severity)) > 0),
    message TEXT NOT NULL CHECK (length(trim(message)) > 0),
    suggestion TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (attempt_id, borg_id)
        REFERENCES planning_attempts(id, borg_id) ON DELETE RESTRICT
);

CREATE INDEX idx_planning_findings_borg_created
    ON planning_findings(borg_id, created_at, id);

CREATE TABLE plan_change_requests (
    id TEXT PRIMARY KEY,
    borg_id TEXT NOT NULL REFERENCES borgs(id) ON DELETE RESTRICT,
    round INTEGER NOT NULL CHECK (round > 0),
    note TEXT NOT NULL CHECK (length(trim(note)) > 0),
    decided_by TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (borg_id, round)
);

CREATE INDEX idx_plan_change_requests_borg_created
    ON plan_change_requests(borg_id, created_at, id);

CREATE TRIGGER planning_attempts_completed_are_append_only
BEFORE UPDATE ON planning_attempts
WHEN OLD.status != 'running'
BEGIN
    SELECT RAISE(ABORT, 'completed planning attempts are append-only');
END;

CREATE TRIGGER planning_attempts_identity_is_immutable
BEFORE UPDATE ON planning_attempts
WHEN NEW.id != OLD.id
  OR NEW.borg_id != OLD.borg_id
  OR NEW.phase != OLD.phase
  OR NEW.round != OLD.round
  OR NEW.adapter != OLD.adapter
  OR NEW.model != OLD.model
  OR NEW.request_json != OLD.request_json
  OR NEW.started_at != OLD.started_at
BEGIN
    SELECT RAISE(ABORT, 'planning attempt identity is immutable');
END;

CREATE TRIGGER planning_attempts_are_not_deleted
BEFORE DELETE ON planning_attempts
BEGIN
    SELECT RAISE(ABORT, 'planning attempts are append-only');
END;

CREATE TRIGGER planning_questions_are_answered_once
BEFORE UPDATE ON planning_questions
WHEN OLD.answers_json IS NOT NULL
  OR NEW.id != OLD.id
  OR NEW.borg_id != OLD.borg_id
  OR NEW.attempt_id IS NOT OLD.attempt_id
  OR NEW.round != OLD.round
  OR NEW.questions_json != OLD.questions_json
  OR NEW.asked_at != OLD.asked_at
BEGIN
    SELECT RAISE(ABORT, 'planning questions are append-only after answering');
END;

CREATE TRIGGER planning_questions_are_not_deleted
BEFORE DELETE ON planning_questions
BEGIN
    SELECT RAISE(ABORT, 'planning questions are append-only');
END;

CREATE TRIGGER planning_findings_are_append_only_on_update
BEFORE UPDATE ON planning_findings
BEGIN
    SELECT RAISE(ABORT, 'planning findings are append-only');
END;

CREATE TRIGGER planning_findings_are_append_only_on_delete
BEFORE DELETE ON planning_findings
BEGIN
    SELECT RAISE(ABORT, 'planning findings are append-only');
END;

CREATE TRIGGER plan_change_requests_are_append_only_on_update
BEFORE UPDATE ON plan_change_requests
BEGIN
    SELECT RAISE(ABORT, 'plan change requests are append-only');
END;

CREATE TRIGGER plan_change_requests_are_append_only_on_delete
BEFORE DELETE ON plan_change_requests
BEGIN
    SELECT RAISE(ABORT, 'plan change requests are append-only');
END;
