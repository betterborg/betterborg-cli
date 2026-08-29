-- Schema version 5. Approved plans and immutable task-generation snapshots.

CREATE TABLE plan_approvals (
    id TEXT PRIMARY KEY,
    borg_id TEXT NOT NULL REFERENCES borgs(id) ON DELETE RESTRICT,
    attempt_id TEXT,
    plan_digest TEXT NOT NULL CHECK (length(trim(plan_digest)) > 0),
    manifest_json TEXT NOT NULL,
    approved_by TEXT,
    approved_at TEXT NOT NULL,
    UNIQUE (id, borg_id),
    FOREIGN KEY (attempt_id, borg_id)
        REFERENCES planning_attempts(id, borg_id) ON DELETE RESTRICT
);

CREATE INDEX idx_plan_approvals_borg_approved
    ON plan_approvals(borg_id, approved_at, id);

CREATE TABLE task_batches (
    id TEXT PRIMARY KEY,
    borg_id TEXT NOT NULL REFERENCES borgs(id) ON DELETE RESTRICT,
    plan_approval_id TEXT NOT NULL,
    attempt_id TEXT,
    round INTEGER NOT NULL CHECK (round > 0),
    summary TEXT NOT NULL DEFAULT '',
    manifest_json TEXT NOT NULL,
    digest TEXT NOT NULL CHECK (length(trim(digest)) > 0),
    created_at TEXT NOT NULL,
    UNIQUE (id, borg_id),
    UNIQUE (id, borg_id, plan_approval_id),
    UNIQUE (borg_id, plan_approval_id, round),
    FOREIGN KEY (plan_approval_id, borg_id)
        REFERENCES plan_approvals(id, borg_id) ON DELETE RESTRICT,
    FOREIGN KEY (attempt_id, borg_id)
        REFERENCES planning_attempts(id, borg_id) ON DELETE RESTRICT
);

CREATE INDEX idx_task_batches_borg_created
    ON task_batches(borg_id, created_at, id);

CREATE TABLE task_findings (
    id TEXT PRIMARY KEY,
    borg_id TEXT NOT NULL REFERENCES borgs(id) ON DELETE RESTRICT,
    batch_id TEXT NOT NULL,
    attempt_id TEXT,
    round INTEGER NOT NULL CHECK (round > 0),
    severity TEXT NOT NULL CHECK (length(trim(severity)) > 0),
    message TEXT NOT NULL CHECK (length(trim(message)) > 0),
    suggestion TEXT,
    task_ref TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (batch_id, borg_id)
        REFERENCES task_batches(id, borg_id) ON DELETE RESTRICT,
    FOREIGN KEY (attempt_id, borg_id)
        REFERENCES planning_attempts(id, borg_id) ON DELETE RESTRICT
);

CREATE INDEX idx_task_findings_borg_created
    ON task_findings(borg_id, created_at, id);
CREATE INDEX idx_task_findings_batch_created
    ON task_findings(batch_id, created_at, id);

CREATE TABLE task_generations (
    id TEXT PRIMARY KEY,
    borg_id TEXT NOT NULL REFERENCES borgs(id) ON DELETE RESTRICT,
    plan_approval_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK (status IN ('preparing', 'current', 'superseded')),
    manifest_json TEXT NOT NULL,
    digest TEXT NOT NULL CHECK (length(trim(digest)) > 0),
    created_at TEXT NOT NULL,
    current_at TEXT,
    superseded_at TEXT,
    UNIQUE (id, borg_id),
    FOREIGN KEY (plan_approval_id, borg_id)
        REFERENCES plan_approvals(id, borg_id) ON DELETE RESTRICT,
    FOREIGN KEY (batch_id, borg_id, plan_approval_id)
        REFERENCES task_batches(id, borg_id, plan_approval_id) ON DELETE RESTRICT,
    CHECK (
        (status = 'preparing' AND current_at IS NULL AND superseded_at IS NULL)
        OR (status = 'current' AND current_at IS NOT NULL AND superseded_at IS NULL)
        OR (status = 'superseded' AND current_at IS NOT NULL AND superseded_at IS NOT NULL)
    )
);

CREATE INDEX idx_task_generations_borg_created
    ON task_generations(borg_id, created_at, id);
CREATE UNIQUE INDEX idx_task_generations_one_current_per_borg
    ON task_generations(borg_id) WHERE status = 'current';

CREATE TABLE task_records (
    id TEXT PRIMARY KEY,
    generation_id TEXT NOT NULL,
    borg_id TEXT NOT NULL,
    task_ref TEXT NOT NULL CHECK (length(trim(task_ref)) > 0),
    stage TEXT NOT NULL CHECK (length(trim(stage)) > 0),
    stem TEXT NOT NULL CHECK (length(trim(stem)) > 0),
    position INTEGER NOT NULL CHECK (position > 0),
    title TEXT NOT NULL CHECK (length(trim(title)) > 0),
    complexity TEXT NOT NULL CHECK (complexity IN ('small', 'medium', 'large')),
    task_json TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    digest TEXT NOT NULL CHECK (length(trim(digest)) > 0),
    created_at TEXT NOT NULL,
    UNIQUE (id, generation_id),
    UNIQUE (generation_id, task_ref),
    UNIQUE (generation_id, stage, stem),
    UNIQUE (generation_id, position),
    FOREIGN KEY (generation_id, borg_id)
        REFERENCES task_generations(id, borg_id) ON DELETE RESTRICT
);

CREATE INDEX idx_task_records_generation_position
    ON task_records(generation_id, position, id);

CREATE TRIGGER task_records_only_while_preparing
BEFORE INSERT ON task_records
WHEN (SELECT status FROM task_generations WHERE id = NEW.generation_id) != 'preparing'
BEGIN
    SELECT RAISE(ABORT, 'task generation is no longer preparing');
END;

CREATE TABLE task_dependencies (
    id TEXT PRIMARY KEY,
    generation_id TEXT NOT NULL REFERENCES task_generations(id) ON DELETE RESTRICT,
    task_id TEXT NOT NULL,
    depends_on_task_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (generation_id, task_id, depends_on_task_id),
    CHECK (task_id != depends_on_task_id),
    FOREIGN KEY (task_id, generation_id)
        REFERENCES task_records(id, generation_id) ON DELETE RESTRICT,
    FOREIGN KEY (depends_on_task_id, generation_id)
        REFERENCES task_records(id, generation_id) ON DELETE RESTRICT
);

CREATE INDEX idx_task_dependencies_generation_task
    ON task_dependencies(generation_id, task_id, depends_on_task_id);

CREATE TRIGGER task_dependencies_only_while_preparing
BEFORE INSERT ON task_dependencies
WHEN (SELECT status FROM task_generations WHERE id = NEW.generation_id) != 'preparing'
BEGIN
    SELECT RAISE(ABORT, 'task generation is no longer preparing');
END;

CREATE TRIGGER plan_approvals_are_append_only_on_update
BEFORE UPDATE ON plan_approvals
BEGIN
    SELECT RAISE(ABORT, 'plan approvals are append-only');
END;

CREATE TRIGGER plan_approvals_are_append_only_on_delete
BEFORE DELETE ON plan_approvals
BEGIN
    SELECT RAISE(ABORT, 'plan approvals are append-only');
END;

CREATE TRIGGER task_batches_are_append_only_on_update
BEFORE UPDATE ON task_batches
BEGIN
    SELECT RAISE(ABORT, 'task batches are append-only');
END;

CREATE TRIGGER task_batches_are_append_only_on_delete
BEFORE DELETE ON task_batches
BEGIN
    SELECT RAISE(ABORT, 'task batches are append-only');
END;

CREATE TRIGGER task_findings_are_append_only_on_update
BEFORE UPDATE ON task_findings
BEGIN
    SELECT RAISE(ABORT, 'task findings are append-only');
END;

CREATE TRIGGER task_findings_are_append_only_on_delete
BEFORE DELETE ON task_findings
BEGIN
    SELECT RAISE(ABORT, 'task findings are append-only');
END;

CREATE TRIGGER task_generations_only_advance_status
BEFORE UPDATE ON task_generations
WHEN NEW.id != OLD.id
  OR NEW.borg_id != OLD.borg_id
  OR NEW.plan_approval_id != OLD.plan_approval_id
  OR NEW.batch_id != OLD.batch_id
  OR NEW.manifest_json != OLD.manifest_json
  OR NEW.digest != OLD.digest
  OR NEW.created_at != OLD.created_at
  OR NOT (
      (OLD.status = 'preparing' AND NEW.status = 'current'
       AND OLD.current_at IS NULL AND NEW.current_at IS NOT NULL
       AND NEW.superseded_at IS NULL)
      OR
      (OLD.status = 'current' AND NEW.status = 'superseded'
       AND NEW.current_at = OLD.current_at
       AND OLD.superseded_at IS NULL AND NEW.superseded_at IS NOT NULL)
  )
BEGIN
    SELECT RAISE(ABORT, 'task generation metadata is immutable');
END;

CREATE TRIGGER task_generations_are_not_deleted
BEFORE DELETE ON task_generations
BEGIN
    SELECT RAISE(ABORT, 'task generations are append-only');
END;

CREATE TRIGGER task_records_are_append_only_on_update
BEFORE UPDATE ON task_records
BEGIN
    SELECT RAISE(ABORT, 'task records are immutable and append-only');
END;

CREATE TRIGGER task_records_are_append_only_on_delete
BEFORE DELETE ON task_records
BEGIN
    SELECT RAISE(ABORT, 'task records are append-only');
END;

CREATE TRIGGER task_dependencies_are_append_only_on_update
BEFORE UPDATE ON task_dependencies
BEGIN
    SELECT RAISE(ABORT, 'task dependencies are append-only');
END;

CREATE TRIGGER task_dependencies_are_append_only_on_delete
BEFORE DELETE ON task_dependencies
BEGIN
    SELECT RAISE(ABORT, 'task dependencies are append-only');
END;
