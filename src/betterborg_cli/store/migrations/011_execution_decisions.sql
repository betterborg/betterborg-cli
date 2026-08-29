-- Schema version 11. Immutable execution decisions bound to task generations.

CREATE TABLE execution_decisions (
    id TEXT PRIMARY KEY,
    borg_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    approved_plan_digest TEXT NOT NULL
        CHECK (length(trim(approved_plan_digest)) > 0),
    task_batch_digest TEXT NOT NULL
        CHECK (length(trim(task_batch_digest)) > 0),
    estimate_version TEXT NOT NULL
        CHECK (length(trim(estimate_version)) > 0),
    source TEXT NOT NULL CHECK (length(trim(source)) > 0),
    snapshot_json TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (length(trim(decision)) > 0),
    decided_at TEXT NOT NULL,
    UNIQUE (borg_id, generation_id),
    FOREIGN KEY (generation_id, borg_id)
        REFERENCES task_generations(id, borg_id) ON DELETE RESTRICT
);

CREATE TRIGGER execution_decisions_match_current_generation
BEFORE INSERT ON execution_decisions
WHEN NOT EXISTS (
    SELECT 1
    FROM task_generations AS generation
    JOIN plan_approvals AS approval
      ON approval.id = generation.plan_approval_id
     AND approval.borg_id = generation.borg_id
    JOIN task_batches AS batch
      ON batch.id = generation.batch_id
     AND batch.borg_id = generation.borg_id
     AND batch.plan_approval_id = approval.id
    WHERE generation.id = NEW.generation_id
      AND generation.borg_id = NEW.borg_id
      AND generation.status = 'current'
      AND approval.plan_digest = NEW.approved_plan_digest
      AND batch.digest = NEW.task_batch_digest
)
BEGIN
    SELECT RAISE(
        ABORT,
        'execution decision does not match the current generation'
    );
END;

CREATE TRIGGER execution_decisions_are_append_only_on_update
BEFORE UPDATE ON execution_decisions
BEGIN
    SELECT RAISE(ABORT, 'execution decisions are immutable');
END;

CREATE TRIGGER execution_decisions_are_append_only_on_delete
BEFORE DELETE ON execution_decisions
BEGIN
    SELECT RAISE(ABORT, 'execution decisions are immutable');
END;
