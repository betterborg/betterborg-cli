-- Schema version 13. Reviewer objections keep a lifecycle across rounds.

CREATE TABLE planning_finding_ledger (
    id TEXT PRIMARY KEY,
    borg_id TEXT NOT NULL REFERENCES borgs(id) ON DELETE RESTRICT,
    -- The plan change request that opened the cycle, or a sentinel for the
    -- first cycle, which has none. A null would not compare equal to itself,
    -- so every lookup would miss the commonest case there is.
    cycle_id TEXT NOT NULL CHECK (length(trim(cycle_id)) > 0),
    attempt_id TEXT NOT NULL,
    first_seen_round INTEGER NOT NULL CHECK (first_seen_round > 0),
    last_seen_round INTEGER NOT NULL
        CHECK (last_seen_round >= first_seen_round),
    status TEXT NOT NULL
        CHECK (status IN ('open', 'resolved', 'regressed')),
    severity TEXT NOT NULL CHECK (length(trim(severity)) > 0),
    message TEXT NOT NULL CHECK (length(trim(message)) > 0),
    suggestion TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (attempt_id, borg_id)
        REFERENCES planning_attempts(id, borg_id) ON DELETE RESTRICT
);

CREATE INDEX idx_planning_finding_ledger_cycle
    ON planning_finding_ledger(borg_id, cycle_id, created_at, id);

CREATE TABLE task_finding_ledger (
    id TEXT PRIMARY KEY,
    borg_id TEXT NOT NULL REFERENCES borgs(id) ON DELETE RESTRICT,
    plan_approval_id TEXT NOT NULL,
    -- The batch the objection was first raised against, which labels where
    -- the task reference came from. Every revision mints new references, so
    -- neither survives into the batch under review.
    batch_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    first_seen_round INTEGER NOT NULL CHECK (first_seen_round > 0),
    last_seen_round INTEGER NOT NULL
        CHECK (last_seen_round >= first_seen_round),
    status TEXT NOT NULL
        CHECK (status IN ('open', 'resolved', 'regressed')),
    severity TEXT NOT NULL CHECK (length(trim(severity)) > 0),
    message TEXT NOT NULL CHECK (length(trim(message)) > 0),
    suggestion TEXT,
    task_ref TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (batch_id, borg_id, plan_approval_id)
        REFERENCES task_batches(id, borg_id, plan_approval_id) ON DELETE RESTRICT,
    FOREIGN KEY (attempt_id, borg_id)
        REFERENCES planning_attempts(id, borg_id) ON DELETE RESTRICT
);

CREATE INDEX idx_task_finding_ledger_approval
    ON task_finding_ledger(borg_id, plan_approval_id, created_at, id);
