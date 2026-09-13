-- Schema version 15. Execution review objections keep a lifecycle across rounds.

CREATE TABLE execution_finding_ledger (
    id TEXT PRIMARY KEY,
    -- The task under review, which is the run of rounds this ledger compares.
    -- Every round reviews the same task, so the scope never moves under it.
    task_id TEXT NOT NULL REFERENCES task_records(id) ON DELETE RESTRICT,
    -- The review attempt that last established the row's status. The rows are
    -- written in the transaction that completes that attempt, so a round
    -- interrupted before it leaves nothing behind for a later round to answer.
    attempt_id TEXT NOT NULL REFERENCES agent_attempts(id) ON DELETE RESTRICT,
    first_seen_round INTEGER NOT NULL CHECK (first_seen_round > 0),
    last_seen_round INTEGER NOT NULL
        CHECK (last_seen_round >= first_seen_round),
    status TEXT NOT NULL
        CHECK (status IN ('open', 'resolved', 'regressed')),
    severity TEXT NOT NULL CHECK (length(trim(severity)) > 0),
    message TEXT NOT NULL CHECK (length(trim(message)) > 0),
    suggestion TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_execution_finding_ledger_task
    ON execution_finding_ledger(task_id, created_at, id);
