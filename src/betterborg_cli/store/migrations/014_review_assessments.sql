-- Schema version 14. Review loops record what each of their rounds showed.

CREATE TABLE review_assessments (
    id TEXT PRIMARY KEY,
    borg_id TEXT NOT NULL REFERENCES borgs(id) ON DELETE RESTRICT,
    -- The loop that assessed itself, left an open string rather than a closed
    -- set: the loops that start recording here later bring their own names
    -- and no storage of their own.
    loop TEXT NOT NULL CHECK (length(trim(loop)) > 0),
    -- The run of rounds the assessment compares, one column per kind of scope
    -- a loop can have, and null in the ones its loop does not. A loop scoped by
    -- a cycle always names one, because the first cycle carries a sentinel
    -- rather than a null.
    cycle_id TEXT CHECK (cycle_id IS NULL OR length(trim(cycle_id)) > 0),
    plan_approval_id TEXT,
    batch_id TEXT,
    task_id TEXT,
    round INTEGER NOT NULL CHECK (round > 0),
    -- The minimum this round ran under, recorded rather than read back from
    -- configuration: a setting edited afterwards does not move a loop that has
    -- already stopped, so it must not move the account of why it stopped.
    minimum INTEGER NOT NULL CHECK (minimum > 0),
    attempt_id TEXT,
    converging INTEGER NOT NULL CHECK (converging IN (0, 1)),
    -- The count the refund compares on, recorded once and never recomputed:
    -- the drain in the evidence beside it is read from current lifecycle
    -- state, so an objection that regresses raises its earlier rounds' counts
    -- after the fact. Null where a loop's own test is not a count, which a
    -- zero would misreport as a round that left nothing open.
    open_findings INTEGER CHECK (open_findings IS NULL OR open_findings >= 0),
    -- Set only on a granted round, the one kind that can earn a refund: a
    -- round inside the minimum is neither charged nor refunded.
    refunded INTEGER CHECK (refunded IS NULL OR refunded IN (0, 1)),
    evidence TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (attempt_id, borg_id)
        REFERENCES planning_attempts(id, borg_id) ON DELETE RESTRICT,
    FOREIGN KEY (plan_approval_id, borg_id)
        REFERENCES plan_approvals(id, borg_id) ON DELETE RESTRICT,
    FOREIGN KEY (batch_id, borg_id)
        REFERENCES task_batches(id, borg_id) ON DELETE RESTRICT
);

-- One assessment per round of one scope, which every loop's scope holds still
-- for the run of rounds it compares. The budget is read by counting the rounds
-- that earned no refund, so a second row for a round that earned one lifts the
-- refund count above the number of grants and the loop never stops. Coalesced
-- because a unique constraint over these columns would enforce nothing: a null
-- does not compare equal to itself, and a loop leaves null every scope it does
-- not have.
CREATE UNIQUE INDEX idx_review_assessments_round
    ON review_assessments(
        borg_id,
        loop,
        COALESCE(cycle_id, ''),
        COALESCE(plan_approval_id, ''),
        COALESCE(batch_id, ''),
        COALESCE(task_id, ''),
        round
    );

CREATE INDEX idx_review_assessments_scope
    ON review_assessments(borg_id, loop, created_at, id);
