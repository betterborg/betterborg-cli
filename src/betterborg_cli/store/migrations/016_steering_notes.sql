-- Schema version 16. A granted round that was steered records the note it ran
-- on, so re-entering it costs nothing and a round that attached nothing can be
-- told from one that was never steered at all.

CREATE TABLE steering_notes (
    id TEXT PRIMARY KEY,
    borg_id TEXT NOT NULL REFERENCES borgs(id) ON DELETE RESTRICT,
    -- The loop whose argument was read, left an open string for the reason the
    -- assessments beside it leave theirs: the loops bring their own names.
    loop TEXT NOT NULL CHECK (length(trim(loop)) > 0),
    -- The run of rounds the note belongs to, one column per kind of scope a
    -- loop can have and null in the ones its loop does not, exactly as the
    -- assessments it is read beside carry theirs.
    cycle_id TEXT CHECK (cycle_id IS NULL OR length(trim(cycle_id)) > 0),
    plan_approval_id TEXT,
    batch_id TEXT,
    task_id TEXT,
    -- The round whose assessment asked for the note, in the ledger's
    -- numbering, and never the number the steering attempt itself is filed
    -- under: that is the steering phase's own running count and would collide
    -- across cycles.
    round INTEGER NOT NULL CHECK (round > 0),
    attempt_id TEXT,
    note TEXT NOT NULL CHECK (length(trim(note)) > 0),
    -- Which note the round actually ran on. A round that fell back attached
    -- something rather than nothing, and only this tells it apart from a round
    -- that was never steered.
    source TEXT NOT NULL CHECK (source IN ('agent', 'assembled')),
    -- The verdict that asked for the note, recorded beside it so the row says
    -- why it exists without a reader going back to the assessment.
    converging INTEGER NOT NULL CHECK (converging IN (0, 1)),
    created_at TEXT NOT NULL,
    FOREIGN KEY (attempt_id, borg_id)
        REFERENCES planning_attempts(id, borg_id) ON DELETE RESTRICT,
    FOREIGN KEY (plan_approval_id, borg_id)
        REFERENCES plan_approvals(id, borg_id) ON DELETE RESTRICT,
    FOREIGN KEY (batch_id, borg_id)
        REFERENCES task_batches(id, borg_id) ON DELETE RESTRICT
);

-- One note per grant, however many times the round carrying it is re-entered:
-- a resumed round finds the note already written for it rather than paying for
-- a second turn. Coalesced for the reason the assessments' index is: a null
-- does not compare equal to itself, and a loop leaves null every scope it does
-- not have.
CREATE UNIQUE INDEX idx_steering_notes_round
    ON steering_notes(
        borg_id,
        loop,
        COALESCE(cycle_id, ''),
        COALESCE(plan_approval_id, ''),
        COALESCE(batch_id, ''),
        COALESCE(task_id, ''),
        round
    );

CREATE INDEX idx_steering_notes_scope
    ON steering_notes(borg_id, loop, created_at, id);
