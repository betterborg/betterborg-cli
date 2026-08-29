-- Schema version 9. Reclaimed task resources and guarded attempt lifecycle events.

DROP TRIGGER compose_resources_are_append_only_on_update;
DROP TRIGGER compose_resources_are_append_only_on_delete;

ALTER TABLE compose_resources RENAME TO compose_resources_v8;
DROP INDEX idx_compose_resources_task_project;

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
    UNIQUE (claim_id, project_name, resource_type, resource_name),
    FOREIGN KEY (claim_id, run_id, task_id)
        REFERENCES task_claims(id, run_id, task_id) ON DELETE RESTRICT
);

INSERT INTO compose_resources
SELECT * FROM compose_resources_v8;

DROP TABLE compose_resources_v8;

CREATE INDEX idx_compose_resources_task_project
    ON compose_resources(task_id, project_name, id);

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
