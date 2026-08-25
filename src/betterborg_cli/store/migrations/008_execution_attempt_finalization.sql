-- Schema version 8. Exactly one durable terminal event per execution attempt.

CREATE UNIQUE INDEX idx_execution_events_one_terminal_per_attempt
    ON execution_events(attempt_id)
    WHERE attempt_id IS NOT NULL
      AND kind IN (
        'environment.attempt_finished',
        'environment.attempt_interrupted',
        'agent.attempt_finished',
        'agent.attempt_interrupted'
      );
