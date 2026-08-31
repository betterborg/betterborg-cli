"""Shared planning-turn cancellation and activity-routing contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.planning.turns import DurablePlanningTurns
from betterborg_cli.progress import AgentActivity, AgentActivityKind
from betterborg_cli.store import SqliteStore

_RESULT_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def test_routes_turn_activity_to_exact_parent_or_child(
    committed_git_repo: Path,
    persist_planning_context,
    recording_progress: Any,
) -> None:
    database = committed_git_repo.parent / "planning-turn-activity.db"
    parent_activity = AgentActivity(AgentActivityKind.READING, "README.md")
    child_activity = AgentActivity(AgentActivityKind.SEARCHING, "tests")
    adapter = MockAdapter().queue(
        MockResponse(payload={"ok": True}, activities=(parent_activity,))
    )
    adapter.queue(
        MockResponse(payload={"ok": True}, activities=(child_activity,))
    )

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "turn-activity"
        )
        parent = _turns(
            repository,
            borg,
            store,
            adapter,
            committed_git_repo,
            progress=recording_progress,
            stage_key="planning",
        )
        child = _turns(
            repository,
            borg,
            store,
            adapter,
            committed_git_repo,
            progress=recording_progress,
            stage_key="planning",
            child_key="revision-2",
        )

        parent.run(
            phase="parent-turn",
            round_number=1,
            schema=_RESULT_SCHEMA,
            system_prompt="Inspect the repository.",
            user_prompt="Return a result.",
        )
        child.run(
            phase="child-turn",
            round_number=2,
            schema=_RESULT_SCHEMA,
            system_prompt="Revise the result.",
            user_prompt="Return a result.",
        )

    assert recording_progress.activities == [("planning", parent_activity)]
    assert recording_progress.child_activities == [
        ("planning", "revision-2", child_activity)
    ]


def _turns(
    repository: Any,
    borg: Any,
    store: SqliteStore,
    adapter: MockAdapter,
    root: Path,
    **progress: Any,
) -> DurablePlanningTurns:
    return DurablePlanningTurns(
        repository,
        borg,
        store,
        adapter,
        role="Planner",
        model="planning-model",
        artifact_dir=root.parent / "planning-turn-artifacts",
        error_factory=RuntimeError,
        cancelled_error_factory=RuntimeError,
        **progress,
    )
