"""Protocol contracts for BetterBorg's typed MCP workflow surface."""

from __future__ import annotations

import json
import select
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import anyio
from mcp.shared.memory import create_connected_server_and_client_session

from betterborg_cli import cli as cli_module
from betterborg_cli import mcp_server
from betterborg_cli.agent_runtime import AgentStatus, AgentUsage, BillingMode
from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.host_execution import HostExecutionResult, HostPreflightPlan
from betterborg_cli.host_execution.scheduler import HostSchedulerResult
from betterborg_cli.planning import TaskPublisher, build_plan_element_catalog
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.store import (
    AgentAttempt,
    Borg,
    BorgState,
    ExecutionRun,
    ExecutionRunStatus,
    PlanApproval,
    PlanningAttempt,
    PlanningAttemptStatus,
    SqliteStore,
    TaskClaim,
    TaskRuntime,
    TaskRuntimeStatus,
)
from betterborg_cli.store.models import utcnow


def _call_tool(name: str, arguments: dict | None = None):
    async def call():
        async with create_connected_server_and_client_session(
            mcp_server.server,
            raise_exceptions=True,
        ) as session:
            return await session.call_tool(name, arguments or {})

    return anyio.run(call)


def _list_tools():
    async def list_tools():
        async with create_connected_server_and_client_session(
            mcp_server.server,
            raise_exceptions=True,
        ) as session:
            return await session.list_tools()

    return anyio.run(list_tools).tools


def _structured(result) -> dict:
    assert result.isError is False
    assert result.structuredContent is not None
    return result.structuredContent


def _task_body() -> dict:
    return {
        "stage": "01-foundation",
        "stem": "01-runtime",
        "title": "Project runtime task status",
        "why": "Consumers need one runtime projection.",
        "scope": ["Expose runtime state."],
        "implementation_notes": [],
        "acceptance_criteria": ["Runtime state is exact."],
        "tests": ["Exercise runtime projection."],
        "dependencies": [],
        "out_of_scope": [],
        "plan_refs": ["P1.deliverable.1"],
        "estimate_complexity": "small",
    }


def _pm_tasks(plan: dict) -> dict:
    refs = [
        element.ref for element in build_plan_element_catalog(plan) if element.required
    ]
    return {
        "summary": "One task covers the approved plan.",
        "tasks": [
            {
                "stage": "01-release-workflow",
                "stem": "01-document-release",
                "title": "Document the release workflow",
                "why": "The approved workflow needs an executable task.",
                "scope": ["Document the release path."],
                "implementation_notes": [],
                "acceptance_criteria": ["The release path is documented."],
                "tests": ["Assert the documented public workflow."],
                "dependencies": [],
                "out_of_scope": [],
                "plan_refs": refs,
                "estimate_complexity": "small",
            }
        ],
    }


def _published_runtime(
    root: Path,
    planning_cli_repository,
    approved_task_generation,
):
    repository, paths = planning_cli_repository(root, "mcp-runtime")
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "mcp-runtime")
        assert borg is not None
        borg = store.compare_and_set_borg_state(
            borg.id,
            expected_state=borg.state,
            expected_version=borg.state_version,
            new_state=BorgState.READY_TO_EXECUTE,
        )
        approval = PlanApproval(
            borg_id=borg.id,
            plan_digest="sha256:mcp-approved-plan",
            manifest={},
        )
        store.append_plan_approval(approval)
        current = approved_task_generation(
            store,
            borg,
            approval,
            body=_task_body(),
            round_number=1,
            task_ref="T-MCP-1",
        )
        publication = TaskPublisher(repository, store).publish(current.generation.id)

        started_at = utcnow()
        run = ExecutionRun(
            borg_id=borg.id,
            generation_id=current.generation.id,
            started_at=started_at,
            heartbeat_at=started_at,
            lease_expires_at=started_at + timedelta(minutes=5),
        )
        runtime = TaskRuntime(
            generation_id=current.generation.id,
            task_id=current.task.id,
            status=TaskRuntimeStatus.FIX,
            state_reason="review requested changes",
            review_round=2,
        )
        claim = TaskClaim(
            run_id=run.id,
            task_id=current.task.id,
            resume_phase="fix",
            claimed_at=started_at,
            lease_expires_at=started_at + timedelta(minutes=2),
        )
        store.add_execution_run(run)
        store.add_task_runtime(runtime)
        store.append_task_claim(claim)
        for index, (billing, cost, duration) in enumerate(
            (
                (BillingMode.API, 0.75, 4.0),
                (BillingMode.SUBSCRIPTION, 8.0, 6.0),
            ),
            start=1,
        ):
            store.append_agent_attempt(
                AgentAttempt(
                    run_id=run.id,
                    claim_id=claim.id,
                    task_id=current.task.id,
                    phase=f"phase-{index}",
                    attempt_number=1,
                    adapter="mock",
                    model="test-model",
                    billing_mode=billing,
                    status=AgentStatus.COMPLETED,
                    log_path=f"artifacts/{index}.log",
                    duration_seconds=duration,
                    usage=AgentUsage(cost_usd=cost),
                    started_at=started_at,
                    finished_at=started_at + timedelta(seconds=duration),
                ),
                run.owner_token,
                claim.claim_token,
                now=started_at,
            )
    return paths, borg, current, publication


def test_tool_inventory_has_typed_results_and_no_removed_gates() -> None:
    tools = _list_tools()

    assert [tool.name for tool in tools] == [
        "init",
        "analyze",
        "create",
        "plan",
        "task_list",
        "execute",
    ]
    assert all(tool.outputSchema is not None for tool in tools)
    assert {"approve_task", "task_approve", "decompose"}.isdisjoint(
        tool.name for tool in tools
    )
    plan_schema = next(tool.inputSchema for tool in tools if tool.name == "plan")
    assert plan_schema["properties"]["action"]["enum"] == [
        "start",
        "show",
        "change",
        "approve",
    ]


def test_init_and_analyze_use_repository_service_with_typed_actions(
    committed_git_repo: Path,
    monkeypatch,
) -> None:
    paths = RepoPaths.discover(committed_git_repo)
    paths.state_dir.mkdir(parents=True)
    score = paths.score_report
    prompt = paths.prompts_dir / "coding.md"
    improvement = paths.improvement_prds_dir / "runtime.md"
    repository = SimpleNamespace(id=uuid4())
    analysis = SimpleNamespace(id=uuid4(), overall_score=81.5, score_delta=4.0)
    previous = SimpleNamespace(overall_score=77.5)
    calls: list[str] = []

    class FakeRepositoryService:
        def __init__(self, service_paths, _store, _factory) -> None:
            assert service_paths == paths

        def initialize(self):
            calls.append("init")
            return SimpleNamespace(
                initialized=True,
                repository=repository,
                analysis=analysis,
                score_path=score,
                prompts=(SimpleNamespace(role="coding", path=prompt),),
                improvement_prds=(
                    SimpleNamespace(
                        path=improvement,
                        suggested_borg_name="runtime-fix",
                    ),
                ),
            )

        def analyze(self):
            calls.append("analyze")
            return SimpleNamespace(
                repository=repository,
                analysis=analysis,
                previous_analysis=previous,
                score_path=score,
                prompts=(SimpleNamespace(role="coding", path=prompt),),
                improvement_prds=(
                    SimpleNamespace(
                        path=improvement,
                        suggested_borg_name="runtime-fix",
                    ),
                ),
            )

    monkeypatch.setattr(mcp_server, "_paths", lambda *, trusted: paths)
    monkeypatch.setattr(mcp_server, "RepositoryService", FakeRepositoryService)

    initialized = _structured(_call_tool("init"))
    analyzed = _structured(_call_tool("analyze"))

    assert calls == ["init", "analyze"]
    assert initialized["status"] == "initialized"
    assert initialized["artifacts"] == [
        {"kind": "score", "path": ".borg/score.md"},
        {"kind": "coding_prompt", "path": ".borg/prompts/coding.md"},
        {"kind": "improvement_prd", "path": ".borg/prds/improvements/runtime.md"},
    ]
    assert initialized["next_actions"] == [
        {
            "tool": "create",
            "arguments": {
                "name": "runtime-fix",
                "source": ".borg/prds/improvements/runtime.md",
            },
        }
    ]
    assert analyzed["status"] == "completed"
    assert analyzed["data"]["previous_score"] == 77.5
    assert analyzed["data"]["delta"] == 4.0


def test_create_and_plan_approval_are_service_backed_and_typed(
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch,
) -> None:
    paths, borg, _current, publication = _published_runtime(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
    )
    prd_path = paths.tracked_dir / "prds" / "new-borg.md"
    approved_plan = paths.tracked_dir / "plans" / f"{borg.name}.md"
    created_borg = Borg(repository_id=borg.repository_id, name="new-borg")
    create_calls: list[tuple[str, Path | None, bool]] = []

    class FakeCreateService:
        def __init__(
            self,
            repository,
            _store,
            _agent,
            *,
            interactive: bool,
        ) -> None:
            assert repository.id == borg.repository_id
            assert interactive is False

        def create(self, name, source, *, confirmed):
            create_calls.append((name, source, confirmed))
            return SimpleNamespace(
                borg=created_borg,
                confirmed=True,
                questions=(),
                body_md="# New Borg\n",
                prd_path=prd_path,
            )

    approval = SimpleNamespace(plan_digest="sha256:mcp-approved-plan")
    approve_calls: list[str] = []

    def approve(service_paths, name):
        approve_calls.append(name)
        assert service_paths == paths
        return borg, approval, approved_plan, publication

    monkeypatch.chdir(committed_git_repo)
    monkeypatch.setattr(mcp_server, "_paths", lambda *, trusted: paths)
    monkeypatch.setattr(mcp_server, "select_agent", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(mcp_server, "CreateService", FakeCreateService)
    monkeypatch.setattr(mcp_server, "_approve_plan", approve)

    created = _structured(
        _call_tool(
            "create",
            {"name": "new-borg", "source": "source.md", "confirmed": True},
        )
    )
    planned = _structured(
        _call_tool("plan", {"name": borg.name, "action": "approve"})
    )

    assert create_calls == [("new-borg", paths.root / "source.md", True)]
    assert created["status"] == "confirmed"
    assert created["artifacts"] == [
        {"kind": "prd", "path": ".borg/prds/new-borg.md"}
    ]
    assert created["next_actions"] == [
        {"tool": "plan", "arguments": {"name": "new-borg", "action": "start"}}
    ]
    assert approve_calls == [borg.name]
    assert planned["status"] == BorgState.READY_TO_EXECUTE.value
    assert planned["data"]["plan_digest"] == approval.plan_digest
    assert planned["artifacts"][0] == {
        "kind": "approved_plan",
        "path": f".borg/plans/{borg.name}.md",
    }
    assert [action["tool"] for action in planned["next_actions"]] == [
        "task_list",
        "execute",
    ]


def test_plan_start_recovers_questions_injects_answers_and_shows_plan(
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    tech_lead_approval_response,
    monkeypatch,
) -> None:
    repository, paths = planning_cli_repository(committed_git_repo, "mcp-start")
    plan = planning_plan_response(summary="MCP plan is ready.")
    adapter = MockAdapter(name="openai").queue(
        MockResponse(
            payload={
                "decision": "ask_more",
                "questions": [
                    {
                        "id": "q1",
                        "question": "Which platforms are required?",
                        "why": "The answer controls the test matrix.",
                    }
                ],
            }
        )
    )
    monkeypatch.chdir(committed_git_repo)
    monkeypatch.setattr(
        mcp_server,
        "_paths",
        lambda *, trusted: paths,
    )
    monkeypatch.setattr(cli_module, "select_agent", lambda *_args, **_kwargs: adapter)

    waiting = _structured(
        _call_tool("plan", {"name": "mcp-start", "action": "start"})
    )

    assert waiting["status"] == BorgState.ARCHITECT_AWAITING_ANSWERS.value
    assert waiting["data"]["questions"] == [
        {
            "id": "q1",
            "question": "Which platforms are required?",
            "why": "The answer controls the test matrix.",
        }
    ]
    assert waiting["next_actions"] == [
        {
            "tool": "plan",
            "arguments": {"name": "mcp-start", "action": "start"},
        }
    ]

    adapter.queue(MockResponse(payload={"decision": "ready_to_plan"}))
    adapter.queue(MockResponse(payload=plan))
    adapter.queue(MockResponse(payload=tech_lead_approval_response()))
    started = _structured(
        _call_tool(
            "plan",
            {
                "name": "mcp-start",
                "action": "start",
                "answers": ["Linux, macOS, and Windows."],
            },
        )
    )
    shown = _structured(
        _call_tool("plan", {"name": "mcp-start", "action": "show"})
    )

    assert started["status"] == BorgState.PLAN_APPROVAL_PENDING.value
    assert [action["arguments"]["action"] for action in started["next_actions"]] == [
        "show",
        "approve",
    ]
    assert shown["status"] == BorgState.PLAN_APPROVAL_PENDING.value
    assert shown["data"] == {"borg": "mcp-start", "plan": plan}
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "mcp-start")
        assert borg is not None
        questions = store.list_planning_questions(borg.id)
    assert questions[0].answers == [
        {"q_id": "q1", "answer": "Linux, macOS, and Windows."}
    ]


def test_plan_change_validates_note_and_preserves_service_history(
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    tech_lead_approval_response,
    monkeypatch,
) -> None:
    repository, paths = planning_cli_repository(committed_git_repo, "mcp-change")
    original = planning_plan_response(summary="Original MCP plan.")
    revised = planning_plan_response(summary="Revised MCP plan.")
    adapter = MockAdapter(name="openai")
    for payload in (
        {"decision": "ready_to_plan"},
        original,
        tech_lead_approval_response(),
    ):
        adapter.queue(MockResponse(payload=payload))
    monkeypatch.chdir(committed_git_repo)
    monkeypatch.setattr(mcp_server, "_paths", lambda *, trusted: paths)
    monkeypatch.setattr(cli_module, "select_agent", lambda *_args, **_kwargs: adapter)

    started = _structured(
        _call_tool("plan", {"name": "mcp-change", "action": "start"})
    )
    assert started["status"] == BorgState.PLAN_APPROVAL_PENDING.value

    invalid = _call_tool(
        "plan",
        {"name": "mcp-change", "action": "change", "note": "   "},
    )
    assert invalid.isError is True
    assert "plan change note must not be empty" in invalid.content[0].text

    adapter.queue(MockResponse(payload=revised))
    adapter.queue(MockResponse(payload=tech_lead_approval_response()))
    changed = _structured(
        _call_tool(
            "plan",
            {
                "name": "mcp-change",
                "action": "change",
                "note": "  Add staged rollout checks.  ",
            },
        )
    )
    shown = _structured(
        _call_tool("plan", {"name": "mcp-change", "action": "show"})
    )

    assert changed["status"] == BorgState.PLAN_APPROVAL_PENDING.value
    assert shown["data"]["plan"] == revised
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "mcp-change")
        assert borg is not None
        attempts = store.list_planning_attempts(borg.id)
        requests = store.list_plan_change_requests(borg.id)
    assert [item.note for item in requests] == ["Add staged rollout checks."]
    assert [
        item.result["summary"]
        for item in attempts
        if item.phase == "architect_plan" and item.result is not None
    ] == ["Original MCP plan.", "Revised MCP plan."]


def test_plan_approval_automatically_decomposes_without_another_gate(
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    monkeypatch,
) -> None:
    plan = planning_plan_response()
    repository, paths = planning_cli_repository(committed_git_repo, "mcp-plan")
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "mcp-plan")
        assert borg is not None
        attempt = PlanningAttempt(
            borg_id=borg.id,
            phase="architect_plan",
            round=1,
            adapter="mock",
            model="test-model",
        )
        store.append_planning_attempt(attempt)
        store.complete_planning_attempt(
            attempt.id,
            status=PlanningAttemptStatus.COMPLETED,
            result=plan,
            summary="Ready for approval.",
        )
        store.compare_and_set_borg_state(
            borg.id,
            expected_state=borg.state,
            expected_version=borg.state_version,
            new_state=BorgState.PLAN_APPROVAL_PENDING,
        )

    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload=_pm_tasks(plan))
    ).queue(
        MockResponse(
            payload={
                "decision": "approve",
                "summary": "The task is ready.",
                "findings": [],
            }
        )
    )
    monkeypatch.chdir(committed_git_repo)
    monkeypatch.setattr(mcp_server, "_paths", lambda *, trusted: paths)
    monkeypatch.setattr(
        mcp_server,
        "select_agent",
        lambda *_args, **_kwargs: adapter,
    )

    result = _structured(
        _call_tool("plan", {"name": "mcp-plan", "action": "approve"})
    )

    assert result["status"] == BorgState.READY_TO_EXECUTE.value
    assert [artifact["kind"] for artifact in result["artifacts"]] == [
        "approved_plan",
        "task",
    ]
    assert [action["tool"] for action in result["next_actions"]] == [
        "task_list",
        "execute",
    ]
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        stored = store.get_borg_by_name(repository.id, "mcp-plan")
        assert stored is not None
        generations = store.list_task_generations(stored.id)
    assert stored.state is BorgState.READY_TO_EXECUTE
    assert len(generations) == 1
    assert not hasattr(mcp_server, "approve_task")
    assert not hasattr(mcp_server, "decompose")


def test_task_list_matches_runtime_projection_and_execute_uses_host_service(
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch,
) -> None:
    paths, borg, current, publication = _published_runtime(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
    )
    operation_id = uuid4()
    invoked: list[tuple] = []

    def invoke(*args):
        invoked.append(args)
        return HostExecutionResult(
            preflight=HostPreflightPlan(
                repository_root=paths.root,
                commands=(),
                prepare_commands=(),
                materialize_commands=(),
                environment_files=(),
                executables=(),
                required_secret_names=(),
                compose_files=(),
                services=(),
            ),
            scheduler=HostSchedulerResult(
                operation_id=operation_id,
                acquired=True,
                status=ExecutionRunStatus.COMPLETED,
                total=1,
                done=1,
                failed=0,
                blocked=0,
                pending=0,
            ),
        )

    monkeypatch.chdir(committed_git_repo)
    monkeypatch.setattr(mcp_server, "_paths", lambda *, trusted: paths)
    monkeypatch.setattr(cli_module, "_invoke_host_execution", invoke)

    listed = _structured(_call_tool("task_list", {"name": borg.name}))
    estimate = _structured(_call_tool("execute", {"name": borg.name}))
    executed = _structured(
        _call_tool("execute", {"name": borg.name, "auto_execute": True})
    )

    assert listed["generation_id"] == str(current.generation.id)
    assert listed["generation_digest"] == publication.generation.digest
    assert listed["approved_plan_digest"] == "sha256:mcp-approved-plan"
    assert listed["tasks"] == [
        {
            "generation_id": str(current.generation.id),
            "task_id": str(current.task.id),
            "task_ref": "T-MCP-1",
            "stage": "01-foundation",
            "stem": "01-runtime",
            "position": 1,
            "title": "Project runtime task status",
            "complexity": "small",
            "status": "fix",
            "state_reason": "review requested changes",
            "review_round": 2,
            "attempt_count": 2,
            "duration_seconds": 10.0,
            "cost": {
                "api_spend_usd": 0.75,
                "api_spend_unknown": False,
                "subscription_included": True,
            },
        }
    ]
    assert estimate["status"] == "estimate_approval_required"
    assert estimate["next_actions"] == [
        {
            "tool": "execute",
            "arguments": {"name": borg.name, "auto_execute": True},
        }
    ]
    assert len(invoked) == 1
    (
        invoked_paths,
        invoked_store,
        invoked_config,
        repository_id,
        borg_id,
        generation_id,
    ) = invoked[0]
    assert invoked_paths == paths
    assert isinstance(invoked_store, SqliteStore)
    assert invoked_config.repository_id == borg.repository_id
    assert (repository_id, borg_id, generation_id) == (
        borg.repository_id,
        borg.id,
        current.generation.id,
    )
    assert executed["status"] == "completed"
    assert executed["data"]["operation_id"] == str(operation_id)
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        decision = store.get_current_execution_decision(borg.id)
    assert decision is not None
    assert decision.source == "mcp_auto_execute"
    assert decision.decision == "bypassed"


def test_stdio_stdout_contains_only_protocol_json() -> None:
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]

    process = subprocess.Popen(
        [str(Path(sys.executable).with_name("borg")), "mcp"],
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    responses = []
    try:
        process.stdin.write(json.dumps(messages[0]) + "\n")
        process.stdin.flush()
        ready, _, _ = select.select([process.stdout], [], [], 5)
        assert ready, "MCP server did not answer initialize"
        responses.append(json.loads(process.stdout.readline()))

        process.stdin.write(json.dumps(messages[1]) + "\n")
        process.stdin.write(json.dumps(messages[2]) + "\n")
        process.stdin.flush()
        ready, _, _ = select.select([process.stdout], [], [], 5)
        assert ready, "MCP server did not answer tools/list"
        responses.append(json.loads(process.stdout.readline()))
        process.stdin.close()
        returncode = process.wait(timeout=5)
        stderr = process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert returncode == 0, stderr
    assert [response["id"] for response in responses] == [1, 2]
    assert all(response["jsonrpc"] == "2.0" for response in responses)
    assert "Processing request" not in "\n".join(map(json.dumps, responses))
    assert "Processing request" in stderr
