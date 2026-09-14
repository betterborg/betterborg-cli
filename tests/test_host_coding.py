"""Coding-agent contracts for guarded materialized host worktrees."""

from __future__ import annotations

import stat
import subprocess
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from steering_test_support import CancellingAgent, UnpersistedNoteAgent

from betterborg_cli.agent_runtime import (
    AgentArtifact,
    AgentStatus,
    AgentUsage,
    BillingMode,
    CancellationToken,
    MockAdapter,
    MockResponse,
    run_captured,
)
from betterborg_cli.agent_runtime.api_tools import READ_ONLY_API_TOOLS
from betterborg_cli.agent_runtime.structured import (
    StructuredResultError,
    validate_structured_result,
)
from betterborg_cli.host_execution import (
    REVIEW_RESULT_SCHEMA,
    EnvironmentMaterializationError,
    HostCodingConfig,
    HostCodingPhase,
    HostEnvironmentManager,
    HostPreflightPlan,
    HostReviewFixConfig,
    HostReviewFixPhase,
    HostWorktreeManager,
    SafeGit,
    ScheduledTaskContext,
)
from betterborg_cli.host_execution._agent_phase import (
    EXISTING_TEST_MERGE_RULE,
    EXISTING_TEST_REVIEW_RULE,
    EXISTING_TEST_RULE,
    REVIEW_FINDING_RULE,
    AgentAttemptArtifacts,
    VerifiedTaskInputs,
)
from betterborg_cli.host_execution.coding import (
    CODING_RESULT_SCHEMA,
    _render_user_prompt,
)
from betterborg_cli.host_execution.guard import (
    CheckoutCondition,
    PrimaryCheckoutContaminationError,
    PrimaryCheckoutGuard,
    checkout_was_changed,
)
from betterborg_cli.host_execution.merge import _render_merge_prompt
from betterborg_cli.host_execution.review import (
    _render_fix_prompt,
    _render_review_prompt,
    _review_round_summaries,
)
from betterborg_cli.planning import (
    approved_plan_digest,
    render_task_markdown,
    task_markdown_digest,
)
from betterborg_cli.planning.grants import TASK_REVIEW_LOOP
from betterborg_cli.planning.steering import (
    STEERING_NOTE_SCHEMA,
    STEERING_SYSTEM_PROMPT,
)
from betterborg_cli.progress import AgentActivity, AgentActivityKind
from betterborg_cli.repo_paths import RepoPaths, ensure_managed_gitignore
from betterborg_cli.repository_config import BlockedTaskPolicy
from betterborg_cli.store import (
    Borg,
    BorgState,
    ExecutionAttemptStatus,
    ExecutionLedgerFinding,
    FindingStatus,
    PlanApproval,
    Repository,
    RepositoryAnalysis,
    RepositoryPackage,
    ReviewAssessment,
    SqliteStore,
    TaskBatch,
    TaskClaim,
    TaskComplexity,
    TaskDependency,
    TaskGeneration,
    TaskRecord,
    TaskRuntimeStatus,
)
from betterborg_cli.store.models import utcnow


@dataclass(frozen=True)
class CodingFixture:
    repository: Path
    database: Path
    borg: Borg
    generation: TaskGeneration
    task: TaskRecord
    dependency: TaskRecord
    run_id: UUID
    owner_token: str
    claim: TaskClaim

    def context(
        self,
        store: SqliteStore,
        *,
        cancel: CancellationToken | None = None,
        activity: Callable[[AgentActivity], None] | None = None,
    ) -> ScheduledTaskContext:
        return ScheduledTaskContext(
            store=store,
            claim=self.claim,
            owner_token=self.owner_token,
            cancel=cancel or CancellationToken(),
            clock=utcnow,
            activity=activity,
        )


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _task_body(stem: str, *, dependencies: list[str]) -> dict:
    return {
        "stage": "07-host-execution",
        "stem": stem,
        "title": f"Implement {stem}",
        "why": "Host coding needs a durable contract.",
        "scope": [f"Implement {stem}."],
        "implementation_notes": [],
        "acceptance_criteria": [f"{stem} works."],
        "tests": [f"Test {stem}."],
        "dependencies": dependencies,
        "out_of_scope": [],
        "plan_refs": ["P1.deliverable.1"],
        "estimate_complexity": "small",
    }


def _record(
    generation_id: UUID,
    borg: Borg,
    *,
    position: int,
    stem: str,
    dependencies: list[str],
) -> TaskRecord:
    body = _task_body(stem, dependencies=dependencies)
    digest = task_markdown_digest(render_task_markdown(body))
    return TaskRecord(
        generation_id=generation_id,
        borg_id=borg.id,
        task_ref=f"07-host-execution/{stem}",
        stage=body["stage"],
        stem=stem,
        position=position,
        title=body["title"],
        complexity=TaskComplexity.SMALL,
        digest=digest,
        task=body,
        manifest={"task.md": digest},
    )


def _coding_fixture(tmp_path: Path) -> CodingFixture:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    _git(repository_root, "init", "--quiet", "--initial-branch=main")
    _git(repository_root, "config", "user.name", "Betterborg Tests")
    _git(repository_root, "config", "user.email", "tests@betterborg.dev")
    (repository_root / "README.md").write_text("# Fixture\n", encoding="utf-8")
    paths = RepoPaths.discover(repository_root)
    ensure_managed_gitignore(paths)
    _git(repository_root, "add", ".")
    _git(repository_root, "commit", "--quiet", "-m", "initial")

    database = tmp_path / "state.sqlite3"
    repository = Repository(root=repository_root)
    borg = Borg(
        repository_id=repository.id,
        name="coding-fixture",
        state=BorgState.READY_TO_EXECUTE,
    )
    approval = PlanApproval(
        borg_id=borg.id,
        plan_digest="sha256:approved-plan",
        manifest={},
    )
    batch = TaskBatch(
        borg_id=borg.id,
        plan_approval_id=approval.id,
        round=1,
        digest="sha256:batch",
        manifest={},
    )
    generation_id = uuid4()
    dependency = _record(
        generation_id,
        borg,
        position=1,
        stem="08-schedule-host-tasks",
        dependencies=[],
    )
    task = _record(
        generation_id,
        borg,
        position=2,
        stem="09-run-coding-agent",
        dependencies=[dependency.task_ref],
    )
    edge = TaskDependency(
        generation_id=generation_id,
        task_id=task.id,
        depends_on_task_id=dependency.id,
    )
    manifest_tasks = [
        {
            "digest": record.digest,
            "path": (
                f".betterborg/tasks/{borg.name}/{generation_id}/"
                f"{record.stage}/{record.stem}.md"
            ),
            "position": record.position,
            "task_ref": record.task_ref,
        }
        for record in (dependency, task)
    ]
    generation_manifest = {
        "approved_plan_digest": approval.plan_digest,
        "batch_digest": batch.digest,
        "dependencies": [
            {
                "task_ref": task.task_ref,
                "depends_on": dependency.task_ref,
            }
        ],
        "plan_approval_id": str(approval.id),
        "tasks": manifest_tasks,
    }
    generation = TaskGeneration(
        id=generation_id,
        borg_id=borg.id,
        plan_approval_id=approval.id,
        batch_id=batch.id,
        digest=approved_plan_digest(generation_manifest),
        manifest=generation_manifest,
    )
    analysis = RepositoryAnalysis(
        repository_id=repository.id,
        head_sha=_git(repository_root, "rev-parse", "HEAD"),
        summary="A test repository.",
        primary_language="Python",
        is_monorepo=False,
        overall_score=4,
        analysis_json={},
    )
    package = RepositoryPackage(
        repository_id=repository.id,
        analysis_id=analysis.id,
        package_path=".",
        package_name="fixture",
        primary_language="Python",
        rubric={},
        overall_score=4,
    )
    durable_root = paths.tasks_dir / borg.name / str(generation.id)

    with SqliteStore.open(database) as store:
        store.add_repository(repository)
        store.add_borg(borg)
        store.append_analysis(analysis, [package])
        store.append_generated_prompt(
            repository_id=repository.id,
            analysis_id=analysis.id,
            role="coding",
            body_md="You are the generated coding agent.\n",
        )
        store.append_plan_approval(approval)
        store.append_task_batch(batch)
        store.add_task_generation(generation, [dependency, task], [edge])
        for record in (dependency, task):
            path = durable_root / record.stage / f"{record.stem}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render_task_markdown(record.task), encoding="utf-8")
        store._promote_published_task_generation(
            generation.id,
            durable_root=durable_root,
            tasks_root=paths.tasks_dir,
            owned_root=paths.tracked_root,
        )

    if paths.tracked_in_repository:
        _git(repository_root, "add", ".")
        _git(repository_root, "commit", "--quiet", "-m", "publish tasks")
    with SqliteStore.open(database) as store:
        acquisition = store.acquire_execution_run(
            borg.id, generation.id, lease_duration=timedelta(hours=1)
        )
        assert acquisition.owner_token is not None
        specs = HostWorktreeManager(
            repository_root,
            tmp_path / "worktrees",
            source_branch="main",
        ).prepare_current_task_worktrees(
            store,
            run_id=acquisition.run_id,
            owner_token=acquisition.owner_token,
            generation_id=generation.id,
            project_name=borg.name,
        )
        dependency_claim = store.claim_dependency_ready_task(
            acquisition.run_id,
            acquisition.owner_token,
            lease_duration=timedelta(minutes=30),
        )
        assert dependency_claim is not None
        store.transition_task_runtime(
            acquisition.run_id,
            acquisition.owner_token,
            dependency_claim.id,
            dependency_claim.claim_token,
            expected_status=TaskRuntimeStatus.CLAIMED,
            new_status=TaskRuntimeStatus.DONE,
        )
        claim = store.claim_dependency_ready_task(
            acquisition.run_id,
            acquisition.owner_token,
            lease_duration=timedelta(minutes=30),
        )
        assert claim is not None and claim.task_id == task.id
        plan = HostPreflightPlan(
            repository_root=repository_root,
            commands=(),
            prepare_commands=(),
            materialize_commands=(),
            required_secret_names=(),
        )
        HostEnvironmentManager(repository_root).materialize_claimed_task(
            store, plan, claim, acquisition.owner_token
        )
    assert len(specs) == 2
    return CodingFixture(
        repository=repository_root,
        database=database,
        borg=borg,
        generation=generation,
        task=task,
        dependency=dependency,
        run_id=acquisition.run_id,
        owner_token=acquisition.owner_token,
        claim=claim,
    )


def _completed_payload(task: TaskRecord) -> dict:
    return {
        "task_file": f"{task.stage}/{task.stem}.md",
        "status": "completed",
        "summary": "Implemented and committed the task.",
        "changed_files": ["feature.txt"],
        "tests_run": ["test feature"],
        "follow_ups": [],
        "blockers": [],
    }


UNRESOLVED_BLOCKER = "tests/test_timeouts.py::test_write_timeout was already red"
UNRESOLVED_FOLLOW_UP = "Resolve the pre-existing warning, then rerun the suite"


def _unfinished_payload(task: TaskRecord, *, status: str) -> dict:
    """One agent report that did the work and says the task is not finished."""
    return {
        **_completed_payload(task),
        "status": status,
        "summary": "Implemented the feature; the suite has an unrelated failure.",
        "blockers": [UNRESOLVED_BLOCKER],
        "follow_ups": [UNRESOLVED_FOLLOW_UP],
    }


def _committing_response(
    task: TaskRecord,
    *,
    usage: AgentUsage | None = None,
    payload: dict | None = None,
):
    def commit(spec):
        (spec.cwd / "feature.txt").write_text("implemented\n", encoding="utf-8")
        _git(spec.cwd, "add", "feature.txt")
        _git(spec.cwd, "commit", "--quiet", "-m", "add feature")
        transcript = spec.cwd / ".betterborg/state/provider-transcript.txt"
        transcript.write_text("immutable transcript\n", encoding="utf-8")
        return MockResponse(
            payload=payload or _completed_payload(task),
            usage=usage,
            billing_mode=spec.billing_mode,
            artifacts=(AgentArtifact(transcript, kind="transcript"),),
        )

    return MockResponse(dynamic=commit)


def test_coding_phase_ready_worktree_reuses_cancellable_git_binding(
    tmp_path: Path,
    real_process_harness,
) -> None:
    fixture = _coding_fixture(tmp_path)
    cancel = CancellationToken()
    resistant = real_process_harness.resistant_argv("ready-worktree-git")
    observed_tokens: list[CancellationToken | None] = []

    def runner(command, **kwargs):
        if tuple(command)[-3:] == ("rev-parse", "--abbrev-ref", "HEAD"):
            observed_tokens.append(kwargs.get("cancel"))
            return run_captured(resistant, **kwargs)
        return run_captured(command, **kwargs)

    with SqliteStore.open(fixture.database) as store:
        context = fixture.context(store, cancel=cancel)
        git = SafeGit(
            fixture.repository,
            cancel=cancel,
            command_runner=runner,
        )
        phase = HostCodingPhase(
            fixture.repository,
            MockAdapter(),
            config=HostCodingConfig(model="coding-model"),
            cancel=cancel,
            git=git,
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(
                phase._require_ready_worktree,  # noqa: SLF001
                context,
            )
            real_process_harness.wait_for_marker("ready-worktree-git.child.pid")
            cancel.cancel()
            with pytest.raises(KeyboardInterrupt):
                result.result(timeout=5)

    real_process_harness.assert_tree_absent("ready-worktree-git")
    assert observed_tokens == [cancel]


def _finding(
    message: str, *, severity: str = "major", repeats: str | None = None
) -> dict:
    """One declared review finding, by default one that holds the task."""
    return {"severity": severity, "message": message, "repeats": repeats}


def _review_payload(
    task: TaskRecord,
    *,
    status: str,
    findings: Sequence[str | dict] | None = None,
    resolved: Sequence[str] = (),
) -> dict:
    return {
        "task_file": f"{task.stage}/{task.stem}.md",
        "status": status,
        "summary": (
            "Implementation approved."
            if status == "approved"
            else "Implementation needs changes."
        ),
        "issues_file": "",
        "resolved": list(resolved),
        "findings": [
            _finding(item) if isinstance(item, str) else item
            for item in findings or ()
        ],
    }


def _ledger_row(
    task: TaskRecord,
    message: str,
    *,
    severity: str = "major",
    first_raised: int = 1,
) -> ExecutionLedgerFinding:
    """One open ledger row, for a prompt rendered without a store behind it."""
    return ExecutionLedgerFinding(
        task_id=task.id,
        attempt_id=uuid4(),
        first_seen_round=first_raised,
        last_seen_round=first_raised,
        severity=severity,
        message=message,
    )


def _steering_adapter(rounds: int, *, confidence: str = "high") -> MockAdapter:
    """Build a steering agent with a note for every pass that can be steered.

    Its own adapter, so a note never comes off the reviewer's script and the
    reviewer's call count still counts reviews.
    """
    agent = MockAdapter()
    for index in range(rounds):
        agent.queue(
            MockResponse(
                payload={
                    "note": f"Steering note {index + 1}.",
                    "confidence": confidence,
                }
            )
        )
    return agent


def _fixing_response(
    task: TaskRecord,
    *,
    usage: AgentUsage | None = None,
    activities: tuple[AgentActivity, ...] = (),
    payload: dict | None = None,
):
    def commit(spec):
        feature = spec.cwd / "feature.txt"
        feature.write_text(feature.read_text() + "fixed\n", encoding="utf-8")
        _git(spec.cwd, "add", "feature.txt")
        _git(spec.cwd, "commit", "--quiet", "-m", "fix review finding")
        return MockResponse(
            payload=payload or _completed_payload(task),
            usage=usage,
            billing_mode=spec.billing_mode,
            activities=activities,
        )

    return MockResponse(dynamic=commit)


def _prepare_review(
    fixture: CodingFixture,
    store: SqliteStore,
    *,
    usage: AgentUsage | None = None,
    coding_payload: dict | None = None,
    coding_config: HostCodingConfig | None = None,
) -> None:
    coding_prompt = store.get_latest_generated_prompts(
        fixture.borg.repository_id
    )["coding"]
    store.append_generated_prompt(
        repository_id=fixture.borg.repository_id,
        analysis_id=coding_prompt.analysis_id,
        role="review",
        body_md="You are the generated read-only review agent.\n",
    )
    status = HostCodingPhase(
        fixture.repository,
        MockAdapter().queue(
            _committing_response(
                fixture.task, usage=usage, payload=coding_payload
            )
        ),
        config=coding_config or HostCodingConfig(model="coding-model"),
    ).run(fixture.context(store))
    assert status is TaskRuntimeStatus.REVIEW


def test_coding_runs_from_digest_verified_inputs_and_persists_billing(
    tmp_path: Path,
) -> None:
    fixture = _coding_fixture(tmp_path)
    usage = AgentUsage(cost_usd=0.42, tokens_input=120, tokens_output=30)
    adapter = MockAdapter().queue(_committing_response(fixture.task, usage=usage))

    with SqliteStore.open(fixture.database) as store:
        phase = HostCodingPhase(
            fixture.repository,
            adapter,
            config=HostCodingConfig(
                model="test-model", billing_mode=BillingMode.API
            ),
        )
        status = phase.run(fixture.context(store))
        attempts = store.list_agent_attempts(fixture.task.id)
        runtime = store.get_task_runtime(fixture.task.id)

    assert status is TaskRuntimeStatus.REVIEW
    assert runtime is not None and runtime.status is TaskRuntimeStatus.REVIEW
    assert len(adapter.calls) == 1
    call = adapter.calls[0]
    assert call.cwd == Path(runtime.worktree_path)
    assert call.system_prompt == "You are the generated coding agent.\n"
    assert fixture.task.digest in call.user_prompt
    assert fixture.dependency.digest in call.user_prompt
    assert render_task_markdown(fixture.task.task).strip() in call.user_prompt
    assert render_task_markdown(fixture.dependency.task).strip() in call.user_prompt
    assert len(attempts) == 1
    assert attempts[0].status is ExecutionAttemptStatus.COMPLETED
    assert attempts[0].billing_mode is BillingMode.API
    assert attempts[0].usage == usage
    metadata = attempts[0].result["_betterborg"]
    artifact_dir = fixture.repository / metadata["artifact_dir"]
    manifest = artifact_dir / "artifact-manifest.json"
    assert manifest.is_file()
    assert not manifest.stat().st_mode & stat.S_IWUSR
    adapter_artifact = metadata["adapter_artifacts"][0]
    transcript = fixture.repository / adapter_artifact["path"]
    assert transcript.read_text() == "immutable transcript\n"
    assert transcript.parent.name == "adapter-artifacts"
    assert not transcript.stat().st_mode & stat.S_IWUSR
    assert _git(fixture.repository, "status", "--porcelain") == ""


def test_coding_binds_labelled_provider_activity_to_the_task(
    tmp_path: Path,
) -> None:
    fixture = _coding_fixture(tmp_path)
    received: list[AgentActivity] = []
    response = _committing_response(fixture.task)
    assert response.dynamic is not None
    commit = response.dynamic

    def commit_with_activity(spec):  # noqa: ANN001
        generated = commit(spec)
        assert isinstance(generated, MockResponse)
        return MockResponse(
            payload=generated.payload,
            artifacts=generated.artifacts,
            activities=(
                AgentActivity(AgentActivityKind.READING, "task.md"),
            ),
        )

    adapter = MockAdapter().queue(MockResponse(dynamic=commit_with_activity))
    with SqliteStore.open(fixture.database) as store:
        status = HostCodingPhase(
            fixture.repository,
            adapter,
            config=HostCodingConfig(model="test-model"),
        ).run(fixture.context(store, activity=received.append))

    assert status is TaskRuntimeStatus.REVIEW
    assert adapter.calls[0].activity_sink is not None
    assert received == [
        AgentActivity(AgentActivityKind.READING, "coding: task.md")
    ]


def test_coding_blocks_and_preserves_work_when_agent_makes_no_commit(
    tmp_path: Path,
) -> None:
    fixture = _coding_fixture(tmp_path)

    def leave_uncommitted(spec):
        (spec.cwd / "unfinished.txt").write_text("keep me\n", encoding="utf-8")
        return _completed_payload(fixture.task)

    adapter = MockAdapter().queue(MockResponse(dynamic=leave_uncommitted))
    with SqliteStore.open(fixture.database) as store:
        status = HostCodingPhase(
            fixture.repository,
            adapter,
            config=HostCodingConfig(model="test-model"),
        ).run(fixture.context(store))
        runtime = store.get_task_runtime(fixture.task.id)
        attempts = store.list_agent_attempts(fixture.task.id)

    assert status is TaskRuntimeStatus.BLOCKED
    assert runtime is not None and "without producing a commit" in runtime.state_reason
    worktree = Path(runtime.worktree_path)
    assert (worktree / "unfinished.txt").read_text() == "keep me\n"
    assert "?? unfinished.txt" in _git(worktree, "status", "--porcelain")
    assert attempts[0].status is ExecutionAttemptStatus.COMPLETED


def test_digest_drift_blocks_before_invocation(
    tmp_path: Path,
) -> None:
    fixture = _coding_fixture(tmp_path)
    with SqliteStore.open(fixture.database) as store:
        runtime = store.get_task_runtime(fixture.task.id)
        assert runtime is not None and runtime.worktree_path is not None
        task_path = (
            Path(runtime.worktree_path)
            / ".betterborg/tasks"
            / fixture.borg.name
            / str(fixture.generation.id)
            / fixture.task.stage
            / f"{fixture.task.stem}.md"
        )
        task_path.write_text("# drifted\n", encoding="utf-8")
        adapter = MockAdapter().queue(_committing_response(fixture.task))
        status = HostCodingPhase(
            fixture.repository,
            adapter,
            config=HostCodingConfig(model="test-model"),
        ).run(fixture.context(store))
        blocked = store.get_task_runtime(fixture.task.id)

    assert status is TaskRuntimeStatus.BLOCKED
    assert blocked is not None and "digest drifted" in blocked.state_reason
    assert adapter.calls == []


def test_missing_materialization_marker_blocks_before_invocation(
    tmp_path: Path,
) -> None:
    fixture = _coding_fixture(tmp_path)
    with SqliteStore.open(fixture.database) as store:
        runtime = store.get_task_runtime(fixture.task.id)
        assert runtime is not None and runtime.worktree_path is not None
        marker = (
            Path(runtime.worktree_path)
            / ".betterborg/state/environment-materialization"
        )
        marker.unlink()
        adapter = MockAdapter().queue(_committing_response(fixture.task))
        status = HostCodingPhase(
            fixture.repository,
            adapter,
            config=HostCodingConfig(model="test-model"),
        ).run(fixture.context(store))
        blocked = store.get_task_runtime(fixture.task.id)

    assert status is TaskRuntimeStatus.BLOCKED
    assert blocked is not None and "marker is missing" in blocked.state_reason
    assert adapter.calls == []


def test_a_drifted_materialization_marker_blocks_before_invocation(
    tmp_path: Path,
) -> None:
    """The stored attempt and the checkout's marker have to agree.

    A completed attempt outlives the dependencies it installed, so an agent
    is only let into a checkout whose marker still names what the store
    recorded for it.
    """
    fixture = _coding_fixture(tmp_path)
    with SqliteStore.open(fixture.database) as store:
        runtime = store.get_task_runtime(fixture.task.id)
        assert runtime is not None and runtime.worktree_path is not None
        marker = (
            Path(runtime.worktree_path)
            / ".betterborg/state/environment-materialization"
        )
        marker.write_text("sha256:another-preparation\n", encoding="utf-8")
        adapter = MockAdapter().queue(_committing_response(fixture.task))
        status = HostCodingPhase(
            fixture.repository,
            adapter,
            config=HostCodingConfig(model="test-model"),
        ).run(fixture.context(store))
        blocked = store.get_task_runtime(fixture.task.id)

    assert status is TaskRuntimeStatus.BLOCKED
    assert blocked is not None and "marker has drifted" in blocked.state_reason
    assert adapter.calls == []


def _relocated_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, CodingFixture]:
    home = tmp_path / "betterborg-home"
    home.mkdir()
    monkeypatch.setenv("BETTERBORG_HOME", str(home))
    return home, _coding_fixture(tmp_path)


def _status(root: Path) -> str:
    return _git(root, "status", "--short", "--untracked-files=all")


def test_coding_under_a_declared_home_leaves_every_checkout_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home, fixture = _relocated_home(tmp_path, monkeypatch)
    paths = RepoPaths.discover(fixture.repository)

    def commit(spec):
        (spec.cwd / "feature.txt").write_text("implemented\n", encoding="utf-8")
        _git(spec.cwd, "add", "feature.txt")
        _git(spec.cwd, "commit", "--quiet", "-m", "add feature")
        return MockResponse(
            payload=_completed_payload(fixture.task),
            billing_mode=spec.billing_mode,
        )

    adapter = MockAdapter().queue(MockResponse(dynamic=commit))
    with SqliteStore.open(fixture.database) as store:
        runtime = store.get_task_runtime(fixture.task.id)
        assert runtime is not None and runtime.worktree_path is not None
        worktree = Path(runtime.worktree_path)
        status = HostCodingPhase(
            fixture.repository,
            adapter,
            config=HostCodingConfig(model="test-model"),
        ).run(fixture.context(store))

    assert status is TaskRuntimeStatus.REVIEW
    # The task the agent read is published outside every checkout, so its
    # absolute path is the only name it has.
    prompt = adapter.calls[0].user_prompt
    published = (
        paths.tasks_dir
        / fixture.borg.name
        / str(fixture.generation.id)
        / fixture.task.stage
        / f"{fixture.task.stem}.md"
    )
    assert f"Task file: {published.as_posix()}" in prompt
    assert f"Implement {fixture.task.stem}" in prompt
    assert f"Implement {fixture.dependency.stem}" in prompt

    # The marker recording what this checkout materialized lives with the
    # rest of Betterborg's state, keyed by the checkout it speaks for.
    markers = list((home / "state/environment-markers").iterdir())
    assert len(markers) == 1 and markers[0].is_file()
    assert not (worktree / ".betterborg").exists()
    assert not (fixture.repository / ".betterborg").exists()
    assert not paths.gitignore.exists()
    assert _status(fixture.repository) == ""
    assert _status(worktree) == ""


def test_missing_relocated_materialization_marker_blocks_before_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home, fixture = _relocated_home(tmp_path, monkeypatch)
    adapter = MockAdapter().queue(_committing_response(fixture.task))

    with SqliteStore.open(fixture.database) as store:
        markers = list((home / "state/environment-markers").iterdir())
        assert len(markers) == 1
        markers[0].unlink()
        status = HostCodingPhase(
            fixture.repository,
            adapter,
            config=HostCodingConfig(model="test-model"),
        ).run(fixture.context(store))
        blocked = store.get_task_runtime(fixture.task.id)

    assert status is TaskRuntimeStatus.BLOCKED
    assert blocked is not None and "marker is missing" in blocked.state_reason
    assert adapter.calls == []


def test_a_marker_directory_outside_the_tracked_one_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The marker still has to sit somewhere Betterborg owns.

    Outside every checkout, no ignore rule of the repository's governs it, so
    the ignore guard no longer applies. What remains is that the directory
    holding it must not lead somewhere else, which is how it would end up
    back in the working tree the relocation exists to keep empty.
    """
    home = tmp_path / "betterborg-home"
    (home / "state").mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (home / "state/environment-markers").symlink_to(
        elsewhere, target_is_directory=True
    )
    monkeypatch.setenv("BETTERBORG_HOME", str(home))

    with pytest.raises(
        EnvironmentMaterializationError, match="escapes the tracked directory"
    ):
        _coding_fixture(tmp_path)

    assert list(elsewhere.iterdir()) == []


def test_a_fresh_worktree_does_not_inherit_the_marker_it_replaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home, fixture = _relocated_home(tmp_path, monkeypatch)
    markers = home / "state/environment-markers"

    with SqliteStore.open(fixture.database) as store:
        runtime = store.get_task_runtime(fixture.task.id)
        assert runtime is not None and runtime.worktree_path is not None
        worktree = Path(runtime.worktree_path)
        assert len(list(markers.iterdir())) == 1

        # A checkout-local marker leaves with its checkout; one kept outside
        # every checkout has to be discarded deliberately.
        _git(fixture.repository, "worktree", "remove", "--force", str(worktree))
        HostWorktreeManager(
            fixture.repository,
            tmp_path / "worktrees",
            source_branch="main",
        ).prepare_current_task_worktrees(
            store,
            run_id=fixture.run_id,
            owner_token=fixture.owner_token,
            generation_id=fixture.generation.id,
            project_name=fixture.borg.name,
        )

        assert list(markers.iterdir()) == []
        adapter = MockAdapter().queue(_committing_response(fixture.task))
        status = HostCodingPhase(
            fixture.repository,
            adapter,
            config=HostCodingConfig(model="test-model"),
        ).run(fixture.context(store))
        blocked = store.get_task_runtime(fixture.task.id)

    assert status is TaskRuntimeStatus.BLOCKED
    assert blocked is not None and "marker is missing" in blocked.state_reason
    assert adapter.calls == []


def test_interrupted_coding_attempt_is_immutable_and_resumable(
    tmp_path: Path,
) -> None:
    fixture = _coding_fixture(tmp_path)
    cancel = CancellationToken()
    cancel.cancel()
    adapter = MockAdapter().queue(
        MockResponse(payload=_completed_payload(fixture.task))
    )

    with SqliteStore.open(fixture.database) as store:
        status = HostCodingPhase(
            fixture.repository,
            adapter,
            config=HostCodingConfig(model="test-model"),
        ).run(fixture.context(store, cancel=cancel))
        attempt = store.list_agent_attempts(fixture.task.id)[0]
        runtime = store.get_task_runtime(fixture.task.id)

    assert status is TaskRuntimeStatus.CODING
    assert runtime is not None and runtime.status is TaskRuntimeStatus.CODING
    assert attempt.status is ExecutionAttemptStatus.CANCELLED
    assert attempt.finished_at is not None
    assert attempt.result["_betterborg"]["outcome_status"] == "coding"
    assert "interrupted" in attempt.result["_betterborg"]["outcome_reason"]


def test_completed_attempt_resumes_transition_without_agent_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _coding_fixture(tmp_path)
    first_adapter = MockAdapter().queue(_committing_response(fixture.task))

    with SqliteStore.open(fixture.database) as store:
        transition = store.transition_task_runtime

        def crash_before_review(*args, **kwargs):
            if kwargs.get("new_status") is TaskRuntimeStatus.REVIEW:
                raise RuntimeError("simulated restart after durable attempt")
            return transition(*args, **kwargs)

        monkeypatch.setattr(store, "transition_task_runtime", crash_before_review)
        with pytest.raises(RuntimeError, match="simulated restart"):
            HostCodingPhase(
                fixture.repository,
                first_adapter,
                config=HostCodingConfig(model="test-model"),
            ).run(fixture.context(store))
        monkeypatch.setattr(store, "transition_task_runtime", transition)
        interrupted = store.get_task_runtime(fixture.task.id)
        assert interrupted is not None
        assert interrupted.status is TaskRuntimeStatus.CODING

        replay_adapter = MockAdapter()
        status = HostCodingPhase(
            fixture.repository,
            replay_adapter,
            config=HostCodingConfig(model="test-model"),
        ).run(fixture.context(store))

    assert status is TaskRuntimeStatus.REVIEW
    assert replay_adapter.calls == []
    with SqliteStore.open(fixture.database) as reopened:
        assert len(reopened.list_agent_attempts(fixture.task.id)) == 1


def test_primary_checkout_guard_blocks_coding_without_discarding_state(
    tmp_path: Path,
) -> None:
    fixture = _coding_fixture(tmp_path)
    (fixture.repository / "README.md").write_text(
        "# operator work\n", encoding="utf-8"
    )
    adapter = MockAdapter().queue(_committing_response(fixture.task))

    with SqliteStore.open(fixture.database) as store:
        status = HostCodingPhase(
            fixture.repository,
            adapter,
            config=HostCodingConfig(model="test-model"),
        ).run(fixture.context(store))
        runtime = store.get_task_runtime(fixture.task.id)

    assert status is TaskRuntimeStatus.BLOCKED
    assert adapter.calls == []
    assert runtime is not None and "primary checkout" in runtime.state_reason
    assert (fixture.repository / "README.md").read_text() == "# operator work\n"


def test_review_approval_persists_immutable_artifacts_and_declared_base(
    tmp_path: Path,
) -> None:
    fixture = _coding_fixture(tmp_path)
    review_usage = AgentUsage(tokens_input=80, tokens_output=10)
    review = MockAdapter().queue(
        MockResponse(
            payload=_review_payload(fixture.task, status="approved"),
            usage=review_usage,
            billing_mode=BillingMode.SUBSCRIPTION,
        )
    )

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        before = store.get_task_runtime(fixture.task.id)
        assert before is not None
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            config=HostReviewFixConfig(
                review_model="review-model",
                review_billing_mode=BillingMode.SUBSCRIPTION,
            ),
        ).run(fixture.context(store))
        runtime = store.get_task_runtime(fixture.task.id)
        attempts = store.list_agent_attempts(fixture.task.id)

    assert status is TaskRuntimeStatus.MERGING
    assert runtime is not None and runtime.status is TaskRuntimeStatus.MERGING
    assert runtime.branch == before.branch
    assert runtime.worktree_path == before.worktree_path
    assert runtime.branch is not None
    assert runtime.branch.rsplit("-", 1)[-1].isalnum()
    assert [attempt.phase for attempt in attempts] == ["coding", "review"]
    review_attempt = attempts[-1]
    assert review_attempt.review_round == 0
    assert review_attempt.billing_mode is BillingMode.SUBSCRIPTION
    assert review_attempt.usage == review_usage
    metadata = review_attempt.result["_betterborg"]
    coding_metadata = attempts[0].result["_betterborg"]
    assert metadata["base_commit"] == coding_metadata["base_commit"]
    assert metadata["commit_sha"] == coding_metadata["commit_sha"]
    artifact_dir = fixture.repository / metadata["artifact_dir"]
    assert not (artifact_dir / "artifact-manifest.json").stat().st_mode & stat.S_IWUSR
    assert "Declared base commit" in review.calls[0].user_prompt
    assert _git(fixture.repository, "status", "--porcelain") == ""


def test_rejection_increments_round_before_fix_and_projects_mixed_billing(
    tmp_path: Path,
) -> None:
    fixture = _coding_fixture(tmp_path)
    finding = "feature.txt must include the reviewed fix"
    received: list[AgentActivity] = []
    review = (
        MockAdapter()
        .queue(
            MockResponse(
                payload=_review_payload(
                    fixture.task,
                    status="issues_found",
                    findings=[finding],
                ),
                usage=AgentUsage(tokens_input=50),
                billing_mode=BillingMode.SUBSCRIPTION,
                activities=(
                    AgentActivity(AgentActivityKind.READING, "feature.txt"),
                ),
            )
        )
        .queue(
            MockResponse(
                payload=_review_payload(fixture.task, status="approved"),
                usage=AgentUsage(tokens_input=40),
                billing_mode=BillingMode.SUBSCRIPTION,
            )
        )
    )
    fix = MockAdapter().queue(
        _fixing_response(
            fixture.task,
            usage=AgentUsage(cost_usd=0.25, tokens_input=100, tokens_output=20),
            activities=(
                AgentActivity(AgentActivityKind.WRITING, "feature.txt"),
            ),
        )
    )

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(
            fixture, store, usage=AgentUsage(cost_usd=0.50, tokens_input=120)
        )
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            config=HostReviewFixConfig(
                review_model="review-model",
                fix_model="fix-model",
                review_billing_mode=BillingMode.SUBSCRIPTION,
                fix_billing_mode=BillingMode.API,
            ),
        ).run(fixture.context(store, activity=received.append))
        runtime = store.get_task_runtime(fixture.task.id)
        attempts = store.list_agent_attempts(fixture.task.id)
        projection = store.list_task_runtime(fixture.borg.id)

    assert status is TaskRuntimeStatus.MERGING
    assert runtime is not None and runtime.review_round == 1
    assert [(attempt.phase, attempt.review_round) for attempt in attempts] == [
        ("coding", 0),
        ("review", 0),
        ("fix", 1),
        ("review", 1),
    ]
    assert attempts[1].result["findings"] == [_finding(finding)]
    assert review.calls[0].activity_sink is not None
    assert fix.calls[0].activity_sink is not None
    assert received == [
        AgentActivity(AgentActivityKind.READING, "review: feature.txt"),
        AgentActivity(AgentActivityKind.WRITING, "fix: feature.txt"),
    ]
    assert finding in fix.calls[0].user_prompt
    assert _git(Path(runtime.worktree_path), "log", "-1", "--pretty=%s") == (
        "fix review finding"
    )
    task_row = next(row for row in projection if row.task_id == fixture.task.id)
    assert task_row.attempt_count == 4
    assert task_row.cost.api_spend_usd == pytest.approx(0.75)
    assert task_row.cost.api_spend_unknown is False
    assert task_row.cost.subscription_included is True


def test_a_task_with_no_grant_budget_blocks_after_persisting_last_findings(
    tmp_path: Path,
) -> None:
    """A budget of nothing asks for exactly the passes and the block they reach.

    The reason it blocks with names what the passes showed rather than the
    number configured, because the number is no longer what ended them.
    """
    fixture = _coding_fixture(tmp_path)
    review = (
        MockAdapter()
        .queue(
            MockResponse(
                payload=_review_payload(
                    fixture.task,
                    status="issues_found",
                    findings=["first finding"],
                )
            )
        )
        .queue(
            MockResponse(
                payload=_review_payload(
                    fixture.task,
                    status="issues_found",
                    findings=["still failing after the fix"],
                )
            )
        )
    )
    fix = MockAdapter().queue(_fixing_response(fixture.task))

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            config=HostReviewFixConfig(
                review_model="review-model",
                review_passes=2,
                grant_budget=0,
            ),
        ).run(fixture.context(store))
        runtime = store.get_task_runtime(fixture.task.id)
        attempts = store.list_agent_attempts(fixture.task.id)

    assert status is TaskRuntimeStatus.BLOCKED
    assert runtime is not None
    assert runtime.review_round == 2
    assert runtime.resume_phase == "review"
    assert runtime.state_reason == (
        "review ended without approval. The loop took no rounds past its minimum "
        "of 2, and its last round was not converging."
    )
    assert [attempt.phase for attempt in attempts] == [
        "coding",
        "review",
        "fix",
        "review",
    ]
    assert attempts[-1].result["findings"] == [
        _finding("still failing after the fix")
    ]
    assert len(fix.calls) == 1


_BROAD_PATTERN = "the media-type pattern is too broad"
_UNBOUNDED_TIMEOUT = "the timeout is unbounded"
_UNNAMED_FIXTURE = "name the fixture the suite already has"


def _ledger_row_for(
    rows: Sequence[ExecutionLedgerFinding], message: str
) -> ExecutionLedgerFinding:
    return next(row for row in rows if row.message == message)


def test_review_findings_reach_the_ledger_with_the_severity_they_declared(
    tmp_path: Path,
) -> None:
    """Severity is a field of the finding rather than a prefix nobody reads.

    Reviewers write "[blocker]" at the front of their findings and no code ever
    read it, which leaves the one signal that tells a stuck loop from a
    productive one to a convention.
    """
    fixture = _coding_fixture(tmp_path)
    review = MockAdapter().queue(
        MockResponse(
            payload=_review_payload(
                fixture.task,
                status="issues_found",
                findings=[
                    _finding(_BROAD_PATTERN, severity="blocker"),
                    _finding(_UNNAMED_FIXTURE, severity="minor"),
                ],
            )
        )
    )

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=1, grant_budget=0
            ),
        ).run(fixture.context(store))
        rows = store.list_execution_ledger_findings(fixture.task.id)
        attempts = store.list_agent_attempts(fixture.task.id)

    assert status is TaskRuntimeStatus.BLOCKED
    assert {(row.severity, row.message) for row in rows} == {
        ("blocker", _BROAD_PATTERN),
        ("minor", _UNNAMED_FIXTURE),
    }
    assert {row.status for row in rows} == {FindingStatus.OPEN}
    assert {row.first_seen_round for row in rows} == {1}
    # Keyed to the attempt that produced it, and written with it, so the round
    # that ran out of passes still left the rows a later grant can read.
    assert {row.attempt_id for row in rows} == {attempts[-1].id}


def test_a_fix_answers_every_open_finding_and_not_just_the_latest_round(
    tmp_path: Path,
) -> None:
    """Silence is not agreement, so a carried objection stays in front of the fixer.

    A fixer shown only the newest round's findings answers less than the ledger
    holds against it: an objection raised in round one and not repeated in
    round two never reaches the one agent that could close it, while its row
    stays open against the task.
    """
    fixture = _coding_fixture(tmp_path)
    review = (
        MockAdapter()
        .queue(
            MockResponse(
                payload=_review_payload(
                    fixture.task,
                    status="issues_found",
                    findings=[_BROAD_PATTERN],
                )
            )
        )
        .queue(
            MockResponse(
                payload=_review_payload(
                    fixture.task,
                    status="issues_found",
                    findings=[_UNBOUNDED_TIMEOUT],
                )
            )
        )
        .queue(MockResponse(payload=_review_payload(fixture.task, status="approved")))
    )
    fix = (
        MockAdapter()
        .queue(_fixing_response(fixture.task))
        .queue(_fixing_response(fixture.task))
    )

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            config=HostReviewFixConfig(
                review_model="review-model",
                fix_model="fix-model",
                review_passes=3,
            ),
        ).run(fixture.context(store))
        rows = store.list_execution_ledger_findings(fixture.task.id)

    assert status is TaskRuntimeStatus.MERGING
    second_fix = fix.calls[1].user_prompt
    assert _BROAD_PATTERN in second_fix
    assert _UNBOUNDED_TIMEOUT in second_fix
    assert "Fix round: 2" in second_fix
    # The approval closed both, including the one no round ever answered.
    assert {row.status for row in rows} == {FindingStatus.RESOLVED}


def test_a_second_review_closes_one_objection_and_raises_another_again(
    tmp_path: Path,
) -> None:
    """The reviewer is given the open ledger and answers for every row on it.

    Across the five rounds of one task the sweep lost, every round named a
    fresh line of one file and none mentioned an earlier round's finding. A
    reviewer that never sees round one's findings can neither close one nor say
    it is back.
    """
    fixture = _coding_fixture(tmp_path)

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)

        def second_review(spec):
            rows = store.list_execution_ledger_findings(fixture.task.id)
            return _review_payload(
                fixture.task,
                status="issues_found",
                resolved=[str(_ledger_row_for(rows, _BROAD_PATTERN).id)],
                findings=[
                    _finding(
                        "the timeout is still unbounded",
                        severity="blocker",
                        repeats=str(
                            _ledger_row_for(rows, _UNBOUNDED_TIMEOUT).id
                        ),
                    )
                ],
            )

        review = (
            MockAdapter()
            .queue(
                MockResponse(
                    payload=_review_payload(
                        fixture.task,
                        status="issues_found",
                        findings=[
                            _finding(_BROAD_PATTERN),
                            _finding(_UNBOUNDED_TIMEOUT, severity="blocker"),
                        ],
                    )
                )
            )
            .queue(MockResponse(dynamic=second_review))
        )
        fix = MockAdapter().queue(_fixing_response(fixture.task))
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=2, grant_budget=0
            ),
        ).run(fixture.context(store))
        rows = store.list_execution_ledger_findings(fixture.task.id)

    assert status is TaskRuntimeStatus.BLOCKED
    closed = _ledger_row_for(rows, _BROAD_PATTERN)
    assert closed.status is FindingStatus.RESOLVED
    assert closed.last_seen_round == 2
    # One objection is one row for as long as the loop argues about it, so the
    # repeat moved the row it named rather than adding a second.
    standing = _ledger_row_for(rows, _UNBOUNDED_TIMEOUT)
    assert len(rows) == 2
    assert standing.status is FindingStatus.REGRESSED
    assert (standing.first_seen_round, standing.last_seen_round) == (1, 2)
    assert standing.severity == "blocker"
    # The ids the second review named came out of the prompt it was handed.
    assert "Open findings this commit has to answer" not in (
        review.calls[0].user_prompt
    )
    second_prompt = review.calls[1].user_prompt
    assert "Review round: 2" in second_prompt
    assert f"- {closed.id} (major, first raised in round 1)" in second_prompt
    assert f"- {standing.id} (blocker, first raised in round 1)" in second_prompt


@pytest.mark.parametrize(
    ("mutate", "missing"),
    [
        pytest.param(
            lambda payload: payload.pop("resolved"), "resolved", id="resolved"
        ),
        pytest.param(
            lambda payload: payload["findings"][0].pop("repeats"),
            "repeats",
            id="repeats",
        ),
    ],
)
def test_a_review_omitting_a_ledger_declaration_fails_its_schema(
    mutate, missing: str, tmp_path: Path
) -> None:
    """Requiring both is what forces a reviewer to answer rather than omit.

    Read as absent, a missing `repeats` says every objection is new and a
    missing `resolved` says the round closed nothing — the two readings that
    leave a ledger which never drains.
    """
    fixture = _coding_fixture(tmp_path)
    payload = _review_payload(
        fixture.task,
        status="issues_found",
        findings=[_finding(_BROAD_PATTERN)],
    )
    validate_structured_result(payload, REVIEW_RESULT_SCHEMA)

    mutate(payload)
    with pytest.raises(StructuredResultError, match=missing):
        validate_structured_result(payload, REVIEW_RESULT_SCHEMA)


_MISSING_ROLLBACK = "the rollback path is untested"


def test_neither_prompt_carries_an_objection_a_round_already_closed(
    tmp_path: Path,
) -> None:
    """The ledger is the row set, so the status filter is the whole narrowing.

    Handed a row its own reviewer closed, the fixer is asked to fix something
    that is already fixed — and the round after it is told nobody has closed an
    objection its predecessor said it closed.
    """
    fixture = _coding_fixture(tmp_path)

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)

        def closing_review(spec):
            rows = store.list_execution_ledger_findings(fixture.task.id)
            return _review_payload(
                fixture.task,
                status="issues_found",
                resolved=[str(_ledger_row_for(rows, _BROAD_PATTERN).id)],
                findings=[_finding(_MISSING_ROLLBACK)],
            )

        review = (
            MockAdapter()
            .queue(
                MockResponse(
                    payload=_review_payload(
                        fixture.task,
                        status="issues_found",
                        findings=[
                            _finding(_BROAD_PATTERN),
                            _finding(_UNBOUNDED_TIMEOUT, severity="blocker"),
                        ],
                    )
                )
            )
            .queue(MockResponse(dynamic=closing_review))
            .queue(
                MockResponse(
                    payload=_review_payload(
                        fixture.task,
                        status="issues_found",
                        findings=[_finding(_UNNAMED_FIXTURE)],
                    )
                )
            )
        )
        fix = MockAdapter()
        for _ in range(2):
            fix.queue(_fixing_response(fixture.task))
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=3, grant_budget=0
            ),
        ).run(fixture.context(store))
        rows = store.list_execution_ledger_findings(fixture.task.id)

    assert status is TaskRuntimeStatus.BLOCKED
    closed = _ledger_row_for(rows, _BROAD_PATTERN)
    assert closed.status is FindingStatus.RESOLVED

    # The fix the closing round asked for, and the round that judged it, were
    # both handed what still stands and not what that round closed.
    second_fix = fix.calls[1].user_prompt
    third_review = review.calls[2].user_prompt
    for prompt in (second_fix, third_review):
        assert str(closed.id) not in prompt
        assert _BROAD_PATTERN not in prompt
        assert _UNBOUNDED_TIMEOUT in prompt
        assert _MISSING_ROLLBACK in prompt


def test_an_approval_whose_every_fault_is_minor_merges(tmp_path: Path) -> None:
    """A minor finding does not hold a task, as it does not hold a batch.

    A reviewer that notices something small has to be able to say so without
    holding the work, or it says nothing and the objection is lost.
    """
    fixture = _coding_fixture(tmp_path)
    review = MockAdapter().queue(
        MockResponse(
            payload=_review_payload(
                fixture.task,
                status="approved",
                findings=[_finding(_UNNAMED_FIXTURE, severity="minor")],
            )
        )
    )

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            config=HostReviewFixConfig(review_model="review-model"),
        ).run(fixture.context(store))
        rows = store.list_execution_ledger_findings(fixture.task.id)

    assert status is TaskRuntimeStatus.MERGING
    assert [(row.severity, row.status) for row in rows] == [
        ("minor", FindingStatus.RESOLVED)
    ]


def test_a_review_reporting_only_minor_issues_still_spends_a_fix_pass(
    tmp_path: Path,
) -> None:
    """What the reviewer decides is left to the reviewer.

    A minor finding does not hold a task the reviewer approved. It does not
    follow that a reviewer asking for changes over one is overruled by its own
    severity.
    """
    fixture = _coding_fixture(tmp_path)
    review = (
        MockAdapter()
        .queue(
            MockResponse(
                payload=_review_payload(
                    fixture.task,
                    status="issues_found",
                    findings=[_finding(_UNNAMED_FIXTURE, severity="minor")],
                )
            )
        )
        .queue(MockResponse(payload=_review_payload(fixture.task, status="approved")))
    )
    fix = MockAdapter().queue(_fixing_response(fixture.task))

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=2
            ),
        ).run(fixture.context(store))

    assert status is TaskRuntimeStatus.MERGING
    assert len(fix.calls) == 1
    assert _UNNAMED_FIXTURE in fix.calls[0].user_prompt


@pytest.mark.parametrize("severity", ["blocker", "major"])
def test_an_approval_carrying_a_holding_finding_still_blocks(
    tmp_path: Path, severity: str
) -> None:
    """A reviewer contradicting itself in one response is caught as it was.

    The check reads the approving round's own findings, and its job is to catch
    that contradiction rather than to make approval an enumeration exercise.
    Minor is the only severity that does not hold the task, so it is the only
    one an approval may carry.
    """
    fixture = _coding_fixture(tmp_path)
    review = MockAdapter().queue(
        MockResponse(
            payload=_review_payload(
                fixture.task,
                status="approved",
                findings=[_finding(_BROAD_PATTERN, severity=severity)],
            )
        )
    )

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            config=HostReviewFixConfig(review_model="review-model"),
        ).run(fixture.context(store))
        runtime = store.get_task_runtime(fixture.task.id)
        rows = store.list_execution_ledger_findings(fixture.task.id)

    assert status is TaskRuntimeStatus.BLOCKED
    assert runtime is not None
    assert runtime.state_reason == (
        "review approval included blocker or major findings"
    )
    assert rows == []


def test_an_unplaceable_resolved_id_leaves_its_row_open_and_a_repeat_regresses(
    tmp_path: Path,
) -> None:
    """The two declarations fall different ways, which is what makes them safe.

    The history a reviewer also reads carries an id for every restatement of an
    objection while the ledger keys each one by its first, so an id the ledger
    cannot place is a mistake a reviewer can make. Left to cost a grant it is
    harmless; treating an unplaceable repeat as fresh discovery would grant a
    loop holding a blocker it has already failed to close.
    """
    fixture = _coding_fixture(tmp_path)
    review = (
        MockAdapter()
        .queue(
            MockResponse(
                payload=_review_payload(
                    fixture.task,
                    status="issues_found",
                    findings=[_BROAD_PATTERN],
                )
            )
        )
        .queue(
            MockResponse(
                payload=_review_payload(
                    fixture.task,
                    status="issues_found",
                    resolved=[str(uuid4())],
                    findings=[
                        _finding(
                            _UNBOUNDED_TIMEOUT,
                            severity="blocker",
                            repeats=str(uuid4()),
                        )
                    ],
                )
            )
        )
    )
    fix = MockAdapter().queue(_fixing_response(fixture.task))

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=2, grant_budget=0
            ),
        ).run(fixture.context(store))
        rows = store.list_execution_ledger_findings(fixture.task.id)

    assert status is TaskRuntimeStatus.BLOCKED
    unclosed = _ledger_row_for(rows, _BROAD_PATTERN)
    assert unclosed.status is FindingStatus.OPEN
    assert (unclosed.first_seen_round, unclosed.last_seen_round) == (1, 2)
    regressed = _ledger_row_for(rows, _UNBOUNDED_TIMEOUT)
    assert regressed.status is FindingStatus.REGRESSED
    assert regressed.first_seen_round == 2


def _assessments(
    store: SqliteStore, fixture: CodingFixture
) -> list[ReviewAssessment]:
    """Return what this task's review passes recorded about themselves."""
    return store.list_review_assessments(
        fixture.borg.id, loop=TASK_REVIEW_LOOP, task_id=fixture.task.id
    )


def _repeating_review(store: SqliteStore, fixture: CodingFixture):
    """A reviewer that raises the one blocker it already raised, by its id."""

    def review(spec):
        rows = store.list_execution_ledger_findings(fixture.task.id)
        return _review_payload(
            fixture.task,
            status="issues_found",
            findings=[
                _finding(
                    _UNBOUNDED_TIMEOUT,
                    severity="blocker",
                    repeats=str(_ledger_row_for(rows, _UNBOUNDED_TIMEOUT).id),
                )
            ],
        )

    return review


def test_a_draining_review_ledger_runs_past_the_configured_passes(
    tmp_path: Path,
) -> None:
    """The configured passes are a minimum, and a closing loop keeps going.

    Three trials of a ten-task sweep lost a task to its pass limit mid-argument.
    A pass that leaves fewer objections open than the pass before it costs the
    task nothing, so a review that is getting somewhere argues on to agreement.
    """
    fixture = _coding_fixture(tmp_path)

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)

        def closing_review(spec):
            rows = store.list_execution_ledger_findings(fixture.task.id)
            return _review_payload(
                fixture.task,
                status="issues_found",
                resolved=[
                    str(_ledger_row_for(rows, _BROAD_PATTERN).id),
                    str(_ledger_row_for(rows, _UNBOUNDED_TIMEOUT).id),
                ],
                findings=[_finding(_UNNAMED_FIXTURE)],
            )

        review = (
            MockAdapter()
            .queue(
                MockResponse(
                    payload=_review_payload(
                        fixture.task,
                        status="issues_found",
                        findings=[_BROAD_PATTERN, _UNBOUNDED_TIMEOUT],
                    )
                )
            )
            .queue(MockResponse(dynamic=closing_review))
            .queue(
                MockResponse(
                    payload=_review_payload(fixture.task, status="approved")
                )
            )
        )
        fix = (
            MockAdapter()
            .queue(_fixing_response(fixture.task))
            .queue(_fixing_response(fixture.task))
        )
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=_steering_adapter(2),
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=1
            ),
        ).run(fixture.context(store))
        rows = store.list_execution_ledger_findings(fixture.task.id)
        recorded = _assessments(store, fixture)

    assert status is TaskRuntimeStatus.MERGING
    assert len(review.calls) == 3
    # The first pass is the minimum and is neither charged nor refunded. Both
    # passes past it left fewer objections open than their predecessor, so both
    # were refunded and the budget paid for none of them.
    assert [
        (item.round, item.open_findings, item.refunded) for item in recorded
    ] == [(1, 2, None), (2, 1, True), (3, 0, True)]
    assert {item.minimum for item in recorded} == {1}
    # Each round keeps the drain it was judged from, so the evidence for a
    # verdict outlives the round that reached it.
    assert [item.evidence["drain"][-1] for item in recorded] == [
        {"new": 2, "open_after": 2, "resolved": 0, "round": 1},
        {"new": 1, "open_after": 1, "resolved": 2, "round": 2},
        {"new": 0, "open_after": 0, "resolved": 1, "round": 3},
    ]
    assert {row.status for row in rows} == {FindingStatus.RESOLVED}


def test_a_review_repeating_one_blocker_spends_its_budget_and_blocks(
    tmp_path: Path,
) -> None:
    """A pass that closes nothing is the pass the budget exists to bound.

    The recorded reason names what the passes showed rather than the number
    configured: the number is no longer what ended them, and an operator
    reading a blocked task wants the evidence the loop stopped on.
    """
    fixture = _coding_fixture(tmp_path)

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        repeat = _repeating_review(store, fixture)
        review = (
            MockAdapter()
            .queue(
                MockResponse(
                    payload=_review_payload(
                        fixture.task,
                        status="issues_found",
                        findings=[
                            _finding(_UNBOUNDED_TIMEOUT, severity="blocker")
                        ],
                    )
                )
            )
            .queue(MockResponse(dynamic=repeat))
            .queue(MockResponse(dynamic=repeat))
        )
        fix = (
            MockAdapter()
            .queue(_fixing_response(fixture.task))
            .queue(_fixing_response(fixture.task))
        )
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=_steering_adapter(2),
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=1, grant_budget=2
            ),
        ).run(fixture.context(store))
        runtime = store.get_task_runtime(fixture.task.id)
        rows = store.list_execution_ledger_findings(fixture.task.id)
        recorded = _assessments(store, fixture)

    assert status is TaskRuntimeStatus.BLOCKED
    assert [(item.round, item.refunded) for item in recorded] == [
        (1, None),
        (2, False),
        (3, False),
    ]
    assert runtime is not None
    assert runtime.state_reason == (
        "review ended without approval. The loop took 2 granted rounds past its "
        "minimum of 1, 2 of them closing nothing, and its last round was not "
        "converging."
    )
    # The terminal state is the one it always was: the reviewed commit is still
    # on the task's retained branch and the objection stands.
    assert runtime.resume_phase == "review" and runtime.branch
    assert [row.status for row in rows] == [FindingStatus.REGRESSED]


def test_a_budget_of_zero_leaves_a_draining_task_at_its_configured_passes(
    tmp_path: Path,
) -> None:
    """Zero asks for exactly today's passes and today's block.

    A draining loop is the one a budget buys passes for, so a repository that
    configures none has to see its configured pass end even that task.
    """
    fixture = _coding_fixture(tmp_path)

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)

        def closing_review(spec):
            rows = store.list_execution_ledger_findings(fixture.task.id)
            return _review_payload(
                fixture.task,
                status="issues_found",
                resolved=[
                    str(_ledger_row_for(rows, _BROAD_PATTERN).id),
                    str(_ledger_row_for(rows, _UNBOUNDED_TIMEOUT).id),
                ],
                findings=[_finding(_UNNAMED_FIXTURE)],
            )

        review = (
            MockAdapter()
            .queue(
                MockResponse(
                    payload=_review_payload(
                        fixture.task,
                        status="issues_found",
                        findings=[_BROAD_PATTERN, _UNBOUNDED_TIMEOUT],
                    )
                )
            )
            .queue(MockResponse(dynamic=closing_review))
        )
        fix = MockAdapter().queue(_fixing_response(fixture.task))
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=2, grant_budget=0
            ),
        ).run(fixture.context(store))
        runtime = store.get_task_runtime(fixture.task.id)
        recorded = _assessments(store, fixture)

    assert status is TaskRuntimeStatus.BLOCKED
    assert len(fix.calls) == 1
    # Both passes are inside the minimum, so neither is a grant, and the loop
    # blocks on a pass it was closing findings in.
    assert [
        (item.round, item.open_findings, item.refunded) for item in recorded
    ] == [(1, 2, None), (2, 1, None)]
    assert runtime is not None and runtime.review_round == 2
    assert runtime.state_reason == (
        "review ended without approval. The loop took no rounds past its minimum "
        "of 2, and its last round was converging."
    )


def test_a_review_grant_budget_below_zero_is_refused() -> None:
    """Zero asks for today's passes; below zero asks for fewer than the minimum."""
    assert (
        HostReviewFixConfig(review_model="review-model", grant_budget=0).grant_budget
        == 0
    )
    with pytest.raises(ValueError, match="grant budget must not be negative"):
        HostReviewFixConfig(review_model="review-model", grant_budget=-1)


def test_a_granted_pass_is_counted_once_across_an_interrupted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed loop must not spend the budget its interrupted round spent.

    The grant is recorded with the attempt that earned it, so the round the
    resume replays is the round the record already holds: reassessing it would
    charge one pass twice and lose the bound the budget promises.
    """
    fixture = _coding_fixture(tmp_path)

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        review = (
            MockAdapter()
            .queue(
                MockResponse(
                    payload=_review_payload(
                        fixture.task,
                        status="issues_found",
                        findings=[
                            _finding(_UNBOUNDED_TIMEOUT, severity="blocker")
                        ],
                    )
                )
            )
            .queue(MockResponse(dynamic=_repeating_review(store, fixture)))
        )
        fix = MockAdapter().queue(_fixing_response(fixture.task))
        config = HostReviewFixConfig(
            review_model="review-model", review_passes=1, grant_budget=1
        )
        transition = store.transition_task_runtime

        def crash_before_blocking(*args, **kwargs):
            if kwargs.get("new_status") is TaskRuntimeStatus.BLOCKED:
                raise RuntimeError("simulated restart after durable review")
            return transition(*args, **kwargs)

        monkeypatch.setattr(store, "transition_task_runtime", crash_before_blocking)
        with pytest.raises(RuntimeError, match="simulated restart"):
            HostReviewFixPhase(
                fixture.repository,
                review,
                fix_adapter=fix,
                steering_adapter=_steering_adapter(1),
                config=config,
            ).run(fixture.context(store))
        monkeypatch.setattr(store, "transition_task_runtime", transition)

        replay = MockAdapter()
        status = HostReviewFixPhase(
            fixture.repository,
            replay,
            fix_adapter=replay,
            steering_adapter=replay,
            config=config,
        ).run(fixture.context(store))
        runtime = store.get_task_runtime(fixture.task.id)
        recorded = _assessments(store, fixture)

    assert status is TaskRuntimeStatus.BLOCKED
    assert len(review.calls) == 2
    assert replay.calls == []
    assert [(item.round, item.refunded) for item in recorded] == [
        (1, None),
        (2, False),
    ]
    assert runtime is not None
    assert "took 1 granted round past its minimum of 1" in (
        runtime.state_reason or ""
    )


@pytest.mark.parametrize(
    "review_passes", [1, 3], ids=["granted-pass", "inside-the-minimum"]
)
def test_a_fix_without_a_commit_blocks_on_the_spot_in_any_pass(
    review_passes: int, tmp_path: Path
) -> None:
    """The one bound a longer loop meets more often, and it is unchanged.

    A fix round whose agent produces no commit blocks the task where it stands,
    with no second try, and a granted pass buys it none.
    """
    fixture = _coding_fixture(tmp_path)

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        review = (
            MockAdapter()
            .queue(
                MockResponse(
                    payload=_review_payload(
                        fixture.task,
                        status="issues_found",
                        findings=[_BROAD_PATTERN],
                    )
                )
            )
            .queue(
                MockResponse(
                    payload=_review_payload(
                        fixture.task,
                        status="issues_found",
                        findings=[_UNBOUNDED_TIMEOUT],
                    )
                )
            )
        )
        fix = (
            MockAdapter()
            .queue(_fixing_response(fixture.task))
            .queue(MockResponse(payload=_completed_payload(fixture.task)))
        )
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=_steering_adapter(2),
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=review_passes
            ),
        ).run(fixture.context(store))
        runtime = store.get_task_runtime(fixture.task.id)

    assert status is TaskRuntimeStatus.BLOCKED
    assert runtime is not None and runtime.review_round == 2
    assert runtime.state_reason == (
        "fix reported completed without producing a commit; worktree preserved"
    )


def test_a_review_that_could_not_review_leaves_the_ledger_untouched(
    tmp_path: Path,
) -> None:
    """Only a review that actually reviewed reconciles.

    A reviewer that reports it could not review still carries the findings key
    its schema requires, and the task is already going to its terminal failed
    state, so recording objections it says it could not form would leave rows
    no later round can answer.
    """
    fixture = _coding_fixture(tmp_path)
    review = (
        MockAdapter()
        .queue(
            MockResponse(
                payload=_review_payload(
                    fixture.task,
                    status="issues_found",
                    findings=[_BROAD_PATTERN],
                )
            )
        )
        .queue(
            MockResponse(
                payload=_review_payload(
                    fixture.task,
                    status="failed",
                    findings=[_finding("the suite will not build", severity="blocker")],
                )
            )
        )
    )
    fix = MockAdapter().queue(_fixing_response(fixture.task))

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=3
            ),
        ).run(fixture.context(store))
        rows = store.list_execution_ledger_findings(fixture.task.id)
        first_review = store.list_agent_attempts(fixture.task.id)[1]

    assert status is TaskRuntimeStatus.FAILED
    assert [(row.message, row.status) for row in rows] == [
        (_BROAD_PATTERN, FindingStatus.OPEN)
    ]
    assert rows[0].last_seen_round == 1
    assert rows[0].attempt_id == first_review.id


class _CancelledHoldingAPayload(MockAdapter):
    """Report a cancellation and a payload in one result.

    No adapter in the product does: each one's cancellation omits the payload,
    so the branch that discards an interrupted round's objections cannot be
    reached through a real one, and the rule it keeps would go untested.
    """

    def run(self, spec, *, cancel=None):
        return replace(
            super().run(spec, cancel=None), status=AgentStatus.CANCELLED
        )


def test_a_cancelled_round_records_nothing_and_its_rerun_records_once(
    tmp_path: Path,
) -> None:
    """An interrupted round left no review, so it left no objections either.

    A round that did not review leaves nothing for the round replacing it to
    say over again — which matters because a re-run mints fresh ids, so rows an
    interrupted round had recorded would stand beside the new ones rather than
    being replaced by them.
    """
    fixture = _coding_fixture(tmp_path)
    interrupted = _CancelledHoldingAPayload().queue(
        MockResponse(
            payload=_review_payload(
                fixture.task,
                status="issues_found",
                findings=[_BROAD_PATTERN],
            )
        )
    )

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        config = HostReviewFixConfig(
            review_model="review-model", review_passes=1, grant_budget=0
        )
        resumable = HostReviewFixPhase(
            fixture.repository, interrupted, config=config
        ).run(fixture.context(store))
        # The round really did hand back a reviewed payload, and it was really
        # discarded: an unconsumed response would prove nothing.
        assert interrupted.responses == []
        after_cancellation = store.list_execution_ledger_findings(fixture.task.id)

        review = MockAdapter().queue(
            MockResponse(
                payload=_review_payload(
                    fixture.task,
                    status="issues_found",
                    findings=[_BROAD_PATTERN],
                )
            )
        )
        status = HostReviewFixPhase(
            fixture.repository, review, config=config
        ).run(fixture.context(store))
        rows = store.list_execution_ledger_findings(fixture.task.id)
        attempts = store.list_agent_attempts(fixture.task.id)

    assert resumable is TaskRuntimeStatus.REVIEW
    assert after_cancellation == []
    assert status is TaskRuntimeStatus.BLOCKED
    assert [(row.message, row.first_seen_round) for row in rows] == [
        (_BROAD_PATTERN, 1)
    ]
    assert rows[0].attempt_id == attempts[-1].id


def test_a_rounds_findings_are_durable_with_the_attempt_that_produced_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume replays a completed attempt's outcome without classifying again.

    So rows written after the attempt finished are rows never written at all:
    the round that raised them never runs again, and the fix it asked for would
    be handed nothing to answer.
    """
    fixture = _coding_fixture(tmp_path)
    interrupted = MockAdapter().queue(
        MockResponse(
            payload=_review_payload(
                fixture.task,
                status="issues_found",
                findings=[_BROAD_PATTERN],
            )
        )
    )

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        transition = store.transition_task_runtime

        def crash_before_fix(*args, **kwargs):
            if kwargs.get("new_status") is TaskRuntimeStatus.FIX:
                raise RuntimeError("simulated restart after durable review")
            return transition(*args, **kwargs)

        monkeypatch.setattr(store, "transition_task_runtime", crash_before_fix)
        with pytest.raises(RuntimeError, match="simulated restart"):
            HostReviewFixPhase(
                fixture.repository,
                interrupted,
                config=HostReviewFixConfig(review_model="review-model"),
            ).run(fixture.context(store))
        monkeypatch.setattr(store, "transition_task_runtime", transition)
        stalled = store.get_task_runtime(fixture.task.id)
        recorded = store.list_execution_ledger_findings(fixture.task.id)

        review = MockAdapter().queue(
            MockResponse(payload=_review_payload(fixture.task, status="approved"))
        )
        fix = MockAdapter().queue(_fixing_response(fixture.task))
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            config=HostReviewFixConfig(review_model="review-model"),
        ).run(fixture.context(store))
        rows = store.list_execution_ledger_findings(fixture.task.id)

    assert stalled is not None and stalled.status is TaskRuntimeStatus.REVIEW
    assert [row.message for row in recorded] == [_BROAD_PATTERN]
    assert status is TaskRuntimeStatus.MERGING
    assert len(interrupted.calls) == 1
    assert _BROAD_PATTERN in fix.calls[0].user_prompt
    assert [row.id for row in rows] == [recorded[0].id]


def test_a_grant_is_decided_on_this_tasks_passes_and_no_others(
    tmp_path: Path,
) -> None:
    """The configured minimum is shared; the rounds that spend it are not.

    Every task of a run records under one loop and one Borg, so a history read
    without the task would compare this round against whichever task happened to
    record the matching round first — and refund a pass that closed nothing.
    """
    fixture = _coding_fixture(tmp_path)

    def repeat(spec):
        rows = store.list_execution_ledger_findings(fixture.task.id)
        if not rows:
            return _review_payload(
                fixture.task,
                status="issues_found",
                findings=[_finding(_BROAD_PATTERN, severity="blocker")],
            )
        return _review_payload(
            fixture.task,
            status="issues_found",
            findings=[
                _finding(
                    _BROAD_PATTERN,
                    severity="blocker",
                    repeats=str(rows[0].id),
                )
            ],
        )

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        # Another task of the same run, one round in, holding more open than
        # this task ever will. Read into this task's history it would make every
        # round of it look like one that closed something.
        store.record_review_assessment(
            ReviewAssessment(
                borg_id=fixture.borg.id,
                loop=TASK_REVIEW_LOOP,
                task_id=uuid4(),
                round=1,
                minimum=1,
                converging=False,
                open_findings=5,
            )
        )
        review = MockAdapter()
        for _ in range(2):
            review.queue(MockResponse(dynamic=repeat))
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=MockAdapter().queue(_fixing_response(fixture.task)),
            steering_adapter=_steering_adapter(1),
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=1, grant_budget=1
            ),
        ).run(fixture.context(store))
        recorded = store.list_review_assessments(
            fixture.borg.id, loop=TASK_REVIEW_LOOP, task_id=fixture.task.id
        )

    # Its own round one left one objection open and round two left the same one,
    # so the grant bought nothing and the single-grant budget is spent.
    assert [(item.round, item.open_findings, item.refunded) for item in recorded] == [
        (1, 1, None),
        (2, 1, False),
    ]
    assert status is TaskRuntimeStatus.BLOCKED


def test_an_assessment_that_cannot_be_recorded_leaves_its_attempt_unfinished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The grant a round spent is durable with the round's own outcome.

    A resume replays a completed attempt without assessing again, so a snapshot
    written outside that step and lost is lost for good — and the round after it
    is charged for a pass that had in fact closed findings.
    """
    fixture = _coding_fixture(tmp_path)
    review = MockAdapter().queue(
        MockResponse(
            payload=_review_payload(
                fixture.task,
                status="issues_found",
                findings=[_BROAD_PATTERN],
            )
        )
    )

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)

        def refuse(assessment):
            raise RuntimeError("simulated assessment failure")

        monkeypatch.setattr(store, "record_review_assessment", refuse)
        with pytest.raises(RuntimeError, match="simulated assessment failure"):
            HostReviewFixPhase(
                fixture.repository,
                review,
                config=HostReviewFixConfig(review_model="review-model"),
            ).run(fixture.context(store))
        monkeypatch.undo()
        attempt = store.list_agent_attempts(fixture.task.id)[-1]
        rows = store.list_execution_ledger_findings(fixture.task.id)
        recorded = store.list_review_assessments(
            fixture.borg.id, loop=TASK_REVIEW_LOOP, task_id=fixture.task.id
        )

    assert attempt.status is ExecutionAttemptStatus.RUNNING
    # The findings rolled back with it, so the round is wholly replayable.
    assert rows == []
    assert recorded == []


def test_findings_that_cannot_be_recorded_leave_their_attempt_unfinished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ledger write and the attempt's completion are one durable step.

    The other half of the same rule: an interruption must not leave a round's
    objections recorded against an attempt that never finished, because the
    resume that replays the attempt would then answer them twice.
    """
    fixture = _coding_fixture(tmp_path)
    review = MockAdapter().queue(
        MockResponse(
            payload=_review_payload(
                fixture.task,
                status="issues_found",
                findings=[_BROAD_PATTERN],
            )
        )
    )

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)

        def refuse(rows):
            raise RuntimeError("simulated ledger failure")

        monkeypatch.setattr(store, "record_execution_ledger_findings", refuse)
        with pytest.raises(RuntimeError, match="simulated ledger failure"):
            HostReviewFixPhase(
                fixture.repository,
                review,
                config=HostReviewFixConfig(review_model="review-model"),
            ).run(fixture.context(store))
        monkeypatch.undo()
        attempt = store.list_agent_attempts(fixture.task.id)[-1]
        rows = store.list_execution_ledger_findings(fixture.task.id)

    assert attempt.phase == "review"
    assert attempt.status is ExecutionAttemptStatus.RUNNING
    assert rows == []


def test_cancelled_review_remains_resumable_with_immutable_attempt(
    tmp_path: Path,
) -> None:
    fixture = _coding_fixture(tmp_path)
    cancel = CancellationToken()
    cancel.cancel()
    review = MockAdapter().queue(
        MockResponse(payload=_review_payload(fixture.task, status="approved"))
    )

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            config=HostReviewFixConfig(review_model="review-model"),
        ).run(fixture.context(store, cancel=cancel))
        runtime = store.get_task_runtime(fixture.task.id)
        attempt = store.list_agent_attempts(fixture.task.id)[-1]

    assert status is TaskRuntimeStatus.REVIEW
    assert runtime is not None and runtime.resume_phase == "review"
    assert attempt.status is ExecutionAttemptStatus.CANCELLED
    assert attempt.result["_betterborg"]["outcome_status"] == "review"
    assert "interrupted" in attempt.result["_betterborg"]["outcome_reason"]
    artifact_dir = fixture.repository / attempt.result["_betterborg"]["artifact_dir"]
    assert not (artifact_dir / "artifact-manifest.json").stat().st_mode & stat.S_IWUSR


def test_completed_review_resumes_transition_without_replaying_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _coding_fixture(tmp_path)
    first_review = MockAdapter().queue(
        MockResponse(payload=_review_payload(fixture.task, status="approved"))
    )

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        transition = store.transition_task_runtime

        def crash_before_merge(*args, **kwargs):
            if kwargs.get("new_status") is TaskRuntimeStatus.MERGING:
                raise RuntimeError("simulated restart after durable review")
            return transition(*args, **kwargs)

        monkeypatch.setattr(store, "transition_task_runtime", crash_before_merge)
        with pytest.raises(RuntimeError, match="simulated restart"):
            HostReviewFixPhase(
                fixture.repository,
                first_review,
                config=HostReviewFixConfig(review_model="review-model"),
            ).run(fixture.context(store))
        monkeypatch.setattr(store, "transition_task_runtime", transition)
        interrupted = store.get_task_runtime(fixture.task.id)
        assert interrupted is not None
        assert interrupted.status is TaskRuntimeStatus.REVIEW

        replay = MockAdapter()
        status = HostReviewFixPhase(
            fixture.repository,
            replay,
            config=HostReviewFixConfig(review_model="review-model"),
        ).run(fixture.context(store))

    assert status is TaskRuntimeStatus.MERGING
    assert len(first_review.calls) == 1
    assert replay.calls == []
    with SqliteStore.open(fixture.database) as reopened:
        assert [
            attempt.phase
            for attempt in reopened.list_agent_attempts(fixture.task.id)
        ] == ["coding", "review"]


def test_every_phase_is_told_not_to_weaken_an_existing_assertion() -> None:
    """A rule the generated role prompt omits is a rule nobody in a run states.

    A run changed what an omitted slice start means and, rather than fail the
    repository's existing parser test, edited that test to assert the opposite.
    Coding, review and the fix round all accepted it, and the fix round is where
    the move is most tempting, because a finding is already asking for a change.
    Betterborg renders these three prompts itself, so the rule arrives whatever
    the model that writes the role prompts produced.
    """
    borg = Borg(repository_id=uuid4(), name="prompt-rules")
    task = _record(
        uuid4(), borg, position=1, stem="09-run-coding-agent", dependencies=[]
    )
    inputs = VerifiedTaskInputs(
        task=task,
        task_path=Path(f"{task.stem}.md"),
        task_markdown=render_task_markdown(task.task),
        dependencies=(),
        system_prompt="You are the generated coding agent.\n",
    )

    assert EXISTING_TEST_RULE in _render_user_prompt(inputs)
    assert EXISTING_TEST_RULE in _render_fix_prompt(
        inputs,
        findings=(
            _ledger_row(
                task,
                "the parser must keep the documented omitted-start value",
            ),
        ),
        review_round=1,
    )
    assert EXISTING_TEST_REVIEW_RULE in _render_review_prompt(
        inputs,
        branch="betterborg/09-run-coding-agent",
        base_commit="a" * 40,
        current_commit="b" * 40,
        review_round=1,
    )
    assert EXISTING_TEST_MERGE_RULE in _render_merge_prompt(
        inputs,
        task_branch="betterborg/09-run-coding-agent",
        project_branch="project/demo",
        approved_commit="b" * 40,
        base_commit="a" * 40,
        unresolved=("evaluator/evaluator_test.go",),
    )


def test_the_rules_keep_an_honest_assertion_change_possible() -> None:
    """A flat prohibition would trade one defect for a worse one.

    Asserting only that a constant appears in a prompt holds for any value of
    that constant, so the half that forbids the dishonest edit is guarded and
    the half that permits the honest one is not. A coding agent left with no
    legal path abandons the finding or defies the rule; a reviewer told to
    judge only the base commit cannot see an assertion an earlier round of the
    same task wrote.
    """
    assert "When the task or a review finding requires" in EXISTING_TEST_RULE
    assert "return status blocked" in EXISTING_TEST_RULE
    assert "unless the assigned task required" in EXISTING_TEST_REVIEW_RULE
    assert "an earlier round of this task" in EXISTING_TEST_REVIEW_RULE
    assert "resolve the code" in EXISTING_TEST_MERGE_RULE
    assert "fail rather than choose one" in EXISTING_TEST_MERGE_RULE


def test_the_reviewer_is_told_the_finding_contract_in_a_rendered_prompt() -> None:
    """Severity and the two declarations reach the reviewer in a rendered prompt.

    The review role prompt is generated per repository, so the requirement it
    comes from can be reworded or dropped by the model that writes it. The
    schema can require the declarations, but a required field with no
    instruction behind it comes back empty every round and a ledger that is
    never told what closed never drains.
    """
    borg = Borg(repository_id=uuid4(), name="review-finding-rule")
    task = _record(
        uuid4(), borg, position=1, stem="09-run-coding-agent", dependencies=[]
    )
    inputs = VerifiedTaskInputs(
        task=task,
        task_path=Path(f"{task.stem}.md"),
        task_markdown=render_task_markdown(task.task),
        dependencies=(),
        system_prompt="You are the generated review agent.\n",
    )

    assert REVIEW_FINDING_RULE in _render_review_prompt(
        inputs,
        branch="betterborg/09-run-coding-agent",
        base_commit="a" * 40,
        current_commit="b" * 40,
        review_round=1,
    )
    assert "severity of blocker, major, or minor" in REVIEW_FINDING_RULE
    assert "A minor finding does not hold the task" in REVIEW_FINDING_RULE
    assert "list in resolved the id of every one" in REVIEW_FINDING_RULE
    assert "set repeats to the id" in REVIEW_FINDING_RULE
    assert "neither resolve nor repeat stays open" in REVIEW_FINDING_RULE


def test_a_committed_partial_reaches_review_rather_than_being_discarded(
    tmp_path: Path,
) -> None:
    fixture = _coding_fixture(tmp_path)
    adapter = MockAdapter().queue(
        _committing_response(
            fixture.task,
            payload=_unfinished_payload(fixture.task, status="partial"),
        )
    )

    with SqliteStore.open(fixture.database) as store:
        status = HostCodingPhase(
            fixture.repository,
            adapter,
            config=HostCodingConfig(model="test-model"),
        ).run(fixture.context(store))
        runtime = store.get_task_runtime(fixture.task.id)

    assert status is TaskRuntimeStatus.REVIEW
    assert runtime is not None and runtime.status is TaskRuntimeStatus.REVIEW
    head = _git(Path(runtime.worktree_path), "rev-parse", "HEAD")
    assert runtime.state_reason == f"coding reported partial and committed {head}"


def test_a_partial_that_committed_nothing_still_blocks(tmp_path: Path) -> None:
    fixture = _coding_fixture(tmp_path)

    def leave_uncommitted(spec):
        (spec.cwd / "unfinished.txt").write_text("keep me\n", encoding="utf-8")
        return _unfinished_payload(fixture.task, status="partial")

    adapter = MockAdapter().queue(MockResponse(dynamic=leave_uncommitted))
    with SqliteStore.open(fixture.database) as store:
        status = HostCodingPhase(
            fixture.repository,
            adapter,
            config=HostCodingConfig(model="test-model"),
        ).run(fixture.context(store))
        runtime = store.get_task_runtime(fixture.task.id)

    assert status is TaskRuntimeStatus.BLOCKED
    assert runtime is not None
    assert "without producing a commit" in runtime.state_reason
    assert (Path(runtime.worktree_path) / "unfinished.txt").is_file()


def test_a_blocked_status_stays_terminal_even_holding_a_commit(
    tmp_path: Path,
) -> None:
    fixture = _coding_fixture(tmp_path)
    adapter = MockAdapter().queue(
        _committing_response(
            fixture.task,
            payload=_unfinished_payload(fixture.task, status="blocked"),
        )
    )

    with SqliteStore.open(fixture.database) as store:
        status = HostCodingPhase(
            fixture.repository,
            adapter,
            config=HostCodingConfig(model="test-model"),
        ).run(fixture.context(store))
        runtime = store.get_task_runtime(fixture.task.id)

    assert status is TaskRuntimeStatus.BLOCKED
    assert runtime is not None
    assert runtime.state_reason == "coding agent reported blocked"


def test_review_is_told_what_the_coding_agent_left_unfinished(
    tmp_path: Path,
) -> None:
    fixture = _coding_fixture(tmp_path)
    review = MockAdapter().queue(
        MockResponse(payload=_review_payload(fixture.task, status="approved"))
    )

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(
            fixture,
            store,
            coding_payload=_unfinished_payload(fixture.task, status="partial"),
        )
        HostReviewFixPhase(
            fixture.repository,
            review,
            config=HostReviewFixConfig(review_model="review-model"),
        ).run(fixture.context(store))

    prompt = review.calls[0].user_prompt
    assert "did not report the task finished" in prompt
    assert "'partial'" in prompt
    assert UNRESOLVED_BLOCKER in prompt
    assert UNRESOLVED_FOLLOW_UP in prompt


def test_review_hears_nothing_unfinished_when_coding_reported_completed(
    tmp_path: Path,
) -> None:
    fixture = _coding_fixture(tmp_path)
    review = MockAdapter().queue(
        MockResponse(payload=_review_payload(fixture.task, status="approved"))
    )

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        HostReviewFixPhase(
            fixture.repository,
            review,
            config=HostReviewFixConfig(review_model="review-model"),
        ).run(fixture.context(store))

    assert "did not report the task finished" not in review.calls[0].user_prompt


def test_a_committed_partial_fix_returns_to_review(tmp_path: Path) -> None:
    fixture = _coding_fixture(tmp_path)
    finding = "feature.txt must include the reviewed fix"
    review = (
        MockAdapter()
        .queue(
            MockResponse(
                payload=_review_payload(
                    fixture.task, status="issues_found", findings=[finding]
                )
            )
        )
        .queue(
            MockResponse(payload=_review_payload(fixture.task, status="approved"))
        )
    )
    fix = MockAdapter().queue(
        _fixing_response(
            fixture.task,
            payload=_unfinished_payload(fixture.task, status="partial"),
        )
    )

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            config=HostReviewFixConfig(
                review_model="review-model", fix_model="fix-model"
            ),
        ).run(fixture.context(store))
        attempts = store.list_agent_attempts(fixture.task.id)

    assert status is TaskRuntimeStatus.MERGING
    assert [attempt.phase for attempt in attempts] == [
        "coding",
        "review",
        "fix",
        "review",
    ]


def test_coding_is_told_when_its_checkout_was_never_installed(
    tmp_path: Path,
) -> None:
    """An agent that is not told spends a turn finding out, or misreports.

    A command failing for want of a dependency is a fact about the checkout
    and not about the change, and only the run knows which it is.
    """
    fixture = _coding_fixture(tmp_path)
    adapter = MockAdapter().queue(_committing_response(fixture.task))
    note = "preparation did not complete: environment command failed"

    with SqliteStore.open(fixture.database) as store:
        status = HostCodingPhase(
            fixture.repository,
            adapter,
            config=HostCodingConfig(model="test-model"),
        ).run(fixture.context(store), preparation_note=note)

    assert status is TaskRuntimeStatus.REVIEW
    prompt = adapter.calls[0].user_prompt
    assert "This checkout is not installed" in prompt
    assert note in prompt
    assert "Do not try to install them" in prompt


def test_a_prepared_checkout_says_nothing_about_itself(tmp_path: Path) -> None:
    fixture = _coding_fixture(tmp_path)
    adapter = MockAdapter().queue(_committing_response(fixture.task))

    with SqliteStore.open(fixture.database) as store:
        HostCodingPhase(
            fixture.repository,
            adapter,
            config=HostCodingConfig(model="test-model"),
        ).run(fixture.context(store))

    assert "This checkout is not installed" not in adapter.calls[0].user_prompt


def test_a_blocked_commit_stops_the_task_unless_a_repository_says_otherwise(
    tmp_path: Path,
) -> None:
    """Blocked is the agent saying the task does not make sense as given.

    Carrying on past that by default would be the run ignoring its own alarm,
    so the commit reaches review only where a repository has said something
    outside Betterborg decides whether the work is worth having.
    """
    stopped = _coding_fixture(tmp_path)
    with SqliteStore.open(stopped.database) as store:
        status = HostCodingPhase(
            stopped.repository,
            MockAdapter().queue(
                _committing_response(
                    stopped.task,
                    payload=_unfinished_payload(stopped.task, status="blocked"),
                )
            ),
            config=HostCodingConfig(model="test-model"),
        ).run(stopped.context(store))
        stopped_runtime = store.get_task_runtime(stopped.task.id)

    assert status is TaskRuntimeStatus.BLOCKED
    assert stopped_runtime is not None
    assert stopped_runtime.state_reason == "coding agent reported blocked"

    reviewed_root = tmp_path / "reviewed"
    reviewed_root.mkdir()
    reviewed = _coding_fixture(reviewed_root)
    with SqliteStore.open(reviewed.database) as store:
        status = HostCodingPhase(
            reviewed.repository,
            MockAdapter().queue(
                _committing_response(
                    reviewed.task,
                    payload=_unfinished_payload(reviewed.task, status="blocked"),
                )
            ),
            config=HostCodingConfig(
                model="test-model",
                blocked_tasks=BlockedTaskPolicy.REVIEW,
            ),
        ).run(reviewed.context(store))
        reviewed_runtime = store.get_task_runtime(reviewed.task.id)

    assert status is TaskRuntimeStatus.REVIEW
    assert reviewed_runtime is not None
    head = _git(Path(reviewed_runtime.worktree_path), "rev-parse", "HEAD")
    assert reviewed_runtime.state_reason == (
        f"coding reported blocked and committed {head}"
    )


def test_a_blocked_commit_that_reaches_review_carries_its_blockers(
    tmp_path: Path,
) -> None:
    """The reason it stopped is exactly what the reviewer needs to judge it."""
    fixture = _coding_fixture(tmp_path)
    review = MockAdapter().queue(
        MockResponse(payload=_review_payload(fixture.task, status="approved"))
    )

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(
            fixture,
            store,
            coding_payload=_unfinished_payload(fixture.task, status="blocked"),
            coding_config=HostCodingConfig(
                model="coding-model", blocked_tasks=BlockedTaskPolicy.REVIEW
            ),
        )
        HostReviewFixPhase(
            fixture.repository,
            review,
            config=HostReviewFixConfig(
                review_model="review-model",
                blocked_tasks=BlockedTaskPolicy.REVIEW,
            ),
        ).run(fixture.context(store))

    prompt = review.calls[0].user_prompt
    assert "'blocked'" in prompt
    assert UNRESOLVED_BLOCKER in prompt


@pytest.mark.parametrize(
    "status", sorted(CODING_RESULT_SCHEMA["properties"]["status"]["enum"])
)
def test_the_coding_schema_offers_only_statuses_the_run_acts_on(
    tmp_path: Path, status: str
) -> None:
    """Every status the coding agent may return has a rule that names it.

    A status offered by the schema and implemented nowhere costs the task
    everything the agent built, so the catch-all is reachable only by a
    payload the schema would have rejected.
    """
    fixture = _coding_fixture(tmp_path)
    adapter = MockAdapter().queue(
        _committing_response(
            fixture.task, payload=_unfinished_payload(fixture.task, status=status)
        )
    )

    with SqliteStore.open(fixture.database) as store:
        HostCodingPhase(
            fixture.repository,
            adapter,
            config=HostCodingConfig(
                model="test-model", blocked_tasks=BlockedTaskPolicy.REVIEW
            ),
        ).run(fixture.context(store))
        runtime = store.get_task_runtime(fixture.task.id)

    assert runtime is not None
    assert (runtime.status, runtime.state_reason) != (
        TaskRuntimeStatus.BLOCKED,
        f"coding agent reported {status}",
    )


@pytest.mark.parametrize(
    "status", sorted(REVIEW_RESULT_SCHEMA["properties"]["status"]["enum"])
)
def test_the_review_schema_offers_only_statuses_the_run_acts_on(
    tmp_path: Path, status: str
) -> None:
    """The same rule over the review agent's own vocabulary."""
    fixture = _coding_fixture(tmp_path)
    review = MockAdapter().queue(
        MockResponse(
            payload=_review_payload(
                fixture.task,
                status=status,
                findings=(
                    ["the media-type pattern is too broad"]
                    if status == "issues_found"
                    else None
                ),
            )
        )
    )

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        HostReviewFixPhase(
            fixture.repository,
            review,
            config=HostReviewFixConfig(review_model="review-model"),
        ).run(fixture.context(store))
        runtime = store.get_task_runtime(fixture.task.id)

    assert runtime is not None
    assert (runtime.status, runtime.state_reason) != (
        TaskRuntimeStatus.BLOCKED,
        f"review agent reported {status}",
    )


def _blocking_review(fixture: CodingFixture) -> MockResponse:
    """One review that requests fixes and leaves a blocker standing."""
    return MockResponse(
        payload=_review_payload(
            fixture.task,
            status="issues_found",
            findings=[_finding(_UNBOUNDED_TIMEOUT, severity="blocker")],
        )
    )


def _approving_review(fixture: CodingFixture, store: SqliteStore) -> MockResponse:
    def approve(_spec):
        rows = store.list_execution_ledger_findings(fixture.task.id)
        return _review_payload(
            fixture.task,
            status="approved",
            resolved=[str(row.id) for row in rows],
        )

    return MockResponse(dynamic=approve)


def _fix_prompts(fix: MockAdapter) -> list[str]:
    return [call.user_prompt for call in fix.calls]


def test_a_granted_fix_the_review_read_as_stuck_carries_a_steering_note(
    tmp_path: Path,
) -> None:
    """The fix is the answering turn in a task's review, so the note joins it.

    The round the granted fix is filed under is the round whose assessment
    asked for the note: the review advanced the runtime's counter as it
    requested the fix, so the write and the resume read agree without
    arithmetic.
    """
    fixture = _coding_fixture(tmp_path)
    usage = AgentUsage(cost_usd=0.11, tokens_input=90, tokens_output=20)

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        review = (
            MockAdapter()
            .queue(_blocking_review(fixture))
            .queue(_approving_review(fixture, store))
        )
        fix = MockAdapter().queue(_fixing_response(fixture.task))
        steering = MockAdapter().queue(
            MockResponse(
                payload={"note": "Answer the timeout first.", "confidence": "high"},
                usage=usage,
            )
        )
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=steering,
            config=HostReviewFixConfig(
                review_model="review-model",
                steering_model="steering-model",
                review_passes=1,
                grant_budget=5,
                review_effort="review-effort",
                steering_effort="steering-effort",
                review_billing_mode=BillingMode.API,
                steering_billing_mode=BillingMode.SUBSCRIPTION,
            ),
        ).run(fixture.context(store))
        rows = store.list_steering_notes(fixture.borg.id)
        attempts = store.list_agent_attempts(fixture.task.id)

    assert status is TaskRuntimeStatus.MERGING
    assert len(steering.calls) == 1
    # Its own instructions, and not the prompt the review turn beside it runs
    # under: the confidence it reports and the file it must not write are
    # asked for here and nowhere else.
    assert steering.calls[0].system_prompt == STEERING_SYSTEM_PROMPT
    assert steering.calls[0].schema == STEERING_NOTE_SCHEMA
    # Its own stage's settings, and not the ones the review turn beside it runs
    # under: the stage is configured separately and every field has to arrive.
    assert steering.calls[0].effort == "steering-effort"
    assert steering.calls[0].billing_mode is BillingMode.SUBSCRIPTION
    assert [call.effort for call in review.calls] == ["review-effort"] * 2
    assert "## Steering note\n\nAnswer the timeout first." in _fix_prompts(fix)[0]
    # The reviewer meets the fix rather than the instruction.
    assert not any("Steering note" in call.user_prompt for call in review.calls)
    assert [
        (row.round, row.source, row.converging, row.task_id) for row in rows
    ] == [(1, "agent", False, fixture.task.id)]

    steering_attempts = [item for item in attempts if item.phase == "steering"]
    assert [item.model for item in steering_attempts] == ["steering-model"]
    # Recorded on its own attempt and nowhere else: no completion-sample
    # bucket prices a steering agent.
    assert [item.usage for item in steering_attempts] == [usage]
    assert [
        item.status for item in steering_attempts
    ] == [ExecutionAttemptStatus.COMPLETED]
    # Its own phase, which the replay and the commit-declaring set both ignore.
    assert [item.phase for item in attempts] == [
        "coding",
        "review",
        "steering",
        "fix",
        "review",
    ]
    assert steering.calls[0].allowed_tools == READ_ONLY_API_TOOLS


def test_a_second_steered_round_is_keyed_to_the_round_it_steers(
    tmp_path: Path,
) -> None:
    """The round a note belongs to is the ledger's, not the minimum it passed.

    A loop that steers once steers exactly on its minimum, so the two numbers
    agree there and only a second grant tells them apart: the row it writes,
    the round its attempt records, the age its fallback note reports and the
    row a re-entry would find are all the round it steers.
    """
    fixture = _coding_fixture(tmp_path)

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)

        def repeating(_spec):
            rows = store.list_execution_ledger_findings(fixture.task.id)
            return _review_payload(
                fixture.task,
                status="issues_found",
                findings=[
                    _finding(
                        _UNBOUNDED_TIMEOUT,
                        severity="blocker",
                        repeats=str(
                            _ledger_row_for(rows, _UNBOUNDED_TIMEOUT).id
                        ),
                    )
                ],
            )

        review = (
            MockAdapter()
            .queue(_blocking_review(fixture))
            .queue(MockResponse(dynamic=repeating))
            .queue(_approving_review(fixture, store))
        )
        fix = (
            MockAdapter()
            .queue(_fixing_response(fixture.task))
            .queue(_fixing_response(fixture.task))
        )
        # The second grant's turn fails, so the note it runs on is assembled
        # for the round it steers rather than written for it.
        steering = (
            MockAdapter()
            .queue(
                MockResponse(
                    payload={
                        "note": "Answer the timeout first.",
                        "confidence": "high",
                    }
                )
            )
            .queue(
                MockResponse(exit_code=1, error="provider refused the request")
            )
        )
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=steering,
            config=HostReviewFixConfig(
                review_model="review-model",
                steering_model="steering-model",
                review_passes=1,
                grant_budget=5,
            ),
        ).run(fixture.context(store))
        rows = store.list_steering_notes(fixture.borg.id)
        attempts = store.list_agent_attempts(fixture.task.id)

    assert status is TaskRuntimeStatus.MERGING
    # Two grants, each steered once: a second turn ran, so the re-entry read
    # of the first round's row did not answer for the second.
    assert len(steering.calls) == 2
    assert [(row.round, row.source) for row in rows] == [
        (1, "agent"),
        (2, "assembled"),
    ]
    # The fallback's age is measured from the round it steers.
    assert (
        f"- {_UNBOUNDED_TIMEOUT} (blocker, first raised in round 1 "
        "and open for 2 rounds)"
    ) in rows[1].note
    assert [
        item.result["_betterborg"]["steered_round"]
        for item in attempts
        if item.phase == "steering"
    ] == [1, 2]


def test_the_steering_turn_is_handed_the_argument_in_the_ledgers_numbering(
    tmp_path: Path,
) -> None:
    """What the turn reads is the argument, numbered as the ledger numbers it.

    The summaries are built from review attempts, which carry the runtime's
    count of the rounds behind them, so each one is named a round later than
    the attempt that recorded it. Numbered from the attempt instead, the turn
    would read summaries one lower than the objections printed above them.
    """
    fixture = _coding_fixture(tmp_path)

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        def repeating(_spec):
            rows = store.list_execution_ledger_findings(fixture.task.id)
            return _review_payload(
                fixture.task,
                status="issues_found",
                findings=[
                    _finding(
                        _BROAD_PATTERN,
                        repeats=str(_ledger_row_for(rows, _BROAD_PATTERN).id),
                    ),
                    _finding(
                        _UNBOUNDED_TIMEOUT,
                        severity="blocker",
                        repeats=str(
                            _ledger_row_for(rows, _UNBOUNDED_TIMEOUT).id
                        ),
                    ),
                ],
            ) | {"summary": "The timeout is still unbounded."}

        review = (
            MockAdapter()
            .queue(
                MockResponse(
                    payload=_review_payload(
                        fixture.task,
                        status="issues_found",
                        findings=[
                            _finding(_BROAD_PATTERN),
                            _finding(_UNBOUNDED_TIMEOUT, severity="blocker"),
                        ],
                    )
                    | {"summary": "Two things stand in the way."}
                )
            )
            .queue(MockResponse(dynamic=repeating))
            .queue(_approving_review(fixture, store))
        )
        fix = (
            MockAdapter()
            .queue(_fixing_response(fixture.task))
            .queue(_fixing_response(fixture.task))
        )
        # One more than the arithmetic allows, so an extra steered round shows
        # up as a wrong count rather than as an adapter running dry.
        steering = _steering_adapter(2)
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=steering,
            config=HostReviewFixConfig(
                review_model="review-model",
                steering_model="steering-model",
                review_passes=2,
                grant_budget=5,
            ),
        ).run(fixture.context(store))

    assert status is TaskRuntimeStatus.MERGING
    assert len(steering.calls) == 1
    prompt = steering.calls[0].user_prompt

    # Who is about to answer, and what the argument is about.
    assert (
        "The fixing agent is about to answer these findings again, on a round "
        "granted because the review of the task's commit is not closing in on "
        "agreement."
    ) in prompt
    assert "Rounds argued so far: 2." in prompt
    assert f"- {_BROAD_PATTERN} (major, first raised in round 1)" in prompt
    assert f"- {_UNBOUNDED_TIMEOUT} (blocker, first raised in round 1)" in prompt
    assert "- Round 1: raised 2," in prompt
    # Each round's own account of what it decided, against the ledger's
    # numbering rather than the attempt's — and the reviews' accounts alone.
    # The coding, fix and steering attempts of the same task carry summaries
    # too, and a section that mixed them in would name two deciders per round.
    decided = prompt.split("## What each round decided")[1].strip().splitlines()
    assert decided == [
        "- Round 1: Two things stand in the way.",
        "- Round 2: The timeout is still unbounded.",
    ]
    # The verdict's own workings do not survive into the assessment and are
    # not here either.
    assert "veto" not in prompt and "converg" not in prompt


def test_a_steering_attempts_artifacts_are_sealed_like_every_other_turns(
    tmp_path: Path,
) -> None:
    """The one turn whose note can misdirect a round is not the unsealed one.

    A directory that cannot be sealed is still not the task's problem: the
    round was granted and the note is already in hand, so the failure is
    recorded on the attempt and the fix runs anyway.
    """
    fixture = _coding_fixture(tmp_path)

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        review = (
            MockAdapter()
            .queue(_blocking_review(fixture))
            .queue(_approving_review(fixture, store))
        )
        fix = MockAdapter().queue(_fixing_response(fixture.task))
        steering = _steering_adapter(1)
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=steering,
            config=HostReviewFixConfig(
                review_model="review-model",
                steering_model="steering-model",
                review_passes=1,
                grant_budget=5,
            ),
        ).run(fixture.context(store))
        attempts = store.list_agent_attempts(fixture.task.id)

    assert status is TaskRuntimeStatus.MERGING
    steering_attempt = next(
        item for item in attempts if item.phase == "steering"
    )
    metadata = steering_attempt.result["_betterborg"]
    assert "artifact_seal_error" not in metadata
    # An execution attempt carries no request context, so the round the note
    # steered and which note was used are recorded here or nowhere.
    assert metadata["steered_round"] == 1
    assert metadata["note_source"] == "agent"
    # And everything the driver's other turns keep, which is the rest of the
    # rule: the turn is one of them but for its outcome.
    coding_metadata = attempts[0].result["_betterborg"]
    assert metadata["base_commit"] == coding_metadata["base_commit"]
    assert metadata["prior_commit"] == coding_metadata["commit_sha"]
    assert metadata["commit_sha"] == coding_metadata["commit_sha"]
    assert metadata["provider"] == coding_metadata["provider"]
    assert metadata["model"] == "steering-model"
    assert metadata["billing_mode"] == BillingMode.API.value
    assert metadata["review_round"] == steering_attempt.review_round
    # The turn resolved and its note was attached, which is the outcome the
    # rest of this row is the record of.
    assert metadata["outcome_reason"] == "steering note attached"
    artifact_dir = fixture.repository / metadata["artifact_dir"]
    manifest = artifact_dir / "artifact-manifest.json"
    assert manifest.is_file()
    assert not manifest.stat().st_mode & stat.S_IWUSR
    assert (artifact_dir / "steering.outcome.json").is_file()


def test_a_steering_directory_that_cannot_be_sealed_still_runs_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every other turn blocks on this; the granted round is not every turn."""
    fixture = _coding_fixture(tmp_path)
    original = AgentAttemptArtifacts.finish

    def refuse(self, result, durable_result):
        if self.phase == "steering":
            raise OSError("artifact directory is read-only")
        return original(self, result, durable_result)

    monkeypatch.setattr(AgentAttemptArtifacts, "finish", refuse)

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        review = (
            MockAdapter()
            .queue(_blocking_review(fixture))
            .queue(_approving_review(fixture, store))
        )
        fix = MockAdapter().queue(_fixing_response(fixture.task))
        steering = _steering_adapter(1)
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=steering,
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=1, grant_budget=5
            ),
        ).run(fixture.context(store))
        attempts = store.list_agent_attempts(fixture.task.id)
        rows = store.list_steering_notes(fixture.borg.id)

    assert status is TaskRuntimeStatus.MERGING
    steering_attempt = next(
        item for item in attempts if item.phase == "steering"
    )
    assert steering_attempt.status is ExecutionAttemptStatus.COMPLETED
    assert (
        "read-only"
        in steering_attempt.result["_betterborg"]["artifact_seal_error"]
    )
    # The note still reached the fix, which is what the round was granted for.
    assert len(rows) == 1
    assert f"## Steering note\n\n{rows[0].note}" in _fix_prompts(fix)[0]


def test_a_fix_that_cannot_find_its_borg_runs_unsteered_rather_than_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Arriving without a note is never a reason to hold the task.

    Only the review is held to finding the Borg its verdict is recorded under.
    A fix that cannot is one more way to arrive unsteered, and the round it was
    granted still runs.
    """
    fixture = _coding_fixture(tmp_path)
    original = SqliteStore.get_execution_run
    lookups: list[int] = []

    def vanishing(self, run_id):
        # Gone for the fix's lookup alone: the review found it and recorded its
        # verdict under it, and every later round finds it again.
        lookups.append(1)
        return None if len(lookups) == 2 else original(self, run_id)

    monkeypatch.setattr(SqliteStore, "get_execution_run", vanishing)

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        review = (
            MockAdapter()
            .queue(_blocking_review(fixture))
            .queue(_approving_review(fixture, store))
        )
        fix = MockAdapter().queue(_fixing_response(fixture.task))
        steering = _steering_adapter(1)
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=steering,
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=1, grant_budget=5
            ),
        ).run(fixture.context(store))
        rows = store.list_steering_notes(fixture.borg.id)

    assert status is TaskRuntimeStatus.MERGING
    # The granted fix ran, and ran unsteered.
    assert len(fix.calls) == 1
    assert not any("Steering note" in prompt for prompt in _fix_prompts(fix))
    assert steering.calls == []
    assert rows == []


def test_the_guard_separates_a_checkout_it_found_dirty_from_one_it_watched_change(
    tmp_path: Path,
) -> None:
    """The discriminator has to come off the guard, not off the reader.

    Both conditions reach a caller as one exception class with a message from a
    private formatter, and a steering turn reads the difference before every
    other arm: found dirty falls back, watched change blocks the task. So the
    guard itself has to be the one asked, over a checkout it really inspected.
    """
    fixture = _coding_fixture(tmp_path)
    guard = PrimaryCheckoutGuard(fixture.repository)

    guard.before_phase("task-1", "steering")
    (fixture.repository / "README.md").write_text("# Written\n", encoding="utf-8")
    with pytest.raises(PrimaryCheckoutContaminationError) as watched:
        guard.after_phase("task-1", "steering")

    # Dirty on entry to a phase of its own, with nothing to blame it on.
    with pytest.raises(PrimaryCheckoutContaminationError) as found:
        PrimaryCheckoutGuard(fixture.repository).before_phase("task-2", "steering")

    assert checkout_was_changed(watched.value) is True
    assert checkout_was_changed(found.value) is False
    assert CheckoutCondition.CHANGED.value in str(watched.value)
    assert CheckoutCondition.DIRTY.value in str(found.value)


def test_a_blank_steering_model_is_refused_like_the_models_beside_it() -> None:
    """A stage configured to nothing is a turn with no model to run on."""
    with pytest.raises(ValueError, match="steering model must not be empty"):
        HostReviewFixConfig(review_model="review-model", steering_model="   ")


def test_a_checkout_that_could_not_be_read_is_not_read_as_a_write(
    tmp_path: Path,
) -> None:
    """The third condition is the one that must not block the task.

    A turn whose checkout could not be inspected has not been shown to have
    written anything, so it falls back like every other failure to produce a
    note. Asked of the guard rather than of a hand-built error, because what
    decides this is the condition the guard attaches.
    """
    outside = tmp_path / "not-a-repository"
    outside.mkdir()

    with pytest.raises(PrimaryCheckoutContaminationError) as unreadable:
        PrimaryCheckoutGuard(outside).before_phase("task-1", "steering")

    assert checkout_was_changed(unreadable.value) is False
    assert CheckoutCondition.UNREADABLE.value in str(unreadable.value)


def test_only_a_completed_review_is_a_rounds_own_account_of_itself() -> None:
    """Three filters decide what the steering turn reads as a round's decision.

    A review cancelled or failed at one round still carries that round's number
    and a summary of how it ended, and the attempt that replaced it carries
    them too — so an unfiltered read names two deciders for one round. The
    other two filters keep out the turns that answered rather than decided.
    """

    @dataclass(frozen=True)
    class _Attempt:
        phase: str
        status: ExecutionAttemptStatus
        review_round: int
        summary: str | None

    attempts = [
        _Attempt("coding", ExecutionAttemptStatus.COMPLETED, 0, "Implemented."),
        _Attempt("review", ExecutionAttemptStatus.CANCELLED, 0, "Interrupted."),
        _Attempt("review", ExecutionAttemptStatus.COMPLETED, 0, "Two things."),
        _Attempt("fix", ExecutionAttemptStatus.COMPLETED, 1, "Fixed one."),
        _Attempt("review", ExecutionAttemptStatus.FAILED, 1, "Unreadable."),
        _Attempt("review", ExecutionAttemptStatus.COMPLETED, 1, "Still open."),
        _Attempt("steering", ExecutionAttemptStatus.COMPLETED, 1, "Note ready."),
        _Attempt("review", ExecutionAttemptStatus.COMPLETED, 2, "   "),
    ]

    class _Store:
        @staticmethod
        def list_agent_attempts(_task_id):
            return attempts

    class _Claim:
        task_id = uuid4()

    class _Context:
        store = _Store()
        claim = _Claim()

    assert _review_round_summaries(_Context()) == [
        (1, "Two things."),
        (2, "Still open."),
    ]


def test_a_closing_review_leaves_the_fix_prompt_alone(tmp_path: Path) -> None:
    """A fix that already has the findings in front of it needs no note."""
    fixture = _coding_fixture(tmp_path)

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)

        def closing(_spec):
            rows = store.list_execution_ledger_findings(fixture.task.id)
            return _review_payload(
                fixture.task,
                status="issues_found",
                resolved=[
                    str(_ledger_row_for(rows, _BROAD_PATTERN).id),
                    str(_ledger_row_for(rows, _UNBOUNDED_TIMEOUT).id),
                ],
                findings=[_finding(_UNNAMED_FIXTURE)],
            )

        review = (
            MockAdapter()
            .queue(
                MockResponse(
                    payload=_review_payload(
                        fixture.task,
                        status="issues_found",
                        findings=[_BROAD_PATTERN, _UNBOUNDED_TIMEOUT],
                    )
                )
            )
            .queue(MockResponse(dynamic=closing))
            .queue(_approving_review(fixture, store))
        )
        fix = (
            MockAdapter()
            .queue(_fixing_response(fixture.task))
            .queue(_fixing_response(fixture.task))
        )
        steering = _steering_adapter(2)
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=steering,
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=2, grant_budget=5
            ),
        ).run(fixture.context(store))
        recorded = _assessments(store, fixture)
        rows = store.list_steering_notes(fixture.borg.id)

    assert status is TaskRuntimeStatus.MERGING
    assert [item.converging for item in recorded] == [False, True, True]
    assert steering.calls == []
    assert rows == []
    assert not any("Steering note" in prompt for prompt in _fix_prompts(fix))


def test_a_tasks_steering_reads_its_own_verdicts_and_no_other_tasks(
    tmp_path: Path,
) -> None:
    """The task is the only thing that separates two tasks' rounds.

    Every task of one execution run shares a Borg and a loop, so a verdict read
    across the run takes the furthest-argued task's and steers a round that is
    still inside its own minimum — on an argument it never had.
    """
    fixture = _coding_fixture(tmp_path)

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        # Another task of the same run, four rounds into a stuck argument.
        store.record_review_assessment(
            ReviewAssessment(
                borg_id=fixture.borg.id,
                loop=TASK_REVIEW_LOOP,
                task_id=uuid4(),
                round=4,
                minimum=2,
                converging=False,
                open_findings=3,
                refunded=False,
            )
        )
        review = (
            MockAdapter()
            .queue(_blocking_review(fixture))
            .queue(_approving_review(fixture, store))
        )
        fix = MockAdapter().queue(_fixing_response(fixture.task))
        steering = _steering_adapter(1)
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=steering,
            config=HostReviewFixConfig(
                review_model="review-model",
                steering_model="steering-model",
                review_passes=2,
                grant_budget=5,
            ),
        ).run(fixture.context(store))
        rows = store.list_steering_notes(fixture.borg.id)

    assert status is TaskRuntimeStatus.MERGING
    # This task's first round is inside its own minimum, whatever the other
    # task's argument has reached.
    assert steering.calls == []
    assert rows == []
    assert not any("Steering note" in call.user_prompt for call in fix.calls)


def test_an_operator_stopping_the_run_leaves_the_task_at_the_same_round(
    tmp_path: Path,
) -> None:
    """Nothing propagates out of a phase whose only handler covers its checks.

    A cancelled steering turn is recorded as cancelled and the phase returns
    the status it came in with, so the task is claimable at the round it was
    steering.
    """
    fixture = _coding_fixture(tmp_path)
    cancel = CancellationToken()

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        review = MockAdapter().queue(_blocking_review(fixture))
        fix = MockAdapter()
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=CancellingAgent(stops=True, cancel=cancel),
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=1, grant_budget=5
            ),
        ).run(fixture.context(store, cancel=cancel))
        runtime = store.get_task_runtime(fixture.task.id)
        attempts = store.list_agent_attempts(fixture.task.id)
        rows = store.list_steering_notes(fixture.borg.id)

    assert status is TaskRuntimeStatus.FIX
    assert runtime is not None and runtime.review_round == 1
    assert runtime.status is TaskRuntimeStatus.FIX
    assert fix.calls == []
    # No note resolved, so no row was written and the resumed round writes one.
    assert rows == []
    steering_attempts = [item for item in attempts if item.phase == "steering"]
    assert [item.status for item in steering_attempts] == [
        ExecutionAttemptStatus.CANCELLED
    ]
    # Read as the operator's stop rather than as the other thing a cancelled
    # status covers, and named for the phase that was stopped.
    assert (
        steering_attempts[0].result["_betterborg"]["outcome_reason"]
        == "steering agent was interrupted"
    )


def test_a_note_written_before_the_stop_reached_it_is_still_recorded(
    tmp_path: Path,
) -> None:
    """A stop does not un-write a note the turn had already written.

    The row is what spares the re-entered round a second turn, so a note that
    resolved before the stop arrived is recorded on the way out even though
    the round it was written for does not run now.
    """
    fixture = _coding_fixture(tmp_path)
    cancel = CancellationToken()

    def note_then_stop(_spec):
        cancel.cancel()
        return {"note": "Answer the timeout first.", "confidence": "high"}

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        review = MockAdapter().queue(_blocking_review(fixture))
        fix = MockAdapter()
        steering = MockAdapter().queue(MockResponse(dynamic=note_then_stop))
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=steering,
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=1, grant_budget=5
            ),
        ).run(fixture.context(store, cancel=cancel))
        runtime = store.get_task_runtime(fixture.task.id)
        rows = store.list_steering_notes(fixture.borg.id)
        attempts = store.list_agent_attempts(fixture.task.id)

    # The operator's stop still leaves the task where it was.
    assert status is TaskRuntimeStatus.FIX
    assert runtime is not None and runtime.review_round == 1
    assert fix.calls == []
    # And the note it had already written is on the record for the re-entry.
    assert [(row.round, row.source, row.note) for row in rows] == [
        (1, "agent", "Answer the timeout first.")
    ]
    assert [
        item.status for item in attempts if item.phase == "steering"
    ] == [ExecutionAttemptStatus.COMPLETED]


def test_a_note_the_stopped_turn_never_finished_is_not_recorded(
    tmp_path: Path,
) -> None:
    """Only the turn's own note survives a stop, and a failed turn has none.

    A failed result can still carry a payload — the native adapters return one
    when they cannot write the result file — so a reader that takes the
    payload without reading the status records a note the turn never
    delivered, and the re-entered round runs on it.
    """
    fixture = _coding_fixture(tmp_path)
    cancel = CancellationToken()

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        review = MockAdapter().queue(_blocking_review(fixture))
        fix = MockAdapter()
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=UnpersistedNoteAgent(
                note="Answer the timeout first.", cancel=cancel
            ),
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=1, grant_budget=5
            ),
        ).run(fixture.context(store, cancel=cancel))
        runtime = store.get_task_runtime(fixture.task.id)
        rows = store.list_steering_notes(fixture.borg.id)

    assert status is TaskRuntimeStatus.FIX
    assert runtime is not None and runtime.review_round == 1
    assert fix.calls == []
    assert rows == []


def test_a_note_the_stopped_turn_was_unsure_of_is_not_recorded(
    tmp_path: Path,
) -> None:
    """The bar a steered round applies still applies on the way out.

    A row kept through a stop is what the re-entered round runs on without a
    second turn, so a note the turn was unsure of would reach a fixer through
    the one path where no round is left to reject it.
    """
    fixture = _coding_fixture(tmp_path)
    cancel = CancellationToken()

    def note_then_stop(_spec):
        cancel.cancel()
        return {"note": "Answer the timeout first.", "confidence": "medium"}

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        review = MockAdapter().queue(_blocking_review(fixture))
        fix = MockAdapter()
        steering = MockAdapter().queue(MockResponse(dynamic=note_then_stop))
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=steering,
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=1, grant_budget=5
            ),
        ).run(fixture.context(store, cancel=cancel))
        runtime = store.get_task_runtime(fixture.task.id)
        rows = store.list_steering_notes(fixture.borg.id)
        attempts = store.list_agent_attempts(fixture.task.id)

    assert status is TaskRuntimeStatus.FIX
    assert runtime is not None and runtime.review_round == 1
    assert fix.calls == []
    # The turn resolved, so this is the bar and not the stop: nothing is
    # recorded, and the re-entry is free to try for a note again.
    assert rows == []
    assert [
        item.status for item in attempts if item.phase == "steering"
    ] == [ExecutionAttemptStatus.COMPLETED]


def test_a_steering_turn_that_raised_under_a_stopped_run_spends_no_more_turns(
    tmp_path: Path,
) -> None:
    """A stopped run reaches this classification as any of its failures.

    Read as one of them the phase would fall back and spend the fix turn the
    operator asked it not to spend.
    """
    fixture = _coding_fixture(tmp_path)
    cancel = CancellationToken()

    def stop_and_raise(_spec):
        cancel.cancel()
        raise RuntimeError("the note turn died with the run")

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        review = MockAdapter().queue(_blocking_review(fixture))
        fix = MockAdapter()
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=MockAdapter().queue(
                MockResponse(dynamic=stop_and_raise)
            ),
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=1, grant_budget=5
            ),
        ).run(fixture.context(store, cancel=cancel))
        runtime = store.get_task_runtime(fixture.task.id)
        rows = store.list_steering_notes(fixture.borg.id)

    assert status is TaskRuntimeStatus.FIX
    assert runtime is not None and runtime.status is TaskRuntimeStatus.FIX
    assert runtime.review_round == 1
    assert fix.calls == []
    assert rows == []


def test_a_steering_turn_its_adapter_gave_up_on_runs_the_fix_anyway(
    tmp_path: Path,
) -> None:
    """An optional turn must not drain every task in flight over a hiccup.

    The adapters report an operator's stop and their own exhausted transient
    retries with one status, and the driver turns the second into a run-wide
    stop for a turn whose work the task needs. This one is not that turn.
    """
    fixture = _coding_fixture(tmp_path)
    cancel = CancellationToken()

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        review = (
            MockAdapter()
            .queue(_blocking_review(fixture))
            .queue(_approving_review(fixture, store))
        )
        fix = MockAdapter().queue(_fixing_response(fixture.task))
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=CancellingAgent(stops=False),
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=1, grant_budget=5
            ),
        ).run(fixture.context(store, cancel=cancel))
        rows = store.list_steering_notes(fixture.borg.id)
        attempts = store.list_agent_attempts(fixture.task.id)

    assert status is TaskRuntimeStatus.MERGING
    assert cancel.is_set() is False
    assert [(row.round, row.source) for row in rows] == [(1, "assembled")]
    assert rows[0].note in _fix_prompts(fix)[0]
    # What the note says, and not only that the row and the prompt agree: both
    # are written from one string, so comparing them holds for any content.
    assert (
        f"- {_UNBOUNDED_TIMEOUT} (blocker, first raised in round 1 "
        "and open for 1 round)"
    ) in rows[0].note
    # The adapter's own account of what it gave up on, and not the wording an
    # operator's stop would have carried.
    assert [
        item.result["_betterborg"]["outcome_reason"]
        for item in attempts
        if item.phase == "steering"
    ] == ["transient provider failures exhausted the retry budget"]


def test_a_steering_turn_the_provider_refused_runs_the_fix_on_the_fallback(
    tmp_path: Path,
) -> None:
    """A turn that reports its failure is the shape the providers report in.

    They return a failed result where the machinery around them raises, so a
    non-zero exit and a rejected schema reach the classification as a status
    and not as an exception, carrying the provider's own account of it.
    """
    fixture = _coding_fixture(tmp_path)

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        review = (
            MockAdapter()
            .queue(_blocking_review(fixture))
            .queue(_approving_review(fixture, store))
        )
        fix = MockAdapter().queue(_fixing_response(fixture.task))
        steering = MockAdapter().queue(
            MockResponse(exit_code=1, error="provider refused the request")
        )
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=steering,
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=1, grant_budget=5
            ),
        ).run(fixture.context(store, cancel=CancellationToken()))
        rows = store.list_steering_notes(fixture.borg.id)
        attempts = store.list_agent_attempts(fixture.task.id)

    assert status is TaskRuntimeStatus.MERGING
    assert [(row.round, row.source) for row in rows] == [(1, "assembled")]
    assert rows[0].note in _fix_prompts(fix)[0]
    steering_attempts = [item for item in attempts if item.phase == "steering"]
    assert [item.status for item in steering_attempts] == [
        ExecutionAttemptStatus.FAILED
    ]
    # What the provider said, and not the bar's account of a missing note.
    assert (
        steering_attempts[0].result["_betterborg"]["outcome_reason"]
        == "provider refused the request"
    )


@pytest.mark.parametrize("failure", ["unwritable-artifacts", "dirty-checkout"])
def test_the_drivers_own_failure_paths_fall_back_rather_than_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """The task's work is not what failed, and the fix is still worth running.

    Each of these ends a turn the task depends on, which is right for the
    driver's own turns and wrong for an optional one.
    """
    fixture = _coding_fixture(tmp_path)

    if failure == "unwritable-artifacts":
        write_text = AgentAttemptArtifacts.write_text

        def refuse(self, name, content):
            if self.phase == "steering":
                raise OSError("no space left on device")
            return write_text(self, name, content)

        monkeypatch.setattr(AgentAttemptArtifacts, "write_text", refuse)
    else:
        before_phase = PrimaryCheckoutGuard.before_phase

        def dirty(self, task_ref, phase_name):
            if phase_name == "steering":
                raise PrimaryCheckoutContaminationError(
                    "primary checkout was dirty before it started",
                    condition=CheckoutCondition.DIRTY,
                )
            return before_phase(self, task_ref, phase_name)

        monkeypatch.setattr(PrimaryCheckoutGuard, "before_phase", dirty)

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        review = (
            MockAdapter()
            .queue(_blocking_review(fixture))
            .queue(_approving_review(fixture, store))
        )
        fix = MockAdapter().queue(_fixing_response(fixture.task))
        steering = _steering_adapter(1)
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=steering,
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=1, grant_budget=5
            ),
        ).run(fixture.context(store))
        rows = store.list_steering_notes(fixture.borg.id)
        runtime = store.get_task_runtime(fixture.task.id)

    assert status is TaskRuntimeStatus.MERGING
    assert runtime is not None and runtime.status is TaskRuntimeStatus.MERGING
    assert [(row.round, row.source) for row in rows] == [(1, "assembled")]
    assert rows[0].note in _fix_prompts(fix)[0]


def test_a_steering_turn_that_wrote_to_the_task_worktree_blocks_the_task(
    tmp_path: Path,
) -> None:
    """Writing is the one thing a steering turn was built not to do.

    Restoring the tree instead is not available: the Git this driver runs
    through withholds the destructive flags as its stated contract, so the
    comparison is a tripwire with a blocked task behind it.
    """
    fixture = _coding_fixture(tmp_path)

    def write(spec):
        (spec.cwd / "feature.txt").write_text("steered\n", encoding="utf-8")
        return MockResponse(
            payload={"note": "Answer the timeout first.", "confidence": "high"}
        )

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        review = MockAdapter().queue(_blocking_review(fixture))
        fix = MockAdapter()
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=MockAdapter().queue(MockResponse(dynamic=write)),
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=1, grant_budget=5
            ),
        ).run(fixture.context(store))
        runtime = store.get_task_runtime(fixture.task.id)
        rows = store.list_steering_notes(fixture.borg.id)

    assert status is TaskRuntimeStatus.BLOCKED
    assert runtime is not None
    assert runtime.state_reason == "steering agent modified the task worktree"
    # The edits never reached the fix, because the fix never ran.
    assert fix.calls == []
    assert rows == []


def test_a_fix_is_not_steered_by_a_note_its_turn_was_unsure_of(
    tmp_path: Path,
) -> None:
    """The bar is the same one the planning loops hold, and this half pays more.

    A merely plausible note misdirects a round that costs a fix turn and the
    review that judges it, where the assembled one leaves the fixer exactly
    where an unsteered round would have.
    """
    fixture = _coding_fixture(tmp_path)

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        review = (
            MockAdapter()
            .queue(_blocking_review(fixture))
            .queue(_approving_review(fixture, store))
        )
        fix = MockAdapter().queue(_fixing_response(fixture.task))
        steering = _steering_adapter(1, confidence="medium")
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=steering,
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=1, grant_budget=5
            ),
        ).run(fixture.context(store))
        rows = store.list_steering_notes(fixture.borg.id)
        attempts = store.list_agent_attempts(fixture.task.id)

    assert status is TaskRuntimeStatus.MERGING
    assert len(steering.calls) == 1
    # The turn's own paragraph is not what the fix was handed.
    assert "Steering note 1." not in _fix_prompts(fix)[0]
    assert [(row.round, row.source) for row in rows] == [(1, "assembled")]
    assert f"## Steering note\n\n{rows[0].note}" in _fix_prompts(fix)[0]
    # The bar's own account of why, and not one of the failure reasons: the
    # turn resolved, and what it returned was not confident enough to use.
    assert [
        item.result["_betterborg"]["outcome_reason"]
        for item in attempts
        if item.phase == "steering"
    ] == ["steering note was not attached"]


@pytest.mark.parametrize("leaves", ["commit", "branch"])
def test_a_steering_turn_that_left_a_clean_tree_behind_it_still_blocks(
    tmp_path: Path, leaves: str
) -> None:
    """A tree can be left changed without being left dirty.

    Every other agent in this driver is told to commit its work, and a steering
    turn under a loosened sandbox can do the same — or leave the worktree on
    another branch. Either way the status comparison is silent, and unnoticed
    the commit is absorbed into the fix's own work and reviewed as the task's.
    """
    fixture = _coding_fixture(tmp_path)

    def commit(spec):
        (spec.cwd / "feature.txt").write_text("steered\n", encoding="utf-8")
        if leaves == "commit":
            _git(spec.cwd, "add", "feature.txt")
            _git(spec.cwd, "commit", "-m", "steering turn commit")
        else:
            _git(spec.cwd, "stash", "--include-untracked")
            _git(spec.cwd, "switch", "-c", "steering-wandered")
        return MockResponse(
            payload={"note": "Answer the timeout first.", "confidence": "high"}
        )

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        review = MockAdapter().queue(_blocking_review(fixture))
        fix = MockAdapter()
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=MockAdapter().queue(MockResponse(dynamic=commit)),
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=1, grant_budget=5
            ),
        ).run(fixture.context(store))
        runtime = store.get_task_runtime(fixture.task.id)
        rows = store.list_steering_notes(fixture.borg.id)

    assert status is TaskRuntimeStatus.BLOCKED
    assert runtime is not None
    assert runtime.state_reason == "steering agent modified the task worktree"
    # The commit never reached the fix, because the fix never ran.
    assert fix.calls == []
    assert rows == []


def test_a_steering_turn_that_wrote_and_then_stopped_the_run_still_blocks(
    tmp_path: Path,
) -> None:
    """A breach is read before a stop, or it outlives the run unattributed.

    Recorded as a stop, the task stays claimable, nothing names the write, and
    the next phase to meet the guard blocks on a dirty tree it did not make.
    """
    fixture = _coding_fixture(tmp_path)
    cancel = CancellationToken()

    def write_then_stop(spec):
        (spec.cwd / "feature.txt").write_text("steered\n", encoding="utf-8")
        cancel.cancel()
        return MockResponse(
            payload={"note": "Answer the timeout first.", "confidence": "high"}
        )

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        review = MockAdapter().queue(_blocking_review(fixture))
        fix = MockAdapter()
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=MockAdapter().queue(
                MockResponse(dynamic=write_then_stop)
            ),
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=1, grant_budget=5
            ),
        ).run(fixture.context(store, cancel=cancel))
        runtime = store.get_task_runtime(fixture.task.id)

    assert status is TaskRuntimeStatus.BLOCKED
    assert runtime is not None
    assert runtime.state_reason == "steering agent modified the task worktree"
    assert fix.calls == []


@pytest.mark.parametrize(
    "raises", [False, True], ids=["guard-raises", "guard-attaches"]
)
def test_a_steering_turn_that_dirtied_the_primary_checkout_blocks_the_task(
    tmp_path: Path, raises: bool
) -> None:
    """Both arms block, including the one attached to a turn that raised first.

    The guard raises the same class for a checkout dirty on entry as for one a
    phase changed while it ran, so an implementer telling them apart by type
    would fall back on exactly the case that most needs blocking.
    """
    fixture = _coding_fixture(tmp_path)

    def contaminate(spec):
        (fixture.repository / "README.md").write_text(
            "# Fixture\nsteered\n", encoding="utf-8"
        )
        if raises:
            raise RuntimeError("the note turn fell over after writing")
        return MockResponse(
            payload={"note": "Answer the timeout first.", "confidence": "high"}
        )

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        review = MockAdapter().queue(_blocking_review(fixture))
        fix = MockAdapter()
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=MockAdapter().queue(
                MockResponse(dynamic=contaminate)
            ),
            config=HostReviewFixConfig(
                review_model="review-model", review_passes=1, grant_budget=5
            ),
        ).run(fixture.context(store))
        runtime = store.get_task_runtime(fixture.task.id)
        rows = store.list_steering_notes(fixture.borg.id)

    assert status is TaskRuntimeStatus.BLOCKED
    assert runtime is not None
    assert (runtime.state_reason or "").startswith(
        "steering agent changed the primary checkout"
    )
    assert fix.calls == []
    assert rows == []


def test_a_steered_round_re_entered_reuses_the_note_its_row_already_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One note per grant, however many times the round carrying it is re-entered.

    The restart here is inside the steered fix itself, which is the re-entry
    that would otherwise pay for a second turn.
    """
    fixture = _coding_fixture(tmp_path)

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        review = (
            MockAdapter()
            .queue(_blocking_review(fixture))
            .queue(_approving_review(fixture, store))
        )
        steering = MockAdapter().queue(
            MockResponse(
                payload={"note": "Answer the timeout first.", "confidence": "high"}
            )
        )
        config = HostReviewFixConfig(
            review_model="review-model", review_passes=1, grant_budget=5
        )
        append = store.append_agent_attempt

        def crash_before_the_fix(attempt, *args, **kwargs):
            if attempt.phase == "fix":
                raise RuntimeError("simulated restart inside the steered fix")
            return append(attempt, *args, **kwargs)

        monkeypatch.setattr(store, "append_agent_attempt", crash_before_the_fix)
        with pytest.raises(RuntimeError, match="simulated restart"):
            HostReviewFixPhase(
                fixture.repository,
                review,
                fix_adapter=MockAdapter(),
                steering_adapter=steering,
                config=config,
            ).run(fixture.context(store))
        monkeypatch.setattr(store, "append_agent_attempt", append)

        interrupted = store.get_task_runtime(fixture.task.id)
        assert interrupted is not None
        assert interrupted.status is TaskRuntimeStatus.FIX
        assert interrupted.review_round == 1
        assert len(steering.calls) == 1

        fix = MockAdapter().queue(_fixing_response(fixture.task))
        resumed = HostReviewFixPhase(
            fixture.repository,
            review,
            fix_adapter=fix,
            steering_adapter=steering,
            config=config,
        ).run(fixture.context(store))
        rows = store.list_steering_notes(fixture.borg.id)

    assert resumed is TaskRuntimeStatus.MERGING
    # The resumed round paid for no second turn and ran on the same note.
    assert len(steering.calls) == 1
    assert [(row.round, row.note) for row in rows] == [
        (1, "Answer the timeout first.")
    ]
    assert "Answer the timeout first." in _fix_prompts(fix)[0]


def test_the_execution_note_row_joins_the_step_that_completes_its_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A round that recorded which note it used but never finished its turn is
    a round the next entry would re-run and charge twice.
    """
    fixture = _coding_fixture(tmp_path)

    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        review = MockAdapter().queue(_blocking_review(fixture))
        fix = MockAdapter()

        def refuse(_note):
            raise RuntimeError("the note row could not be written")

        monkeypatch.setattr(store, "record_steering_note", refuse)
        with pytest.raises(RuntimeError, match="note row"):
            HostReviewFixPhase(
                fixture.repository,
                review,
                fix_adapter=fix,
                steering_adapter=_steering_adapter(1),
                config=HostReviewFixConfig(
                    review_model="review-model", review_passes=1, grant_budget=5
                ),
            ).run(fixture.context(store))
        monkeypatch.undo()

        rows = store.list_steering_notes(fixture.borg.id)
        attempts = store.list_agent_attempts(fixture.task.id)

    assert rows == []
    assert fix.calls == []
    assert [
        item.status for item in attempts if item.phase == "steering"
    ] == [ExecutionAttemptStatus.RUNNING]
