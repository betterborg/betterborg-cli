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
    VerifiedTaskInputs,
)
from betterborg_cli.host_execution.coding import (
    CODING_RESULT_SCHEMA,
    _render_user_prompt,
)
from betterborg_cli.host_execution.merge import _render_merge_prompt
from betterborg_cli.host_execution.review import (
    _render_fix_prompt,
    _render_review_prompt,
)
from betterborg_cli.planning import (
    approved_plan_digest,
    render_task_markdown,
    task_markdown_digest,
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


def test_review_pass_cap_blocks_after_persisting_last_findings(
    tmp_path: Path,
) -> None:
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
            ),
        ).run(fixture.context(store))
        runtime = store.get_task_runtime(fixture.task.id)
        attempts = store.list_agent_attempts(fixture.task.id)

    assert status is TaskRuntimeStatus.BLOCKED
    assert runtime is not None
    assert runtime.review_round == 2
    assert runtime.resume_phase == "review"
    assert runtime.state_reason == "review pass limit 2 reached"
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
                review_model="review-model", review_passes=1
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
                review_model="review-model", review_passes=2
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
                review_model="review-model", review_passes=3
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
                review_model="review-model", review_passes=2
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
        config = HostReviewFixConfig(review_model="review-model", review_passes=1)
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
