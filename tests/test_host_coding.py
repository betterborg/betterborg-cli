"""Coding-agent contracts for guarded materialized host worktrees."""

from __future__ import annotations

import stat
import subprocess
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from betterborg_cli.agent_runtime import (
    AgentArtifact,
    AgentUsage,
    BillingMode,
    CancellationToken,
    MockAdapter,
    MockResponse,
)
from betterborg_cli.host_execution import (
    HostCodingConfig,
    HostCodingPhase,
    HostEnvironmentManager,
    HostPreflightPlan,
    HostReviewFixConfig,
    HostReviewFixPhase,
    HostWorktreeManager,
    ScheduledTaskContext,
)
from betterborg_cli.planning import (
    approved_plan_digest,
    render_task_markdown,
    task_markdown_digest,
)
from betterborg_cli.repo_paths import RepoPaths, ensure_managed_gitignore
from betterborg_cli.store import (
    Borg,
    BorgState,
    ExecutionAttemptStatus,
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
        self, store: SqliteStore, *, cancel: CancellationToken | None = None
    ) -> ScheduledTaskContext:
        return ScheduledTaskContext(
            store=store,
            claim=self.claim,
            owner_token=self.owner_token,
            cancel=cancel or CancellationToken(),
            clock=utcnow,
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
    ensure_managed_gitignore(RepoPaths.discover(repository_root))
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
                f".borg/tasks/{borg.name}/{generation_id}/"
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
    durable_root = (
        repository_root / ".borg/tasks" / borg.name / str(generation.id)
    )

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
            generation.id, durable_root=durable_root
        )

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
            environment_files=(),
            executables=(),
            required_secret_names=(),
            compose_files=(),
            services=(),
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


def _committing_response(task: TaskRecord, *, usage: AgentUsage | None = None):
    def commit(spec):
        (spec.cwd / "feature.txt").write_text("implemented\n", encoding="utf-8")
        _git(spec.cwd, "add", "feature.txt")
        _git(spec.cwd, "commit", "--quiet", "-m", "add feature")
        transcript = spec.cwd / ".borg/state/provider-transcript.txt"
        transcript.write_text("immutable transcript\n", encoding="utf-8")
        return MockResponse(
            payload=_completed_payload(task),
            usage=usage,
            billing_mode=spec.billing_mode,
            artifacts=(AgentArtifact(transcript, kind="transcript"),),
        )

    return MockResponse(dynamic=commit)


def _review_payload(
    task: TaskRecord,
    *,
    status: str,
    findings: list[str] | None = None,
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
        "findings": findings or [],
    }


def _fixing_response(task: TaskRecord, *, usage: AgentUsage | None = None):
    def commit(spec):
        feature = spec.cwd / "feature.txt"
        feature.write_text(feature.read_text() + "fixed\n", encoding="utf-8")
        _git(spec.cwd, "add", "feature.txt")
        _git(spec.cwd, "commit", "--quiet", "-m", "fix review finding")
        return MockResponse(
            payload=_completed_payload(task),
            usage=usage,
            billing_mode=spec.billing_mode,
        )

    return MockResponse(dynamic=commit)


def _prepare_review(
    fixture: CodingFixture,
    store: SqliteStore,
    *,
    usage: AgentUsage | None = None,
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
        MockAdapter().queue(_committing_response(fixture.task, usage=usage)),
        config=HostCodingConfig(model="coding-model"),
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
            / ".borg/tasks"
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
            / ".borg/state/environment-materialization"
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
        ).run(fixture.context(store))
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
    assert attempts[1].result["findings"] == [finding]
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
    assert attempts[-1].result["findings"] == ["still failing after the fix"]
    assert len(fix.calls) == 1


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
