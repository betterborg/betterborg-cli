"""Real-Git contracts for guarded project-base task merging."""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import pytest
from test_host_coding import (
    CodingFixture,
    _coding_fixture,
    _completed_payload,
    _prepare_review,
    _review_payload,
)

from betterborg_cli.agent_runtime import MockAdapter, MockResponse
from betterborg_cli.host_execution import (
    HostMergeConfig,
    HostMergePhase,
    HostReviewFixConfig,
    HostReviewFixPhase,
    SafeGit,
    UnsafeGitError,
)
from betterborg_cli.host_execution._locking import path_lock
from betterborg_cli.store import SqliteStore, TaskRuntimeStatus


class RecordingLock:
    """Observable wrapper around the shared repository lock input."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.entries = 0

    def __enter__(self) -> None:
        self._lock.acquire()
        self.entries += 1

    def __exit__(self, *args: object) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()

    def __call__(self) -> AbstractContextManager[None]:
        return self


def _git(root: Path, *arguments: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=check,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _project_branch(fixture: CodingFixture) -> str:
    return f"project/{fixture.borg.name}"


def _approved_merge_fixture(tmp_path: Path) -> CodingFixture:
    fixture = _coding_fixture(tmp_path)
    review = MockAdapter().queue(
        MockResponse(payload=_review_payload(fixture.task, status="approved"))
    )
    with SqliteStore.open(fixture.database) as store:
        _prepare_review(fixture, store)
        coding_prompt = store.get_latest_generated_prompts(
            fixture.borg.repository_id
        )["coding"]
        store.append_generated_prompt(
            repository_id=fixture.borg.repository_id,
            analysis_id=coding_prompt.analysis_id,
            role="merge",
            body_md="You are the generated merge conflict resolver.\n",
        )
        status = HostReviewFixPhase(
            fixture.repository,
            review,
            config=HostReviewFixConfig(review_model="review-model"),
        ).run(fixture.context(store))
    assert status is TaskRuntimeStatus.MERGING
    return fixture


def _advance_project_base(
    fixture: CodingFixture, filename: str, content: str
) -> str:
    (fixture.repository / filename).write_text(content, encoding="utf-8")
    _git(fixture.repository, "add", filename)
    _git(fixture.repository, "commit", "--quiet", "-m", "advance project base")
    project_branch = _project_branch(fixture)
    previous = _git(fixture.repository, "rev-parse", project_branch)
    destination = _git(fixture.repository, "rev-parse", "main")
    _git(
        fixture.repository,
        "update-ref",
        f"refs/heads/{project_branch}",
        destination,
        previous,
    )
    return _git(fixture.repository, "rev-parse", project_branch)


def _phase(
    fixture: CodingFixture,
    adapter: MockAdapter,
    repository_lock: Callable[[], AbstractContextManager[None]],
) -> HostMergePhase:
    return HostMergePhase(
        fixture.repository,
        adapter,
        config=HostMergeConfig(model="merge-model"),
        repository_lock=repository_lock,
    )


def test_clean_merge_produces_tip_without_agent_or_base_advance(
    tmp_path: Path,
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    base_commit = _advance_project_base(fixture, "base.txt", "base progress\n")
    adapter = MockAdapter()
    repository_lock = RecordingLock()

    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, adapter, repository_lock).run(
            fixture.context(store)
        )
        runtime = store.get_task_runtime(fixture.task.id)
        merge_events = store.list_task_execution_events(fixture.task.id)
        phases = [
            attempt.phase
            for attempt in store.list_agent_attempts(fixture.task.id)
        ]

    assert result.status is TaskRuntimeStatus.MERGING
    assert result.tip is not None and not result.tip.agent_used
    assert result.tip.project_branch == _project_branch(fixture)
    assert result.tip.base_commit == base_commit
    assert adapter.calls == []
    assert phases == ["coding", "review"]
    merge_event_kinds = [
        event.kind for event in merge_events if event.kind.startswith("merge.")
    ]
    assert merge_event_kinds == [
        "merge.started",
        "merge.completed",
    ]
    completion = merge_events[-1].payload
    assert completion["approved_commit"] == result.tip.approved_commit
    assert completion["base_commit"] == base_commit
    assert completion["commit_sha"] == result.tip.commit_sha
    assert runtime is not None and runtime.status is TaskRuntimeStatus.MERGING
    assert repository_lock.entries == 1
    assert (
        _git(fixture.repository, "rev-parse", _project_branch(fixture))
        == base_commit
    )
    assert SafeGit(Path(runtime.worktree_path)).is_ancestor(
        base_commit, result.tip.commit_sha
    )


def test_completed_clean_merge_resumes_without_agent(
    tmp_path: Path,
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    base_commit = _advance_project_base(fixture, "base.txt", "base progress\n")
    adapter = MockAdapter()

    with SqliteStore.open(fixture.database) as store:
        phase = _phase(fixture, adapter, RecordingLock())
        initial = phase.run(fixture.context(store))
        resumed = phase.run(fixture.context(store))
        attempts = store.list_agent_attempts(fixture.task.id)

    assert initial.tip is not None and not initial.tip.agent_used
    assert resumed.status is TaskRuntimeStatus.MERGING
    assert resumed.tip is not None and not resumed.tip.agent_used
    assert resumed.tip.commit_sha == initial.tip.commit_sha
    assert resumed.tip.base_commit == base_commit
    assert adapter.calls == []
    assert [attempt.phase for attempt in attempts] == ["coding", "review"]


def test_clean_merge_resumes_after_commit_before_completion_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    base_commit = _advance_project_base(fixture, "base.txt", "base progress\n")
    with SqliteStore.open(fixture.database) as store:
        runtime = store.get_task_runtime(fixture.task.id)
    assert runtime is not None
    worktree = Path(runtime.worktree_path)
    original_run = SafeGit.run

    def interrupt_after_merge(self, arguments, **kwargs):
        result = original_run(self, arguments, **kwargs)
        if self.cwd == worktree and arguments[0] == "merge":
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(SafeGit, "run", interrupt_after_merge)
    with SqliteStore.open(fixture.database) as store:
        with pytest.raises(KeyboardInterrupt):
            _phase(fixture, MockAdapter(), RecordingLock()).run(
                fixture.context(store)
            )
        started = store.list_task_execution_events(
            fixture.task.id, kind="merge.started"
        )
        completed = store.list_task_execution_events(
            fixture.task.id, kind="merge.completed"
        )

    merged_commit = _git(worktree, "rev-parse", "HEAD")
    assert len(started) == 1
    assert started[0].payload["expected_commit"] == merged_commit
    assert completed == []

    monkeypatch.setattr(SafeGit, "run", original_run)
    adapter = MockAdapter()
    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, adapter, RecordingLock()).run(
            fixture.context(store)
        )
        completed = store.list_task_execution_events(
            fixture.task.id, kind="merge.completed"
        )

    assert result.status is TaskRuntimeStatus.MERGING
    assert result.tip is not None and not result.tip.agent_used
    assert result.tip.commit_sha == merged_commit
    assert result.tip.base_commit == base_commit
    assert adapter.calls == []
    assert len(completed) == 1


def test_unreviewed_commit_is_not_accepted_as_resumable_merge_tip(
    tmp_path: Path,
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    with SqliteStore.open(fixture.database) as store:
        runtime = store.get_task_runtime(fixture.task.id)
    assert runtime is not None
    worktree = Path(runtime.worktree_path)
    (worktree / "unreviewed.txt").write_text("not reviewed\n", encoding="utf-8")
    _git(worktree, "add", "unreviewed.txt")
    _git(worktree, "commit", "--quiet", "-m", "unreviewed change")
    adapter = MockAdapter()

    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, adapter, RecordingLock()).run(
            fixture.context(store)
        )

    assert result.status is TaskRuntimeStatus.BLOCKED
    assert result.tip is None
    assert "approved task commit no longer matches" in result.reason
    assert adapter.calls == []


def test_forged_two_parent_commit_is_not_a_resumable_clean_merge(
    tmp_path: Path,
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    base_commit = _advance_project_base(fixture, "base.txt", "base progress\n")
    with SqliteStore.open(fixture.database) as store:
        runtime = store.get_task_runtime(fixture.task.id)
        approved_commit = store.list_agent_attempts(fixture.task.id)[-1].result[
            "_betterborg"
        ]["commit_sha"]
    assert runtime is not None and runtime.branch is not None
    worktree = Path(runtime.worktree_path)
    (worktree / "unreviewed.txt").write_text("not reviewed\n", encoding="utf-8")
    _git(worktree, "add", "unreviewed.txt")
    tree = _git(worktree, "write-tree")
    forged = _git(
        worktree,
        "commit-tree",
        tree,
        "-p",
        approved_commit,
        "-p",
        base_commit,
        "-m",
        "forged merge",
    )
    _git(
        worktree,
        "update-ref",
        f"refs/heads/{runtime.branch}",
        forged,
        approved_commit,
    )

    adapter = MockAdapter()
    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, adapter, RecordingLock()).run(
            fixture.context(store)
        )

    assert result.status is TaskRuntimeStatus.BLOCKED
    assert result.tip is None
    assert "approved task commit no longer matches" in result.reason
    assert adapter.calls == []


def test_unattested_active_merge_is_preserved_and_blocked(
    tmp_path: Path,
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    base_commit = _advance_project_base(
        fixture, "feature.txt", "project version\n"
    )
    with SqliteStore.open(fixture.database) as store:
        runtime = store.get_task_runtime(fixture.task.id)
    assert runtime is not None
    worktree = Path(runtime.worktree_path)
    _git(worktree, "merge", "--no-edit", base_commit, check=False)
    assert _git(worktree, "rev-parse", "MERGE_HEAD") == base_commit

    adapter = MockAdapter()
    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, adapter, RecordingLock()).run(
            fixture.context(store)
        )

    assert result.status is TaskRuntimeStatus.BLOCKED
    assert result.tip is None
    assert "lacks a durable host attestation" in result.reason
    assert adapter.calls == []
    assert _git(worktree, "rev-parse", "MERGE_HEAD") == base_commit


def test_host_attested_active_merge_resumes_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    base_commit = _advance_project_base(
        fixture, "feature.txt", "project version\n"
    )
    with SqliteStore.open(fixture.database) as store:
        runtime = store.get_task_runtime(fixture.task.id)
    assert runtime is not None
    worktree = Path(runtime.worktree_path)
    original_run = SafeGit.run

    def interrupt_after_merge(self, arguments, **kwargs):
        result = original_run(self, arguments, **kwargs)
        if self.cwd == worktree and arguments[0] == "merge":
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(SafeGit, "run", interrupt_after_merge)
    with SqliteStore.open(fixture.database) as store:
        with pytest.raises(KeyboardInterrupt):
            _phase(fixture, MockAdapter(), RecordingLock()).run(
                fixture.context(store)
            )
        merge_events = store.list_task_execution_events(
            fixture.task.id, kind="merge.started"
        )
    assert len(merge_events) == 1
    assert merge_events[0].payload["base_commit"] == base_commit
    assert _git(worktree, "rev-parse", "MERGE_HEAD") == base_commit

    monkeypatch.setattr(SafeGit, "run", original_run)

    def resolve(spec):
        (spec.cwd / "feature.txt").write_text("resolved\n", encoding="utf-8")
        _git(spec.cwd, "add", "feature.txt")
        _git(spec.cwd, "commit", "--quiet", "-m", "resolve project merge")
        return MockResponse(payload=_completed_payload(fixture.task))

    adapter = MockAdapter().queue(MockResponse(dynamic=resolve))
    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, adapter, RecordingLock()).run(
            fixture.context(store)
        )

    assert result.status is TaskRuntimeStatus.MERGING
    assert result.tip is not None and result.tip.agent_used
    assert result.tip.base_commit == base_commit
    assert len(adapter.calls) == 1


def test_conflict_invokes_agent_outside_lock_and_persists_merge_attempt(
    tmp_path: Path,
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    base_commit = _advance_project_base(
        fixture, "feature.txt", "project version\n"
    )
    repository_lock = RecordingLock()

    def resolve(spec):
        assert not repository_lock.locked()
        assert _git(spec.cwd, "diff", "--name-only", "--diff-filter=U") == (
            "feature.txt"
        )
        (spec.cwd / "feature.txt").write_text(
            "implemented\nproject version\n", encoding="utf-8"
        )
        _git(spec.cwd, "add", "feature.txt")
        _git(spec.cwd, "commit", "--quiet", "-m", "resolve project merge")
        return MockResponse(payload=_completed_payload(fixture.task))

    adapter = MockAdapter().queue(MockResponse(dynamic=resolve))
    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, adapter, repository_lock).run(
            fixture.context(store)
        )
        attempts = store.list_agent_attempts(fixture.task.id)
        runtime = store.get_task_runtime(fixture.task.id)

    assert result.status is TaskRuntimeStatus.MERGING
    assert result.tip is not None and result.tip.agent_used
    assert len(adapter.calls) == 1
    assert [attempt.phase for attempt in attempts] == [
        "coding",
        "review",
        "merge",
    ]
    assert attempts[-1].result["_betterborg"]["commit_sha"] == (
        result.tip.commit_sha
    )
    assert "feature.txt" in adapter.calls[0].user_prompt
    assert runtime is not None
    assert _git(Path(runtime.worktree_path), "status", "--porcelain") == ""
    assert (
        _git(fixture.repository, "rev-parse", _project_branch(fixture))
        == base_commit
    )


def test_conflict_verification_reacquires_real_path_lock_factory(
    tmp_path: Path,
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    _advance_project_base(fixture, "feature.txt", "project version\n")
    lock_entries = 0

    def repository_lock() -> AbstractContextManager[None]:
        nonlocal lock_entries
        lock_entries += 1
        return path_lock(tmp_path / "repository.lock")

    def resolve(spec):
        (spec.cwd / "feature.txt").write_text("resolved\n", encoding="utf-8")
        _git(spec.cwd, "add", "feature.txt")
        _git(spec.cwd, "commit", "--quiet", "-m", "resolve project merge")
        return MockResponse(payload=_completed_payload(fixture.task))

    adapter = MockAdapter().queue(MockResponse(dynamic=resolve))
    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, adapter, repository_lock).run(
            fixture.context(store)
        )

    assert result.status is TaskRuntimeStatus.MERGING
    assert result.tip is not None and result.tip.agent_used
    assert len(adapter.calls) == 1
    assert lock_entries == 2


def test_completed_conflict_merge_resumes_without_replaying_agent(
    tmp_path: Path,
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    _advance_project_base(fixture, "feature.txt", "project version\n")

    def resolve(spec):
        (spec.cwd / "feature.txt").write_text(
            "implemented\nproject version\n", encoding="utf-8"
        )
        _git(spec.cwd, "add", "feature.txt")
        _git(spec.cwd, "commit", "--quiet", "-m", "resolve project merge")
        return MockResponse(payload=_completed_payload(fixture.task))

    first = MockAdapter().queue(MockResponse(dynamic=resolve))
    with SqliteStore.open(fixture.database) as store:
        initial = _phase(fixture, first, RecordingLock()).run(
            fixture.context(store)
        )
        replay = MockAdapter()
        resumed = _phase(fixture, replay, RecordingLock()).run(
            fixture.context(store)
        )

    assert initial.tip is not None
    assert resumed.status is TaskRuntimeStatus.MERGING
    assert resumed.tip is not None and resumed.tip.agent_used
    assert resumed.tip.commit_sha == initial.tip.commit_sha
    assert replay.calls == []


def test_conflict_agent_moving_project_base_is_detected(
    tmp_path: Path,
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    base_commit = _advance_project_base(
        fixture, "feature.txt", "project version\n"
    )
    with SqliteStore.open(fixture.database) as store:
        approved_commit = store.list_agent_attempts(fixture.task.id)[-1].result[
            "_betterborg"
        ]["commit_sha"]
    repository_lock = RecordingLock()

    def resolve(spec):
        _git(
            spec.cwd,
            "update-ref",
            f"refs/heads/{_project_branch(fixture)}",
            approved_commit,
            base_commit,
        )
        (spec.cwd / "feature.txt").write_text("resolved\n", encoding="utf-8")
        _git(spec.cwd, "add", "feature.txt")
        _git(spec.cwd, "commit", "--quiet", "-m", "resolve project merge")
        return MockResponse(payload=_completed_payload(fixture.task))

    adapter = MockAdapter().queue(MockResponse(dynamic=resolve))
    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, adapter, repository_lock).run(
            fixture.context(store)
        )
        attempt = store.list_agent_attempts(fixture.task.id)[-1]

    assert result.status is TaskRuntimeStatus.BLOCKED
    assert result.tip is None
    assert "project base moved" in result.reason
    assert repository_lock.entries == 2
    assert _git(fixture.repository, "rev-parse", _project_branch(fixture)) == (
        approved_commit
    )
    assert attempt.result["_betterborg"]["commit_sha"] is None


def test_agent_claiming_completion_with_unresolved_paths_blocks_and_preserves(
    tmp_path: Path,
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    base_commit = _advance_project_base(
        fixture, "feature.txt", "project version\n"
    )
    adapter = MockAdapter().queue(
        MockResponse(payload=_completed_payload(fixture.task))
    )

    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, adapter, RecordingLock()).run(
            fixture.context(store)
        )
        runtime = store.get_task_runtime(fixture.task.id)

    assert result.status is TaskRuntimeStatus.BLOCKED
    assert result.tip is None
    assert runtime is not None and runtime.status is TaskRuntimeStatus.BLOCKED
    worktree = Path(runtime.worktree_path)
    assert worktree.is_dir()
    assert _git(worktree, "diff", "--name-only", "--diff-filter=U") == (
        "feature.txt"
    )
    assert "unresolved paths" in (runtime.state_reason or "")
    assert (
        _git(fixture.repository, "rev-parse", _project_branch(fixture))
        == base_commit
    )


def test_safe_git_denial_blocks_without_invoking_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    _advance_project_base(fixture, "base.txt", "base progress\n")
    with SqliteStore.open(fixture.database) as store:
        runtime = store.get_task_runtime(fixture.task.id)
    assert runtime is not None
    worktree = Path(runtime.worktree_path)
    original_run = SafeGit.run

    def deny_merge(self, arguments, **kwargs):
        if self.cwd == worktree and arguments[0] == "merge":
            raise UnsafeGitError("test policy denied merge")
        return original_run(self, arguments, **kwargs)

    monkeypatch.setattr(SafeGit, "run", deny_merge)
    adapter = MockAdapter()
    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, adapter, RecordingLock()).run(
            fixture.context(store)
        )
        runtime = store.get_task_runtime(fixture.task.id)

    assert result.status is TaskRuntimeStatus.BLOCKED
    assert adapter.calls == []
    assert runtime is not None and "policy denied" in (runtime.state_reason or "")
    assert worktree.is_dir()


def test_primary_checkout_contamination_blocks_after_clean_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    base_commit = _advance_project_base(fixture, "base.txt", "base progress\n")
    with SqliteStore.open(fixture.database) as store:
        runtime = store.get_task_runtime(fixture.task.id)
    assert runtime is not None
    worktree = Path(runtime.worktree_path)
    original_run = SafeGit.run

    def contaminate(self, arguments, **kwargs):
        if self.cwd == worktree and arguments[0] == "merge":
            (fixture.repository / "escaped.txt").write_text(
                "escaped\n", encoding="utf-8"
            )
        return original_run(self, arguments, **kwargs)

    monkeypatch.setattr(SafeGit, "run", contaminate)
    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, MockAdapter(), RecordingLock()).run(
            fixture.context(store)
        )
        runtime = store.get_task_runtime(fixture.task.id)

    assert result.status is TaskRuntimeStatus.BLOCKED
    assert runtime is not None and "primary checkout" in (
        runtime.state_reason or ""
    )
    assert worktree.is_dir()
    assert (fixture.repository / "escaped.txt").read_text() == "escaped\n"
    assert (
        _git(fixture.repository, "rev-parse", _project_branch(fixture))
        == base_commit
    )
