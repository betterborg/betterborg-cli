"""Real-Git contracts for guarded project-base task merging."""

from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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

from betterborg_cli.agent_runtime import (
    CancellationToken,
    MockAdapter,
    MockResponse,
    run_captured,
)
from betterborg_cli.host_execution import (
    HostMergeConfig,
    HostMergePhase,
    HostReviewFixConfig,
    HostReviewFixPhase,
    SafeGit,
    UnsafeGitError,
)
from betterborg_cli.host_execution._locking import path_lock
from betterborg_cli.progress import AgentActivity, AgentActivityKind
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


_IDENTITY_VARIABLES = (
    "EMAIL",
    "GIT_AUTHOR_NAME",
    "GIT_AUTHOR_EMAIL",
    "GIT_COMMITTER_NAME",
    "GIT_COMMITTER_EMAIL",
)


def _clear_environment_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave the repository's own configuration as the only source of identity.

    Git reads these variables ahead of configuration, and falls back to global
    and system configuration when the repository supplies nothing, so either
    would otherwise decide what these tests observe.
    """
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    for variable in _IDENTITY_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def _strip_git_identity(fixture: CodingFixture) -> None:
    """Leave the fixture repository with no identity Git may commit under."""
    _git(fixture.repository, "config", "--unset", "user.name")
    _git(fixture.repository, "config", "--unset", "user.email")
    # Without this Git invents an identity from the account and hostname,
    # which would make "no identity configured" mean different things on
    # different machines.
    _git(fixture.repository, "config", "user.useConfigOnly", "true")


def _advance_project_base_past_task_tip(fixture: CodingFixture) -> str:
    """Move the project base onto a commit that already contains the task tip.

    Merging then only moves the branch pointer, which is the shape in which
    Git writes no commit and so needs no identity.
    """
    with SqliteStore.open(fixture.database) as store:
        runtime = store.get_task_runtime(fixture.task.id)
    assert runtime is not None
    tip = _git(Path(runtime.worktree_path), "rev-parse", "HEAD")
    descendant = _git(
        fixture.repository,
        "commit-tree",
        f"{tip}^{{tree}}",
        "-p",
        tip,
        "-m",
        "advance project base past the task tip",
    )
    project_branch = _project_branch(fixture)
    _git(
        fixture.repository,
        "update-ref",
        f"refs/heads/{project_branch}",
        descendant,
        _git(fixture.repository, "rev-parse", project_branch),
    )
    return descendant


def _phase(
    fixture: CodingFixture,
    adapter: MockAdapter,
    repository_lock: Callable[[], AbstractContextManager[None]],
    *,
    cancel: CancellationToken | None = None,
    git: SafeGit | None = None,
) -> HostMergePhase:
    return HostMergePhase(
        fixture.repository,
        adapter,
        config=HostMergeConfig(model="merge-model"),
        repository_lock=repository_lock,
        cancel=cancel,
        git=git,
    )


def test_merge_phase_reuses_bound_git_and_reaps_cancelled_project_tip_probe(
    tmp_path: Path,
    real_process_harness,
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    cancel = CancellationToken()
    resistant = real_process_harness.resistant_argv("merge-project-tip-git")
    observed_tokens: list[CancellationToken | None] = []

    def runner(command, **kwargs):  # noqa: ANN001, ANN003
        arguments = tuple(command)
        if (
            arguments[1:3] == ("rev-parse", "--verify")
            and arguments[-1].startswith("refs/heads/project/")
        ):
            observed_tokens.append(kwargs.get("cancel"))
            return run_captured(resistant, **kwargs)
        return run_captured(command, **kwargs)

    git = SafeGit(
        fixture.repository,
        cancel=cancel,
        command_runner=runner,
    )
    with SqliteStore.open(fixture.database) as store:
        phase = _phase(
            fixture,
            MockAdapter(),
            RecordingLock(),
            cancel=cancel,
            git=git,
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(phase.run, fixture.context(store, cancel=cancel))
            real_process_harness.wait_for_marker(
                "merge-project-tip-git.child.pid"
            )
            cancel.cancel()
            with pytest.raises(KeyboardInterrupt):
                result.result(timeout=5)

    real_process_harness.assert_tree_absent("merge-project-tip-git")
    assert observed_tokens == [cancel]


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


def test_merge_commit_carries_the_configured_git_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    _clear_environment_identity(monkeypatch)
    _advance_project_base(fixture, "base.txt", "base progress\n")

    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, MockAdapter(), RecordingLock()).run(
            fixture.context(store)
        )

    assert result.tip is not None
    identity = _git(
        fixture.repository,
        "show",
        "--no-patch",
        "--format=%an%n%ae%n%cn%n%ce%n%P",
        result.tip.commit_sha,
    )
    name, email, committer, committer_email, parents = identity.splitlines()
    assert (name, email) == ("Betterborg Tests", "tests@betterborg.dev")
    assert (committer, committer_email) == (name, email)
    assert len(parents.split()) == 2


def test_merge_commit_prefers_an_identity_the_environment_supplies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    _clear_environment_identity(monkeypatch)
    for variable in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(variable, "Release Robot")
    for variable in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(variable, "robot@betterborg.dev")
    _advance_project_base(fixture, "base.txt", "base progress\n")

    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, MockAdapter(), RecordingLock()).run(
            fixture.context(store)
        )

    assert result.tip is not None
    identity = _git(
        fixture.repository,
        "show",
        "--no-patch",
        "--format=%an%n%ae%n%cn%n%ce",
        result.tip.commit_sha,
    )
    assert identity.splitlines() == [
        "Release Robot",
        "robot@betterborg.dev",
        "Release Robot",
        "robot@betterborg.dev",
    ]


def test_merge_commit_keeps_its_pre_attested_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    _clear_environment_identity(monkeypatch)
    # A merge that stopped setting these would stamp wall-clock time, which on
    # a UTC host matches the attested value in every field including the zone.
    monkeypatch.setenv("TZ", "Asia/Kolkata")
    _advance_project_base(fixture, "base.txt", "base progress\n")

    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, MockAdapter(), RecordingLock()).run(
            fixture.context(store)
        )
        started = next(
            event
            for event in store.list_task_execution_events(fixture.task.id)
            if event.kind == "merge.started"
        )

    assert result.tip is not None
    merge_date = started.payload["merge_date"]
    assert merge_date.endswith(" +0000")
    dates = _git(
        fixture.repository,
        "show",
        "--no-patch",
        "--format=%ad%n%cd",
        "--date=raw",
        result.tip.commit_sha,
    )
    expected = merge_date.removeprefix("@")
    assert dates.splitlines() == [expected, expected]


def test_merge_without_a_configured_identity_blocks_before_merging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    base_commit = _advance_project_base(fixture, "base.txt", "base progress\n")
    _clear_environment_identity(monkeypatch)
    _strip_git_identity(fixture)
    adapter = MockAdapter()

    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, adapter, RecordingLock()).run(
            fixture.context(store)
        )
        runtime = store.get_task_runtime(fixture.task.id)
        merge_events = [
            event.kind
            for event in store.list_task_execution_events(fixture.task.id)
            if event.kind.startswith("merge.")
        ]

    assert result.status is TaskRuntimeStatus.BLOCKED
    # Git's own banner names user.name and user.email too, so the reason has
    # to lead with Betterborg's sentence for this to say anything.
    assert result.reason.startswith("Git could not resolve a commit identity")
    assert "user.name" in result.reason and "user.email" in result.reason
    # Both roles are missing here, and Git reports the committer first.
    assert "no committer" in result.reason
    # Only Git's verdict survives its ten-line banner, and Git translates
    # that banner, so the reduction is pinned by shape rather than by text.
    assert "\n" not in result.reason
    assert "Please tell me who you are" not in result.reason
    assert adapter.calls == []
    assert merge_events == []
    assert runtime is not None and "user.email" in (runtime.state_reason or "")
    assert (
        _git(fixture.repository, "rev-parse", _project_branch(fixture))
        == base_commit
    )
    assert SafeGit(Path(runtime.worktree_path)).is_clean()


@pytest.mark.parametrize(
    ("supplied", "missing"),
    [("GIT_AUTHOR", "committer"), ("GIT_COMMITTER", "author")],
)
def test_an_identity_for_one_role_does_not_mask_the_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, supplied: str, missing: str
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    _advance_project_base(fixture, "base.txt", "base progress\n")
    _clear_environment_identity(monkeypatch)
    _strip_git_identity(fixture)
    monkeypatch.setenv(f"{supplied}_NAME", "Release Robot")
    monkeypatch.setenv(f"{supplied}_EMAIL", "robot@betterborg.dev")

    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, MockAdapter(), RecordingLock()).run(
            fixture.context(store)
        )

    assert result.status is TaskRuntimeStatus.BLOCKED
    assert f"Git resolves no {missing}" in result.reason


def test_unreadable_identity_is_not_reported_as_a_missing_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    _advance_project_base(fixture, "base.txt", "base progress\n")
    _clear_environment_identity(monkeypatch)
    original_run = SafeGit.run

    def unreadable_ident(self, arguments, **kwargs):  # noqa: ANN001, ANN003
        result = original_run(self, arguments, **kwargs)
        if arguments[0] == "var":
            return subprocess.CompletedProcess(
                result.args, 0, "Dev Name <no-timestamp>\n", ""
            )
        return result

    monkeypatch.setattr(SafeGit, "run", unreadable_ident)
    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, MockAdapter(), RecordingLock()).run(
            fixture.context(store)
        )

    assert result.status is TaskRuntimeStatus.BLOCKED
    assert result.reason.startswith("Git reported a committer Betterborg cannot read")
    assert "user.email" not in result.reason


def test_an_undecodable_identity_is_not_reported_as_a_missing_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    _advance_project_base(fixture, "base.txt", "base progress\n")
    _clear_environment_identity(monkeypatch)
    # Git writes configured bytes back verbatim, whatever encoding they are in.
    config = fixture.repository / ".git" / "config"
    config.write_bytes(
        config.read_bytes()
        + b"[user]\n\tname = Jos\xe9 Muller\n\temail = jose@example.com\n"
    )

    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, MockAdapter(), RecordingLock()).run(
            fixture.context(store)
        )

    assert result.status is TaskRuntimeStatus.BLOCKED
    assert result.reason.startswith("Git reported a committer Betterborg cannot decode")
    assert "user.email" not in result.reason


def test_merge_already_containing_the_base_needs_no_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    _clear_environment_identity(monkeypatch)
    _strip_git_identity(fixture)
    adapter = MockAdapter()

    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, adapter, RecordingLock()).run(
            fixture.context(store)
        )

    assert result.status is TaskRuntimeStatus.MERGING
    assert adapter.calls == []


def test_an_unusable_ambient_commit_date_is_not_a_missing_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    _advance_project_base(fixture, "base.txt", "base progress\n")
    _clear_environment_identity(monkeypatch)
    # Git parses this when resolving an identity, and refuses the whole
    # command when it cannot; the merge sets its own dates regardless.
    monkeypatch.setenv("GIT_AUTHOR_DATE", "2024-01-01")

    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, MockAdapter(), RecordingLock()).run(
            fixture.context(store)
        )

    assert result.status is TaskRuntimeStatus.MERGING
    assert result.tip is not None


def test_clean_merge_pins_the_identity_its_attestation_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configuration change mid-merge cannot move the pre-attested sha."""
    fixture = _approved_merge_fixture(tmp_path)
    _advance_project_base(fixture, "base.txt", "base progress\n")
    _clear_environment_identity(monkeypatch)
    original_run = SafeGit.run

    def rewrite_identity_after_attestation(self, arguments, **kwargs):  # noqa: ANN001, ANN003
        result = original_run(self, arguments, **kwargs)
        if arguments[0] == "commit-tree":
            _git(fixture.repository, "config", "user.email", "moved@betterborg.dev")
        return result

    monkeypatch.setattr(SafeGit, "run", rewrite_identity_after_attestation)
    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, MockAdapter(), RecordingLock()).run(
            fixture.context(store)
        )
        started = next(
            event
            for event in store.list_task_execution_events(fixture.task.id)
            if event.kind == "merge.started"
        )

    assert result.status is TaskRuntimeStatus.MERGING
    assert result.tip is not None
    assert result.tip.commit_sha == started.payload["expected_commit"]
    assert (
        _git(
            fixture.repository,
            "show",
            "--no-patch",
            "--format=%ae",
            result.tip.commit_sha,
        )
        == "tests@betterborg.dev"
    )


def test_clean_merge_attests_under_the_identity_it_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A change between resolving the identity and attesting cannot split them."""
    fixture = _approved_merge_fixture(tmp_path)
    _advance_project_base(fixture, "base.txt", "base progress\n")
    _clear_environment_identity(monkeypatch)
    original_run = SafeGit.run

    def rewrite_identity_after_probe(self, arguments, **kwargs):  # noqa: ANN001, ANN003
        result = original_run(self, arguments, **kwargs)
        if tuple(arguments) == ("var", "GIT_AUTHOR_IDENT"):
            _git(fixture.repository, "config", "user.email", "moved@betterborg.dev")
        return result

    monkeypatch.setattr(SafeGit, "run", rewrite_identity_after_probe)
    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, MockAdapter(), RecordingLock()).run(
            fixture.context(store)
        )
        started = next(
            event
            for event in store.list_task_execution_events(fixture.task.id)
            if event.kind == "merge.started"
        )

    assert result.status is TaskRuntimeStatus.MERGING
    assert result.tip is not None
    assert result.tip.commit_sha == started.payload["expected_commit"]


def test_merge_that_cannot_fast_forward_is_not_refused_by_merge_ff_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    _clear_environment_identity(monkeypatch)
    # Whether this merge writes a commit is Betterborg's decision, and the sha
    # it attested to assumes it; an operator who only ever fast-forwards would
    # otherwise have Git refuse the merge outright.
    _git(fixture.repository, "config", "merge.ff", "only")
    _advance_project_base(fixture, "base.txt", "base progress\n")

    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, MockAdapter(), RecordingLock()).run(
            fixture.context(store)
        )
        started = next(
            event
            for event in store.list_task_execution_events(fixture.task.id)
            if event.kind == "merge.started"
        )

    assert result.status is TaskRuntimeStatus.MERGING
    assert result.tip is not None
    assert result.tip.commit_sha == started.payload["expected_commit"]
    parents = _git(
        fixture.repository,
        "show",
        "--no-patch",
        "--format=%P",
        result.tip.commit_sha,
    )
    assert len(parents.split()) == 2


def test_fast_forward_merge_needs_no_configured_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    base_commit = _advance_project_base_past_task_tip(fixture)
    _clear_environment_identity(monkeypatch)
    # An operator who never fast-forwards would otherwise turn this into a
    # two-parent commit, which needs the identity this test removes.
    _git(fixture.repository, "config", "merge.ff", "false")
    _strip_git_identity(fixture)
    adapter = MockAdapter()

    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, adapter, RecordingLock()).run(
            fixture.context(store)
        )

    assert result.status is TaskRuntimeStatus.MERGING
    assert result.tip is not None
    assert result.tip.commit_sha == base_commit
    assert adapter.calls == []


def test_resumed_merge_without_an_identity_blocks_before_the_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _approved_merge_fixture(tmp_path)
    _advance_project_base(fixture, "feature.txt", "project version\n")
    with SqliteStore.open(fixture.database) as store:
        runtime = store.get_task_runtime(fixture.task.id)
    assert runtime is not None
    worktree = Path(runtime.worktree_path)
    original_run = SafeGit.run

    def interrupt_after_merge(self, arguments, **kwargs):  # noqa: ANN001, ANN003
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
    monkeypatch.setattr(SafeGit, "run", original_run)
    assert _git(worktree, "rev-parse", "--verify", "--quiet", "MERGE_HEAD")

    _clear_environment_identity(monkeypatch)
    _strip_git_identity(fixture)
    adapter = MockAdapter()
    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, adapter, RecordingLock()).run(
            fixture.context(store)
        )

    assert result.status is TaskRuntimeStatus.BLOCKED
    assert result.reason.startswith("Git could not resolve a commit identity")
    # The agent resolves conflicts by committing; reaching it without an
    # identity invites it to invent one.
    assert adapter.calls == []


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
    received: list[AgentActivity] = []

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
        return MockResponse(
            payload=_completed_payload(fixture.task),
            activities=(
                AgentActivity(AgentActivityKind.WRITING, "feature.txt"),
            ),
        )

    adapter = MockAdapter().queue(MockResponse(dynamic=resolve))
    with SqliteStore.open(fixture.database) as store:
        result = _phase(fixture, adapter, repository_lock).run(
            fixture.context(store, activity=received.append)
        )
        attempts = store.list_agent_attempts(fixture.task.id)
        runtime = store.get_task_runtime(fixture.task.id)

    assert result.status is TaskRuntimeStatus.MERGING
    assert result.tip is not None and result.tip.agent_used
    assert len(adapter.calls) == 1
    assert adapter.calls[0].activity_sink is not None
    assert received == [
        AgentActivity(AgentActivityKind.WRITING, "merge: feature.txt")
    ]
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
