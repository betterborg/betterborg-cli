"""Three-door onboarding dispatch and machine handoff contracts."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from progress_test_support import FailingStringIO, TTYStringIO

from betterborg_cli.agent_runtime.base import CancellationToken
from betterborg_cli.onboarding import OnboardingDispatcher, create_commands
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.progress import RunProgress, StageSpec
from betterborg_cli.repo_analysis import ImprovementPrd
from betterborg_cli.store import Borg, Repository, SqliteStore


class RecordingCreator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path | None]] = []

    def create(self, name: str, source: Path | None = None):
        self.calls.append((name, source))
        return self.calls[-1]


def _io(answers: Iterator[str | None], output: list[str]) -> InteractiveIO:
    return InteractiveIO(
        prompt=lambda _message: next(answers),
        confirm=lambda _message, _default: False,
        write=output.append,
    )


def _documents(root: Path) -> tuple[ImprovementPrd, ...]:
    directory = root / ".betterborg/prds/improvements"
    return (
        ImprovementPrd(
            theme_key="checks",
            title="Reliable checks",
            predicted_impact=0.25,
            effort="S",
            suggested_borg_name="sentinel",
            path=directory / "checks.md",
            body_md="# Reliable checks\n",
        ),
        ImprovementPrd(
            theme_key="docs",
            title="Approachable docs",
            predicted_impact=0.125,
            effort="M",
            suggested_borg_name="scribe",
            path=directory / "docs.md",
            body_md="# Approachable docs\n",
        ),
    )


@pytest.fixture
def onboarding_context(committed_git_repo: Path):
    repository = Repository(root=committed_git_repo)
    with SqliteStore.open(
        committed_git_repo / ".betterborg/state/betterborg.sqlite3"
    ) as store:
        store.add_repository(repository)
        yield repository, store


@pytest.mark.parametrize(
    ("theme_answer", "expected_name", "expected_filename"),
    [
        ("", "sentinel", "checks.md"),
        ("2", "scribe", "docs.md"),
    ],
    ids=["top-theme-default", "second-ranked-theme"],
)
def test_fix_door_lists_every_ranked_theme_and_dispatches_exact_source(
    onboarding_context,
    theme_answer: str,
    expected_name: str,
    expected_filename: str,
) -> None:
    repository, store = onboarding_context
    output: list[str] = []
    creator = RecordingCreator()
    dispatcher = OnboardingDispatcher(
        repository,
        store,
        _io(iter(["1", theme_answer, ""]), output),
        creator,
        _documents(repository.root),
    )

    dispatcher.run()

    assert creator.calls == [
        (
            expected_name,
            repository.root
            / ".betterborg"
            / "prds"
            / "improvements"
            / expected_filename,
        )
    ]
    rendered = "\n".join(output)
    assert "Reliable checks — predicted impact +0.25; effort S" in rendered
    assert "Approachable docs — predicted impact +0.125; effort M" in rendered


def test_fix_door_edits_a_colliding_suggested_name(onboarding_context) -> None:
    repository, store = onboarding_context
    store.add_borg(Borg(repository_id=repository.id, name="sentinel"))
    output: list[str] = []
    creator = RecordingCreator()
    dispatcher = OnboardingDispatcher(
        repository,
        store,
        _io(iter(["1", "", "", "Guardian"]), output),
        creator,
        _documents(repository.root),
    )

    dispatcher.run()

    assert creator.calls[0][0] == "Guardian"
    assert "already exists" in "\n".join(output)


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        (("2", "incoming.md", ""), ("incoming", "incoming.md")),
        (("3", "NewIdea"), ("NewIdea", None)),
    ],
    ids=["existing-prd", "brainstorm"],
)
def test_other_doors_use_the_same_creator(
    onboarding_context,
    answers: tuple[str, ...],
    expected: tuple[str, str | None],
) -> None:
    repository, store = onboarding_context
    output: list[str] = []
    creator = RecordingCreator()

    OnboardingDispatcher(
        repository,
        store,
        _io(iter(answers), output),
        creator,
        _documents(repository.root),
    ).run()

    expected_source = (
        repository.root / expected[1] if expected[1] is not None else None
    )
    assert creator.calls == [(expected[0], expected_source)]
    assert output[:4] == [
        "What would you like to do next?",
        "  1. Fix the repo",
        "  2. Improve an existing PRD",
        "  3. Brainstorm a new PRD",
    ]


def test_cancellation_does_not_dispatch_or_mutate(onboarding_context) -> None:
    repository, store = onboarding_context
    output: list[str] = []
    creator = RecordingCreator()

    result = OnboardingDispatcher(
        repository,
        store,
        _io(iter(["q"]), output),
        creator,
        _documents(repository.root),
    ).run()

    assert result is None
    assert creator.calls == []
    with store.locked_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM borgs").fetchone()[0] == 0
        assert (
            connection.execute("SELECT COUNT(*) FROM prd_sessions").fetchone()[0]
            == 0
        )


def test_token_cancellation_during_menu_starts_no_selected_door(
    onboarding_context,
) -> None:
    repository, store = onboarding_context
    cancel = CancellationToken()
    creator = RecordingCreator()

    def cancel_while_choosing(_message: str) -> str:
        cancel.cancel()
        return "1"

    result = OnboardingDispatcher(
        repository,
        store,
        InteractiveIO(
            prompt=cancel_while_choosing,
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        ),
        creator,
        _documents(repository.root),
        cancel=cancel,
    ).run()

    assert result is None
    assert creator.calls == []


@pytest.mark.parametrize(
    "answers_before_name",
    [("1", "1"), ("2", "incoming.md"), ("3",)],
    ids=["fix-repo", "improve-prd", "brainstorm"],
)
def test_token_cancellation_during_name_prompt_starts_no_prd_session(
    onboarding_context,
    answers_before_name: tuple[str, ...],
) -> None:
    repository, store = onboarding_context
    cancel = CancellationToken()
    creator = RecordingCreator()
    answers = iter(answers_before_name)

    def cancel_while_naming(message: str) -> str:
        if message.startswith("Borg name"):
            cancel.cancel()
            return "cancelled-name"
        return next(answers)

    result = OnboardingDispatcher(
        repository,
        store,
        InteractiveIO(
            prompt=cancel_while_naming,
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        ),
        creator,
        _documents(repository.root),
        cancel=cancel,
    ).run()

    assert result is None
    assert creator.calls == []
    with store.locked_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM borgs").fetchone()[0] == 0
        assert (
            connection.execute("SELECT COUNT(*) FROM prd_sessions").fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("environment", [{}, {"NO_COLOR": "1"}, {"TERM": "dumb"}])
def test_menu_suspension_crosses_heartbeat_without_overdrawing_prompts(
    onboarding_context,
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
) -> None:
    repository, store = onboarding_context
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    stream = TTYStringIO()
    progress = RunProgress(
        [StageSpec("active", "Active work")],
        stream=stream,
        heartbeat_interval=0.02,
    )
    progress.start("active")
    snapshots: list[tuple[str, str]] = []

    def wait_at_prompt(_message: str) -> str:
        before = stream.getvalue()
        time.sleep(0.05)
        snapshots.append((before, stream.getvalue()))
        return "q"

    result = OnboardingDispatcher(
        repository,
        store,
        InteractiveIO(
            prompt=wait_at_prompt,
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        ),
        RecordingCreator(),
        _documents(repository.root),
        progress=progress,
    ).run()

    assert result is None
    assert snapshots and all(before == after for before, after in snapshots)
    progress.stop_display()


def test_menu_suspension_propagates_the_first_renderer_failure(
    onboarding_context,
) -> None:
    repository, store = onboarding_context
    stream = FailingStringIO()
    progress = RunProgress(
        [StageSpec("active", "Active work")],
        stream=stream,
        heartbeat_interval=0.01,
    )
    progress.start("active")
    stream.fail_next_write()
    deadline = time.monotonic() + 1
    while progress._cadence_worker is not None:
        assert time.monotonic() < deadline
        time.sleep(0.005)

    with pytest.raises(RuntimeError, match="progress heartbeat failed"):
        OnboardingDispatcher(
            repository,
            store,
            InteractiveIO(
                prompt=lambda _message: pytest.fail("prompt must not run"),
                confirm=lambda _message, _default: False,
                write=lambda _message: pytest.fail("menu must not render"),
            ),
            RecordingCreator(),
            _documents(repository.root),
            progress=progress,
        ).run()

    progress.raise_if_render_failed()
    progress.stop_display()


def test_machine_handoff_commands_are_exact_and_mutation_free(
    onboarding_context,
) -> None:
    repository, store = onboarding_context

    commands = create_commands(repository.root, _documents(repository.root))

    assert commands == (
        (
            "betterborg",
            "create",
            "sentinel",
            "--prd",
            ".betterborg/prds/improvements/checks.md",
        ),
        (
            "betterborg",
            "create",
            "scribe",
            "--prd",
            ".betterborg/prds/improvements/docs.md",
        ),
    )
    with store.locked_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM borgs").fetchone()[0] == 0
        assert (
            connection.execute("SELECT COUNT(*) FROM prd_sessions").fetchone()[0]
            == 0
        )
