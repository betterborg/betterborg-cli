"""Three-door onboarding dispatch and machine handoff contracts."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from betterborg_cli.onboarding import OnboardingDispatcher, create_commands
from betterborg_cli.prd_session import InteractiveIO
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
    directory = root / ".borg/prds/improvements"
    return (
        ImprovementPrd(
            theme_key="checks",
            title="Reliable checks",
            predicted_impact=0.25,
            effort="S",
            suggested_borg_name="Sentinel",
            path=directory / "checks.md",
            body_md="# Reliable checks\n",
        ),
        ImprovementPrd(
            theme_key="docs",
            title="Approachable docs",
            predicted_impact=0.125,
            effort="M",
            suggested_borg_name="Scribe",
            path=directory / "docs.md",
            body_md="# Approachable docs\n",
        ),
    )


@pytest.fixture
def onboarding_context(committed_git_repo: Path):
    repository = Repository(root=committed_git_repo)
    with SqliteStore.open(
        committed_git_repo / ".borg/state/borg.sqlite3"
    ) as store:
        store.add_repository(repository)
        yield repository, store


@pytest.mark.parametrize(
    ("theme_answer", "expected_name", "expected_filename"),
    [
        ("", "Sentinel", "checks.md"),
        ("2", "Scribe", "docs.md"),
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
            / ".borg"
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
    store.add_borg(Borg(repository_id=repository.id, name="Sentinel"))
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


def test_machine_handoff_commands_are_exact_and_mutation_free(
    onboarding_context,
) -> None:
    repository, store = onboarding_context

    commands = create_commands(repository.root, _documents(repository.root))

    assert commands == (
        (
            "borg",
            "create",
            "--name",
            "Sentinel",
            "--prd",
            ".borg/prds/improvements/checks.md",
        ),
        (
            "borg",
            "create",
            "--name",
            "Scribe",
            "--prd",
            ".borg/prds/improvements/docs.md",
        ),
    )
    with store.locked_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM borgs").fetchone()[0] == 0
        assert (
            connection.execute("SELECT COUNT(*) FROM prd_sessions").fetchone()[0]
            == 0
        )
