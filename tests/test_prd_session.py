"""Shared PRD interviewing, review, and confirmation contracts."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.prd_session import InteractiveIO, PrdSession, PrdSessionError
from betterborg_cli.store import Repository, SqliteStore


def _responses(*payloads: dict[str, object]) -> MockAdapter:
    adapter = MockAdapter()
    for payload in payloads:
        adapter.queue(MockResponse(payload=payload))
    return adapter


def _io(
    *,
    answers: Iterator[str | None] | None = None,
    confirmations: Iterator[bool] | None = None,
    output: list[str] | None = None,
) -> InteractiveIO:
    rendered = output if output is not None else []
    supplied_answers = answers if answers is not None else iter(())
    supplied_confirmations = confirmations if confirmations is not None else iter(())
    return InteractiveIO(
        prompt=lambda _message: next(supplied_answers),
        confirm=lambda _message, _default: next(supplied_confirmations),
        write=rendered.append,
    )


@pytest.fixture
def repository_store(committed_git_repo: Path):
    repository = Repository(root=committed_git_repo)
    with SqliteStore.open(
        committed_git_repo / ".borg/state/borg.sqlite3"
    ) as store:
        store.add_repository(repository)
        yield repository, store


def test_empty_brainstorm_asks_material_questions_and_confirms(
    repository_store,
) -> None:
    repository, store = repository_store
    adapter = _responses(
        {
            "questions": ["Who should use the first release?"],
            "prd_markdown": None,
        },
        {
            "questions": [],
            "prd_markdown": "# Guided setup\n\nHelp new maintainers configure a repo.",
        },
    )
    output: list[str] = []
    session = PrdSession(
        repository,
        store,
        adapter,
        io=_io(
            answers=iter(["Maintainers adopting the CLI"]),
            confirmations=iter([True]),
            output=output,
        ),
    )

    result = session.run("Guide")

    assert result.confirmed
    assert not result.cancelled
    assert result.prd_path == repository.root / ".borg/prds/Guide.md"
    assert result.prd_path.read_text(encoding="utf-8") == (
        "# Guided setup\n\nHelp new maintainers configure a repo.\n"
    )
    assert store.get_borg_by_name(repository.id, "Guide") == result.borg
    assert store.get_prd_session(result.session.id) == result.session
    turns = store.list_prd_turns(result.session.id)
    assert [turn.role for turn in turns] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert turns[2].content == "Maintainers adopting the CLI"
    assert len(adapter.calls) == 2
    assert "Maintainers adopting the CLI" in adapter.calls[1].user_prompt
    assert output == [result.body_md]
    assert store.list_operations(repository.id) == []


@pytest.mark.parametrize(
    "relative_source",
    [Path("incoming.md"), Path(".borg/prds/improvements/theme-ci.md")],
    ids=["standalone-markdown", "generated-theme"],
)
def test_markdown_doors_share_editing_and_preserve_their_input(
    repository_store,
    relative_source: Path,
) -> None:
    repository, store = repository_store
    source = repository.root / relative_source
    source.parent.mkdir(parents=True, exist_ok=True)
    original = "# Existing input\n\nKeep the source byte-for-byte.\n"
    source.write_text(original, encoding="utf-8")
    adapter = _responses(
        {
            "questions": [],
            "prd_markdown": "# Improved draft\n\nAgent revision.",
        }
    )
    edited_inputs: list[str] = []

    def edit(body: str) -> str:
        edited_inputs.append(body)
        return "# Human-reviewed draft\n\nFinal requirements."

    session = PrdSession(
        repository,
        store,
        adapter,
        io=_io(confirmations=iter([True, True])),
        editor=edit,
    )

    result = session.run("Editor", source)

    assert result.confirmed
    assert source.read_text(encoding="utf-8") == original
    assert result.prd_path.read_text(encoding="utf-8") == (
        "# Human-reviewed draft\n\nFinal requirements.\n"
    )
    assert edited_inputs == ["# Improved draft\n\nAgent revision.\n"]
    turns = store.list_prd_turns(result.session.id)
    assert turns[0].content == original
    assert turns[-1].content == result.body_md
    assert "Never start planning" in adapter.calls[0].system_prompt
    assert adapter.calls[0].allowed_tools == (
        "list_files",
        "read_file",
        "search_text",
    )


@pytest.mark.parametrize(
    "questions",
    [["   "], ["Who is this for?", " Who is this for? "]],
    ids=["empty-after-trimming", "duplicate-after-trimming"],
)
def test_agent_questions_must_be_material_after_normalization(
    repository_store,
    questions: list[str],
) -> None:
    repository, store = repository_store
    prompted: list[str] = []
    session = PrdSession(
        repository,
        store,
        _responses({"questions": questions, "prd_markdown": None}),
        io=InteractiveIO(
            prompt=lambda message: prompted.append(message) or "answer",
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        ),
    )

    with pytest.raises(PrdSessionError, match="must not be (empty|duplicated)"):
        session.run("InvalidQuestion")

    assert prompted == []


def test_cancelling_a_material_question_keeps_only_the_draft_records(
    repository_store,
) -> None:
    repository, store = repository_store
    adapter = _responses(
        {
            "questions": ["Which compatibility promise matters most?"],
            "prd_markdown": None,
        }
    )
    session = PrdSession(
        repository,
        store,
        adapter,
        io=_io(answers=iter([None])),
    )

    result = session.run("Cancelled")

    assert result.cancelled
    assert not result.confirmed
    assert not result.prd_path.exists()
    assert store.get_borg(result.borg.id) == result.borg
    assert store.get_prd_session(result.session.id) == result.session
    assert [turn.role for turn in store.list_prd_turns(result.session.id)] == [
        "user",
        "assistant",
    ]


def test_rejecting_final_confirmation_does_not_publish_the_draft(
    repository_store,
) -> None:
    repository, store = repository_store
    session = PrdSession(
        repository,
        store,
        _responses(
            {"questions": [], "prd_markdown": "# Not yet\n\nNeeds review."}
        ),
        io=_io(confirmations=iter([False])),
    )

    result = session.run("NotYet")

    assert result.cancelled
    assert result.body_md == "# Not yet\n\nNeeds review.\n"
    assert not result.prd_path.exists()
    assert store.get_borg(result.borg.id) == result.borg


def test_machine_mode_uses_explicit_confirmation_without_io_or_editor(
    repository_store,
) -> None:
    repository, store = repository_store
    editor_calls = 0

    def forbidden_editor(_body: str) -> str:
        nonlocal editor_calls
        editor_calls += 1
        raise AssertionError("machine-readable sessions must not launch an editor")

    session = PrdSession(
        repository,
        store,
        _responses(
            {"questions": [], "prd_markdown": "# Automated\n\nExplicitly approved."}
        ),
        interactive=False,
        editor=forbidden_editor,
    )

    result = session.run("Automated", confirmed=True)

    assert result.confirmed
    assert result.prd_path.read_text(encoding="utf-8") == result.body_md
    assert editor_calls == 0


def test_machine_mode_returns_questions_instead_of_prompting(
    repository_store,
) -> None:
    repository, store = repository_store
    session = PrdSession(
        repository,
        store,
        _responses(
            {
                "questions": ["Which users are in scope?"],
                "prd_markdown": None,
            }
        ),
        interactive=False,
    )

    result = session.run("MachineQuestions", confirmed=True)

    assert not result.confirmed
    assert not result.cancelled
    assert result.questions == ("Which users are in scope?",)
    assert not result.prd_path.exists()


def test_source_must_be_nonempty_local_markdown_before_creating_a_borg(
    repository_store,
    tmp_path: Path,
) -> None:
    repository, store = repository_store
    source = tmp_path / "requirements.txt"
    source.write_text("not Markdown", encoding="utf-8")
    session = PrdSession(
        repository,
        store,
        MockAdapter(),
        io=_io(),
    )

    with pytest.raises(ValueError, match="local Markdown"):
        session.run("Invalid", source)

    assert store.get_borg_by_name(repository.id, "Invalid") is None


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "nested/name",
        "CON",
        "bad\\name",
        "bad\x00name",
        "bad<name",
        "bad>name",
        "bad:name",
        'bad"name',
        "bad|name",
        "bad?name",
        "bad*name",
    ],
)
def test_invalid_borg_name_is_rejected_before_creating_records(
    repository_store,
    name: str,
) -> None:
    repository, store = repository_store
    session = PrdSession(
        repository,
        store,
        MockAdapter(),
        io=_io(),
    )

    with pytest.raises(ValueError, match="filename"):
        session.run(name)

    assert store.get_borg_by_name(repository.id, name) is None


def test_confirmed_prd_does_not_overwrite_a_racing_destination(
    repository_store,
) -> None:
    repository, store = repository_store
    destination = repository.root / ".borg/prds/Race.md"

    def create_destination(body: str) -> str:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("# Existing\n", encoding="utf-8")
        return body

    session = PrdSession(
        repository,
        store,
        _responses({"questions": [], "prd_markdown": "# Candidate"}),
        io=_io(confirmations=iter([True, True])),
        editor=create_destination,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        session.run("Race")

    assert destination.read_text(encoding="utf-8") == "# Existing\n"
