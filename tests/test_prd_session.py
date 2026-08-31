"""Shared PRD interviewing, review, and confirmation contracts."""

from __future__ import annotations

from collections.abc import Iterator
from io import StringIO
from pathlib import Path

import pytest

from betterborg_cli import prd_session as prd_session_module
from betterborg_cli.agent_runtime.api_tools import ApiAgentRole
from betterborg_cli.agent_runtime.base import AgentCapabilities, CancellationToken
from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.agent_runtime.selection import SelectedAgent
from betterborg_cli.prd_session import InteractiveIO, PrdSession, PrdSessionError
from betterborg_cli.progress import (
    AgentActivity,
    AgentActivityKind,
    RunProgress,
    StageState,
)
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.store import Repository, SqliteStore


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TTYStringIO(StringIO):
    def isatty(self) -> bool:
        return True


def _responses(*payloads: dict[str, object]) -> MockAdapter:
    adapter = MockAdapter(name="openai")
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


def test_requirements_progress_records_activity_turns_and_confirmed_publication(
    repository_store,
) -> None:
    repository, store = repository_store
    activity = AgentActivity(AgentActivityKind.READING, "pyproject.toml")
    adapter = MockAdapter(name="openai").queue(
        MockResponse(
            payload={
                "questions": [],
                "prd_markdown": "# Durable draft\n\nPublish after confirmation.",
            },
            activities=(activity,),
        )
    )
    progress = RunProgress(stream=StringIO())
    session = PrdSession(
        repository,
        store,
        adapter,
        io=_io(confirmations=iter([True])),
        progress=progress,
    )

    result = session.run("Progress")

    record = progress.stages["requirements"]
    assert result.confirmed
    assert record.state is StageState.COMPLETED
    assert record.result == "PRD 'Progress' confirmed"
    assert record.activity == activity
    assert record.detail == "2 turns recorded"
    assert [turn.role for turn in store.list_prd_turns(result.session.id)] == [
        "user",
        "assistant",
    ]
    assert result.prd_path.read_text(encoding="utf-8") == result.body_md


def test_cancellation_reconciles_the_durable_assistant_turn_before_stopping(
    repository_store,
) -> None:
    repository, store = repository_store
    cancel = CancellationToken()

    def cancel_with_questions(_spec):
        cancel.cancel()
        return {
            "questions": ["Which compatibility promise matters most?"],
            "prd_markdown": None,
        }

    adapter = MockAdapter(name="openai").queue(
        MockResponse(dynamic=cancel_with_questions)
    )
    progress = RunProgress(stream=StringIO())
    session = PrdSession(
        repository,
        store,
        adapter,
        io=InteractiveIO(
            prompt=lambda _message: pytest.fail("cancelled work must not prompt"),
            confirm=lambda _message, _default: pytest.fail(
                "cancelled work must not confirm"
            ),
            write=lambda _message: pytest.fail("cancelled work must not render"),
        ),
        cancel=cancel,
        progress=progress,
    )

    result = session.run("CancelledAfterTurn")

    assert result.cancelled
    assert progress.stages["requirements"].state is StageState.STOPPED
    assert progress.stages["requirements"].result == "interrupted"
    assert [turn.role for turn in store.list_prd_turns(result.session.id)] == [
        "user",
        "assistant",
    ]
    assert len(adapter.calls) == 1
    assert not result.prd_path.exists()


def test_cancellation_keeps_a_recorded_answer_and_starts_no_later_turn(
    repository_store,
) -> None:
    repository, store = repository_store
    cancel = CancellationToken()
    adapter = _responses(
        {
            "questions": ["Who needs this behavior?"],
            "prd_markdown": None,
        }
    )

    def answer_then_cancel(_message: str) -> str:
        cancel.cancel()
        return "Repository maintainers"

    progress = RunProgress(stream=StringIO())
    session = PrdSession(
        repository,
        store,
        adapter,
        io=InteractiveIO(
            prompt=answer_then_cancel,
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        ),
        cancel=cancel,
        progress=progress,
    )

    result = session.run("CancelledAfterAnswer")

    turns = store.list_prd_turns(result.session.id)
    assert [turn.role for turn in turns] == ["user", "assistant", "user"]
    assert turns[-1].content == "Repository maintainers"
    assert progress.stages["requirements"].state is StageState.STOPPED
    assert len(adapter.calls) == 1
    assert not result.prd_path.exists()


def test_cancellation_before_run_creates_no_session_records(
    repository_store,
) -> None:
    repository, store = repository_store
    cancel = CancellationToken()
    adapter = _responses(
        {"questions": [], "prd_markdown": "# Must not run\n"}
    )
    progress = RunProgress(stream=StringIO())
    session = PrdSession(
        repository,
        store,
        adapter,
        io=InteractiveIO(
            prompt=lambda _message: pytest.fail("cancelled work must not prompt"),
            confirm=lambda _message, _default: pytest.fail(
                "cancelled work must not confirm"
            ),
            write=lambda _message: pytest.fail("cancelled work must not render"),
        ),
        cancel=cancel,
        progress=progress,
    )
    cancel.cancel()

    result = session.run("NeverStarted")

    assert result.cancelled
    assert adapter.calls == []
    assert progress.stages["requirements"].state is StageState.PENDING
    with store.locked_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM borgs").fetchone()[0] == 0
        assert (
            connection.execute("SELECT COUNT(*) FROM prd_sessions").fetchone()[0]
            == 0
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM prd_turns").fetchone()[0]
            == 0
        )


def test_cancellation_during_editor_confirmation_does_not_launch_editor(
    repository_store,
) -> None:
    repository, store = repository_store
    cancel = CancellationToken()
    adapter = _responses(
        {"questions": [], "prd_markdown": "# Draft\n\nReview this."}
    )
    confirmations: list[str] = []

    def cancel_editor_confirmation(message: str, _default: bool) -> bool:
        confirmations.append(message)
        cancel.cancel()
        return True

    progress = RunProgress(stream=StringIO())
    result = PrdSession(
        repository,
        store,
        adapter,
        io=InteractiveIO(
            prompt=lambda _message: pytest.fail("draft needs no answers"),
            confirm=cancel_editor_confirmation,
            write=lambda _message: None,
        ),
        editor=lambda _body: pytest.fail("cancelled work must not launch editor"),
        cancel=cancel,
        progress=progress,
    ).run("CancelledEditor")

    assert result.cancelled
    assert confirmations == ["Review and edit this PRD in your editor?"]
    assert progress.stages["requirements"].state is StageState.STOPPED
    assert [turn.role for turn in store.list_prd_turns(result.session.id)] == [
        "user",
        "assistant",
    ]
    assert not result.prd_path.exists()


@pytest.mark.parametrize("environment", [{}, {"NO_COLOR": "1"}, {"TERM": "dumb"}])
def test_draft_editor_and_confirmations_suspend_renderer_heartbeats(
    repository_store,
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
) -> None:
    repository, store = repository_store
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    stream = TTYStringIO()
    clock = FakeClock()
    progress = RunProgress(
        stream=stream,
        clock=clock,
        heartbeat_interval=5,
    )
    snapshots: list[tuple[str, str]] = []

    def cross_heartbeat() -> None:
        before = stream.getvalue()
        clock.now += 10
        progress.refresh()
        snapshots.append((before, stream.getvalue()))

    confirmations = iter((True, True))

    def confirm(_message: str, _default: bool) -> bool:
        cross_heartbeat()
        return next(confirmations)

    def write(_message: str) -> None:
        cross_heartbeat()

    def edit(_body: str) -> str:
        cross_heartbeat()
        return "# Edited draft\n\nConfirmed by a human."

    def answer(_message: str) -> str:
        cross_heartbeat()
        return "Repository maintainers"

    result = PrdSession(
        repository,
        store,
        _responses(
            {
                "questions": ["Who needs the first release?"],
                "prd_markdown": None,
            },
            {"questions": [], "prd_markdown": "# Draft\n\nNeeds review."}
        ),
        io=InteractiveIO(prompt=answer, confirm=confirm, write=write),
        editor=edit,
        progress=progress,
    ).run("Suspended")

    assert result.confirmed
    assert len(snapshots) == 6
    assert all(before == after for before, after in snapshots)


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
    progress = RunProgress(stream=StringIO())
    session = PrdSession(
        repository,
        store,
        _responses(
            {"questions": [], "prd_markdown": "# Not yet\n\nNeeds review."}
        ),
        io=_io(confirmations=iter([False])),
        progress=progress,
    )

    result = session.run("NotYet")

    assert result.cancelled
    assert result.body_md == "# Not yet\n\nNeeds review.\n"
    assert not result.prd_path.exists()
    assert store.get_borg(result.borg.id) == result.borg
    assert progress.stages["requirements"].state is StageState.STOPPED
    assert progress.stages["requirements"].result == "not confirmed"


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


@pytest.mark.parametrize(
    "capabilities, message",
    [
        (AgentCapabilities(), "read-only tool allowlist"),
        (
            AgentCapabilities(
                tool_allowlist=True,
                host_capable=True,
            ),
            "wrapped by SelectedAgent",
        ),
    ],
    ids=["no-tool-allowlist", "raw-host-capable"],
)
def test_prd_session_rejects_unconfined_adapters(
    repository_store,
    capabilities: AgentCapabilities,
    message: str,
) -> None:
    repository, store = repository_store
    adapter = MockAdapter(name="openai", capabilities=capabilities)

    with pytest.raises(PrdSessionError, match=message):
        PrdSession(repository, store, adapter, interactive=False)

    assert adapter.calls == []


def test_selected_host_capable_agent_requires_workspace_trust(
    repository_store,
) -> None:
    repository, store = repository_store
    adapter = MockAdapter(
        name="openai",
        capabilities=AgentCapabilities(
            tool_allowlist=True,
            host_capable=True,
        ),
    ).queue(
        MockResponse(
            payload={"questions": [], "prd_markdown": "# Trusted\n\nRead only."}
        )
    )
    trusted: list[Path] = []
    selected = SelectedAgent(
        role=ApiAgentRole.ANALYSIS,
        adapter=adapter,
        paths=RepoPaths.discover(repository.root),
        trust_requirement=lambda paths, **_kwargs: trusted.append(paths.root),
    )
    session = PrdSession(repository, store, selected, interactive=False)

    result = session.run("Trusted", confirmed=True)

    assert result.confirmed
    assert trusted == [repository.root]
    assert adapter.calls[0].allowed_tools == (
        "list_files",
        "read_file",
        "search_text",
    )


def test_prd_model_must_be_explicit_for_an_unknown_adapter(
    repository_store,
) -> None:
    repository, store = repository_store
    adapter = MockAdapter(name="custom")

    with pytest.raises(PrdSessionError, match="model must be configured"):
        PrdSession(repository, store, adapter, interactive=False)

    assert adapter.calls == []


def test_prd_model_prefers_the_selected_agent_override(
    repository_store,
) -> None:
    repository, store = repository_store
    adapter = MockAdapter(name="custom").queue(
        MockResponse(
            payload={"questions": [], "prd_markdown": "# Model\n\nConfigured."}
        )
    )
    selected = SelectedAgent(
        role=ApiAgentRole.ANALYSIS,
        adapter=adapter,
        paths=RepoPaths.discover(repository.root),
        model="configured-model",
    )
    session = PrdSession(repository, store, selected, interactive=False)

    session.run("ConfiguredModel")

    assert adapter.calls[0].model == "configured-model"


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
        model="test-model",
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
        model="test-model",
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


def test_confirmed_publication_winning_cancellation_race_completes_after_reconcile(
    repository_store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, store = repository_store
    cancel = CancellationToken()
    progress = RunProgress(stream=StringIO())
    publish = prd_session_module._publish_confirmed_prd

    def publish_then_interrupt(path: Path, body: str, *, root: Path) -> None:
        publish(path, body, root=root)
        cancel.cancel()
        raise KeyboardInterrupt("interrupted after stable publication")

    monkeypatch.setattr(
        prd_session_module,
        "_publish_confirmed_prd",
        publish_then_interrupt,
    )
    session = PrdSession(
        repository,
        store,
        _responses(
            {
                "questions": [],
                "prd_markdown": "# Confirmed\n\nThe stable file won the race.",
            }
        ),
        io=_io(confirmations=iter([True])),
        cancel=cancel,
        progress=progress,
    )

    result = session.run("PublicationRace")

    assert result.confirmed
    assert result.prd_path.read_text(encoding="utf-8") == result.body_md
    assert progress.stages["requirements"].state is StageState.COMPLETED
