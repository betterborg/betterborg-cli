"""Tests for the public CLI bootstrap."""

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner
from pytest import MonkeyPatch

from betterborg_cli import __version__
from betterborg_cli import cli as cli_module
from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.cli import cli
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.store import Repository, SqliteStore
from betterborg_cli.workspace_trust import TrustStore, WorkspaceIdentity


@pytest.fixture
def initialized_cli_repository(
    committed_git_repo: Path,
) -> Iterator[tuple[Repository, RepoPaths]]:
    paths = RepoPaths.discover(committed_git_repo)
    repository = Repository(root=committed_git_repo)
    paths.tracked_dir.mkdir(parents=True)
    paths.tracked_dir.joinpath("config.toml").write_text(
        "\n".join(
            (
                "version = 1",
                "",
                "[repository]",
                f'id = "{repository.id}"',
                'default_branch = "main"',
                "",
            )
        ),
        encoding="utf-8",
    )
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        store.add_repository(repository)
    yield repository, paths


def _invoke_create(
    cli_runner: CliRunner,
    monkeypatch: MonkeyPatch,
    configure_interactive_cli: Callable,
    repository: Repository,
    adapter: MockAdapter,
    io: InteractiveIO,
    *arguments: str,
    editor=None,
):
    configure_interactive_cli(
        repository.root,
        adapter,
        io,
        state_home=repository.root.parent / "machine-state",
    )
    monkeypatch.setattr(cli_module, "_edit_markdown", editor)
    return cli_runner.invoke(cli, ["create", *arguments, "--yes"])


def test_help_lists_bootstrap_commands(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "Work with BetterBorg" in result.output
    assert "create" in result.output
    assert "init" in result.output
    assert "trust" in result.output
    assert "version" in result.output


def test_create_help_registers_required_positional_name(
    cli_runner: CliRunner,
) -> None:
    result = cli_runner.invoke(cli, ["create", "--help"])

    assert result.exit_code == 0
    assert "Usage: cli create [OPTIONS] NAME" in result.output
    assert "--prd FILE" in result.output
    assert "--name" not in result.output


@pytest.mark.parametrize(
    "name",
    ["Not-Kebab", "not kebab", "not_kebab", "double--hyphen"],
)
def test_create_rejects_non_kebab_name_before_session_runs(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    monkeypatch: MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.chdir(committed_git_repo)
    monkeypatch.setattr(cli_module, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr(
        cli_module,
        "select_agent",
        lambda *_args, **_kwargs: pytest.fail("agent selection must not run"),
    )

    result = cli_runner.invoke(cli, ["create", name, "--yes"])

    assert result.exit_code == 1
    assert "kebab-case" in result.output
    assert not RepoPaths.discover(committed_git_repo).tracked_dir.exists()


def test_create_without_prd_runs_brainstorm_and_prints_plan_handoff(
    cli_runner: CliRunner,
    initialized_cli_repository: tuple[Repository, RepoPaths],
    monkeypatch: MonkeyPatch,
    configure_interactive_cli: Callable,
) -> None:
    repository, paths = initialized_cli_repository
    adapter = MockAdapter(name="openai").queue(
        MockResponse(
            payload={
                "questions": [],
                "prd_markdown": "# Release guide\n\nHelp maintainers ship safely.",
            }
        )
    )
    output: list[str] = []
    confirmations = iter((False, True))
    io = InteractiveIO(
        prompt=lambda _message: pytest.fail("brainstorm draft needs no answers"),
        confirm=lambda _message, _default: next(confirmations),
        write=output.append,
    )

    result = _invoke_create(
        cli_runner,
        monkeypatch,
        configure_interactive_cli,
        repository,
        adapter,
        io,
        "release-guide",
        editor=lambda _body: pytest.fail("editor was declined"),
    )

    assert result.exit_code == 0, result.output
    assert "No starting PRD was supplied" in adapter.calls[0].user_prompt
    confirmed = paths.tracked_dir / "prds" / "release-guide.md"
    assert confirmed.read_text(encoding="utf-8") == (
        "# Release guide\n\nHelp maintainers ship safely.\n"
    )
    assert output == ["# Release guide\n\nHelp maintainers ship safely.\n"]
    assert result.output.endswith("borg plan start release-guide\n")
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        assert store.get_borg_by_name(repository.id, "release-guide") is not None
        assert store.list_operations(repository.id) == []


def test_create_with_prd_propagates_and_preserves_exact_input(
    cli_runner: CliRunner,
    initialized_cli_repository: tuple[Repository, RepoPaths],
    monkeypatch: MonkeyPatch,
    configure_interactive_cli: Callable,
) -> None:
    repository, paths = initialized_cli_repository
    source = repository.root / "incoming.md"
    original = "# Existing PRD\n\nPreserve this input byte-for-byte.\n"
    source.write_text(original, encoding="utf-8")
    adapter = MockAdapter(name="openai").queue(
        MockResponse(
            payload={
                "questions": [],
                "prd_markdown": "# Improved PRD\n\nAgent revision.",
            }
        )
    )
    confirmations = iter((True, True))
    io = InteractiveIO(
        prompt=lambda _message: pytest.fail("improvement needs no answers"),
        confirm=lambda _message, _default: next(confirmations),
        write=lambda _message: None,
    )
    edited_inputs: list[str] = []

    def edit(body: str) -> str:
        edited_inputs.append(body)
        return "# Reviewed PRD\n\nHuman-approved requirements."

    result = _invoke_create(
        cli_runner,
        monkeypatch,
        configure_interactive_cli,
        repository,
        adapter,
        io,
        "improve-docs",
        "--prd",
        str(source),
        editor=edit,
    )

    assert result.exit_code == 0, result.output
    assert adapter.calls[0].user_prompt.startswith(f"## User\n\n{original}")
    assert source.read_text(encoding="utf-8") == original
    assert edited_inputs == ["# Improved PRD\n\nAgent revision.\n"]
    confirmed = paths.tracked_dir / "prds" / "improve-docs.md"
    assert confirmed.read_text(encoding="utf-8") == (
        "# Reviewed PRD\n\nHuman-approved requirements.\n"
    )
    assert result.output.endswith("borg plan start improve-docs\n")
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "improve-docs")
        assert borg is not None
        with store.locked_connection() as connection:
            session_borg_ids = connection.execute(
                "SELECT borg_id FROM prd_sessions WHERE repository_id = ?",
                (str(repository.id),),
            ).fetchall()
        assert [row["borg_id"] for row in session_borg_ids] == [str(borg.id)]
        assert store.list_operations(repository.id) == []


def test_create_does_not_print_plan_handoff_before_confirmation(
    cli_runner: CliRunner,
    initialized_cli_repository: tuple[Repository, RepoPaths],
    monkeypatch: MonkeyPatch,
    configure_interactive_cli: Callable,
) -> None:
    repository, paths = initialized_cli_repository
    adapter = MockAdapter(name="openai").queue(
        MockResponse(
            payload={
                "questions": [],
                "prd_markdown": "# Unconfirmed\n\nWait for approval.",
            }
        )
    )
    confirmations = iter((False, False))
    io = InteractiveIO(
        prompt=lambda _message: pytest.fail("draft needs no answers"),
        confirm=lambda _message, _default: next(confirmations),
        write=lambda _message: None,
    )

    result = _invoke_create(
        cli_runner,
        monkeypatch,
        configure_interactive_cli,
        repository,
        adapter,
        io,
        "wait-for-approval",
        editor=lambda _body: pytest.fail("editor was declined"),
    )

    assert result.exit_code == 0, result.output
    assert "borg plan start" not in result.output
    assert "draft saved without a confirmed PRD" in result.output
    assert not paths.tracked_dir.joinpath("prds/wait-for-approval.md").exists()
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        assert (
            store.get_borg_by_name(repository.id, "wait-for-approval") is not None
        )
        assert store.list_operations(repository.id) == []


def test_version_does_not_initialize_repository(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = cli_runner.invoke(cli, ["version"])

    assert result.exit_code == 0
    assert result.output == f"borg {__version__}\n"
    assert list(tmp_path.iterdir()) == []


def test_explicit_trust_command_records_current_workspace(
    cli_runner: CliRunner,
    git_repo: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    state_home = git_repo.parent / "machine-state"
    monkeypatch.chdir(git_repo)
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))

    result = cli_runner.invoke(cli, ["trust", "--yes"])

    assert result.exit_code == 0
    assert result.output == f"Trusted workspace: {git_repo}\n"
    store = TrustStore()
    assert store.path.is_relative_to(state_home)
    assert store.is_trusted(WorkspaceIdentity.discover(RepoPaths.discover(git_repo)))


def test_untrusted_noninteractive_command_rejects_before_callback(
    cli_runner: CliRunner,
    git_repo: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.chdir(git_repo)
    monkeypatch.setenv("XDG_STATE_HOME", str(git_repo.parent / "machine-state"))

    result = cli_runner.invoke(cli, ["trust"])

    assert result.exit_code == 1
    assert "workspace is not trusted" in result.output
    assert "Trusted workspace:" not in result.output


def test_interactive_trust_command_explains_host_access(
    cli_runner: CliRunner,
    git_repo: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.chdir(git_repo)
    monkeypatch.setenv("XDG_STATE_HOME", str(git_repo.parent / "machine-state"))
    monkeypatch.setattr(cli_module, "_stdin_is_interactive", lambda: True)

    result = cli_runner.invoke(cli, ["trust"], input="y\n")

    assert result.exit_code == 0
    assert "read and modify files" in result.output
    assert "execute commands on this machine" in result.output
    assert f"Trusted workspace: {git_repo}" in result.output
