"""Tests for the public CLI bootstrap."""

import io
import os
import selectors
import signal
import socket
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import click
import pytest
from click.testing import CliRunner
from pytest import MonkeyPatch

from betterborg_cli import __version__
from betterborg_cli import cli as cli_module
from betterborg_cli import run_control as run_control_module
from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.cli import cli
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.progress import RunProgress, StageSpec, StageState
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


@pytest.mark.parametrize(
    ("terminal_method", "terminal_state"),
    [("complete", StageState.COMPLETED), ("stop", StageState.STOPPED)],
)
def test_main_sigint_acknowledges_without_terminalizing_active_work(
    monkeypatch: MonkeyPatch,
    terminal_method: str,
    terminal_state: StageState,
) -> None:
    stream = io.StringIO()
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        cli_module,
        "RunProgress",
        lambda **kwargs: RunProgress(stream=stream, **kwargs),
    )

    @click.command()
    @click.pass_obj
    def command(run: cli_module.CliRunContext) -> None:
        observed["run"] = run
        run.progress.declare(StageSpec("work", "Work"))
        run.progress.start("work")
        observed["before_interrupt"] = stream.getvalue()
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except KeyboardInterrupt:
            deadline = time.monotonic() + 1
            while not run.progress.cancelling and time.monotonic() < deadline:
                time.sleep(0.001)
            observed["cancelled"] = run.cancellation.is_set()
            observed["acknowledged_state"] = run.progress.stages["work"].state
            observed["acknowledgement"] = stream.getvalue()
            getattr(run.progress, terminal_method)("work", "reconciled")
            observed["terminal_state"] = run.progress.stages["work"].state

    monkeypatch.setattr(cli_module, "cli", command)

    exit_code = cli_module.main([], prog_name="borg")

    assert exit_code == 130
    assert isinstance(observed["run"], cli_module.CliRunContext)
    assert observed["cancelled"] is True
    assert observed["acknowledged_state"] is StageState.RUNNING
    assert observed["acknowledgement"] == (
        f"{observed['before_interrupt']}stopping...\n"
    )
    assert observed["terminal_state"] is terminal_state
    run = observed["run"]
    assert isinstance(run, cli_module.CliRunContext)
    assert run.progress.closed
    assert "Error:" not in stream.getvalue()


def test_main_sigint_queues_acknowledgement_while_output_is_suspended(
    monkeypatch: MonkeyPatch,
) -> None:
    stream = io.StringIO()
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        cli_module,
        "RunProgress",
        lambda **kwargs: RunProgress(stream=stream, **kwargs),
    )

    @click.command()
    @click.pass_obj
    def command(run: cli_module.CliRunContext) -> None:
        interruption: KeyboardInterrupt | None = None
        run.progress.declare(StageSpec("work", "Work"))
        run.progress.start("work")
        with run.progress.suspend():
            observed["before_interrupt"] = stream.getvalue()
            try:
                os.kill(os.getpid(), signal.SIGINT)
            except KeyboardInterrupt as error:
                interruption = error
                deadline = time.monotonic() + 1
                while not run.progress.cancelling and time.monotonic() < deadline:
                    time.sleep(0.001)
                observed["cancelled"] = run.cancellation.is_set()
                observed["acknowledged_state"] = run.progress.stages["work"].state
                observed["suspended_output"] = stream.getvalue()
                run.progress.stop("work", "reconciled")
                observed["terminal_state"] = run.progress.stages["work"].state
                observed["terminal_output"] = stream.getvalue()
        run.progress.close()
        assert interruption is not None
        raise click.ClickException(
            "workflow interruption reconciled"
        ) from interruption

    monkeypatch.setattr(cli_module, "cli", command)

    exit_code = cli_module.main([], prog_name="borg")

    assert exit_code == 130
    assert observed["cancelled"] is True
    assert observed["acknowledged_state"] is StageState.RUNNING
    assert observed["terminal_state"] is StageState.STOPPED
    assert observed["suspended_output"] == observed["before_interrupt"]
    assert observed["terminal_output"] == observed["before_interrupt"]
    assert stream.getvalue().startswith(
        f"{observed['before_interrupt']}stopping...\n"
    )
    assert "stopped" in stream.getvalue()


def test_main_waits_for_first_sigint_dispatch_before_closing_progress(
    monkeypatch: MonkeyPatch,
) -> None:
    stream = io.StringIO()
    cancellation_started = threading.Event()
    release_cancellation = threading.Event()
    observed: dict[str, object] = {}
    original_cancel = cli_module.CancellationToken.cancel

    monkeypatch.setattr(
        cli_module,
        "RunProgress",
        lambda **kwargs: RunProgress(stream=stream, **kwargs),
    )

    def delayed_cancel(token: cli_module.CancellationToken) -> None:
        cancellation_started.set()
        assert release_cancellation.wait(1)
        original_cancel(token)

    monkeypatch.setattr(cli_module.CancellationToken, "cancel", delayed_cancel)

    @click.command()
    @click.pass_obj
    def command(run: cli_module.CliRunContext) -> None:
        observed["run"] = run
        os.kill(os.getpid(), signal.SIGINT)

    def inspect_before_release() -> None:
        assert cancellation_started.wait(1)
        run = observed["run"]
        assert isinstance(run, cli_module.CliRunContext)
        observed["cancelled_before_release"] = run.cancellation.is_set()
        observed["cancelling_before_release"] = run.progress.cancelling
        observed["closed_before_release"] = run.progress.closed
        release_cancellation.set()

    observer = threading.Thread(target=inspect_before_release)
    monkeypatch.setattr(cli_module, "cli", command)
    observer.start()
    try:
        exit_code = cli_module.main([], prog_name="borg")
    finally:
        release_cancellation.set()
        observer.join(timeout=1)

    run = observed["run"]
    assert isinstance(run, cli_module.CliRunContext)
    assert exit_code == 130
    assert not observer.is_alive()
    assert observed["cancelled_before_release"] is False
    assert observed["cancelling_before_release"] is False
    assert observed["closed_before_release"] is False
    assert run.cancellation.is_set()
    assert run.progress.cancelling
    assert run.progress.closed
    assert stream.getvalue().splitlines() == [
        "stopping...",
        "summary: 0 completed, 0 failed, 0 stopped — 0 retained",
    ]


def test_main_dispatches_sigint_with_socket_only_selector(
    monkeypatch: MonkeyPatch,
) -> None:
    selector_factory = run_control_module.selectors.DefaultSelector
    dispatcher_ready = threading.Event()

    class SocketOnlySelector:
        def __init__(self) -> None:
            self._selector = selector_factory()

        def register(
            self, fileobj: object, events: int
        ) -> selectors.SelectorKey:
            if not isinstance(fileobj, socket.socket):
                raise OSError(10038, "not a socket")
            key = self._selector.register(fileobj, events)
            dispatcher_ready.set()
            return key

        def select(
            self, timeout: float | None = None
        ) -> list[tuple[selectors.SelectorKey, int]]:
            return self._selector.select(timeout)

        def close(self) -> None:
            self._selector.close()

    @click.command()
    def command() -> None:
        assert dispatcher_ready.wait(1)
        os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(
        run_control_module.selectors,
        "DefaultSelector",
        SocketOnlySelector,
    )
    monkeypatch.setattr(cli_module, "cli", command)

    assert cli_module.main([], prog_name="borg") == 130


def test_main_preserves_click_usage_error_formatting(
    capsys: pytest.CaptureFixture,
) -> None:
    exit_code = cli_module.main(["version", "unexpected"], prog_name="borg")

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "Usage: borg version [OPTIONS]" in captured.err
    assert "Error: Got unexpected extra argument (unexpected)" in captured.err


def test_main_preserves_click_exception_formatting(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    @click.command()
    def command() -> None:
        raise click.ClickException("ordinary failure")

    monkeypatch.setattr(cli_module, "cli", command)

    exit_code = cli_module.main([], prog_name="borg")

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "Error: ordinary failure\n"


def test_main_surfaces_reconciliation_failure_after_interrupt(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    @click.command()
    def command() -> None:
        try:
            try:
                raise KeyboardInterrupt
            except KeyboardInterrupt as interruption:
                raise RuntimeError("durability reconciliation failed") from interruption
        except RuntimeError as failure:
            raise click.ClickException(
                "could not reconcile interruption"
            ) from failure

    monkeypatch.setattr(cli_module, "cli", command)

    exit_code = cli_module.main([], prog_name="borg")

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "Error: could not reconcile interruption\n"


@pytest.mark.parametrize("wrapper", [click.ClickException, click.Abort])
def test_main_refuses_interrupted_exit_with_unreconciled_started_work(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture,
    wrapper: type[click.ClickException] | type[click.Abort],
) -> None:
    observed: dict[str, cli_module.CliRunContext] = {}

    @click.command()
    @click.pass_obj
    def command(run: cli_module.CliRunContext) -> None:
        observed["run"] = run
        run.progress.declare(StageSpec("work", "Work"))
        run.progress.start("work")
        raise wrapper("workflow interrupted") from KeyboardInterrupt()

    monkeypatch.setattr(cli_module, "cli", command)

    exit_code = cli_module.main([], prog_name="borg")

    captured = capsys.readouterr()
    progress = observed["run"].progress
    assert exit_code == 1
    assert progress.closed is False
    assert progress.stages["work"].state is StageState.RUNNING
    assert captured.out == ""
    assert captured.err.endswith(
        "Error: cannot close progress with unresolved started records: "
        "stage 'work'\n"
    )


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
