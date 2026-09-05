"""Tests for the public CLI bootstrap."""

import io
import multiprocessing
import os
import re
import runpy
import selectors
import signal
import socket
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner
from progress_test_support import FakeClock, TTYStringIO
from pytest import MonkeyPatch

from betterborg_cli import __version__
from betterborg_cli import cli as cli_module
from betterborg_cli import repository_service as repository_service_module
from betterborg_cli import run_control as run_control_module
from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.cli import cli
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.progress import RunProgress, StageSpec, StageState
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_config import AgentStage
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
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
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
    assert "Work with Betterborg" in result.output
    assert "create" in result.output
    assert "init" in result.output
    assert "trust" in result.output
    assert "version" in result.output


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], cli_module.RootInvocation(None, False)),
        (["-h"], cli_module.RootInvocation(None, True)),
        (["--help"], cli_module.RootInvocation(None, True)),
        (["init", "--yes"], cli_module.RootInvocation("init", False)),
        (["init", "-h"], cli_module.RootInvocation("init", True)),
        (["init", "--help"], cli_module.RootInvocation("init", True)),
        (["--", "init", "--help"], cli_module.RootInvocation("init", True)),
        (["init", "--", "--help"], cli_module.RootInvocation("init", False)),
        (["init", "--", "-h"], cli_module.RootInvocation("init", False)),
        (["version", "--help"], cli_module.RootInvocation("version", True)),
    ],
)
def test_root_invocation_classifies_click_help_without_business_parsing(
    arguments: list[str],
    expected: cli_module.RootInvocation,
) -> None:
    assert cli_module._root_invocation(arguments) == expected


@pytest.mark.parametrize(
    "arguments",
    [["-h"], ["--help"], ["init", "-h"], ["init", "--help"]],
)
def test_main_root_and_init_help_have_no_startup_projection(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture,
    arguments: list[str],
) -> None:
    expected = CliRunner().invoke(
        cli,
        arguments,
        prog_name="betterborg",
    )
    stream = TTYStringIO()
    construction: list[dict[str, object]] = []

    def progress_factory(**kwargs) -> RunProgress:
        construction.append(kwargs)
        return RunProgress(stream=stream, **kwargs)

    monkeypatch.setattr(cli_module, "RunProgress", progress_factory)

    exit_code = cli_module.main(
        arguments,
        prog_name="betterborg",
    )

    captured = capsys.readouterr()
    assert expected.exit_code == 0
    assert exit_code == 0
    assert captured.out == expected.output
    assert captured.err == ""
    assert construction == [{"machine_readable": False}]
    assert stream.getvalue() == ""


@pytest.mark.parametrize(
    ("arguments", "startup_expected"),
    [
        (["init", "--yes"], True),
        (["init", "--", "--help"], True),
        (["init", "--yes", "--json"], False),
        (["version"], False),
        ([], False),
    ],
)
def test_main_selects_startup_projection_only_for_direct_human_init(
    monkeypatch: MonkeyPatch,
    arguments: list[str],
    startup_expected: bool,
) -> None:
    stream = TTYStringIO()
    construction: list[dict[str, object]] = []

    def progress_factory(**kwargs) -> RunProgress:
        construction.append(kwargs)
        return RunProgress(stream=stream, **kwargs)

    class StubRoot:
        def main(self, **kwargs) -> int:
            kwargs["obj"].progress.stop_display()
            return 0

    monkeypatch.setattr(cli_module, "RunProgress", progress_factory)
    monkeypatch.setattr(cli_module, "cli", StubRoot())

    assert cli_module.main(arguments, prog_name="betterborg") == 0
    assert len(construction) == 1
    if startup_expected:
        assert construction[0] == {
            "machine_readable": False,
            "startup_label": "Starting betterborg init",
            "startup_pending": (
                "Discover evidence",
                "Analyze repository",
                "Generate role prompts",
                "Draft improvement PRDs",
            ),
        }
    else:
        assert "startup_label" not in construction[0]
        assert "startup_pending" not in construction[0]


def test_progress_observed_work_requires_a_nonpending_stage() -> None:
    progress = RunProgress(enabled=False)

    assert cli_module._progress_has_observed_work(progress) is False

    progress.declare(StageSpec("waiting", "Waiting"))
    assert cli_module._progress_has_observed_work(progress) is False
    assert progress.stages["waiting"].state is StageState.PENDING

    progress.declare(StageSpec("active", "Active"))
    progress.start("active")
    assert cli_module._progress_has_observed_work(progress) is True
    assert progress.stages["waiting"].state is StageState.PENDING


@pytest.mark.parametrize(
    ("failure", "expected_error", "expected_stage_keys"),
    [
        ("trust", "workspace was not trusted", ()),
        ("path-discovery", "repository path unavailable", ()),
        (
            "configuration",
            "invalid repository configuration",
            ("discover", "analyze", "prompts", "improvement-prds"),
        ),
        (
            "default-branch",
            "default branch unavailable",
            ("discover", "analyze", "prompts", "improvement-prds"),
        ),
        (
            "agent-selection",
            "bootstrap agent unavailable",
            ("discover", "analyze", "prompts", "improvement-prds"),
        ),
    ],
)
def test_init_prework_failures_leave_root_progress_unobserved(
    committed_git_repo: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture,
    failure: str,
    expected_error: str,
    expected_stage_keys: tuple[str, ...],
) -> None:
    paths = RepoPaths.discover(committed_git_repo)
    stream = io.StringIO()
    reporters: list[RunProgress] = []

    def progress_factory(**kwargs: object) -> RunProgress:
        progress = RunProgress(stream=stream, enabled=False, **kwargs)
        reporters.append(progress)
        return progress

    arguments = ["init", "--yes"]
    monkeypatch.chdir(committed_git_repo)
    monkeypatch.setenv(
        "XDG_STATE_HOME", str(committed_git_repo.parent / "machine-state")
    )
    monkeypatch.setattr(cli_module, "RunProgress", progress_factory)
    monkeypatch.setattr(cli_module, "_stdin_is_interactive", lambda: False)

    if failure == "trust":
        arguments = ["init"]
        monkeypatch.setattr(cli_module, "_stdin_is_interactive", lambda: True)
        monkeypatch.setattr(
            cli_module.click, "confirm", lambda *_args, **_kwargs: False
        )
    elif failure == "path-discovery":
        monkeypatch.setattr(
            RepoPaths,
            "discover",
            lambda **_kwargs: (_ for _ in ()).throw(
                ValueError("repository path unavailable")
            ),
        )
    elif failure == "configuration":
        paths.tracked_dir.mkdir()
        paths.tracked_dir.joinpath("config.toml").write_text(
            "version = [\n", encoding="utf-8"
        )
    elif failure == "default-branch":
        monkeypatch.setattr(
            repository_service_module,
            "_default_branch",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("default branch unavailable")
            ),
        )
    elif failure == "agent-selection":
        monkeypatch.setattr(
            cli_module,
            "select_agent",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("bootstrap agent unavailable")
            ),
        )
    else:  # pragma: no cover - the parameter table is exhaustive
        raise AssertionError(f"unknown failure fixture: {failure}")

    try:
        exit_code = cli_module.main(arguments, prog_name="betterborg")
    finally:
        for reporter in reporters:
            reporter.stop_display()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert expected_error in captured.err
    assert len(reporters) == 1
    assert tuple(reporters[0].records) == expected_stage_keys
    assert all(
        record.state is StageState.PENDING
        for record in reporters[0].records.values()
    )
    assert cli_module._progress_has_observed_work(reporters[0]) is False
    assert stream.getvalue() == ""


def test_main_enables_multiprocessing_before_click_dispatch(
    monkeypatch: MonkeyPatch,
) -> None:
    events: list[str] = []

    @click.command()
    def command() -> None:
        events.append("click")

    monkeypatch.setattr(
        cli_module.multiprocessing,
        "freeze_support",
        lambda: events.append("freeze-support"),
    )
    monkeypatch.setattr(cli_module, "cli", command)

    assert cli_module.main([], prog_name="betterborg") == 0
    assert events == ["freeze-support", "click"]


def test_executable_module_enables_multiprocessing_before_cli_dispatch(
    monkeypatch: MonkeyPatch,
) -> None:
    events: list[str] = []

    monkeypatch.setattr(
        multiprocessing,
        "freeze_support",
        lambda: events.append("freeze-support"),
    )
    monkeypatch.setattr(
        cli_module,
        "main",
        lambda: events.append("click") or 23,
    )

    with pytest.raises(SystemExit, match="23"):
        runpy.run_module("betterborg_cli", run_name="__main__")

    assert events == ["freeze-support", "click"]


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

    exit_code = cli_module.main([], prog_name="betterborg")

    assert exit_code == 130
    assert isinstance(observed["run"], cli_module.CliRunContext)
    assert observed["cancelled"] is True
    assert observed["acknowledged_state"] is StageState.RUNNING
    assert observed["acknowledgement"] == (
        f"{observed['before_interrupt']}stopping…\n"
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

    exit_code = cli_module.main([], prog_name="betterborg")

    assert exit_code == 130
    assert observed["cancelled"] is True
    assert observed["acknowledged_state"] is StageState.RUNNING
    assert observed["terminal_state"] is StageState.STOPPED
    assert observed["suspended_output"] == observed["before_interrupt"]
    assert observed["terminal_output"] == observed["before_interrupt"]
    assert stream.getvalue().startswith(
        f"{observed['before_interrupt']}stopping…\n"
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
        exit_code = cli_module.main([], prog_name="betterborg")
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
    lines = stream.getvalue().splitlines()
    assert len(lines) == 2
    assert lines[0] == "stopping…"
    assert re.fullmatch(
        r"0 of 0 stages finished in \d+:\d{2}; none failed or stopped\.",
        lines[1],
    )


def test_json_init_reconciles_cancellation_on_run_control_reporter(
    committed_git_repo: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    stream = io.StringIO()
    reporters: list[RunProgress] = []

    def progress_factory(**kwargs) -> RunProgress:
        progress = RunProgress(stream=stream, **kwargs)
        reporters.append(progress)
        return progress

    class StubRepositoryService:
        def __init__(
            self,
            paths: RepoPaths,
            _store: SqliteStore,
            _agent_factory,
            *,
            cancel: cli_module.CancellationToken,
            progress: RunProgress,
        ) -> None:
            assert reporters == [progress]
            self.paths = paths
            self.cancel = cancel
            self.progress = progress

        def initialize(self):
            self.progress.declare(StageSpec("work", "Structured work"))
            self.progress.start("work")
            try:
                os.kill(os.getpid(), signal.SIGINT)
            except KeyboardInterrupt:
                deadline = time.monotonic() + 1
                while not self.progress.cancelling and time.monotonic() < deadline:
                    time.sleep(0.001)
                assert self.cancel.is_set()
                self.progress.stop("work", "reconciled")
            return SimpleNamespace(
                repository=Repository(root=self.paths.root),
                initialized=False,
                improvement_prds=(),
                analysis=SimpleNamespace(overall_score=3.0),
            )

    monkeypatch.chdir(committed_git_repo)
    monkeypatch.setenv(
        "XDG_STATE_HOME", str(committed_git_repo.parent / "machine-state")
    )
    monkeypatch.setattr(cli_module, "RunProgress", progress_factory)
    monkeypatch.setattr(cli_module, "RepositoryService", StubRepositoryService)

    exit_code = cli_module.main(
        ["init", "--yes", "--json"],
        prog_name="betterborg",
    )

    captured = capsys.readouterr()
    assert exit_code == 130
    assert len(reporters) == 1
    assert reporters[0].stages["work"].state is StageState.STOPPED
    assert reporters[0].closed
    assert stream.getvalue() == ""
    assert captured.out == ""
    assert captured.err == ""


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

    assert cli_module.main([], prog_name="betterborg") == 130


def test_main_preserves_click_usage_error_formatting(
    capsys: pytest.CaptureFixture,
) -> None:
    exit_code = cli_module.main(["version", "unexpected"], prog_name="betterborg")

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "Usage: betterborg version [OPTIONS]" in captured.err
    assert "Error: Got unexpected extra argument (unexpected)" in captured.err


def test_main_preserves_click_exception_formatting(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    @click.command()
    def command() -> None:
        raise click.ClickException("ordinary failure")

    monkeypatch.setattr(cli_module, "cli", command)

    exit_code = cli_module.main([], prog_name="betterborg")

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "Error: ordinary failure\n"


@pytest.mark.parametrize("environment", [{}, {"NO_COLOR": "1"}, {"TERM": "dumb"}])
def test_init_cli_suspends_root_progress_across_every_interactive_boundary(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    monkeypatch: MonkeyPatch,
    environment: dict[str, str],
) -> None:
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
    run = cli_module.CliRunContext(cli_module.CancellationToken(), progress)
    adapter = MockAdapter(name="openai").queue(
        MockResponse(
            payload={
                "questions": ["Who needs this first?"],
                "prd_markdown": None,
            }
        )
    ).queue(
        MockResponse(
            payload={
                "questions": [],
                "prd_markdown": "# CLI draft\n\nReady for review.",
            }
        ),
    )
    snapshots: list[tuple[str, str, str]] = []

    def cross_heartbeat(boundary: str) -> None:
        before = stream.getvalue()
        clock.now += 10
        progress.refresh()
        snapshots.append((boundary, before, stream.getvalue()))

    class StubRepositoryService:
        def __init__(
            self,
            paths: RepoPaths,
            store: SqliteStore,
            _agent_factory,
            *,
            cancel=None,
            progress: RunProgress | None = None,
        ) -> None:
            assert cancel is run.cancellation
            assert progress is run.progress
            self.paths = paths
            self.store = store
            self.progress = progress

        def initialize(self):
            repository = Repository(root=self.paths.root)
            self.store.add_repository(repository)
            self.paths.tracked_dir.mkdir(parents=True, exist_ok=True)
            self.paths.tracked_dir.joinpath("config.toml").write_text(
                "version = 1\n\n"
                "[repository]\n"
                f'id = "{repository.id}"\n'
                'default_branch = "main"\n',
                encoding="utf-8",
            )
            assert self.progress is not None
            self.progress.declare(StageSpec("active", "Active work"))
            self.progress.start("active")
            return SimpleNamespace(
                repository=repository,
                analysis=SimpleNamespace(overall_score=3.0),
                initialized=True,
                improvement_prds=(),
            )

    original_echo = click.echo
    original_prompt = click.prompt
    original_confirm = click.confirm

    def echo(message=None, *args, **kwargs):
        cross_heartbeat(f"output:{message}")
        return original_echo(message, *args, **kwargs)

    def prompt(*args, **kwargs):
        cross_heartbeat("prompt")
        return original_prompt(*args, **kwargs)

    def confirm(*args, **kwargs):
        cross_heartbeat("confirmation")
        return original_confirm(*args, **kwargs)

    def edit(_body: str) -> str:
        cross_heartbeat("editor")
        return "# Edited CLI draft\n\nApproved requirements."

    monkeypatch.chdir(committed_git_repo)
    monkeypatch.setenv(
        "XDG_STATE_HOME", str(committed_git_repo.parent / "machine-state")
    )
    monkeypatch.setattr(cli_module, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr(cli_module, "RepositoryService", StubRepositoryService)
    monkeypatch.setattr(
        cli_module, "select_agent", lambda *_args, **_kwargs: adapter
    )
    monkeypatch.setattr(cli_module.click, "echo", echo)
    monkeypatch.setattr(cli_module.click, "prompt", prompt)
    monkeypatch.setattr(cli_module.click, "confirm", confirm)
    monkeypatch.setattr(cli_module, "_edit_markdown", edit)

    result = cli_runner.invoke(
        cli,
        ["init", "--yes"],
        input="3\ncli-suspended\nMaintainers\ny\ny\n",
        obj=run,
    )

    assert result.exit_code == 0, result.output
    assert any(
        boundary.startswith("output:Initialized repository")
        for boundary, *_ in snapshots
    )
    assert any(boundary == "prompt" for boundary, *_ in snapshots)
    assert any(boundary == "confirmation" for boundary, *_ in snapshots)
    assert any(boundary == "editor" for boundary, *_ in snapshots)
    assert snapshots and all(before == after for _, before, after in snapshots)
    assert committed_git_repo.joinpath(
        ".betterborg/prds/cli-suspended.md"
    ).read_text(encoding="utf-8") == (
        "# Edited CLI draft\n\nApproved requirements.\n"
    )


def test_main_surfaces_reconciliation_failure_after_sigint(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    observed: dict[str, cli_module.CliRunContext] = {}

    @click.command()
    @click.pass_obj
    def command(run: cli_module.CliRunContext) -> None:
        observed["run"] = run
        run.progress.declare(StageSpec("work", "Work"))
        run.progress.start("work")
        try:
            try:
                os.kill(os.getpid(), signal.SIGINT)
            except KeyboardInterrupt as interruption:
                deadline = time.monotonic() + 1
                while not run.progress.cancelling and time.monotonic() < deadline:
                    time.sleep(0.001)
                raise RuntimeError("durability reconciliation failed") from interruption
        except RuntimeError as failure:
            run.progress.fail("work", str(failure))
            raise click.ClickException(
                "could not reconcile interruption"
            ) from failure

    monkeypatch.setattr(cli_module, "cli", command)

    exit_code = cli_module.main([], prog_name="betterborg")

    captured = capsys.readouterr()
    assert exit_code == 1
    run = observed["run"]
    assert run.cancellation.is_set()
    assert run.progress.cancelling
    assert run.progress.stages["work"].state is StageState.FAILED
    assert captured.out == ""
    assert "✖ Work" in captured.err
    assert "durability reconciliation failed" in captured.err
    assert captured.err.endswith("Error: could not reconcile interruption\n")


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

    exit_code = cli_module.main([], prog_name="betterborg")

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
    selected_stages: list[AgentStage] = []

    configure_interactive_cli(
        repository.root,
        adapter,
        io,
        state_home=repository.root.parent / "machine-state",
    )

    def select_mock(_config, stage, _paths, **_kwargs):
        selected_stages.append(stage)
        return adapter

    monkeypatch.setattr(cli_module, "select_agent", select_mock)
    monkeypatch.setattr(
        cli_module,
        "_edit_markdown",
        lambda _body: pytest.fail("editor was declined"),
    )

    result = cli_runner.invoke(cli, ["create", "release-guide", "--yes"])

    assert result.exit_code == 0, result.output
    assert selected_stages == [AgentStage.REQUIREMENTS]
    assert "No starting PRD was supplied" in adapter.calls[0].user_prompt
    confirmed = paths.tracked_dir / "prds" / "release-guide.md"
    assert confirmed.read_text(encoding="utf-8") == (
        "# Release guide\n\nHelp maintainers ship safely.\n"
    )
    assert output == ["# Release guide\n\nHelp maintainers ship safely.\n"]
    assert result.output.endswith("betterborg plan start release-guide\n")
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
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
    assert result.output.endswith("betterborg plan start improve-docs\n")
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
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
    assert "betterborg plan start" not in result.output
    assert "draft saved without a confirmed PRD" in result.output
    assert not paths.tracked_dir.joinpath("prds/wait-for-approval.md").exists()
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        assert (
            store.get_borg_by_name(repository.id, "wait-for-approval") is not None
        )
        assert store.list_operations(repository.id) == []


@pytest.mark.parametrize(
    ("confirmation_number", "name"),
    [(1, "interrupt-editor-review"), (2, "interrupt-final-approval")],
    ids=["editor-confirmation", "final-confirmation"],
)
def test_main_sigint_during_prd_confirmation_stops_without_publishing(
    initialized_cli_repository: tuple[Repository, RepoPaths],
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture,
    confirmation_number: int,
    name: str,
) -> None:
    repository, paths = initialized_cli_repository
    adapter = MockAdapter(name="openai").queue(
        MockResponse(
            payload={
                "questions": [],
                "prd_markdown": "# Interrupted draft\n\nNever publish this.",
            }
        )
    )
    reporters: list[RunProgress] = []

    def progress_factory(**kwargs) -> RunProgress:
        progress = RunProgress(stream=io.StringIO(), **kwargs)
        reporters.append(progress)
        return progress

    prompts = 0

    def interrupt_at_confirmation(_prompt: str) -> str:
        nonlocal prompts
        prompts += 1
        if prompts == confirmation_number:
            os.kill(os.getpid(), signal.SIGINT)
        return "n"

    monkeypatch.chdir(repository.root)
    monkeypatch.setenv(
        "XDG_STATE_HOME", str(repository.root.parent / "machine-state")
    )
    monkeypatch.setattr(cli_module, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr(
        cli_module, "select_agent", lambda *_args, **_kwargs: adapter
    )
    monkeypatch.setattr(cli_module, "RunProgress", progress_factory)
    monkeypatch.setattr(click.termui, "visible_prompt_func", interrupt_at_confirmation)
    monkeypatch.setattr(
        cli_module,
        "_edit_markdown",
        lambda _body: pytest.fail("interrupted work must not launch the editor"),
    )

    exit_code = cli_module.main(
        ["create", name, "--yes"],
        prog_name="betterborg",
    )

    captured = capsys.readouterr()
    assert exit_code == 130
    assert prompts == confirmation_number
    assert len(reporters) == 1
    assert reporters[0].stages["requirements"].state is StageState.STOPPED
    assert reporters[0].stages["requirements"].result == "interrupted"
    assert reporters[0].closed
    assert captured.err == ""
    assert "Error:" not in captured.out
    assert "Aborted!" not in captured.out
    assert "betterborg plan start" not in captured.out
    assert "draft saved" not in captured.out
    assert not paths.tracked_dir.joinpath("prds", f"{name}.md").exists()
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, name)
        assert borg is not None
        session = store.get_prd_session_for_borg(borg.id)
        assert session is not None
        assert [turn.role for turn in store.list_prd_turns(session.id)] == [
            "user",
            "assistant",
        ]


def test_version_does_not_initialize_repository(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = cli_runner.invoke(cli, ["version"])

    assert result.exit_code == 0
    assert result.output == f"betterborg {__version__}\n"
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


def test_trust_command_forwards_root_cancellation_token(
    cli_runner: CliRunner,
    git_repo: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    cancel = cli_module.CancellationToken()
    run = cli_module.CliRunContext(cancel, RunProgress())
    observed: list[tuple[str, object]] = []
    original_discover = RepoPaths.discover.__func__

    def discover(cls, start=None, *, cancel=None, command_runner=None):
        observed.append(("paths", cancel))
        if command_runner is None:
            return original_discover(cls, start or git_repo)
        return original_discover(
            cls,
            start or git_repo,
            command_runner=command_runner,
        )

    def trust(_paths: RepoPaths, **kwargs) -> None:
        observed.append(("trust", kwargs["cancel"]))

    monkeypatch.chdir(git_repo)
    monkeypatch.setattr(RepoPaths, "discover", classmethod(discover))
    monkeypatch.setattr(cli_module, "require_workspace_trust", trust)

    result = cli_runner.invoke(cli, ["trust", "--yes"], obj=run)

    assert result.exit_code == 0, result.output
    assert observed == [("paths", cancel), ("trust", cancel)]


def test_trusted_callback_suspends_progress_while_trust_is_required(
    cli_runner: CliRunner,
    git_repo: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    run = cli_module.CliRunContext(
        cli_module.CancellationToken(),
        RunProgress(enabled=False),
    )
    paths = RepoPaths.discover(git_repo)
    events: list[str] = []

    class Suspension:
        def __enter__(self) -> None:
            events.append("suspend")

        def __exit__(self, *_args: object) -> None:
            events.append("resume")

    @click.command()
    @cli_module._trusted_workspace_callback
    def command(paths: RepoPaths, cancel: object) -> None:
        events.append("callback")

    def require_trust(_paths: RepoPaths, **_kwargs: object) -> None:
        assert _paths is paths
        events.append("trust")

    monkeypatch.setattr(RepoPaths, "discover", lambda **_kwargs: paths)
    monkeypatch.setattr(
        cli_module,
        "_suspend_progress",
        lambda progress: Suspension()
        if progress is run.progress
        else pytest.fail("root progress was not suspended"),
    )
    monkeypatch.setattr(cli_module, "require_workspace_trust", require_trust)

    result = cli_runner.invoke(command, obj=run)

    assert result.exit_code == 0, result.output
    assert events == ["suspend", "trust", "resume", "callback"]


def test_trusted_callback_reuses_discovered_paths_and_root_token(
    cli_runner: CliRunner,
    git_repo: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    cancel = cli_module.CancellationToken()
    run = cli_module.CliRunContext(cancel, RunProgress())
    paths = RepoPaths.discover(git_repo)
    discoveries: list[object] = []
    callback_arguments: list[tuple[RepoPaths, object]] = []

    def discover(cls, start=None, *, cancel=None, command_runner=None):
        discoveries.append(cancel)
        return paths

    @click.command()
    @cli_module._trusted_workspace_callback
    def command(paths: RepoPaths, cancel: object) -> None:
        callback_arguments.append((paths, cancel))

    monkeypatch.chdir(git_repo)
    monkeypatch.setattr(RepoPaths, "discover", classmethod(discover))
    monkeypatch.setattr(
        cli_module,
        "require_workspace_trust",
        lambda _paths, **_kwargs: None,
    )

    result = cli_runner.invoke(command, obj=run)

    assert result.exit_code == 0, result.output
    assert discoveries == [cancel]
    assert callback_arguments == [(paths, cancel)]


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
