"""Streamed subprocess behavior for agent CLI adapters."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from conftest import RealProcessHarness

from betterborg_cli.agent_runtime import (
    CancellationRegistrationWindow,
    CancellationToken,
    run_streamed,
)
from betterborg_cli.agent_runtime.process import terminate_process


def test_run_streamed_persists_exact_observed_text_and_passes_stdin(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "process.log"
    observed: list[str] = []
    script = (
        "import sys; "
        "data=sys.stdin.read(); "
        "sys.stdout.buffer.write(('stdout:' + data + '\\r\\n').encode()); "
        "sys.stdout.buffer.write(b'partial'); "
        "sys.stdout.buffer.flush()"
    )

    exit_code = run_streamed(
        [sys.executable, "-c", script],
        tmp_path,
        "input-value",
        log_path,
        on_line=observed.append,
    )

    assert exit_code == 0
    with log_path.open(encoding="utf-8", newline="") as log_file:
        assert "".join(observed) == log_file.read()
    assert observed == ["stdout:input-value\r\n", "partial"]


def test_run_streamed_uses_argv_without_shell_interpretation(tmp_path: Path) -> None:
    log_path = tmp_path / "argv.log"

    exit_code = run_streamed(
        [sys.executable, "-c", "import sys; print(sys.argv[1])", "$(touch nope)"],
        tmp_path,
        "",
        log_path,
    )

    assert exit_code == 0
    assert log_path.read_text(encoding="utf-8").strip() == "$(touch nope)"
    assert not (tmp_path / "nope").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
def test_run_streamed_cancellation_reaps_resistant_descendant(
    real_process_harness: RealProcessHarness,
) -> None:
    cancel = CancellationToken(grace_seconds=0.1)

    exit_code = run_streamed(
        real_process_harness.resistant_argv("cancelled"),
        real_process_harness.root,
        "",
        real_process_harness.root / "cancelled.log",
        cancel,
        on_line=lambda _line: cancel.cancel(),
    )

    assert exit_code == -1
    real_process_harness.assert_tree_absent("cancelled")


def test_run_streamed_cancels_while_large_stdin_is_blocked(tmp_path: Path) -> None:
    cancel = CancellationToken(grace_seconds=0.1)

    exit_code = run_streamed(
        [
            sys.executable,
            "-c",
            "import time; print('ready', flush=True); time.sleep(30)",
        ],
        tmp_path,
        "x" * (10 * 1024 * 1024),
        tmp_path / "blocked-stdin.log",
        cancel,
        on_line=lambda _line: cancel.cancel(),
    )

    assert exit_code == -1


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
def test_terminate_process_reaps_group_after_leader_exit(
    real_process_harness: RealProcessHarness,
) -> None:
    process = subprocess.Popen(
        real_process_harness.resistant_argv("exited", leader_exits=True),
        start_new_session=True,
    )
    real_process_harness.processes.append(process)
    real_process_harness.wait_for_marker("exited.parent.pid")
    real_process_harness.wait_for_marker("exited.child.pid")
    assert process.wait(timeout=2) == 0

    terminate_process(process, pgid=process.pid, force_deadline=time.monotonic())

    parent_pid = int(real_process_harness.wait_for_marker("exited.parent.pid"))
    child_pid = int(real_process_harness.wait_for_marker("exited.child.pid"))
    assert not real_process_harness._pid_exists(parent_pid)
    assert not real_process_harness._pid_exists(child_pid)
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
def test_concurrent_processes_share_one_cancellation_deadline(
    real_process_harness: RealProcessHarness,
) -> None:
    grace_seconds = 0.35
    cancel = CancellationToken(grace_seconds=grace_seconds)
    results: list[int] = []
    errors: list[BaseException] = []

    def run(name: str) -> None:
        try:
            results.append(
                run_streamed(
                    real_process_harness.resistant_argv(name),
                    real_process_harness.root,
                    "",
                    real_process_harness.root / f"{name}.log",
                    cancel,
                )
            )
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=run, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for name in ("a", "b"):
        real_process_harness.wait_for_marker(f"{name}.parent.pid")
        real_process_harness.wait_for_marker(f"{name}.child.pid")

    started = time.monotonic()
    cancel.cancel()
    for thread in threads:
        thread.join(timeout=2)
    elapsed = time.monotonic() - started

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert results == [-1, -1]
    assert elapsed < grace_seconds * 1.75
    real_process_harness.assert_tree_absent("a")
    real_process_harness.assert_tree_absent("b")


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
def test_forced_cancellation_immediately_reaps_process_tree(
    real_process_harness: RealProcessHarness,
) -> None:
    cancel = CancellationToken()
    results: list[int] = []
    thread = threading.Thread(
        target=lambda: results.append(
            run_streamed(
                real_process_harness.resistant_argv("forced"),
                real_process_harness.root,
                "",
                real_process_harness.root / "forced.log",
                cancel,
            )
        )
    )
    thread.start()
    real_process_harness.wait_for_marker("forced.parent.pid")
    real_process_harness.wait_for_marker("forced.child.pid")

    started = time.monotonic()
    cancel.force()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert time.monotonic() - started < 0.5
    assert results == [-1]
    real_process_harness.assert_tree_absent("forced")


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
def test_observer_exception_cleans_process_tree(
    real_process_harness: RealProcessHarness,
) -> None:
    def fail_observer(_line: str) -> None:
        raise RuntimeError("observer failed")

    with pytest.raises(RuntimeError, match="observer failed"):
        run_streamed(
            real_process_harness.resistant_argv("observer"),
            real_process_harness.root,
            "",
            real_process_harness.root / "observer.log",
            on_line=fail_observer,
        )

    real_process_harness.assert_tree_absent("observer")


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals required")
def test_sigint_during_registration_reaches_late_process_and_exits_130(
    real_process_harness: RealProcessHarness,
) -> None:
    process = real_process_harness.launch_streamed_registration_wrapper(
        real_process_harness.resistant_argv("late-signal"),
        name="late-signal",
    )
    real_process_harness.wait_for_marker("late-signal.registration-gate")
    real_process_harness.wait_for_marker("late-signal.parent.pid")
    real_process_harness.wait_for_marker("late-signal.child.pid")

    real_process_harness.signal(process, signal.SIGINT)
    real_process_harness.wait_for_marker("late-signal.cancelled")
    real_process_harness.release("release-late-signal")

    assert real_process_harness.wait_for_exit(process) == 130
    real_process_harness.assert_tree_absent("late-signal")


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
def test_registration_exception_directly_cleans_created_process(
    real_process_harness: RealProcessHarness,
) -> None:
    process = real_process_harness.launch_streamed_registration_wrapper(
        real_process_harness.resistant_argv("registration-error"),
        name="registration-error",
        fail_registration=True,
    )
    real_process_harness.wait_for_marker("registration-error.registration-gate")
    real_process_harness.wait_for_marker("registration-error.parent.pid")
    real_process_harness.wait_for_marker("registration-error.child.pid")

    real_process_harness.release("release-registration-error")

    assert real_process_harness.wait_for_exit(process) == 73
    assert (
        real_process_harness.wait_for_marker("registration-error.error")
        == "injected registration failure"
    )
    assert (
        real_process_harness.wait_for_marker("registration-error.active-windows")
        == "0"
    )
    real_process_harness.assert_tree_absent("registration-error")


def test_run_streamed_already_cancelled_creates_empty_log(tmp_path: Path) -> None:
    cancel = CancellationToken()
    cancel.cancel()
    log_path = tmp_path / "cancelled-before-spawn.log"

    exit_code = run_streamed(
        [sys.executable, "-c", "raise SystemExit('must not run')"],
        tmp_path,
        "",
        log_path,
        cancel,
    )

    assert exit_code == -1
    assert log_path.read_text(encoding="utf-8") == ""


def test_run_streamed_force_before_registration_returns_cancelled(
    tmp_path: Path,
) -> None:
    class ForceBeforeRegistrationToken(CancellationToken):
        def registration_window(self) -> CancellationRegistrationWindow:
            self.force()
            return super().registration_window()

    cancel = ForceBeforeRegistrationToken()
    marker = tmp_path / "spawned"
    log_path = tmp_path / "forced-before-registration.log"

    exit_code = run_streamed(
        [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
        tmp_path,
        "",
        log_path,
        cancel,
    )

    assert exit_code == -1
    assert log_path.read_text(encoding="utf-8") == ""
    assert not marker.exists()


def test_run_streamed_rejects_shell_command_string(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="argv sequence"):
        run_streamed("echo unsafe", tmp_path, "", tmp_path / "log")


def test_run_streamed_preserves_nonzero_exit(tmp_path: Path) -> None:
    exit_code = run_streamed(
        [sys.executable, "-c", "raise SystemExit(23)"],
        tmp_path,
        "",
        tmp_path / "failed.log",
    )

    assert exit_code == 23
