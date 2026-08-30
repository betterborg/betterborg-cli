"""Signal dispatch, force arbitration, and real-process run-control tests."""

from __future__ import annotations

import io
import os
import signal
import threading
import time
from pathlib import Path

import pytest
from conftest import RealProcessHarness

from betterborg_cli.agent_runtime import CancellationToken, ForceTarget
from betterborg_cli.run_control import (
    DEFAULT_FORCE_GRACE_SECONDS,
    INTERRUPTED_EXIT_CODE,
    RunControl,
)


class _LockedProgress:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.called = threading.Event()

    def begin_cancellation(self) -> bool:
        with self.lock:
            self.called.set()
        return True


def _press_sigint() -> None:
    os.kill(os.getpid(), signal.SIGINT)


def test_first_sigint_defers_protected_exception_and_application_locks() -> None:
    cancel = CancellationToken()
    progress = _LockedProgress()
    exits: list[int] = []
    control = RunControl(cancel, progress=progress, exit_function=exits.append)
    cancel._lock.acquire()
    progress.lock.acquire()
    section = control.protected()
    control.install()
    section.__enter__()
    try:
        _press_sigint()
        assert not cancel.is_set()
        assert not progress.called.is_set()
    finally:
        cancel._lock.release()
        progress.lock.release()

    assert control.wait_for_cancellation(1)
    assert progress.called.wait(1)
    with pytest.raises(KeyboardInterrupt):
        section.__exit__(None, None, None)
    control.close()
    assert exits == []


def test_second_sigint_waits_for_every_window_and_rejects_new_windows() -> None:
    cancel = CancellationToken()
    exits: list[int] = []
    output = io.StringIO()
    control = RunControl(
        cancel,
        force_stream=output,
        exit_function=exits.append,
    ).install()
    windows = (cancel.registration_window(), cancel.registration_window())
    assert windows[0] is not windows[1]
    for window in windows:
        window.resource_created()
    releases = (threading.Event(), threading.Event())
    forced = (threading.Event(), threading.Event())
    registrations = []

    def publish(index: int) -> None:
        releases[index].wait()
        registrations.append(
            windows[index].register(
                lambda: None,
                forced[index].set,
                force_target=ForceTarget(f"worker-{index}"),
            )
        )

    workers = [threading.Thread(target=publish, args=(index,)) for index in range(2)]
    for worker in workers:
        worker.start()
    section = control.protected()
    section.__enter__()
    _press_sigint()
    assert control.wait_for_cancellation(1)
    _press_sigint()
    assert control.wait_for_force(1)
    with pytest.raises(RuntimeError, match="after force"):
        cancel.registration_window()

    releases[0].set()
    workers[0].join(timeout=1)
    assert forced[0].is_set()
    assert exits == []

    releases[1].set()
    workers[1].join(timeout=1)
    deadline = time.monotonic() + 1
    while not exits and time.monotonic() < deadline:
        time.sleep(0.01)
    assert forced[1].is_set()
    assert exits == [INTERRUPTED_EXIT_CODE]
    assert output.getvalue() == "Force stopping...\n"
    for registration in registrations:
        registration.unregister()
    with pytest.raises(KeyboardInterrupt):
        section.__exit__(None, None, None)
    control.close()


def test_precreation_failure_settles_forced_exit_without_target() -> None:
    cancel = CancellationToken()
    window = cancel.registration_window()
    exits: list[int] = []
    control = RunControl(cancel, exit_function=exits.append).install()
    section = control.protected()
    section.__enter__()
    _press_sigint()
    _press_sigint()
    assert control.wait_for_force(1)
    assert exits == []

    window.no_resource()
    deadline = time.monotonic() + 1
    while not exits and time.monotonic() < deadline:
        time.sleep(0.01)
    assert exits == [INTERRUPTED_EXIT_CODE]
    with pytest.raises(KeyboardInterrupt):
        section.__exit__(None, None, None)
    control.close()


def test_watchdog_uses_bounded_grace_and_forces_after_deadline() -> None:
    assert DEFAULT_FORCE_GRACE_SECONDS == 1.0
    cancel = CancellationToken(grace_seconds=0.02)
    exits: list[int] = []
    control = RunControl(
        cancel,
        force_grace_seconds=0.02,
        exit_function=exits.append,
    ).install()
    section = control.protected()
    section.__enter__()
    _press_sigint()

    assert control.wait_for_force(1)
    deadline = time.monotonic() + 1
    while not exits and time.monotonic() < deadline:
        time.sleep(0.01)
    assert exits == [INTERRUPTED_EXIT_CODE]
    with pytest.raises(KeyboardInterrupt):
        section.__exit__(None, None, None)
    control.close()


def test_force_includes_window_published_during_snapshot_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancel = CancellationToken()
    publication_started = threading.Event()
    release_publication = threading.Event()
    force_worker_started = threading.Event()
    release_force_worker = threading.Event()
    original_refresh = cancel._refresh_registry_snapshots
    original_force = cancel.force

    def delayed_refresh() -> None:
        publication_started.set()
        release_publication.wait()
        original_refresh()

    def delayed_force() -> None:
        force_worker_started.set()
        release_force_worker.wait()
        original_force()

    monkeypatch.setattr(cancel, "_refresh_registry_snapshots", delayed_refresh)
    monkeypatch.setattr(cancel, "force", delayed_force)
    exits: list[int] = []
    control = RunControl(cancel, exit_function=exits.append).install()
    section = control.protected()
    section.__enter__()
    _press_sigint()
    assert control.wait_for_cancellation(1)

    windows = []
    opener = threading.Thread(
        target=lambda: windows.append(cancel.registration_window())
    )
    opener.start()
    assert publication_started.wait(1)
    _press_sigint()
    assert force_worker_started.wait(1)
    assert exits == []

    release_publication.set()
    opener.join(timeout=1)
    assert not opener.is_alive()
    window = windows[0]
    window.resource_created()
    deadline = time.monotonic() + 1
    while not control._force_snapshot_captured and time.monotonic() < deadline:
        time.sleep(0.01)
    assert control._forced_windows == (window,)
    assert not window.is_settled
    assert exits == []

    forced = threading.Event()
    registration = window.register(
        lambda: None,
        forced.set,
        force_target=ForceTarget("opening-worker"),
    )
    assert forced.is_set()
    assert window.is_settled
    deadline = time.monotonic() + 1
    while not exits and time.monotonic() < deadline:
        time.sleep(0.01)
    assert exits == [INTERRUPTED_EXIT_CODE]

    release_force_worker.set()
    assert control.wait_for_force(1)
    assert registration.unregister()
    with pytest.raises(KeyboardInterrupt):
        section.__exit__(None, None, None)
    control.close()


def test_watchdog_escalates_while_cancellation_and_progress_locks_are_held(
) -> None:
    cancel = CancellationToken(grace_seconds=0.02)
    progress = _LockedProgress()
    exits: list[int] = []
    control = RunControl(
        cancel,
        progress=progress,
        force_grace_seconds=0.02,
        exit_function=exits.append,
    ).install()
    section = control.protected()
    section.__enter__()
    cancel._lock.acquire()
    progress.lock.acquire()
    try:
        _press_sigint()
        deadline = time.monotonic() + 1
        while not exits and time.monotonic() < deadline:
            time.sleep(0.01)
        assert exits == [INTERRUPTED_EXIT_CODE]
        assert control._force_requested
        assert not cancel.is_set()
        assert not progress.called.is_set()
    finally:
        cancel._lock.release()
        progress.lock.release()

    assert control.wait_for_force(1)
    assert progress.called.wait(1)
    with pytest.raises(KeyboardInterrupt):
        section.__exit__(None, None, None)
    control.close()


def test_failed_force_delivery_prevents_exit() -> None:
    cancel = CancellationToken()
    exits: list[int] = []

    def fail_force() -> None:
        raise RuntimeError("force delivery failed")

    registration = cancel.register(
        lambda: None,
        fail_force,
        force_target=ForceTarget("worker-without-process-group"),
    )
    control = RunControl(cancel, exit_function=exits.append).install()
    section = control.protected()
    section.__enter__()
    _press_sigint()
    assert control.wait_for_cancellation(1)
    _press_sigint()

    deadline = time.monotonic() + 1
    while control.dispatcher_error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert isinstance(control.dispatcher_error, RuntimeError)
    assert str(control.dispatcher_error) == "force delivery failed"
    assert not control.wait_for_force(0)
    assert exits == []

    assert registration.unregister()
    with pytest.raises(KeyboardInterrupt):
        section.__exit__(None, None, None)
    control.close()


def test_second_sigint_defers_exit_until_force_notice_is_written() -> None:
    output = io.StringIO()
    exits: list[tuple[int, str]] = []
    control = RunControl(
        force_stream=output,
        exit_function=lambda code: exits.append((code, output.getvalue())),
    ).install()
    section = control.protected()
    section.__enter__()
    _press_sigint()
    assert control.wait_for_cancellation(1)

    _press_sigint()

    deadline = time.monotonic() + 1
    while not exits and time.monotonic() < deadline:
        time.sleep(0.01)
    assert exits == [(INTERRUPTED_EXIT_CODE, "Force stopping...\n")]
    with pytest.raises(KeyboardInterrupt):
        section.__exit__(None, None, None)
    control.close()


@pytest.mark.skipif(
    not hasattr(signal, "setitimer"), reason="POSIX interval timer required"
)
def test_second_sigint_does_not_block_on_full_force_stream_pipe() -> None:
    read_fd, write_fd = os.pipe()
    force_stream = os.fdopen(write_fd, "w", closefd=False)
    exits: list[int] = []
    control = RunControl(
        force_stream=force_stream,
        exit_function=exits.append,
    ).install()
    section = control.protected()
    section.__enter__()
    previous_alarm_handler = signal.getsignal(signal.SIGALRM)

    try:
        os.set_blocking(write_fd, False)
        while True:
            try:
                os.write(write_fd, b"x" * 4096)
            except BlockingIOError:
                break
        os.set_blocking(write_fd, True)
        assert os.get_blocking(write_fd)

        _press_sigint()
        assert control.wait_for_cancellation(1)

        def fail_if_handler_blocks(_signum: int, _frame: object) -> None:
            raise TimeoutError("SIGINT handler blocked on the force stream")

        signal.signal(signal.SIGALRM, fail_if_handler_blocks)
        signal.setitimer(signal.ITIMER_REAL, 0.2)
        try:
            _press_sigint()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

        assert os.get_blocking(write_fd)
        assert exits == [INTERRUPTED_EXIT_CODE]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm_handler)
        with pytest.raises(KeyboardInterrupt):
            section.__exit__(None, None, None)
        control.close()
        force_stream.close()
        os.close(write_fd)
        os.close(read_fd)


def test_install_and_close_restore_handler_and_wakeup_fd() -> None:
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    previous_handler = signal.getsignal(signal.SIGINT)
    previous_wakeup = signal.set_wakeup_fd(write_fd, warn_on_full_buffer=False)
    control = RunControl(exit_function=lambda _code: None)
    try:
        control.install()
        assert signal.getsignal(signal.SIGINT) != previous_handler
        control.close()
        assert signal.getsignal(signal.SIGINT) == previous_handler
        restored_wakeup = signal.set_wakeup_fd(-1)
        assert restored_wakeup == write_fd
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        signal.set_wakeup_fd(previous_wakeup)
        os.close(read_fd)
        os.close(write_fd)


def test_real_process_harness_deadline_diagnostic_and_cleanup(
    real_process_harness: RealProcessHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TimeoutError, match="missing.*no tracked processes"):
        real_process_harness.wait_for_marker("missing", timeout=0.01)

    process = real_process_harness.launch_python(
        "import time\nwhile True: time.sleep(1)\n", name="cleanup"
    )
    real_process_harness.cleanup()
    assert process.poll() is not None

    checked: list[tuple[int, int]] = []
    monkeypatch.setattr(Path, "is_dir", lambda _path: False)
    monkeypatch.setattr(os, "kill", lambda pid, sig: checked.append((pid, sig)))
    assert real_process_harness._pid_exists(12345)
    assert checked == [(12345, 0)]


_WRAPPER_SOURCE = r"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from betterborg_cli.agent_runtime import CancellationToken, ForceTarget
from betterborg_cli.run_control import RunControl

root = Path(sys.argv[1])
helper = Path(sys.argv[2])
names = sys.argv[3:]
cancel = CancellationToken()
windows = [cancel.registration_window() for _name in names]

def worker(name, window):
    process = subprocess.Popen(
        [sys.executable, str(helper), str(root), name, "parent"],
        start_new_session=True,
    )
    window.resource_created()
    (root / f"{name}.spawned").write_text(str(process.pid))
    gate = root / f"release-{name}"
    while not gate.exists():
        time.sleep(0.01)

    def terminate():
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def force_and_reap():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    window.register(
        terminate,
        force_and_reap,
        force_target=ForceTarget(process.pid, process_group_id=process.pid),
    )

threads = [
    threading.Thread(target=worker, args=(name, window))
    for name, window in zip(names, windows, strict=True)
]
class Progress:
    def begin_cancellation(self):
        (root / "wrapper.cancelled").write_text("cancelled")
        return True

control = RunControl(cancel, progress=Progress())
with control:
    with control.protected():
        for thread in threads:
            thread.start()
        (root / "wrapper.ready").write_text(str(os.getpid()))
        while True:
            time.sleep(1)
"""


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
def test_rapid_double_sigint_kills_single_resistant_tree(
    real_process_harness: RealProcessHarness,
) -> None:
    helper = real_process_harness.resistant_argv("unused")[1]
    process = real_process_harness.launch_python(
        _WRAPPER_SOURCE,
        str(real_process_harness.root),
        helper,
        "single",
        name="single-wrapper",
    )
    real_process_harness.wait_for_marker("wrapper.ready")
    real_process_harness.wait_for_marker("single.parent.pid")
    real_process_harness.wait_for_marker("single.child.pid")

    real_process_harness.signal(process, signal.SIGINT)
    real_process_harness.wait_for_marker("wrapper.cancelled")
    real_process_harness.signal(process, signal.SIGINT)
    real_process_harness.release("release-single")

    assert real_process_harness.wait_for_exit(process) == INTERRUPTED_EXIT_CODE
    real_process_harness.assert_tree_absent("single")


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
def test_two_creation_windows_settle_before_wrapper_exits(
    real_process_harness: RealProcessHarness,
) -> None:
    helper = real_process_harness.resistant_argv("unused")[1]
    process = real_process_harness.launch_python(
        _WRAPPER_SOURCE,
        str(real_process_harness.root),
        helper,
        "first",
        "second",
        name="fanout-wrapper",
    )
    real_process_harness.wait_for_marker("wrapper.ready")
    for name in ("first", "second"):
        real_process_harness.wait_for_marker(f"{name}.parent.pid")
        real_process_harness.wait_for_marker(f"{name}.child.pid")

    real_process_harness.signal(process, signal.SIGINT)
    real_process_harness.wait_for_marker("wrapper.cancelled")
    real_process_harness.signal(process, signal.SIGINT)
    real_process_harness.release("release-first")
    real_process_harness.assert_tree_absent("first")
    assert process.poll() is None

    real_process_harness.release("release-second")
    assert real_process_harness.wait_for_exit(process) == INTERRUPTED_EXIT_CODE
    real_process_harness.assert_tree_absent("second")


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals required")
def test_real_precreation_failure_must_settle_before_exit(
    real_process_harness: RealProcessHarness,
) -> None:
    source = r"""
import os
import sys
import time
from pathlib import Path

from betterborg_cli.agent_runtime import CancellationToken
from betterborg_cli.run_control import RunControl

root = Path(sys.argv[1])
cancel = CancellationToken()
window = cancel.registration_window()
class Progress:
    def begin_cancellation(self):
        (root / "failure.cancelled").write_text("cancelled")
        return True
control = RunControl(cancel, progress=Progress())
with control:
    with control.protected():
        (root / "failure.ready").write_text(str(os.getpid()))
        while not (root / "release-failure").exists():
            time.sleep(0.01)
        window.no_resource()
        while True:
            time.sleep(1)
"""
    process = real_process_harness.launch_python(
        source,
        str(real_process_harness.root),
        name="failure-wrapper",
    )
    real_process_harness.wait_for_marker("failure.ready")
    real_process_harness.signal(process, signal.SIGINT)
    real_process_harness.wait_for_marker("failure.cancelled")
    real_process_harness.signal(process, signal.SIGINT)
    time.sleep(0.05)
    assert process.poll() is None

    real_process_harness.release("release-failure")
    assert real_process_harness.wait_for_exit(process) == INTERRUPTED_EXIT_CODE
