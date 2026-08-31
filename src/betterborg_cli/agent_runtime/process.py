"""Stream non-shell subprocess output with cooperative cancellation."""

from __future__ import annotations

import contextlib
import io
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from betterborg_cli.agent_runtime.base import (
    CancellationDeliveryError,
    CancellationRegistration,
    CancellationRegistrationRejected,
    CancellationToken,
    ForceTarget,
)

_IO_CHUNK_SIZE = 64 * 1024
_PROCESS_POLL_SECONDS = 0.05


class ProcessRunner(Protocol):
    """Callable contract used by native CLI adapters."""

    def __call__(
        self,
        command: Sequence[str],
        cwd: Path,
        stdin_text: str,
        log_path: Path,
        cancel: CancellationToken | None = None,
        env: Mapping[str, str] | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> int: ...


def run_captured(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    input: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    check: bool = False,
    cancel: CancellationToken | None = None,
    terminate_on_cancel: bool = True,
    deadline: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an argv sequence and capture text output with registered cleanup.

    Ordinary work stops when ``cancel`` is set. Shutdown cleanup may opt out of
    first-cancel termination, but only while bounded by the token's absolute
    force deadline or an explicit absolute deadline. Forced cancellation always
    terminates the process group.
    """
    if isinstance(command, str | bytes) or not command:
        raise ValueError("command must be a non-empty argv sequence")
    argv = list(command)
    _validate_cleanup_mode(cancel, terminate_on_cancel, deadline)
    if cancel is not None and cancel.is_set() and terminate_on_cancel:
        return subprocess.CompletedProcess(argv, -1, "", "")

    window = None
    process: subprocess.Popen[str] | None = None
    registration: CancellationRegistration | None = None
    process_group_id: int | None = None
    force_target: ForceTarget | None = None
    stdout: str | None = None
    stderr: str | None = None
    timed_out = False
    deadline_expired = False

    try:
        try:
            window = (
                cancel.registration_window() if cancel is not None else None
            )
        except CancellationRegistrationRejected:
            return subprocess.CompletedProcess(argv, -1, "", "")
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                stdin=subprocess.PIPE if input is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(env) if env is not None else None,
                text=True,
                start_new_session=os.name == "posix",
            )
        except BaseException:
            if window is not None:
                window.no_resource()
            raise

        if window is not None:
            window.resource_created()
        process_group_id = _validated_process_group(process)
        if window is not None:
            force_target = ForceTarget(
                process.pid,
                process_group_id=process_group_id,
            )
            try:
                registration = window.register(
                    lambda: _request_termination(process, process_group_id),
                    lambda: terminate_process(
                        process,
                        pgid=process_group_id,
                        force_deadline=time.monotonic(),
                    ),
                    terminate_on_cancel=terminate_on_cancel,
                    force_target=force_target,
                )
            except CancellationDeliveryError as error:
                registration = error.registration
                raise

        started = time.monotonic()
        input_text = input
        timeout_deadline = None if timeout is None else started + timeout
        while True:
            if cancel is not None and cancel.is_set() and terminate_on_cancel:
                break

            active_deadline = _captured_deadline(
                timeout_deadline,
                deadline,
                cancel,
                terminate_on_cancel,
            )
            if active_deadline is not None and active_deadline <= time.monotonic():
                if cancel is not None and cancel.is_set():
                    deadline_expired = True
                elif (
                    timeout_deadline is not None
                    and active_deadline == timeout_deadline
                ):
                    timed_out = True
                else:
                    deadline_expired = True
                break

            wait_timeout = (
                _PROCESS_POLL_SECONDS
                if active_deadline is None
                else min(
                    _PROCESS_POLL_SECONDS,
                    max(0.0, active_deadline - time.monotonic()),
                )
            )
            try:
                stdout, stderr = process.communicate(
                    input=input_text,
                    timeout=wait_timeout,
                )
                break
            except subprocess.TimeoutExpired:
                # ``communicate`` retains both buffered output and unwritten
                # input across retries. Input must only be supplied initially.
                input_text = None
                continue
    finally:
        if process is not None:
            force_deadline = (
                time.monotonic()
                if timed_out or deadline_expired
                else (cancel.force_deadline if cancel is not None else None)
            )
            terminate_process(
                process,
                pgid=process_group_id,
                force_deadline=force_deadline,
            )
            if stdout is None or stderr is None:
                stdout, stderr = process.communicate()
            if registration is not None:
                registration.unregister()
            elif window is not None and not window.is_settled:
                if force_target is None:
                    force_target = ForceTarget(
                        process.pid,
                        process_group_id=process_group_id,
                    )
                window.publish_cleaned_resource(
                    lambda: _request_termination(process, process_group_id),
                    lambda: terminate_process(
                        process,
                        pgid=process_group_id,
                        force_deadline=time.monotonic(),
                    ),
                    force_target=force_target,
                )

    assert process is not None
    assert stdout is not None
    assert stderr is not None
    cancelled = cancel is not None and (
        (terminate_on_cancel and cancel.is_set())
        or cancel.is_forced()
        or (deadline_expired and cancel.is_set())
    )
    if cancelled:
        return subprocess.CompletedProcess(argv, -1, stdout, stderr)
    if timed_out or deadline_expired:
        duration = timeout if timed_out else max(0.0, time.monotonic() - started)
        raise subprocess.TimeoutExpired(
            argv,
            duration,
            output=stdout,
            stderr=stderr,
        )
    result = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result


def run_streamed(
    command: Sequence[str],
    cwd: Path,
    stdin_text: str,
    log_path: Path,
    cancel: CancellationToken | None = None,
    env: Mapping[str, str] | None = None,
    on_line: Callable[[str], None] | None = None,
) -> int:
    """Run an argv sequence and persist and observe its combined output.

    No shell is involved. Output is pumped in bounded pieces; each piece passed
    to ``on_line`` is exactly the text written to the log. Cancellation
    terminates the complete child process group and returns ``-1``.
    """
    if isinstance(command, str | bytes) or not command:
        raise ValueError("command must be a non-empty argv sequence")
    if cancel is not None and cancel.is_set():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        return -1

    log_path.parent.mkdir(parents=True, exist_ok=True)
    window = None
    process: subprocess.Popen[bytes] | None = None
    registration: CancellationRegistration | None = None
    process_group_id: int | None = None
    force_target: ForceTarget | None = None
    pump_errors: list[BaseException] = []
    stdout_thread: threading.Thread | None = None
    stdin_thread: threading.Thread | None = None
    cancelled = False

    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        try:
            window = (
                cancel.registration_window() if cancel is not None else None
            )
        except CancellationRegistrationRejected:
            return -1
        try:
            try:
                process = subprocess.Popen(
                    list(command),
                    cwd=cwd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=dict(env) if env is not None else None,
                    start_new_session=os.name == "posix",
                )
            except BaseException:
                if window is not None:
                    window.no_resource()
                raise

            if window is not None:
                window.resource_created()
            process_group_id = _validated_process_group(process)
            if window is not None:
                force_target = ForceTarget(
                    process.pid,
                    process_group_id=process_group_id,
                )
                try:
                    registration = window.register(
                        lambda: _request_termination(process, process_group_id),
                        lambda: terminate_process(
                            process,
                            pgid=process_group_id,
                            force_deadline=time.monotonic(),
                        ),
                        force_target=force_target,
                    )
                except CancellationDeliveryError as error:
                    registration = error.registration
                    raise

            assert process.stdout is not None
            assert process.stdin is not None
            stdout_stream = io.TextIOWrapper(
                process.stdout,
                encoding="utf-8",
                errors="replace",
                newline="",
            )
            stdout_thread = threading.Thread(
                target=_pump_stdout,
                args=(stdout_stream, log_file, on_line, pump_errors),
                name="betterborg-process-stdout",
                daemon=True,
            )
            stdin_thread = threading.Thread(
                target=_pump_stdin,
                args=(process.stdin, stdin_text, pump_errors),
                name="betterborg-process-stdin",
                daemon=True,
            )
            stdout_thread.start()
            stdin_thread.start()

            while process.poll() is None:
                if pump_errors:
                    raise pump_errors[0]
                if cancel is not None and cancel.is_set():
                    cancelled = True
                    break
                try:
                    process.wait(timeout=_PROCESS_POLL_SECONDS)
                except subprocess.TimeoutExpired:
                    continue

            if cancel is not None and cancel.is_set():
                cancelled = True
        finally:
            if process is not None:
                terminate_process(
                    process,
                    pgid=process_group_id,
                    force_deadline=(
                        cancel.force_deadline if cancel is not None else None
                    ),
                )
                _join_pump(stdin_thread)
                _join_pump(stdout_thread)
                if registration is not None:
                    registration.unregister()
                elif window is not None and not window.is_settled:
                    # The process has been verified absent, but a failure around
                    # the registration boundary returned no cleanup handle. A
                    # created window must still publish a validated identity;
                    # it cannot be misreported as a pre-creation failure.
                    if force_target is None:
                        force_target = ForceTarget(
                            process.pid,
                            process_group_id=process_group_id,
                        )
                    window.publish_cleaned_resource(
                        lambda: _request_termination(
                            process, process_group_id
                        ),
                        lambda: terminate_process(
                            process,
                            pgid=process_group_id,
                            force_deadline=time.monotonic(),
                        ),
                        force_target=force_target,
                    )

    if pump_errors:
        raise pump_errors[0]
    if cancelled:
        return -1
    assert process is not None
    return process.returncode


def _validate_cleanup_mode(
    cancel: CancellationToken | None,
    terminate_on_cancel: bool,
    deadline: float | None,
) -> None:
    if terminate_on_cancel:
        return
    if cancel is None:
        raise ValueError("cleanup mode requires a cancellation token")
    token_deadline = cancel.force_deadline
    if deadline is None and token_deadline is None:
        raise ValueError("cleanup mode requires an absolute deadline")
    if (
        deadline is not None
        and token_deadline is not None
        and deadline > token_deadline
    ):
        raise ValueError("cleanup deadline cannot exceed the token deadline")


def _captured_deadline(
    timeout_deadline: float | None,
    deadline: float | None,
    cancel: CancellationToken | None,
    terminate_on_cancel: bool,
) -> float | None:
    deadlines = [
        value for value in (timeout_deadline, deadline) if value is not None
    ]
    if (
        not terminate_on_cancel
        and cancel is not None
        and cancel.force_deadline is not None
    ):
        deadlines.append(cancel.force_deadline)
    return min(deadlines, default=None)


def terminate_process(
    process: subprocess.Popen[object],
    *,
    pgid: int | None = None,
    force_deadline: float | None = None,
) -> None:
    """Terminate a process group, verify cleanup, and reap its direct child.

    On POSIX the known group is checked and signalled even if its leader has
    already exited. ``force_deadline`` is an absolute monotonic deadline shared
    by every process cancelled by the same token.
    """
    deadline = (
        time.monotonic() + CancellationToken.DEFAULT_GRACE_SECONDS
        if force_deadline is None
        else force_deadline
    )
    if os.name != "posix":
        _terminate_single_process(process, deadline)
        return

    process_group_id = process.pid if pgid is None else pgid
    _signal_process_group(process_group_id, signal.SIGTERM)
    while _process_group_exists(process_group_id):
        process.poll()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(_PROCESS_POLL_SECONDS, remaining))

    if _process_group_exists(process_group_id):
        _signal_process_group(process_group_id, signal.SIGKILL)
    _reap_direct_child(process)
    cleanup_deadline = time.monotonic() + CancellationToken.DEFAULT_GRACE_SECONDS
    if not _wait_for_process_group_absence(process_group_id, cleanup_deadline):
        raise TimeoutError(
            f"process group {process_group_id} still exists after SIGKILL"
        )


def _validated_process_group(process: subprocess.Popen[object]) -> int | None:
    if os.name != "posix":
        return None
    process_group_id = process.pid
    try:
        actual_process_group_id = os.getpgid(process.pid)
    except ProcessLookupError:
        # Popen's successful start_new_session handshake establishes this
        # identity even when a very short-lived leader exits before getpgid.
        return process_group_id
    if actual_process_group_id != process_group_id:
        raise RuntimeError("child process group identity could not be validated")
    return process_group_id


def _pump_stdout(
    stdout: io.TextIOWrapper,
    log_file: io.TextIOBase,
    on_line: Callable[[str], None] | None,
    errors: list[BaseException],
) -> None:
    try:
        while line := stdout.readline(_IO_CHUNK_SIZE):
            log_file.write(line)
            log_file.flush()
            if on_line is not None:
                on_line(line)
    except BaseException as error:
        errors.append(error)
    finally:
        with contextlib.suppress(OSError, ValueError):
            stdout.close()


def _pump_stdin(
    stdin: io.BufferedWriter,
    stdin_text: str,
    errors: list[BaseException],
) -> None:
    try:
        for offset in range(0, len(stdin_text), _IO_CHUNK_SIZE):
            stdin.write(
                stdin_text[offset : offset + _IO_CHUNK_SIZE].encode("utf-8")
            )
            stdin.flush()
    except (BrokenPipeError, ConnectionResetError):
        pass
    except BaseException as error:
        errors.append(error)
    finally:
        with contextlib.suppress(BrokenPipeError, OSError, ValueError):
            stdin.close()


def _join_pump(thread: threading.Thread | None) -> None:
    if thread is not None:
        thread.join()


def _request_termination(
    process: subprocess.Popen[object], process_group_id: int | None
) -> None:
    if os.name == "posix":
        assert process_group_id is not None
        _signal_process_group(process_group_id, signal.SIGTERM)
    elif process.poll() is None:
        with contextlib.suppress(ProcessLookupError, OSError):
            process.terminate()


def _signal_process_group(process_group_id: int, signum: int) -> None:
    try:
        os.killpg(process_group_id, signum)
    except ProcessLookupError:
        pass


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for_process_group_absence(
    process_group_id: int, deadline: float
) -> bool:
    while _process_group_exists(process_group_id):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(_PROCESS_POLL_SECONDS, remaining))
    return True


def _reap_direct_child(process: subprocess.Popen[object]) -> None:
    try:
        process.wait(timeout=CancellationToken.DEFAULT_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, OSError):
            process.kill()
        process.wait()


def _terminate_single_process(
    process: subprocess.Popen[object], deadline: float
) -> None:
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError, OSError):
            process.terminate()
    remaining = max(0.0, deadline - time.monotonic())
    try:
        process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, OSError):
            process.kill()
        process.wait()
