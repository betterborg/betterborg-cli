"""Stream non-shell subprocess output with cooperative cancellation."""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from betterborg_cli.agent_runtime.base import CancellationToken

ProcessRunner = Callable[
    [
        Sequence[str],
        Path,
        str,
        Path,
        CancellationToken | None,
        Mapping[str, str] | None,
    ],
    int,
]


def run_streamed(
    command: Sequence[str],
    cwd: Path,
    stdin_text: str,
    log_path: Path,
    cancel: CancellationToken | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """Run an argv sequence and stream combined output to ``log_path``.

    No shell is involved. Cancellation terminates the complete child process
    group and returns ``-1``.
    """
    if isinstance(command, str | bytes) or not command:
        raise ValueError("command must be a non-empty argv sequence")
    if cancel is not None and cancel.is_set():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        return -1

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(env) if env is not None else None,
            start_new_session=os.name == "posix",
        )
        try:
            input_text: str | None = stdin_text
            while True:
                if cancel is not None and cancel.is_set():
                    _terminate_process(process)
                    return -1
                try:
                    process.communicate(input_text, timeout=0.1)
                except subprocess.TimeoutExpired:
                    # communicate() retains unwritten input after a timeout;
                    # subsequent calls must resume without passing it again.
                    input_text = None
                    continue

                if cancel is not None and cancel.is_set():
                    return -1
                return process.returncode
        finally:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        return
    process.wait(timeout=5)
