"""Streamed subprocess behavior for agent CLI adapters."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

from betterborg_cli.agent_runtime import CancellationToken, run_streamed


def test_run_streamed_combines_output_and_passes_stdin(tmp_path: Path) -> None:
    log_path = tmp_path / "process.log"
    script = (
        "import sys; "
        "data=sys.stdin.read(); "
        "print('stdout:' + data); "
        "print('stderr-line', file=sys.stderr)"
    )

    exit_code = run_streamed(
        [sys.executable, "-c", script],
        tmp_path,
        "input-value",
        log_path,
    )

    assert exit_code == 0
    output = log_path.read_text(encoding="utf-8")
    assert "stdout:input-value" in output
    assert "stderr-line" in output


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


def test_run_streamed_terminates_process_group_on_cancellation(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "cancel.log"
    started = tmp_path / "started"
    cancel = CancellationToken()

    def trigger_cancel() -> None:
        for _ in range(100):
            if started.exists():
                cancel.cancel()
                return
            cancel.wait(0.01)
        cancel.cancel()

    thread = threading.Thread(target=trigger_cancel)
    thread.start()
    try:
        exit_code = run_streamed(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; import time; "
                f"Path({str(started)!r}).write_text('yes'); time.sleep(30)",
            ],
            tmp_path,
            "",
            log_path,
            cancel,
        )
    finally:
        thread.join(timeout=2)

    assert exit_code == -1
    assert not thread.is_alive()


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
