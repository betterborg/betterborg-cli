"""Cross-process file locking for host-owned lifecycle operations."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

if os.name == "nt":  # pragma: no cover - exercised on Windows hosts
    import msvcrt
else:  # pragma: no branch - exactly one platform lock implementation is loaded
    import fcntl

_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


@contextmanager
def path_lock(lock_path: Path) -> Iterator[None]:
    """Serialize one host lifecycle across threads and worker processes."""
    key = str(lock_path.resolve())
    with _PATH_LOCKS_GUARD:
        process_lock = _PATH_LOCKS.setdefault(key, threading.Lock())
    with process_lock:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock_file:
            _lock_file(lock_file)
            try:
                yield
            finally:
                _unlock_file(lock_file)


def _lock_file(lock_file) -> None:
    if os.name == "nt":  # pragma: no cover - exercised on Windows hosts
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        while True:
            try:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(0.05)
    else:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)


def _unlock_file(lock_file) -> None:
    if os.name == "nt":  # pragma: no cover - exercised on Windows hosts
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
