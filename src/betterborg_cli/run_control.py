"""Root-owned, signal-safe cancellation and force-exit arbitration."""

from __future__ import annotations

import contextlib
import os
import selectors
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator
from types import FrameType
from typing import Protocol, TextIO

from betterborg_cli.agent_runtime.base import (
    CancellationRegistrationWindow,
    CancellationToken,
    ForceTarget,
)

DEFAULT_FORCE_GRACE_SECONDS = 1.0
INTERRUPTED_EXIT_CODE = 130
_FORCE_NOTICE = b"Force stopping...\n"
_STOP_BYTE = b"\x00"


class CancellationProgress(Protocol):
    """The nonterminal progress operation RunControl is allowed to request."""

    def begin_cancellation(self) -> bool: ...


class RunControl:
    """Translate SIGINT ingress into cancellation and bounded force shutdown.

    The installed Python handler deliberately performs no application callback,
    rendering, locking, waiting, or joining. First-press work is dispatched from
    a wakeup pipe. On a second press, only validated process-group identities are
    signalled in the handler; all callback delivery and creation-window waiting
    remains owned by the dispatcher.
    """

    def __init__(
        self,
        cancellation: CancellationToken | None = None,
        *,
        progress: CancellationProgress | None = None,
        force_grace_seconds: float = DEFAULT_FORCE_GRACE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        force_stream: TextIO | None = None,
        exit_function: Callable[[int], object] = os._exit,
    ) -> None:
        if (
            force_grace_seconds < 0
            or force_grace_seconds > DEFAULT_FORCE_GRACE_SECONDS
        ):
            raise ValueError("force grace must be between zero and one second")
        self.cancellation = cancellation or CancellationToken(
            grace_seconds=force_grace_seconds,
            clock=clock,
        )
        self._progress = progress
        self._force_grace_seconds = force_grace_seconds
        self._clock = clock
        self._force_stream = force_stream if force_stream is not None else sys.stderr
        try:
            self._force_descriptor = self._force_stream.fileno()
        except (AttributeError, OSError):
            self._force_descriptor = -1
        self._exit_function = exit_function

        self._read_fd = -1
        self._write_fd = -1
        self._previous_handler: signal.Handlers | None = None
        self._previous_wakeup_fd = -1
        self._dispatcher: threading.Thread | None = None
        self._installed = False
        self._closed = False

        # These values are assigned by the main-thread handler and read by the
        # dispatcher. Assigning an int, bool, or tuple reference is atomic under
        # CPython and, critically, does not acquire an application lock.
        self._interrupt_count = 0
        self._handled_interrupt_count = 0
        self._protected_depth = 0
        self._deferred_interrupt = False
        self._force_requested = False
        self._forced_windows: tuple[CancellationRegistrationWindow, ...] = ()
        self._exit_invoked = False
        self._force_notice_written = False

        self._cancel_dispatched = threading.Event()
        self._force_dispatched = threading.Event()
        self._dispatcher_stopped = threading.Event()
        self._dispatcher_error: BaseException | None = None
        self._force_deadline: float | None = None

    @property
    def is_installed(self) -> bool:
        """Return whether this controller currently owns SIGINT ingress."""
        return self._installed and not self._closed

    @property
    def active_windows(self) -> tuple[CancellationRegistrationWindow, ...]:
        """Return the token's immutable concurrent creation snapshot."""
        return self.cancellation.active_windows

    @property
    def dispatcher_error(self) -> BaseException | None:
        """Return an exception raised by deferred cancellation delivery."""
        return self._dispatcher_error

    def install(self) -> RunControl:
        """Install SIGINT and wakeup dispatch, returning this controller."""
        if self._installed:
            raise RuntimeError("run control is already installed")
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("run control must be installed on the main thread")

        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        os.set_blocking(write_fd, False)
        self._read_fd = read_fd
        self._write_fd = write_fd
        try:
            self._previous_handler = signal.getsignal(signal.SIGINT)
            self._previous_wakeup_fd = signal.set_wakeup_fd(
                write_fd, warn_on_full_buffer=False
            )
            signal.signal(signal.SIGINT, self._handle_sigint)
        except BaseException:
            with contextlib.suppress(ValueError, OSError):
                signal.set_wakeup_fd(self._previous_wakeup_fd)
            if self._previous_handler is not None:
                with contextlib.suppress(ValueError, OSError):
                    signal.signal(signal.SIGINT, self._previous_handler)
            os.close(read_fd)
            os.close(write_fd)
            self._read_fd = -1
            self._write_fd = -1
            raise

        self._installed = True
        self._dispatcher = threading.Thread(
            target=self._dispatch,
            name="betterborg-signal-dispatcher",
            daemon=True,
        )
        self._dispatcher.start()
        return self

    def close(self) -> None:
        """Restore process signal state and stop the wakeup dispatcher."""
        if not self._installed or self._closed:
            return
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("run control must be closed on the main thread")

        self._closed = True
        signal.signal(signal.SIGINT, self._previous_handler)
        signal.set_wakeup_fd(self._previous_wakeup_fd)
        self._wake_dispatcher(_STOP_BYTE)
        dispatcher = self._dispatcher
        if dispatcher is not None and dispatcher is not threading.current_thread():
            dispatcher.join(timeout=1.0)
        for descriptor in (self._read_fd, self._write_fd):
            if descriptor >= 0:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
        self._read_fd = -1
        self._write_fd = -1

    def __enter__(self) -> RunControl:
        return self.install()

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    @contextlib.contextmanager
    def protected(self) -> Iterator[None]:
        """Defer a first-press ``KeyboardInterrupt`` until locks are released."""
        self._protected_depth += 1
        try:
            yield
        finally:
            self._protected_depth -= 1
            if self._protected_depth == 0 and self._deferred_interrupt:
                self._deferred_interrupt = False
                raise KeyboardInterrupt

    def wait_for_cancellation(self, timeout: float | None = None) -> bool:
        """Wait until the dispatcher has recorded first cancellation."""
        return self._cancel_dispatched.wait(timeout)

    def wait_for_force(self, timeout: float | None = None) -> bool:
        """Wait until the dispatcher has delivered recorded force callbacks."""
        return self._force_dispatched.wait(timeout)

    def _handle_sigint(
        self, _signum: int, _frame: FrameType | None
    ) -> None:
        self._interrupt_count += 1
        if self._interrupt_count == 1:
            if self._protected_depth:
                self._deferred_interrupt = True
                return
            raise KeyboardInterrupt

        if self._force_requested:
            return
        self._force_requested = True
        self.cancellation._force_requested_signal = True
        windows = self.cancellation.active_windows
        targets = self.cancellation.force_targets
        self._forced_windows = windows
        undeliverable_target = self._signal_process_groups(targets)
        self._force_notice_written = self._write_force_notice_signal_safe()
        if (
            not windows
            and not self.cancellation.has_window_opening
            and not undeliverable_target
        ):
            self._invoke_exit()

    def _dispatch(self) -> None:
        selector = selectors.DefaultSelector()
        try:
            selector.register(self._read_fd, selectors.EVENT_READ)
            while not self._closed:
                timeout = self._deadline_timeout()
                events = selector.select(timeout)
                if events:
                    try:
                        payload = os.read(self._read_fd, 4096)
                    except BlockingIOError:
                        payload = b""
                    if _STOP_BYTE in payload and self._closed:
                        break
                self._dispatch_interrupts()
                if self._deadline_reached():
                    self._request_force(())
        except BaseException as error:
            self._dispatcher_error = error
        finally:
            selector.close()
            self._dispatcher_stopped.set()

    def _dispatch_interrupts(self) -> None:
        interrupt_count = self._interrupt_count
        if self._handled_interrupt_count == 0 and interrupt_count >= 1:
            self._handled_interrupt_count = 1
            self.cancellation.cancel()
            if self._progress is not None:
                self._progress.begin_cancellation()
            self._force_deadline = self._clock() + self._force_grace_seconds
            self._cancel_dispatched.set()
        if self._handled_interrupt_count < 2 and interrupt_count >= 2:
            self._handled_interrupt_count = 2
            self._request_force(self._forced_windows)

    def _request_force(
        self, windows: tuple[CancellationRegistrationWindow, ...]
    ) -> None:
        if self._force_dispatched.is_set():
            return
        self._force_requested = True
        self.cancellation._force_requested_signal = True
        if not self._forced_windows:
            self._forced_windows = windows or self.cancellation.active_windows
        self._signal_process_groups(self.cancellation.force_targets)
        if not self._force_notice_written:
            self._write_force_notice_deferred()
        try:
            self.cancellation.force()
        except BaseException as error:
            self._dispatcher_error = error
        finally:
            self._force_dispatched.set()

        for window in self._forced_windows:
            window.wait()
        if all(window.is_settled for window in self._forced_windows):
            self._invoke_exit()

    def _deadline_timeout(self) -> float | None:
        if self._force_deadline is None or self._force_dispatched.is_set():
            return None
        return max(0.0, self._force_deadline - self._clock())

    def _deadline_reached(self) -> bool:
        return (
            self._force_deadline is not None
            and not self._force_dispatched.is_set()
            and self._clock() >= self._force_deadline
        )

    @staticmethod
    def _signal_process_groups(targets: tuple[ForceTarget, ...]) -> bool:
        undeliverable = False
        for target in targets:
            process_group_id = target.process_group_id
            if process_group_id is None:
                undeliverable = True
                continue
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        return undeliverable

    def _write_force_notice_signal_safe(self) -> bool:
        if self._force_descriptor < 0:
            return False
        try:
            os.write(self._force_descriptor, _FORCE_NOTICE)
        except OSError:
            return False
        return True

    def _write_force_notice_deferred(self) -> None:
        try:
            self._force_stream.write(_FORCE_NOTICE.decode("ascii"))
            self._force_stream.flush()
            self._force_notice_written = True
        except (AttributeError, OSError, ValueError):
            pass

    def _wake_dispatcher(self, payload: bytes) -> None:
        if self._write_fd < 0:
            return
        with contextlib.suppress(BlockingIOError, OSError):
            os.write(self._write_fd, payload)

    def _invoke_exit(self) -> None:
        if self._exit_invoked:
            return
        self._exit_invoked = True
        self._exit_function(INTERRUPTED_EXIT_CODE)
