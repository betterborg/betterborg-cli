"""Root-owned, signal-safe cancellation and force-exit arbitration."""

from __future__ import annotations

import contextlib
import os
import select
import selectors
import signal
import stat
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
_ARBITRATION_POLL_SECONDS = 0.01


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
            self._force_stream_descriptor = self._force_stream.fileno()
        except (AttributeError, OSError):
            self._force_stream_descriptor = -1
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
        self._force_snapshot_captured = False
        self._direct_force_complete = False
        self._force_delivery_started = False
        self._exit_invoked = False
        self._force_notice_written = False
        self._force_notice_attempted = False

        self._cancel_dispatched = threading.Event()
        self._force_dispatched = threading.Event()
        self._dispatcher_stopped = threading.Event()
        self._dispatcher_error: BaseException | None = None
        self._force_deadline: float | None = None
        self._cancel_worker: threading.Thread | None = None
        self._force_worker: threading.Thread | None = None

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

    @property
    def interruption_requested(self) -> bool:
        """Return whether this controller received a first SIGINT."""
        return self._interrupt_count >= 1

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
        self._force_descriptor = self._open_nonblocking_force_descriptor()
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
            if self._force_descriptor >= 0:
                os.close(self._force_descriptor)
            self._read_fd = -1
            self._write_fd = -1
            self._force_descriptor = -1
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
        for descriptor in (
            self._read_fd,
            self._write_fd,
            self._force_descriptor,
        ):
            if descriptor >= 0:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
        self._read_fd = -1
        self._write_fd = -1
        self._force_descriptor = -1

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
        """Wait until first cancellation and progress acknowledgement finish."""
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
        opening_in_progress = self.cancellation.has_window_opening
        windows = self.cancellation.active_windows
        targets = self.cancellation.force_targets
        self._forced_windows = windows
        # An opener that passed its post-force check is already part of the
        # force boundary even when it has not published its window object yet.
        # Finalize the immutable snapshot in the dispatcher after such openers
        # drain; reading the opening marker before the snapshot means that a
        # newly starting, post-force opener can only be rejected.
        self._force_snapshot_captured = not opening_in_progress
        self._direct_force_complete = self._signal_process_groups(targets)
        self._force_notice_written = self._write_force_notice_signal_safe()
        if (
            self._force_snapshot_captured
            and not windows
            and self._direct_force_complete
            and self._force_notice_allows_exit()
        ):
            self._invoke_exit()

    def _dispatch(self) -> None:
        selector = selectors.DefaultSelector()
        try:
            selector.register(self._read_fd, selectors.EVENT_READ)
            while True:
                timeout = self._deadline_timeout()
                events = selector.select(timeout)
                stop_requested = False
                if events:
                    try:
                        payload = os.read(self._read_fd, 4096)
                    except BlockingIOError:
                        payload = b""
                    stop_requested = _STOP_BYTE in payload and self._closed
                self._dispatch_interrupts()
                if self._deadline_reached():
                    self._request_force(())
                self._arbitrate_exit()
                if stop_requested:
                    break
        except BaseException as error:
            self._dispatcher_error = error
        finally:
            if self._interrupt_count >= 1 and self._cancel_worker is None:
                self._cancel_dispatched.set()
            selector.close()
            self._dispatcher_stopped.set()

    def _dispatch_interrupts(self) -> None:
        interrupt_count = self._interrupt_count
        if self._handled_interrupt_count == 0 and interrupt_count >= 1:
            self._handled_interrupt_count = 1
            self._force_deadline = self._clock() + self._force_grace_seconds
            self._cancel_worker = threading.Thread(
                target=self._deliver_cancellation,
                name="betterborg-cancellation-dispatch",
                daemon=True,
            )
            self._cancel_worker.start()
        if self._handled_interrupt_count < 2 and interrupt_count >= 2:
            self._handled_interrupt_count = 2
            self._request_force(self._forced_windows)

    def _deliver_cancellation(self) -> None:
        try:
            self.cancellation.cancel()
            if self._progress is not None:
                self._progress.begin_cancellation()
        except BaseException as error:
            self._record_dispatcher_error(error)
        finally:
            self._cancel_dispatched.set()

    def _request_force(
        self, windows: tuple[CancellationRegistrationWindow, ...]
    ) -> None:
        if self._force_delivery_started:
            return
        self._force_requested = True
        self.cancellation._force_requested_signal = True
        if not self._force_snapshot_captured:
            self._forced_windows = windows
            self._finalize_force_window_snapshot()
        self._direct_force_complete = self._signal_process_groups(
            self.cancellation.force_targets
        )
        if not self._force_notice_written:
            self._write_force_notice_deferred()
        self._force_delivery_started = True
        self._force_worker = threading.Thread(
            target=self._deliver_force,
            name="betterborg-force-dispatch",
            daemon=True,
        )
        self._force_worker.start()
        if self._direct_force_complete:
            self._arbitrate_exit()

    def _deliver_force(self) -> None:
        try:
            self.cancellation.force()
        except BaseException as error:
            self._record_dispatcher_error(error)
        else:
            self._force_dispatched.set()
        finally:
            self._wake_dispatcher(b"f")

    def _arbitrate_exit(self) -> None:
        if not self._force_delivery_started or self._exit_invoked:
            return
        if not self._finalize_force_window_snapshot():
            return
        if any(not window.is_settled for window in self._forced_windows):
            return
        if not self._force_notice_allows_exit():
            return
        if self._direct_force_complete:
            self._invoke_exit()
        elif self._force_dispatched.is_set():
            self._invoke_exit()

    def _finalize_force_window_snapshot(self) -> bool:
        """Capture every window whose opening began before force was requested."""
        if self._force_snapshot_captured:
            return True
        if self.cancellation.has_window_opening:
            return False

        windows = list(self._forced_windows)
        for window in self.cancellation.active_windows:
            if window not in windows:
                windows.append(window)
        self._forced_windows = tuple(windows)
        self._force_snapshot_captured = True
        return True

    def _record_dispatcher_error(self, error: BaseException) -> None:
        if self._dispatcher_error is None:
            self._dispatcher_error = error

    def _deadline_timeout(self) -> float | None:
        if self._force_delivery_started:
            return _ARBITRATION_POLL_SECONDS
        if self._force_deadline is None:
            return None
        return max(0.0, self._force_deadline - self._clock())

    def _deadline_reached(self) -> bool:
        return (
            self._force_deadline is not None
            and not self._force_delivery_started
            and self._clock() >= self._force_deadline
        )

    @staticmethod
    def _signal_process_groups(targets: tuple[ForceTarget, ...]) -> bool:
        delivery_complete = True
        for target in targets:
            process_group_id = target.process_group_id
            if process_group_id is None:
                delivery_complete = False
                continue
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except (OSError, TypeError):
                delivery_complete = False
        return delivery_complete

    def _write_force_notice_signal_safe(self) -> bool:
        if self._force_descriptor < 0:
            return False
        try:
            os.write(self._force_descriptor, _FORCE_NOTICE)
        except OSError:
            return False
        return True

    def _write_force_notice_deferred(self) -> None:
        if self._force_descriptor >= 0:
            self._force_notice_written = self._write_force_notice_signal_safe()
            return
        if self._force_stream_descriptor >= 0:
            self._force_notice_written = self._write_force_notice_if_ready()
            self._force_notice_attempted = True
            return
        try:
            self._force_stream.write(_FORCE_NOTICE.decode("ascii"))
            self._force_stream.flush()
            self._force_notice_written = True
        except (AttributeError, OSError, ValueError) as error:
            self._record_dispatcher_error(error)

    def _force_notice_allows_exit(self) -> bool:
        """Return whether notification is complete or safely best-effort.

        A prepared nonblocking descriptor lets the handler attempt notification
        without waiting, even when the destination is already full. On platforms
        that cannot independently reopen the descriptor, the dispatcher instead
        performs a zero-time readiness attempt. Descriptor-less streams must
        confirm the ordinary stream write before exit.
        """
        return (
            self._force_notice_written
            or self._force_descriptor >= 0
            or self._force_notice_attempted
        )

    def _write_force_notice_if_ready(self) -> bool:
        """Attempt descriptor output without waiting for write capacity.

        BSD ``/dev/fd`` descriptors share file status flags with the original,
        so they cannot provide the independently nonblocking handle available
        through Linux ``/proc``. A zero-time readiness check is the portable
        fallback and restores the caller's descriptor mode after the attempt.
        The notice is smaller than ``PIPE_BUF``, so a write-ready pipe accepts it
        atomically; a temporary nonblocking guard also closes a concurrent-writer
        race between the readiness check and the write.
        """
        descriptor = self._force_stream_descriptor
        try:
            _readable, writable, _exceptional = select.select(
                [], [descriptor], [], 0
            )
            if not writable:
                return False
            was_blocking = os.get_blocking(descriptor)
            if was_blocking:
                os.set_blocking(descriptor, False)
            try:
                os.write(descriptor, _FORCE_NOTICE)
            finally:
                if was_blocking:
                    os.set_blocking(descriptor, True)
        except (OSError, ValueError):
            return False
        return True

    def _open_nonblocking_force_descriptor(self) -> int:
        """Open an independently nonblocking descriptor for handler output."""
        descriptor = self._force_stream_descriptor
        if descriptor < 0:
            return -1
        try:
            mode = os.fstat(descriptor).st_mode
        except OSError:
            return -1
        if not (stat.S_ISFIFO(mode) or stat.S_ISCHR(mode)):
            return -1

        # ``dup`` would share O_NONBLOCK with the caller's descriptor. Reopening
        # the Linux descriptor path creates an independent open file description,
        # so the signal handler cannot block and the caller retains its mode.
        flags = os.O_WRONLY | os.O_NONBLOCK
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOCTTY", 0)
        try:
            return os.open(f"/proc/self/fd/{descriptor}", flags)
        except OSError:
            return -1

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
