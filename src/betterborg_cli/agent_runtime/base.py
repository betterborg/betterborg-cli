"""Shared contracts for invoking BetterBorg agent adapters."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from betterborg_cli.progress import AgentActivity


class AgentStatus(StrEnum):
    """Stable adapter-level outcome classifications."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BillingMode(StrEnum):
    """How an agent invocation is billed."""

    API = "api"
    SUBSCRIPTION = "subscription"


class CancellationState(StrEnum):
    """Lifecycle states shared by cancellation-aware resources."""

    ACTIVE = "active"
    CANCELLED = "cancelled"
    FORCED = "forced"


@dataclass(frozen=True, slots=True)
class ForceTarget:
    """Stable identity for a resource that has been validated for force delivery.

    The resource owner is responsible for constructing this record only after it
    has validated the identity against the live resource. Keeping the identity
    separate from the callback prevents an untracked force callback from being
    registered and lets cleanup code prove the registration remains retained.
    """

    identity: Hashable
    process_group_id: int | None = None

    def __post_init__(self) -> None:
        if self.identity is None or isinstance(self.identity, bool):
            raise ValueError("force target identity must be a stable value")
        try:
            hash(self.identity)
        except TypeError as error:
            raise TypeError("force target identity must be hashable") from error
        if isinstance(self.identity, int) and self.identity <= 0:
            raise ValueError("force target integer identity must be positive")
        if isinstance(self.identity, str) and not self.identity:
            raise ValueError("force target string identity must not be empty")
        if self.process_group_id is not None:
            if isinstance(self.process_group_id, bool) or not isinstance(
                self.process_group_id, int
            ):
                raise TypeError("process group identity must be an integer")
            if self.process_group_id <= 0:
                raise ValueError("process group identity must be a positive integer")


@dataclass(slots=True)
class _CancellationEntry:
    on_cancel: Callable[[], None]
    on_force: Callable[[], None] | None
    terminate_on_cancel: bool
    force_target: ForceTarget | None
    window_id: int | None = None
    cancel_claimed: bool = False
    force_claimed: bool = False
    cancel_started: threading.Event = field(default_factory=threading.Event)
    cancel_finished: threading.Event = field(default_factory=threading.Event)
    force_finished: threading.Event = field(default_factory=threading.Event)
    force_error: Exception | None = None


@dataclass(slots=True)
class _RegistrationWindowEntry:
    window: CancellationRegistrationWindow
    resource_created: bool = False
    publication_started: bool = False


class CancellationRegistration:
    """Handle retained until a registered resource has verified cleanup."""

    def __init__(
        self,
        token: CancellationToken,
        registration_id: int,
        force_target: ForceTarget | None,
    ) -> None:
        self._token = token
        self._registration_id = registration_id
        self.force_target = force_target

    def unregister(self) -> bool:
        """Remove the registration after cleanup, returning whether it was active."""
        return self._token._unregister(self._registration_id)


class CancellationDeliveryError(RuntimeError):
    """Late callback failure that retains the registration cleanup handle."""

    def __init__(
        self,
        registration: CancellationRegistration,
        errors: tuple[Exception, ...],
    ) -> None:
        if not errors:
            raise ValueError("at least one delivery error is required")
        detail = str(errors[0]) if len(errors) == 1 else f"{len(errors)} errors"
        super().__init__(f"late cancellation delivery failed: {detail}")
        self.registration = registration
        self.errors = errors


class CancellationRegistrationRejected(RuntimeError):
    """A creation window rejected because forced cancellation already began."""


class CancellationRegistrationWindow:
    """Pre-creation entry retained until resource creation is resolved."""

    def __init__(self, token: CancellationToken, window_id: int) -> None:
        self._token = token
        self._window_id = window_id
        self._settled = threading.Event()

    @property
    def is_settled(self) -> bool:
        """Return whether creation has been conclusively resolved."""
        return self._settled.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for target registration or conclusive creation failure."""
        return self._settled.wait(timeout)

    def register(
        self,
        on_cancel: Callable[[], None],
        on_force: Callable[[], None] | None = None,
        *,
        terminate_on_cancel: bool = True,
        force_target: ForceTarget | None = None,
    ) -> CancellationRegistration:
        """Publish a created resource and atomically register its callbacks."""
        return self._token._register(
            on_cancel,
            on_force,
            terminate_on_cancel=terminate_on_cancel,
            force_target=force_target,
            window_id=self._window_id,
        )

    def resource_created(self) -> None:
        """Record successful creation before publication can be delayed."""
        self._token._mark_resource_created(self._window_id)

    def publish_cleaned_resource(
        self,
        on_cancel: Callable[[], None],
        on_force: Callable[[], None],
        *,
        force_target: ForceTarget,
    ) -> None:
        """Publish and retire a created resource after verified local cleanup.

        Resource owners use this recovery path only when their normal
        registration boundary failed before returning a cleanup handle. The
        validated target is still published through the token's registration
        machinery, so a recorded force request is delivered before the window
        settles, and the temporary registration is then removed immediately.
        """
        registration = self._token._register(
            on_cancel,
            on_force,
            terminate_on_cancel=True,
            force_target=force_target,
            window_id=self._window_id,
        )
        registration.unregister()

    def no_resource(self) -> None:
        """Settle a window after creation conclusively produced no resource."""
        self._token._settle_no_resource(self._window_id)


class CancellationToken:
    """Atomic active/cancelled/forced lifecycle for cooperative resources."""

    DEFAULT_GRACE_SECONDS = 1.0

    def __init__(
        self,
        *,
        grace_seconds: float = DEFAULT_GRACE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if grace_seconds < 0 or grace_seconds > self.DEFAULT_GRACE_SECONDS:
            raise ValueError("cancellation grace must be between zero and one second")
        self._event = threading.Event()
        self._force_event = threading.Event()
        self._cancel_delivery_started = threading.Event()
        self._force_delivery_finished = threading.Event()
        self._lock = threading.Lock()
        self._window_opening_lock = threading.Lock()
        self._clock = clock
        self._grace_seconds = grace_seconds
        self._state = CancellationState.ACTIVE
        self._force_requested_signal = False
        self._force_deadline: float | None = None
        self._force_delivery_error: Exception | None = None
        self._next_registration_id = 0
        self._next_window_id = 0
        self._window_opening_count = 0
        self._registrations: dict[int, _CancellationEntry] = {}
        self._registration_windows: dict[int, _RegistrationWindowEntry] = {}
        self._force_targets_snapshot: tuple[ForceTarget, ...] = ()
        self._active_windows_snapshot: tuple[
            CancellationRegistrationWindow, ...
        ] = ()

    @property
    def state(self) -> CancellationState:
        """Return a consistent snapshot of the lifecycle state."""
        with self._lock:
            return self._state

    @property
    def force_deadline(self) -> float | None:
        """Return the absolute monotonic deadline fixed by first cancellation."""
        with self._lock:
            return self._force_deadline

    @property
    def force_targets(self) -> tuple[ForceTarget, ...]:
        """Return a lock-free immutable snapshot retained for signal delivery."""
        return self._force_targets_snapshot

    @property
    def active_windows(self) -> tuple[CancellationRegistrationWindow, ...]:
        """Return a lock-free immutable snapshot of unresolved creation windows."""
        return self._active_windows_snapshot

    @property
    def has_window_opening(self) -> bool:
        """Return whether a caller is entering the registry but not published yet."""
        return self._window_opening_count > 0

    def registration_window(self) -> CancellationRegistrationWindow:
        """Register a unique creation window before an OS resource is created."""
        with self._window_opening_lock:
            self._window_opening_count += 1
        try:
            with self._lock:
                if (
                    self._state is CancellationState.FORCED
                    or self._force_requested_signal
                ):
                    raise CancellationRegistrationRejected(
                        "cannot open a registration window after force"
                    )
                window_id = self._next_window_id
                self._next_window_id += 1
                window = CancellationRegistrationWindow(self, window_id)
                self._registration_windows[window_id] = _RegistrationWindowEntry(
                    window
                )
                self._refresh_registry_snapshots()
                return window
        finally:
            with self._window_opening_lock:
                self._window_opening_count -= 1

    def register(
        self,
        on_cancel: Callable[[], None],
        on_force: Callable[[], None] | None = None,
        *,
        terminate_on_cancel: bool = True,
        force_target: ForceTarget | None = None,
    ) -> CancellationRegistration:
        """Atomically register lifecycle callbacks and deliver recorded state.

        A returned registration is deliberately not removed after callback
        delivery. The resource owner must unregister it only after cleanup has
        verified the target no longer exists.
        """
        return self._register(
            on_cancel,
            on_force,
            terminate_on_cancel=terminate_on_cancel,
            force_target=force_target,
        )

    def cancel(self) -> None:
        """Record first cancellation and concurrently broadcast eligible callbacks."""
        with self._lock:
            if self._state is not CancellationState.ACTIVE:
                return
            self._state = CancellationState.CANCELLED
            self._force_deadline = self._clock() + self._grace_seconds
            self._event.set()
            callbacks = self._claim_cancel_callbacks()

        self._broadcast(callbacks, "cancel", wait_for_completion=False)
        self._cancel_delivery_started.set()

    def force(self) -> None:
        """Record forced cancellation and broadcast every validated force callback."""
        self._force_requested_signal = True
        with self._lock:
            if self._state is CancellationState.FORCED:
                already_forced = True
                was_active = False
                cancel_callbacks = []
                force_callbacks = []
            else:
                already_forced = False
                was_active = self._state is CancellationState.ACTIVE
                if was_active:
                    self._force_deadline = self._clock()
                    self._event.set()
                self._state = CancellationState.FORCED
                self._force_event.set()
                cancel_callbacks = self._claim_cancel_callbacks()
                force_callbacks = self._claim_force_callbacks()

        if already_forced:
            self._force_delivery_finished.wait()
            with self._lock:
                force_delivery_error = self._force_delivery_error
            if force_delivery_error is not None:
                raise force_delivery_error
            return

        try:
            if was_active:
                self._broadcast(
                    cancel_callbacks, "cancel", wait_for_completion=False
                )
                self._cancel_delivery_started.set()
            else:
                self._cancel_delivery_started.wait()
            self._broadcast(force_callbacks, "force", wait_for_completion=True)
        except Exception as error:
            with self._lock:
                self._force_delivery_error = error
            raise
        finally:
            self._force_delivery_finished.set()

    def is_set(self) -> bool:
        """Return whether cancellation was requested."""
        return self._event.is_set()

    def start_if_active(self, start: Callable[[], None]) -> bool:
        """Run one short work-start transition unless cancellation won the race."""
        if not callable(start):
            raise TypeError("start must be callable")
        with self._lock:
            if (
                self._state is not CancellationState.ACTIVE
                or self._force_requested_signal
            ):
                return False
            start()
            return True

    def is_forced(self) -> bool:
        """Return whether forced cancellation was requested."""
        return self._force_event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for cancellation, returning whether it was requested."""
        return self._event.wait(timeout)

    def wait_for_force(self, timeout: float | None = None) -> bool:
        """Wait for forced cancellation, returning whether it was requested."""
        return self._force_event.wait(timeout)

    def _unregister(self, registration_id: int) -> bool:
        while True:
            with self._lock:
                entry = self._registrations.get(registration_id)
                if entry is None:
                    return False
                pending_deliveries = [
                    finished
                    for claimed, finished in (
                        (entry.cancel_claimed, entry.cancel_finished),
                        (entry.force_claimed, entry.force_finished),
                    )
                    if claimed and not finished.is_set()
                ]
                if not pending_deliveries:
                    self._registrations.pop(registration_id)
                    if entry.window_id is not None:
                        window_entry = self._registration_windows.pop(
                            entry.window_id, None
                        )
                        if window_entry is not None:
                            window_entry.window._settled.set()
                    self._refresh_registry_snapshots()
                    return True

            for finished in pending_deliveries:
                finished.wait()

    def _register(
        self,
        on_cancel: Callable[[], None],
        on_force: Callable[[], None] | None = None,
        *,
        terminate_on_cancel: bool,
        force_target: ForceTarget | None,
        window_id: int | None = None,
    ) -> CancellationRegistration:
        if not callable(on_cancel):
            raise TypeError("on_cancel must be callable")
        if on_force is not None and not callable(on_force):
            raise TypeError("on_force must be callable")
        if (on_force is None) != (force_target is None):
            raise ValueError("on_force and force_target must be provided together")
        if not terminate_on_cancel and on_force is None:
            raise ValueError("cleanup registrations must remain force-deliverable")
        if window_id is not None and on_force is None:
            raise ValueError(
                "registration windows require a validated force target"
            )

        with self._lock:
            window_entry = None
            if window_id is not None:
                window_entry = self._registration_windows.get(window_id)
                if window_entry is None or window_entry.publication_started:
                    raise RuntimeError("registration window is already settled")
                if not window_entry.resource_created:
                    raise RuntimeError(
                        "registration window must record resource creation "
                        "before publication"
                    )
                window_entry.publication_started = True
            registration_id = self._next_registration_id
            self._next_registration_id += 1
            entry = _CancellationEntry(
                on_cancel=on_cancel,
                on_force=on_force,
                terminate_on_cancel=terminate_on_cancel,
                force_target=force_target,
                window_id=window_id,
            )
            self._registrations[registration_id] = entry
            self._refresh_registry_snapshots()
            cancel_callback = self._claim_cancel(entry)
            force_callback = self._claim_force(entry)
            window_settled = False
            if (
                window_id is not None
                and self._state is CancellationState.ACTIVE
                and not self._force_requested_signal
            ):
                settled_entry = self._registration_windows.pop(window_id)
                settled_entry.window._settled.set()
                self._refresh_registry_snapshots()
                window_settled = True

        registration = CancellationRegistration(
            self, registration_id, force_target
        )
        # Late delivery is synchronous, but callbacks are never invoked while
        # the token lock is held.
        callback_errors: list[Exception] = []
        for callback in (cancel_callback, force_callback):
            if callback is None:
                continue
            try:
                callback()
            except Exception as error:
                callback_errors.append(error)
        if window_id is not None and not window_settled:
            try:
                self._settle_published_window(window_id, entry)
            except Exception as error:
                if all(
                    error is not callback_error
                    for callback_error in callback_errors
                ):
                    callback_errors.append(error)
        if callback_errors:
            delivery_error = CancellationDeliveryError(
                registration, tuple(callback_errors)
            )
            if len(callback_errors) == 1:
                raise delivery_error from callback_errors[0]
            raise delivery_error from ExceptionGroup(
                "late cancellation callbacks failed", callback_errors
            )
        return registration

    def _mark_resource_created(self, window_id: int) -> None:
        with self._lock:
            entry = self._registration_windows.get(window_id)
            if entry is None or entry.publication_started:
                raise RuntimeError("registration window is already settled")
            if entry.resource_created:
                raise RuntimeError("resource creation is already recorded")
            entry.resource_created = True

    def _settle_no_resource(self, window_id: int) -> None:
        with self._lock:
            entry = self._registration_windows.get(window_id)
            if entry is None or entry.publication_started:
                raise RuntimeError("registration window is already settled")
            if entry.resource_created:
                raise RuntimeError("created resource must be published")
            self._registration_windows.pop(window_id)
            entry.window._settled.set()
            self._refresh_registry_snapshots()

    def _settle_published_window(
        self, window_id: int, cancellation_entry: _CancellationEntry
    ) -> None:
        with self._lock:
            if (
                self._state is not CancellationState.FORCED
                and not self._force_requested_signal
            ):
                entry = self._registration_windows.pop(window_id, None)
                if entry is None:
                    raise RuntimeError("registration window is already settled")
                entry.window._settled.set()
                self._refresh_registry_snapshots()
                return
            force_callback = self._claim_force(cancellation_entry)
            wait_for_force = (
                force_callback is None
                and not cancellation_entry.force_finished.is_set()
            )

        if force_callback is not None:
            force_callback()
        elif wait_for_force:
            cancellation_entry.force_finished.wait()

        if cancellation_entry.force_error is not None:
            raise cancellation_entry.force_error

        with self._lock:
            entry = self._registration_windows.pop(window_id, None)
            if entry is None:
                raise RuntimeError("registration window is already settled")
            entry.window._settled.set()
            self._refresh_registry_snapshots()

    def _refresh_registry_snapshots(self) -> None:
        """Publish registry state atomically while the token lock is held."""
        self._force_targets_snapshot = tuple(
            entry.force_target
            for entry in self._registrations.values()
            if entry.force_target is not None
        )
        self._active_windows_snapshot = tuple(
            entry.window for entry in self._registration_windows.values()
        )

    def _claim_cancel(self, entry: _CancellationEntry) -> Callable[[], None] | None:
        if (
            self._state is CancellationState.ACTIVE
            or entry.cancel_claimed
            or not entry.terminate_on_cancel
        ):
            return None
        entry.cancel_claimed = True

        def deliver_cancel() -> None:
            try:
                entry.cancel_started.set()
                entry.on_cancel()
            finally:
                entry.cancel_finished.set()

        return deliver_cancel

    def _claim_force(self, entry: _CancellationEntry) -> Callable[[], None] | None:
        if (
            (
                self._state is not CancellationState.FORCED
                and not self._force_requested_signal
            )
            or entry.force_claimed
            or entry.on_force is None
            or entry.force_target is None
        ):
            return None
        entry.force_claimed = True

        def deliver_force() -> None:
            try:
                if entry.cancel_claimed:
                    entry.cancel_started.wait()
                if entry.on_force is not None:
                    entry.on_force()
            except Exception as error:
                entry.force_error = error
                raise
            finally:
                entry.force_finished.set()

        return deliver_force

    def _claim_cancel_callbacks(self) -> list[Callable[[], None]]:
        return [
            callback
            for entry in self._registrations.values()
            if (callback := self._claim_cancel(entry)) is not None
        ]

    def _claim_force_callbacks(self) -> list[Callable[[], None]]:
        return [
            callback
            for entry in self._registrations.values()
            if (callback := self._claim_force(entry)) is not None
        ]

    def _broadcast(
        self,
        callbacks: list[Callable[[], None]],
        label: str,
        *,
        wait_for_completion: bool,
    ) -> None:
        started: list[threading.Event] = []
        completed: list[threading.Event] = []
        errors: list[Exception | None] = [None] * len(callbacks)
        release = threading.Event()
        for index, callback in enumerate(callbacks):
            callback_started = threading.Event()
            callback_completed = threading.Event()
            started.append(callback_started)
            completed.append(callback_completed)
            thread = threading.Thread(
                target=self._invoke_broadcast,
                args=(
                    callback,
                    callback_started,
                    callback_completed,
                    release,
                    errors,
                    index,
                    wait_for_completion,
                ),
                name=f"betterborg-{label}-callback",
                daemon=True,
            )
            thread.start()
        for callback_started in started:
            callback_started.wait()
        release.set()
        if wait_for_completion:
            for callback_completed in completed:
                callback_completed.wait()
            errors = [error for error in errors if error is not None]
            if len(errors) == 1:
                raise errors[0]
            if errors:
                raise ExceptionGroup(f"{label} callbacks failed", errors)

    def _invoke_broadcast(
        self,
        callback: Callable[[], None],
        started: threading.Event,
        completed: threading.Event,
        release: threading.Event,
        errors: list[Exception | None],
        index: int,
        capture_errors: bool,
    ) -> None:
        started.set()
        release.wait()
        try:
            callback()
        except Exception as error:
            if not capture_errors:
                raise
            errors[index] = error
        finally:
            completed.set()


@dataclass(frozen=True, slots=True)
class AgentCapabilities:
    """Provider-independent metadata used to match adapters to runs."""

    billing_modes: frozenset[BillingMode] = frozenset({BillingMode.API})
    structured_output: bool = True
    streaming: bool = False
    tool_allowlist: bool = False
    resumable: bool = False
    host_capable: bool = False
    read_only_sandbox: bool = False

    def supports_billing(self, mode: BillingMode) -> bool:
        """Return whether the adapter can run under ``mode``."""
        return mode in self.billing_modes


@dataclass(frozen=True, slots=True)
class AgentArtifact:
    """A file or URI produced by an agent invocation."""

    path: Path | str
    kind: str = "file"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("artifact kind must not be empty")


@dataclass(frozen=True, slots=True)
class AgentRunSpec:
    """Provider-independent inputs for one agent invocation."""

    system_prompt: str
    user_prompt: str
    schema: Mapping[str, Any]
    cwd: Path
    model: str
    log_path: Path
    result_path: Path
    allowed_tools: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    effort: str | None = None
    billing_mode: BillingMode = BillingMode.API
    resume_token: str | None = None
    activity_sink: Callable[[AgentActivity], None] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cwd", Path(self.cwd))
        object.__setattr__(self, "log_path", Path(self.log_path))
        object.__setattr__(self, "result_path", Path(self.result_path))
        object.__setattr__(self, "allowed_tools", tuple(self.allowed_tools))
        object.__setattr__(self, "billing_mode", BillingMode(self.billing_mode))
        if not self.model:
            raise ValueError("agent model must not be empty")
        if not isinstance(self.schema, Mapping):
            raise TypeError("agent result schema must be a mapping")


@dataclass(frozen=True, slots=True)
class AgentUsage:
    """Optional resource accounting reported by an adapter.

    ``tokens_input`` is the uncached input portion. Cache reads and writes are
    recorded separately so pricing never counts the same input twice.
    """

    cost_usd: float | None = None
    tokens_input: int | None = None
    tokens_output: int | None = None
    tokens_cache_read: int | None = None
    tokens_cache_write: int | None = None
    num_turns: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "cost_usd",
            "tokens_input",
            "tokens_output",
            "tokens_cache_read",
            "tokens_cache_write",
            "num_turns",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"usage field {name} must not be negative")


def combine_agent_usage(usages: Iterable[AgentUsage | None]) -> AgentUsage | None:
    """Sum reported usage, preserving unknown fields as ``None``."""
    reported = [usage for usage in usages if usage is not None]
    if not reported:
        return None

    def total(name: str) -> int | float | None:
        values = [
            value for usage in reported if (value := getattr(usage, name)) is not None
        ]
        return sum(values) if values else None

    return AgentUsage(
        cost_usd=total("cost_usd"),
        tokens_input=total("tokens_input"),
        tokens_output=total("tokens_output"),
        tokens_cache_read=total("tokens_cache_read"),
        tokens_cache_write=total("tokens_cache_write"),
        num_turns=total("num_turns"),
    )


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Durable outcome of an adapter invocation."""

    status: AgentStatus
    log_path: Path
    exit_code: int | None = None
    payload: dict[str, Any] | None = None
    error: str | None = None
    result_path: Path | None = None
    duration_seconds: float = 0.0
    usage: AgentUsage | None = None
    billing_mode: BillingMode = BillingMode.API
    provider: str | None = None
    model: str | None = None
    artifacts: tuple[AgentArtifact, ...] = ()
    attempts: int = 1
    retryable: bool = False
    resume_token: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", AgentStatus(self.status))
        object.__setattr__(self, "log_path", Path(self.log_path))
        object.__setattr__(self, "billing_mode", BillingMode(self.billing_mode))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        if self.result_path is not None:
            object.__setattr__(self, "result_path", Path(self.result_path))
        if self.duration_seconds < 0:
            raise ValueError("duration must not be negative")
        if self.attempts < 0:
            raise ValueError("attempt count must not be negative")
        if self.provider == "":
            raise ValueError("provider must not be empty")
        if self.model == "":
            raise ValueError("model must not be empty")

    @property
    def resumable(self) -> bool:
        """Return whether a later run can resume this outcome."""
        return self.retryable or self.resume_token is not None


@runtime_checkable
class AgentAdapter(Protocol):
    """Protocol implemented by every native, API, CLI, and mock adapter."""

    name: str
    capabilities: AgentCapabilities

    def run(
        self,
        spec: AgentRunSpec,
        *,
        cancel: CancellationToken | None = None,
    ) -> AgentResult: ...
