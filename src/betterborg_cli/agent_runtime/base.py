"""Shared contracts for invoking BetterBorg agent adapters."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping
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


class CancellationToken:
    """Thread-safe cooperative cancellation signal."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Signal cancellation and wake any waiters."""
        self._event.set()

    def is_set(self) -> bool:
        """Return whether cancellation was requested."""
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for cancellation, returning whether it was requested."""
        return self._event.wait(timeout)


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
