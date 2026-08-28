"""Bounded transient retry with cancellation-aware backoff."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from betterborg_cli.agent_runtime.base import (
    AgentArtifact,
    AgentResult,
    AgentRunSpec,
    AgentStatus,
    AgentUsage,
    CancellationToken,
)

DEFAULT_TRANSIENT_BACKOFF_SECONDS = 300.0
DEFAULT_TRANSIENT_MAX_ATTEMPTS = 6


@dataclass(frozen=True, slots=True)
class RetryOutcome:
    """Process outcome plus the state needed to resume after exhaustion."""

    exit_code: int
    attempts: int
    transient_reason: str | None = None

    @property
    def cancelled(self) -> bool:
        return self.exit_code == -1 and self.transient_reason is None

    @property
    def exhausted(self) -> bool:
        return self.transient_reason is not None

    @property
    def resumable(self) -> bool:
        return self.exhausted


def run_with_transient_retry(
    run_once: Callable[[], int],
    classify: Callable[[int], str | None],
    *,
    cancel: CancellationToken | None = None,
    backoff_seconds: float = DEFAULT_TRANSIENT_BACKOFF_SECONDS,
    max_attempts: int = DEFAULT_TRANSIENT_MAX_ATTEMPTS,
) -> RetryOutcome:
    """Retry classified transient exits and retain exhaustion as resumable."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    if backoff_seconds < 0:
        raise ValueError("backoff_seconds must not be negative")
    attempts = 0
    while attempts < max_attempts:
        if cancel is not None and cancel.is_set():
            return RetryOutcome(exit_code=-1, attempts=attempts)
        attempts += 1
        exit_code = run_once()
        if exit_code == -1:
            return RetryOutcome(exit_code=-1, attempts=attempts)
        reason = classify(exit_code)
        if reason is None:
            return RetryOutcome(exit_code=exit_code, attempts=attempts)
        if attempts == max_attempts:
            return RetryOutcome(
                exit_code=exit_code,
                attempts=attempts,
                transient_reason=reason,
            )
        if cancel is not None:
            if cancel.wait(backoff_seconds):
                return RetryOutcome(exit_code=-1, attempts=attempts)
        elif backoff_seconds:
            time.sleep(backoff_seconds)
    raise AssertionError("retry loop ended without an outcome")


def retry_outcome_to_result(
    outcome: RetryOutcome,
    spec: AgentRunSpec,
    *,
    duration_seconds: float,
    usage: AgentUsage | None = None,
    artifacts: Sequence[AgentArtifact] = (),
) -> AgentResult:
    """Convert cancellation or exhaustion into a resumable agent result."""
    if not outcome.cancelled and not outcome.exhausted:
        raise ValueError("only cancelled or exhausted retries can be converted")
    error = (
        f"transient retry exhausted: {outcome.transient_reason}"
        if outcome.exhausted
        else "agent run cancelled"
    )
    return AgentResult(
        status=AgentStatus.CANCELLED,
        exit_code=outcome.exit_code,
        error=error,
        log_path=spec.log_path,
        duration_seconds=duration_seconds,
        usage=usage,
        billing_mode=spec.billing_mode,
        artifacts=tuple(artifacts),
        attempts=outcome.attempts,
        retryable=True,
        resume_token=spec.resume_token,
    )
