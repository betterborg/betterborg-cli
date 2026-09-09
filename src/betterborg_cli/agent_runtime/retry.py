"""Bounded retry for transient service errors and for missed result schemas."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from betterborg_cli.agent_runtime.base import (
    AgentArtifact,
    AgentResult,
    AgentRunSpec,
    AgentStatus,
    AgentUsage,
    CancellationToken,
)
from betterborg_cli.agent_runtime.structured import StructuredResultError

DEFAULT_TRANSIENT_BACKOFF_SECONDS = 300.0
DEFAULT_TRANSIENT_MAX_ATTEMPTS = 6
DEFAULT_SCHEMA_MAX_ATTEMPTS = 3

_SCHEMA_FAILURE = """
## Rejected result

Your previous result did not satisfy the required JSON Schema:

{error}

Send a result that satisfies every schema constraint above.
""".strip()

_SCHEMA_CORRECTION = """
## Rejected result

Your previous result did not satisfy the required JSON Schema:

{error}

That rejected result is below. Fix what the failure names, then re-check the
whole result for anything else the schema would reject before returning it:
validation stops at the first value it rejects, so a rejection usually means
more remain. Return the whole result again.

```json
{result}
```
""".strip()


@dataclass(slots=True)
class SchemaRetry:
    """Bounded, immediate retry of a result that missed its schema.

    A missed schema is a property of one sampled result rather than a service
    fault, so the next attempt starts at once: the transient backoff exists for
    rate limits and outages and would buy nothing here. Each retry carries the
    validating error, so the agent is told what to correct instead of being
    asked the same question again. Every adapter shares this budget and this
    wording; only the transport that delivers the correction differs.

    The correction carries the rejected result as well as the failing
    constraint, so an attempt that starts in a fresh process repairs the value
    the failure names instead of sampling a whole new result from the
    distribution that produced the mistake. It carries the result whole,
    because the agent is being asked to return a whole result and validation
    stops at the first value it rejects, so the parts the failure does not name
    are the ones it still has to get right.

    The rejected result travels in the correction and never in the validating
    error, which reaches logs, exceptions and stored state. The correction is
    the agent's own output returning to the provider that produced it.

    A caller that repairs malformed output itself multiplies its own budget by
    this one, which is bounded but paid in whole turns.
    """

    max_attempts: int = DEFAULT_SCHEMA_MAX_ATTEMPTS
    attempts: int = 0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")

    def correction(
        self,
        error: StructuredResultError,
        rejected: Mapping[str, Any],
    ) -> str | None:
        """Return what to tell the agent, or ``None`` once attempts run out."""
        self.attempts += 1
        if self.attempts >= self.max_attempts:
            return None
        try:
            result = json.dumps(rejected, indent=2, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError):
            # A result holding a non-finite number cannot be shown as the JSON
            # it must be sent back as, so this attempt names the constraint
            # alone rather than fencing something the agent cannot copy.
            return _SCHEMA_FAILURE.format(error=error)
        return _SCHEMA_CORRECTION.format(error=error, result=result)


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
