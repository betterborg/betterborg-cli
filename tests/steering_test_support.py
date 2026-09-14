"""Adapters that reproduce the two things a cancelled status covers.

The provider adapters report an operator's stop and a bounded transient retry
budget run dry with the same status, and only the run's cancellation token
tells them apart. Every loop that treats the two differently needs both, so
one shape produces both.
"""

from __future__ import annotations

from betterborg_cli.agent_runtime.base import (
    AgentCapabilities,
    AgentResult,
    AgentRunSpec,
    AgentStatus,
    BillingMode,
    CancellationToken,
)


class CancellingAgent:
    """Report a cancellation, optionally stopping the run on the way out."""

    def __init__(
        self,
        *,
        stops: bool,
        cancel: CancellationToken | None = None,
        name: str = "openai",
    ) -> None:
        self.name = name
        self.capabilities = AgentCapabilities(
            billing_modes=frozenset(BillingMode),
            structured_output=True,
            tool_allowlist=True,
        )
        self.calls: list[AgentRunSpec] = []
        self._stops = stops
        self._cancel = cancel

    def run(
        self,
        spec: AgentRunSpec,
        *,
        cancel: CancellationToken | None = None,
    ) -> AgentResult:
        self.calls.append(spec)
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        spec.log_path.write_text("cancelled\n", encoding="utf-8")
        token = self._cancel or cancel
        if self._stops and token is not None:
            token.cancel()
        return AgentResult(
            status=AgentStatus.CANCELLED,
            error=(
                "operator stopped the run"
                if self._stops
                else "transient provider failures exhausted the retry budget"
            ),
            log_path=spec.log_path,
            billing_mode=spec.billing_mode,
            retryable=True,
        )


class UnpersistedNoteAgent:
    """Fail while still returning the payload the turn produced.

    The native adapters do exactly this when they cannot write the result
    file: the payload has already passed structured validation, and the turn
    fails on the write. So a failed result can carry a usable note, and a
    reader that takes the payload without reading the status takes it.
    """

    def __init__(
        self,
        *,
        note: str,
        cancel: CancellationToken | None = None,
        name: str = "openai",
    ) -> None:
        self.name = name
        self.capabilities = AgentCapabilities(
            billing_modes=frozenset(BillingMode),
            structured_output=True,
            tool_allowlist=True,
        )
        self.calls: list[AgentRunSpec] = []
        self._note = note
        self._cancel = cancel

    def run(
        self,
        spec: AgentRunSpec,
        *,
        cancel: CancellationToken | None = None,
    ) -> AgentResult:
        self.calls.append(spec)
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        spec.log_path.write_text("failed\n", encoding="utf-8")
        token = self._cancel or cancel
        if token is not None:
            token.cancel()
        return AgentResult(
            status=AgentStatus.FAILED,
            payload={"note": self._note, "confidence": "high"},
            error="unable to persist result: disk full",
            log_path=spec.log_path,
            billing_mode=spec.billing_mode,
        )
