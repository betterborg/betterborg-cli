"""Shared progress fault injection for planning lifecycle tests."""

from __future__ import annotations

from betterborg_cli.progress import ChildRecord, RunProgress, StageRecord


class BoundaryInterruptProgress(RunProgress):
    """Interrupt once at a named progress lifecycle boundary."""

    def __init__(self, interrupt_at: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.interrupt_at = interrupt_at

    def start(self, stage_key: str) -> StageRecord:
        record = super().start(stage_key)
        self._interrupt("after-start")
        return record

    def complete(
        self, stage_key: str, result: object | None = None
    ) -> StageRecord:
        self._interrupt("before-complete")
        return super().complete(stage_key, result)

    def complete_child(
        self,
        stage_key: str,
        child_key: str,
        result: object | None = None,
    ) -> ChildRecord:
        self._interrupt("before-complete-child")
        return super().complete_child(stage_key, child_key, result)

    def _interrupt(self, boundary: str) -> None:
        if self.interrupt_at == boundary:
            self.interrupt_at = ""
            raise KeyboardInterrupt(f"{boundary} interrupted")
