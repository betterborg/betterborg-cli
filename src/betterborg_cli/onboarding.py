"""Interactive onboarding doors and shared Borg creation dispatch."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Protocol

from betterborg_cli.agent_runtime.base import AgentAdapter, CancellationToken
from betterborg_cli.agent_runtime.selection import SelectedAgent
from betterborg_cli.prd_session import (
    Editor,
    InteractiveIO,
    PrdSession,
    PrdSessionResult,
    validate_borg_name,
)
from betterborg_cli.progress import RunProgress
from betterborg_cli.repo_analysis import ImprovementPrd
from betterborg_cli.repo_analysis.text_rendering import terminal_text
from betterborg_cli.store import Repository, SqliteStore


class BorgCreator(Protocol):
    """The creation boundary shared by every onboarding door."""

    def create(self, name: str, source: Path | None = None) -> PrdSessionResult:
        """Create one named Borg from an optional source PRD."""


class CreateService:
    """Configure and invoke the shared PRD session for one repository."""

    def __init__(
        self,
        repository: Repository,
        store: SqliteStore,
        agent: AgentAdapter | SelectedAgent,
        *,
        io: InteractiveIO | None = None,
        editor: Editor | None = None,
        interactive: bool = True,
        cancel: CancellationToken | None = None,
        progress: RunProgress | None = None,
    ) -> None:
        self._session = PrdSession(
            repository,
            store,
            agent,
            io=io,
            editor=editor,
            interactive=interactive,
            cancel=cancel,
            progress=progress,
        )

    def create(
        self,
        name: str,
        source: Path | None = None,
        *,
        confirmed: bool = False,
    ) -> PrdSessionResult:
        """Run the common interview, review, and final confirmation flow."""
        return self._session.run(name, source, confirmed=confirmed)


class OnboardingDispatcher:
    """Present the three first-run doors and dispatch one selected PRD."""

    _DOORS = (
        "Fix the repo",
        "Improve an existing PRD",
        "Brainstorm a new PRD",
    )

    def __init__(
        self,
        repository: Repository,
        store: SqliteStore,
        io: InteractiveIO,
        creator: BorgCreator,
        improvement_prds: Sequence[ImprovementPrd],
        *,
        cancel: CancellationToken | None = None,
        progress: RunProgress | None = None,
    ) -> None:
        self.repository = repository
        self.store = store
        self.io = io
        self.creator = creator
        self.improvement_prds = tuple(improvement_prds)
        self.cancel = cancel
        self.progress = progress

    def run(self) -> PrdSessionResult | None:
        """Choose a door and dispatch it, or return ``None`` on cancellation."""
        if self._cancelled():
            return None
        with self._suspend_output():
            self.io.write("What would you like to do next?")
            for index, door in enumerate(self._DOORS, start=1):
                self.io.write(f"  {index}. {door}")

            door = self._choose(
                "Choose a door (1-3, or q to cancel)",
                count=len(self._DOORS),
            )
        if door is None:
            return None
        if door == 1:
            return self._fix_the_repo()
        if door == 2:
            return self._improve_prd()
        return self._brainstorm()

    def _fix_the_repo(self) -> PrdSessionResult | None:
        if self._cancelled():
            return None
        if not self.improvement_prds:
            with self._suspend_output():
                self.io.write("No repository improvement themes were generated.")
            return None
        with self._suspend_output():
            self.io.write("Ranked repository improvement themes:")
            for index, document in enumerate(self.improvement_prds, start=1):
                self.io.write(
                    f"  {index}. {terminal_text(document.title)} — predicted impact "
                    f"+{document.predicted_impact:g}; effort {document.effort}"
                )
            selection = self._choose(
                f"Choose a theme [1] (1-{len(self.improvement_prds)}, or q to cancel)",
                count=len(self.improvement_prds),
                default=1,
            )
        if selection is None:
            return None
        document = self.improvement_prds[selection - 1]
        with self._suspend_output():
            name = self._prompt_name(document.suggested_borg_name)
        if name is None or self._cancelled():
            return None
        return self.creator.create(name, document.path)

    def _improve_prd(self) -> PrdSessionResult | None:
        if self._cancelled():
            return None
        with self._suspend_output():
            answer = self.io.prompt("Local Markdown PRD path (or q to cancel)")
        if self._cancelled():
            return None
        if answer is None or answer.casefold() == "q":
            return None
        source = Path(answer)
        if not source.is_absolute():
            source = self.repository.root / source
        with self._suspend_output():
            name = self._prompt_name(source.stem)
        if name is None or self._cancelled():
            return None
        return self.creator.create(name, source)

    def _brainstorm(self) -> PrdSessionResult | None:
        if self._cancelled():
            return None
        with self._suspend_output():
            name = self._prompt_name()
        if name is None or self._cancelled():
            return None
        return self.creator.create(name)

    def _prompt_name(self, suggested: str | None = None) -> str | None:
        while True:
            if self._cancelled():
                return None
            suffix = f" [{suggested}]" if suggested else ""
            answer = self.io.prompt(f"Borg name{suffix} (or q to cancel)")
            if self._cancelled():
                return None
            if answer is None or answer.casefold() == "q":
                return None
            name = suggested if not answer and suggested is not None else answer
            try:
                validate_borg_name(name)
            except ValueError as error:
                self.io.write(f"Invalid Borg name: {error}")
                continue
            destination = self.repository.root / ".borg" / "prds" / f"{name}.md"
            if (
                self.store.get_borg_by_name(self.repository.id, name) is not None
                or destination.exists()
                or destination.is_symlink()
            ):
                self.io.write(f"Borg name already exists: {name!r}. Choose another.")
                suggested = None
                continue
            return name

    def _choose(
        self,
        message: str,
        *,
        count: int,
        default: int | None = None,
    ) -> int | None:
        while True:
            if self._cancelled():
                return None
            answer = self.io.prompt(message)
            if self._cancelled():
                return None
            if answer is None or answer.casefold() == "q":
                return None
            if not answer and default is not None:
                return default
            try:
                selection = int(answer)
            except ValueError:
                selection = 0
            if 1 <= selection <= count:
                return selection
            self.io.write(f"Choose a number from 1 to {count}, or q to cancel.")

    def _cancelled(self) -> bool:
        return self.cancel is not None and self.cancel.is_set()

    def _suspend_output(self) -> AbstractContextManager[object]:
        return (
            self.progress.suspend()
            if self.progress is not None
            else nullcontext()
        )


def create_commands(
    repository_root: Path, improvement_prds: Sequence[ImprovementPrd]
) -> tuple[tuple[str, ...], ...]:
    """Return one shell-free create argv for each ranked generated theme."""
    return tuple(
        (
            "betterborg",
            "create",
            document.suggested_borg_name,
            "--prd",
            document.path.relative_to(repository_root).as_posix(),
        )
        for document in improvement_prds
    )
