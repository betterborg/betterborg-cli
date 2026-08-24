"""Shared interactive interviewing and confirmation for Borg PRDs."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, TypeAlias
from uuid import uuid4

from betterborg_cli.agent_runtime.base import (
    AgentAdapter,
    AgentRunSpec,
    AgentStatus,
    CancellationToken,
)
from betterborg_cli.agent_runtime.selection import SelectedAgent
from betterborg_cli.agent_runtime.structured import validate_structured_result
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.store import (
    Borg,
    Repository,
    SqliteStore,
)
from betterborg_cli.store import (
    PrdSession as StoredPrdSession,
)

PRD_SESSION_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "BetterBorg PRD interview turn",
    "type": "object",
    "additionalProperties": False,
    "required": ["questions", "prd_markdown"],
    "properties": {
        "questions": {
            "type": "array",
            "maxItems": 8,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "prd_markdown": {
            "type": ["string", "null"],
            "minLength": 1,
            "description": "The complete improved PRD, or null while asking questions.",
        },
    },
}

_SYSTEM_PROMPT = """You facilitate a concise product-requirements interview.
Improve the supplied Markdown when it exists; otherwise help brainstorm a useful
PRD from scratch. Ask only material product questions that cannot be answered by
repository inspection. Ask at most eight concise questions in one turn. Return
questions with prd_markdown set to null when answers are needed. Otherwise return
an empty questions array and the complete PRD as Markdown. Never start planning,
design an implementation, or modify files. Return only the required JSON object.
"""
_BRAINSTORM_OPENING = (
    "No starting PRD was supplied. Help me brainstorm and write a product "
    "requirements document."
)
_DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-8",
    "claude": "claude-opus-4-8",
    "codex": "gpt-5",
    "openai": "gpt-5",
}
_WINDOWS_RESERVED_BASENAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}
_WINDOWS_FORBIDDEN_FILENAME_CHARACTERS = frozenset('<>:"/\\|?*')

Prompt: TypeAlias = Callable[[str], str | None]
Confirm: TypeAlias = Callable[[str, bool], bool]
Write: TypeAlias = Callable[[str], None]
Editor: TypeAlias = Callable[[str], str | None]


class PrdSessionError(RuntimeError):
    """Raised when a PRD session cannot safely produce a confirmed draft."""


class _PrdSessionCancelled(Exception):
    """Stop a cooperative agent cancellation without treating it as failure."""


class InteractiveIO:
    """Small injectable boundary for interactive prompts and rendering."""

    def __init__(self, *, prompt: Prompt, confirm: Confirm, write: Write) -> None:
        self._prompt = prompt
        self._confirm = confirm
        self._write = write

    def prompt(self, message: str) -> str | None:
        """Return an answer, or ``None`` when the user cancels."""
        return self._prompt(message)

    def confirm(self, message: str, *, default: bool = False) -> bool:
        """Ask one yes/no question with an explicit default."""
        return self._confirm(message, default)

    def write(self, message: str) -> None:
        """Render reviewable text for the user."""
        self._write(message)


@dataclass(frozen=True, slots=True)
class PrdSessionResult:
    """Outcome of a shared PRD session, including an unconfirmed draft."""

    borg: Borg
    session: StoredPrdSession
    prd_path: Path
    confirmed: bool
    body_md: str | None = None
    questions: tuple[str, ...] = ()

    @property
    def cancelled(self) -> bool:
        """Return whether the session ended without confirmation or questions."""
        return not self.confirmed and not self.questions


class PrdSession:
    """Run the common PRD interview used by every onboarding entry door."""

    def __init__(
        self,
        repository: Repository,
        store: SqliteStore,
        agent: AgentAdapter | SelectedAgent,
        *,
        io: InteractiveIO | None = None,
        editor: Editor | None = None,
        interactive: bool = True,
        artifact_dir: Path | None = None,
        model: str | None = None,
        cancel: CancellationToken | None = None,
    ) -> None:
        if store.get_repository(repository.id) != repository:
            raise ValueError("repository must already be present in the supplied store")
        paths = RepoPaths.discover(repository.root)
        if paths.root != repository.root:
            raise ValueError("repository root does not match its discovered Git root")
        if interactive and io is None:
            raise ValueError("interactive PRD sessions require InteractiveIO")

        self.repository = repository
        self.store = store
        self.agent = agent
        self.paths = paths
        self.io = io
        self.editor = editor
        self.interactive = interactive
        self.artifact_dir = Path(artifact_dir or paths.artifacts_dir / "prd-sessions")
        self.model = model or _selected_model(agent)
        self.cancel = cancel

    def run(
        self,
        name: str,
        source: Path | None = None,
        *,
        confirmed: bool = False,
    ) -> PrdSessionResult:
        """Interview, review, and optionally confirm one named Borg PRD.

        ``source`` is always read as local Markdown and is never edited. In a
        noninteractive caller, ``confirmed=True`` is the explicit final
        confirmation; material questions are returned to the caller instead of
        being prompted.
        """
        _validate_borg_name(name)
        initial_markdown = _read_source(source) if source is not None else None
        relative_prd_path = Path(".borg") / "prds" / f"{name}.md"
        prd_path = self.repository.root / relative_prd_path
        if source is not None and source.resolve() == prd_path.resolve():
            raise ValueError("source PRD cannot also be the confirmed output path")
        if self.store.get_borg_by_name(self.repository.id, name) is not None:
            raise ValueError(f"Borg name already exists in this repository: {name!r}")
        if prd_path.exists() or prd_path.is_symlink():
            raise FileExistsError(f"confirmed Borg PRD already exists: {prd_path}")

        borg = Borg(repository_id=self.repository.id, name=name)
        session = StoredPrdSession(
            repository_id=self.repository.id,
            borg_id=borg.id,
            prd_path=relative_prd_path,
        )
        with self.store.transaction():
            self.store.add_borg(borg)
            self.store.add_prd_session(session)
            self.store.append_prd_turn(
                session_id=session.id,
                role="user",
                content=initial_markdown or _BRAINSTORM_OPENING,
            )

        base_result = {
            "borg": borg,
            "session": session,
            "prd_path": prd_path,
        }
        if self.cancel is not None and self.cancel.is_set():
            return PrdSessionResult(**base_result, confirmed=False)

        round_number = 0
        while True:
            round_number += 1
            try:
                payload = self._run_agent(session, round_number)
            except _PrdSessionCancelled:
                return PrdSessionResult(**base_result, confirmed=False)
            questions = tuple(question.strip() for question in payload["questions"])
            draft = payload["prd_markdown"]
            if bool(questions) == bool(draft):
                raise PrdSessionError(
                    "PRD agent must return either material questions or a draft"
                )

            if questions:
                self.store.append_prd_turn(
                    session_id=session.id,
                    role="assistant",
                    content="\n".join(f"- {question}" for question in questions),
                )
                if not self.interactive:
                    return PrdSessionResult(
                        **base_result,
                        confirmed=False,
                        questions=questions,
                    )
                if not self._answer_questions(session, questions):
                    return PrdSessionResult(**base_result, confirmed=False)
                continue

            body_md = _normalize_draft(draft)
            self.store.append_prd_turn(
                session_id=session.id,
                role="assistant",
                content=body_md,
            )
            break

        if self.interactive:
            assert self.io is not None
            self.io.write(body_md)
            if self.editor is not None and self.io.confirm(
                "Review and edit this PRD in your editor?", default=False
            ):
                edited = self.editor(body_md)
                if edited is not None:
                    body_md = _normalize_draft(edited)
                    self.store.append_prd_turn(
                        session_id=session.id,
                        role="user",
                        content=body_md,
                    )
                    self.io.write(body_md)
            confirmed = self.io.confirm(
                f"Create Borg {name!r} with this PRD?", default=False
            )

        if not confirmed:
            return PrdSessionResult(
                **base_result,
                confirmed=False,
                body_md=body_md,
            )

        _publish_confirmed_prd(prd_path, body_md, root=self.repository.root)
        return PrdSessionResult(
            **base_result,
            confirmed=True,
            body_md=body_md,
        )

    def _run_agent(
        self, session: StoredPrdSession, round_number: int
    ) -> dict[str, Any]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        history = self.store.list_prd_turns(session.id)
        user_prompt = "\n\n".join(
            f"## {turn.role.title()}\n\n{turn.content}" for turn in history
        )
        spec = AgentRunSpec(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=PRD_SESSION_OUTPUT_SCHEMA,
            cwd=self.repository.root,
            model=self.model,
            log_path=self.artifact_dir / f"{session.id}.round-{round_number}.log",
            result_path=self.artifact_dir / f"{session.id}.round-{round_number}.json",
        )
        try:
            result = self.agent.run(spec, cancel=self.cancel)
        except Exception as error:
            raise PrdSessionError(f"PRD agent crashed: {error}") from error
        if result.status is AgentStatus.CANCELLED:
            raise _PrdSessionCancelled
        if result.status is not AgentStatus.COMPLETED or result.payload is None:
            raise PrdSessionError(
                result.error or f"PRD agent returned {result.status.value}"
            )
        validate_structured_result(result.payload, PRD_SESSION_OUTPUT_SCHEMA)
        return result.payload

    def _answer_questions(
        self, session: StoredPrdSession, questions: tuple[str, ...]
    ) -> bool:
        assert self.io is not None
        for question in questions:
            answer = self.io.prompt(question)
            if answer is None:
                return False
            answer = answer.strip()
            if not answer:
                raise PrdSessionError("material question answers must not be empty")
            self.store.append_prd_turn(
                session_id=session.id,
                role="user",
                content=answer,
            )
        return True


def _selected_model(agent: AgentAdapter | SelectedAgent) -> str:
    if isinstance(agent, SelectedAgent) and agent.model is not None:
        return agent.model
    return _DEFAULT_MODELS.get(agent.name, "prd-session-model")


def _validate_borg_name(name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Borg name must not be empty")
    if name != name.strip() or "\n" in name or "\r" in name:
        raise ValueError("Borg name must be a single trimmed line")
    if any(
        ord(character) < 32
        or character in _WINDOWS_FORBIDDEN_FILENAME_CHARACTERS
        for character in name
    ):
        raise ValueError("Borg name must be a portable filename stem")
    path = Path(name)
    windows_path = PureWindowsPath(name)
    if (
        path.name != name
        or windows_path.name != name
        or name in {".", ".."}
        or name.endswith((".", " "))
    ):
        raise ValueError("Borg name must be a portable filename stem")
    if name.casefold().split(".", 1)[0] in _WINDOWS_RESERVED_BASENAMES:
        raise ValueError("Borg name must not be a reserved filename")


def _read_source(source: Path) -> str:
    source = Path(source)
    if source.suffix.casefold() != ".md":
        raise ValueError("PRD input must be a local Markdown file")
    if not source.is_file():
        raise ValueError(f"PRD input is not a readable local file: {source}")
    try:
        body = source.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read PRD input {source}: {error}") from error
    if not body.strip():
        raise ValueError("PRD input must not be empty")
    return body


def _normalize_draft(body: object) -> str:
    if not isinstance(body, str) or not body.strip():
        raise PrdSessionError("PRD draft must not be empty")
    return body if body.endswith("\n") else f"{body}\n"


def _publish_confirmed_prd(path: Path, body: str, *, root: Path) -> None:
    parent = path.parent
    if not parent.resolve().is_relative_to(root):
        raise PrdSessionError(f"Borg PRD directory escapes repository: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = parent.resolve(strict=True)
    if not resolved_parent.is_relative_to(root):
        raise PrdSessionError(f"Borg PRD directory escapes repository: {parent}")

    destination = resolved_parent / path.name
    temporary = resolved_parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as output:
            output.write(body)
        os.link(temporary, destination)
    except FileExistsError as error:
        raise FileExistsError(f"confirmed Borg PRD already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)
