"""Shared interactive interviewing and confirmation for Borg PRDs."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, TypeAlias

from betterborg_cli.agent_runtime.api_tools import READ_ONLY_API_TOOLS
from betterborg_cli.agent_runtime.base import (
    AgentAdapter,
    AgentRunSpec,
    AgentStatus,
    CancellationToken,
)
from betterborg_cli.agent_runtime.selection import (
    AgentSelectionError,
    SelectedAgent,
    require_read_only_agent,
    resolve_agent_model,
)
from betterborg_cli.agent_runtime.structured import validate_structured_result
from betterborg_cli.progress import RunProgress, StageSpec, StageState
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_files import (
    RepositoryPathError,
    is_windows_reserved_filename,
    publish_repository_text,
    read_repository_text,
)
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
    "title": "Betterborg PRD interview turn",
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
_WINDOWS_FORBIDDEN_FILENAME_CHARACTERS = frozenset('<>:"/\\|?*')
_REQUIREMENTS_STAGE_KEY = "requirements"

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
        progress: RunProgress | None = None,
    ) -> None:
        if store.get_repository(repository.id) != repository:
            raise ValueError("repository must already be present in the supplied store")
        paths = RepoPaths.discover(repository.root, cancel=cancel)
        if paths.root != repository.root:
            raise ValueError("repository root does not match its discovered Git root")
        if interactive and io is None:
            raise ValueError("interactive PRD sessions require InteractiveIO")
        require_read_only_agent(agent, role="PRD", error_factory=PrdSessionError)
        try:
            resolved_model = resolve_agent_model(agent, model)
        except AgentSelectionError as error:
            raise PrdSessionError(str(error)) from error

        self.repository = repository
        self.store = store
        self.agent = agent
        self.paths = paths
        self.io = io
        self.editor = editor
        self.interactive = interactive
        self.artifact_dir = Path(artifact_dir or paths.artifacts_dir / "prd-sessions")
        self.model = resolved_model
        self.cancel = cancel
        self.progress = progress
        if progress is not None:
            progress.declare(
                StageSpec(_REQUIREMENTS_STAGE_KEY, "Gather requirements")
            )

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
        borg, session, prd_path = _new_borg_records(self.repository, name)
        base_result = {
            "borg": borg,
            "session": session,
            "prd_path": prd_path,
        }
        if self._cancelled():
            return PrdSessionResult(**base_result, confirmed=False)

        initial_markdown = _read_source(source) if source is not None else None
        _require_unclaimed_borg(self.store, self.repository, name, prd_path, source)
        _open_prd_session(
            self.store,
            borg,
            session,
            initial_markdown or _BRAINSTORM_OPENING,
        )

        if self._cancelled():
            return PrdSessionResult(**base_result, confirmed=False)

        if self.progress is not None:
            self.progress.start(_REQUIREMENTS_STAGE_KEY)
            self.progress.update(_REQUIREMENTS_STAGE_KEY, "1 turn recorded")

        try:
            round_number = 0
            while True:
                round_number += 1
                payload = self._run_agent(session, round_number)
                questions = _normalize_questions(payload["questions"])
                draft = payload["prd_markdown"]
                if bool(questions) == bool(draft):
                    raise PrdSessionError(
                        "PRD agent must return either material questions or a draft"
                    )

                if questions:
                    self._append_turn(
                        session,
                        role="assistant",
                        content="\n".join(
                            f"- {question}" for question in questions
                        ),
                    )
                    if self._cancelled():
                        return self._stopped_result(base_result, "interrupted")
                    if not self.interactive:
                        return self._stopped_result(
                            base_result,
                            "questions pending",
                            questions=questions,
                        )
                    if not self._answer_questions(session, questions):
                        reason = (
                            "interrupted" if self._cancelled() else "cancelled"
                        )
                        return self._stopped_result(base_result, reason)
                    continue

                body_md = _normalize_draft(draft)
                self._append_turn(
                    session,
                    role="assistant",
                    content=body_md,
                )
                break

            if self._cancelled():
                return self._stopped_result(
                    base_result,
                    "interrupted",
                    body_md=body_md,
                )

            if self.interactive:
                assert self.io is not None
                with self._suspend_output():
                    self.io.write(body_md)
                    edit_requested = False
                    if not self._cancelled() and self.editor is not None:
                        edit_requested = self.io.confirm(
                            "Review and edit this PRD in your editor?",
                            default=False,
                        )
                    if edit_requested and not self._cancelled():
                        edited = self.editor(body_md)
                        if edited is not None:
                            body_md = _normalize_draft(edited)
                            self._append_turn(
                                session,
                                role="user",
                                content=body_md,
                            )
                            if not self._cancelled():
                                self.io.write(body_md)
                    if not self._cancelled():
                        confirmed = self.io.confirm(
                            f"Create Borg {name!r} with this PRD?", default=False
                        )

            if self._cancelled():
                return self._stopped_result(
                    base_result,
                    "interrupted",
                    body_md=body_md,
                )

            if not confirmed:
                return self._stopped_result(
                    base_result,
                    "not confirmed",
                    body_md=body_md,
                )

            try:
                _publish_confirmed_prd(
                    prd_path,
                    body_md,
                    root=self.repository.root,
                )
            except FileExistsError:
                raise
            except BaseException:
                if not _confirmed_prd_matches(
                    prd_path,
                    body_md,
                    root=self.repository.root,
                ):
                    raise
            result = PrdSessionResult(
                **base_result,
                confirmed=True,
                body_md=body_md,
            )
            if self.progress is not None:
                self.progress.complete(
                    _REQUIREMENTS_STAGE_KEY,
                    f"PRD {name!r} confirmed",
                )
            return result
        except _PrdSessionCancelled:
            return self._stopped_result(base_result, "interrupted")
        except BaseException as error:
            if self.progress is not None:
                record = self.progress.stages[_REQUIREMENTS_STAGE_KEY]
                if record.state is StageState.RUNNING:
                    if _is_interruption(error, self.cancel):
                        self.progress.stop(_REQUIREMENTS_STAGE_KEY, "interrupted")
                    else:
                        self.progress.fail(
                            _REQUIREMENTS_STAGE_KEY,
                            str(error) or type(error).__name__,
                        )
            raise

    def _append_turn(
        self,
        session: StoredPrdSession,
        *,
        role: str,
        content: str,
    ) -> None:
        self.store.append_prd_turn(
            session_id=session.id,
            role=role,
            content=content,
        )
        if self.progress is not None:
            turn_count = len(self.store.list_prd_turns(session.id))
            noun = "turn" if turn_count == 1 else "turns"
            self.progress.update(
                _REQUIREMENTS_STAGE_KEY,
                f"{turn_count} {noun} recorded",
            )

    def _stopped_result(
        self,
        base_result: dict[str, object],
        reason: str,
        *,
        body_md: str | None = None,
        questions: tuple[str, ...] = (),
    ) -> PrdSessionResult:
        if self.progress is not None:
            record = self.progress.stages[_REQUIREMENTS_STAGE_KEY]
            if record.state is StageState.RUNNING:
                self.progress.stop(_REQUIREMENTS_STAGE_KEY, reason)
        return PrdSessionResult(
            **base_result,
            confirmed=False,
            body_md=body_md,
            questions=questions,
        )

    def _cancelled(self) -> bool:
        return self.cancel is not None and self.cancel.is_set()

    def _suspend_output(self) -> AbstractContextManager[object]:
        return (
            self.progress.suspend()
            if self.progress is not None
            else nullcontext()
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
            allowed_tools=READ_ONLY_API_TOOLS,
            log_path=self.artifact_dir / f"{session.id}.round-{round_number}.log",
            result_path=self.artifact_dir / f"{session.id}.round-{round_number}.json",
            activity_sink=(
                lambda activity: self.progress.activity(
                    _REQUIREMENTS_STAGE_KEY, activity
                )
                if self.progress is not None
                else None
            ),
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
            if self._cancelled():
                return False
            with self._suspend_output():
                answer = self.io.prompt(question)
            if answer is None:
                return False
            answer = answer.strip()
            if not answer:
                raise PrdSessionError("material question answers must not be empty")
            self._append_turn(
                session,
                role="user",
                content=answer,
            )
            if self._cancelled():
                return False
        return True


def adopt_prd(
    repository: Repository,
    store: SqliteStore,
    name: str,
    source: Path,
    *,
    cancel: CancellationToken | None = None,
) -> PrdSessionResult:
    """Create one named Borg from an authoritative PRD, adopted verbatim.

    Adoption is the agent-free counterpart to :class:`PrdSession`. The caller
    already owns the requirements, so there is nothing to interview about and
    no draft to improve, and the source is published verbatim as UTF-8 text.
    Everything
    a confirmed interview records is recorded here too, so nothing downstream
    can tell an adopted Borg from an interviewed one.
    """
    if store.get_repository(repository.id) != repository:
        raise ValueError("repository must already be present in the supplied store")
    paths = RepoPaths.discover(repository.root, cancel=cancel)
    if paths.root != repository.root:
        raise ValueError("repository root does not match its discovered Git root")
    borg, session, prd_path = _new_borg_records(repository, name)
    body_md = _read_source(source)
    _require_unclaimed_borg(store, repository, name, prd_path, source)
    _open_prd_session(store, borg, session, body_md)
    # An adopted PRD tolerates a failed publish on the same terms an
    # interviewed one does: if the file on disk already holds the confirmed
    # body, the publish did its job and a late error unwinding it would
    # otherwise strand a fully written Borg behind a claimed name.
    try:
        _publish_confirmed_prd(prd_path, body_md, root=repository.root)
    except FileExistsError:
        raise
    except BaseException:
        if not _confirmed_prd_matches(prd_path, body_md, root=repository.root):
            raise
    return PrdSessionResult(
        borg=borg,
        session=session,
        prd_path=prd_path,
        confirmed=True,
        body_md=body_md,
    )


def validate_borg_name(name: str) -> None:
    """Require a nonempty Borg name that is a portable filename stem."""
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
    if is_windows_reserved_filename(name):
        raise ValueError("Borg name must not be a reserved filename")


def _new_borg_records(
    repository: Repository, name: str
) -> tuple[Borg, StoredPrdSession, Path]:
    """Return the Borg, its stored session, and its confirmed PRD path."""
    validate_borg_name(name)
    relative_prd_path = Path(".betterborg") / "prds" / f"{name}.md"
    borg = Borg(repository_id=repository.id, name=name)
    session = StoredPrdSession(
        repository_id=repository.id,
        borg_id=borg.id,
        prd_path=relative_prd_path,
    )
    return borg, session, repository.root / relative_prd_path


def _require_unclaimed_borg(
    store: SqliteStore,
    repository: Repository,
    name: str,
    prd_path: Path,
    source: Path | None,
) -> None:
    """Reject a name or output path the store, the tree, or the source holds."""
    if source is not None and source.resolve() == prd_path.resolve():
        raise ValueError("source PRD cannot also be the confirmed output path")
    if store.get_borg_by_name(repository.id, name) is not None:
        raise ValueError(f"Borg name already exists in this repository: {name!r}")
    if prd_path.exists() or prd_path.is_symlink():
        raise FileExistsError(f"confirmed Borg PRD already exists: {prd_path}")


def _open_prd_session(
    store: SqliteStore,
    borg: Borg,
    session: StoredPrdSession,
    opening: str,
) -> None:
    """Record the Borg, its session, and its opening turn in one transaction."""
    with store.transaction():
        store.add_borg(borg)
        store.add_prd_session(session)
        store.append_prd_turn(
            session_id=session.id,
            role="user",
            content=opening,
        )


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


def _normalize_questions(questions: list[str]) -> tuple[str, ...]:
    normalized = tuple(question.strip() for question in questions)
    if any(not question for question in normalized):
        raise PrdSessionError("material questions must not be empty")
    if len(set(normalized)) != len(normalized):
        raise PrdSessionError("material questions must not be duplicated")
    return normalized


def _publish_confirmed_prd(path: Path, body: str, *, root: Path) -> None:
    try:
        publish_repository_text(path, body, root=root, overwrite=False)
    except RepositoryPathError as error:
        raise PrdSessionError(
            f"Borg PRD directory escapes repository: {path.parent}"
        ) from error
    except FileExistsError as error:
        raise FileExistsError(f"confirmed Borg PRD already exists: {path}") from error


def _confirmed_prd_matches(path: Path, body: str, *, root: Path) -> bool:
    try:
        return read_repository_text(path, root=root) == body
    except (OSError, UnicodeError, RepositoryPathError):
        return False


def _is_interruption(
    error: BaseException,
    cancel: CancellationToken | None,
) -> bool:
    if isinstance(error, KeyboardInterrupt) or (
        cancel is not None and cancel.is_set()
    ):
        return True
    cause = error.__cause__
    if cause is None:
        cause = error.__context__
    return isinstance(cause, KeyboardInterrupt)
