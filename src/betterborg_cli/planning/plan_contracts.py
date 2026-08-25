"""Deterministic validation and portable rendering for Architect plans."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from betterborg_cli.agent_runtime.structured import (
    StructuredResultError,
    validate_structured_result,
)
from betterborg_cli.repo_analysis.text_rendering import (
    markdown_code_span,
    markdown_text,
)


class PlanValidationError(ValueError):
    """Raised when an Architect plan cannot safely enter Tech Lead review."""


_PHASE_NAME = re.compile(r"^[0-9]{2}-[a-z0-9]+(?:-[a-z0-9]+)*$")


def validate_plan_json(rendered_json: str, repository_root: Path) -> dict[str, Any]:
    """Parse and validate one strict JSON rendering of an Architect plan."""
    try:
        plan = json.loads(rendered_json)
    except json.JSONDecodeError as error:
        raise PlanValidationError(f"plan is not valid JSON: {error.msg}") from error
    if not isinstance(plan, dict):
        raise PlanValidationError("plan JSON must contain an object")
    validate_plan(plan, repository_root)
    return plan


def validate_plan(plan: Mapping[str, Any], repository_root: Path) -> None:
    """Validate schema, sequencing, dependencies, paths, and completeness.

    The JSON schema owns field-level shape. These checks deliberately stay
    deterministic and repository-aware so malformed plans fail before a Tech
    Lead agent is invoked.
    """
    # Imported lazily to keep the schema's existing public location while the
    # Architect module consumes this validator.
    from betterborg_cli.planning.architect import ARCHITECT_PLAN_SCHEMA

    try:
        validate_structured_result(plan, ARCHITECT_PLAN_SCHEMA)
    except StructuredResultError as error:
        raise PlanValidationError(f"plan does not match its schema: {error}") from error

    _validate_completeness(plan)

    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise PlanValidationError(f"repository root does not exist: {root}")

    phases = plan["phases"]
    phase_names = [phase["name"] for phase in phases]
    for index, phase in enumerate(phases, start=1):
        name = phase["name"]
        if _PHASE_NAME.fullmatch(name) is None:
            raise PlanValidationError(
                f"phase name {name!r} must use the NN-kebab format"
            )
        expected_prefix = f"{index:02d}-"
        if not name.startswith(expected_prefix):
            raise PlanValidationError(
                f"phase {name!r} is out of sequence; expected number {index:02d}"
            )

        dependencies = phase.get("dependencies_on", [])
        if len(dependencies) != len(set(dependencies)):
            raise PlanValidationError(f"phase {name!r} has duplicate dependencies")
        earlier = set(phase_names[: index - 1])
        for dependency in dependencies:
            if dependency not in earlier:
                raise PlanValidationError(
                    f"phase {name!r} dependency {dependency!r} must name an "
                    "earlier phase"
                )

        seen_paths: set[str] = set()
        for entry in phase["files_touched"]:
            path = entry["path"]
            if path in seen_paths:
                raise PlanValidationError(
                    f"phase {name!r} lists file path {path!r} more than once"
                )
            seen_paths.add(path)
            _validate_planned_path(root, path, entry["role"], phase=name)

    for pointer in plan.get("code_pointers", []):
        _validate_existing_path(root, pointer["path"], field="code pointer")

    if plan.get("open_questions"):
        raise PlanValidationError(
            "plan is incomplete while open_questions remain unanswered"
        )


def _validate_completeness(plan: Mapping[str, Any]) -> None:
    """Reject schema-shaped values that contain no semantic content."""
    for field in ("title", "summary", "overall_approach"):
        _require_nonblank(plan[field], field)
    _require_nonblank_items(plan.get("risks", []), "risks")
    _require_nonblank_items(plan.get("open_questions", []), "open_questions")

    for index, repository in enumerate(plan.get("repositories", [])):
        _require_nonblank(repository["id"], f"repositories[{index}].id")

    for phase_index, phase in enumerate(plan["phases"]):
        prefix = f"phases[{phase_index}]"
        for field in (
            "name",
            "title",
            "goal",
            "technical_approach",
            "test_strategy",
        ):
            _require_nonblank(phase[field], f"{prefix}.{field}")
        for field in (
            "repositories",
            "acceptance_criteria",
            "dependencies_on",
            "deliverables",
            "constraints",
            "risks",
        ):
            _require_nonblank_items(phase.get(field, []), f"{prefix}.{field}")

        for file_index, entry in enumerate(phase["files_touched"]):
            file_prefix = f"{prefix}.files_touched[{file_index}]"
            _require_nonblank(entry["path"], f"{file_prefix}.path")
            if "repo" in entry:
                _require_nonblank(entry["repo"], f"{file_prefix}.repo")
        for contract_index, contract in enumerate(phase.get("contracts", [])):
            contract_prefix = f"{prefix}.contracts[{contract_index}]"
            _require_nonblank(contract["spec"], f"{contract_prefix}.spec")
            if "repo" in contract:
                _require_nonblank(contract["repo"], f"{contract_prefix}.repo")

    for index, pointer in enumerate(plan.get("code_pointers", [])):
        prefix = f"code_pointers[{index}]"
        _require_nonblank(pointer["path"], f"{prefix}.path")
        _require_nonblank(pointer["why"], f"{prefix}.why")


def _require_nonblank(value: str, field: str) -> None:
    if not value.strip():
        raise PlanValidationError(f"plan field {field} must not be blank")


def _require_nonblank_items(values: Sequence[str], field: str) -> None:
    for index, value in enumerate(values):
        _require_nonblank(value, f"{field}[{index}]")


def _validate_planned_path(root: Path, raw_path: str, role: str, *, phase: str) -> None:
    candidate = _repository_path(root, raw_path, field=f"phase {phase!r} file")
    if role == "new":
        if candidate.exists():
            raise PlanValidationError(
                f"phase {phase!r} marks existing path {raw_path!r} as new"
            )
        return
    if not candidate.is_file():
        raise PlanValidationError(
            f"phase {phase!r} {role} path {raw_path!r} is not a repository file"
        )


def _validate_existing_path(root: Path, raw_path: str, *, field: str) -> None:
    candidate = _repository_path(root, raw_path, field=field)
    if not candidate.exists():
        raise PlanValidationError(
            f"{field} {raw_path!r} is not grounded in the repository"
        )


def _repository_path(root: Path, raw_path: str, *, field: str) -> Path:
    if "\\" in raw_path:
        raise PlanValidationError(f"{field} {raw_path!r} must use POSIX separators")
    path = PurePosixPath(raw_path)
    if path.as_posix() != raw_path or path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise PlanValidationError(
            f"{field} {raw_path!r} must be a normalized repository-relative path"
        )
    candidate = root.joinpath(*path.parts).resolve()
    if not candidate.is_relative_to(root):
        raise PlanValidationError(f"{field} {raw_path!r} escapes the repository")
    return candidate


def render_plan_markdown(plan: Mapping[str, Any] | None) -> str:
    """Render a plan as portable GFM, tolerating partial legacy input."""
    if not plan or not isinstance(plan, Mapping):
        return ""

    blocks: list[str] = []

    def add(block: str) -> None:
        if block and block.strip():
            blocks.append(block.rstrip())

    title = _string(plan.get("title"))
    if title:
        add(f"# {markdown_text(title)}")
    summary = _string(plan.get("summary"))
    add(markdown_text(summary) if summary else "")

    approach = _string(plan.get("overall_approach"))
    if approach:
        add("## Overall approach")
        add(markdown_text(approach))

    phases = plan.get("phases")
    if isinstance(phases, list) and phases:
        add("## Phases")
        for index, phase in enumerate(phases, start=1):
            blocks.extend(_phase_blocks(phase, index))

    pointer_lines: list[str] = []
    pointers = plan.get("code_pointers")
    if isinstance(pointers, list):
        for pointer in pointers:
            if not isinstance(pointer, Mapping):
                continue
            path = _string(pointer.get("path"))
            if not path:
                continue
            why = _string(pointer.get("why"))
            label = markdown_code_span(path)
            pointer_lines.append(
                f"- {label} — {markdown_text(why)}" if why else f"- {label}"
            )
    if pointer_lines:
        add("## Code pointers")
        add("\n".join(pointer_lines))

    _add_string_list(add, "## Risks", plan.get("risks"))
    _add_string_list(add, "## Open questions", plan.get("open_questions"))
    return "\n\n".join(blocks).strip() + "\n" if blocks else ""


def _phase_blocks(phase: Any, index: int) -> list[str]:
    if not isinstance(phase, Mapping):
        return []
    blocks: list[str] = []
    name = _string(phase.get("name"))
    title = _string(phase.get("title"))
    heading = (
        " — ".join(markdown_text(item) for item in (name, title) if item)
        or f"Phase {index}"
    )
    blocks.append(f"### {heading}")

    goal = _string(phase.get("goal"))
    if goal:
        blocks.append(f"**Goal:** {markdown_text(goal)}")
    approach = _string(phase.get("technical_approach"))
    if approach:
        blocks.append(f"**Technical approach:** {markdown_text(approach)}")

    file_lines: list[str] = []
    files = phase.get("files_touched")
    if isinstance(files, list):
        for entry in files:
            if not isinstance(entry, Mapping):
                continue
            path = _string(entry.get("path"))
            if not path:
                continue
            label = markdown_code_span(path)
            role = _string(entry.get("role"))
            description = _string(entry.get("description"))
            if role:
                label += f" ({markdown_text(role)})"
            if description:
                label += f" — {markdown_text(description)}"
            file_lines.append(f"- {label}")
    if file_lines:
        blocks.append("**Files touched:**\n" + "\n".join(file_lines))

    contract_lines: list[str] = []
    contracts = phase.get("contracts")
    if isinstance(contracts, list):
        for contract in contracts:
            if not isinstance(contract, Mapping):
                continue
            spec = _string(contract.get("spec"))
            if not spec:
                continue
            kind = _string(contract.get("kind"))
            rendered_spec = markdown_text(spec)
            contract_lines.append(
                f"- _{markdown_text(kind)}_ — {rendered_spec}"
                if kind
                else f"- {rendered_spec}"
            )
    if contract_lines:
        blocks.append("**Contracts:**\n" + "\n".join(contract_lines))

    strategy = _string(phase.get("test_strategy"))
    if strategy:
        blocks.append(f"**Test strategy:** {markdown_text(strategy)}")
    for label, key in (
        ("Acceptance criteria", "acceptance_criteria"),
        ("Deliverables", "deliverables"),
        ("Dependencies", "dependencies_on"),
        ("Constraints", "constraints"),
        ("Risks", "risks"),
    ):
        items = _nonempty_strings(phase.get(key))
        if items:
            blocks.append(
                f"**{label}:**\n"
                + "\n".join(f"- {markdown_text(item)}" for item in items)
            )
    return blocks


def _add_string_list(add, heading: str, value: Any) -> None:
    items = _nonempty_strings(value)
    if items:
        add(heading)
        add("\n".join(f"- {markdown_text(item)}" for item in items))


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _nonempty_strings(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


__all__ = [
    "PlanValidationError",
    "render_plan_markdown",
    "validate_plan",
    "validate_plan_json",
]
