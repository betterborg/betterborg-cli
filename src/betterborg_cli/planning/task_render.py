"""Canonical Markdown rendering for immutable task generations."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

_SECTIONS = (
    ("Why", "why", False),
    ("Scope", "scope", True),
    ("Implementation Notes", "implementation_notes", True),
    ("Acceptance Criteria", "acceptance_criteria", True),
    ("Tests", "tests", True),
    ("Dependencies", "dependencies", True),
    ("Out of Scope", "out_of_scope", True),
)
_EMPTY = "(none)"


def render_task_markdown(task: Mapping[str, Any]) -> str:
    """Render one structured task in a byte-stable, inspectable format."""
    title = _text(task.get("title"))
    if not title:
        stage = _text(task.get("stage"))
        stem = _text(task.get("stem"))
        title = f"{stage}/{stem}" if stage or stem else "Untitled task"

    lines = [f"# {title}", ""]
    for heading, key, is_list in _SECTIONS:
        lines.append(f"## {heading}")
        if is_list:
            items = _items(task.get(key))
            lines.extend((f"- {item}" for item in items) if items else [_EMPTY])
        else:
            lines.append(_text(task.get(key)) or _EMPTY)
        lines.append("")
    while lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def task_markdown_digest(body: str | bytes) -> str:
    """Return the digest persisted beside rendered task Markdown."""
    encoded = body.encode("utf-8") if isinstance(body, str) else body
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int | float | bool):
        return str(value).strip()
    return ""


def _items(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := _text(item))]


__all__ = ["render_task_markdown", "task_markdown_digest"]
