"""Safe text rendering shared by human-readable analysis documents."""

from __future__ import annotations

import re
import unicodedata

_MARKDOWN_SPECIALS = frozenset(r"\`*_{}[]<>#+!|")
_BACKTICK_RUN = re.compile(r"`+")


def terminal_text(value: object) -> str:
    """Flatten text and remove characters that can control a terminal."""
    cleaned: list[str] = []
    for character in str(value):
        if character.isspace():
            cleaned.append(" ")
        elif not unicodedata.category(character).startswith("C"):
            cleaned.append(character)
    return " ".join("".join(cleaned).split())


def markdown_text(value: object) -> str:
    """Render analyzer-controlled text as one escaped Markdown fragment."""
    return "".join(
        f"\\{character}" if character in _MARKDOWN_SPECIALS else character
        for character in terminal_text(value)
    )


def markdown_code_span(value: object, *, table_cell: bool = False) -> str:
    """Render flattened, control-free text as one Markdown code span."""
    text = terminal_text(value)
    longest_run = max(
        (len(match.group()) for match in _BACKTICK_RUN.finditer(text)),
        default=0,
    )
    delimiter = "`" * (longest_run + 1)
    if table_cell:
        text = text.replace("|", r"\|")
    padding = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{delimiter}{padding}{text}{padding}{delimiter}"
