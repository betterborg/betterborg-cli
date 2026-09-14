"""Terminal presentation for the decisions betterborg puts to its operator.

Every prompt the CLI shows a person goes through here so they share one
look: a bordered card that frames the decision, then one input line marked
with ``›``. Colours follow the progress display: cyan for work that wants
attention, dim for supporting detail, green for an approval step and red for
a warning. ``NO_COLOR`` and non-terminal output degrade to plain text.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

import click
from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from betterborg_cli.execution_estimate import DUMMY_PRIOR_LABEL
from betterborg_cli.prd_session import InteractiveIO, PromptCard
from betterborg_cli.workspace_trust import TRUST_PROMPT_PREFIX

PROMPT_MARKER = "› "
CARD_WIDTH = 80

_TONE_STYLES: Mapping[str, str] = {
    "question": "cyan",
    "review": "cyan",
    "approve": "green",
    "warning": "red",
}


def console() -> Console:
    """Return a console on the current stdout, plain when it is not a terminal."""
    return Console(
        file=click.get_text_stream("stdout"),
        highlight=False,
        no_color="NO_COLOR" in os.environ,
    )


def styled_prompt(message: str) -> str:
    """Prefix one input line with the marker every prompt shares."""
    return click.style(PROMPT_MARKER, fg="cyan", bold=True) + click.style(
        message, bold=True
    )


def prompt(message: str) -> str | None:
    """Read one free-text answer, or ``None`` when the user aborts."""
    try:
        return click.prompt(styled_prompt(message), default="", show_default=False)
    except click.Abort:
        return None


def confirm(message: str, default: bool = False) -> bool:
    """Ask one yes/no question with the shared marker."""
    return click.confirm(styled_prompt(message), default=default)


def present(card: PromptCard) -> None:
    """Draw one card as a bordered panel."""
    console().print(render_card(card))


def interactive_io() -> InteractiveIO:
    """Build the terminal-backed prompt boundary the CLI hands to workflows."""
    return InteractiveIO(
        prompt=prompt,
        confirm=confirm,
        write=click.echo,
        present=present,
    )


def render_card(card: PromptCard) -> Panel:
    """Lay a card out as the panel ``present`` draws."""
    colour = _TONE_STYLES.get(card.tone, "cyan")
    parts: list[RenderableType] = []
    if card.question:
        parts.append(Text(card.question, style="bold"))
    if card.details:
        if parts:
            parts.append(Text(""))
        parts.append(_detail_rows(card.details))
    if card.markdown is not None:
        if parts:
            parts.append(Text(""))
        parts.append(Markdown(card.markdown))
    return _panel(
        Group(*parts) if parts else Text(""),
        title=card.title,
        footer=card.footer,
        colour=colour,
    )


def render_execution_estimate(name: str, estimate: Mapping[str, object]) -> Panel:
    """Lay the execution estimate out as the card the execution gate shows."""
    mix = estimate["task_mix"]
    time = estimate["time"]
    rows = Table.grid(padding=(0, 3))
    rows.add_column(style="dim", no_wrap=True)
    rows.add_column()
    rows.add_row("Generation", str(estimate["generation_id"]))
    rows.add_row(
        "Task mix",
        f"{mix['small']} small · {mix['medium']} medium · "
        f"{mix['large']} large · {mix['unsized']} unsized",
    )
    rows.add_row(
        "Agent work",
        Text.assemble(
            ("P50 ", "dim"),
            (format_duration(time["p50"]), "bold"),
            ("   P80 ", "dim"),
            (format_duration(time["p80"]), "bold"),
            ("  (not calendar time)", "dim"),
        ),
    )
    if time["unknown_tasks"]:
        rows.add_row("Unknown time", f"{time['unknown_tasks']} task(s)")
    rows.add_row("Sample", f"{estimate['sample_size']} local completion(s)")

    per = Table(box=None, padding=(0, 2), header_style="dim")
    for heading, justify in (
        ("complexity", "left"),
        ("tasks", "right"),
        ("n", "right"),
        ("source", "left"),
        ("P50", "right"),
        ("P80", "right"),
    ):
        per.add_column(heading, justify=justify)
    for item in estimate["per_complexity"]:
        per.add_row(
            str(item["complexity"]),
            str(item["task_count"]),
            str(item["sample_size"]),
            str(item["source"]),
            format_duration(item["time"]["p50"]),
            format_duration(item["time"]["p80"]),
        )

    billing = estimate["billing"]
    api = billing["api"]
    if api["unknown"]:
        api_line = "unknown (billing, usage, or model price is missing)"
    elif api["estimate"] is None:
        api_line = "not used"
    else:
        api_line = (
            f"P50 ${api['estimate']['p50']:.4f}   P80 ${api['estimate']['p80']:.4f} USD"
        )
    money = Table.grid(padding=(0, 3))
    money.add_column(style="dim", no_wrap=True)
    money.add_column()
    money.add_row("API estimate", api_line)
    subscription = billing["subscription"]
    if subscription["included"]:
        money.add_row(
            "Subscription",
            f"work included for {', '.join(subscription['phases'])}; "
            "USD: unknown/not applicable",
        )
    if billing["unknown_phases"]:
        money.add_row("Billing unknown", ", ".join(billing["unknown_phases"]))

    body = Group(
        Text(DUMMY_PRIOR_LABEL, style="red"),
        Text(""),
        rows,
        Text(""),
        per,
        Text(""),
        money,
    )
    return _panel(
        body, title="Execution estimate", footer=f"borg {name!r}", colour="green"
    )


def format_duration(value: object) -> str:
    """Render seconds compactly: ``45s``, ``12.5m``, ``2.3h``."""
    if value is None:
        return "unknown duration"
    seconds = float(value)
    if seconds < 60:
        return f"{seconds:g}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def confirm_workspace_trust(consequence: str) -> bool:
    """Warn about host access before asking whether to trust a workspace.

    The consequence arrives as one sentence so transcripts and MCP prompts
    read naturally; here it is split back into the workspace and the warning.
    """
    details: tuple[tuple[str, str], ...] = ()
    question = consequence
    head, separator, warning = consequence.partition("? ")
    if separator and head.startswith(TRUST_PROMPT_PREFIX):
        details = (("Workspace", head[len(TRUST_PROMPT_PREFIX) :]),)
        question = warning
    present(
        PromptCard(
            title="Trust this workspace?",
            question=question,
            details=details,
            tone="warning",
        )
    )
    return confirm("Trust this workspace", default=False)


def _detail_rows(details: tuple[tuple[str, str], ...]) -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim cyan", no_wrap=True)
    grid.add_column(style="dim")
    for label, text in details:
        grid.add_row(label, text)
    return grid


def _panel(
    body: RenderableType, *, title: str, footer: str | None, colour: str
) -> Panel:
    return Panel(
        body,
        title=Text(title, style=f"bold {colour}"),
        title_align="left",
        subtitle=Text(footer, style="dim") if footer else None,
        subtitle_align="right",
        border_style=colour,
        padding=(1, 2),
        width=min(console().width, CARD_WIDTH),
    )
