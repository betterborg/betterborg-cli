"""The shared card layer and its terminal presentation."""

from __future__ import annotations

from io import StringIO

import click
from click.testing import CliRunner
from rich.console import Console

from betterborg_cli import terminal_prompts
from betterborg_cli.prd_session import InteractiveIO, PromptCard


def _plain(renderable) -> str:
    console = Console(file=StringIO(), width=80, record=True, force_terminal=False)
    console.print(renderable)
    return console.export_text()


def test_card_without_presenter_flattens_to_the_old_lines() -> None:
    prompts: list[str] = []
    written: list[str] = []
    io = InteractiveIO(
        prompt=lambda message: prompts.append(message) or "Linux",
        confirm=lambda _message, _default: False,
        write=written.append,
    )
    card = PromptCard(
        title="Architect question 1 of 1",
        question="Which platforms?",
        details=(
            ("Why this matters", "Test matrix."),
            ("Answer guidance", "OS names."),
        ),
        footer="round 1 of 3",
    )

    assert io.ask(card) == "Linux"
    assert prompts == ["Which platforms?"]
    assert written == ["Why this matters: Test matrix.", "Answer guidance: OS names."]


def test_card_with_presenter_prompts_for_the_answer_label() -> None:
    prompts: list[str] = []
    presented: list[PromptCard] = []
    io = InteractiveIO(
        prompt=lambda message: prompts.append(message) or "note",
        confirm=lambda _message, _default: False,
        write=lambda _message: None,
        present=presented.append,
    )
    card = PromptCard(
        title="Plan change", question="What changes?", answer_label="Note"
    )

    assert io.ask(card) == "note"
    assert presented == [card]
    assert prompts == ["Note"]


def test_render_card_frames_question_details_and_footer() -> None:
    text = _plain(
        terminal_prompts.render_card(
            PromptCard(
                title="Architect question 2 of 5",
                question="Should invitees see drafts?",
                details=(("Why this matters", "Decides visibility."),),
                footer="round 1 of 3 · borg 'invite'",
            )
        )
    )

    lines = text.splitlines()
    assert "Architect question 2 of 5" in lines[0]
    assert "Should invitees see drafts?" in text
    assert "Why this matters  Decides visibility." in text
    assert "round 1 of 3 · borg 'invite'" in lines[-1]


def test_render_card_renders_markdown_documents() -> None:
    text = _plain(
        terminal_prompts.render_card(
            PromptCard(title="PRD draft", markdown="# Title\n\n- one\n- two\n")
        )
    )

    assert "PRD draft" in text
    assert "Title" in text
    assert "• one" in text


def test_render_execution_estimate_lists_every_figure() -> None:
    estimate = {
        "generation_id": "gen-1",
        "task_mix": {"small": 1, "medium": 2, "large": 0, "unsized": 0},
        "time": {"p50": 1800.0, "p80": 3600.0, "unknown_tasks": 1},
        "sample_size": 4,
        "per_complexity": [
            {
                "complexity": "small",
                "task_count": 1,
                "sample_size": 4,
                "source": "local",
                "time": {"p50": 600.0, "p80": 900.0},
            }
        ],
        "billing": {
            "api": {"unknown": False, "estimate": {"p50": 1.5, "p80": 2.25}},
            "subscription": {"included": True, "phases": ["coding"]},
            "unknown_phases": ["merge"],
        },
    }

    text = _plain(terminal_prompts.render_execution_estimate("demo", estimate))

    assert "Execution estimate" in text
    assert "DUMMY DATA" in text
    assert "gen-1" in text
    assert "1 small · 2 medium · 0 large · 0 unsized" in text
    assert "P50 30.0m   P80 1.0h  (not calendar time)" in text
    assert "Unknown time" in text and "1 task(s)" in text
    assert "4 local completion(s)" in text
    assert "local" in text and "10.0m" in text and "15.0m" in text
    assert "P50 $1.5000   P80 $2.2500 USD" in text
    assert "work included for coding" in text
    assert "Billing unknown" in text and "merge" in text
    assert "borg 'demo'" in text


def test_prompts_share_the_marker_and_trust_shows_its_warning() -> None:
    seen: dict[str, object] = {}

    @click.command()
    def command() -> None:
        seen["trust"] = terminal_prompts.confirm_workspace_trust(
            "Trust workspace /repo/demo? Agents may read and modify files."
        )
        seen["answer"] = terminal_prompts.prompt("Answer")

    result = CliRunner().invoke(command, input="y\nLinux\n")

    assert result.exit_code == 0, result.output
    assert seen == {"trust": True, "answer": "Linux"}
    assert "Trust this workspace?" in result.output
    assert "Workspace  /repo/demo" in result.output
    assert "Agents may read and modify files." in result.output
    assert "Trust workspace /repo/demo?" not in result.output
    assert "› Trust this workspace [y/N]: " in result.output
    assert "› Answer: " in result.output
