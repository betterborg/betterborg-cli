"""Tracked repository configuration contracts."""

from dataclasses import fields
from pathlib import Path
from uuid import UUID

import pytest

from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_config import (
    AgentChoice,
    AgentChoices,
    AgentStage,
    ExecutionLimits,
    RepositoryConfigError,
    load_repository_config,
)

REPOSITORY_ID = "bd3b21f9-693b-4c58-b7cf-a90417809e1f"


def _write_config(git_repo: Path, content: str) -> RepoPaths:
    paths = RepoPaths.discover(git_repo)
    paths.tracked_dir.mkdir()
    (paths.tracked_dir / "config.toml").write_text(content, encoding="utf-8")
    return paths


def test_loads_typed_repository_identity_and_execution_limits(
    git_repo: Path,
) -> None:
    paths = _write_config(
        git_repo,
        f"""
version = 1

[repository]
id = "{REPOSITORY_ID}"
default_branch = "main"

[execution]
jobs = 10
review_passes = 1
""",
    )

    config = load_repository_config(paths)

    assert config.version == 1
    assert config.repository_id == UUID(REPOSITORY_ID)
    assert config.default_branch == "main"
    assert config.execution == ExecutionLimits(jobs=10, review_passes=1)
    assert config.agents == AgentChoices()


@pytest.mark.parametrize("stage", list(AgentStage))
def test_loads_optional_agent_choices_for_each_stage(
    git_repo: Path, stage: AgentStage
) -> None:
    paths = _write_config(
        git_repo,
        f"""
version = 1

[repository]
id = "{REPOSITORY_ID}"
default_branch = "trunk"

[agents.{stage.value}]
adapter = "codex"
model = "stage-model"
effort = "high"
""",
    )

    config = load_repository_config(paths)

    assert getattr(config.agents, stage.value) == AgentChoice(
        adapter="codex", model="stage-model", effort="high"
    )
    assert config.execution == ExecutionLimits()


def test_loads_agent_defaults(git_repo: Path) -> None:
    paths = _write_config(
        git_repo,
        f"""
version = 1

[repository]
id = "{REPOSITORY_ID}"
default_branch = "trunk"

[agents.defaults]
adapter = "codex"
model = "default-model"
effort = "medium"
""",
    )

    config = load_repository_config(paths)

    assert config.agents.defaults == AgentChoice(
        adapter="codex", model="default-model", effort="medium"
    )


@pytest.mark.parametrize("stage", list(AgentStage))
@pytest.mark.parametrize("setting", ["adapter", "model", "effort"])
def test_resolving_one_stage_setting_does_not_change_other_stages(
    stage: AgentStage, setting: str
) -> None:
    defaults = AgentChoice(
        adapter="default-adapter", model="default-model", effort="medium"
    )
    override = AgentChoice(**{setting: f"stage-{setting}"})
    choices = AgentChoices(defaults=defaults, **{stage.value: override})

    for other_stage in AgentStage:
        resolved = choices.resolve(other_stage)
        expected = (
            AgentChoice(
                adapter=override.adapter or defaults.adapter,
                model=override.model or defaults.model,
                effort=override.effort or defaults.effort,
            )
            if other_stage is stage
            else defaults
        )
        assert resolved == expected


@pytest.mark.parametrize(
    ("override_setting", "expected"),
    [
        (
            'effort = "high"',
            AgentChoice(adapter="codex", model="default-model", effort="high"),
        ),
        (
            'model = "stage-model"',
            AgentChoice(adapter="codex", model="stage-model", effort="medium"),
        ),
    ],
)
def test_stage_overrides_inherit_each_other_field_independently(
    git_repo: Path, override_setting: str, expected: AgentChoice
) -> None:
    paths = _write_config(
        git_repo,
        f"""
version = 1

[repository]
id = "{REPOSITORY_ID}"
default_branch = "trunk"

[agents.defaults]
adapter = "codex"
model = "default-model"
effort = "medium"

[agents.architect]
{override_setting}
""",
    )

    config = load_repository_config(paths)

    assert config.agents.resolve(AgentStage.ARCHITECT) == expected


@pytest.mark.parametrize(
    "choice_name", ["defaults", *(stage.value for stage in AgentStage)]
)
@pytest.mark.parametrize("setting", ["adapter", "model", "effort"])
def test_rejects_empty_agent_choice_values(
    git_repo: Path, choice_name: str, setting: str
) -> None:
    paths = _write_config(
        git_repo,
        f"""
version = 1

[repository]
id = "{REPOSITORY_ID}"
default_branch = "trunk"

[agents.{choice_name}]
{setting} = " "
""",
    )

    with pytest.raises(
        RepositoryConfigError,
        match=rf"agents\.{choice_name}\.{setting} must be a non-empty string",
    ):
        load_repository_config(paths)


def test_agent_choices_fields_match_agent_stage_catalog() -> None:
    stage_fields = {field.name for field in fields(AgentChoices)} - {"defaults"}

    assert stage_fields == {stage.value for stage in AgentStage}


def test_loads_legacy_three_stage_agent_choices(git_repo: Path) -> None:
    paths = _write_config(
        git_repo,
        f"""
version = 1

[repository]
id = "{REPOSITORY_ID}"
default_branch = "trunk"

[agents.coding]
adapter = "codex"
model = "coding-model"
effort = "high"

[agents.review]
adapter = "claude"

[agents.merge]
model = "merge-model"
effort = "low"
""",
    )

    config = load_repository_config(paths)

    assert config.agents.coding == AgentChoice(
        adapter="codex", model="coding-model", effort="high"
    )
    assert config.agents.review == AgentChoice(adapter="claude")
    assert config.agents.merge == AgentChoice(model="merge-model", effort="low")


@pytest.mark.parametrize(
    ("execution", "message"),
    [
        ("jobs = 0", "jobs must be in 1..10"),
        ("jobs = 11", "jobs must be in 1..10"),
        ("review_passes = 0", "review_passes must be at least 1"),
    ],
)
def test_rejects_execution_limits_outside_planned_ranges(
    git_repo: Path, execution: str, message: str
) -> None:
    paths = _write_config(
        git_repo,
        f"""
version = 1
[repository]
id = "{REPOSITORY_ID}"
default_branch = "main"
[execution]
{execution}
""",
    )

    with pytest.raises(RepositoryConfigError, match=message):
        load_repository_config(paths)


@pytest.mark.parametrize(
    ("unsafe_setting", "message"),
    [
        ('api_key = "do-not-track-this"', "secret setting"),
        ('rules_path = "/home/operator/rules"', "absolute machine path"),
        ('rules_path = "C:\\\\Users\\\\operator\\\\rules"', "absolute machine path"),
    ],
)
def test_rejects_secrets_and_absolute_machine_paths_before_schema_validation(
    git_repo: Path, unsafe_setting: str, message: str
) -> None:
    paths = _write_config(
        git_repo,
        f"""
version = 1
[repository]
id = "{REPOSITORY_ID}"
default_branch = "main"
{unsafe_setting}
""",
    )

    with pytest.raises(RepositoryConfigError, match=message):
        load_repository_config(paths)


@pytest.mark.parametrize(
    ("version", "repository_id", "message"),
    [
        (2, REPOSITORY_ID, "unsupported configuration version"),
        (1, "not-a-uuid", "repository.id must be a valid UUID"),
    ],
)
def test_rejects_unsupported_version_and_invalid_repository_identity(
    git_repo: Path, version: int, repository_id: str, message: str
) -> None:
    paths = _write_config(
        git_repo,
        f"""
version = {version}
[repository]
id = "{repository_id}"
default_branch = "main"
""",
    )

    with pytest.raises(RepositoryConfigError, match=message):
        load_repository_config(paths)
