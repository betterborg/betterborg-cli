"""Tracked repository configuration contracts."""

from pathlib import Path
from uuid import UUID

import pytest

from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_config import (
    AgentChoice,
    AgentChoices,
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


def test_loads_optional_agent_choices(git_repo: Path) -> None:
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
    assert config.execution == ExecutionLimits()


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
