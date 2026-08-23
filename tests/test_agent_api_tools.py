"""Containment and grants for provider API agent tools."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from betterborg_cli.agent_runtime import (
    ApiAgentRole,
    ApiToolError,
    CancellationToken,
    ContainedApiTools,
    PathContainmentError,
    ToolGrantError,
)


def test_contained_file_tools_read_list_search_and_patch(tmp_path: Path) -> None:
    source = tmp_path / "src" / "example.py"
    source.parent.mkdir()
    source.write_text("first line\nneedle here\n", encoding="utf-8")
    obsolete = tmp_path / "obsolete.txt"
    obsolete.write_text("remove me\n", encoding="utf-8")
    tools = ContainedApiTools(tmp_path, ApiAgentRole.ANALYSIS)

    assert tools.list_files() == ("obsolete.txt", "src/example.py")
    assert tools.read_file("src/example.py") == "first line\nneedle here\n"
    matches = tools.search_text("needle")
    assert [(match.path, match.line, match.text) for match in matches] == [
        ("src/example.py", 2, "needle here")
    ]

    changed = tools.apply_patch(
        """*** Begin Patch
*** Update File: src/example.py
@@
 first line
-needle here
+replacement
*** Add File: notes.txt
+new file
*** Delete File: obsolete.txt
*** End Patch"""
    )

    assert changed == ("src/example.py", "notes.txt", "obsolete.txt")
    assert source.read_text(encoding="utf-8") == "first line\nreplacement\n"
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "new file\n"
    assert not obsolete.exists()


@pytest.mark.parametrize("path", ["../outside.txt", "nested/../../outside.txt"])
def test_all_file_tools_reject_traversal(tmp_path: Path, path: str) -> None:
    tools = ContainedApiTools(tmp_path, ApiAgentRole.ANALYSIS)

    with pytest.raises(PathContainmentError, match="traversal"):
        tools.list_files(path)
    with pytest.raises(PathContainmentError, match="traversal"):
        tools.search_text("secret", path)
    with pytest.raises(PathContainmentError, match="traversal"):
        tools.read_file(path)
    with pytest.raises(PathContainmentError, match="traversal"):
        tools.apply_patch(
            f"*** Begin Patch\n*** Add File: {path}\n+content\n*** End Patch"
        )


def test_all_file_tools_reject_absolute_paths(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    tools = ContainedApiTools(tmp_path, ApiAgentRole.ANALYSIS)
    absolute = str(outside.resolve())

    with pytest.raises(PathContainmentError, match="absolute"):
        tools.list_files(absolute)
    with pytest.raises(PathContainmentError, match="absolute"):
        tools.search_text("secret", absolute)
    with pytest.raises(PathContainmentError, match="absolute"):
        tools.read_file(absolute)
    with pytest.raises(PathContainmentError, match="absolute"):
        tools.apply_patch(
            "*** Begin Patch\n"
            f"*** Update File: {absolute}\n"
            "@@\n-secret\n+changed\n*** End Patch"
        )

    assert outside.read_text(encoding="utf-8") == "secret\n"


def test_all_file_tools_reject_escaping_symlinks(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    (run_directory / "escape.txt").symlink_to(outside)
    tools = ContainedApiTools(run_directory, ApiAgentRole.ANALYSIS)

    with pytest.raises(PathContainmentError, match="escapes"):
        tools.list_files()
    with pytest.raises(PathContainmentError, match="escapes"):
        tools.search_text("secret")
    with pytest.raises(PathContainmentError, match="escapes"):
        tools.read_file("escape.txt")
    with pytest.raises(PathContainmentError, match="escapes"):
        tools.apply_patch(
            """*** Begin Patch
*** Update File: escape.txt
@@
-secret
+changed
*** End Patch"""
        )

    assert outside.read_text(encoding="utf-8") == "secret\n"


@pytest.mark.parametrize("role", [ApiAgentRole.ANALYSIS, ApiAgentRole.PLANNING])
def test_analysis_and_planning_never_receive_command_runner(
    tmp_path: Path,
    role: ApiAgentRole,
) -> None:
    tools = ContainedApiTools(tmp_path, role, workspace_trusted=True)

    assert "run_command" not in tools.available_tools
    with pytest.raises(ToolGrantError, match="never granted"):
        tools.run_command((sys.executable, "-c", "raise SystemExit(0)"))


@pytest.mark.parametrize(
    "role",
    [ApiAgentRole.CODING, ApiAgentRole.REVIEW, ApiAgentRole.MERGE],
)
def test_execution_roles_receive_command_runner_only_after_trust(
    tmp_path: Path,
    role: ApiAgentRole,
) -> None:
    untrusted = ContainedApiTools(tmp_path, role)
    trusted = ContainedApiTools(tmp_path, role, workspace_trusted=True)

    assert "run_command" not in untrusted.available_tools
    with pytest.raises(ToolGrantError, match="trusted workspace"):
        untrusted.run_command((sys.executable, "-c", "raise SystemExit(0)"))
    assert "run_command" in trusted.available_tools
    assert trusted.run_command(
        (sys.executable, "-c", "print('trusted')")
    ).stdout == "trusted\n"


def test_run_command_treats_shell_metacharacters_as_literal_argv(
    tmp_path: Path,
) -> None:
    tools = ContainedApiTools(
        tmp_path,
        ApiAgentRole.CODING,
        workspace_trusted=True,
    )
    injected = "; touch command-injection"

    result = tools.run_command(
        (sys.executable, "-c", "import sys; print(sys.argv[1])", injected)
    )

    assert result.returncode == 0
    assert result.stdout == f"{injected}\n"
    assert not (tmp_path / "command-injection").exists()


def test_run_command_terminates_when_cancelled(tmp_path: Path) -> None:
    tools = ContainedApiTools(
        tmp_path,
        ApiAgentRole.CODING,
        workspace_trusted=True,
    )
    started = tmp_path / "started"
    finished = tmp_path / "finished"
    cancel = CancellationToken()

    def cancel_when_started() -> None:
        deadline = time.monotonic() + 1
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert started.exists()
        cancel.cancel()

    canceller = threading.Thread(target=cancel_when_started)
    canceller.start()
    result = tools.run_command(
        (
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import time; "
                "Path('started').write_text('yes'); time.sleep(3); "
                "Path('finished').write_text('no')"
            ),
        ),
        cancel=cancel,
    )
    canceller.join()

    assert result.returncode == -1
    assert started.exists()
    assert not finished.exists()


def test_patch_validates_every_path_before_writing(tmp_path: Path) -> None:
    tools = ContainedApiTools(tmp_path, ApiAgentRole.CODING)
    outside = tmp_path.parent / "atomic-outside.txt"

    with pytest.raises(PathContainmentError, match="traversal"):
        tools.apply_patch(
            """*** Begin Patch
*** Add File: would-have-been-created.txt
+content
*** Add File: ../atomic-outside.txt
+escaped
*** End Patch"""
        )

    assert not (tmp_path / "would-have-been-created.txt").exists()
    assert not outside.exists()


def test_patch_rejects_a_nonmatching_hunk_without_modifying_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("actual\n", encoding="utf-8")
    tools = ContainedApiTools(tmp_path, ApiAgentRole.CODING)

    with pytest.raises(ApiToolError, match="does not match"):
        tools.apply_patch(
            """*** Begin Patch
*** Update File: target.txt
@@
-different
+changed
*** End Patch"""
        )

    assert target.read_text(encoding="utf-8") == "actual\n"
