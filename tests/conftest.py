"""Shared fixtures for BetterBorg CLI tests."""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import pytest
from click.testing import CliRunner
from pytest import MonkeyPatch

from betterborg_cli import cli as cli_module
from betterborg_cli.agent_runtime.mock import MockAdapter
from betterborg_cli.planning import (
    approved_plan_digest,
    render_task_markdown,
    task_markdown_digest,
)
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.progress import AgentActivity
from betterborg_cli.repo_analysis import DIMENSIONS
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.store import (
    Borg,
    PlanApproval,
    PlanningAttempt,
    PlanningAttemptStatus,
    PrdSession,
    Repository,
    RepositoryAnalysis,
    RepositoryPackage,
    SqliteStore,
    TaskBatch,
    TaskComplexity,
    TaskGeneration,
    TaskRecord,
)


@dataclass(frozen=True)
class TaskGenerationFixture:
    """A persisted generation and its sole task record."""

    generation: TaskGeneration
    task: TaskRecord


@dataclass
class RecordingProgress:
    """Record provider-neutral parent and child progress updates for tests."""

    updates: list[tuple[str, str | None]] = field(default_factory=list)
    activities: list[tuple[str, AgentActivity]] = field(default_factory=list)
    child_updates: list[tuple[str, str, str | None]] = field(default_factory=list)
    child_activities: list[tuple[str, str, AgentActivity]] = field(
        default_factory=list
    )

    def update(self, stage_key: str, detail: str | None) -> None:
        self.updates.append((stage_key, detail))

    def activity(self, stage_key: str, activity: AgentActivity) -> None:
        self.activities.append((stage_key, activity))

    def update_child(
        self, stage_key: str, child_key: str, detail: str | None
    ) -> None:
        self.child_updates.append((stage_key, child_key, detail))

    def child_activity(
        self, stage_key: str, child_key: str, activity: AgentActivity
    ) -> None:
        self.child_activities.append((stage_key, child_key, activity))


@dataclass
class RealProcessHarness:
    """Own real-process markers, gates, signals, deadlines, and cleanup."""

    root: Path
    deadline_seconds: float = 5.0
    processes: list[subprocess.Popen[str]] = field(default_factory=list)

    def launch_python(
        self,
        source: str,
        *arguments: str,
        name: str = "wrapper",
    ) -> subprocess.Popen[str]:
        """Launch a generated Python wrapper in a separately killable session."""
        script = self.root / f"{name}.py"
        script.write_text(source, encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, str(script), *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        self.processes.append(process)
        return process

    def resistant_argv(
        self, name: str, *, leader_exits: bool = False
    ) -> tuple[str, ...]:
        """Return argv for a SIGTERM-resistant process group with a descendant."""
        helper = self.root / "resistant_descendant.py"
        if not helper.exists():
            helper.write_text(
                """from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
name = sys.argv[2]
mode = sys.argv[3]
signal.signal(signal.SIGTERM, signal.SIG_IGN)
if mode == "child":
    (root / f"{name}.child.pid").write_text(str(__import__("os").getpid()))
    while True:
        time.sleep(1)
child = subprocess.Popen([sys.executable, __file__, str(root), name, "child"])
(root / f"{name}.parent.pid").write_text(str(__import__("os").getpid()))
while not (root / f"{name}.child.pid").exists():
    time.sleep(0.01)
print("process-tree-ready", flush=True)
if mode == "exit":
    raise SystemExit(0)
child.wait()
""",
                encoding="utf-8",
            )
        mode = "exit" if leader_exits else "parent"
        return (sys.executable, str(helper), str(self.root), name, mode)

    def launch_streamed_registration_wrapper(
        self,
        command: tuple[str, ...],
        *,
        name: str,
        fail_registration: bool = False,
    ) -> subprocess.Popen[str]:
        """Run ``run_streamed`` behind a post-creation registration gate."""
        return self._launch_registration_wrapper(
            command,
            name=name,
            runner="streamed",
            fail_registration=fail_registration,
        )

    def launch_captured_registration_wrapper(
        self,
        command: tuple[str, ...],
        *,
        name: str,
        fail_registration: bool = False,
    ) -> subprocess.Popen[str]:
        """Run ``run_captured`` behind a post-creation registration gate."""
        return self._launch_registration_wrapper(
            command,
            name=name,
            runner="captured",
            fail_registration=fail_registration,
        )

    def launch_url_registration_wrapper(
        self,
        url: str,
        *,
        name: str,
        fail_registration: bool = False,
    ) -> subprocess.Popen[str]:
        """Run a URL request behind a post-start registration gate."""
        source = r'''
from __future__ import annotations

import sys
import time
from pathlib import Path

from betterborg_cli.agent_runtime import (
    CancellationRegistrationWindow,
    CancellationToken,
    MultiprocessUrlRequest,
    UrlRequestSpec,
)
from betterborg_cli.run_control import RunControl


def main() -> None:
    root = Path(sys.argv[1])
    name = sys.argv[2]
    fail_registration = sys.argv[3] == "fail"
    url = sys.argv[4]
    cancel = CancellationToken()
    original_register = CancellationRegistrationWindow.register

    def gated_register(self, *args, **kwargs):
        target = kwargs["force_target"]
        (root / f"{name}.request.pid").write_text(str(target.identity))
        (root / f"{name}.registration-gate").write_text("blocked")
        while not (root / f"release-{name}").exists():
            time.sleep(0.01)
        if fail_registration:
            raise RuntimeError("injected registration failure")
        return original_register(self, *args, **kwargs)

    CancellationRegistrationWindow.register = gated_register

    class Progress:
        def begin_cancellation(self):
            (root / f"{name}.cancelled").write_text("cancelled")
            return True

    try:
        request = MultiprocessUrlRequest(
            UrlRequestSpec(
                url,
                "GET",
                {"authorization": "Bearer request-private-value"},
                None,
            ),
            cancel,
        )
        if fail_registration:
            request.run()
        else:
            control = RunControl(cancel, progress=Progress())
            with control:
                with control.protected():
                    request.run()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except RuntimeError as error:
        (root / f"{name}.error").write_text(str(error))
        (root / f"{name}.active-windows").write_text(
            str(len(cancel.active_windows))
        )
        raise SystemExit(73) from None


if __name__ == "__main__":
    main()
'''
        mode = "fail" if fail_registration else "signal"
        return self.launch_python(
            source,
            str(self.root),
            name,
            mode,
            url,
            name=f"{name}-wrapper",
        )

    def _launch_registration_wrapper(
        self,
        command: tuple[str, ...],
        *,
        name: str,
        runner: str,
        fail_registration: bool,
    ) -> subprocess.Popen[str]:
        source = r'''
from __future__ import annotations

import sys
import time
from pathlib import Path

from betterborg_cli.agent_runtime import (
    CancellationRegistrationWindow,
    CancellationToken,
    run_captured,
    run_streamed,
)
from betterborg_cli.run_control import RunControl

root = Path(sys.argv[1])
name = sys.argv[2]
fail_registration = sys.argv[3] == "fail"
runner = sys.argv[4]
command = sys.argv[5:]
cancel = CancellationToken()
original_register = CancellationRegistrationWindow.register

def gated_register(self, *args, **kwargs):
    (root / f"{name}.registration-gate").write_text("blocked")
    while not (root / f"release-{name}").exists():
        time.sleep(0.01)
    if fail_registration:
        raise RuntimeError("injected registration failure")
    return original_register(self, *args, **kwargs)

CancellationRegistrationWindow.register = gated_register

class Progress:
    def begin_cancellation(self):
        (root / f"{name}.cancelled").write_text("cancelled")
        return True

try:
    if fail_registration:
        if runner == "streamed":
            run_streamed(command, root, "", root / f"{name}.log", cancel)
        else:
            run_captured(command, cwd=root, input="", cancel=cancel)
    else:
        control = RunControl(cancel, progress=Progress())
        with control:
            with control.protected():
                if runner == "streamed":
                    run_streamed(
                        command, root, "", root / f"{name}.log", cancel
                    )
                else:
                    run_captured(command, cwd=root, input="", cancel=cancel)
except KeyboardInterrupt:
    raise SystemExit(130) from None
except RuntimeError as error:
    (root / f"{name}.error").write_text(str(error))
    (root / f"{name}.active-windows").write_text(
        str(len(cancel.active_windows))
    )
    raise SystemExit(73) from None
'''
        mode = "fail" if fail_registration else "signal"
        return self.launch_python(
            source,
            str(self.root),
            name,
            mode,
            runner,
            *command,
            name=f"{name}-wrapper",
        )

    def marker(self, name: str) -> Path:
        return self.root / name

    def wait_for_marker(
        self, name: str, *, timeout: float | None = None
    ) -> str:
        """Wait for a nonempty marker or raise with process-state diagnostics."""
        marker = self.marker(name)
        deadline = time.monotonic() + (
            self.deadline_seconds if timeout is None else timeout
        )
        while time.monotonic() < deadline:
            if marker.exists():
                value = marker.read_text(encoding="utf-8").strip()
                if value:
                    return value
            time.sleep(0.01)
        states = ", ".join(
            f"pid={process.pid}:returncode={process.poll()}"
            for process in self.processes
        ) or "no tracked processes"
        existing = ", ".join(path.name for path in sorted(self.root.iterdir()))
        raise TimeoutError(
            f"deadline waiting for marker {name!r}; {states}; markers=[{existing}]"
        )

    def release(self, name: str) -> None:
        """Open a named file gate."""
        self.marker(name).write_text("released\n", encoding="utf-8")

    @staticmethod
    def signal(process: subprocess.Popen[str], signum: int) -> None:
        """Deliver a signal to a tracked wrapper process."""
        os.kill(process.pid, signum)

    def wait_for_exit(
        self, process: subprocess.Popen[str], *, timeout: float | None = None
    ) -> int:
        """Wait for wrapper exit with captured-output diagnostics."""
        duration = self.deadline_seconds if timeout is None else timeout
        try:
            return process.wait(timeout=duration)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(
                f"deadline waiting for pid {process.pid}; "
                f"returncode={process.poll()}; "
                f"markers={[path.name for path in sorted(self.root.iterdir())]}"
            ) from error

    def assert_tree_absent(
        self, name: str, *, timeout: float | None = None
    ) -> None:
        """Assert a named resistant process group and descendant are both gone."""
        pids = (
            int(self.wait_for_marker(f"{name}.parent.pid")),
            int(self.wait_for_marker(f"{name}.child.pid")),
        )
        deadline = time.monotonic() + (
            self.deadline_seconds if timeout is None else timeout
        )
        while time.monotonic() < deadline:
            if all(not self._pid_exists(pid) for pid in pids):
                return
            time.sleep(0.01)
        surviving = [pid for pid in pids if self._pid_exists(pid)]
        raise AssertionError(f"process tree {name!r} survived: {surviving}")

    def assert_pid_absent(
        self, pid: int, *, timeout: float | None = None
    ) -> None:
        """Assert one marked process identity is absent before a deadline."""
        deadline = time.monotonic() + (
            self.deadline_seconds if timeout is None else timeout
        )
        while time.monotonic() < deadline:
            if not self._pid_exists(pid):
                return
            time.sleep(0.01)
        raise AssertionError(f"process {pid} survived")

    def cleanup(self) -> None:
        """Kill all tracked wrappers and every marked resistant process group."""
        for marker in self.root.glob("*.parent.pid"):
            with contextlib.suppress(ValueError, ProcessLookupError, PermissionError):
                os.killpg(int(marker.read_text(encoding="utf-8")), signal.SIGKILL)
        for process in self.processes:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(process.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=1)

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

        proc_root = Path("/proc")
        if not proc_root.is_dir():
            return True
        process_stat = proc_root / str(pid) / "stat"
        try:
            fields = process_stat.read_text(encoding="utf-8").split()
        except FileNotFoundError:
            return False
        return len(fields) < 3 or fields[2] != "Z"


@pytest.fixture
def cli_runner() -> CliRunner:
    """Return Click's isolated command-line test runner."""
    return CliRunner()


@pytest.fixture
def recording_progress() -> RecordingProgress:
    """Return a progress recorder that understands only shared activity types."""
    return RecordingProgress()


@pytest.fixture
def real_process_harness(tmp_path: Path) -> Iterator[RealProcessHarness]:
    """Provide the project-wide real-process cancellation harness."""
    harness = RealProcessHarness(tmp_path)
    try:
        yield harness
    finally:
        harness.cleanup()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create an initialized temporary Git repository."""
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "BetterBorg Tests"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "config",
            "user.email",
            "tests@betterborg.dev",
        ],
        check=True,
    )
    return tmp_path


@pytest.fixture
def committed_git_repo(git_repo: Path) -> Path:
    """Create a temporary Git repository with one tracked commit."""
    (git_repo / "README.md").write_text(
        "# Test repository\n\nBuild and test docs.\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(git_repo), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(git_repo), "commit", "--quiet", "-m", "initial"],
        check=True,
    )
    return git_repo


@pytest.fixture
def sqlite_db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a temporary SQLite database and close it after the test."""
    connection = sqlite3.connect(tmp_path / "test.sqlite3")
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def write_repository_config():
    """Return the shared repository-config writer for planning tests."""
    return _write_repository_config


@pytest.fixture
def persist_repository_analysis():
    """Return the shared analysis persistence helper for planning tests."""
    return _persist_repository_analysis


@pytest.fixture
def persist_planning_context():
    """Return the shared complete planning-context factory."""
    return _persist_planning_context


@pytest.fixture
def planning_cli_repository(persist_planning_context):
    """Return a factory for a planning repository with CLI-visible state."""

    def create(root: Path, name: str):
        paths = RepoPaths.discover(root)
        fixture_database = root.parent / f"{name}.sqlite3"
        with SqliteStore.open(fixture_database) as store:
            repository, _borg = persist_planning_context(root, store, name)
        paths.state_dir.mkdir(parents=True)
        shutil.copyfile(fixture_database, paths.state_dir / "borg.sqlite3")
        return repository, paths

    return create


@pytest.fixture
def approved_task_generation():
    """Return the shared factory for one approved persisted task generation."""
    return _add_approved_task_generation


@pytest.fixture
def planning_plan_response():
    """Build a valid plan response shared by planning lifecycle tests."""
    return _planning_plan_response


@pytest.fixture
def tech_lead_approval_response():
    """Build an approving Tech Lead response."""
    return _tech_lead_approval_response


@pytest.fixture
def tech_lead_change_request_response():
    """Build a Tech Lead response that requests a plan revision."""
    return _tech_lead_change_request_response


@pytest.fixture
def configure_interactive_cli(monkeypatch: MonkeyPatch):
    """Return the shared interactive CLI dependency configurator."""

    def configure(
        root: Path,
        adapter: MockAdapter,
        io: InteractiveIO,
        *,
        state_home: Path,
    ) -> None:
        monkeypatch.chdir(root)
        monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
        monkeypatch.setattr(cli_module, "_stdin_is_interactive", lambda: True)
        monkeypatch.setattr(
            cli_module, "select_agent", lambda *_args, **_kwargs: adapter
        )
        monkeypatch.setattr(cli_module, "_interactive_io", lambda: io)

    return configure


def _write_repository_config(root: Path, repository: Repository) -> None:
    (root / ".borg").mkdir()
    (root / ".borg/config.toml").write_text(
        "version = 1\n\n"
        "[repository]\n"
        f'id = "{repository.id}"\n'
        'default_branch = "main"\n',
        encoding="utf-8",
    )


def _add_approved_task_generation(
    store: SqliteStore,
    borg: Borg,
    approval: PlanApproval,
    *,
    body: dict,
    round_number: int,
    task_ref: str | None = None,
) -> TaskGenerationFixture:
    attempt = PlanningAttempt(
        borg_id=borg.id,
        phase="supervisor_review",
        round=round_number,
        adapter="mock",
        model="test-model",
    )
    batch = TaskBatch(
        borg_id=borg.id,
        plan_approval_id=approval.id,
        attempt_id=attempt.id,
        round=round_number,
        digest=f"sha256:batch-{round_number}",
        manifest={},
    )
    generation_id = uuid4()
    digest = task_markdown_digest(render_task_markdown(body))
    task = TaskRecord(
        generation_id=generation_id,
        borg_id=borg.id,
        task_ref=task_ref or f"T-{generation_id.hex}",
        stage=body["stage"],
        stem=body["stem"],
        position=1,
        title=body["title"],
        complexity=TaskComplexity(body["estimate_complexity"]),
        digest=digest,
        task=body,
        manifest={"approved_plan_digest": approval.plan_digest, "task.md": digest},
    )
    relative_path = (
        f".borg/tasks/{borg.name}/{generation_id}/{task.stage}/{task.stem}.md"
    )
    manifest = {
        "approved_plan_digest": approval.plan_digest,
        "batch_digest": batch.digest,
        "dependencies": [],
        "plan_approval_id": str(approval.id),
        "tasks": [
            {
                "digest": digest,
                "path": relative_path,
                "position": task.position,
                "task_ref": task.task_ref,
            }
        ],
    }
    generation = TaskGeneration(
        id=generation_id,
        borg_id=borg.id,
        plan_approval_id=approval.id,
        batch_id=batch.id,
        digest=approved_plan_digest(manifest),
        manifest=manifest,
    )
    attempt = PlanningAttempt(
        id=attempt.id,
        borg_id=attempt.borg_id,
        phase=attempt.phase,
        round=attempt.round,
        adapter=attempt.adapter,
        model=attempt.model,
        request={
            "batch_id": str(batch.id),
            "generation_id": str(generation.id),
        },
    )
    store.append_planning_attempt(attempt)
    store.append_task_batch(batch)
    store.add_task_generation(generation, [task])
    store.complete_planning_attempt(
        attempt.id,
        status=PlanningAttemptStatus.COMPLETED,
        result={"decision": "approve", "summary": "Ready.", "findings": []},
        summary="Ready.",
    )
    return TaskGenerationFixture(generation=generation, task=task)


def _persist_repository_analysis(
    store: SqliteStore, repository: Repository
) -> tuple[RepositoryAnalysis, list[RepositoryPackage]]:
    head_sha = subprocess.run(
        ["git", "-C", str(repository.root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    analysis = RepositoryAnalysis(
        repository_id=repository.id,
        head_sha=head_sha,
        summary="A compact test repository.",
        primary_language="Python",
        is_monorepo=False,
        overall_score=4.0,
        analysis_json={
            "packages": [{"path": "."}],
            "themes": [],
            "command_catalog": {"commands": []},
            "environment": {"files": []},
            "required_secrets": [],
            "service_dependencies": [],
        },
    )
    package = RepositoryPackage(
        repository_id=repository.id,
        analysis_id=analysis.id,
        package_path=".",
        package_name="test-repository",
        primary_language="Python",
        rubric={dimension: {"score": 4} for dimension in DIMENSIONS},
        overall_score=4.0,
    )
    packages = [package]
    store.append_analysis(analysis, packages)
    return analysis, packages


def _persist_planning_context(
    root: Path, store: SqliteStore, name: str
) -> tuple[Repository, Borg]:
    repository = Repository(root=root)
    borg = Borg(repository_id=repository.id, name=name)
    _write_repository_config(root, repository)
    prd_path = Path(".borg/prds") / f"{name}.md"
    (root / prd_path).parent.mkdir(parents=True)
    (root / prd_path).write_text(f"# {name}\n", encoding="utf-8")
    store.add_repository(repository)
    store.add_borg(borg)
    store.add_prd_session(
        PrdSession(
            repository_id=repository.id,
            borg_id=borg.id,
            prd_path=prd_path,
        )
    )
    _persist_repository_analysis(store, repository)
    return repository, borg


def _planning_plan_response(
    *, summary: str = "Add a small, tested release workflow."
) -> dict:
    return {
        "title": "Release workflow",
        "summary": summary,
        "overall_approach": (
            "Extend the existing repository conventions and verify public behavior."
        ),
        "phases": [
            {
                "name": "01-release-workflow",
                "title": "Add release workflow",
                "goal": "Document and test the release path.",
                "technical_approach": "Update the tracked README convention.",
                "files_touched": [
                    {
                        "path": "README.md",
                        "role": "modified",
                        "description": "Document the release workflow.",
                    }
                ],
                "test_strategy": "Assert the documented public workflow.",
                "acceptance_criteria": ["The release path is documented."],
                "deliverables": ["Release workflow documentation."],
                "dependencies_on": [],
            }
        ],
        "code_pointers": [
            {"path": "README.md", "why": "It owns repository guidance."}
        ],
        "risks": [],
        "open_questions": [],
    }


def _tech_lead_approval_response() -> dict:
    return {
        "decision": "approve",
        "summary": "The plan is ready for human approval.",
        "findings": [],
    }


def _tech_lead_change_request_response(message: str) -> dict:
    return {
        "decision": "request_changes",
        "summary": message,
        "findings": [
            {
                "severity": "major",
                "message": message,
                "suggestion": "Clarify the plan and its verification.",
            }
        ],
    }
