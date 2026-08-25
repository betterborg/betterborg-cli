"""Command-line entry point for BetterBorg."""

import hashlib
import json
import re
import shlex
import subprocess
from functools import wraps
from pathlib import Path

import click

from betterborg_cli import __version__
from betterborg_cli.agent_runtime.api_tools import ApiAgentRole
from betterborg_cli.agent_runtime.selection import select_agent
from betterborg_cli.onboarding import (
    CreateService,
    OnboardingDispatcher,
    create_commands,
)
from betterborg_cli.planning import (
    ArchitectCancelled,
    ArchitectLoop,
    SupervisorCancelled,
    SupervisorLoop,
    TaskPublisher,
    TechLeadCancelled,
    TechLeadLoop,
    approved_plan_digest,
    render_plan_markdown,
    validate_plan,
)
from betterborg_cli.prd_session import InteractiveIO, validate_borg_name
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_config import load_repository_config
from betterborg_cli.repository_files import publish_repository_text
from betterborg_cli.repository_service import RepositoryService
from betterborg_cli.store import (
    Borg,
    BorgState,
    PlanApproval,
    PlanChangeRequest,
    PlanningAttemptStatus,
    SqliteStore,
)
from betterborg_cli.workspace_trust import (
    UntrustedWorkspaceError,
    require_workspace_trust,
)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Work with BetterBorg from the command line."""


@cli.command()
def version() -> None:
    """Print the installed BetterBorg CLI version."""
    click.echo(f"borg {__version__}")


def _stdin_is_interactive() -> bool:
    return click.get_text_stream("stdin").isatty()


def _trusted_workspace_callback(function):
    """Gate a callback before it can load repository-controlled context."""

    @wraps(function)
    def guarded(*args, explicit_trust: bool, **kwargs):
        try:
            paths = RepoPaths.discover()
            interactive = _stdin_is_interactive() and not kwargs.get(
                "json_output", False
            )
            require_workspace_trust(
                paths,
                explicit=explicit_trust,
                interactive=interactive,
                confirm=lambda prompt: click.confirm(prompt, default=False),
            )
        except (UntrustedWorkspaceError, ValueError, RuntimeError) as error:
            raise click.ClickException(str(error)) from error
        return function(*args, repository_path=paths.root, **kwargs)

    return guarded


@cli.command(name="trust")
@click.option(
    "--yes",
    "explicit_trust",
    is_flag=True,
    help="Trust without prompting after accepting the host-access consequence.",
)
@_trusted_workspace_callback
def trust_workspace(repository_path: Path) -> None:
    """Trust this workspace for host-capable agent operations."""
    click.echo(f"Trusted workspace: {repository_path}")


@cli.command(name="init")
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit machine-readable initialization output without prompting.",
)
@click.option(
    "--yes",
    "explicit_trust",
    is_flag=True,
    help="Trust this workspace without prompting and initialize it.",
)
@_trusted_workspace_callback
def initialize_repository(
    repository_path: Path,
    json_output: bool,
) -> None:
    """Register and analyze the current Git repository."""
    paths = RepoPaths.discover(repository_path)
    database = paths.state_dir / "borg.sqlite3"
    interactive = _stdin_is_interactive() and not json_output
    try:
        if not database.resolve().is_relative_to(paths.root):
            raise ValueError(f"repository state path escapes repository: {database}")
        with SqliteStore.open(database) as store:
            service = RepositoryService(
                paths,
                store,
                lambda config: select_agent(
                    config,
                    ApiAgentRole.ANALYSIS,
                    paths,
                    interactive=interactive,
                ),
            )
            result = service.initialize()

            if result.initialized and interactive:
                _write_initialized(result)
                config = load_repository_config(paths)
                io = _interactive_io()
                creator = CreateService(
                    result.repository,
                    store,
                    select_agent(
                        config,
                        ApiAgentRole.PLANNING,
                        paths,
                        interactive=True,
                    ),
                    io=io,
                    editor=_edit_markdown,
                )
                OnboardingDispatcher(
                    result.repository,
                    store,
                    io,
                    creator,
                    result.improvement_prds,
                ).run()
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    commands = create_commands(paths.root, result.improvement_prds)
    if json_output:
        click.echo(
            json.dumps(
                {
                    "repository_id": str(result.repository.id),
                    "initialized": result.initialized,
                    "score": result.analysis.overall_score,
                    "create_commands": [shlex.join(command) for command in commands],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    elif not result.initialized:
        click.echo(f"Repository already initialized: {result.repository.id}")
    elif not interactive:
        _write_initialized(result)
        for command in commands:
            click.echo(shlex.join(command))


@cli.command(name="analyze")
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit the analysis result as machine-readable JSON.",
)
@click.option(
    "--yes",
    "explicit_trust",
    is_flag=True,
    help="Trust this workspace without prompting before analyzing it.",
)
@_trusted_workspace_callback
def analyze_repository(
    repository_path: Path,
    json_output: bool,
) -> None:
    """Re-analyze an initialized Git repository and refresh its outputs."""
    paths = RepoPaths.discover(repository_path)
    database = paths.state_dir / "borg.sqlite3"
    interactive = _stdin_is_interactive() and not json_output
    try:
        if not database.resolve().is_relative_to(paths.root):
            raise ValueError(f"repository state path escapes repository: {database}")
        with SqliteStore.open(database) as store:
            result = RepositoryService(
                paths,
                store,
                lambda config: select_agent(
                    config,
                    ApiAgentRole.ANALYSIS,
                    paths,
                    interactive=interactive,
                ),
            ).analyze()
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    analysis = result.analysis
    previous_score = result.previous_analysis.overall_score
    score_delta = analysis.overall_score - previous_score
    if json_output:
        click.echo(
            json.dumps(
                {
                    "analysis_id": str(analysis.id),
                    "repository_id": str(result.repository.id),
                    "score": analysis.overall_score,
                    "previous_score": previous_score,
                    "delta": score_delta,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    else:
        click.echo(
            f"Analyzed repository {result.repository.id}: "
            f"score {analysis.overall_score:.2f}/5 "
            f"(previous {previous_score:.2f}/5, "
            f"delta {score_delta:+.2f})."
        )


@cli.command(name="create")
@click.argument("name")
@click.option(
    "--prd",
    "source",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    help="Optional local Markdown PRD to improve.",
)
@click.option(
    "--yes",
    "explicit_trust",
    is_flag=True,
    help="Trust this workspace without prompting before creating the Borg.",
)
@_trusted_workspace_callback
def create_borg(
    repository_path: Path,
    name: str,
    source: Path | None,
) -> None:
    """Brainstorm or improve a PRD and create a named Borg."""
    try:
        _validate_create_name(name)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    if not _stdin_is_interactive():
        raise click.ClickException("borg create requires an interactive terminal")
    paths = RepoPaths.discover(repository_path)
    try:
        config = load_repository_config(paths)
        with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
            repository = store.get_repository(config.repository_id)
            if repository is None:
                raise ValueError("repository is not initialized; run 'borg init' first")
            io = _interactive_io()
            result = CreateService(
                repository,
                store,
                select_agent(
                    config,
                    ApiAgentRole.PLANNING,
                    paths,
                    interactive=True,
                ),
                io=io,
                editor=_edit_markdown,
            ).create(name, source)
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    if result.confirmed:
        click.echo(f"Created Borg {result.borg.name!r}: {result.prd_path}")
        click.echo(f"borg plan start {result.borg.name}")
    elif result.questions:
        click.echo("Borg PRD needs more input before it can be created.")
    else:
        click.echo("Borg draft saved without a confirmed PRD.")


@cli.group()
def plan() -> None:
    """Create and review implementation plans for a Borg."""


@plan.command(name="start")
@click.argument("name")
@click.option(
    "--yes",
    "explicit_trust",
    is_flag=True,
    help="Trust this workspace without prompting before planning.",
)
@_trusted_workspace_callback
def start_plan(repository_path: Path, name: str) -> None:
    """Start or resume planning for the named Borg."""
    borg = _continue_planning(repository_path, name)
    _write_planning_gate(name, borg, changed=False)


@plan.command(name="show")
@click.argument("name")
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit the stored plan as validated machine-readable JSON.",
)
def show_plan(name: str, json_output: bool) -> None:
    """Show the latest complete plan for the named Borg."""
    try:
        paths = RepoPaths.discover()
        config = load_repository_config(paths)
        with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
            repository = store.get_repository(config.repository_id)
            if repository is None:
                raise ValueError("repository is not initialized; run 'borg init' first")
            borg = store.get_borg_by_name(repository.id, name)
            if borg is None:
                raise ValueError(
                    f"Borg {name!r} does not exist; run 'borg create {name}' first"
                )
            attempt = next(
                (
                    item
                    for item in reversed(store.list_planning_attempts(borg.id))
                    if item.phase == "architect_plan"
                    and item.status is PlanningAttemptStatus.COMPLETED
                    and item.result is not None
                ),
                None,
            )
            if attempt is None:
                raise ValueError(
                    f"Borg {name!r} does not have a stored plan; "
                    f"run 'borg plan start {name}' first"
                )
            stored_plan = attempt.result
            validate_plan(
                stored_plan,
                paths.root,
                check_repository_state=False,
            )
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    if json_output:
        click.echo(json.dumps(stored_plan, sort_keys=True, separators=(",", ":")))
    else:
        click.echo(render_plan_markdown(stored_plan), nl=False)


@plan.command(name="approve")
@click.argument("name")
@click.option(
    "--yes",
    "explicit_trust",
    is_flag=True,
    help="Trust this workspace without prompting before decomposition.",
)
@_trusted_workspace_callback
def approve_plan(repository_path: Path, name: str) -> None:
    """Approve the current plan and prepare its executable task generation."""
    paths = RepoPaths.discover(repository_path)
    resumable = False
    try:
        config = load_repository_config(paths)
        with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
            repository = store.get_repository(config.repository_id)
            if repository is None:
                raise ValueError("repository is not initialized; run 'borg init' first")
            borg = store.get_borg_by_name(repository.id, name)
            if borg is None:
                raise ValueError(
                    f"Borg {name!r} does not exist; run 'borg create {name}' first"
                )

            approval, plan_path = _bind_plan_approval(paths, store, borg)
            borg = store.get_borg(borg.id)
            if borg is None:
                raise RuntimeError(f"Borg {name!r} disappeared during approval")
            resumable = True

            if borg.state in {BorgState.PM_WORKING, BorgState.SUPERVISOR_WORKING}:
                agent = select_agent(
                    config,
                    ApiAgentRole.PLANNING,
                    paths,
                    interactive=_stdin_is_interactive(),
                )
                borg = SupervisorLoop(
                    repository,
                    borg,
                    store,
                    agent,
                    pm_agent=agent,
                    approved_plan=approval.manifest["plan"],
                    plan_approval=approval,
                ).run().borg

            publication = None
            if borg.state is BorgState.READY_TO_EXECUTE:
                publication = TaskPublisher(repository, store).reconcile(borg.id)
                if publication is None:
                    raise RuntimeError(
                        f"Borg {name!r} is ready to execute but has no current tasks"
                    )
            elif borg.state is not BorgState.BLOCKED:
                raise RuntimeError(
                    f"decomposition stopped in unexpected state {borg.state.value!r}"
                )
    except (SupervisorCancelled, KeyboardInterrupt) as error:
        message = str(error).strip()
        detail = f" ({message})" if message else ""
        raise click.ClickException(
            f"Decomposition for Borg {name!r} was interrupted{detail}. "
            f"Run 'borg plan approve {name}' to resume."
        ) from error
    except (OSError, RuntimeError, ValueError) as error:
        if resumable:
            message = str(error).strip()
            detail = f" ({message})" if message else ""
            raise click.ClickException(
                f"Decomposition for Borg {name!r} could not continue{detail}. "
                f"Run 'borg plan approve {name}' to resume."
            ) from error
        raise click.ClickException(str(error)) from error

    relative_plan = plan_path.relative_to(paths.root).as_posix()
    click.echo(f"Approved plan: {relative_plan} ({approval.plan_digest})")
    if borg.state is BorgState.READY_TO_EXECUTE:
        click.echo(f"Borg {name!r} is ready to execute.")
        click.echo("Current tasks:")
        for item in publication.files:
            click.echo(f"  {item.path.relative_to(paths.root).as_posix()}")
    else:
        click.echo(f"Task decomposition blocked for Borg {name!r}.")


def _bind_plan_approval(
    paths: RepoPaths,
    store: SqliteStore,
    borg: Borg,
) -> tuple[PlanApproval, Path]:
    """Bind or recover one approval for the latest exact Architect plan."""
    plan_attempt = next(
        (
            item
            for item in reversed(store.list_planning_attempts(borg.id))
            if item.phase == "architect_plan"
            and item.status is PlanningAttemptStatus.COMPLETED
            and item.result is not None
        ),
        None,
    )
    if plan_attempt is None:
        raise ValueError(
            f"Borg {borg.name!r} does not have a complete plan to approve"
        )
    current_plan = plan_attempt.result
    validate_plan(current_plan, paths.root, check_repository_state=False)
    digest = approved_plan_digest(current_plan)
    body = render_plan_markdown(current_plan)
    body_digest = "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()
    relative_path = Path(".borg/plans") / f"{borg.name}.md"
    plan_path = paths.root / relative_path

    approvals = store.list_plan_approvals(borg.id)
    if approvals:
        approval = approvals[-1]
        if borg.state is BorgState.PLAN_APPROVAL_PENDING:
            raise ValueError(f"Borg {borg.name!r} already has a plan approval")
        manifest_plan = approval.manifest.get("plan")
        if (
            approval.attempt_id != plan_attempt.id
            or approval.plan_digest != digest
            or manifest_plan != current_plan
            or approval.manifest.get("plan.md") != body_digest
            or approval.manifest.get("plan_path") != relative_path.as_posix()
        ):
            raise ValueError(
                f"Borg {borg.name!r} approval does not match its current plan"
            )
        try:
            existing = plan_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            publish_repository_text(
                plan_path, body, root=paths.root, overwrite=True
            )
        else:
            if existing != body:
                raise ValueError(f"approved plan Markdown drifted: {relative_path}")
        _require_git_trackable(paths.root, relative_path)
        return approval, plan_path

    if borg.state is not BorgState.PLAN_APPROVAL_PENDING:
        raise ValueError(
            f"Borg {borg.name!r} cannot approve a plan from state "
            f"{borg.state.value!r}; a plan must be awaiting approval"
        )
    publish_repository_text(plan_path, body, root=paths.root, overwrite=True)
    _require_git_trackable(paths.root, relative_path)
    approval = PlanApproval(
        borg_id=borg.id,
        attempt_id=plan_attempt.id,
        plan_digest=digest,
        manifest={
            "plan": current_plan,
            "plan.md": body_digest,
            "plan_path": relative_path.as_posix(),
        },
        approved_by="operator",
    )
    with store.transaction():
        store.append_plan_approval(approval)
        store.compare_and_set_borg_state(
            borg.id,
            expected_state=borg.state,
            expected_version=borg.state_version,
            new_state=BorgState.PM_WORKING,
        )
    return approval, plan_path


def _require_git_trackable(root: Path, relative_path: Path) -> None:
    ignored = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "--quiet", "--", str(relative_path)],
        check=False,
    )
    if ignored.returncode == 0:
        raise ValueError(f"approved plan path is ignored by Git: {relative_path}")
    if ignored.returncode not in {0, 1}:
        raise RuntimeError("could not verify approved plan Git tracking status")


@plan.command(name="change")
@click.argument("name")
@click.option(
    "--note",
    help="Plain-language changes for the Architect to apply.",
)
@click.option(
    "--yes",
    "explicit_trust",
    is_flag=True,
    help="Trust this workspace without prompting before revising the plan.",
)
@_trusted_workspace_callback
def change_plan(repository_path: Path, name: str, note: str | None) -> None:
    """Request changes to a plan awaiting human approval."""
    if note is None:
        note = _prompt("Change note")
    if note is None or not note.strip():
        raise click.ClickException("plan change note must not be empty")
    note = note.strip()

    borg = _continue_planning(repository_path, name, change_note=note)
    _write_planning_gate(name, borg, changed=True)


def _continue_planning(
    repository_path: Path,
    name: str,
    *,
    change_note: str | None = None,
) -> Borg:
    """Load and drain one initial or change-request planning lifecycle."""
    paths = RepoPaths.discover(repository_path)
    change_requested = change_note is not None
    resumable = False
    try:
        config = load_repository_config(paths)
        with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
            repository = store.get_repository(config.repository_id)
            if repository is None:
                raise ValueError("repository is not initialized; run 'borg init' first")
            borg = store.get_borg_by_name(repository.id, name)
            if borg is None:
                raise ValueError(
                    f"Borg {name!r} does not exist; run 'borg create {name}' first"
                )

            if change_note is not None:
                if borg.state is not BorgState.PLAN_APPROVAL_PENDING:
                    raise ValueError(
                        f"Borg {name!r} cannot change its plan from state "
                        f"{borg.state.value!r}; a plan must be awaiting approval"
                    )
                requests = store.list_plan_change_requests(borg.id)
                request = PlanChangeRequest(
                    borg_id=borg.id,
                    round=max((item.round for item in requests), default=0) + 1,
                    note=change_note,
                )
                with store.transaction():
                    store.append_plan_change_request(request)
                    borg = store.compare_and_set_borg_state(
                        borg.id,
                        expected_state=borg.state,
                        expected_version=borg.state_version,
                        new_state=BorgState.ARCHITECT_WORKING,
                    )

            if borg.state not in {
                BorgState.PLAN_APPROVAL_PENDING,
                BorgState.BLOCKED,
            }:
                resumable = True
                agent = select_agent(
                    config,
                    ApiAgentRole.PLANNING,
                    paths,
                    interactive=_stdin_is_interactive(),
                )
                io = _interactive_io()
                if borg.state is BorgState.DRAFT:
                    borg = ArchitectLoop(
                        repository,
                        borg,
                        store,
                        agent,
                        io=io,
                    ).run().borg
                borg = TechLeadLoop(
                    repository,
                    borg,
                    store,
                    agent,
                    architect_agent=agent,
                    io=io,
                ).run().borg
    except (ArchitectCancelled, TechLeadCancelled, KeyboardInterrupt) as error:
        message = str(error).strip()
        detail = f" ({message})" if message else ""
        action = "Plan change" if change_requested else "Planning"
        raise click.ClickException(
            f"{action} for Borg {name!r} was interrupted{detail}. "
            f"Run 'borg plan start {name}' to resume."
        ) from error
    except (OSError, RuntimeError, ValueError) as error:
        if resumable:
            message = str(error).strip()
            detail = f" ({message})" if message else ""
            action = "Plan change" if change_requested else "Planning"
            raise click.ClickException(
                f"{action} for Borg {name!r} could not continue{detail}. "
                f"Run 'borg plan start {name}' to resume."
            ) from error
        raise click.ClickException(str(error)) from error
    return borg


def _write_planning_gate(name: str, borg: Borg, *, changed: bool) -> None:
    """Report the actionable terminal gate reached by a planning lifecycle."""
    if borg.state is BorgState.PLAN_APPROVAL_PENDING:
        suffix = " after applying the change" if changed else ""
        click.echo(f"Plan approval pending for Borg {name!r}{suffix}.")
        click.echo(f"Review it with: borg plan show {name}")
    elif borg.state is BorgState.BLOCKED:
        suffix = " while applying the change" if changed else ""
        click.echo(f"Planning blocked for Borg {name!r}{suffix}.")
        click.echo(f"Review the saved Tech Lead findings with: borg plan show {name}")
    else:
        raise click.ClickException(
            f"Planning stopped in unexpected state {borg.state.value!r}. "
            f"Run 'borg plan start {name}' to resume."
        )


def _write_initialized(result) -> None:
    click.echo(
        f"Initialized repository {result.repository.id} "
        f"with score {result.analysis.overall_score:.2f}/5."
    )


def _validate_create_name(name: str) -> None:
    validate_borg_name(name)
    if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None:
        raise ValueError(
            "Borg name must use kebab-case lowercase letters and numbers"
        )


def _interactive_io() -> InteractiveIO:
    return InteractiveIO(
        prompt=_prompt,
        confirm=lambda message, default: click.confirm(message, default=default),
        write=click.echo,
    )


def _prompt(message: str) -> str | None:
    try:
        return click.prompt(message, default="", show_default=False)
    except click.Abort:
        return None


def _edit_markdown(body: str) -> str | None:
    return click.edit(body, extension=".md")
