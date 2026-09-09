# Betterborg command bootstrap

`betterborg --help` is the authoritative command index. These are the commands needed
to get a new checkout ready without hiding trust or provider decisions.

## Verify and initialize

```console
betterborg version
betterborg trust
betterborg init
```

`betterborg version` and `betterborg --help` do not initialize or trust a repository.
`betterborg trust` records a machine-local decision for the current Git worktree.
`betterborg init` registers and analyzes it, then offers interactive onboarding.

For automation, make trust explicit and provide one agent transport. A
`claude` or `codex` CLI on `PATH` is used ahead of any provider credential,
and must already be logged in because selection tests only for the
executable:

```console
betterborg init --yes --json
```

With neither CLI on `PATH`, export `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
instead. `--json` disables prompts and returns the repository identifier,
initialization status, score, and suggested create commands. Credentials
belong in the process environment or a secret manager and must not be
committed.

## Agent configuration

The first `betterborg init` creates `.betterborg/config.toml`. Alongside the
generated repository identity, its agent configuration has this complete
shape (this example shows the values produced when Codex is selected):

```toml
[agents.defaults]
adapter = "codex"
model = "gpt-5.6-sol"
effort = "high"

[agents.analysis]

[agents.requirements]

[agents.architect]

[agents.tech_lead]

[agents.pm]

[agents.supervisor]

[agents.coding]

[agents.review]

[agents.merge]
```

Each table accepts optional, non-empty `adapter`, `model`, and `effort`
strings. Betterborg resolves each of those three settings independently: the
stage value wins when present, then `[agents.defaults]`, then the built-in
selection or value. This means, for example, that a stage can override only
its effort and continue to inherit its adapter and model.

Without an adapter in either applicable table, selection is native-first:
Betterborg looks for the `claude` and then `codex` executables before trying
the `anthropic` and `openai` API adapters. The built-in current model is
`claude-opus-5` for `claude` or `anthropic`, and `gpt-5.6-sol` for `codex` or
`openai`; built-in effort is `high` for every stage.

Fresh initialization pins the selected adapter, its corresponding current
model, and high effort in `[agents.defaults]`. If that initial selection is a
native adapter, the pin intentionally prevents a later invocation from
switching automatically merely because another adapter becomes available.
An adapter named explicitly by a stage or by the defaults is authoritative:
an unknown adapter, a missing native executable, or a missing API credential
causes selection to fail with setup guidance instead of falling back to a
different adapter.

Provider credentials are environment-only. Keep `ANTHROPIC_API_KEY` and
`OPENAI_API_KEY` in the process environment or a secret manager; they are not
valid tracked configuration and `betterborg init` never writes them to this
file.

Codex runs a read-only phase under a read-only sandbox and every other phase
with sandboxing off. Inside a container that is already the boundary, the
read-only sandbox usually cannot start, because it is built from an
unprivileged user namespace that such a container is normally refused. Set
`BETTERBORG_SANDBOX=host` to declare the environment already isolated, and
Codex then runs without a sandbox in every phase.

Weigh what that gives up. Every read-only phase under Codex is held to reading
by this sandbox alone, creating a Borg and generating prompts as much as
analysis and planning, and what the sandbox denies is writing anywhere and
reaching the network. It does not confine reads: under either setting Codex can
read outside the workspace it was given. Set `host` only where something around
Betterborg is genuinely the boundary. Claude is unaffected either way, because
it is held to reading by a tool allowlist rather than by a sandbox.

The variable accepts `auto` and `host`, in any case and with surrounding
spaces. Unset is the default and an empty value is read the same way; any other
value fails any run that would launch Codex. Like a credential it belongs to
whoever starts Betterborg, so it is environment-only and not valid tracked
Betterborg configuration.

## Betterborg's own files

Betterborg keeps its configuration, prompts, PRDs, plans, published tasks and
score in `.betterborg` inside the repository, and adds a managed block to the
repository's `.gitignore` so its state directory stays out of Git. For a team
that owns the repository this is the point: the configuration is reviewed and
shared like any other checked-in file.

Working on a repository you are only passing through, set `BETTERBORG_HOME` to
an absolute path outside it:

```console
BETTERBORG_HOME=~/.betterborg/acme betterborg init --yes
```

Configuration, prompts, PRDs, plans, tasks, score, state and artifacts all
move there together, and the repository's working tree and `.gitignore` are
left exactly as Betterborg found them: with nothing of Betterborg's inside the
repository there is nothing to ignore, so no managed block is written. The
task worktrees Betterborg mints are unaffected; they are siblings of the
repository under `.betterborg-worktrees`, and are not the operator's to place.

Unset is the default and an empty value is read the same way. A path that
resolves inside the repository fails the run, because it would reintroduce
exactly what the variable exists to keep out; so does a path that contains the
repository, which would put the whole working tree inside what Betterborg
owns, and so does a relative path. One directory serves one repository: a
directory already holding another repository's configuration is refused rather
than serving both, and it goes on refusing after its state directory is
deleted, because a relocated directory records the repository it serves beside
that configuration. Like the sandbox declaration it belongs to whoever starts
Betterborg, so it is environment-only and not valid tracked Betterborg
configuration.

## Host integrations

```console
betterborg plugins install --all
betterborg plugins install --host claude
betterborg plugins install --host codex
```

The standalone binary installer runs `--all` after verification; after a pip
or uv install, run it yourself. Re-run it after installing or logging in to a
host. Activation requires a persistent install and is refused under the `npx`
and `uvx` wrappers. Open Claude Code sessions require
`/reload-plugins` after installation or upgrade; start a new Codex session when
the installer requests it.

## Continue repository work

Once initialization completes, use `betterborg create`, `betterborg plan`, `betterborg task`,
and `betterborg execute` as shown by `betterborg COMMAND --help`. Before executing a
published task generation, `betterborg task estimate NAME` shows its P50/P80 work and
billing-mode estimate.

## Run on a host that has less than the repository names

`betterborg execute` requires what the run will use, not everything the
repository has ever been able to do. The toolchains and package managers the
analysis lists are an inventory written for a person to read, so a host with no
program by one of those names is not refused; every command that runs already
requires the program it invokes.

The catalog lists what a repository can do, and only part of it settles
whether a change broke anything. Each catalogued command says whether running
it verifies the repository: a test run, a linter, a type checker or other
static analysis, a formatter in a check mode that reports rather than rewrites,
or a build whose output the repository ignores. Everything else does not,
including a command that serves, watches, publishes, waits for input, measures
rather than checks, or writes anything the repository tracks or would report as
untracked. `.gitignore` decides which outputs count, and where it cannot be
told, the command is not a check. A run whose catalog declares no check at all is refused
before any task is coded, because nothing in it could prove a change safe.
The sanity gate runs the ones that do. An analysis recorded before the question
was asked says nothing, and every command in it still runs.

A catalog command whose program is missing is dropped from the run rather than
refusing it, and never silently. The Preflight stage names each dropped
command, and so does the result of every task that would have run it:

```text
completed Preflight — 1 sanity command dropped: cargo test: host executable is
not available: cargo (evidence: Cargo.toml)
```

A dropped check cannot fail, so a task that skipped one is not the same as a
task that passed it. The checks the host can run still run, and a host that can
run none of them is refused rather than spending the run. Commands that build
the run itself are never dropped: a worktree is prepared by the analysis's
materialize commands when it declares any and by its prepare commands
otherwise, and a host that cannot run a program that list names is refused.

Secrets follow the commands. One that nothing left in the run consumes does not
block, and a secret the analysis names twice blocks only when the two records
disagree, in which case the refusal says what they disagree on.

## Adopt an existing PRD

When the PRD is already written and authoritative, `--adopt` publishes it as
the Borg's confirmed PRD unchanged, including whether it ends in a newline.
Line endings are read the way every other PRD source is read, so a CRLF file
is published with newlines:

```console
betterborg create my-feature --prd spec.md --adopt --yes
```

Adoption holds no requirements interview, so it selects no agent, needs no
provider credential, reports no progress stages, and does not require an
interactive terminal. It still requires a trusted workspace, which is what
`--yes` grants above. `--adopt` requires `--prd`, because a PRD that has yet to
be brainstormed is not one that can be adopted. The Borg it creates is the one
an interview would have confirmed, so continue with `betterborg plan start
NAME` as usual.

## Plan without a terminal

The Architect asks when the requirements do not settle something it needs, and
with a terminal the operator answers. `--unattended` supplies the missing
party: the Architect is told nobody can be asked, so it settles each
uncertainty itself, on the reading the evidence it already read best supports.

```console
betterborg plan start my-feature --yes --unattended
```

The plan names those decisions itself, and `betterborg plan show` renders them
under `## Assumptions`, so the gaps a run closed on its own stay in front of
whoever reads it. A plan that says nothing about them is asked once,
shown what it decided, and told to say what it still rests on; if it says
nothing again the recorded decisions are published in its place. A revision
that no longer rests on anything assumed says so with an empty list, which
retires what it would otherwise inherit. An Architect that asks a question anyway
is answered the same way instead of ending the run, and that answer is stored
beside its question, marked as assumed rather than answered.

`betterborg plan change NAME --note ... --unattended` revises the same way, so
a Borg planned without a terminal can be changed without one.

Without `--unattended` planning still prompts, and a prompt that returns
nothing still stops the run for a person to resume. A plan written that way
claims no assumptions of its own, because every requirement there was given; it
keeps the ones the plan it revises already carried, minus any over a question
the operator has since answered.

An unattended run's questions are bounded per planning cycle, so a Borg that
spent its budget planning can still be revised. An Architect that keeps asking
past that budget ends the run with its unanswered round preserved, so
`betterborg plan start NAME` resumes it with a person answering.

A run that blocks says so and exits non-zero, so a script driving Betterborg
with nobody watching stops on it rather than carrying on. A plan waiting for
approval is the ordinary end of `betterborg plan start NAME` and exits zero,
because it is what an unattended plan is meant to reach.

## Choose how many revisions a plan gets

The Tech Lead reviews the Architect's plan, and where it finds something wrong
the Architect revises and it reviews again. Three rounds is what a plan gets,
and one the Tech Lead still will not approve then blocks with its findings
kept, so `betterborg plan show NAME` says what stood in the way.

A repository that wants more or fewer attempts at agreement sets its own
budget in `.betterborg/config.toml`:

```toml
[planning]
review_rounds = 5
```

Decomposition has the same shape and the same knob. The Supervisor reviews the
Project Manager's task batch, sends it back where it finds something wrong, and
after its rounds a batch it still will not approve blocks with its findings
kept. `decomposition_rounds` sets how many it gets:

```toml
[planning]
decomposition_rounds = 5
```

Both are read when a run starts and govern that run. A plan or a batch that has
already blocked stays blocked whatever the setting becomes afterwards.

The value is a whole number of at least one, and anything else is refused when
the configuration is read rather than part-way through a review. Each review
is told the round it is on and the budget it has. Raising the budget buys
further rounds, never approval: a plan that spends the larger budget
unapproved blocks exactly as one that spends the default does.

## Progress output

Agent-backed terminal commands (`init`, `analyze`, `create`, the planning
commands, and `execute`) report their work as stages. Progress goes to stderr;
the command's result continues to go to stdout. On an interactive terminal,
running stages are live, transient rows that are updated in place. Completed,
failed, and stopped stages become permanent lines. Commands that close a full
run, including `execute` and reconciled interruptions, also print a summary.
For example, an analysis can display a live row like this while the agent is
working:

```text
⠋ Analyze repository     0:12  reading src/betterborg_cli/cli.py
```

When the stage finishes, the transient row is replaced by canonical permanent
output such as:

```text
✔ Analyze repository     0:18  score 4.20/5
```

The leading mark communicates the state: `✔` completed, `✖` failed, and
`■` stopped. Running rows use an animated dots spinner, while pending child rows
use `◦`. Durations use `M:SS` below one hour and `H:MM:SS` from one hour
onward. Current work uses product language such as `thinking`, `reading PATH`,
`searching "TERM"`, `running COMMAND`, or `writing PATH`. Durations, results,
and spinner frames naturally vary by repository and refresh. Long labels and
activity details are truncated to the terminal width, and at most eight live
rows are shown; permanent outcome lines still record every stage.

When stderr is redirected or is otherwise noninteractive, the same progress is
plain, newline-delimited text with no cursor control or color. A line is written
when work starts. After a running stage crosses the 30-second heartbeat
interval, the next progress refresh can repeat its active row. Activity and
stage changes cause refreshes; the optional `execute` push and pull-request
stages also refresh periodically while their commands run. A periodic push
heartbeat can look like this:

```text
⠋ Push project branch    0:00  thinking
⠋ Push project branch    0:30  thinking
✔ Push project branch    0:35  Pushed project/example to origin.
```

Heartbeats are refresh-driven rather than a timer guarantee for every stage. A
silent agent or provider wait may therefore go longer than 30 seconds without
another line.

Interactive questions, confirmations, editor sessions, and ordinary command
results temporarily suspend the live display. Progress lines produced during
that boundary are queued, so prompts remain readable and progress resumes after
input completes.

Successful structured commands keep stdout machine-readable. In particular,
`--json` suppresses progress rather than mixing it with either output stream:

```console
betterborg analyze --yes --json >analysis.json 2>progress.log
```

`analysis.json` contains only the documented JSON result and `progress.log` is
empty unless the command reports a non-progress diagnostic. MCP stdio is also
headless: its stdout remains protocol JSON and does not use the terminal
progress renderer.

## Interrupting work

Press Ctrl+C once to request cooperative cancellation. Betterborg prints
`stopping…`, stops starting new work, cancels active agent, provider, and
local-command processes, and waits for their cleanup and durable state to be
reconciled. A normally reconciled interruption exits with status 130 and can
end with output like:

```text
stopping…
■ Analyze repository     0:12  interrupted
1 of 2 stages finished in 0:12; 0 failed and 1 stopped.
```

If cooperative cleanup has not finished after one second, Betterborg requests
a forced stop of registered child process groups and prints `Force stopping...`.
Pressing Ctrl+C a second time is the safety valve that requests this force path
immediately instead of waiting for the deadline. A forced exit may occur before
the normal closing summary, but its process exit status is still 130.

Cancellation does not roll back durable work that already completed. When a
later invocation can reuse a stage outcome, its result says
`reused from earlier run`; this is reuse of a safe persisted checkpoint, not
continuation of an in-flight process:

```text
✔ Analyze repository     —  score 4.20/5 · reused from earlier run
```

What is retained depends on the command:

- `betterborg create` stores the Borg name, its PRD session, and each completed
  conversation turn as soon as the session begins. After exit 130, check
  `.betterborg/prds/NAME.md`. If it is absent, interruption happened before
  publication; the current CLI cannot resume the stored session and rejects
  another `betterborg create` using the same name, so retry with a different Borg
  name. If it exists, cancellation raced with atomic publication and
  reconciliation retained the confirmed PRD; do not create it again, and
  continue with `betterborg plan start NAME`.
- `betterborg init` reuses a completed repository analysis and any completed role
  prompts, then generates only missing initialization outputs.
- `betterborg plan start`, `betterborg plan change`, and `betterborg plan approve` reuse completed
  planning attempts, revisions, approvals, and safely published task state.
- `betterborg execute` retains tasks already recorded as `DONE`. It records the owned
  run as `CANCELLED` and releases or cleanup-fences unfinished claims before
  returning; a later execution starts new work from the resulting durable task
  state.

Unfinished agent requests, HTTP requests, and local processes are never
resumable in place. After an interrupted `betterborg plan change NAME`, run
`betterborg plan start NAME`: the change request has already been saved, so submitting
another change is rejected. Except for that recovery command and the two
`betterborg create` outcomes above, re-run the command you invoked. Every retry starts
a new invocation and uses only the command-specific durable state listed above.
