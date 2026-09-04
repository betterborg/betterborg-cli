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
summary: 1 completed, 0 failed, 1 stopped — 0 retained
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
