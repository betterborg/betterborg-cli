# BetterBorg command bootstrap

`borg --help` is the authoritative command index. These are the commands needed
to get a new checkout ready without hiding trust or provider decisions.

## Verify and initialize

```console
borg version
borg trust
borg init
```

`borg version` and `borg --help` do not initialize or trust a repository.
`borg trust` records a machine-local decision for the current Git worktree.
`borg init` registers and analyzes it, then offers interactive onboarding.

For automation, make trust explicit and provide one agent transport. A
`claude` or `codex` CLI on `PATH` is used ahead of any provider credential,
and must already be logged in because selection tests only for the
executable:

```console
borg init --yes --json
```

With neither CLI on `PATH`, export `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
instead. `--json` disables prompts and returns the repository identifier,
initialization status, score, and suggested create commands. Credentials
belong in the process environment or a secret manager and must not be
committed.

## Host integrations

```console
borg plugins install --all
borg plugins install --host claude
borg plugins install --host codex
```

The persistent installer runs `--all` after binary verification. Re-run it
after installing or logging in to a host. Open Claude Code sessions require
`/reload-plugins` after installation or upgrade; start a new Codex session when
the installer requests it.

## Continue repository work

Once initialization completes, use `borg create`, `borg plan`, `borg task`,
and `borg execute` as shown by `borg COMMAND --help`. Before executing a
published task generation, `borg task estimate NAME` shows its P50/P80 work and
billing-mode estimate.

## Progress output

Agent-backed terminal commands (`init`, `analyze`, `create`, the planning
commands, and `execute`) report their work as stages. Progress goes to stderr;
the command's result continues to go to stdout. On an interactive terminal,
running stages are live, transient rows that are updated in place. Completed,
failed, and stopped stages become permanent lines, followed by a summary. For
example, an analysis can display a live row like this while the agent is
working:

```text
running Analyze repository (12.4s) — reading: src/betterborg_cli/cli.py
```

When the stage finishes, the transient row is replaced by canonical permanent
output such as:

```text
completed Analyze repository — score 4.20/5 (18.7s)
summary: 4 completed, 0 failed, 0 stopped — 0 retained
```

Durations and results naturally vary by repository. Long labels and activity
details are truncated to the terminal width, and at most eight live rows are
shown; permanent outcome lines still record every stage.

When stderr is redirected or is otherwise noninteractive, the same progress is
plain, newline-delimited text with no cursor control or color. A line is written
when work starts, and active work is repeated every 30 seconds as a heartbeat:

```text
running Analyze repository (0.0s)
running Analyze repository (30.0s) — reading: pyproject.toml
completed Analyze repository — score 4.20/5 (35.2s)
```

Interactive questions, confirmations, editor sessions, and ordinary command
results temporarily suspend the live display. Progress lines produced during
that boundary are queued, so prompts remain readable and progress resumes after
input completes.

Successful structured commands keep stdout machine-readable. In particular,
`--json` suppresses progress rather than mixing it with either output stream:

```console
borg analyze --yes --json >analysis.json 2>progress.log
```

`analysis.json` contains only the documented JSON result and `progress.log` is
empty unless the command reports a non-progress diagnostic. MCP stdio is also
headless: its stdout remains protocol JSON and does not use the terminal
progress renderer.

## Interrupting work

Press Ctrl+C once to request cooperative cancellation. BetterBorg prints
`stopping...`, stops starting new work, cancels active agent, provider, and
local-command processes, and waits for their cleanup and durable state to be
reconciled. A normally reconciled interruption exits with status 130 and can
end with output like:

```text
stopping...
stopped Analyze repository — interrupted (12.4s)
summary: 1 completed, 0 failed, 1 stopped — 0 retained
```

If cooperative cleanup has not finished after one second, BetterBorg requests
a forced stop of registered child process groups and prints `Force stopping...`.
Pressing Ctrl+C a second time is the safety valve that requests this force path
immediately instead of waiting for the deadline. A forced exit may occur before
the normal closing summary, but its process exit status is still 130.

Cancellation does not roll back durable work that already completed. On the
next invocation, reused stage outcomes are marked `[retained]`; this is reuse of
a safe persisted checkpoint, not continuation of an in-flight process:

```text
completed Analyze repository — score 4.20/5 (—) [retained]
```

What is retained depends on the command:

- `borg init` reuses a completed repository analysis and any completed role
  prompts, then generates only missing initialization outputs.
- `borg plan start`, `borg plan change`, and `borg plan approve` reuse completed
  planning attempts, revisions, approvals, and safely published task state.
- `borg execute` retains tasks already recorded as `DONE`. It records the owned
  run as `CANCELLED` and releases or cleanup-fences unfinished claims before
  returning; a later execution starts new work from the resulting durable task
  state.

Unfinished agent requests, HTTP requests, and local processes are never
resumable in place. Re-run the command named by the interruption message; it
starts a new invocation and uses only the command-specific durable state listed
above.
