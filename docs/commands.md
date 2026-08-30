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

The standalone binary installer runs `--all` after verification; after a pip
or uv install, run it yourself. Re-run it after installing or logging in to a
host. Activation requires a persistent install and is refused under the `npx`
and `uvx` wrappers. Open Claude Code sessions require
`/reload-plugins` after installation or upgrade; start a new Codex session when
the installer requests it.

## Continue repository work

Once initialization completes, use `borg create`, `borg plan`, `borg task`,
and `borg execute` as shown by `borg COMMAND --help`. Before executing a
published task generation, `borg task estimate NAME` shows its P50/P80 work and
billing-mode estimate.
