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

For automation, provide exactly one usable provider credential and make trust
explicit:

```console
export OPENAI_API_KEY='your-provider-key'
borg init --yes --json
```

`ANTHROPIC_API_KEY` is the supported alternative. `--json` disables prompts
and returns the repository identifier, initialization status, score, and
suggested create commands. Credentials belong in the process environment or a
secret manager and must not be committed.

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

Once initialization completes, use `borg create`, `borg plan`, `borg tasks`,
and `borg execute` as shown by `borg COMMAND --help`. Before executing a
published task generation, `borg task estimate NAME` shows its P50/P80 work and
billing-mode estimate.
