# @betterborg/cli

Betterborg is an AI engineering team for substantial software projects, not a chat window that makes an isolated edit. You give it a task or a PRD; agents investigate your real code, write a reviewed technical plan, decompose it into a dependency graph of tasks, and implement them, each in its own isolated Git worktree. You approve the plan before anything is decomposed, and the estimate before anything runs. It works locally, on your own Claude Code or Codex subscription.

> [!NOTE]
> Pre-alpha. Interfaces may change between releases.

This package is a launcher, not the CLI itself. It resolves a `betterborg` matching its own version, then forwards your arguments, stdio, and signals to it. Nothing is downloaded at install time.

## Try it

```bash
npx --yes @betterborg/cli version
```

## Install

```bash
npm install --global @betterborg/cli
```

Then activate the agent-host integrations, which need a persistent install rather than `npx`:

```bash
betterborg plugins install --all
```

Run `/reload-plugins` in any open Claude Code session, and start a new Codex session when prompted.

From inside a Git repository, run the loop:

```bash
betterborg trust                     # trust this worktree (machine-local)
betterborg init                      # register and analyze the repository
betterborg create my-feature         # or: betterborg create my-feature --prd spec.md
betterborg plan start my-feature     # agents plan, review, and iterate
betterborg plan approve my-feature   # your gate: nothing is decomposed before this
betterborg task estimate my-feature  # P50/P80 work and cost
betterborg execute my-feature        # approve the estimate, then run
```

`betterborg execute` hands off reviewed local branches. Add `--push` or `--pr` to publish them. `betterborg --help` is the full command index.

## How the launcher finds the CLI

On each run, in order:

1. **An installed `betterborg` on `PATH`**, if `betterborg version` reports this package's exact version. A `pip install betterborg` or the standalone installer takes over from here, and the launcher adds nothing.
2. **The standalone release binary** for macOS and Linux on arm64 or x86_64. It is fetched from the matching GitHub release, verified against the published SHA-256 digest, and cached under `$XDG_CACHE_HOME/betterborg/cli/<version>`, or `~/.cache/betterborg/cli/<version>`, so later runs reuse it. A binary that fails verification is discarded, never executed.
3. **`uvx --from betterborg==<version> betterborg`**, if [uv](https://docs.astral.sh/uv/) is on `PATH`.

If none of those resolve, the launcher exits with what it tried and how to fix it.

## Requirements

- Node.js 18+ to run the launcher.
- macOS or Linux. On Windows, use a WSL2 shell.
- One agent transport: a `claude` or `codex` CLI on `PATH` and **already logged in**, or `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` exported in your shell.

Step 3 additionally needs Python 3.11+, which `uv` will provide.

Keep provider keys in your shell or a secret manager, never in `.betterborg/config.toml` or any tracked file.

## Documentation

- [Installation guide](https://github.com/betterborg/betterborg-cli/blob/main/docs/installation.md): version pinning, WSL2, providers, recovery
- [Command guide](https://github.com/betterborg/betterborg-cli/blob/main/docs/commands.md): bootstrap and initialization
- [Repository](https://github.com/betterborg/betterborg-cli)

## Reporting bugs

File a [GitHub issue](https://github.com/betterborg/betterborg-cli/issues).

## License

Copyright 2026 Betterborg. Licensed under the [MIT License](https://github.com/betterborg/betterborg-cli/blob/main/LICENSE). See [NOTICE](https://github.com/betterborg/betterborg-cli/blob/main/NOTICE) for attribution.
