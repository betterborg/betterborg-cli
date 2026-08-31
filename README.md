# Betterborg CLI

[![PyPI](https://img.shields.io/pypi/v/betterborg.svg?style=flat-square)](https://pypi.org/project/betterborg/)
[![npm](https://img.shields.io/npm/v/@betterborg/cli.svg?style=flat-square)](https://www.npmjs.com/package/@betterborg/cli)
![Python](https://img.shields.io/badge/Python-3.11%2B-brightgreen?style=flat-square)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](LICENSE)

Betterborg is an AI engineering team for substantial software projects, not a chat window that makes an isolated edit. You give it a task or a PRD; agents investigate your real code, write a reviewed technical plan, decompose it into a dependency graph of tasks, and implement them, each in its own isolated Git worktree. You approve the plan before anything is decomposed, and the estimate before anything runs. It works locally, on your own Claude Code or Codex subscription.

> [!NOTE]
> Pre-alpha. Interfaces may change between releases.

## Get started

1. Install the CLI:

    ```bash
    pip install betterborg
    ```

    Or install the standalone release binary for Darwin or Linux, which needs
    no Python toolchain. It selects ARM64 or x86_64 from the release manifest,
    verifies the SHA-256 digest and exact version before replacing
    `~/.local/bin/betterborg`, and then runs step 2 for you:

    ```bash
    curl -fsSL https://install.betterborg.ai | sh
    ```

2. Activate the agent-host integrations:

    ```bash
    betterborg plugins install --all
    ```

    Then run `/reload-plugins` in any open Claude Code session, and start a new Codex session when prompted.

3. From inside a Git repository, run the loop:

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

To try it without installing:

```bash
npx --yes @betterborg/cli version
```

Plugin activation needs a persistent install, so it is skipped under `npx`.

## Requirements

- Python 3.11+, on macOS or Linux. On Windows, use a WSL2 shell.
- One agent transport: a `claude` or `codex` CLI on `PATH` and **already logged in**, or `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` exported in your shell.

Keep provider keys in your shell or a secret manager, never in `.betterborg/config.toml` or any tracked file.

## Documentation

- [Installation guide](docs/installation.md): version pinning, WSL2, providers, recovery
- [Command guide](docs/commands.md): bootstrap and initialization
- [Release runbook](docs/releasing.md): for maintainers

## Reporting bugs

File a [GitHub issue](https://github.com/betterborg/betterborg-cli/issues).

## Contributing

See [AGENTS.md](AGENTS.md) for repository rules and the standard `make` targets.

## License

Copyright 2026 Betterborg. Licensed under the [MIT License](LICENSE). See [NOTICE](NOTICE) for attribution.
