# The discovery adapter

This is the throwaway adapter from the PRD's first gate. It exists to reproduce
`plans/benchmark-harness-adapters/GAPS.md`, not to become the benchmark adapter.
Every improvisation in it is a product gap that has not been fixed yet, so
carrying it forward would carry those gaps forward disguised as working code.

It is tracked for one reason: a run has to be reproducible. `BETTERBORG_COMMIT`
below names the build that was under test, and this file names the harness that
drove it; neither is recoverable from the other. That is the whole argument for
keeping it in the repository, and it does not make it the adapter we ship.
Delete it once the gaps it stands on are closed and the real adapter exists.

## Running it

Pier and the DeepSWE task set are external to this repository. From a directory
holding this package on `PYTHONPATH`:

    env PYTHONPATH=. CODEX_FORCE_AUTH_JSON=1 pier run \
      -p /path/to/deep-swe/tasks/abs-module-cache-flags \
      --agent-import-path pier_adapter.agent:BetterborgPierAgent \
      -m gpt-5.6-sol --ae CODEX_FORCE_AUTH_JSON=1 \
      -o <out>/jobs --job-name <name> -n 1 -k 1

`CODEX_FORCE_AUTH_JSON=1` is not optional: without it no `auth.json` reaches the
container, every model call is refused, and the run aborts before `init`.

`BETTERBORG_COMMIT` in `agent.py` pins the Betterborg build the container
installs from GitHub. It must be pushed to `origin` before launching, or the
image build resolves nothing.

## What each improvisation stands on

| Step | Gap | What it fakes |
|---|---|---|
| `docker-stub` | 1 | a `docker` that logs and exits 0, because preparation's program is always required |
| `npm-shim` | 2 | rewrites a plain `npm install` to `npm ci`, because preparation may not modify tracked files |
| `land` | 3 | merges `project/benchmark` onto the checkout, because execution leaves its work on its own branch |
| running as `root` | 4 | the worktrees directory is derived from the checkout's parent and cannot be placed |
| `seed-config` | 5 | a hand-written `config.toml`, because configuration cannot be supplied before analysis runs |
| `git-identity` | 6 | a global git identity, because the coding phase requires a commit and supplies none |
| the note appended in `write-prd` | 7 | states a plan-schema rule no prompt states; this one contaminates the task contract and is the most urgent to remove |
| the greps in `plan` and `approve` | 11 | reads output for a blocked gate; the pinned build exits non-zero on one, so this now only guards against an older build |

## What is not an improvisation

`model-patch` writes `/logs/artifacts/model.patch` because the task declares the
artifact and, separately, the command that produces it, and the installed Pier
reads an older task schema and drops the second declaration. Producing a
declared artifact is harness work. Without this step the verifier grades a
pristine checkout and every score is the untouched repository's.

`BETTERBORG_SANDBOX=host` is a released operator declaration, not a workaround:
Codex's own sandbox cannot start in a task container because bubblewrap needs an
unprivileged user namespace the default seccomp profile refuses.

The saved planning database, task graph, execution state and step report exist
so a failure can be diagnosed after the container is gone. The execution
snapshot is taken after `execute` and carries the per-attempt artifacts, because
why a task blocked at review is recorded nowhere else and does not survive the
container.
