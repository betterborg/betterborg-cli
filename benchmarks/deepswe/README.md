# DeepSWE adapter

A Pier agent that runs Betterborg unattended against a DeepSWE task. It turns
the task instructions into a PRD, plans and decomposes the work, executes the
tasks it produced, and submits the resulting diff as the task's patch.

Pier and the DeepSWE task set are external to this repository.

## Running it

From this directory, which holds the package on `PYTHONPATH`:

    env PYTHONPATH=. CODEX_FORCE_AUTH_JSON=1 pier run \
      -p /path/to/deep-swe/tasks/<task> \
      --agent-import-path pier_adapter.agent:BetterborgPierAgent \
      -m gpt-5.6-sol --ae CODEX_FORCE_AUTH_JSON=1 \
      -o <output-dir> --job-name <name> -n 1 -k 1

Point `-p` at one task directory, or at the task set and select tasks with
`-i <name>` repeated.

## Two things that cost a run

`CODEX_FORCE_AUTH_JSON=1` is not optional. Without it no `auth.json` reaches
the container, every model call is refused, and the run aborts during setup.
Pass it with `--ae` so the command carries its own requirement.

`BETTERBORG_COMMIT` in `agent.py` pins the Betterborg build the container
installs from GitHub. It must be pushed to `origin` before launching, or the
image build resolves nothing.
