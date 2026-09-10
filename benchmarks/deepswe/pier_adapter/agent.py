"""Pier agent that runs Betterborg unattended in a DeepSWE task container.

Every Betterborg step is run and logged, and a failing step does not end the
job: the report records what each step did, so a run can be read after the
container is gone.

    PYTHONPATH=<this dir> pier run -p /tmp/deep-swe/tasks/<task> \
        --agent-import-path pier_adapter.agent:BetterborgPierAgent
"""

from __future__ import annotations

import json
from pathlib import Path

from pier.agents.installed.codex import Codex
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.network import NetworkAllowlist

# Betterborg build the container installs. A commit rather than a release, so
# a result names the exact code that produced it. It must be on the remote.
BETTERBORG_COMMIT = "288504478509071df931e944fbdf9e88f3b658c7"
BETTERBORG_REPO = "https://github.com/betterborg/betterborg-cli"

_APP = "/app"
_STATE = "/logs/agent/betterborg"
_VENV = "/opt/betterborg-venv"
_BB = f"{_VENV}/bin/betterborg"
# Trust store and XDG state must live outside the repository under test:
# Betterborg refuses to trust a workspace whose trust store is inside it.
_HOME = "/opt/betterborg-home"
# Betterborg's own configuration, prompts, PRDs, tasks and score live here
# rather than in the repository. Everything it writes would otherwise land in
# the working tree and be graded as the change under review: run 12's patch
# was 892 lines of scaffolding and no solution.
_TRACKED = "/opt/betterborg-tracked"


class _FailedStep:
    """Stands in for a step whose command never returned a result."""

    return_code = 1
    stdout = ""

    def __init__(self, detail: str) -> None:
        self.stderr = detail


class BetterborgPierAgent(Codex):
    """Drives Betterborg instead of `codex exec`, reusing Codex install/auth."""

    # Pier constructs one agent per trial, so a step log held here is that
    # trial's alone. A module-level one is shared by every trial the
    # process runs and each report then carries its predecessors' steps.
    @property
    def _steps(self) -> list[dict[str, object]]:
        log = getattr(self, "_step_log", None)
        if log is None:
            log = []
            self._step_log = log
        return log

    _base_sha: str = ""

    @staticmethod
    def name() -> str:
        return "betterborg-pier"

    def version(self) -> str | None:
        return f"betterborg-{BETTERBORG_COMMIT[:12]}"

    def network_allowlist(self) -> NetworkAllowlist:
        # Codex's own allowlist covers the model gateway. PyPI is not needed at
        # run time: Betterborg is installed during the build step below.
        base = super().network_allowlist()
        return NetworkAllowlist(domains=[*base.domains, ".chatgpt.com", ".openai.com"])

    def install_spec(self) -> AgentInstallSpec:
        # Build time still has network, so Betterborg is installed here rather
        # than in setup(), which runs under the task's no-network policy.
        spec = super().install_spec()
        install_betterborg = (
            "set -eux; "
            "if command -v apt-get >/dev/null 2>&1; then "
            "  apt-get update; "
            "  DEBIAN_FRONTEND=noninteractive apt-get install -y "
            "    python3 python3-venv python3-pip git; "
            "elif command -v apk >/dev/null 2>&1; then "
            "  apk add --no-cache python3 py3-pip git; "
            "fi; "
            f"python3 -m venv {_VENV}; "
            f"{_VENV}/bin/python -m pip install --quiet --upgrade pip; "
            f"{_VENV}/bin/python -m pip install --quiet "
            f'"betterborg @ git+{BETTERBORG_REPO}@{BETTERBORG_COMMIT}"; '
            f"{_BB} version"
        )
        # Pier installs codex as the `agent` user through nvm, then symlinks it
        # as root, where `which codex` finds nothing. Glob both homes instead so
        # a root-run Betterborg can actually see the binary.
        expose_agent_bins = (
            "set -eu; "
            "for bin in node codex; do "
            '  dest="/usr/local/bin/$bin"; '
            "  for candidate in "
            "    /root/.nvm/versions/node/*/bin/$bin "
            "    /home/agent/.nvm/versions/node/*/bin/$bin "
            "    /root/.local/bin/$bin /home/agent/.local/bin/$bin "
            "    /usr/local/bin/$bin /usr/bin/$bin; do "
            '    if [ -x "$candidate" ]; then '
            '      [ "$candidate" != "$dest" ] && ln -sf "$candidate" "$dest"; '
            "      break; "
            "    fi; "
            "  done; "
            "done; "
            "codex --version"
        )
        return AgentInstallSpec(
            agent_name=spec.agent_name,
            version=spec.version,
            steps=[
                *spec.steps,
                InstallStep(user="root", run=expose_agent_bins),
                InstallStep(user="root", run=install_betterborg),
            ],
        )

    async def _step(
        self,
        environment: BaseEnvironment,
        label: str,
        command: str,
        *,
        cwd: str = _APP,
        timeout_sec: int = 1800,
    ) -> object:
        """Run one command, record whether it worked, and keep going."""
        # build_process_env merges the job's agent env, and agent_process_env is
        # where a filtered-egress environment scopes in its proxy variables. A
        # bare dict here replaces both, leaving the agent unable to reach the
        # model gateway at all: it blocks with no CPU and no connection.
        env = self.build_process_env(
            {
                "HOME": _HOME,
                "XDG_STATE_HOME": f"{_HOME}/state",
                "XDG_CONFIG_HOME": f"{_HOME}/config",
                "XDG_CACHE_HOME": f"{_HOME}/cache",
                "CODEX_HOME": self._REMOTE_CODEX_HOME.as_posix(),
                "PATH": f"{_VENV}/bin:/usr/local/bin:/usr/bin:/bin",
                # The task container is already the boundary, and Codex's own
                # read-only sandbox cannot start inside it: bubblewrap needs an
                # unprivileged user namespace that the default seccomp profile
                # refuses. Measured, not assumed. Without this every command
                # the agent runs fails and it plans without reading the repo.
                "BETTERBORG_SANDBOX": "host",
                "BETTERBORG_HOME": _TRACKED,
            }
        )
        env = environment.agent_process_env(env) or env
        # A step that times out raises, and the report is the only account of
        # the run that survives the container. Record the failure and carry on
        # so the steps after it, and the report itself, still happen: a run
        # that died at execute is otherwise indistinguishable from one that
        # never reached it.
        try:
            result = await environment.exec(
                command,
                cwd=cwd,
                env=env,
                timeout_sec=timeout_sec,
                user="root",
            )
        except Exception as exc:  # noqa: BLE001 - recorded, not handled
            result = _FailedStep(f"{type(exc).__name__}: {exc}")
        code = getattr(result, "return_code", getattr(result, "exit_code", None))
        self._steps.append(
            {
                "step": label,
                "command": command,
                "exit_code": code,
                "ok": code == 0,
                "stdout": (getattr(result, "stdout", "") or "")[-4000:],
                "stderr": (getattr(result, "stderr", "") or "")[-4000:],
            }
        )
        self._write_report()
        return result

    def _write_report(self) -> None:
        """Keep the on-disk report current after every step.

        Best effort: the report is a diagnostic, and failing to write one must
        never be what ends a run that is otherwise working.
        """
        try:
            path = Path(self.logs_dir) / "pier-report.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"base": self._base_sha, "steps": self._steps}, indent=2)
            )
        except Exception:  # noqa: BLE001 - a diagnostic, never a failure
            pass

    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)
        await self._step(
            environment,
            "prepare-state",
            f"mkdir -p {_STATE} {_HOME}/state {_HOME}/config {_HOME}/cache "
            f"{_TRACKED}",
            cwd="/",
        )
        # The repository's Makefile declares `docker build` as a preparation
        # step, and betterborg runs the preparation a repository declares
        # rather than guessing which parts matter. This container has no
        # daemon and the task is Go source, so the image is irrelevant to it:
        # a recorded no-op keeps the declared step satisfiable without
        # pretending the run built anything. It logs every call so a run that
        # did depend on a real image is still diagnosable.
        await self._step(
            environment,
            "docker-stub",
            "printf '%s\\n' '#!/bin/sh' "
            "'echo \"[docker-stub] skipped: $*\" "
            ">> /logs/agent/betterborg/docker-stub.log' "
            "'exit 0' > /usr/local/bin/docker && chmod +x /usr/local/bin/docker "
            "&& docker build -t probe . && cat /logs/agent/betterborg/docker-stub.log",
            cwd="/",
        )
        # The coding phase requires each task to leave a commit behind, and
        # unlike the merge phase it supplies no identity of its own. The image
        # configures none at any level, so without this every coding task ends
        # with git refusing to commit and no attestation to show for it.
        await self._step(
            environment,
            "git-identity",
            'git config --global user.email "betterborg@example.invalid" && '
            'git config --global user.name "Betterborg" && '
            "git config --global --get user.email",
            cwd="/",
        )
        # Codex creates CODEX_HOME and injects auth.json inside its own run(),
        # which this agent replaces, so that setup has to happen here instead.
        codex_home = self._REMOTE_CODEX_HOME.as_posix()
        secrets = self._REMOTE_CODEX_SECRETS_DIR.as_posix()
        auth_target = f"{secrets}/auth.json"
        await self._step(
            environment,
            "codex-home",
            f"mkdir -p {codex_home} {secrets}",
            cwd="/",
        )
        auth_source = self._resolve_auth_json_path()
        if auth_source is None:
            self._steps.append(
                {
                    "step": "codex-auth",
                    "ok": False,
                    "stderr": "no auth.json resolved; set CODEX_FORCE_AUTH_JSON=1",
                }
            )
            return
        await environment.upload_file(auth_source, auth_target)
        await self._step(
            environment,
            "codex-auth",
            f"ln -sf {auth_target} {codex_home}/auth.json && "
            f"test -s {codex_home}/auth.json && echo auth-linked",
            cwd="/",
        )
        # No config is written here on purpose. `init` generates it with the
        # repository identity and a required top-level version, and refuses to
        # overwrite an existing file, so a hand-written one breaks the run.
        # Only `codex` is installed, so native-first selection lands on it.

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        base = await self._step(
            environment, "base-commit", "git rev-parse HEAD"
        )
        base_sha = (getattr(base, "stdout", "") or "").strip()
        self._base_sha = base_sha

        # The Pier instruction is the PRD, written through unaltered. A note
        # restating the plan's phase-naming rule used to be appended here,
        # because the schema enforced a shape no prompt stated; the Architect
        # states it itself now, and anything added here is contamination of
        # the task contract.
        await self._step(
            environment,
            "write-prd",
            f"mkdir -p {_STATE} && cat > {_STATE}/prd.md <<'BBEOF'\n"
            f"{instruction}\nBBEOF",
            cwd="/",
        )

        # The repository's docs install rewrites its lockfile, adding "peer"
        # annotations, and Betterborg refuses an environment command that
        # dirties the checkout: anything preparation writes into the working
        # tree would land in the graded diff as if the run had authored it.
        # `npm ci` installs the identical tree from the same lockfile and does
        # not write it back, so a plain install becomes that form. Named
        # packages, other subcommands and a directory with no lockfile run
        # unchanged, and a lockfile out of sync with package.json falls back
        # rather than failing the run. Disabling the lockfile instead would
        # have npm ignore the pins and resolve fresh, which changes the
        # dependency versions the repository under test is given. Both halves
        # measured in this image, not assumed.
        # The shim is written with the real npm's path baked in, so it can
        # delegate without finding itself on PATH.
        shim_body = r"""cat > /usr/local/bin/npm.new <<SHIM
#!/bin/sh
case "\$1" in
  i|install)
    shift
    for a in "\$@"; do case "\$a" in -*) ;; *) exec $REAL install "\$@";; esac; done
    if [ -f package-lock.json ]; then
      $REAL ci "\$@" && exit 0
      echo "[npm-shim] npm ci failed; falling back to npm install" >&2
    fi
    exec $REAL install "\$@"
    ;;
esac
exec $REAL "\$@"
SHIM
"""
        await self._step(
            environment,
            "npm-shim",
            'REAL=$(command -v npm) && ' + shim_body
            + "chmod +x /usr/local/bin/npm.new && "
            "mv /usr/local/bin/npm.new /usr/local/bin/npm && "
            'echo "shimmed npm -> $REAL" && npm --version',
            timeout_sec=300,
        )
        await self._step(environment, "version", f"{_BB} version")
        await self._step(environment, "trust", f"{_BB} trust --yes")
        # `init` writes the config only after adapter selection succeeds, and
        # then immediately runs analysis with it, so there is no moment between
        # the two to lower the effort. It does accept an existing config, so
        # write a complete valid one up front: version, a repository identity,
        # and the stage tables it would have generated.
        # The sanity gate is off because the grader is the judgement that
        # counts here, and a repository check that fails for an environment
        # reason would otherwise discard a reviewed, merged task. Preparation
        # is optional for the same reason one layer earlier: an install this
        # analysis got wrong should cost the agent its tooling, not the task.
        seed = (
            "import uuid, subprocess, pathlib\n"
            "branch = subprocess.run("
            "['git','-C','/app','rev-parse','--abbrev-ref','HEAD'],"
            "capture_output=True, text=True).stdout.strip() or 'master'\n"
            "stages = ['analysis','requirements','architect','tech_lead','pm',"
            "'supervisor','coding','review','merge']\n"
            "body = 'version = 1\\n\\n[repository]\\n'\n"
            "body += 'id = \"%s\"\\n' % uuid.uuid4()\n"
            "body += 'default_branch = \"%s\"\\n\\n' % branch\n"
            "body += '[planning]\\nreview_rounds = 8\\n"
            "decomposition_rounds = 6\\n\\n'\n"
            "body += '[execution]\\njobs = 4\\nreview_passes = 5\\n"
            "sanity = false\\npreparation = \\'optional\\'\\n\\n'\n"
            "body += '[agents.defaults]\\nadapter = \"codex\"\\n"
            "model = \"gpt-5.6-sol\"\\neffort = \"low\"\\n\\n'\n"
            "body += ''.join('[agents.%s]\\n\\n' % s for s in stages)\n"
            f"p = pathlib.Path('{_TRACKED}'); p.mkdir(parents=True, exist_ok=True)\n"
            "(p / 'config.toml').write_text(body)\n"
            "print(body)\n"
        )
        await self._step(
            environment,
            "seed-config",
            f"cat > /tmp/seed.py <<'BBPY'\n{seed}BBPY\npython3 /tmp/seed.py",
            timeout_sec=300,
        )
        # Prove the model gateway is reachable before committing to a long
        # analysis. Without this a broken egress is indistinguishable from a
        # slow stage until the timeout fires.
        probe = await self._step(
            environment,
            "egress-probe",
            # Unpiped: a pipeline reports the exit code of its last command,
            # so `| tail` reported success for a failed probe and defeated the
            # abort below.
            "codex exec --skip-git-repo-check --ephemeral "
            "-c model_reasoning_effort=low 'Reply with the single word: ready'",
            timeout_sec=180,
        )
        probe_code = getattr(probe, "return_code", getattr(probe, "exit_code", 1))
        if probe_code != 0:
            self._steps.append(
                {
                    "step": "abort",
                    "ok": False,
                    "stderr": "egress probe failed; skipping init to avoid a "
                    "long hang on an unreachable gateway",
                }
            )
        else:
            await self._step(
                environment, "init", f"{_BB} init --yes --json", timeout_sec=2400
            )
        # Codex session recordings are the only evidence of what analysis did.
        # Without them a timeout cannot be told apart from a hang.
        await self._step(
            environment,
            "save-codex-sessions",
            f"cp -r {self._REMOTE_CODEX_HOME.as_posix()}/sessions "
            f"{_STATE}/ 2>/dev/null; "
            f"find {_STATE} -name '*.jsonl' | head -5; "
            f"du -sh {_STATE} 2>/dev/null",
            cwd="/",
        )
        # The PRD is already written, so adoption is the whole point: no
        # interview, no agent, no terminal. Deliberately unpiped, because a
        # pipeline reports the exit code of its last command and would hide a
        # failure here behind a successful `tail`.
        await self._step(
            environment,
            "create",
            f"{_BB} create benchmark --prd {_STATE}/prd.md --adopt --yes",
            timeout_sec=3600,
        )
        # Unattended: nobody is at a terminal, so the Architect is told to
        # settle its own questions rather than ask them. Without this a single
        # genuine question ends the run holding a plan it already reasoned out.
        # The grep is belt and braces: the pinned build exits non-zero on a
        # blocked gate, but the build before it exited 0 either way, and the
        # output check is what makes this step honest against both.
        await self._step(
            environment,
            "plan",
            f"{_BB} plan start benchmark --yes --unattended > /tmp/plan.out 2>&1; "
            "rc=$?; cat /tmp/plan.out; test $rc -eq 0 && "
            "! grep -q 'Planning blocked' /tmp/plan.out",
            timeout_sec=3600,
        )
        # The plan and the analysis it rests on are the inputs to every later
        # failure, and neither survives the container.
        await self._step(
            environment,
            "save-plan",
            f"mkdir -p {_STATE}/planning && "
            f"cp {_TRACKED}/state/betterborg.sqlite3 {_STATE}/planning/ && "
            f"{{ cp -r {_TRACKED}/plans {_STATE}/planning/ "
            "|| echo 'NO PLANS DIR'; } && "
            f"{{ cp -r {_TRACKED}/state/planning/context {_STATE}/planning/ "
            "|| echo 'NO CONTEXT DIR'; } && "
            f"find {_STATE}/planning -type f | head -30",
            cwd="/",
        )
        # `plan start` stops at PLAN_APPROVAL_PENDING. Decomposition into
        # executable tasks happens here, and `execute` has nothing to run
        # without it.
        await self._step(
            environment,
            "approve",
            f"{_BB} plan approve benchmark --yes > /tmp/approve.out 2>&1; "
            "rc=$?; cat /tmp/approve.out; test $rc -eq 0 && "
            "! grep -q 'decomposition blocked' /tmp/approve.out",
            timeout_sec=3600,
        )
        # `approve` is where the plan becomes a task graph, so the snapshot
        # taken before it holds no tasks. Take a second one here: the graph is
        # the input to every execution failure, and it does not survive the
        # container.
        await self._step(
            environment,
            "save-tasks",
            f"mkdir -p {_STATE}/tasks && "
            f"cp {_TRACKED}/state/betterborg.sqlite3 {_STATE}/tasks/ && "
            f"ls -l {_STATE}/tasks",
            cwd="/",
        )
        await self._step(
            environment,
            "execute",
            f"{_BB} execute benchmark --auto-execute",
            timeout_sec=7200,
        )
        # Why a task blocked lives in the database and in the per-attempt
        # artifacts, and both die with the container. The two snapshots above
        # are taken before execution, so neither holds anything about it: a run
        # that blocked at review is otherwise reported as a bare exit code.
        # Best effort on the artifacts, exact on the database.
        await self._step(
            environment,
            "save-execution",
            f"mkdir -p {_STATE}/execution && "
            f"cp {_TRACKED}/state/betterborg.sqlite3 {_STATE}/execution/ && "
            f"{{ cp -r {_TRACKED}/state/artifacts {_STATE}/execution/ "
            "|| echo 'NO ARTIFACTS DIR'; } && "
            f"find {_STATE}/execution -type f -size +2M -exec sh -c "
            "'tail -c 2000000 \"$1\" > \"$1.tail\" && mv \"$1.tail\" \"$1\"' _ {} \\; "
            "; "
            f"du -sh {_STATE}/execution && "
            f"find {_STATE}/execution -type f | wc -l",
            cwd="/",
        )
        # Betterborg delivers onto its own project branch, advancing it once
        # per merged task. The graded diff is taken from the checked-out
        # branch, which never moved, so the work has to be brought onto it or
        # the run reads as having changed nothing.
        await self._step(
            environment,
            "land",
            "git rev-parse --abbrev-ref HEAD && "
            "git log --oneline -1 project/benchmark && "
            "git merge --ff-only project/benchmark || "
            "git -c user.email=a@b -c user.name=bb merge --no-edit "
            "project/benchmark",
        )
        # A task that blocks keeps its commits on its own branch and nothing
        # merges them, so a review budget that runs out discards work that was
        # coded, reviewed and fixed. The grader is the judgement here, so the
        # adapter lands what the run built and abandons only what will not
        # merge. Branch order is the stem order the Project Manager numbered,
        # which is the order the tasks were meant to run in.
        await self._step(
            environment,
            "land-blocked",
            "for branch in "
            "$(git for-each-ref --format='%(refname:short)' "
            "refs/heads/betterborg-tasks/); do "
            'if git merge-base --is-ancestor "$branch" HEAD; then '
            'echo "[land] already landed: $branch"; '
            "elif git -c user.email=a@b -c user.name=bb merge --no-edit "
            '"$branch"; then '
            'echo "[land] landed: $branch"; '
            "else "
            "git merge --abort || true; "
            'echo "[land] conflict, abandoned: $branch"; '
            "fi; "
            "done; "
            "git status --porcelain",
        )
        await self._step(
            environment,
            "diff",
            "git add -A && git -c user.email=a@b "
            "-c user.name=bb commit -q -m benchmark "
            f"|| true; git diff --stat {base_sha} HEAD",
        )
        # The task declares the artifact pier collects, and separately declares
        # the command that produces it. The installed pier reads an older task
        # schema and drops that second declaration, so the file is never
        # written and the verifier grades a pristine checkout. Producing a
        # declared artifact is harness work, so the adapter writes it here.
        await self._step(
            environment,
            "model-patch",
            f"mkdir -p /logs/artifacts && "
            f"git diff --binary {base_sha} HEAD > /logs/artifacts/model.patch && "
            f"wc -c < /logs/artifacts/model.patch",
        )

        # Uploaded rather than written through a command: the report grew
        # past the argument-length limit and the step that carried it failed
        # with E2BIG, losing the account of the run it existed to keep.
        self._write_report()
        try:
            await environment.upload_file(
                Path(self.logs_dir) / "pier-report.json", f"{_STATE}/report.json"
            )
            self._steps.append({"step": "save-report", "ok": True})
        except Exception as exc:  # noqa: BLE001 - a diagnostic, never a failure
            self._steps.append(
                {
                    "step": "save-report",
                    "ok": False,
                    "stderr": f"{type(exc).__name__}: {exc}",
                }
            )
        self._write_report()

    def populate_context_post_run(self, context: AgentContext) -> None:
        # Codex's trajectory parsing expects its own session layout; this run
        # produces Betterborg's, so skip it rather than fail the trial.
        try:
            super().populate_context_post_run(context)
        except Exception:
            pass
