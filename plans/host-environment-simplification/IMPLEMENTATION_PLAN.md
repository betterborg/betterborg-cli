# The host is the environment

Betterborg-cli runs on a developer's machine, against a repository that already
builds there. It reconstructs that repository's build recipe from analyzer
evidence, runs it in a synthesized environment with a private home and private
package caches, caches the result under a content fingerprint, and refuses to
start a task when the reconstruction disagrees with the repository. Five changes
remove the reconstruction and keep the parts that outlive it.

The design being removed belongs to a different product. In the cloud, a task
runs in a microVM that boots from a base image with nothing installed, so
something has to install the repository's dependencies and cache the result;
`SandboxEnvironmentSpec`, the environment cache and the descriptor contract all
serve that. The cloud's own host executor, `HostSandbox`, is a bare subprocess
runner and does none of it. The CLI has no mode but the host, and inherited the
machinery without inheriting the reason for it.

What the machinery costs is measurable: `preflight.py`, `environment.py` and
`compose.py` are 4,396 lines against 3,323 for the whole coding, review, merge
and sanity pipeline they serve. In a five-task benchmark sweep, three runs
produced no code at all, and every one of those three died in this subsystem.

One rule governs everything the five changes strand: a symbol whose last caller
goes is deleted with its export, and a durable payload key whose last writer
goes is deleted with it. Nothing is left exported with nothing to call it.

Nothing here preserves what an earlier version wrote. Betterborg-cli has three
users on one team, so a schema change drops what it no longer needs without
repairing the rows it orphans, and a run left in a strange state by an earlier
version is cleared by hand or by starting a fresh state directory. That
assumption is load-bearing in Stage 4 and it is the only reason these changes
carry no migration logic beyond dropping a table.

Two things are not part of what goes.

**Worktree isolation stays.** A task runs in a private git worktree so an agent
cannot dirty the operator's checkout, and so `[execution] jobs` above one is
possible at all. Worktrees are cheap — they share the object store.

**The record that a worktree was prepared stays.** Two things read it. The
reuse check skips preparation when a worktree is already prepared for the same
key, which is what makes preparation happen once across a claim, a resume and a
re-claim. `require_ready_worktree` refuses to invoke a coding, review or merge
agent unless the store holds a completed `materialize` attempt and the
checkout's marker agrees with it, which stops an agent running against a
half-prepared tree. Both survive. What narrows is the key they share.

## Stage 1: Betterborg runs repository commands in the operator's environment

**Goal**: Every process Betterborg runs against the repository — a preparation
command, an agent, a sanity catalog command — runs in the environment the
operator would have run it in.

Those processes share one environment object today, and it is synthetic: six
variables from an allowlist (`LANG`, `LC_ALL`, `PATH`, `PATHEXT`, `SYSTEMROOT`,
`TMPDIR`), a `HOME` inside a cache directory named by the content fingerprint,
and package-manager caches redirected into that same directory. On a machine
with no user and no caches that is necessary. On a developer's machine it
substitutes an environment nobody has run the repository in for one that
demonstrably works, and cold-starts caches that are already warm. A repository
whose toolchain lives under the real home — nvm, mise, corepack shims, a global
package-manager prefix — is invisible from it, and the build fails in a way the
operator cannot reproduce by hand.

The environment stays a single object, and that object becomes the operator's.
Agents are not carved out: an agent's whole job is to run the repository's build
and tests, so preparing a worktree against the operator's package store and then
running its tests against an empty one is the same defect in a new place.

Cache redirection is not only the package-manager variables. The function that
builds them sets `BETTERBORG_ENVIRONMENT_ROOT` and the three `XDG_*` directories
first, before it looks at what the repository declares, so every command and
every agent runs today with an XDG cache inside the fingerprint directory
whatever the analysis says. All of it goes together — the function and its
export with it — because half a removal leaves the agent's tests reading an
empty XDG cache instead of an empty package store.

Three synthesized variables are not cache redirection and are decided on their
own merits. `GIT_TERMINAL_PROMPT=0` stays: preparation commands run with
no timeout and inherit a stdin, so a command that reaches a private dependency
would otherwise block on a credential prompt with nothing to end it.
`PIP_DISABLE_PIP_VERSION_CHECK` goes with the rest — it changes output, not
behaviour, and the operator does not set it. `CI=true`, which the sanity gate
sets for its catalog commands alone, stays: a check running as part of a
judgement is a CI run whatever the operator's shell says.

Secrets keep a rule, and the rule gets weaker in a way worth stating. Secret
values are read out of the operator's environment, and commands run today with a
*closed* environment, so a command sees a declared secret only when its stage
declares it, and sees no other credential at all. Once the environment is
inherited, only the first half can survive: a stage still does not see a
**declared** secret name it did not declare, but every undeclared credential in
the operator's shell now reaches every command — including the same credential
exported under a second name. That is the price of running commands the way the
operator runs them.

It is not the same exposure the operator's own run has, and the difference is
the part worth stating. Betterborg persists what these commands print: a
preparation command's stdout and stderr go into the environment attempt's
result, and a catalog command's last four thousand characters of each go into
the `sanity.completed` event. Both are redacted against the values of
*declared* secrets only. Today an undeclared credential cannot be in a
command's environment, so it cannot be echoed into that store; afterwards it
can, and one verbose failing install is enough. The operator's own run leaves
such a value in scrollback; this leaves it in the state directory.

Keeping the surviving half needs a new operation. Today a stage's secrets are
added to a closed environment; afterwards they must also be *subtracted* from an
inherited one, and two call sites need it — the environment manager, and the
sanity gate, which composes its command environment on its own. Addition and
subtraction are one operation with one owner: the function that already computes
a stage's secret environment takes the base environment and returns it with that
stage's secrets present and every other declared secret absent. Neither call
site filters on its own; two independently written filters would disagree the
next time the secret model changes. It returns the environment and nothing else:
the mask values it also returns today are discarded at both call sites, which
take their masks from the declared-secret helper instead.

Two existing suites are rewritten with this stage rather than merely adjusted.
The harness the environment suite runs on is built from the two things this
stage and the next remove. Its fake package manager aborts under `set -u` unless
`XDG_CACHE_HOME` is set, and writes the audit logs that suite counts installs by
into that directory; the counting helpers find those logs by walking the cache
root, and the fixture passes a cache root and a preparation root into the
environment manager. All of it goes, and what replaces the counting is the
`prepare` and `materialize` attempts recorded in the store, and the commands a
recording runner saw. The fixture's manager injects no runner, so the stub runs
as a real subprocess and several tests depend on its filesystem effects; a
recording runner that appends the argv and then delegates to the real one keeps
both, which is the idiom the sanity suite already uses. Until that substitution
is made, every environment command in the file fails for a reason unrelated to
what its test is about — including two that read the cache directly, one
counting preparations under the cache root and one reading a file out of a
materialization's cache path.

The sanity suite asserts the inverse of this stage: it injects an operator
variable and asserts no command sees it, then reads
`HOME` and four package-manager cache paths out of the command environment and
requires each to sit under the fingerprint directory. The undeclared-variable
assertion inverts — that exposure is what this stage accepts. The three secret
assertions stay exactly as they are: an agent-scoped secret and a secret
declared for another stage are still excluded by the subtraction, and a value
supplied to the run but declared nowhere is in neither the declaration nor the
injected operator environment, so nothing can put it there. The cache-shaped
assertions all go, including the one reading a cache path out of the recorded
attempt — with the operator's environment inherited there is no synthetic `HOME`
to subscript, and the injected environment has none of its own. The test
covering the cache-variable function goes with the function.

**Success Criteria**:
- A preparation command, an agent and a sanity catalog command all inherit the
  operator's environment, including `HOME` and package-manager cache locations.
- Betterborg sets no synthetic `HOME` and none of the variables the cache
  environment produces, including `BETTERBORG_ENVIRONMENT_ROOT` and the `XDG_*`
  directories it sets whatever the repository declares.
- `GIT_TERMINAL_PROMPT=0` is still set for every repository-declared command.
- A command's environment excludes every declared secret name its stage does not
  declare, and one function decides that for every call site.

**Tests**: only the first can fail against today's code. The other three pass
against the closed environment as they will against the inherited one, and are
kept because they pin what must stay true afterwards.

- A preparation command, a sanity catalog command and an agent each observe a
  variable set in the operator's environment and absent from the six-name
  allowlist, and each observe the operator's `HOME` and `XDG_CACHE_HOME`. Both
  of those are overridden today and are the halves that can fail for an agent,
  whose adapter already merges every other operator variable.
- A secret declared for one stage is absent from another stage's environment
  **while present in the operator's environment** — the case that distinguishes
  a subtractive filter from an allowlist that never contained the value.
- The subtraction holds for a sanity catalog command.
- A command Betterborg runs has `GIT_TERMINAL_PROMPT=0`.

**Status**: Complete

## Stage 2: A worktree is prepared by running the repository's own command

**Goal**: A task worktree becomes runnable by running the repository's declared
preparation command in that worktree, and by nothing else.

Preparation is a content-addressed cache. Declared file contents are hashed into
a fingerprint, the fingerprint names a cache directory, a `prepare` stage
populates it in a disposable worktree whose outputs are deliberately discarded,
a `materialize` stage runs per worktree, and a marker inside the cache records
that the cache is populated. The cache exists so that N worktrees do not each
pay a full install where no shared cache exists. On a host there is already a
shared cache: the operator's.

Removing it leaves one install per worktree against that warm cache. The
disposable preparation pass goes with the cache, because populating a shared
thing is the only work it ever did — all of it, from the environment manager's
preparation entry point up through the task runtime's public wrapper, the
pre-dispatch block in the service that calls it inside the setup lease, and the
runtime double the service tests use to satisfy that seam. With it goes the lock
that serialized access to the shared directory. Nothing replaces that lock: with
`jobs` above one, N installs run concurrently against the operator's store,
which is what N terminals would do, and whether they interfere is a property of
npm, pip or cargo rather than of anything Betterborg builds.

Stages 1 and 2 are one release. Between them, preparation still populates a
cache directory whose only remaining occupant is its marker, so a run in that
intermediate state pays a full install for nothing. Nothing breaks; there is no
reason to ship it.

The analysis keeps declaring both command lists and the existing selection rule
is unchanged: the materialize list when present, otherwise the prepare list.
That rule gets one owner, for the same reason the secret filter does: a free
function in `preflight.py` taking the two lists and returning the selected one.
Three sites call it — the materialization that executes the commands, the
reuse key, and Stage 5's check of which programs the host must be able to run —
and none restates it. It belongs in `preflight.py` rather than beside the
materialization because the dependency runs environment → preflight, and it
takes two sequences rather than a plan because preflight needs the answer before
it has a validated plan to hang it on.

The reuse key holding the selected list and not both is what stops an edit to a
command that will never run from re-installing every worktree.

Two further sites look like they should narrow and cannot, because they are
keyed by stage name and both declared lists carry the same one. Preflight stamps
every prepare and every materialize command with the stage `environment`, and
the analyzer's schema for those commands admits no stage of its own, so the set
of stages that reach a run is identical whichever list is selected. The builder
that composes a stage's command environment and the gate that decides which
secrets reach the run are therefore left alone: for any analysis Betterborg can
produce, narrowing them would change nothing.

No analysis is rewritten and no schema changes — but one behaviour does change
for an analysis declaring both lists. Today both execute, prepare in the
disposable worktree and materialize per checkout; afterwards only the selected
list runs. After Stage 1 the prepare list's only surviving effect is warming
caches the operator already has, which makes the loss small rather than nothing.

The per-checkout materialization marker is not the cache's marker and stays. It
is half of what `require_ready_worktree` checks, and it is written in one of two
places depending on whether Betterborg's state lives inside the repository.

Reuse narrows rather than moves. The key today mixes six things: the contents of
every declared environment file, both command lists, the declared package
managers, the repository root, every declared secret's name, scope and users,
and every program preflight resolved, with its version — which is not only the
declared toolchains but every catalog command's program and every declared
package manager. Afterwards it carries the selected command list alone. The
repository root goes with the rest: it was there to stop two repositories
sharing one machine-local cache directory, and with no shared directory the key
is only ever compared against this repository's own attempt rows and marker —
and `materialize_claimed_task` already refuses a plan belonging to another
repository. Six kinds of edit stop invalidating a prepared worktree: a
package-manager list, a secret's scope, a resolved program's version, a catalog
command whose program resolves, the repository's own path, and the contents of a
declared file.

The last of those six has a cost worth naming. A task whose agent adds a
dependency and is then interrupted re-claims against what is already installed,
and installs what it added the same way it did the first time. The merged-tip
case is not left to that rule.

What holds the key keeps its shape. The environment attempt's `fingerprint`
column is `NOT NULL` and `require_ready_worktree` compares the marker's text to
it, so that column and that marker hold the preparation key; renaming a column
would cost a migration for nothing.

The names Betterborg controls do change, because a field naming a cache or a
fingerprint while holding neither is how the next reader gets it wrong: the
contract version constant, `EnvironmentMaterialization`'s field and the
`sanity.completed` event key all name the preparation key. Two payloads shed
the cache with it — `EnvironmentMaterialization` drops `cache_path` and
`preparation_reused`, and the environment attempt's result JSON drops
`cache_path` and `prepared_before_dispatch` on both the success and the failure
path. `preparation_reused` and `prepared_before_dispatch` lose their producer
with the disposable pass; `cache_path` is produced by the materialization itself
and goes with the cache directory it names.

`environment_fingerprint`, the public function that computes the old key, is
nothing but the declared-file read and the digest that both go here. It goes
with them, export included, rather than outliving the code it is made of.

The sanity gate asks for a fresh tree and gets one unconditionally. It prepares
the merged tip inside the repository lock before running the catalog, because
the catalog judges that tip and a digest of the selected commands cannot tell a
merged tree from the tree it replaced. Asking is an argument the gate passes
through the materialization entry point to the reuse check, because the marker
written before coding otherwise matches at sanity and skips the install.

That install is not free and is not parallel. Today the gate's call almost
always reuses and runs nothing; afterwards every task pays a full install there,
inside the repository lock the gate already holds — so with `jobs` above one,
sanity installs run one at a time even though materialization installs do not.

Two things the pass leaves behind go with it. The filter that separates
run-owned cache attempts from task-local ones becomes a no-op once nothing
writes that kind, so it goes rather than surviving behind a comment that has
stopped being true; the only rows it would still classify are ones an earlier
version wrote, and those are not this plan's problem. The attempt lookup is the
same case — every call becomes `materialize` for one task, so its optional task
filter and the docstring explaining when to omit it describe a distinction that
no longer exists.

The schema is where this stops. The attempt table's nullable `claim_id` and the
constraint permitting it only for a `prepare` attempt stay, because both remain
satisfiable once nothing writes that kind and relaxing a constraint costs a
migration that buys nothing.

**Success Criteria**:
- A claimed task worktree becomes runnable by the declared preparation command
  run in that worktree and by nothing else.
- The sanity gate prepares the merged tip before running the catalog, whatever
  the reuse rule would otherwise say.
- No fingerprint of declared file bytes, no shared cache directory and no
  disposable preparation worktree is computed, written or read, and
  `environment_fingerprint` is gone with its export.
- The environment attempt's `fingerprint` column holds the preparation key, and
  no durable payload carries a cache path.
- A completed preparation is not repeated for the same worktree and the same
  declared commands across a resume or a re-claim.
- The materialization marker and the completed `materialize` attempt still agree
  before any agent runs.
- A repository that declares no preparation command reaches coding without one.

**Tests**: four of these can fail against an implementation that did nothing —
the prepare command never running, the re-claim pair, the gitignored lockfile,
and the absence of the cache directory and its marker. The gitignored-lockfile
test is the one that fails hardest today: it is the motivating defect, and today
the declared file is absent from every task worktree, so every task blocks.

The other three are guards. The merged-dependency test passes today because the
current key hashes declared file digests, and fails against the realistic
partial implementation — a narrowed key without the forced preparation. The
no-preparation-command test and the marker-conjunction pair both hold against
today's code unchanged: they pin behaviour this stage preserves rather than
introduces.

- A repository declaring both command lists never runs its prepare command at
  all: no `prepare` environment attempt is recorded and the command runner is
  never invoked with it. The assertion has to be about the command not running,
  not about the worktree lacking its output — that output is discarded with the
  disposable worktree today, so a worktree-shaped assertion holds either way. A
  repository declaring only a prepare list is prepared by it; that half passes
  today and guards the fallback.
- A task merging a dependency change is judged by the catalog on the merged
  dependencies.
- A re-claim whose declared environment file changed but whose declared command
  did not reuses the prepared worktree; a re-claim whose declared command
  changed prepares again.
- A repository whose only lockfile is gitignored reaches coding in every task.
- No cache directory and no cache marker is created; the materialization marker
  is still written, in both of its layouts, and an agent still refuses to run
  when it disagrees with the stored attempt.
- A repository declaring no preparation command reaches coding.
- A worktree whose marker disagrees with the computed key is prepared again
  even though a completed `materialize` attempt exists for that key, and a
  worktree whose marker agrees is not — the pair that pins the conjunction
  rather than either conjunct alone.

**Status**: Complete

## Stage 3: Declared environment evidence stops gating the run

**Goal**: No run is refused because a file the analysis named is not where
Betterborg looked for it, or because a version scraped from that file disagrees
with the host.

`environment.files` is enforced in three places that cannot agree. Preflight
requires each declared file to exist in the primary checkout, where an untracked
lockfile is visible. Materialization requires the same path inside a task
worktree, which by construction contains only tracked files — so a repository
that gitignores its lockfile, a normal convention, passes preflight and then
blocks every task it has. Preflight also reads those files as evidence for a
toolchain version pin, and refuses a host whose program does not match a version
string scraped out of them.

All three go. The materialization read dies with the fingerprint in Stage 2,
which is where the blocked task is fixed; the two preflight reads die here. The
version pin is the reconstruction this plan exists to remove, applied to the
operator's own toolchain: the operator's node is the node the repository builds
with, and a version scraped from a file is not better evidence than the build
itself. The `--version` probe that fed the comparison loses its last consumer
and goes with it, and so do the two helpers used only inside the block being
deleted — the one that matches a version string against probe output, and the
one that resolves an evidence file's path.

The declaration stays. It is evidence a person reads in an analysis report,
exactly as declared services and package managers are, and nothing is served by
removing one kind of environment evidence while keeping the others. What stops
is the enforcement.

**Success Criteria**:
- Nothing in the execution path reads the contents of a declared environment
  file, and no host program's version is compared against one. The analyzer
  still reads repository files to produce the declaration.
- Nothing refuses a run because a declared environment file is missing, because
  a toolchain's cited version source is missing, because a version does not
  appear in the file that cites it, or because a toolchain version disagrees
  with the host's program. The first belongs to the declared-file check; the
  other three live inside the version block and go with it.
- `environment.files` remains in the analyzer schema and in the analysis report.

**Tests**:
- A repository declaring a file that does not exist is not refused.
- A repository declaring a toolchain version pin that the host's program does
  not satisfy is not refused. The comparison sits behind four guards and the
  test has to clear all of them or it proves nothing: the pin's file must exist,
  the version string must be non-empty and must literally appear in that file,
  the toolchain's program must resolve on the host, and a validated command must
  invoke it. Clearing only some of them exercises the adjacent "must appear in
  its evidence file" refusal instead. The existing toolchain-version test
  arranges all four and is the one to adapt. Two more change for their own
  reasons: the aggregate-failure test counts the missing-file refusal this stage
  removes among the failures it expects, and the missing-cited-file test asserts
  the *other* file refusal — the one inside the version block, over a
  toolchain's cited source, which need not be a declared environment file at
  all. A third stops meaning anything: the test pinning that
  a version pin on a program the run never invokes does not block exists to
  cover a guard inside the block being deleted, so it passes afterwards for an
  unrelated reason and goes with the guard, and so does the test named for the
  Go version probe's argv branch, whose stub stops being invoked and whose only
  assertion is on the declared version. The three tests asserting a probed
  binary's resolved path are untouched here — what they assert is the path and
  the *declared* version, neither of which comes from the probe — and change in
  Stage 5, with the inventory.

  The cancellation harness is the larger piece of work and belongs to this stage
  and Stage 4 together. That file identifies five ordered direct probes by argv
  shape and parametrizes four tests across all five, including a real-subprocess
  test that requires an interrupt at each one. This stage removes the
  executable-version probe and Stage 4 removes the Compose version and topology
  probes, leaving the two git probes; the harness, its probe-naming helper and
  the parametrization shrink to those two rather than being deleted, and the
  ordinary-probe-failure test loses the refusal it asserts. Those tests also
  pass `external_urls`, which stops existing in Stage 4.
- An analysis report still renders its declared environment files — a guard on
  the reporting path, which reads the analysis and not the validated plan.

**Status**: Not Started

## Stage 4: Service orchestration leaves the CLI

**Goal**: Betterborg does not start, stop or validate a service stack, and
leaves no bookkeeping behind that can strand a task.

`compose.py`, the service and Compose half of preflight, and the service and
Compose fields of the validated plan all serve one capability: starting a
Compose stack per claimed task, validating its topology for isolation, resolving
service URLs, and refusing a service the analyzer inferred rather than named
exactly. On a developer's machine the developer runs their own stack. All of it
is removed here, in one place, so that no later stage has to claim the same
code.

The cost is not zero and both service kinds pay it, in the same coin. A service
is selected because a *surviving catalog command* declares it — one the sanity
gate would run — and both kinds are refused before the run today: a Compose
service when its stack cannot be validated, an external one when the variable
naming its URL is unset or does not hold an absolute URL. Afterwards neither is.
A repository whose tests genuinely need a service stops being refused before any
task is claimed and starts failing at the sanity gate after a full coding,
review and merge spend, with no message naming the stack to start or the
variable to export. Only the remedy the operator is no longer told about
differs.

The bookkeeping goes with the manager, all of it. A claim carrying an
unconfirmed `compose_resources` row is held unreleased today at three points —
terminal transition, run interruption and claim expiry — and the only code that
confirms cleanup and lifts all three lives in the manager being deleted. So the
table goes, in a migration that does nothing but drop it, along with every
accessor that reads or writes it, the three gates that consult it, and the
branch that unblocks a task blocked by a cleanup failure. With no row left to
consult, a claim releases at all three points on its own.

The migration repairs nothing, under the rule this plan opens with. A database
carrying rows from an earlier version can hold a task whose claim was never
released and whose runtime is `blocked` or mid-phase; after the upgrade nothing
will move it, because `blocked` has no outgoing transition and claiming needs
both a `pending` runtime and no unreleased claim. Repairing that in SQL means
reading a durable event to tell a teardown failure from a review failure, and
writing timestamps that satisfy two ordering constraints or the migration rolls
back on every open and the store stops opening at all. For three users the
answer is to clear such a task by hand.

A stack an earlier version left running is consequently not stopped by the
upgrade, and the operator stops it — finding it in `docker compose ls`, which is
where they would look anyway, since Betterborg never showed the table's rows to
a person. That is the honest outcome. The alternatives are keeping enough of the
manager to run `docker compose down`, which is the capability this stage
removes, and recording a cleanup event for containers that are still up, which
puts a false statement in the durable log.

Two things outlive their last callers here and go with them. `path_lock`, the
cross-process file lock, is used only by the Compose lifecycle and by the
preparation lock Stage 2 removes, so its module goes. It did guard something
real: the store's run lease is scoped to a Borg, and one repository can hold
several, so two `betterborg execute` processes on different Borgs of the same
repository both acquire and then share one cache directory under the
repository's state directory. That directory is what `path_lock` serialized —
and Stage 2 removes it, along with the Compose lifecycle that was its only other
user, so nothing is left to serialize. The merge phase's conflict-verification
test uses `path_lock` as a deliberately non-reentrant lock factory to prove the
phase releases before re-acquiring; it keeps proving that with an in-process
non-reentrant lock built in the test.

Preflight's probe environment is the other: its last two callers are the Compose
version and topology probes, so it goes here.

Reconciliation keeps one of its two duties and loses the other. Returning stale
rows so that something can stop them has nothing left to serve; sweeping expired
execution runs is needed whether or not Compose exists, and it is the only sweep
that reaches a Borg other than the one being acquired — acquisition interrupts
an expired run for its own Borg and no other. So the service helper that calls
the sweep survives, calling it for its effect and returning nothing, and the
scheduler keeps invoking it on cancellation and on lost ownership; both stop
being named for a cleanup they no longer do. The sweep and
`interrupt_execution_run` stop returning Compose rows, `HostExecutionResult`
stops carrying a `cleanup` field, and `validate` and `run` stop taking an
`external_urls` argument that never had a producer.

Seven public names go with the code behind them: `ComposeCleanupResult`,
`ComposeStackError`, `ComposeStack`, `HostComposeManager`, `HostService`,
`compose_project_name` and `service_url_environment`. `ComposeResource` leaves
the store's public surface with the table it modelled. Nothing outside tests
reads any of them.

**Success Criteria**:
- No Compose stack is started, stopped or validated, and no preflight refusal
  concerns a service, a Compose file or a topology.
- The validated plan carries no service or Compose field, and no public entry
  point takes `external_urls`.
- A repository declaring services and Compose files executes its tasks.
- No `compose_resources` table, accessor or claim gate remains, and the
  migration that drops the table does nothing else.
- With the gates gone, a claim releases at all three points.
- Expired execution runs are still swept, including a run belonging to a Borg
  nobody is acquiring.

**Tests**:
- A repository whose analysis declares services and Compose files executes its
  tasks with no service started.
- A claim releases on terminal transition, on interruption and on expiry with
  no `awaiting Compose cleanup` reason anywhere. This one cannot fail after the
  stage — with the table gone there is no unconfirmed row to hold a live claim,
  and nothing left that writes that reason — so it is a regression guard on the
  gates having been removed cleanly.
- An expired run belonging to a Borg that nobody acquires is still swept, which
  acquisition alone would not do. This passes today too; it guards against the
  sweep helper being deleted along with the Compose work it used to feed.
- Existing coverage that changes: services are still recorded for a reader; the
  six suites asserting the applied-migration list gain this stage's migration;
  and three suites build their plan fixtures with the service, Compose,
  package-manager and executable fields this stage and Stage 5 remove, so each
  raises on import until its fixture is rewritten. One of them also asserts the
  external service URL a catalog command receives — the value survives, because
  the command inherits it from the operator's environment, but the mechanism
  that injected it does not.

**Status**: Not Started

## Stage 5: Preflight validates what outlives it

**Goal**: Preflight validates the inputs the rest of the run consumes, and
nothing else.

With the cache, the descriptor enforcement and service orchestration gone, most
of what preflight validates has no consumer. What survives it is this.

The **preparation commands**, in two parts that narrow differently. Both
declared lists keep their shape checked — each argument list well formed, each
working directory a real repository-relative path — because a malformed command
is a malformed analysis whichever list is selected, and because the selection
rule reads the shape-checked lists, so shape checking has to precede the
selection rather than depend on it.

What the host must be able to *run* narrows to the list the selection rule
picks. A
program one of those commands needs is required rather than dropped, because a
host that cannot run it fails every task individually instead of refusing the
run once. A program named only by the list that will not run is neither
required nor dropped: nothing will invoke it. The secret gate does not narrow
with it, for the reason Stage 2 gives: both lists share one stage name, so no
analysis can express a secret only the unselected list consumes.

The **sanity catalog**, for the reason it exists: it is the only judge of
whether a merged tree is sound. A catalog command whose program this host lacks
is dropped rather than refused, and a repository with no runnable command left
is refused. The record of what was dropped travels with the run — into the
task's durable state reason and the report a person reads — because a dropped
check cannot fail, and a task that skipped one is not a task that passed it.

**Declared secrets** and their scopes. Five refusals survive, and each names
something the operator or the analysis can change: a name declared twice with
conflicting records; a command referencing a secret nobody declared; a
declaration with an invalid scope or no named user; a secret scoped to the
agents but named by a command that runs; and a secret that reaches the run with
no value in the operator's environment. The last is the one that protects the
spend — without it a missing build-scoped secret stops being a refusal before
the run and becomes a materialization failure that blocks each task in turn.

The gate does not confine what an agent can read, because agent processes
inherit the operator's environment and a declared secret's value comes from
there. The narrower guarantee is the one to state; the wider one was never true.

**Workspace trust**, required before any repository-controlled command runs.
Preflight performs it today and it is independent of everything else here.

Analyzer evidence that no longer has a consumer — declared environment files,
package managers, services — is still recorded in the analysis for a reader and
no longer validated, and the validated plan stops carrying it. Each field loses
its last reader earlier — `environment_files` and `package_managers` with Stage
2's fingerprint, the service and Compose fields with Stage 4, and the resolved
executable inventory with them, its two readers being that fingerprint and the
Compose manager's Docker check — and preflight stops producing it here.
`HostExecutable` goes with the inventory and its export, and so does the loop
that fed it: preflight stops resolving a program for each declared package
manager and each declared toolchain, since those requests block nothing and
their only outputs were the inventory and a key the catalog drop never consults.
The rule they were prose for — an inventory name the host lacks never refuses a
run — stays true and is easier to hold once nothing resolves them. The check
that a program is runnable stays and still produces a refusal; only the record
of what was resolved is dropped.

**Success Criteria**:
- Preflight validates preparation commands, the sanity catalog, declared secrets
  and their scopes, and workspace trust.
- Both declared command lists are still shape-checked; only runnability narrows
  to the selected list.
- A program the selected preparation command needs is still required; a catalog
  program is still dropped, and what was dropped is still reported.
- Preflight computes nothing that has no consumer.
- Every remaining refusal names something the operator or the analysis can
  change.

**Tests**: two of these can fail against an implementation that did nothing —
the runnability narrowing and the absent plan fields. The rest are existing
coverage kept as regression guards, because this stage rewrites the file they
cover — and several of those guards assert on the resolved inventory this stage
removes, so they change with it rather than surviving untouched.

- A host that cannot run the selected preparation command's program is refused,
  naming it; a host that cannot run a program named only by the list that will
  not run is not.
- In the list that will not run, both halves of the shape check still refuse the
  analysis: a malformed argument list, and a working directory that does not
  exist. The second is the one an implementer relaxing the check along with
  runnability would drop.
- A host missing one catalog program drops that check and runs; a host missing
  every catalog program is refused; the dropped check appears in the task's
  state reason and its report.
- A secret scoped to the agents and named by a surviving command is refused,
  naming the secret.
- A secret that reaches the run with no value in the operator's environment is
  refused, naming it.
- An untrusted workspace is refused before any repository-controlled command
  runs.
- A repository declaring environment files, package managers and services —
  including a toolchain and a package manager this host cannot run — is
  accepted, and the plan it validates carries no `environment_files`, no
  `package_managers` and no resolved executables. Acceptance is green once
  Stage 4 lands, as is the absence of the service and Compose fields, which that
  stage removes; these three absences are what this stage adds.

**Status**: Not Started

## Documentation

`docs/commands.md` is the only place a reader learns two things this plan
changes.

One sentence says that the analysis's prepare *and* materialize commands are
never dropped and that a host unable to run one of them is refused. Stage 5
requires only the programs of the list Stage 2 runs, and rewrites that sentence
to name that list. The catalog half of the same rule, and the dropped-command
example above it, are unchanged and stay true.

Another says that the task and environment worktrees Betterborg mints are
siblings of the repository, under `.betterborg-worktrees` and
`.betterborg-environments`. Stage 2 removes the disposable preparation worktree,
so the second directory stops existing and the sentence keeps only the first.

## Decision log

**Worktrees stay.** They keep an agent out of the operator's checkout and make
`[execution] jobs` above one possible. Executing in place was rejected: it would
delete more, but the guard against primary-checkout contamination is worth
keeping independently of parallelism.

**The preparation record stays; only its key narrows.** Rejected: removing the
environment attempt and the per-checkout marker along with the cache. They are
what `require_ready_worktree` checks before every agent phase and what makes
preparation idempotent across resumes; removing them makes every task block
before its first agent runs.

**Reuse is keyed by the selected command list alone.** Rejected: keeping the
declared files' digests. Keeping them keeps the read inside the task worktree
that blocks every task in a repository with a gitignored lockfile — the defect
that motivates the change, and the read Stage 2 removes along with the
fingerprint. Rejected: keeping the repository root, which distinguished
repositories sharing one cache directory and distinguishes nothing once that
directory is gone.

**The sanity gate prepares unconditionally.** Rejected: keying reuse on
something that distinguishes a merged tree by itself, such as the worktree's
`HEAD`. The agent commits during coding, so a `HEAD`-derived key would
re-install after every commit; an explicit request at the one call site that
needs a fresh tree states the rule where it applies.

**Agents run in the operator's environment like every other repository
command.** Rejected: keeping a Betterborg-controlled home for agents on the
grounds that an agent should not see `~/.ssh`. A synthetic `HOME` is not that
boundary — agent adapters already merge the operator's whole environment, so the
tokens are already there — and it would leave the agent running tests against an
empty package store while the worktree was prepared against the operator's.
Real isolation is a sandbox, which is a different design, not a `HOME` override.

**Preparation runs per worktree rather than once into a shared cache.**
Rejected: keeping a prepared directory and linking it into each worktree. It
preserves the cache's benefit and also its invalidation rules and failure modes,
which is most of what this plan removes. If concurrent installs are ever
measured to hurt, copying a prepared directory into a new worktree is a smaller
mechanism than the one being removed.

**Declared environment evidence stops being enforced but keeps being recorded.**
Rejected: removing `environment.files` from the schema. It is evidence a person
reads, like declared services, and an analysis describing an environment is not
what makes runs fail.

**Service orchestration is removed rather than made optional.** Rejected: a flag
that skips service validation. A flag leaves the code, its tests and its failure
modes in place and adds a mode nobody exercises.

**The Compose bookkeeping goes with the Compose manager, and nothing repairs
what that strands.** Rejected: keeping the table's confirmation path so a row
written by an earlier version could still be cleared — its only caller lives in
the manager being deleted, so the path would survive with nothing to call it and
the claim would stay held forever. Rejected: confirming cleanup without
performing it, which releases the claim by recording that containers were
stopped when they may still be running. Rejected: keeping the table unread,
which preserves nothing an operator cannot get from `docker compose ls` and
strands every accessor. Rejected: a migration that repairs the claims those rows
stranded. It has to read a durable event to tell a failed teardown from a failed
review, and write timestamps that satisfy two ordering constraints or roll back
on every open and stop the store opening at all — real risk, run unattended on
every upgrade, to save three people one manual fix.
