# Headless Operation

Betterborg cannot complete a run without a terminal. Creating a Borg refuses to
start unless stdin is a TTY, and the same command rejects an adapter whose
read-only boundary is a sandbox rather than a tool allowlist, which is the
shape every native CLI adapter has. Planning then stops on a repository the
operator has already trusted, because it runs in a worktree Betterborg itself
generated and that worktree is a workspace of its own. A turn whose result
misses the schema by one field ends the run outright, however well the rest of
it went. Separately, a Compose stack created during preflight can outlive the
run that created it, and enough survivors exhaust Docker's address pool until
nothing on the machine can create a network.

Together these block every unattended use: CI, cron, a queue worker, and a
benchmark container. The work here is five independent changes, each closing
one of them.

## Stage 1: A read-only sandbox satisfies the PRD session

**Goal**: `betterborg create` accepts any adapter that can hold a read-only
boundary, matching the rule every other read-only stage already applies.

The PRD session is the only read-only role that does not call the shared
`require_read_only_agent` helper. It checks `tool_allowlist` alone, while the
shared helper accepts `tool_allowlist` or `read_only_sandbox`. A native Codex
adapter has the sandbox and not the allowlist, so analysis runs it under a
read-only sandbox while creating a Borg rejects it as unable to enforce a
boundary it demonstrably enforces.

**Success Criteria**:
- The PRD session applies the shared read-only requirement rather than its own.
- An adapter offering only `read_only_sandbox` is accepted.
- An adapter offering only `tool_allowlist` is still accepted.
- An adapter offering neither is still rejected, with the shared message.
- A host-capable adapter still has to arrive wrapped for workspace trust.

**Tests**:
- A Codex-shaped adapter (`read_only_sandbox` true, `tool_allowlist` false) is
  accepted where it is currently rejected.
- An allowlist-only adapter remains accepted.
- An adapter with neither capability raises the shared read-only error.
- An unwrapped host-capable adapter still raises the existing trust error.

**Status**: Complete

## Stage 2: Adopt a PRD without an interview

**Goal**: A Borg can be created from an existing Markdown PRD with no terminal
and no agent call.

Creating a Borg today always runs the requirements agent to interview the user
and improve a draft. When the PRD is already written and authoritative, that
interview is not merely unnecessary: it is the reason the command needs a
terminal and an agent at all. Adopting a finished PRD verbatim removes both
requirements from the path, and the adopted Borg must be indistinguishable
from an interviewed one to everything downstream.

**Success Criteria**:
- A distinct, explicit option adopts a PRD file's contents verbatim as the
  Borg's PRD.
- That path selects and invokes no agent adapter, and needs no provider
  credential.
- That path runs with stdin closed or not a terminal.
- Without that option the existing interactive behaviour is unchanged,
  including its terminal requirement.
- Planning proceeds from an adopted Borg exactly as from an interviewed one.
- Adopting requires a PRD source; the option cannot be combined with a request
  to brainstorm one.

**Tests**:
- Adopting with stdin not a terminal creates a Borg whose stored PRD equals the
  source file's contents.
- The adopt path performs zero agent invocations, asserted against an adapter
  that records every call.
- Without the option, a non-terminal stdin still fails with the existing error.
- `plan start` succeeds against an adopted Borg.
- Adopting without a PRD source is rejected before any state is written.

**Status**: Complete

## Stage 3: Preflight releases every Compose resource it creates

**Goal**: A Compose stack created during preflight is torn down on every exit
path, including failure and interruption, so repeated runs cannot accumulate
networks or volumes.

Teardown runs a secret-free cleanup model through `compose down`, and that
model declares the claim-owned networks and volumes without attaching them to
any service. Compose releases only the resources its services reference, so
every teardown, success included, returns zero while leaving the project's
volume behind, and its network too wherever the repository names one. A
service that declares no network is normalized onto the implicit default, so
that one network is referenced and released; a named network no cleanup
service joins is the case that survives. A failure the startup path does not
name by type reaches no teardown at all and leaves the whole stack running.
The failure is silent and cumulative: each leftover consumes one of Docker's
predefined subnets, and once they are gone every later run fails to create a
network for reasons that appear unrelated.

Teardown can only release what preflight discovered. The topology is read from
the repository once, while startup runs against the task worktree's copy of the
same file, so a task branch that adds a network or volume to it creates a
resource the cleanup model never names. Closing that gap means revalidating the
worktree's topology, which is a wider change than teardown and carries the more
serious half of the same problem: a branch can also introduce a writable bind
mount or a host network mode that the discovered topology never checked. Both
belong to that work, not this one.

**Success Criteria**, each for the topology preflight discovered:
- A preflight that fails after starting a stack leaves no Compose project,
  network or volume behind.
- The same holds when the run is interrupted rather than failing.
- Repeated failing runs do not increase the number of Docker networks.
- A successful run releases its network and volume along with its containers
  and images.

**Tests**:
- Two preflights forced to fail after their stacks start each leave their
  project owning no network, volume, container or image.
- The same assertion for an interruption path, and for a failure the startup
  path does not name by type.
- A failure while blocking the task still releases the started stack.
- A failure before the project is recorded keeps its own error rather than
  raising over it from a teardown that has nothing to release.
- The existing healthy-stack isolation test continues to pass.

**Status**: Complete

## Stage 4: Planning trusts the worktrees it manages

**Goal**: `plan start` runs to completion on a trusted repository without
asking the operator to trust a path Betterborg minted during the run.

Planning materializes a managed worktree under the repository's worktrees
directory and selects its agents against that worktree. Trust is an exact
identity, keyed on the Git common directory together with the checkout path,
so a generated worktree is a different workspace from the repository it came
from. Its path carries identifiers minted during the run, so an unattended
caller cannot trust it beforehand, and nothing runs between its creation and
its use that could trust it. Execution already resolves exactly this: its
coding, review and merge stages reuse the primary checkout's trust for the
worktrees they run in. The architect, tech lead, PM and supervisor stages do
not, so planning fails where execution would have succeeded.

**Success Criteria**:
- The planning stages reuse the primary checkout's trust when they run in a
  worktree the repository manages.
- That reuse is confined to worktrees under the repository's managed
  worktrees directory; any other path is trusted on its own identity.
- An untrusted repository still refuses to plan.
- Analysis and requirements, which run in the checkout itself rather than a
  worktree, keep trusting the path they run in.

**Tests**:
- Planning succeeds on a trusted repository whose planning worktree was never
  trusted on its own.
- Planning still refuses on an untrusted repository.
- A run path outside the managed worktrees directory is not granted the
  primary checkout's trust.
- The existing execution trust behaviour continues to pass.

**Status**: Complete

## Stage 5: A missed schema is retried, not fatal

**Goal**: An agent turn whose structured result fails schema validation is
retried automatically, so one malformed field does not end an unattended run.

A turn's result is validated against a schema after the process has already
exited, and a failure is terminal: the adapter reports the turn failed and the
run stops with the validating error, telling a human to resume. The retry that
does exist classifies process exit codes for transient service errors, so it
never sees a schema miss, and its backoff is measured in minutes because it
exists for rate limits rather than for a model that wrote `q01` where `q1` was
required.

The distinction matters because this failure is unlike the others here. They
were deterministic: the same run failed the same way until the cause was
fixed. A missed schema is a property of one sampled result, so it strikes some
fraction of turns, and an unattended sweep loses that fraction outright while
a human retrying by hand would very likely get a conforming result the second
time.

**Success Criteria**:
- A turn whose result misses the schema is retried without human action, and
  succeeds when a later attempt conforms.
- The retry tells the agent what was wrong with the previous result, so the
  attempt differs from the one that failed.
- Retries are bounded, and a turn that never conforms still fails with the
  validating error rather than looping.
- A schema miss is retried promptly, not on the backoff that exists for
  rate limits and service outages.
- Every adapter that validates a structured result behaves the same way.
- A result that cannot be read as a payload at all is a different failure and
  stays fatal here; only a payload that was read and then missed the schema is
  retried.

**Tests**:
- A turn whose first result misses the schema and whose second conforms
  completes, and the run continues.
- A turn that never conforms fails with the validating error, after the
  bounded number of attempts and no more.
- The retried attempt carries the previous validation error.
- A schema miss does not wait for the transient backoff.
- Planning survives a first architect result that misses the schema.

**Status**: Complete
