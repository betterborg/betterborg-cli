# Headless Operation

Betterborg cannot complete a run without a terminal. Creating a Borg refuses to
start unless stdin is a TTY, and the same command rejects an adapter whose
read-only boundary is a sandbox rather than a tool allowlist, which is the
shape every native CLI adapter has. Separately, a Compose stack created during
preflight can outlive the run that created it, and enough survivors exhaust
Docker's address pool until nothing on the machine can create a network.

Together these block every unattended use: CI, cron, a queue worker, and a
benchmark container. The work here is three independent changes, each closing
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
