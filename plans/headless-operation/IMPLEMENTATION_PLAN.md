# Headless Operation

Betterborg cannot complete a run without a terminal. Creating a Borg refuses to
start unless stdin is a TTY, and the same command rejects an adapter whose
read-only boundary is a sandbox rather than a tool allowlist, which is the
shape every native CLI adapter has. Planning then stops on a repository the
operator has already trusted, because it runs in a worktree Betterborg itself
generated and that worktree is a workspace of its own. A turn whose result
misses the schema by one field ends the run outright, however well the rest of
it went, and the retry that would rescue it is told which field was wrong
without being told what would have been right. Separately, a Compose stack
created during preflight can outlive the run that created it, and enough
survivors exhaust Docker's address pool until nothing on the machine can
create a network.

A plan that satisfies its schema can still fail the checks that follow it, and
that failure ends the run too, although it names what it rejected. A retry that
could rescue any of these is never shown the result it is correcting, so it
writes a new one rather than repairing what it sent.

Under all of that sits a sandbox Betterborg cannot be talked out of. Codex is
always asked for a sandbox of its own, and the read-only one needs a user
namespace that the surrounding container is usually not allowed to create. No
setting anywhere says the environment is already isolated. When that sandbox
fails to start, Codex still exits zero and still answers, so the run carries on
with a result written without the repository it could never read.

Betterborg also keeps its own configuration, prompts and score inside the
repository it is working on, which is right for a team that owns that
repository and wrong for an operator only passing through: the scaffolding
becomes indistinguishable from the change under review.

Last, the Architect may find the requirements genuinely ambiguous and ask,
which is the right instinct and the correct thing to do with a terminal in
front of it. Unattended there is nobody to answer, so a run that read its
repository carefully and reasoned well stops anyway, on its best work.

Together these block every unattended use, and spoil the output of the runs
that do finish: CI, cron, a queue worker, and a benchmark container. The work
here is twelve independent changes, each closing one of them.

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

## Stage 6: A rejected result says what would have been accepted

**Goal**: A validation error names the constraint the value violated, so the
agent correcting it knows what to produce.

A rejected string reports that it does not match a pattern without saying
which pattern; a rejected length says the string is too short without saying
the bound; a rejected number, array length, enum member and branch are the
same. The agent is told which field is wrong and nothing about what would be
right, so correcting the result is guesswork. Retrying an uninformative
correction spends turns without changing the odds, which is how a value that
misses a pattern by its shape rather than by chance exhausts a whole budget.

The values that describe a constraint come from the schema, which Betterborg
wrote. Reporting them tells the agent only what it was already asked for, and
keeps the property that matters about a message: it reaches logs, exceptions
and stored state, so the rejected value itself is never quoted in one.

A constraint is rendered as JSON, because JSON is what the agent has to send
back; a Python rendering would offer it True and None, neither of which it can
use. A schema value that cannot be rendered as JSON describes a constraint no
agent could satisfy, so it is refused when the schema is validated rather than
described in a correction.

A constraint carries no length bound of its own, so a long one is shortened by
dropping whole members and saying how many were dropped. A value with no
members to drop, a pattern or a bound, is shown whole however long it runs. A
value cut part way through reads as a shorter value that the schema would
reject just as surely, and an abbreviated pattern is not even a pattern, which
would leave the message worse than the silence it replaced.

**Success Criteria**:
- A violated constraint is reported with the value it required, for patterns,
  string and array lengths, numeric bounds, enum membership, and the branches
  of a rejected anyOf or oneOf.
- A constraint is rendered as the JSON the agent is being asked to produce.
- A message never quotes the rejected value, nor any part of the payload
  beyond the property names already reported.
- A message stays a single line, whatever a constraint holds and whatever the
  payload named, and a shortened constraint shows whole members only and says
  how many it left out.

**Tests**:
- Each constraint's message names its required value.
- A payload value that violates a constraint does not appear in the message.
- An agent correcting a rejected pattern receives the pattern.
- A constraint too long to show keeps whole members and reports the remainder.
- A constraint with no members to drop is shown whole.
- A rejected branch names the alternatives it required.
- A payload property name cannot split a message across lines.
- A constraint that is not JSON is refused as a broken schema.

**Status**: Complete

## Stage 7: A plan that fails its contract is asked to fix it

**Goal**: A plan the agent could correct is sent back for correction, so a
recoverable mistake does not end the run.

An Architect plan that satisfies the schema still has to pass the checks that
follow it: that its phases are numbered in sequence, that a dependency names an
earlier phase, that the paths it says a phase touches are files the repository
contains, and a dozen rules like them. A plan that fails one of them is not
resent. The attempt is marked failed, planning stops, and an operator is told
to resume.

The agent already has everything it needs to fix such a plan, because the
failure names what it rejected. What it does not get is another turn. The
retry that rescues a missed schema cannot reach this: that one lives in the
adapter and answers a rejected payload, while this failure is raised by
planning against a payload the adapter already accepted.

These checks stop at the first value they reject, so a rejection usually means
more remain. A correction that repaired only the value it was handed would
spend the budget one violation at a time on a plan that was a single pass from
valid, so it asks for a pass over the whole plan instead. What it must not do
is restate the rules: the checks own them, and a second copy in a prompt would
drift out of step with the first.

A broken plan contract is a property of one sampled result, exactly as a missed
schema is. Some fraction of runs produce one, and an unattended sweep loses
that fraction outright. Correcting it costs one turn; failing it costs the run.

**Success Criteria**:
- A plan that fails a deterministic check is sent back with the failure, and
  planning continues when a later plan passes.
- The correction asks for a pass over the whole plan, in terms that fit every
  check rather than only the one that rejected the plan.
- The corrections are bounded within a run, and exhausting them fails with the
  last failure rather than a summary of all of them. A resumed run plans afresh
  and buys its own corrections, as it does for a missed schema.
- A plan that passes is unaffected and costs no extra turn.

**Tests**:
- Planning survives a first plan that fails a deterministic check.
- The correction the agent receives names the check that failed.
- The correction asks for a whole-plan pass without naming a single check.
- Several violations are correctable inside the bound.
- A rejected revision is corrected against its persisted findings.
- A correction does not outlive the turn it was built for.
- A cancelled run stops before it validates the plan it received.
- Exhausting the bound fails with the last failure.
- A plan that passes runs exactly one turn.

**Status**: Complete

## Stage 8: A correction shows the agent the result it sent

**Goal**: An agent asked to correct a rejected result can see what it produced,
so it repairs one field instead of producing another whole result.

A native transport retries a rejected result by rebuilding the prompt as the
original request plus the validation failure, in a fresh process. The agent is
told which path was wrong and what was required, and is never shown the result
it sent, so it cannot repair that result. It writes a new one from the same
starting point, and each attempt is an independent sample from the distribution
that produced the mistake. A constraint the agent tends to miss is missed
again.

The constraints this costs most are the ones the transport cannot carry. A
native CLI that takes a schema drops the keywords its provider cannot express,
so the properties and types come back right and the lengths, bounds and
patterns are guarded by local validation alone. Those are exactly the
rejections a retry has to repair, and exactly the ones it currently rerolls.

Planning already answers this one layer up: a plan rejected by its contract is
handed back through the channel a revision uses, so the agent revises what it
wrote. The adapter has no equivalent and needs one.

The result travelling back to the agent is the agent's own output returning to
the provider that produced it, which discloses nothing new. It travels in the
prompt, which leaves the rule about messages untouched.

**Success Criteria**:
- A correction carries the rejected result alongside the failure that rejected
  it, in whatever form that result is already safe to keep: a transport that
  redacts a submission carries back the redacted one.
- A result that cannot be rendered as JSON is not quoted at all, and the
  correction names the failure alone.
- No validation message gains a payload value, and Betterborg writes the
  correction to no log of its own.
- A result that validates is unaffected, and a first attempt carries no
  rejected result.
- Every adapter that retries a missed schema carries back the result of the
  attempt it is correcting.

**Tests**:
- A corrected attempt receives the result the previous attempt sent.
- A first attempt carries no rejected result.
- A validation message still names no payload value.
- A result that cannot be rendered as JSON still corrects, without crashing.
- The correction reaches the transport that carries it and no log of ours.

**Status**: Complete

## Stage 9: An operator can declare the environment already isolated

**Goal**: An operator running Betterborg inside a container that is already
sealed can say so, and Codex then runs without a second sandbox of its own.

Codex is always launched with a sandbox: read-only for a read-only tool set,
full access otherwise. On Linux the read-only one is bubblewrap, which has to
create an unprivileged user namespace, and a container started under a default
seccomp profile is refused that syscall. A benchmark task container is exactly
where that refusal happens and also exactly where a second sandbox buys
nothing, because the container is already the boundary.

The setting has to come from whoever runs Betterborg and never from the
repository being worked on. Tracked configuration is written by the repository
under test, which is the party a sandbox defends against, and it is already
treated that way: secrets and absolute machine paths are rejected there
outright. An environment variable belongs to the operator who started the
process, so that is the channel.

**Success Criteria**:
- With nothing set, Codex is sandboxed exactly as it is today.
- An operator can declare the environment already isolated, and Codex then
  skips its own sandbox whatever the tool set.
- The declaration cannot be made by tracked repository configuration.
- An unrecognised value stops the run and names what is accepted, rather than
  quietly choosing either boundary.

**Tests**:
- An unset variable yields a read-only sandbox for a read-only tool set and
  full access otherwise.
- The isolated declaration yields full access for a read-only tool set.
- Tracked configuration carrying the setting does not change the sandbox.
- An unrecognised value fails, and the message names the accepted values.

**Status**: Complete

## Stage 10: A sandbox that cannot start fails the run

**Goal**: A Codex run whose sandbox never initialised fails, instead of
returning an answer composed without the repository.

When bubblewrap cannot create its namespace, every command Codex runs under the
sandbox fails with the same launcher error, and Codex reports it the only way
it can, as failed command output the model then reads. The model says so in its
answer, but Codex exits zero and produces a result, so neither path that could
stop the run is on: transient classification inspects a non-zero exit only, and
terminal extraction runs only when no payload arrives. The cost is not a crash
but a plan whose confidence is unearned.

The signal is unambiguous and already in the log Betterborg keeps. A command
Codex ran failed, and its output is the sandbox launcher saying it could not
build the namespace. A failure of that shape is never partial, because it
denies every sandboxed command equally, so one occurrence settles it.

**Success Criteria**:
- A log carrying a sandbox launcher failure fails the run even when Codex
  exited zero and returned a valid result.
- The failure names the sandbox as the cause and points at the setting that
  resolves it, rather than reporting the model's answer.
- The failure is terminal, because a sandbox that cannot start will not start
  on a retry.
- A run whose commands failed on their own merits is unaffected.
- A log carrying no command output at all is unaffected.

**Tests**:
- A Codex log holding the bubblewrap namespace failure fails a zero-exit run
  that produced a schema-valid result.
- The error names the sandbox and the setting that resolves it.
- A log whose only failed command is an ordinary non-zero exit still succeeds.
- The failure is not retried.
- A run under the isolated declaration never trips the check.

**Status**: Complete

## Stage 11: Betterborg's own files can live outside the repository

**Goal**: An operator working on a repository whose history they do not own can
put Betterborg's tracked directory somewhere else, leaving that repository's
working tree exactly as Betterborg found it.

Betterborg keeps its configuration, prompts, PRDs and score inside the
repository at `.betterborg`, and adds a managed block to that repository's
`.gitignore` to hide its state directory. For a team that owns the repository
this is the point: the configuration is reviewed and shared like any other
checked-in file. For an operator pointed at a repository they are only passing
through, it is wrong in both directions. Betterborg's scaffolding becomes
indistinguishable from the change under review, and a diff meant to carry one
piece of work carries a configuration file, three prompts, a PRD, a score and
an edited `.gitignore` instead.

Every path Betterborg writes already derives from a single tracked directory,
and one line decides where that directory is. The relocation is that line. When
the directory sits outside the repository there is nothing inside the
repository left to ignore, so the managed `.gitignore` block stops being
written rather than being written somewhere it does not belong.

Like the sandbox declaration this belongs to the operator who started the
process and not to the repository being worked on, and for the same reason.
Betterborg already holds state this way: the trust store is located by
`XDG_STATE_HOME` and is refused outright when it resolves inside the
repository it vouches for. `BETTERBORG_HOME` names one repository's directory
and follows both halves of that rule.

One directory serves one repository. Configuration already carries the
repository it belongs to, so a directory holding another repository's
configuration is refused rather than quietly shared between them.

**Success Criteria**:
- With nothing set, the tracked directory stays at `.betterborg` inside the
  repository and the managed ignore block is still written.
- An operator can place the tracked directory elsewhere, and configuration,
  prompts, PRDs, tasks, score, state and artifacts all follow it.
- A repository worked on under a relocated directory ends with no Betterborg
  file in its working tree and an unmodified `.gitignore`.
- A relocated directory that resolves inside the repository is rejected,
  because it would silently reintroduce what the setting exists to prevent.
- A relocated directory already holding another repository's configuration is
  refused, rather than serving both from one set of files.

**Tests**:
- An unset variable leaves every derived path where it is today.
- A relocated directory moves configuration, prompts, PRDs, tasks, score,
  state and artifacts together.
- A run under a relocated directory writes no managed ignore block.
- A relocated directory pointing inside the repository fails and says why.
- A relocated directory belonging to another repository fails and says why.

**Status**: Not Started

## Stage 12: An unattended run answers the Architect rather than stopping

**Goal**: Planning started unattended resolves the Architect's questions by
deciding them, and says in the plan that it decided them.

The Architect asks when the requirements do not settle something it needs, and
that is the behaviour to keep: a run that invents a requirement silently is
worse than one that stops. With a terminal the operator answers. Without one
the prompt returns nothing and the run ends holding a plan it had already
reasoned its way to, which is the most expensive way to fail.

Someone has to decide. Unattended, the only party present is the Architect
itself, so it answers its own question the way a careful engineer does when
nobody is reachable: pick the reading the evidence best supports, say plainly
that it was assumed rather than given, and carry on. The value of that is
entirely in the saying. An assumed requirement that reads like a given one
turns an honest gap into a false certainty, and the operator loses the one
signal telling them where to look first.

The question round is already durable, so the answers belong in it, marked as
the Architect's own, beside the question that prompted them.

**Success Criteria**:
- Started unattended, a question round is answered by the Architect and
  planning continues to a plan.
- Each answer records that the Architect assumed it, and what it assumed, in
  the stored round beside its question.
- The resulting plan carries those assumptions where a reader meets them,
  rather than only in the store.
- Without the unattended option the existing behaviour is unchanged, including
  stopping when a prompt returns nothing.
- The bounded round cap that governs question rounds still governs them, so an
  Architect that keeps asking still ends the run rather than looping.

**Tests**:
- A question round started unattended yields a plan, with no prompt issued.
- The stored round carries the Architect's answer and marks it assumed.
- The plan a reader receives names the assumptions it rests on.
- Interactive planning still prompts, and still stops on a cancelled prompt.
- An Architect that asks past the round cap still ends the run.

**Status**: Not Started
