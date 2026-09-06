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

Finally, a run that plans and decomposes without a terminal still cannot
execute anywhere but the machine the repository was written on. Preflight reads
the analyzer's description of the repository as a list of things this run must
have: every command found anywhere in it, every toolchain any file implies,
every secret any workflow names. A container built to run one task is refused
over tools nothing in the run would ever invoke.

Together these block every unattended use, and spoil the output of the runs
that do finish: CI, cron, a queue worker, and a benchmark container. The work
here is fourteen independent changes, each closing one of them.

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

**Status**: Complete

## Stage 12: An unattended run decides rather than stopping

**Goal**: Planning run without a terminal settles the questions the Architect
would otherwise ask, and says in the plan which requirements it settled.

The Architect asks when the requirements do not settle something it needs, and
that is the behaviour to keep: a run that invents a requirement silently is
worse than one that stops. With a terminal the operator answers. Without one
the prompt returns nothing and the run ends holding a plan it had already
reasoned its way to, which is the most expensive way to fail.

Someone has to decide, and unattended the only party present is the Architect
itself. It is told so before it asks, because with nobody to answer, a question
buys nothing and costs the run: the judgement that would have gone into asking
goes into deciding instead, on the reading the evidence best supports. A
question it asks regardless is answered the same way rather than ending the
run. The question round is already durable, so that answer belongs in it,
marked as the Architect's own beside the question that prompted it. The turn
that decides a round raised by a plan is given that plan, because it is a fresh
agent holding none of the reasoning that raised the question, and a workspace
without the plan would have it decide from the requirements alone.

The value of all of it is in the saying. A requirement the Architect settled
for itself that reads like one it was given turns an honest gap into a false
certainty, and the operator loses the one signal telling them where to look
first. So the plan carries them where a reader meets them, under a heading
that says who decided them.

The plan names its own assumptions, and it is the only party that can. A
question has no identity beyond the words it was asked in, and every path that
re-raises one writes those words afresh: an Architect dissatisfied with its own
earlier reading restates it as the plan's open question, in a new sentence
generated by a different turn. Nothing outside the plan can tell a decision
restated from a second decision, so nothing outside the plan can say which
reading is live. Betterborg matching them by text would publish the reading the
Architect abandoned beside the one it kept, with nothing marking which.

The record keeps the one thing it can still judge: silence. It cannot say which
of several readings stands, but it knows the Architect decided something, so a
plan mentioning nothing has left the operator no sign of it. Such a plan is
asked once, shown the decisions on record, and told to state in its own words
whatever it still rests on. If it names them, that is what the plan carries. If
it is asked and still says nothing, the record is added to what the plan
already stands on: an approximate account of the run beats one that reads as
though nothing was decided for it.

Added, rather than put in its place, and only for the rounds answered since the
last plan that spoke. Where the record and the inherited list cover one
question, the record holds the later reading of it: a plan that says nothing
inherits what its predecessor named, and a question reopened after that plan
was answered again since. Keeping the inherited entry there would publish the
reading the run left and suppress the one it planned against. The record knows the question rounds and nothing else, so
a requirement the Architect settled without asking appears nowhere in it, and a
plan that already spoke has accounted for everything decided before it.
Publishing the record over such a plan would delete the assumptions it named
that no round produced, and reinstate ones a later plan retired.

Spoke, not merely finished, and not merely holding a list. A plan turn that
raises open questions completes on its way to having them answered, without
being asked to name anything, so it accounts for nothing. What it holds by then
is what it inherited, because every plan passes through this settling before it
is stored, so the list alone cannot say who wrote it. The questions it raised
can: they are what mark a plan that was never asked. Measured from such a plan,
the boundary would move past decisions no plan has named and no later window
would reach back for them.

An empty list is the exception, because it is the one list that says the same
thing whoever wrote it: this plan rests on nothing assumed. Inherited, it
retires nothing there was to retire; stated, it retires what came before.
Reaching back past it would reinstate exactly what it disclaimed.

Silence is what the plan itself said, not what it was handed. A plan naming
none of its own inherits the list the plan it supersedes carried, and judging
it after that arrives would find a list there and count it as having spoken, so
the ask would never fire for any Borg whose earlier plan carried anything.

Naming none and saying nothing are also different answers. An empty list is a
statement, made by a revision the review has settled, and it retires what came
before. A missing field is silence, and the Architect that forgot must not
thereby retire what the plan before it carried. Reading both as "none" would
leave it no way to retire an assumption except by inventing another.

The price of taking one account rather than merging them is that a plan naming
some of its assumptions is trusted to have named them all. That is the narrower
failure. An incomplete list is still a list of real decisions, while a merged
one is a history in which the standing decision cannot be picked out.

What an attended run cannot do is originate one. Every requirement there was
either read from the confirmed PRD or got by asking, so a plan claiming an
assumption is describing a conversation that did not happen, and the claim is
dropped. What it publishes instead is what the plan it supersedes published: an
assumption an earlier unattended pass made is no more confirmed for having been
revised by hand.

One rule holds across both, as far as it can be told: an assumption over ground
a person has answered is dropped, whoever offers it. Their answer is a
requirement, and listing it under decisions nobody confirmed sends them to
audit the one piece of ground they settled themselves. A question is recognised
here by its words too, so this removes the case it can see rather than every
case.

Starting a plan and changing one are the same lifecycle, and a revision is
where the Architect first meets a requirement the original plan did not need.
Both are run the same way, or a Borg planned without a terminal can only be
revised with one. The revision the review loop runs is the Architect the
command never built itself, and it decides on the same terms as the rest.

**Success Criteria**:
- Run unattended, the Architect is instructed to decide rather than ask, and
  planning reaches a plan without a prompt being issued.
- A question asked regardless is answered by the Architect, against the plan
  that raised it where a plan raised it, and the stored round records what it
  assumed beside the question that prompted it.
- The plan a reader receives names the requirements the run settled itself and
  no more: the plan's own account rather than several merged, and nothing over
  ground a person has since answered.
- A plan that says nothing about assumptions is asked once to state them, and
  the rounds answered since the last plan that spoke are added if it still says
  nothing. What it inherits does not count as having spoken.
- A plan naming an empty list has said it rests on nothing assumed, and retires
  what it inherits.
- Without the unattended option the existing behaviour is unchanged, including
  stopping when a prompt returns nothing and refusing an assumption a plan
  claims for itself.
- The bounded round cap that governs question rounds still governs them, so an
  Architect that keeps asking still ends the run rather than looping. It bounds
  one planning cycle, so a Borg that spent its budget planning can still be
  revised.
- Changing a plan, and revising one for the review, run unattended on the same
  terms as starting one.

**Tests**:
- An unattended run's Architect turns carry the instruction to decide, and an
  attended run's carry neither half of it.
- A plan run unattended names the requirements it settled for itself.
- A question reopened in different words publishes only the reading the plan
  follows, and a plan that names its assumptions owns the list.
- A plan that says nothing about assumptions is shown the decisions on record
  and asked again, including when it inherits a list; one that says nothing
  twice keeps what it stands on and gains the rounds answered since.
- A plan that already spoke is not asked again, and a retirement it made is not
  reinstated by the plan after it.
- A plan that only raised questions does not close the window over the
  decisions answered before it, including when it holds a list it inherited
  and two such plans follow one another; one that retires with an empty list
  while raising a question closes it, and the retirement stands.
- The record retires a question a person answered before it is published, and
  an attended run is never asked to name assumptions at all.
- A plan naming an empty list retires what it inherits, and one that says
  nothing keeps it.
- Where the record and an inherited list cover one question, the plan publishes
  the later reading and not the one the run moved on from, and the record keeps
  the later of two readings it holds itself.
- A plan asked to name its assumptions is shown the plan it is told to restate.
- An attended revision carries the superseded plan's assumptions, except over
  a question the operator answered in that run.
- A question round run unattended yields a plan with no prompt issued, its
  stored round marks the answer assumed, and a round raised by a plan is
  decided with that plan in the workspace while one raised by the questions
  phase is not attributed to it.
- An answer that repeats a question, skips one, or says nothing is refused, an
  abandoned answers turn is not recovered for a later round, and a decision
  interrupted before its round is answered leaves neither half behind.
- Interactive planning still prompts, still stops on a cancelled prompt, and
  still drops an assumption its plan claims.
- An Architect that asks past the round cap still ends the run, and a new
  cycle begins with its budget restored on both halves of it: a cycle
  following a spent one still asks its first question.
- A plan changed unattended, and a plan the review sends back, both assume the
  questions their revisions raise.

**Status**: Complete

## Stage 13: Preflight requires what the run will use

**Goal**: A run is refused for what it is about to do, not for everything the
repository has ever been able to do.

The analyzer describes a repository: the commands it found, the toolchains its
files imply, the secrets its workflows name. Preflight reads that description
as this run's requirements and refuses when the host cannot satisfy all of it.
On the machine the repository was written on the two are nearly the same thing,
and the refusal is a real service: a missing tool found at preflight is a run
that would have failed later, further in, having spent more. Somewhere else
they are not the same thing at all, and the refusal is over tools nothing in
the run would invoke.

Three different things are being required, and only one of them is used.

The toolchains and package managers are an inventory, not a command list. They
are resolved by looking for an executable of the same name, but nothing ever
said the name was an executable: the analyzer writes them for a person to read,
and "Go modules" and "Node.js" name no program. The inventory also adds no
coverage, because every command that runs already requires the program it
invokes. It stops being a source of requirements, and that covers the version
it pins as much as the program itself: a pin is this run's requirement only
where this run invokes the program, so a patch-level mismatch on a runtime no
command calls is the same refusal arriving by a different route.

The commands are used, and requiring what they invoke is right. What changes is
the answer when the host cannot invoke one: the command is dropped from the run
and recorded as dropped, rather than the run being refused. This is the trade
the stage makes, and it is a real one. Sanity is how a task proves it did not
break the repository, so a check that does not run is a check that cannot fail,
and dropping the wrong one lets a bad change through. What makes it the better
side of the trade is that the alternative is not a stricter run but no run at
all, and that dropping is only tolerable while it is visible: every dropped
command is named in the preflight result and in the sanity result of every task
that would have run it, so a green run that skipped its tests cannot be
mistaken for a green run that passed them.

The trade stops paying when nothing survives it. A run holding no check cannot
publish anything, so every task would be coded, reviewed and merged and every
one would then block. The alternative there is not a stricter run but the same
no run, after the whole spend, so a host that can run none of the catalogued
checks is refused as it was before.

The secrets follow the commands. One a workflow names but no command that runs
asks for is not this run's requirement. Which commands ask is answered first by
the command that names the secret, and only then by the stages the secret's own
record names: nothing in the analyzer contract makes those stages spell a
catalog stage, and a record read off a workflow plausibly names the job. And a
secret named twice is only ambiguous when the two records disagree; refusing a
repetition that says the same thing twice refuses over nothing.

**Success Criteria**:
- A host missing a program that only the toolchain inventory named runs, and
  is not refused.
- A host missing a program a catalogued command invokes runs, that command is
  dropped rather than the run refused, and the drop is named in the preflight
  result and in each affected task's sanity result.
- A run left with no check to run is refused, before any task is coded,
  whether the host could run none of them or the analysis declared none.
- A version pin on a program the run never invokes does not block.
- A secret no command that will run requires does not block the run, and one a
  surviving command names does block, however that secret's record spells the
  commands that use it, and reaches the command that named it when it runs.
- Everything reported about a run that quotes the analysis is masked, including
  the checks it names as skipped, on every surface that carries them: the
  terminal, the task's durable reason, the headless payload, and the pull
  request body that is pushed.
- A record that scopes a secret to the agents while a running command names it
  is refused, because no phase could both require and deliver it.
- A secret named more than once blocks only when the records disagree.
- A host that can satisfy everything behaves exactly as it does today.
- Nothing is dropped silently: a run that dropped a command can be told apart
  from one that ran it, without reading a log, over every surface that reports
  a run.

**Tests**:
- A toolchain the analyzer named for a person, with no executable of that name,
  does not block a run.
- A command whose program is missing is dropped, named in the result, and
  named in the sanity result of a task that would have run it.
- The commands that can run still run, and still fail the task when they fail.
- A host missing the only catalogued check is refused rather than spending the
  run and blocking every task at the end of it, and so is a catalog that
  declares no check for the host to miss.
- A toolchain version mismatch blocks where a command invokes the program and
  does not where none does.
- A secret required only by a dropped command does not block the run; one a
  surviving command names blocks even when its record names no catalog stage,
  and is handed to that command when it runs.
- A dropped command's own words are masked in the task outcome that names it
  and in the pull request body that leaves the host.
- A command naming an agent-scoped secret is refused before the run is spent.
- A materialize command the host cannot run refuses the run rather than being
  dropped, and a program named by path is resolved where its command runs.
- Identical repeated secret records are accepted; conflicting ones are refused
  and say what disagrees.
- A headless caller is told what was dropped whether it started the run or
  found one already going.
- With every program present and every secret configured, the plan preflight
  produces is unchanged.

**Status**: Complete

## Stage 14: A project decides how many revisions its plans get

**Goal**: The number of times the Tech Lead may send a plan back is a
project's decision, not a constant.

The Tech Lead reviews a plan, and where it finds something wrong the Architect
revises and it reviews again. After a fixed three rounds a plan it still will
not approve blocks, which is right: executing a plan the reviewer rejects is
the silent erosion of quality that having a reviewer prevents. What is not
right is that three is the only answer. A project whose plans are large enough
that its reviewer is still converging at the third round ends every one of them
the same way as a project whose reviewer has given up.

Execution already treats this kind of bound as a project's to set:
`[execution] review_passes` decides how many times a coding task may be sent
back. Planning gets the same knob for the same reason, defaulting to what it
does today so no repository changes behaviour by upgrading.

The budget bounds revisions, and nothing else changes. It is read when a run
starts and governs that run: a plan the Tech Lead has not approved blocks when
the budget runs out and stays blocked whatever the setting becomes afterwards,
because whether a rejection revised or blocked was settled when it completed
and is held in the Borg's state. Reading the record back through a number
raised since would deny a plainly terminal plan and answer with an error naming
a state. Raising the budget buys the plans that follow more attempts at
agreement, never agreement itself.

Blocking keeps the findings, and keeping them is worth something only if a
reader can reach them, so the command the run names for reading them shows
them, on the surface a person reads and on the one a program calls: a headless
caller is handed the findings and the call that reaches them, having no
terminal to be told in. What stands is narrower than what the record holds. A
finding belongs to the round that wrote it, rounds restart with each planning
cycle, and a revision the reviewer went on to approve answered the finding that
asked for it. Shown whole, the list reads as a page of outstanding objections
with round numbers that repeat. They are the reviewer's own words rendered into a
document, so they are escaped like everything else the plan carries. A revision already under way outlives a budget lowered beneath it, and
the round it leads to is the last one: that is what the reviewer is told, since
a round numbered past its own budget describes nothing it can use.

**Success Criteria**:
- A repository can set the number of Tech Lead review rounds its plans get.
- With nothing set, planning behaves exactly as it does today.
- A plan still unapproved when the budget runs out still blocks, with its
  findings preserved, readable on every surface that reports the block, and
  the run resumable.
- A surface reporting a block names the call that reaches the findings.
- A blocked plan re-entered later reports what the record holds, whatever the
  budget has since become, and reconstructs its progress rather than raising.
- The budget is reported where a reader can see which round they are in, and a
  round past its budget is named the final one rather than given a number its
  budget contradicts.
- A budget below one, or not a whole number, is refused when configuration is
  loaded rather than part-way through a review.

**Tests**:
- An unset budget leaves the number of rounds exactly as it is today.
- A raised budget lets the Tech Lead send a plan back more times, and an
  approval on the later round completes planning.
- A lowered budget blocks sooner.
- A plan unapproved at the budget blocks with its findings intact, and showing
  the plan shows them; a headless caller is handed the same findings and a
  call that reaches them. A finding from a cycle a change request closed, and
  one a later approval answered, are not shown.
- A finding carrying Markdown of its own renders as text, not as structure.
- A blocked plan re-entered with a raised budget reports the same result and
  reviews nothing further; one re-entered with progress attached reconstructs
  the revisions that ran.
- A revision under way when the budget is lowered still finishes, and the round
  it leads to is named the final one.
- A budget of zero, a negative one, and a fractional one are each refused with
  a message naming the setting.

**Status**: Complete


## Stage 15: The sanity gate runs the repository's checks, not its catalog

**Goal**: A merged task is proved by the commands that verify the repository,
and never by one that serves, watches, or publishes.

The analyzer catalogs what a repository can do, and describes the list in its
own words as the development commands: a Makefile's targets and a package's
scripts, gathered for a person to choose among. The sanity gate has exactly one
input and it is that list, so it runs all of it. Beside the test target it runs
the docs watch server, the interactive shell, and the release pipeline. The
first of those never exits, and the last publishes.

Nothing ever told the analyzer the list would be used this way, so nothing in
an entry says whether running it proves anything. The gate cannot infer it: a
docs watch server and a docs build are one word apart in the same manifest, and
the difference between them is what the command does, not what it is called. So
the entry says it. Each catalogued command declares whether running it to
completion verifies the repository, and the gate runs the ones that do.

The rule that decides it is narrow on purpose. A check runs to completion and
reports, and the gate reads its exit code and then requires the worktree to be
unchanged, so a command whose purpose is to rewrite files cannot be one however
useful it is: a formatter qualifies in the mode that reports and not in the mode
that writes. Everything the rule does not positively admit falls outside it,
because the two ways of being wrong are not equal. A check wrongly skipped is a
gap in what the run proved; a server wrongly admitted never exits, and every
task blocks when the gate times out, which is the failure this stage exists to
remove.

A catalog that declares no check leaves the run holding nothing that could
prove a change safe. That is the same run a host missing every check leaves, so
it is refused in the same place and for the same reason. Cataloguing nothing at
all leaves it too, and by the shortest route: the catalog is optional and the
analyzer is told to omit a category it has no evidence for. The refusal follows
the empty gate rather than the way it came to be empty.

An entry written before the analyzer was asked declares nothing, and is run.
Reading that silence as "not a check" would quietly stop running a repository's
tests, and a run that checked nothing would be indistinguishable from one that
passed. Too much in the gate is visible in the result; too little is not.

A command the gate will not run states no requirement for the run, so the
secrets rule already in place follows it: a secret only a non-verifying command
names does not block.

**Success Criteria**:
- Each catalogued command declares whether running it verifies the repository,
  and analysis is refused when one does not.
- The sanity gate runs the commands that declare they do, and no others.
- A catalog recorded before the declaration existed still runs in full.
- A run holding no check refuses before any task is coded, whether the catalog
  declared none or catalogued nothing at all.
- A non-verifying command is not a dropped one: it requires no program of the
  host, and is reported as no loss.
- A secret or service only a non-verifying command names does not block the run.
- Prepare and materialize commands are unaffected; they build the run itself.

**Tests**:
- A catalog mixing verifying and non-verifying entries runs only the former.
- A catalog that declares nothing runs every command.
- A catalog declaring no check is refused, naming the declaration as the cause,
  and so is an analysis with no catalog, no commands key, or an empty one.
- A non-verifying command whose program is absent neither blocks the run nor
  appears among the checks the host could not run.
- A secret, and a service, named only by a non-verifying command do not block.
- The analyzer schema refuses a catalogued command that does not declare.

**Status**: Complete


## Stage 16: A command's directory belongs to the repository

**Goal**: A command the analyzer reports runs somewhere Betterborg can reach,
and analysis that names anywhere else is refused where it was written.

Betterborg runs a repository's commands in a checkout, so the only directory
any of them can name is one relative to the repository root. A Dockerfile
states a working directory too, and it reads exactly like one: `WORKDIR /abs`
is a real path to a real directory, inside an image nothing in the run will
enter. An analyzer reading a Dockerfile for a repository's prepare steps takes
that path along with them.

Preflight already refuses it, correctly and with the right message, but it
refuses at the start of execution: analysis, planning, review and decomposition
have all been paid for by then, and the answer is that a directory named in the
first of them was never usable. The schema the analyzer answers is where that
belongs, because it is the one place the producer is told before it writes.

The rule is the narrow one the failure shows: a directory a command names is
written relative to the repository root. Reaching outside the checkout by
another route is preflight's to catch, where the checkout is known.

**Success Criteria**:
- A command whose directory is absolute is refused when analysis is validated.
- The analyzer is told the rule as part of the contract it answers.
- A repository-relative directory, including the root itself, is accepted.
- The rule covers catalogued commands and environment commands alike.
- Preflight's own containment check is unchanged.

**Tests**:
- An analysis whose prepare command names an absolute directory is refused,
  and nothing is stored.
- A catalogued command naming a relative directory is accepted.

**Status**: Complete


## Stage 17: A project decides how many revisions its task batches get

**Goal**: The number of times the Supervisor may send a task batch back is a
project's decision, on the same terms the Tech Lead's reviews already are.

Decomposition has the shape planning has. The Project Manager writes a batch,
the Supervisor reviews it, and where it finds something wrong the batch is
revised and reviewed again. After three rounds a batch it still will not
approve blocks, which is right for the same reason it is right one stage
earlier: publishing tasks the reviewer rejects is the erosion that having a
reviewer prevents. Three being the only answer is wrong for the same reason
too, and a project large enough to want more rounds of one wants more of the
other.

So it is the same knob, in the same table, read the same way: `[planning]
decomposition_rounds`, defaulting to what decomposition does today. The budget
is read when a run starts and governs that run; a batch that has already
blocked stays blocked whatever the setting becomes.

The three rules the review budget already settled hold here unchanged, because
they are properties of reading a record rather than of what the record is
about. Whether a rejection revised or blocked was settled when it completed, so
the record is not re-judged against a budget raised since. A revision already
under way outlives a budget lowered beneath it, and the round it leads to is
the last one rather than a number its budget contradicts. And the rejection
that blocked never revises, so it declares no revision to reconstruct.

**Success Criteria**:
- A repository can set the number of Supervisor review rounds its batches get.
- With nothing set, decomposition behaves exactly as it does today.
- A batch still unapproved when the budget runs out still blocks, with its
  findings preserved.
- A blocked batch re-entered later reports what the record holds, whatever the
  budget has since become.
- A round past its budget is named the final round.
- A budget below one, or not a whole number, is refused when configuration is
  loaded, and by the loop that is handed one directly.

**Tests**:
- An unset budget leaves the number of rounds exactly as it is today.
- A lowered budget blocks on its only round, and the round says so.
- The configured budget reaches decomposition through `plan approve`.
- A blocked batch re-entered with a raised budget reports the same result and
  reviews nothing further.
- A budget of zero, a negative one, and a fractional one are each refused with
  a message naming the setting.

**Status**: Complete
