# ADR 0084: Async task ownership and cancellation

Status: Accepted

## Context

An idle future may have no queued poll job. Its event source may never wake it
again, so queue draining alone cannot settle its result during cancellation or
shutdown. The task registry already owns a cancellation flag and lifecycle
tokens serialize future access through the existing polling phase.

## Decision

Keep a live-future waker table in the hybrid executor. The table retains tasks
between polls; each task holds only a weak table reference. Settlement removes
the entry after future destruction, publication, callbacks, and caught-payload
cleanup. No additional cancellation flag or erased callback is introduced.

Cancellation publishes the registry flag before waking the task. Only the poll
owner reads the lifecycle cell and drops the future, at a poll boundary. A poll
already returning a value may win the cancellation race. Running synchronous
and blocking closures continue to completion because they cannot be safely
preempted.

Shutdown closes table admission and snapshots live entries under the table mutex.
It releases that mutex before publishing cancellation and waking tasks, then
joins scheduler workers. Each concurrent shutdown caller can settle the same
live tasks; settlement alone removes entries after callbacks and cleanup. Spawn
racing closure either
registers before closure or receives `ShuttingDown`. Rejected scheduler admission
retains its exclusive queued epoch until it unregisters and drops its unpolled
future. The initial created phase prevents cancellation wakes from scheduling
a first poll before the admission owner. Executor destruction uses the same shutdown path.
External event sources are not required to wake cancelled tasks.

Timed shutdown performs cancellation and scheduler drain in its helper; user
notification code cannot hold the caller before the timed wait begins. Setup
time counts against the caller budget. The existing destructor still joins
the drain to preserve its documented shutdown contract; this is an exception
to the default nonblocking-destructor preference. Non-yielding work remains
nonpreemptible, so only the explicit timed call has a caller deadline.

Future destruction stays under the poll or rejected-admission owner. A destructor
panic reports `TaskError::Panicked` while tracking remains owned through state,
metrics, result and lifecycle publication, callbacks, and caught-payload cleanup.
Lifecycle publication returns an explicit
notification receipt; completion-owned callbacks run after result publication.
A concurrently registered waiter may observe the published lifecycle timestamp
while result publication is still in progress; registration runs on that
consumer thread, with no registry lock held during its callback. The public
status API remains eventually consistent with the one-shot handle. Both
notification channels are attempted and retained output/payload destruction is
guarded before a callback panic can reach the scheduler boundary. The
`publication_panics` metric counts result-publication and lifecycle-notification
failures, including destruction of uncollected results; shutdown contains each wake so
one failure cannot skip cancellation of the remaining tasks. Caught payloads and displaced results are dropped
after settlement using the scheduler panic boundary. A panic in that cleanup
payload, or in final-owner fallback destruction without a completion owner,
aborts under the existing fatal destructor policy.

Unrun closure captures are destroyed under an unwind guard before cancellation
notifications. A capture destruction failure reports `Panicked`; lifecycle
cancellation still records that the body never ran.

Synchronous, blocking, and detached jobs use the same result-publication and
notification operation as async tasks. An external exhaustive `ExecutorMetrics`
literal must initialize `publication_panics`, or use `ExecutorMetrics::new()`;
adding the observable field is a breaking public struct change.

The runtime implements the existing `TaskManager` trait so applications can
cancel and observe tasks through the public facade. Blocking `join` retains its
existing result contract. Task-wait timeout does not cancel its underlying task.

## Alternatives

Weak-only tracking fails to own a pending future that stores no waker. Dropping
futures directly from the cancellation thread can race an active poll and violate
pinning and exclusive access. A second cancellation flag duplicates registry
state and creates disagreement between task status and execution.

## Consequences

Task registration and removal acquire the lifecycle table mutex. Polling and
ordinary wake dispatch do not. The table has one entry per outstanding future;
entries disappear when their futures complete or are dropped. A future that retains a runtime
clone requires explicit shutdown to break that application-owned lifetime.

## Verification

Production-facade tests retain an external waker through cancellation and
shutdown, assert `Cancelled`, and count exactly one future drop. Additional
tests cancel during an actively held poll and retain a future that saves no
waker. The synchronous running-task contract remains covered by existing tests.
Admission tests saturate the real scheduler and assert that rejected futures
are never polled and drop once. A shutdown test retains an earlier snapshot
while another shutdown settles the same idle tasks. These are deterministic
regression cases, not an exhaustive interleaving proof. Structural benchmark
checks follow the receipt return type while retaining their elapsed-time
arithmetic and single-sample requirements.

## Overturning evidence

A race that permits concurrent future access, an unresolved cancelled handle,
or an outstanding task after shutdown requires revisiting ownership transfer.
