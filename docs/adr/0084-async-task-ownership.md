# ADR 0084: Async task ownership and cancellation

Status: Accepted

## Context

An idle future may have no queued poll job. Its event source may never wake it
again, so queue draining alone cannot settle its result during cancellation or
shutdown. The task registry already owns a cancellation flag and lifecycle
tokens serialize future access through the existing polling phase.

## Decision

Keep a live-future waker table in the hybrid executor. The table retains tasks
between polls; each task holds only a weak table reference. Completion removes
the entry. No additional cancellation flag or erased callback is introduced.

Cancellation publishes the registry flag before waking the task. Only the poll
owner reads the lifecycle cell and drops the future, at a poll boundary. A poll
already returning a value may win the cancellation race. Running synchronous
and blocking closures continue to completion because they cannot be safely
preempted.

Shutdown closes table admission and snapshots live entries under the table mutex.
It releases that mutex before publishing cancellation and waking tasks, then
joins scheduler workers. Each concurrent shutdown caller can settle the same
live tasks; only completion removes their entries. Spawn racing closure either
registers before closure or receives `ShuttingDown`. Rejected scheduler admission
unregisters the task and drops its unpolled future. Executor destruction uses the same shutdown path.
External event sources are not required to wake cancelled tasks.

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
regression cases, not an exhaustive interleaving proof.

## Overturning evidence

A race that permits concurrent future access, an unresolved cancelled handle,
or an outstanding task after shutdown requires revisiting ownership transfer.
