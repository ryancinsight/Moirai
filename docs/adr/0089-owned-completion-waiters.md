# ADR 0089: Owned task-completion waiters

Status: Accepted

Revision 2026-10-29: use the standard mutex for the lazy set pointer;
cross-target checks preserve the task-state size bound without a custom lock.

## Context

A completion wait currently registers strong waker clones in the target task.
The registration outlives a dropped or timed-out wait. Repeated abandoned waits
against one unfinished target therefore retain historical waker payloads and
increase registration scans. Limiting active tasks does not bound this history.
A retained runtime waker can also retain its scheduler and registry.

Independent review of the current implementation establishes this ownership
path; it does not establish a production leak measurement. Existing completion
and timer tests do not exercise repeated abandoned registrations.

## Decision

Give each wait an owned, non-cloneable registration with stable identity.
Re-poll updates that registration instead of adding a historical waker.
Terminal poll and Drop remove it. Removal distinguishes independent waits even
when they present the same caller waker. Completion detaches or closes the
registration set before notifying, and attempts every notification before
propagating the first callback panic.

The registry owns registration storage. A handle keeps that storage valid
independently of registry block retirement; it must not own the scheduler or
the whole registry. Empty slots are reusable, and an empty set releases retained
wakers and its variable storage. Sequential abandoned waits must consume space
bounded by concurrent registrations rather than historical registrations.

Use a lazy shared set outside the task-state block and owned slot keys inside
that set. Keep the first slot inline in the shared set; allocate spill storage
only for additional concurrent registrations. Tokens own the set rather than a
task-state pointer. Completion closes registration before notification; handles
can still remove pending entries while other notifications execute.

Protect the lazy pointer with `Mutex<Option<Arc<WaiterSet>>>`. Pointer mutation
never invokes a caller callback. A separate set mutex owns slot bookkeeping;
the pointer lock is released before acquiring that mutex. Standard mutex
layouts fit the complete task-state budget on Windows, Linux and macOS.

Clone, wake and destruction of caller wakers occur outside every registry or
registration lock. Slot mutation contains only owned moves and bookkeeping.
Unwinding cannot leave a lock held or erase another waiter's notification.
Removal may briefly acquire the internal registration lock; it performs no
await, user callback or operating-system wait while holding that lock.

Preserve the existing 64-byte TaskState ceiling. A lazy shared registration set
may allocate when a task first acquires waiters. This correctness tradeoff
requires measurement before making allocation or throughput claims. It does
not change task-result ownership or make non-yielding work preemptible.

Replace the public registration operation in place and migrate every in-tree
caller. The 0.12 migration guide documents the owned return value. No persistent
legacy registration API remains alongside the new operation.

## Alternatives

Keeping strong wakers until completion retains the demonstrated history.
Weak entries without eager removal retain accumulating tombstones. Removing by
the caller's will_wake identity conflates independent waits and relies on a
best-effort identity test. A second global table adds ownership and lock order
without removing the target task's registration responsibility.

## Verification

Exercise actual completion waits with tracked waker payloads. Abandon and expire
successive waits on one unfinished target; each payload must be released while
a surviving wait still receives completion. Re-poll replaces and releases the
old waker. Concurrent completion/removal and block retirement preserve memory
safety and exactly-once notification obligations. Adversarial clone, wake and
drop callbacks reenter or panic without stranding other waiters.

Run focused Nextest tests in development and release profiles and Miri for
reachable unsafe storage. Check public callers, documentation and SemVer at the
0.12 boundary. These are behavioral and executed-case memory checks, not a
proof of every possible schedule or a performance claim.

The Loom model bounds the protocol to two waiter identities, two slots, one
registrar and one completion publisher, with the default 1,000-branch bound
and no preemption or permutation cap. It tracks the pointer mutex and
per-slot claims. It abstracts callbacks, allocation, updates and block
retirement; actual-code tests cover removal while another callback blocks.
The [delivery PR](https://github.com/ryancinsight/Moirai/pull/650) records the
exact candidate and collected gate results.

The internal completion-wait operation accepts its deadline future as an input
so tests can deliver expiry through an event without sleeping or assuming when
an operating-system timer fires. Production supplies the existing PAL timer.

## Overturning evidence

Historical growth after abandonment, a callback under a lock, notification
loss, unsafe retirement, or an expanded TaskState invalidates this design.
Measured registration cost may justify changing its representation while
preserving the same ownership and cancellation semantics.
