# ADR 0086: Task-wait deadlines

Status: Accepted

## Context

A deadline checked only when a completion waiter is polled does not cause an
event-driven executor to poll it again. Waiting on a task that never completes
therefore requires an independent timer notification.

## Decision

Each timed task wait owns one PAL timer, constructed once and polled with the
completion waiter. The PAL already sits below the executor, avoiding an executor
dependency on the higher-level async crate. Timer failures become typed task
errors. Untimed waits allocate no timer. Dropping the wait cancels its timer and
releases its registered waker.

Completion visible at a poll wins a simultaneous timeout observation. A timeout
ends only the wait; it does not cancel the task. A separate cancellation request
controls task execution.

The existing PAL sleeper uses a cancellation-aware condition variable. It
rechecks the deadline after every wake, including spurious notifications, and
exits promptly when cancelled. Waker destruction and notification happen after
releasing internal mutexes. Completion also clears a borrowed timer's waker.

## Alternatives

Polling the clock without registering a wake strands idle consumers. Importing
the async timer into the executor creates a dependency cycle. Copying that
timer's heap implementation creates two owners of the same protocol.

## Consequences

The current PAL provider retains one sleeper thread per pending timer. This
change establishes deadline and cancellation correctness; it does not establish
high-fan-out timer efficiency or parity with a shared timer driver.

## Verification

Executor tests assert a positive deadline's wake and typed timeout, completion
before expiry, and the existing zero and unknown-task outcomes. PAL tests retain
real pending/wake/ready coverage and assert cancellation and waker release. The
obsolete benchmark-source assertion requiring `std::thread::sleep` is removed;
the runtime tests establish behavior independent of that implementation spelling.

## Overturning evidence

A missed deadline notification, early completion after a spurious wake, retained
waker after completion, or timer resource exhaustion invalidates the relevant
contract. A shared service requires explicit fallible initialization and one
upstream owner before replacing the per-timer provider.
