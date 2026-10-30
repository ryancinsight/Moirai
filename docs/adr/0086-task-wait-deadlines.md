# ADR 0086: Task-wait deadlines

Status: Accepted

Revision 2026-10-29: task waits now distinguish unknown identifiers, elapsed
deadlines, and timer-service failures while preserving the timer's `io::Error`.
PAL timer construction validates the full requested duration without clamping.

## Context

A deadline checked only when a completion waiter is polled does not cause an
event-driven executor to poll it again. Waiting on a task that never completes
therefore requires an independent timer notification.

## Decision

Each timed task wait owns one PAL timer, constructed once when `wait_for_task`
is called and polled with the completion waiter. The PAL already sits below the
executor, avoiding an executor dependency on the higher-level async crate.
Untimed waits allocate no timer. Each wait owns a completion registration under
[ADR 0089](0089-owned-completion-waiters.md). Dropping the wait cancels its timer
and unregisters its waker. Completion, timeout and error paths dispose both
resources before returning `Ready`; destruction of the future is not required.

The executor reports `UnknownTask { id }`, `WaitTimedOut { id }`, and, when the
standard library is available, `TimerFailed { id, source }`. `TimerFailed` owns
the original `std::io::Error`; its `Error::source` is that same error, preserving
raw operating-system codes and custom payloads. `ExecutorError` therefore does
not implement `Clone`, `PartialEq`, or `Eq`. Callers inspect variants and fields
instead of requiring an I/O error to support traits it does not provide.

`Timer::new` and `timer::sleep` return `io::Result<Timer>`. Construction computes
`Instant::checked_add` with the complete requested duration before allocating
state or starting a sleeper. An unrepresentable deadline returns
`ErrorKind::InvalidInput`; representable durations retain their exact deadline.
The future continues to return the original thread-builder error if sleeper
creation fails.

Completion visible at a poll wins a simultaneous successful or failed timer
observation. An unknown task is reported before the timer is polled. A timeout
ends only the wait; it does not cancel the task. A separate cancellation request
controls task execution.

The existing PAL sleeper uses a cancellation-aware condition variable. It
rechecks the deadline after every wake, including spurious notifications, and
exits promptly when cancelled. Waker destruction and notification happen after
releasing internal mutexes. Completion also clears a borrowed timer's waker.

## Alternatives

Polling the clock without registering a wake strands idle consumers. Importing
the async timer into the executor creates a dependency cycle. Copying that
timer's heap implementation creates two owners of the same protocol. Clamping
large durations or substituting the current instant silently changes the
requested deadline and hides invalid input.

## Consequences

The current PAL provider retains one sleeper thread per pending timer. This
change establishes deadline and cancellation correctness; it does not establish
high-fan-out timer efficiency or parity with a shared timer driver.

## Verification

Executor tests assert exact identifiers for timeout and unknown-task outcomes,
completion precedence over ready success and failure, preserved I/O sources,
non-cancelling timeout behavior, and release of owned registrations and timer
wakers while the completed wait remains allocated. PAL tests retain real
pending/wake/ready coverage and reject `Duration::MAX` and `u64::MAX` seconds.
A fixed `Instant` verifies that a representable duration is retained exactly.
Bounded worker events establish sleeper exit after cancellation before or during
its wait, spurious-wake re-parking without completion, and borrowed-ready waker
release.

## Overturning evidence

A missed deadline notification, early completion after a spurious wake, retained
waker after completion, or timer resource exhaustion invalidates the relevant
contract. A shared service requires explicit fallible initialization and one
upstream owner before replacing the per-timer provider.
