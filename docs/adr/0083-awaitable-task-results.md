# ADR 0083: Awaitable task results

Status: Accepted

## Context

The runtime returns `moirai_core::TaskHandle` for spawned futures and closures.
Its blocking `join` parks the caller. An async parent cannot use it to collect
an unfinished child that needs the same sole worker without blocking progress.
The existing `ResultCell` already supports serialized waiter replacement.

## Decision

Implement `Future` on the existing handle, sharing its completion cell with
blocking `join`. The output is `Option<Result<T, TaskError>>`, preserving the
existing detached-handle and task-error meanings. A private waiter enum carries
either a thread or the current task waker. Re-polling replaces the waker, and
switching from a pending poll to blocking `join` replaces it with the thread.
The consumer checks readiness again after registering, closing the concurrent
completion window. No second result allocation or completion protocol is added.
Dropping an awaited handle removes its registered waiter without cancelling the
producer. A producer that already owns notification may finish that callback.

## Alternatives

A separate async handle returned by the facade would split result collection
by spawn method and require a public return-type migration. Blocking inside
`poll` prevents other tasks on the same worker from making progress.

## Consequences

Existing blocking callers keep their signatures. Async callers await handles
instead of blocking workers. Dropping a handle detaches result collection and
releases a registered waiter; it does not request task cancellation. The
producer owns result publication. After a borrowed await consumes the result,
`is_finished` remains true; polling again or calling `join` returns `None`.

## Verification

Co-located handle tests cover ready and detached results, wakeup, replacement,
sender cancellation, dropped consumers, and poll-to-join handoff. The tests
exercise the production `ResultCell`; Loom bounds the producer/removal race and
removal followed by check-register-recheck to four preemptions. Caller-owned
destructors remain outside that finite-progress claim.
The implementation is delivered through
[PR 646](https://github.com/ryancinsight/Moirai/pull/646).

## Overturning evidence

A failing producer/consumer interleaving or an external contract requiring a
different detached result would require revisiting the cell or result contract.
