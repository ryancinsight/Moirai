# ADR 0045: Cancellable browser-local tasks

Status: Proposed

Date: 2026-09-07

Driver: [MOI-WASM-TASK-2026-09-07](../backlog.md#MOI-WASM-TASK-2026-09-07),
[Metis async](../../metis/backlog.md#METIS-ASYNC-001).

## Context

Moirai's browser PAL owns WebSocket receive waiters, deadline timers and DOM
listener callbacks, but `spawn_local` currently provides only fire-and-forget
execution. A Metis browser request can therefore lose its caller while the
outer task remains pending. Dropping an application state handle alone does
not wake that task, so the inner future can retain a browser callback until a
message or timer arrives.

## Decision

Keep `spawn_local` as the fire-and-forget entry point and add
`spawn_local_with_handle`. The new entry point returns a non-cloneable
`LocalTaskHandle`; calling `cancel` or dropping the handle sets a browser-local
cancellation flag and wakes the task's most recent executor waker. The wrapper
future checks the flag before polling its child and returns immediately when
cancelled, which drops the child future and its owned PAL resources. The
handle is single-owner so cancellation has one clear lifetime authority.

The cancellation state uses `Rc` and browser-thread wakers because WASM PAL
tasks are thread-affine. The wrapper remains generic and statically dispatched;
there is no second executor, unbounded registry, or vtable on the task path.
`spawn_local` reuses the same wrapper and intentionally detaches its handle to
preserve fire-and-forget semantics.

## Alternatives

Leaving cancellation to each consumer would duplicate task ownership and
would let a dropped request retain PAL callbacks. A global task registry would
add shared mutable state and an unbounded lifetime table. Replacing the
browser executor with a third-party runtime would split the Atlas PAL boundary
and introduce a second scheduler.

## Verification

The cancellation state has native value tests for pre-poll cancellation,
pending-task wakeup and child-drop, handle-drop cancellation, and completed
task behavior. `moirai-pal` is checked for `wasm32-unknown-unknown`, and
warning-denied native/WASM Clippy plus the focused native test suite run against
the exact provider revision. Metis will consume the handle in its browser host
and add a teardown export; a real browser trace remains required for
JavaScript-object allocation counts.

## Limits

Cancellation is cooperative: a task that never reaches an await point can
still occupy the browser thread until it returns. The handle does not abort
JavaScript work that has already completed, and this decision does not add
worker scheduling, fetch, or a browser WebSocket server.
