# ADR 0014: Reactor-Bound Async I/O and Readiness Integration

Status: Accepted

**Date**: 2026-05-25

**Revision 2026-09-30**: Rewritten to the as-built state audited under
MOI-NATIVE-REACTOR-001 (PR 570). The Windows gaps against that item move to ADR 0067;
the Unix level-triggered/edge-triggered conflict is recorded under Residual Risk.

**Revision 2026-09-29**: Waiter cancellation is the readiness contract on every
native target; Unix removal treats `EBADF`/`ENOENT` as a retired registration.

**Revision 2026-09-23**: File syscalls run on a bounded blocking pool, never
inside `poll` (MOI-ASYNC-FS-BLOCKING-2026-09-23).

**Revision 2026-09-17**: Windows poll snapshots hold socket leases through
`WSAPoll` (PR 390).

**Revision 2026-09-16**: `POLLNVAL` invalidation is generation-tagged and a
platform iteration error is terminal.

## Context

Socket futures perform a non-blocking syscall and, on `WouldBlock`, register the
task waker with a reactor. Earlier revisions of this record claimed the reactor
removed idle CPU use and that Windows used "readiness structures". The audit
found the Windows backend is `WSAPoll`, not IOCP (an earlier IOCP backend was
deleted because completions are not socket readiness; CHANGELOG "real readiness reactor on Windows" entry), and
that the reactor thread wakes on a timer while idle.

## Decision

1. **One central reactor.** `IoReactor` (`moirai-pal/src/reactor/core`) owns the
   platform backend chosen by the compile target (`PlatformReactor`,
   `moirai-pal/src/lib.rs`): `EpollReactor` on Linux, `KqueueReactor` on
   macOS/BSD, `WsaPollReactor` on Windows, `WebReactor` on wasm32 (cooperative,
   ADR 0007). It is driven by `IoReactor::run` on the process-global thread
   `moirai-global-reactor` (started on first use, `reactor/tls.rs`) or by an
   executor calling `run_iteration`, which blocks without a timeout when its
   run queue is empty (`moirai-async/src/executor/core.rs`).
2. **Registration after `WouldBlock`, one-shot delivery.** A socket operation
   registers (descriptor, interest, waker) only after its syscall reports
   `WouldBlock`. Dispatch consumes exactly the interest an event reports,
   narrows the platform registration, and wakes the wakers registered for that
   descriptor and interest (`reactor/core/event_dispatch.rs`). A task still
   blocked re-registers on its next poll.
3. **Backends are level-triggered.** `EPOLLET` and `EV_CLEAR` are not used: a
   waker armed after `WouldBlock` would miss readiness that arrived in the
   window, and an earlier edge-triggered revision hung tasks that way (CHANGELOG
   lost-edge entry). Level-triggering plus one-shot narrowing at dispatch gives
   edge-like delivery without the window.
4. **Flags are wake reasons, errors come from the retried syscall.** Backends
   report `Event { error, hangup }`; dispatch treats both as readiness for every
   registered direction. The woken task retries its syscall, which returns the
   typed `std::io::Error` (`ECONNRESET`, `EPIPE`) or a zero-length read at end of
   stream. No error is synthesized from a flag.
5. **Generation-bound registrations.** Every platform registration carries a
   generation; a polled event applies only while its generation is current, so a
   reused descriptor value never inherits a stale interest or consumes a
   replacement (`reactor/registration.rs`). Waiter replacement publishes the
   waker and its cancellation identity under one lock and destroys displaced
   values after releasing it.
6. **Waiter cancellation.** A waiter owner retires its waker and platform
   interest on drop while the descriptor is still open (`WaiterCancellation`,
   `reactor/waiter_cancellation.rs`), so its owner declares the waiter before
   the socket. Unix readiness syscalls hold no user memory, so no lease is
   needed; a lease type holding the descriptor open is rejected because it
   delays `close` past the cancelled operation without a memory-safety gain.
7. **Windows `WSAPoll` ownership.** PAL sockets store the OS socket in an `Arc`;
   registrations keep a weak owner and each poll snapshot upgrades it to a lease
   held through the kernel call, excluding concurrent `closesocket` (Winsock
   prohibits it). A closed raw socket surfaces as `POLLNVAL` and invalidates
   exactly its generation, waking and removing its waiters. A pending connect
   re-probes with `select` every 100 ms because `WSAPoll` before Windows 10
   2004 never reports a failed connect (`net/connect/reprobe.rs`).
8. **Terminal driver failure.** A driven iteration error is retained once,
   every waiter and generation is removed and woken outside locks, and later
   registrations fail with the retained error as source. There is no retry and
   no driver switch (`reactor/driver_failure.rs`).
9. **Files use a bounded blocking pool.** File syscalls block on every
   platform. `moirai-pal::fs::File` exposes blocking `&self` primitives and
   `moirai-async::fs` runs each on a bounded pool with async admission, one
   stream operation in flight per handle. No file syscall runs inside `poll`.
10. **No active reactor means cooperative self-wake.** When `with_current`
    yields no reactor (wasm32 outside `with_active`, reactor or thread
    construction failure, test suppression), a `WouldBlock` poll wakes its own
    task and yields the thread (`net.rs` `wake_without_active_reactor`). This is
    a busy poll on every target, not a Windows mechanism.

## Rationale

- Level-triggered backends plus one-shot dispatch close the register-after-
  `WouldBlock` race without an arm-before-syscall redesign.
- Keeping platform descriptors inside `moirai-pal` leaves `moirai-async` with
  the readiness-driven facade only.

## Verification

- Dispatch, generation, cancellation, and terminal-failure behavior:
  `moirai-pal/src/reactor/tests/` (`readiness_dispatch`, `socket_generation`,
  `owned_waiter_cancellation`, `terminal_failure`, `backend_update_failure`);
  Windows snapshot leases: `moirai-pal/src/windows/poll/tests.rs`.
- Wake counts: `net::tests::self_wake_fallback_wakes_once_per_pending_poll`
  (N pending polls without a reactor produce N wakes) and
  `net::tests::reactor_readiness_wakes_an_idle_read_exactly_once` (zero wakes
  over 1000 idle reactor iterations, one wake on peer write).
- Unix backends are type-checked for Linux and macOS targets; their runtime
  tests did not execute on the Windows development host.

## Residual Risk

- **Windows gaps** (ADR 0067): the driver wakes every 10 ms while idle
  (`reactor/core/lifecycle.rs` `run`, about 67 loop iterations per second
  measured on Windows 11 with the 15.6 ms timer tick); each iteration rebuilds an
  O(n) `WSAPoll` snapshot under the registration mutex and returns a `Vec`;
  every registration sends a loopback datagram to interrupt the poll. The
  acceptance text "pins overlapped operations, binds handles once" is unmet.
- **Unix acceptance conflict**: MOI-NATIVE-REACTOR-001 asks for edge-triggered
  interests. Decision 3 rejects edge-triggering for the armed-after-`WouldBlock`
  protocol. Satisfying the item as worded needs arm-before-syscall
  registration, a respecification for the judgment tier.
- Platform asynchronous file I/O (IOCP, io_uring) is not built; ADR 0067 states
  the Windows plan.
