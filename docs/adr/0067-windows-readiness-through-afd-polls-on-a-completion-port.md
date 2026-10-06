# ADR 0067: Windows Readiness Through AFD Polls on One Completion Port

Status: Proposed

**Date**: 2026-09-30

Driving item: MOI-NATIVE-REACTOR-001. Audit and claim: PR 570.

## Context

ADR 0014 records the as-built Windows backend: `WsaPollReactor`, a
level-triggered `WSAPoll` loop. Against the item's Windows acceptance (pinned
overlapped operations, handles bound once, completions mapped to wakers without
thread contention or heap allocation in the poll loop) the audit found:

| Property | As built | Location |
| --- | --- | --- |
| Idle behavior | Driver wakes every 10 ms (67 iterations per second measured with the 15.6 ms Windows timer tick) | `reactor/core/lifecycle.rs` `run` |
| Poll cost | Snapshot of every registration rebuilt each iteration under two mutexes; cost O(registered sockets) | `windows/poll/polling.rs` |
| Registration cost | Every registration sends a loopback UDP datagram to interrupt the poll and contends with the poller on the same mutexes | `windows/poll/registration.rs` |
| Allocation in the loop | A `Vec` of events is returned per iteration; the dispatch path consumes it | `polling.rs`, `lifecycle.rs` `run_iteration` |
| Lifetime | Weak socket owners upgraded to leases so `closesocket` cannot race `WSAPoll` | `windows/poll/types.rs` |
| Closed-socket detection | `POLLNVAL`, generation-tagged | `polling.rs` |
| Connect failure | `select` re-probe every 100 ms for pre-2004 `WSAPoll` | `net/connect/reprobe.rs` |
| Files | Blocking pool; no overlapped I/O | ADR 0014 decision 9 |

The "cooperative fallback" in the item is the no-reactor self-wake
(`net.rs` `wake_without_active_reactor`: `wake_by_ref` plus `yield_now`). It is
not a Windows mechanism and is live only when no reactor exists. Measured on
this host with a counting waker over 2 s on an idle socket read: the fallback
performed 2,681,402 polls and 2,681,402 wakes and consumed 1015.6 ms of process
CPU; the reactor path performed 0 wakes from 134 driver iterations (process CPU
below the 15.6 ms accounting quantum). Host-specific, single run; the wake
counts are pinned as tests (ADR 0014 Verification), the CPU figure is not.

An earlier IOCP backend was deleted because overlapped completions are not
socket readiness and could not drive `net.rs`'s try-then-register futures
(CHANGELOG entry "real readiness reactor on Windows"). That constraint stands:
any replacement must still deliver readiness.

## Decision (recommended; sign-off asynchronous)

1. **Readiness is delivered by `IOCTL_AFD_POLL` requests completed through one
   I/O completion port.** This is the mechanism of mio (`src/sys/windows/afd.rs`,
   `selector.rs`, version 1.2.3) and wepoll. Each armed interest is one pinned
   `IO_STATUS_BLOCK` plus `AFD_POLL_INFO` issued with `NtDeviceIoControlFile` on
   an AFD device handle; the completion packet names that record.
   The socket-facing `net.rs` surface and the `Reactor` trait seam are
   unchanged, so Windows shares the readiness protocol of ADR 0014 with Unix.
2. **Handles bind once.** AFD handles are opened with `NtCreateFile` on
   `\Device\Afd\Moirai`, associated with the port once, and set to skip the
   handle event (`FILE_SKIP_SET_EVENT_ON_HANDLE`). mio groups 32 sockets per
   AFD handle without stating a reason in its source; slice 1 adopts the
   grouping unmeasured and slice 2's registration-churn benchmark decides
   whether it stays. Sockets are polled by their base
   handle (`SIO_BASE_HANDLE`, with mio's `SIO_BSP_HANDLE_*` fallbacks for
   layered providers) and are never associated with the port; a socket with no
   base handle fails registration with a typed error.
3. **Slot table, no allocation in dispatch.** Records live in a table of
   fixed-address slots sized at registration time, never moved. A slot carries
   a generation; a `Token` is (index, generation). Free slots are claimed and
   released through an atomic bitmap, so neither the claim nor the dispatch
   path takes a lock. The poll loop reads completions into a preallocated
   `OVERLAPPED_ENTRY` buffer and delivers readiness through a caller-provided
   sink (`FnMut(Token, io::Result<Event>)`; a failure status reaches it as `Err`), replacing the per-iteration `Vec`.
4. **One-shot per arm, one arm per socket.** An AFD poll completes once with
   the current readiness, matching ADR 0014 decision 2; re-arming after a
   `WouldBlock` has no lost-edge window because the poll reports state already
   present. mio's source states that an AFD poll cannot be modified or
   synchronously cancelled and that only one may be active per (socket,
   completion port); a change of interest is cancel plus re-arm with the union
   mask, and the caller (the reactor's registration table) guarantees one armed
   token per socket.
5. **Completion-after-cancel race.** A slot returns to free only when its
   completion packet is dequeued, whether it finished or was cancelled
   (`NtCancelIoFileEx`). A cancel moves the slot `Armed -> Cancelling` for its
   generation and sets a sticky `Requested` bit that no later step clears; a
   stale token (older generation) is a no-op and cannot cancel a later arm of
   the reused slot. A completion for a slot with `Requested` set is suppressed
   whether its packet was queued before or after the cancel call, so a cancel
   that loses the race to a readiness packet still reports nothing. Slot
   generations are 32 bits and wrap; a token reused after 2^32 arms of one slot
   would match, which a registration table holding at most one live token per
   socket does not allow to be live.
6. **Wake.** `PostQueuedCompletionStatus` with a reserved key replaces the
   loopback UDP socket; a wake posted before the poller blocks is not lost.
7. **Shutdown.** Drop cancels every armed slot, then drains the port until no
   slot is armed, bounded by a deadline of five seconds. The bound is chosen, not derived: the
   driver documents no cancellation latency, and a cancelled poll completes
   without waiting on a peer, so the deadline is reached only when the driver
   misbehaves. If a cancellation is refused or the deadline passes, the table
   is leaked rather than freed: the kernel may still write to it.
8. **Start status.** `NtDeviceIoControlFile` returns before the poll
   completes. The handle is asynchronous and does not skip the port on
   success, so a status of success, informational, or warning severity
   (`STATUS_PENDING`, a synchronous `STATUS_SUCCESS`, `STATUS_BUFFER_OVERFLOW`)
   leaves the request with the kernel and a packet will arrive; the slot stays
   armed and the packet's status is delivered as usual. Only an error-severity
   status (top two bits set) guarantees no packet and releases the slot at
   once.
9. **Typed errors.** `NTSTATUS` maps through `RtlNtStatusToDosError` to
   `io::Error`; `AFD_POLL_ABORT`/`CONNECT_FAIL` map to `Event::error`,
   `DISCONNECT` to `Event::hangup`, `LOCAL_CLOSE` to error plus hangup
   with neither direction set.
10. **Dispatch thread.** The existing driver (`IoReactor::run` or an executor's
   `run_iteration`) polls the port. No second thread is added. One thread
   polls at a time: a concurrent `poll` returns `WouldBlock` immediately
   rather than waiting behind the first, so a timeout bounds every call. The
   port is created with the maximum concurrency, so it never throttles: the
   limit counts threads that dequeued earlier and are still running, not
   threads blocked in a dequeue, so any finite limit lets that many busy
   threads starve a blocked poller (a limit of one hung a test; the processor
   count delayed a wake by seconds in a reviewer's probe). The entries mutex
   already admits one thread into the wait.

## Slices

1. **Port core** (`windows/afd`, delivered by the slice 1 PR): port, AFD
   groups, fixed-capacity slot table, arm, cancel, poll with sink, wake,
   drain-on-drop, tested against loopback sockets, with the arm-and-dispatch
   allocation contract. The table is fixed-capacity and refuses with
   `QuotaExceeded`; slice 2 makes it grow by appending fixed-address chunks.
2. **Reactor swap**: `Reactor` implementation over the port replaces
   `WsaPollReactor`, `SocketLease`, `POLLNVAL` generation handling, and the
   connect re-probe in one change, and `AfdPort` narrows from `pub` (kept only
   because the lib build rejects an unused `pub(crate)` item) to `pub(crate)`; the registration table grows by appending
   fixed-address chunks, and `AfdPort::poll` gains a distinct signal for a
   local close, which the slice 1 `Event` mapping folds into error plus hangup.
3. **Allocation-free dispatch and idle**: sink-based `poll_registered_events`
   on every backend; `run` blocks without a timeout since `stop` and every
   registration wake the poller.
4. **Native file I/O** (separate ADR): overlapped `ReadFile`/`WriteFile` with
   owned buffers on the same port, replacing pool offload for Windows files.

## Acceptance test design

- Value assertions on real loopback sockets: readable after peer write,
  writable on a fresh connection, readable plus hangup after peer close.
- Cancellation: a cancel racing readiness (barrier-synchronized) and a cancel
  after the packet was queued, for readable and writable polls, both report
  nothing and free the slot.
- A second device group (40 slots), local close, a refused second poller,
  concurrent zero-timeout pollers delivering each completion once, the start
  and failure status rules, and generation wrap.
- Stale token cannot cancel a later arm of a reused slot (capacity one).
- Drop with armed polls completes within the drain bound.
- Wake posted before and after the poller blocks returns it.
- No allocation in arm and dispatch, measured with a counting allocator
  (`moirai-pal/tests/afd_poll_allocations.rs`, slice 1); the same contract over
  `IoReactor::run_iteration` lands with slice 3.
- No sleeps; synchronization by channels, barriers, and bounded blocking waits.

## Rejected alternatives

- **Completion-model sockets (overlapped `WSARecv`/`WSASend`).** Changes buffer
  ownership across the shared `net.rs` surface and adds the cancel race to
  every read and write; its benefits do not reach the stated acceptance more
  directly than readiness over the same port.
- **Keep `WSAPoll`, remove allocations and the tick only.** Leaves the O(n)
  snapshot, loopback wake, and lease machinery; fails "binds handles once".
- **Zero-byte overlapped `WSARecv` for read readiness.** Documented API but no
  write or connect readiness equivalent.
- **Linux first.** The epoll backend implements generation-bound readiness; its
  gap is the acceptance wording (ADR 0014 Residual Risk), a respecification
  rather than a mechanism. Windows is the measured gap and the only target
  executable on the development host.

## Risks and overturning evidence

- `IOCTL_AFD_POLL`, `AFD_POLL_INFO`, the event bits, and the device name are
  undocumented by Microsoft. Their layout is taken from mio 1.2.3
  (`src/sys/windows/afd.rs`, read in the local cargo registry); mio ships on
  them. Slice 1 pins the layout with size and offset assertions and a creation
  self-check that fails with a typed error rather than degrading.
- Overturn if the port's completion latency or per-registration cost measures
  worse than `WSAPoll` on the registration-churn benchmark, or if the driver
  rejects the 32-slot grouping on supported Windows versions.
