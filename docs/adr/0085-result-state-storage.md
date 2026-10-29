# ADR 0085: Result state storage

Status: Accepted

## Context

`ResultCell` serves both blocking and async consumers. Its former single atomic
protocol waits for the producer while registering a consumer and drops replaced
waiters while claiming a state the producer needs. A waker destructor can itself
complete the result, making that destruction reentrant.

The public unsafe `StateWord` storage seam exposes a mutable reference during
cell destruction, although the consumer only reads its final value. Loom stores
atomic values in its execution model and can expose them only within a closure;
returning a reference to a copied value would violate the storage contract.

## Decision

Keep one result cell with independent result-publication and waiter-ownership
atomics. The producer publishes the result before notifying the waiter machine.
If registration owns the waiter, notification marks a pending wake and returns;
registration then takes responsibility for that wake. Otherwise the producer
owns notification. Both release waiter ownership before invoking any waiter
destructor or wake callback. Registration never waits for producer progress.

Replace `StateWord::get_mut` with `exclusive_state(&mut self) -> u8`. This returns
the actual stored state under exclusive access and grants no mutable reference.
Update all first-party implementations and consumers without a forwarding shim.
This is a breaking public trait change; no version or publication is implied.

## Alternatives

Keeping the combined machine blocks async consumers behind preempted producers.
Self-waking retries replace blocking with polling rather than completing the
ownership protocol. A copied Loom implementation can drift from production.
A snapshot pretending to be the atomic's mutable storage violates the unsafe
trait contract. A second storage trait only to preserve the overstrong getter
duplicates the same role.

## Consequences

The cell retains one result allocation and monomorphized storage. The waiter
machine adds internal atomic bookkeeping; no performance improvement is claimed.
External `StateWord` implementors migrate the getter as described in
[the migration guide](../migrations/task-results.md).

## Verification

Production-cell tests gate result publication, replace wakers that own result
senders, and assert exactly-once values and wakeups. Loom exercises the production
algorithm with instrumented atomics under a finite preemption bound. Its payload
cells remain ordinary Rust cells, so the model establishes atomic protocol
evidence rather than exhaustive raw-memory access checking. Miri supplies a
separate memory-access check for the executed regression cases.
The provider is reviewed in [PR 647](https://github.com/ryancinsight/Moirai/pull/647).

## Overturning evidence

A lost wake, concurrent waiter access, reentrant deadlock, or storage backend that
needs mutation at destruction invalidates the corresponding protocol or seam.
