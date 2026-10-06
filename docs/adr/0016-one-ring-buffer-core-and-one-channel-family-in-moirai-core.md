# ADR 0016: One Ring-Buffer Core and One Channel Family in moirai-core

Status: Accepted

- Date: 2026-07-02
- Change class: [arch]
- Revision: 2026-09-30 — rewritten to the as-built decision. The 2026-07-02
  proposal (five sibling rings, three error enums, blocking-policy strategy
  types) was implemented in part by #429, #430 and #433; the record now states
  what the tree holds and what was not adopted.

## Context

`moirai-core` shipped five sibling implementations of one ring-buffer
algorithm family (an SPSC Lamport ring, an SPSC channel that cloned it, a
mutex-locked "unified" ring, a CAS-spin-locked "memory-mapped" ring, and a
Vyukov-style MPMC ring) and three channel error enums that each repeated
`Full`/`Empty`/`Closed`/`WouldBlock`. The variation between them is exactly
producer/consumer cardinality and blocking policy, a bounded set that needs no
cloned implementation.

## Decision

1. Two algorithm cores remain, one per cardinality: the SPSC Lamport ring
   `communication::RingBuffer` (`moirai-core/src/communication/ring_buffer.rs`)
   and the sequence-numbered bounded ring `moirai_utils::queue::LockFreeQueue`
   (`moirai-utils/src/queue/ring.rs`) behind the MPMC channel.
2. `channel::spsc::SpscChannel` (crate-private, `channel/spsc/ring.rs`)
   composes `RingBuffer` with a `closed` flag and a spin-then-yield schedule;
   it holds no second copy of the Lamport protocol. The public halves
   (`SpscSender`/`SpscReceiver`, `SpscRing::split`) are the ADR 0024 capability
   wrappers over it.
3. One channel error enum, `channel::error::ChannelError`. The duplicate
   ring types (`UnifiedRingBuffer`, `MemoryMappedRing`), the `zero_copy`
   subsystem, and the extra error enums (`UnifiedChannelError`,
   `ZeroCopyError`) are deleted, with every call site updated in the same
   change and no alias kept.
4. `channel::unified::UnifiedChannel` stays as a channel over `LockFreeQueue`
   with an overflow queue; its remaining consumer is `moirai-iter`'s
   `advanced_patterns`. `ipc::SharedQueue` stays a separate ring because of its
   cross-process `Pod` contract.

## Not adopted

The proposed `NonBlocking`/`SpinThenPark` zero-sized blocking-policy types over
the cores do not exist. Each channel keeps its own measured schedule
(`SPSC_BLOCK_SPINS` in the SPSC channel, `MPMC_BLOCK_SPINS` with a condvar
fallback in the MPMC channel, a retry loop in `LockFreeQueue::enqueue`).
Consolidating the schedules is tracked by
`MOI-SPIN-BACKOFF-CONSOLIDATION-2026-09-30` in `backlog.md`, which starts from
the shared spin budget rather than from policy types on the channels, because
the schedules differ by design and by measured fallback path.

## Consequences

The parallel ring implementations and two error enums are gone; the MPMC wake
protocol is unchanged (its Dekker pair is documented at
`channel/mpmc/channel.rs` and modeled in `moirai-core/tests/loom_mpmc_waiter.rs`). Adding a
new cardinality or blocking policy means extending one of the two cores, not
cloning one.
