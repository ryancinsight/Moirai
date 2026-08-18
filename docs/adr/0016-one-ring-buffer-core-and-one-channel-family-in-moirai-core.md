# ADR-016: One Ring-Buffer Core and One Channel Family in moirai-core

Status: Proposed

- Status note: awaiting sign-off; implements the 2026-07-02 structural audit's
  S1/S2 findings
- Change class: [arch]

### Context

moirai-core ships five sibling implementations of the same ring-buffer
algorithm family: `communication::RingBuffer` (lock-free SPSC, CachePadded
sequences, MaybeUninit slots), `channel::spsc::SpscChannel` (a line-for-line
clone of that Lamport ring plus a `closed` flag and spin-blocking),
`memory::UnifiedRingBuffer` (the same ring, mutex-locked — its "lock-free
zero-copy" doc is false), `communication::zero_copy::MemoryMappedRing` (the
same ring behind CAS spin-locks; not memory-mapped despite the name), and
`channel::mpmc::BoundedMpmcQueue` (Vyukov — the one genuinely distinct
algorithm). Above them sit four channel bounded-contexts (`channel/`,
`unified_channel/`, `communication::zero_copy/`, plus the bare
`communication::RingBuffer`) with three duplicated error enums
(`ChannelError`, `UnifiedChannelError`, `ZeroCopyError`) all repeating
Full/Empty/Closed/WouldBlock. Only `MpmcChannel` is consumed by the live
runtime (`moirai/src/runtime.rs`, moirai-transport); `unified_channel` is
consumed solely by `moirai-iter::advanced_patterns` (itself a prune candidate,
ADR-017); `HybridChannel` and `zero_copy` are consumed only by benchmarks and
contract tests. `ipc::SharedQueue` is a justified separate ring (cross-process
Pod contract) and stays.

### Decision (proposed)

The variation dimensions across the five rings are exactly producer/consumer
cardinality and blocking policy — a bounded set expressible without cloning:

1. Keep TWO algorithm cores: the SPSC Lamport ring (canonical home:
   `communication::RingBuffer`) and the Vyukov MPMC (`BoundedMpmcQueue`).
2. Express blocking policy as a ZST strategy over those cores (the crate
   already has this exact pattern in `task::handle::ResultWaitPolicy`):
   `NonBlocking` / `SpinThenPark`, monomorphized so the non-blocking path
   compiles to the bare ring.
3. `SpscChannel` becomes a thin `RingBuffer + closed-flag + policy`
   composition (the shape `HybridChannel` already proves); delete
   `UnifiedRingBuffer` and `MemoryMappedRing`, retargeting `unified_channel`
   (or deleting it with moirai-iter's advanced_patterns per ADR-017) and
   `zero_copy` consumers onto the canonical cores.
4. ONE channel error enum in `channel::error`; the other two enums' extra
   variants (InvalidConfig, the zero-copy set) become variants or per-call
   typed errors. Every call site updated in the same change; no aliases.

### Consequences

Deletes roughly 1.5-2k lines of parallel implementations while keeping every
live capability; the 18-round-audited MPMC/hybrid protocols are preserved
as-is (this ADR relocates and dedups shells, it does not restructure the
verified CAS protocols). Consumers to update: moirai (runtime), transport,
benchmarks/contract tests, and moirai-iter's advanced_patterns (interlocks
with ADR-017 — implement after that decision).
