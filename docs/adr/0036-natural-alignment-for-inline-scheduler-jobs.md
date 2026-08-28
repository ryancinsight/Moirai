# ADR-036: Natural alignment for inline scheduler jobs

Status: Accepted

- Date: 2026-08-28
- Change class: [arch] [patch]
- Refs: `MOI-QUEUE-RETENTION-036`, ADR-005, ADR-034, Apollo PR #158
- Revision: 2026-08-28 — accepted after the exact Apollo retained-footprint
  probe and saved Criterion comparison passed the stated stop conditions.

## Context

`ScheduledJob` stores 14 machine words plus monomorphized execute and drop
function pointers, giving `InlineJob` a 128-byte size on the profiled x86-64
target. `InlineJob` also forces 64-byte alignment. That alignment lets closures
with alignment up to 64 remain inline, but it propagates through
`(Priority, ScheduledJob)` into every slot of the per-worker bounded MPMC
injector.

Apollo's exact retained-allocation probe attributes 24 surviving 65,536-byte
blocks to 24 worker injectors with 256 slots each. The blocks total 1,572,864
bytes of the 1,857,224-byte pool-warmup retained window. The current layout is
therefore 256 bytes per slot on the measured target. This storage is allocated
eagerly at executor construction and retained independently of queue occupancy.

Per-job cache-line alignment is not a synchronization boundary. The MPMC
queue's sequence number grants exclusive data access to one producer or
consumer, and its head and tail counters carry the cache-line isolation needed
for cross-thread mutation. Forced alignment on the payload increases every
empty slot as well as every occupied slot.

The entry baseline uses Criterion sample size 20, 500 ms warmup, and 2 s
measurement windows. Relevant slope estimates and 95% confidence intervals
are:

| Path | Entry slope estimate |
| --- | --- |
| fresh priority queue push/pop | 893.02 ns [880.83, 905.17] |
| fresh submission publication | 887.91 ns [882.50, 897.57] |
| maximum-inline construct/drop | 1.3362 ns [1.3138, 1.3618] |
| maximum-inline construct/execute | 7.5957 ns [7.3216, 7.8086] |
| oversized construct/drop | 22.208 ns [21.781, 22.789] |
| oversized construct/execute | 24.280 ns [24.007, 24.653] |
| fresh maximum-inline queue round trip | 890.89 ns [876.71, 914.37] |
| fresh oversized queue round trip | 917.32 ns [909.12, 928.88] |
| retained-worker maximum-inline dequeue | 100.16 ns [99.793, 100.61] |
| retained-worker oversized dequeue | 122.51 ns [122.02, 122.88] |

The fresh-queue rows include construction and allocation; the retained-worker
rows isolate steady-state queue use. Both classes are required because this
decision changes retained construction storage but must not regress worker
execution.

## Decision

Keep the 14-word inline capacity and monomorphized function-pointer dispatch,
but remove the forced 64-byte alignment from `InlineJob`. The storage uses its
natural machine-word alignment. Closures that fit the byte budget but require
greater alignment take the existing typed `Box<F>` fallback; this changes no
public API or execution semantics.

Pin the scheduler payload and generic queue-slot layouts with target-width
derived assertions. On 64-bit targets, the expected injector payload is 17
machine words and the slot is 18 machine words (144 bytes), reducing a
256-slot injector from 65,536 to 36,864 requested bytes. At 24 workers the
requested injector storage falls from 1,572,864 to 884,736 bytes, a reduction
of 688,128 bytes, while the executor-wide admission capacity remains 6,144
tasks under ADR-034.

The exact Apollo retained-allocation probe and retained-worker Criterion
confidence intervals are the acceptance gates. A later regression reopens this
decision; it does not authorize shrinking coverage or widening the bound.

## Failure modes

- An over-aligned closure must use the typed boxed trampoline and still execute
  or drop exactly once.
- The inline byte capacity must remain 14 words; shrinking it would move common
  captures onto the allocator and is outside this decision.
- Queue capacity, selected-worker routing, priority ordering, saturation, and
  wake progress must remain unchanged.
- The MPMC sequence protocol and its memory ordering are unchanged. Any queue
  algorithm rewrite requires a separate concurrency decision and Loom model.

## Alternatives rejected

1. Reduce the inline word count to force a power-of-two slot size. Rejected
   because it adds per-task allocation for existing 14-word captures.
2. Split priority into separate queues. Rejected because fixed per-priority
   partitions change the aggregate admission contract and can reject work while
   capacity remains in another priority.
3. Replace worker injectors with one global queue. Rejected by ADR-034 because
   it removes selected-worker placement and adds a shared contention point.
4. Lazily allocate each fixed injector. Rejected because pool warmup reaches
   every worker, so it defers rather than removes the measured retained state.
5. Introduce a segmented or reclaiming MPMC queue. Rejected for this bounded
   experiment because reclamation expands the lock-free proof surface while
   natural alignment addresses the measured amplification without changing the
   algorithm.

## Verification plan

- Assert exact inline storage, scheduler payload, and representative MPMC slot
  layouts from machine-word widths.
- Verify inline, oversized, and over-aligned jobs execute and drop exactly once.
- Preserve queue round-trip, priority, exact capacity, saturation, wake-progress,
  and scheduler shutdown tests.
- Run focused and workspace Nextest, warning-denied Clippy, doctests, rustdoc,
  release tests, and the applicable Loom suites.
- Compare the saved same-machine Criterion baseline against the candidate and
  rerun Apollo's exact retained-footprint instrument against the provider
  revision.

## Evidence

The layout tests establish a 17-machine-word injector payload and an
18-machine-word queue slot. The 64-bit build therefore requests 36,864 bytes
for each 256-slot worker injector.

Apollo's exact release-mode probe, rebuilt against this provider tree, reports
936 pool-warmup allocations, a 1,173,414-byte peak, and 1,169,112 retained
bytes. The prior exact retained value was 1,857,224 bytes. The change removes
688,112 retained bytes (37.1%), and no allocation at or above the 65,536-byte
ledger floor remains in the pool-warmup window. The probe's FFT workload and
value assertions complete in 0.02 seconds. The 16-byte difference from the
requested-byte model is allocator or harness bookkeeping below the queue-slot
layout boundary.

The saved same-machine Criterion slope-estimate comparison reports:

| Path | Accepted slope estimate | Change classification |
| --- | --- | --- |
| fresh priority queue push/pop | 871.14 ns [867.28, 875.51] | -1.35%, within noise threshold |
| fresh submission publication | 880.49 ns [874.54, 885.86] | -3.25%, improved |
| maximum-inline construct/drop | 1.3328 ns [1.2974, 1.3919] | no detected change |
| maximum-inline construct/execute | 7.0290 ns [6.9876, 7.0945] | no detected change |
| oversized construct/drop | 22.317 ns [22.176, 22.505] | no detected change |
| oversized construct/execute | 21.564 ns [21.434, 21.771] | -8.40%, improved |
| fresh maximum-inline queue round trip | 896.59 ns [882.66, 917.96] | no detected change |
| fresh oversized queue round trip | 927.27 ns [899.83, 948.20] | no detected change |
| retained-worker maximum-inline dequeue | 100.69 ns [100.42, 100.96] | no detected change |
| retained-worker oversized dequeue | 122.64 ns [122.29, 122.98] | +0.76%, within noise threshold |

Debug and optimized Nextest pass 152/152 executor and utility tests and 68/68
benchmark contracts. Full-workspace Nextest passes 881/881 tests with six
configured skips in 12.0 seconds. Workspace doctests pass, with one documented
ignored example, and warning-denied workspace Clippy and Rustdoc pass. Focused
Miri passes all 9 inline-job tests, including exactly-once drop coverage before
and after execution for inline, oversized, and over-aligned captures. The four
executor Loom models pass 6/6 release tests, and warning-denied AArch64
all-target compilation passes. These checks establish value, layout,
diagnostics, and modeled scheduler-handshake behavior; they do not model the
unchanged generic MPMC implementation itself.
