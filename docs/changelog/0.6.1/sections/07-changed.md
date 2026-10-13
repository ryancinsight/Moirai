### Changed

- Scheduler construction now clears per-worker NUMA assignments unless at
  least two distinct nodes are represented. Single-node and partially known
  layouts therefore skip the same-node steal pass instead of scanning the same
  victims before the unchanged full-ring pass. Multi-node assignments and
  fallback semantics are unchanged; a synchronized three-worker regression now
  proves that a same-node miss reaches a cross-node victim. This is a source-
  and value-test-backed branch-elision result; no latency claim is made without
  a controlled benchmark.
- Cache and owned parallel map now share one panic-safe direct-output owner.
  Each worker writes its disjoint range into the final allocation, publishes a
  compact completion endpoint only after full initialization, and drops its
  initialized prefix if mapping panics; the outer owner drops completed peer
  ranges exactly once. For the 2,097,153-element `u64` fan-out case, the exact
  release ledger remains two allocations while gross bytes fall from
  16,777,624 to 16,777,424: the 16,777,224-byte returned vector is unchanged
  and ancillary metadata falls from 400 bytes of borrowed chunk descriptors to
  200 bytes of completion endpoints. This rejects the original second-output-
  allocation hypothesis because Rust already reused the uninitialized vector
  allocation during collection. Chunk ranges derive their ragged end from the
  remaining logical length, preventing overflow for valid zero-sized sources
  whose length approaches `usize::MAX`. A same-instrument Criterion comparison
  against the pre-change source reports no detected Moirai regression for the new
  fan-out map row (1.4987 ms baseline, 1.5027 ms candidate; change estimate
  -1.68% to +7.96%, p=0.32); no throughput claim is made. The cache
  implementation and allocation-test harness are also split into focused
  modules without changing their public paths.
- Bounded ordered async map/filter now stores completed outputs by retained
  physical slot and reuses each slot's occupancy-discriminated metadata as the
  input-order chain, replacing the separate `Vec<Option<Output>>` position
  ring. The unchanged 1,024-item pending-map ledger at concurrency limits 1 /
  8 / 24 retains 15 / 15 / 15 allocation calls while gross bytes fall from
  16,560 / 17,064 / 18,216 to 16,552 / 17,000 / 18,024, exactly eight bytes per
  reachable `u64` output. An unknown-size sequential source with an unbounded
  configured limit now retains one output cell instead of growing output
  storage with every yielded position. Ordered values, head-of-line behavior,
  geometric growth, stale wakes, cancellation, and exact drop ownership remain
  unchanged. Forced-rebuild pinned-core measurements show the affected ready,
  pending, and sparse Moirai medians moving -2.08%, -1.77%, and -0.30%; no row
  reaches the 5% regression threshold, while control movement precludes a
  throughput claim. This changes no public API, concurrency bound, scheduler
  policy, wake protocol, workload, or timeout.
- Retained async future slots now overlap the output index and intrusive vacancy
  link in one occupancy-discriminated metadata word. On the measured 64-bit
  host, `FutureSlot<PendingOnce<u64>>` falls from 48 to 40 bytes. The unchanged
  1,024-item pending-map ledger at concurrency limits 1 / 8 / 24 retains 15 /
  15 / 15 allocation calls while gross bytes fall from 16,568 / 17,128 /
  18,408 to 16,560 / 17,064 / 18,216, exactly eight bytes per reachable slot.
  Full-width output indices, including `usize::MAX`, remain valid because slot
  occupancy—not a packed sentinel—discriminates the metadata. This changes no
  public API, concurrency bound, scheduler policy, wake protocol, or workload.
  Forced-rebuild pinned-core measurements show no retained Moirai Criterion
  median regressing by 5% or more; host variance precludes a throughput claim.
- Retained async slot blocks now own stable wake identities in one shared
  `WakeBlock` instead of allocating one `Arc<WakeToken>` per in-flight slot.
  The unchanged 1,024-item pending-map ledger at explicit concurrency limits
  1 / 8 / 24 falls from 16 / 23 / 39 allocation calls to 15 / 15 / 15 while
  gross bytes move from 16,568 / 17,296 / 18,960 to 16,568 / 17,128 / 18,408,
  preserving ordered values, cross-thread wake routing, cloned-waker lifetime,
  cancellation, and panic-unwind ownership. Same-binary paired measurements
  show no retained Moirai Criterion median regressing by 5% or more; control
  movement and broad baseline intervals preclude a speedup claim. This changes
  no public API, concurrency bound, scheduler policy, or workload.
- Bounded ordered async map/filter and completion-only for-each now retain
  pinned future blocks and atomic ready bitsets instead of allocating one
  futures-util task node per item. Exact-size sources allocate only their
  clamped reachable concurrency; sources without an upper bound grow pinned
  blocks geometrically after admission, so a large configured ceiling does not
  reserve unreachable storage. On the measured 24-logical-worker x86-64
  Windows host, warmed 1,024-item ready/pending ordered maps fall from 1,035
  allocations / 114,816 gross bytes to 39 / 18,768 and 39 / 18,960; pending
  for-each falls from 1,026 / 98,464 to 29 / 2,232. A fixed block-vacancy
  bitset and intrusive per-slot links make repeated refill O(1) without another
  allocation. Criterion median estimates fall from 66.008 us (95% CI
  65.220-66.690) to 25.326 us (25.225-25.742) for ready map and from
  124.001 us (123.369-124.082) to 47.085 us (46.419-47.254) for one-pending
  map. A same-binary 1,000-slot sparse-wake comparison measures 125.115 us
  (123.034-126.554) for retained storage versus 177.750 us
  (175.229-179.719) for futures-util. Ordered values, configured concurrency,
  cancellation, pinned addresses, and exact drop behavior are unchanged; the
  public completion-order stream remains on futures-util.
- Parallel-context async iterator terminals now reuse the process-wide cached
  parallelism count instead of materializing CPU topology on every call. Map,
  filter, and for-each borrow their operation closure rather than placing it in
  an `Arc`; for-each also drains completion directly instead of collecting a
  throwaway `Vec<()>`. For a warmed 1,024-item ready-future map on the measured
  x86-64 Windows host, the ledger falls from 1,118 allocations / 152,616 gross
  bytes to 1,035 / 114,816, while every ordered result remains exact. The same
  Criterion row's median falls from 81.025 us (95% CI 79.601-82.117 us) to
  63.307 us (95% CI 63.041-63.545 us), a 21.9% reduction. Explicit Async and
  Hybrid concurrency limits are unchanged.
- `cache::ZeroCopyParallelIter` construction now uses the process-wide cached
  parallelism count instead of materializing a NUMA/cache topology snapshot for
  every borrowed slice. Rust 1.97's Linux process-parallelism query reads cgroup
  files and allocates temporary path/read buffers, so the shared cache also
  removes that repeated cost from `ParallelIter` construction. For a 1,024-item
  sequential `u64` map, warmed iterator construction falls from 82 allocations
  and 13,184 gross bytes to zero; map execution retains its sole 8,192-byte
  output allocation. Cache-line prefetch spacing is also clamped to one element,
  so parallel iteration remains defined when an element exceeds one cache line.
  On the measured x86-64 Windows host, the retained
  zero-copy map median falls from 7.0749 us (95% CI 6.8996-7.1410 us) to
  368.13 ns (95% CI 362.14-368.34 ns), a 94.8% reduction.
- `iter_ops::ParallelIter::map` now initializes ordered chunk results directly
  in one full-output allocation. Per-chunk guards drop initialized prefixes if
  a mapper panics, and completed peer ranges remain owned until the scheduler
  joins. Iterator construction reuses the process-wide cached parallelism count
  rather than materializing topology or Linux cgroup-query storage solely to
  derive chunk size. The warmed path allocates exactly three buffers: input chunk
  views, completion ranges, and final output. For 131,072 `u64` outputs on the
  measured x86-64 Windows host, the retained Criterion median falls from 1.3115
  ms to 0.29612 ms (77.4%; disjoint 95% confidence intervals).
- `moirai_async::Notify::notify_waiters` and the shared broadcast-grant path now
  drain pending waiter identifiers directly into one pre-reserved owned-waker
  result. The pending identifier count is a safe capacity upper bound when
  cancelled entries leave lazy holes. At 64 registered waiters this removes the
  transient identifier buffer and geometric result growth, reducing warmed
  allocations from six to one and the retained x86-64 Windows Criterion median
  from 666.99 ns to 343.12 ns (48.6%; disjoint confidence intervals). The
  remaining allocation owns the wakers until callers release the state lock and
  wake them.

[Version 0.6.1 index](../index.md) · [Current changelog](../../../../CHANGELOG.md)
