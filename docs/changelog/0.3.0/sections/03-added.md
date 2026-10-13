### Added
- `moirai-executor`: added `block_on`, a public current-thread parking wait
  primitive over the existing scheduler waker driver. This gives Moirai-owned
  crates a lightweight replacement for external `pollster::block_on` without
  constructing the global scheduler.
- `moirai-async`: added a loopback TCP `timeout(read)` regression that lets the
  timer complete before peer data arrives, then forces the socket reactor to
  wake the stale read waker after the async task has completed.
- `moirai-parallel`: added `for_each_chunk_mut_with_state`, a mutable chunk
  primitive with one reusable scratch state per scheduled worker shard. This
  fills the provider gap for consumers replacing Rayon `for_each_with`/
  `for_each_init` chunk kernels without allocating scratch per logical chunk.
- `moirai-parallel`: added `for_each_chunk_triple_mut_enumerated_with` and
  `for_each_chunk_quad_mut_enumerated_with` for fused provider-owned updates
  across three or four equal-length mutable output buffers.
- `moirai-executor`: loom model of the scheduler park/wake handshake
  (`tests/loom_wake_handshake.rs`, `cfg(loom)`-gated) — exhaustively verifies
  the `pending_tasks`/idle-bitset `SeqCst` Dekker handshake never loses a
  wakeup, the model-checked evidence that the `SeqCst` ordering is necessary
  and sufficient. The crate now wires the `cfg(loom)` dev-dependency mirroring
  `moirai-scheduler`.
- `moirai-iter`: Added bounded concurrent stream adapters
  `ConcurrentStreamExt::concurrent_filter_map` and
  `ConcurrentStreamExt::concurrent_filter`.
- Scheduler DIP seam: `moirai_executor::schedule::{WorkScheduler, WorkSubmit,
  SchedulerControl, DataParallel}` — ISP-segregated role traits implemented by
  `ThreadScheduler`. `HybridExecutor<S: WorkScheduler = ThreadScheduler>` now
  depends on this contract rather than the concrete scheduler, so a substitute
  (e.g. a single-threaded `wasm32` scheduler) can be plugged in via
  `HybridExecutor<S>`; the default type parameter keeps every existing call site
  unchanged.

