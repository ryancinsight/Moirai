### Changed
- Replaced `moirai-iter::base` adapter dead-field suppressions with live
  accessor and `into_parts` APIs for `BaseIterator`, `MapAdapter`,
  `FilterAdapter`, and `BatchAdapter`; moved base tests into a vertical
  `base/tests.rs` leaf.
- Split `moirai-iter::async_iter` into vertical traits, sources, adapters,
  consumers, and bounded-parallel leaves; removed dead source cursor fields so
  `AsyncVecIter<T>` stays the size of `Vec<T>` and `AsyncRangeIter` stays the
  size of `Range<usize>`.
- Consolidated executor idle-maintenance and PAL active-reactor thread-local
  state through Melinoe `thread_cached!` leaves, removing duplicated nightly
  `#[thread_local]` and stable `thread_local!` branches.
- Removed obsolete `remote_stage("gpu-cluster", ...)` examples from historical
  architecture notes and replaced them with the fixed-capability routed facade
  boundary.
- Removed dead thread-local cache declaration from `moirai-core::pool::GlobalPool::get`; the active implementation uses the global pool path.
- Removed the stale root Mnemosyne Git patch override that still forced
  `../mnemosyne` path resolution; `Cargo.lock` now resolves Mnemosyne crates
  from upstream GitHub `main` commit
  `8a428c4ce72786ff4a28a94342d8e724a36319a3`.
- Removed the local Mnemosyne Git patch override and obsolete repository-local dependency copy, locked Moirai's GitHub Mnemosyne dependency to upstream `ryancinsight/Mnemosyne` `main` commit `4f8d84b91780d2b1f7b27ede29580dffe2bff9c9`, and reran allocator, TLS, Rayon-facing parallel iterator, cache iterator, and process/server routed execution benchmarks.
- Replaced `moirai-metrics` placeholder storage with vertical collector,
  counter, gauge, histogram, snapshot, and exporter leaves backed by real
  shared state and deterministic value snapshots.
- Replaced the PAL timer immediate-ready placeholder with a deadline-sensitive
  future that registers a waker, returns `Pending` before the deadline, and
  wakes at completion.
- Replaced the distributed iterator fixed 10 second completion estimate with an
  input-sensitive model using task count, node CPU capacity, reliability,
  latency, and aggregate bandwidth, including saturating behavior for extreme
  telemetry.
- Changed borrowed vector `positions`, borrowed copied map/filter/sum, chunked map/sum, vector-backed indexed step/interleave enumerate/map/sum, nested flatten/map/filter/sum, and zip_eq/map/filter/collect parallel iterator shapes to avoid intermediate materialization on the focused regression paths.
- Moved the public `moirai-iter::MoiraiIterator` facade into a vertical `facade` module, preserved execution contexts through the `ExecutionContext` enum instead of string matching, and removed silent error-to-empty fallback branches.
- Changed the public Moirai facade documentation to state that cross-machine remote closure execution is outside the active API until a transport-backed task contract exists.
- Changed the benchmark crate feature declarations to forward Moirai `async`, `iter`, `local`, and `mnemosyne` features explicitly instead of inheriting `moirai` defaults implicitly.
- Changed Mnemosyne OS TLS key fast-path lookup to use a relaxed scalar load; the published value is only the OS TLS key, not allocator slot contents.
- Changed standalone deque steal paths to remove steal-side `SeqCst` fences while retaining acquire observations and `SeqCst` ownership CAS operations.
- Changed utility SIMD benchmarks and benchmark setup contracts to call generic `moirai_utils::simd` operations instead of type-suffixed vector functions.
- Changed generic `f32` SIMD dispatch to process native vector prefixes plus scalar tails for non-lane-multiple lengths and record those operations as vectorized when a native backend is available.
- Changed utility SIMD wide-real native dispatch reporting to stay x86 AVX2-specific, matching the implemented backend instead of over-reporting native support on unsupported architectures.
- Moved `HybridExecutor` public task ID allocation into `TaskRegistry::register_next_task`, removing the executor-local `AtomicU64` while reusing the existing registry registration boundary.
- Split `result_handle_diagnostics` wrapper, scheduler-tail, primitive, and registry rows into dedicated vertical leaves while preserving benchmark names and production-path contracts.
- Changed `moirai_async::fs::copy` to delegate to the PAL platform copy operation instead of allocating a user-space 64 KiB transfer buffer.
- Changed `moirai_async::fs::write` to delegate to the PAL platform write operation over the caller-provided byte slice instead of constructing a facade file handle, updating stats, looping through writes, and unconditionally syncing.
- Changed `moirai_async::fs::append` to delegate to the PAL platform append operation over the caller-provided byte slice instead of constructing a facade file handle, updating stats, looping through writes, and unconditionally syncing.
- Changed async file facade metadata lookup to use PAL platform metadata without constructing an async file handle or stats state.
- Changed async file facade rename to use PAL platform rename without reading or copying file contents through user-space buffers.
- Changed async file facade removal to use PAL platform remove without constructing an async file handle or stats state.
- Changed async directory facade creation and removal to use PAL platform directory operations instead of direct async-layer `std::fs` ownership.
- Replaced async iterator terminal placeholder futures with value-semantic `collect`, `for_each`, `fold`, and `reduce` execution over the logical iterator stream.
- Changed `moirai_iter::iter_ops::ParallelIter` to own `Vec<T>` directly, borrow scoped chunks without `Arc<Vec<T>>`, accept non-`'static` closures, and use the bounded scheduler batch capacity as the scoped-spawn cost gate.
- Changed `moirai_iter::cache::ZeroCopyParallelIter` map execution to borrow scoped slices and closures directly instead of allocating `Arc` wrappers for borrowed data and functions.
- Changed `moirai_iter::cache::ZeroCopyParallelIter` reduce execution to move owned intermediate partials through pair compaction, accept non-`Clone` reducer closures, and gate scoped OS-thread fanout behind a scheduler-batch cache-chunk floor.
- Changed `moirai_iter::execution` direct iterator execution to move owned chunks instead of cloning chunk slices, removing `T: Clone` from direct execution-context map bounds.
- Changed `moirai_iter::numa` map and reduce execution to consume owned batches instead of cloning chunk slices, removing `T: Clone` from NUMA map and extension bounds.
- Changed `moirai_iter::distributed` map, reduce, and partition helpers to consume owned partitions, produce value-semantic map results, and remove clone-bound direct distributed item paths.
- Changed `moirai_iter::multi_system` partition, distribution, and heterogeneous map helpers to consume owned partitions without clone-bound direct item paths and to return value-semantic mapped results.
- Changed `moirai_iter::parallel::IntoParallelRefIterator for Vec<T>` to borrow non-`Clone` values with `T: Send + Sync + 'data` instead of requiring `T: Clone + 'static`.
- Changed exact-size vector source collection to bulk-move owned items into caller-provided `Vec` spare capacity without cloning.
- Changed `ParAsyncMap`, `ParAsyncFilter`, and `ParAsyncForEach` to use bounded in-flight polling through their `concurrency` parameter.
- Raised the unstable sorting sequential threshold so medium slices use Rust's optimized unstable sort until worker dispatch amortizes.
- Replaced placeholder-only async file tests with value-semantic file operation tests.
- Removed the obsolete async file placeholder future wrapper and documented Tokio reactor-native I/O compatibility as a deferred PAL boundary rather than a covered facade benchmark.
- Changed PAL TCP/UDP `WouldBlock` fallback without an active reactor to wake the current task before returning `Pending`.
- Changed async file, TCP, and UDP Tokio comparison benchmarks so Moirai rows use `Moirai::block_on` instead of `futures::executor::block_on`.
- Changed the Moirai TCP stream facade to expose TCP_NODELAY for low-latency stream benchmark parity with Tokio.
- Changed Moirai TCP stream shutdown from a no-op to PAL write-side socket shutdown.
- Changed the Moirai TCP stream facade to expose `from_std` so tests and benchmarks can wrap preconfigured sockets without copying payload buffers or depending on Tokio.
- Changed `TokioCompat<T>` and `MoiraiCompat<T>` into transparent conversion wrappers with `From<T>` constructors and value-semantic compatibility tests.
- Changed the persistent TCP stream benchmark to use the production Moirai `AsyncReadExt::read_exact` and `AsyncWriteExt::write_all` futures.
- Replaced the PAL reactor pending-only task handle with per-task atomic completion state and waker publication.
- Replaced PAL platform reactor boxing with compile-target `PlatformReactor` static dispatch.
- Replaced PAL reactor queued future `Pin<Box<dyn Future>>` storage with bounded inline future storage and monomorphized poll/drop future erasure.
- Replaced the Linux epoll reactor no-op wake placeholder with an internal `eventfd` wake path.
- Replaced public core and scheduler `Box<dyn BoxedTask>` / `dyn Scheduler` task surfaces with `ScheduledTask` inline storage and monomorphized execute/drop/context erasure.
- Replaced standalone scheduler `ChaseLevDeque` per-item boxed task nodes with contiguous `UnsafeCell<MaybeUninit<T>>` ring slots, sealed zero-sized quiescent reclamation, and opt-in shared epoch reclamation.
- Changed all current comparison benchmark targets to carry explicit Criterion measurement and warm-up bounds under benchmark-contract coverage.

