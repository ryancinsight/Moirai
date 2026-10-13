### Added
- Default `parallel` and `mnemosyne-memory` feature contracts across every
  Moirai package. Crates with existing Mnemosyne-backed runtime behavior expose
  `mnemosyne-memory` as the canonical default forwarding feature; provider-free
  leaf crates expose zero-dependency markers.
- Registered Moirai's global scheduler as Melinoe's `std` partition executor, so branded partition writes can run on the existing Moirai worker pool instead of spawning raw scoped threads.
- Routed default `moirai-executor` worker idle maintenance through Mnemosyne when the runtime is waiting for work, so Moirai's default provider stack performs allocator defragmentation sweeps without Rayon/Tokio involvement.
- Added Apollo-facing public facade contract tests for chunked mutable
  scheduling and non-`Clone` collect-into-existing-storage behavior.
- Removed the stale duplicate top-level `moirai` `par_benchmarks` target
  declaration; `moirai-parallel` remains the benchmark owner.
- Added `moirai_parallel::{join, join_with}` as a Rayon-style two-closure join
  surface with static `ExecutionPolicy` dispatch, scoped scheduler flush plus
  caller-lane execution for forced parallel joins, borrowed non-`'static` tests,
  source contracts, and a value-checked Rayon comparison benchmark row.
- Added `parallel_iterator_regression`, a focused multi-size Moirai/Rayon benchmark matrix for parallel iterator map/reduce, zip/filter collect, borrowed positions, borrowed copied reduce, collect-into-existing-storage, nested flatten/reduce, chunked map/reduce, indexed step/interleave, partition/unzip, and position/find paths.
- Added `moirai_executor::schedule::HybridRouter<P>` with sealed zero-sized route policies and concrete thread/process/server/async-lane route values, plus `process_server_scheduler_routing` for value-checked route-decision benchmarks.
- Added sealed accelerator route metadata for CPU/GPU/TPU/NPU placement with `AcceleratorRoutePolicy`, `AcceleratorCounts`, `AcceleratorKind`, vertical route leaf modules, and value-checked `scheduler_route_accelerator_metadata_summary` benchmark rows.
- Added sealed device payload ownership regions for accelerator route handoff, with pointer-transfer rejection and value-checked `device_region_owned_handoff` benchmark coverage.
- Added the `moirai-transport/scheduler-routes` feature with route-to-address binding for `SchedulerRoute` values and archived local route send/receive helpers.
- Added bounded length-prefixed TCP byte transfer for `NetworkTransport` and `TransportManager` remote addresses.
- Added fixed-format remote task envelopes/results for explicit echo and wrapping-sum request/response execution over remote byte transport.
- Added selected server-route remote task execution through `RoutedRemoteTaskClient<P>`.
- Added real OS process lifecycle primitives in `moirai-transport::process`, including `ProcessSupervisor`, `ProcessSpec`, explicit drop policy, bounded wait, termination, and typed process outcomes.
- Added selected process-route fixed-format task execution through `RoutedProcessTaskClient<P>` and supervised `ProcessEndpoint` child processes.
- Added public fixed-capability routed execution through `Moirai::execute_routed_server_task` and `Moirai::execute_routed_process_task`, requiring sealed `RemoteCapabilityToken<C>` values instead of arbitrary remote closures.
- Split communication primitives into vertical broadcast, collective, message, pub/sub, ring-buffer, and router leaves under `moirai_core::communication`.
- Added `BoundedRemoteTaskServer` for fixed-format remote task execution with persistent listener ownership, bounded request queue capacity, bounded worker count, and accepted/completed value stats.
- Added sealed zero-sized remote capability tokens for constructing only admitted fixed-format remote task operations and keeping arbitrary closure remoting outside process/server transport routes.
- Added sealed thread/process/server/device `TransportPayload<R>` ownership regions for archive byte handoff, with process/server/device pointer-transfer rejection and Mnemosyne global allocator evidence pinned at the top-level crate feature.
- Added `process_server_routed_execution`, a value-checked benchmark for selected server-route and process-route fixed-format task execution through real TCP request/result frames and supervised child processes.
- Added `metrics_collector_comparison`, a value-checked Criterion target for
  shared counter handles, fixed-size metric snapshots, and Prometheus export.
- Added `async_iterator_comparison`, a value-checked `moirai-iter::AsyncIterator` ready-pipeline benchmark against Tokio `JoinSet` fan-out.
- Added `AsyncIterator::take` and `AsyncIterator::skip` logical-window adapters with value tests and a Tokio `JoinSet` comparison row.
- Added `AsyncIterator::enumerate` and `AsyncIterator::zip` logical-position/pairing adapters with value tests and a Tokio `JoinSet` comparison row.
- Added `ParallelSliceMut` sorting comparison coverage against Rayon `ParallelSliceMut`.
- Added `async_fs_comparison`, a value-checked `moirai_async::fs::read` benchmark against Tokio `fs::read`.
- Added an `async_fs_write_file` row to `async_fs_comparison`, comparing Moirai platform-write facade behavior against Tokio `fs::write` over the same 64 KiB payload.
- Added an `async_fs_append_file` row to `async_fs_comparison`, comparing Moirai platform-append facade behavior against Tokio append-open/write behavior over the same 64 KiB payload.
- Added `moirai_async::fs::metadata` and an `async_fs_metadata_file` row to `async_fs_comparison`, comparing Moirai platform-metadata facade behavior against Tokio `fs::metadata` over the same 64 KiB file.
- Added `moirai_async::fs::rename` and an `async_fs_rename_file` row to `async_fs_comparison`, comparing Moirai platform-rename facade behavior against Tokio `fs::rename` over prepared 64 KiB source files.
- Added an `async_fs_remove_file` row to `async_fs_comparison`, comparing Moirai platform-remove facade behavior against Tokio `fs::remove_file` over prepared 64 KiB source files.
- Added `async_fs_dir_comparison`, comparing Moirai directory create/remove and recursive create/remove facade behavior against Tokio directory facade operations.
- Added an `async_fs_copy_file` row to `async_fs_comparison`, comparing Moirai platform-copy facade behavior against Tokio `fs::copy` over the same 64 KiB file.
- Added TCP and UDP loopback value tests for the Moirai-owned async network facade.
- Added `async_tcp_comparison`, a value-checked Moirai TCP facade echo benchmark against Tokio `TcpListener`/`TcpStream`.
- Added a persistent TCP stream echo row to `async_tcp_comparison` to isolate established-stream read/write behavior against Tokio `TcpStream`.
- Added a TCP write-shutdown row to `async_tcp_comparison` to verify payload delivery and peer EOF against Tokio.
- Added `async_tcp_backpressure_comparison`, a value-checked TCP write-backpressure benchmark against Tokio over bounded socket buffers.
- Added `async_tcp_readiness_comparison`, a value-checked TCP read-readiness benchmark against Tokio that asserts `Poll::Pending` before peer data and exact payload delivery after release.
- Added `async_tcp_cancel_safety_comparison`, a value-checked TCP pending-read cancellation benchmark against Tokio that drops a pending borrowed read future before asserting unchanged caller buffer state and later payload delivery.
- Added `async_io_compat_comparison`, a value-checked benchmark for feature-gated Tokio I/O trait compatibility wrappers over native Moirai readers and writers.
- Added zero-copy `AsyncReadExt::read_exact` and `AsyncWriteExt::shutdown` futures for the native Moirai I/O trait surface.
- Added `async_udp_comparison`, a value-checked Moirai UDP facade receive benchmark against Tokio `UdpSocket::recv_from`.
- Added `map_with` and `map_init` stateful map adapters to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `update` mutation adapter to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `intersperse` separator adapter to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `flatten` nested-stream adapter to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `flat_map_iter` and `flatten_iter` serial-inner adapter methods to the Rayon-style parallel iterator subset with value tests and corrected same-run Rayon comparison rows.
- Added `take_any` and `skip_any` bounded-window adapters to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `take_any_while` and `skip_any_while` deterministic predicate-window adapters to the Rayon-style parallel iterator subset with value tests and a full-pass same-run Rayon benchmark row.
- Added `real_application_mixed_workload`, a value-checked mixed async/parallel/channel benchmark against a Tokio plus Rayon reference path.
- Refreshed public result-handle, async wake, scheduler handoff, and Criterion variance attribution evidence against Tokio and Rayon comparison rows.
- Added `sum`, `product`, `min`, and `max` terminal reducers to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `min_by`, `max_by`, `min_by_key`, and `max_by_key` ordered terminal reducers to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `find_map_first` and `find_map_any` predicate terminals to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `find_last` and `find_map_last` reverse-order predicate terminals to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `position_first`, `position_any`, and `position_last` predicate terminals to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `positions` to the Rayon-style parallel iterator subset with owned, borrowed, and mapped value tests plus a same-run Rayon benchmark row.
- Added `for_each_with` and `for_each_init` stateful side-effect terminals to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `try_for_each_with` and `try_for_each_init` fallible stateful side-effect terminals to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `while_some` optional-stream adapter to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `try_for_each` fallible side-effect terminal to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `try_reduce` fallible reduction terminal to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `try_reduce_with` fallible no-identity reduction over sealed `Option<T>` and `Result<T, E>` stream items with value tests and a same-run Rayon benchmark row.
- Added `copied` and `cloned` borrowed-reference adapters to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `iterator_adapter_non_clone_ref_map`, a value-checked borrowed `Vec<T>::par_iter` benchmark row against Rayon over non-`Clone` values.
- Added `IndexedParallelIterator::collect_into_vec` for exact-size source iterators with non-`Clone` move tests and a same-run Rayon benchmark row.
- Added `IndexedParallelIterator::unzip_into_vecs` for exact-size pair sources with non-`Clone` move tests and a same-run Rayon benchmark row over caller-provided output storage.
- Added indexed `interleave` and `interleave_shortest` adapters for exact-size sources with non-`Clone` move tests, shortest-tail drop checks, and a same-run Rayon benchmark row.
- Added indexed `step_by` for exact-size sources with non-`Clone` move tests, skipped-value drop checks, exact-length tests, and a same-run Rayon benchmark row.
- Added indexed `by_exponential_blocks` and `by_uniform_blocks` logical-output block adapters with non-`Clone` move tests, zero-sized policy markers, zero-size rejection, and a same-run Rayon benchmark row.
- Added `collect_vec_list` to the Rayon-style parallel iterator subset with non-`Clone` move tests and a same-run Rayon benchmark row.
- Added `zip_eq` equal-length pairing to the Rayon-style parallel iterator subset with value tests, mismatch-panic coverage, and a same-run Rayon benchmark row.
- Added `partition_map` with a public `Either<L, R>` sum type to the Rayon-style parallel iterator subset with value tests and a same-run Rayon benchmark row.
- Added `unzip` pair-stream collector to the Rayon-style parallel iterator subset with value tests and benchmark rows.
- Added `IndexedParallelIterator::{len, is_empty}` for exact-size parallel iterator sources with value tests and a same-run Rayon metadata benchmark row.
- Added `iter_ops_parallel_comparison`, a value-checked scoped `ParallelIter` map/reduce benchmark against Rayon.
- Added `cache_iterator_comparison`, a value-checked borrowed-slice `ZeroCopyParallelIter` map/reduce benchmark against Rayon, including a large borrowed reduce row that exercises the cache scoped-spawn gate.
- Added `execution_context_comparison`, a value-checked owned `ParallelContext::execute_iter` benchmark against Rayon.
- Added `numa_context_comparison`, a value-checked owned `NumaContext::execute_iter` benchmark against Rayon.
- Added `distributed_context_comparison`, a value-checked owned `DistributedContext::execute_distributed_map` benchmark against Rayon.
- Added `multi_system_context_comparison`, a value-checked owned `MultiSystemContext::execute_heterogeneous_compute` benchmark against Rayon.
- Added PAL async file value tests, PAL TCP/UDP delayed loopback progress tests for the no-active-reactor self-wake path, and a Linux epoll wake-path test.
- Added a PAL reactor task-handle completion regression test for spawned ready tasks.
- Added `standalone_deque_reclaim_policy`, a value-checked diagnostic benchmark for `ChaseLevDeque` quiescent versus shared epoch reclamation.
- Added bounded channel matrix coverage to the Tokio comparison audit, comparison report, and benchmark-contract surface.
- Added sealed generic `moirai_utils::simd` scalar contracts and a benchmark source contract rejecting the removed type-suffixed utility SIMD public surface.
- Added a vector-prefix/tail SIMD benchmark row for non-lane-multiple `f32` slice lengths with value assertions.
- Added a value-checked wide-real SIMD vector-addition benchmark row under the generic `moirai_utils::simd::add<T>` public API.
- Added registry-local and external-ID token lifecycle diagnostic rows to `result_handle_diagnostics`, with benchmark contracts tying both rows to the production registry token path.
- Added a registry-owned after-send quiescent diagnostic row to separate public result availability from worker-tail metrics completion.
- Added a refreshed native Rayon/Tokio gap-closure benchmark snapshot covering result handles, scheduler rows, async iterator rows, and selected Rayon adapter rows.

