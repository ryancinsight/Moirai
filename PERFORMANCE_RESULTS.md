# Moirai Performance Benchmark Results

This document reports executable Criterion benchmark results for the unified scheduler comparison work. Tokio and Rayon are used only as benchmark dependencies.

## 2026-06-13 Iterator Base Adapter Cleanup Refresh

Command:
```bash
cargo bench -p moirai-benchmarks --bench iter_ops_parallel_comparison -- --quick --quiet
```

Workload: same value-checked `moirai-iter::iter_ops::ParallelIter` map/reduce
rows after replacing base adapter dead-field suppressions with accessor and
`into_parts` APIs.

| Benchmark | Moirai | Reference |
| --- | ---: | ---: |
| Iter ops parallel map, 8,192 | 7.4445-7.6436 us | Rayon 57.798-57.920 us |
| Iter ops parallel reduce, 8,192 | 1.7509-1.7544 us | Rayon 51.995-52.786 us |

Interpretation: the adapter-base cleanup does not regress the covered
monomorphized iterator map/reduce paths against the same-run Rayon references.

## 2026-06-13 Async Iterator Vertical Split Refresh

Command:
```bash
cargo bench -p moirai-benchmarks --bench async_iterator_comparison -- --quick --quiet
```

Workload: same value-checked async iterator pipelines after splitting
`moirai-iter::async_iter` into vertical leaves and removing unused source
cursor fields. Each Moirai and Tokio row asserts equal output before timing.

| Benchmark | Moirai | Reference |
| --- | ---: | ---: |
| Async iterator ready pipeline | 373.81-377.53 us | Tokio `JoinSet` 26.226-27.019 ms |
| Async iterator take/skip pipeline | 154.00-154.96 us | Tokio `JoinSet` 25.101-26.116 ms |
| Async iterator enumerate/zip pipeline | 572.72-579.71 us | Tokio `JoinSet` 48.887-50.156 ms |
| Bounded async iterator pipeline | 2.0485-2.0568 ms | Tokio `JoinSet` 10.801-10.935 ms |

Interpretation: the vertical split and source-layout cleanup preserve the
covered async iterator advantage over the same-run Tokio `JoinSet` references.
This is empirical benchmark evidence for the audited async iterator subset, not
a claim of full Tokio stream ecosystem parity.

## 2026-06-01 Public Distributed Facade Cleanup

Command:
```bash
cargo bench -p moirai-benchmarks --bench distributed_context_comparison -- distributed_context_owned_map --quiet
```

Workload: `moirai-iter::distributed::DistributedContext::execute_distributed_map` consumes owned partitions and is compared with Rayon `into_par_iter` over equivalent owned vectors. The benchmark asserts equal checksums before timing. The public `Moirai` facade does not expose remote-closure methods; later routed execution coverage admits only sealed fixed-format capability tasks.

| Benchmark | Moirai | Reference |
| --- | ---: | ---: |
| Owned distributed context map, 512 | 357.70-361.39 ns | Rayon 26.111-29.445 us |

Interpretation: the distributed helper row remains value-checked and below the same-run Rayon owned-map reference for the audited helper boundary. This does not claim full distributed networking, Rayon adapter parity, or facade-level remote closure execution.

## 2026-06-01 All-Target Example Cleanup Benchmark Rerun

Commands:
```bash
cargo bench -p moirai-benchmarks --no-run
cargo bench -p moirai-benchmarks -- --quiet
cargo bench -p moirai-benchmarks --bench simd_benchmarks -- vector_prefix_tail_addition --quiet
cargo bench -p moirai-benchmarks --bench simd_benchmarks -- vector_addition_wide --quiet
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- public_result_handle_ready --quiet
cargo bench -p moirai-benchmarks --bench thread_schedule_comparison -- --quiet
cargo bench -p moirai-benchmarks --bench thread_schedule_comparison -- standalone_deque_reclaim_policy --quiet
cargo bench -p moirai-benchmarks --bench iterator_adapter_comparison -- --quiet
cargo bench -p moirai-benchmarks --bench async_tcp_comparison -- --quiet
```

`cargo bench -p moirai-benchmarks --no-run` compiled all benchmark targets. The full package run exceeded the 300 second local gate after completing async file, async directory, async I/O, async iterator, TCP backpressure, TCP cancellation, and part of TCP comparison coverage. The maintained comparison targets below completed individually under the same 300 second per-command bound.

| Benchmark | Moirai | Reference |
| --- | ---: | ---: |
| SIMD prefix/tail add, 65 | 10.593-11.496 ns | Scalar 54.657-85.843 ns |
| SIMD prefix/tail add, 4,099 | 303.97-497.13 ns | Scalar 3.4924-5.9176 us |
| SIMD prefix/tail add, 16,385 | 1.5658-2.0635 us | Scalar 14.469-20.229 us |
| SIMD wide add, 64 | 12.688-13.492 ns | Scalar 51.079-53.204 ns |
| SIMD wide add, 4,096 | 523.56-574.79 ns | Scalar 3.3056-3.5587 us |
| SIMD wide add, 16,384 | 2.5845-2.6198 us | Scalar 14.573-15.380 us |
| Ready result handle | 509.05-681.55 ns | Tokio 1.5601-2.5055 us |
| Scoped ready schedule | 12.700-12.955 us | Tokio 2.1801-3.5275 ms; Rayon 47.117-98.944 us |
| Indexed reduce schedule | 454.94-658.77 ns | Rayon 3.8028-6.6470 us |
| Mixed unified schedule | 39.542-40.067 us | Tokio plus Rayon 605.57-629.84 us |
| Real application mixed workload | 92.002-94.008 us | Tokio plus Rayon 677.24-694.88 us |
| Standalone deque quiescent reclaim | 2.1955-2.2040 us | Shared epoch reclaim 6.3355-6.4715 us |
| Iterator indexed collect-into-vec | 52.772-54.366 us | Rayon 94.521-99.820 us |
| Iterator collect-vec-list | 75.452-84.155 us | Rayon 471.67-490.59 us |
| TCP loopback echo, 24 bytes | 309.05-339.12 us | Tokio 358.13-370.59 us |
| TCP persistent stream echo, 24 bytes | 17.764-19.724 us | Tokio 23.766-24.201 us |
| TCP write shutdown, 19 bytes | 445.28-461.16 us | Tokio 494.87-503.21 us |

Interpretation: example cleanup does not regress the audited scheduler, iterator, SIMD, or TCP comparison rows. The TCP target now creates persistent sockets immediately before the persistent-stream group, preventing the benchmark harness from timing out an idle persistent stream while earlier groups run.

## 2026-06-01 SIMD Vector Prefix/Tail Addition Row

Command:
```bash
cargo bench -p moirai-benchmarks --bench simd_benchmarks -- vector_prefix_tail_addition --quiet
```

Workload: generic `moirai_utils::simd::add<f32>` runs over non-lane-multiple sizes and is compared with a scalar loop. The benchmark asserts output equality before timing. This is a zero-cost utility invariant row, not a Rayon/Tokio competitive row.

| Elements | Generic prefix/tail | Scalar |
| ---: | ---: | ---: |
| 65 | 11.753-11.796 ns | 50.164-50.480 ns |
| 4,099 | 123.85-124.32 ns | 3.1101-3.1287 us |
| 16,385 | 1.0764-1.0870 us | 12.988-13.049 us |

Interpretation: non-lane-multiple `f32` slices now use a native vector prefix plus scalar tail when the native backend is available, and dispatch accounting records the covered operation as vectorized.

## 2026-06-01 SIMD Wide Vector Addition Row

Command:
```bash
cargo bench -p moirai-benchmarks --bench simd_benchmarks -- vector_addition_wide --quiet
```

Workload: generic `moirai_utils::simd::add<T>` runs over wide real slices and is compared with a scalar loop. The benchmark asserts output equality before timing. This is a zero-cost utility invariant row, not a Rayon/Tokio competitive row.

| Elements | Generic wide | Scalar |
| ---: | ---: | ---: |
| 64 | 12.688-13.492 ns | 51.079-53.204 ns |
| 256 | 18.065-24.843 ns | 202.88-217.19 ns |
| 1,024 | 60.959-65.341 ns | 796.22-876.43 ns |
| 4,096 | 523.56-574.79 ns | 3.3056-3.5587 us |
| 16,384 | 2.5845-2.6198 us | 14.573-15.380 us |

Interpretation: the generic public API dispatches to the private x86 AVX2 wide backend when available, falls back to scalar elsewhere, and keeps benchmark setup value-checked before timing.

## 2026-06-01 Iterator Collect-Vec-List Row

Command:
```bash
cargo bench -p moirai-benchmarks --bench iterator_adapter_comparison -- iterator_adapter_collect_vec_list --quiet
```

Workload: owned streams are mapped, filtered, collected through `ParallelIterator::collect_vec_list`, and summarized by flattening the returned segment list. The benchmark source asserts equal flattened `(len, sum, xor)` summaries for Moirai and Rayon before timing. Unit tests cover non-`Clone` value movement and empty-list behavior.

| Benchmark | Moirai | Reference |
| --- | ---: | ---: |
| Iterator `collect_vec_list` | 18.349-18.558 µs | Rayon 315.88-327.29 µs |

Interpretation: the bounded Rayon-style subset now includes the `collect_vec_list` terminal return shape without clone bounds or dynamic strategy state. Segment count is not part of the asserted semantic contract.

## 2026-06-01 Iterator Indexed Block Adapter Row

Command:
```bash
cargo bench -p moirai-benchmarks --bench iterator_adapter_comparison -- iterator_indexed_blocks --quiet
```

Workload: exact-size owned sources are routed through `IndexedParallelIterator::by_exponential_blocks` before a `find_first` query and through `IndexedParallelIterator::by_uniform_blocks(257)` before a mapped/filtering collection. The benchmark source asserts equal Moirai and Rayon `(first, collected)` outputs before timing. Unit tests cover non-`Clone` value movement, zero-sized block policy markers, and zero-size rejection for uniform blocks.

| Benchmark | Moirai | Reference |
| --- | ---: | ---: |
| Iterator indexed block adapters | 30.128-32.300 µs | Rayon 4.4301-4.5698 ms |

Interpretation: the bounded indexed source boundary now includes Rayon-named block adapter methods as value-preserving logical-output adapters without dynamic strategy objects or clone bounds. Full Rayon indexed producer/consumer block-scheduling semantics remain outside the audited subset.

## 2026-06-01 Iterator Serial-Inner Flatten Rows

Commands:
```bash
cargo bench -p moirai-benchmarks --bench iterator_adapter_comparison -- iterator_adapter_filter_flat_pipeline --quiet
cargo bench -p moirai-benchmarks --bench iterator_adapter_comparison -- iterator_adapter_flatten --quiet
```

Workload: owned streams are routed through `filter_map` plus `flat_map_iter`, and nested owned streams are routed through `flatten_iter`. The benchmark source asserts equal Moirai and Rayon output vectors before timing. Unit tests cover left-to-right flattened value order, a non-`Sync` serial inner iterator for `flat_map_iter`, and serial range flattening for `flatten_iter`.

| Benchmark | Moirai | Reference |
| --- | ---: | ---: |
| Iterator `filter_map`/`flat_map_iter` | 79.134-123.35 µs | Rayon 393.06-405.26 µs |
| Iterator `flatten_iter` | 73.234-74.541 µs | Rayon 150.08-155.19 µs |

Interpretation: the audited Rayon-style subset now exposes Rayon-named serial-inner flattening methods and benchmarks them against the matching Rayon serial-inner APIs. This does not claim Rayon's nested-parallel `flat_map`/`flatten` producer model.

## 2026-06-01 Iterator Indexed Step-By Row

Command:
```bash
cargo bench -p moirai-benchmarks --bench iterator_adapter_comparison -- iterator_indexed_step_by --quiet
```

Workload: exact-size owned sources are filtered through `IndexedParallelIterator::step_by(3)`, mapped, and collected. The benchmark source asserts equal Moirai and Rayon output vectors before timing, and unit tests cover non-`Clone` value movement, exact indexed length, zero-step rejection, and skipped-value drops.

| Benchmark | Moirai | Reference |
| --- | ---: | ---: |
| Iterator indexed `step_by` | 24.335-25.830 us | Rayon 65.191-67.990 us |

Interpretation: the bounded indexed source boundary now includes fixed-stride exact-size source selection without clone bounds, boxed dispatch, or runtime strategy objects. Full Rayon indexed producer/consumer parity remains outside the audited subset.

## 2026-06-01 Iterator Indexed Interleave Row

Command:
```bash
cargo bench -p moirai-benchmarks --bench iterator_adapter_comparison -- iterator_indexed_interleave --quiet
```

Workload: exact-size owned sources are alternated through `IndexedParallelIterator::interleave` and `interleave_shortest`, then collected. The benchmark source asserts equal Moirai and Rayon full-interleave and shortest-interleave vectors before timing, and unit tests cover non-`Clone` value movement plus exact tail drops through both shortest-input shapes.

| Benchmark | Moirai | Reference |
| --- | ---: | ---: |
| Iterator indexed `interleave` / `interleave_shortest` | 401.13-439.28 us | Rayon 433.44-453.31 us |

Interpretation: the bounded indexed source boundary now includes alternating exact-size source composition without adding boxed dispatch, dynamic strategy state, or clone bounds. Full Rayon indexed producer/consumer parity remains outside the audited subset.

## 2026-05-29 Iterator Indexed Unzip-Into-Vecs Row

Command:
```bash
cargo bench -p moirai-benchmarks --bench iterator_adapter_comparison -- iterator_indexed_unzip_into_vecs --quiet
```

Workload: exact-size owned pair sources are split into caller-provided left and right `Vec` storage through `IndexedParallelIterator::unzip_into_vecs`. The benchmark source asserts equal Moirai and Rayon output vectors plus side checksums before timing, and unit tests cover non-`Clone` pair movement into preallocated storage.

| Benchmark | Moirai | Reference |
| --- | ---: | ---: |
| Iterator indexed `unzip_into_vecs` | 256.72-273.34 us | Rayon 268.81-303.00 us |

Interpretation: the bounded indexed source boundary now includes caller-provided pair splitting in addition to `collect_into_vec`. The implementation reuses destination allocations and moves pair sides exactly once without adding boxed dispatch, dynamic strategy state, or clone bounds.

## 2026-05-29 Iterator Take/Skip-Any-While Adapter Row

Command:
```bash
cargo bench -p moirai-benchmarks --bench iterator_adapter_comparison -- iterator_adapter_take_skip_any_while --quiet
```

Workload: mapped owned streams are routed through full-pass predicate-window `take_any_while` and `skip_any_while` paths. The benchmark source asserts equal Moirai and Rayon vectors before timing, and unit tests cover deterministic prefix/suffix early-stop semantics. Threshold early-stop equality is not claimed because Rayon permits unordered early-stop behavior for these APIs.

| Benchmark | Moirai | Reference |
| --- | ---: | ---: |
| Iterator `take_any_while`/`skip_any_while` | 91.813-102.11 us | Rayon 729.10-756.49 us |

Interpretation: the audited Rayon-style predicate-window subset now includes deterministic prefix/suffix APIs without adding dynamic dispatch or boxed strategy state. The comparison row is intentionally limited to a full-pass predicate window where Rayon and Moirai have identical retained values.

## 2026-05-29 Iterator Positions Adapter Row

Command:
```bash
cargo bench -p moirai-benchmarks --bench iterator_adapter_comparison -- iterator_adapter_positions --quiet
```

Workload: owned vector streams are mapped and then filtered through `positions` to collect every matching logical index. The benchmark source asserts equal Moirai and Rayon index vectors before timing, and unit tests cover owned, borrowed, and mapped streams.

| Benchmark | Moirai | Reference |
| --- | ---: | ---: |
| Iterator `positions` | 11.248-11.339 us | Rayon 234.78-239.80 us |

Interpretation: the audited Rayon-style predicate subset now includes an index-stream adapter in addition to single-index terminals. The mapped path routes through a fused `MapPositions` adapter, so mapped values are consumed directly while only matching indices are materialized.

## 2026-05-29 Iterator Try-Reduce-With Terminal Row

Command:
```bash
cargo bench -p moirai-benchmarks --bench iterator_adapter_comparison -- iterator_adapter_try_reduce_with --quiet
```

Workload: owned vector streams are mapped into `Result<u64, u64>` values and reduced with `try_reduce_with` without an identity value. The benchmark source asserts equal Moirai and Rayon `Option<Result<_, _>>` outputs before timing, and unit tests cover success, first-error, empty, and `Option::None` paths.

| Benchmark | Moirai | Reference |
| --- | ---: | ---: |
| Iterator `try_reduce_with` | 8.5426-8.7513 us | Rayon 64.753-66.248 us |

Interpretation: the audited Rayon-style fallible reducer subset now includes no-identity fallible reduction over a sealed local `TryStreamItem` contract. The mapped fast path streams transformed fallible values directly into the reducer instead of materializing an intermediate mapped vector.

## 2026-05-29 Iterator Partition-Map Adapter Row

Command:
```bash
cargo bench -p moirai-benchmarks --bench iterator_adapter_comparison -- iterator_adapter_partition_map --quiet
```

Workload: owned vector streams are mapped and split through `partition_map` using the public `Either<L, R>` sum type. The benchmark source asserts equal Moirai and Rayon left/right output vectors before timing, and unit tests cover side-local output order.

| Benchmark | Moirai | Reference |
| --- | ---: | ---: |
| Iterator `partition_map` | 32.468-32.719 us | Rayon 587.36-620.15 us |

Interpretation: the audited Rayon-style adapter subset now includes mapped `Either` splitting without adding a runtime dependency on Rayon or a boxed dispatch path.

## 2026-05-29 Iterator Zip-Eq Adapter Row

Command:
```bash
cargo bench -p moirai-benchmarks --bench iterator_adapter_comparison -- iterator_adapter_zip_eq --quiet
```

Workload: equal-length owned vector streams are paired with `zip_eq`, mapped, filtered, and bounded with `take`. The benchmark source asserts equal Moirai and Rayon output vectors before timing, and unit tests cover both equal-length pairing and mismatch panic semantics.

| Benchmark | Moirai | Reference |
| --- | ---: | ---: |
| Iterator `zip_eq` | 107.34-142.67 us | Rayon 364.99-373.05 us |

Interpretation: the audited Rayon-style adapter subset now includes equal-length pair-stream semantics. Full Rayon ecosystem parity and full indexed producer/consumer compatibility remain separate documented boundaries.

## 2026-05-28 Native Rayon/Tokio Gap Closure Refresh

Commands:
```bash
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- public_result_handle_ready --quiet
cargo bench -p moirai-benchmarks --bench thread_schedule_comparison -- "ready_task_schedule|indexed_reduce_schedule|mixed_unified_schedule|real_application_mixed_workload" --quiet
cargo bench -p moirai-benchmarks --bench async_iterator_comparison -- "async_iterator_(ready_pipeline|take_skip_pipeline|enumerate_zip_pipeline|bounded_yield_pipeline)" --quiet
cargo bench -p moirai-benchmarks --bench iterator_adapter_comparison -- "iterator_(indexed_collect_into_vec|adapter_(indexed|filter_flat|flatten|take_skip_any|update|intersperse|try_reduce|terminal_reducers|find_map|position|ref_copy_clone|non_clone_ref_map|unzip))" --quiet
cargo bench -p moirai-benchmarks --bench iter_ops_parallel_comparison -- iter_ops_parallel --quiet
cargo bench -p moirai-benchmarks --bench cache_iterator_comparison -- "cache_iterator_zero_copy(_large)?_reduce|cache_iterator_zero_copy_map" --quiet
cargo bench -p moirai-benchmarks --bench execution_context_comparison -- execution_context_owned_map --quiet
cargo bench -p moirai-benchmarks --bench numa_context_comparison -- numa_context_owned_map --quiet
cargo bench -p moirai-benchmarks --bench distributed_context_comparison -- distributed_context_owned_map --quiet
cargo bench -p moirai-benchmarks --bench multi_system_context_comparison -- multi_system_context_owned_map --quiet
```

Workload: same-run native scheduler, async iterator, and Rayon-style adapter rows after registry-owned task-ID allocation and diagnostic tree splitting. Every row keeps value assertions inside the benchmark source before timing.

| Benchmark | Moirai | Reference |
| --- | ---: | ---: |
| Ready result handle | 544.01-571.75 ns | Tokio 1.3583-1.6227 us |
| Captured result handle | 487.46-506.54 ns | Tokio 1.2880-1.3826 us |
| Oversized captured result handle | 638.02-736.37 ns | Tokio 1.4152-1.5104 us |
| Async wake-once result handle | 734.80-894.09 ns | Tokio 1.4993-1.5182 us |
| Single scoped completion | 534.89-543.65 ns | Rayon 632.16-637.41 ns |
| Ready scoped schedule | 10.634-11.466 us | Tokio 81.088-82.047 us; Rayon 82.964-83.842 us |
| Indexed reduction | 879.79-915.68 ns | Rayon 7.9438-8.0862 us |
| Mixed unified workload | 39.666-40.156 us | Tokio plus Rayon 51.772-53.105 us |
| Real application mixed workload | 89.559-90.721 us | Tokio plus Rayon 106.88-108.04 us |
| Async iterator ready pipeline | 296.24-297.32 us | Tokio `JoinSet` 24.665-24.867 ms |
| Async iterator take/skip pipeline | 86.505-87.440 us | Tokio `JoinSet` 23.828-24.211 ms |
| Async iterator enumerate/zip pipeline | 272.30-274.50 us | Tokio `JoinSet` 45.560-46.273 ms |
| Bounded async iterator pipeline | 2.0134-2.0244 ms | Tokio `JoinSet` 10.282-10.360 ms |
| Iterator indexed collect_into_vec | 54.745-75.638 us | Rayon 95.255-102.59 us |
| Iterator indexed pipeline | 110.68-177.31 us | Rayon 318.24-323.68 us |
| Iterator filter/flat pipeline | 22.482-22.693 us | Rayon 2.8352-3.0191 ms |
| Iterator flatten | 75.239-76.573 us | Rayon 1.2744-1.3018 ms |
| Iterator take/skip-any | 14.639-14.739 us | Rayon 1.0011-1.0244 ms |
| Iterator update | 17.161-17.385 us | Rayon 355.51-361.33 us |
| Iterator intersperse | 35.564-36.352 us | Rayon 337.54-348.60 us |
| Iterator try-reduce | 8.6967-8.7589 us | Rayon 74.804-76.607 us |
| Iterator terminal reducers | 36.051-36.607 us | Rayon 210.00-221.47 us |
| Iterator find-map | 44.891-45.368 us | Rayon 257.55-271.53 us |
| Iterator position | 23.683-23.933 us | Rayon 221.01-228.09 us |
| Iterator copied/cloned materialization | 1.8444-1.8703 ms | Rayon 2.8979-2.9487 ms |
| Iterator non-Clone borrowed ref map | 16.446-17.165 us | Rayon 57.328-79.918 us |
| Iterator unzip | 29.351-29.744 us | Rayon 493.15-537.41 us |
| Scoped `iter_ops::ParallelIter` map | 7.0830-7.5290 us | Rayon 46.176-47.066 us |
| Scoped `iter_ops::ParallelIter` reduce | 1.7471-1.7582 us | Rayon 47.637-50.345 us |
| Borrowed cache zero-copy map | 422.36-444.66 ns | Rayon 101.42-289.01 us |
| Borrowed cache zero-copy reduce | 297.25-303.37 ns | Rayon 64.054-165.09 us |
| Borrowed cache zero-copy large reduce | 4.0282-4.1575 us | Rayon 67.143-83.126 us |
| Owned execution context map | 120.53-122.07 ns | Rayon 29.323-30.104 us |
| Owned NUMA context map | 175.50-204.96 ns | Rayon 45.097-142.69 us |
| Owned distributed context map | 389.05-428.30 ns | Rayon 72.092-75.365 us |
| Owned multi-system context map | 348.11-354.81 ns | Rayon 61.837-78.097 us |

Interpretation: no active comparison gap remains in the native scheduler/result-handle/indexed-reduction scope. Exact-size indexed sources now support `collect_into_vec` over caller-provided storage; owned vector sources bulk-move items into existing spare capacity and keep the same-run row below Rayon for the audited source boundary. The legacy `iter_ops::ParallelIter` helper now removes the old `Arc<Vec<T>>` data-sharing path and keeps scoped OS-thread fanout behind the bounded scheduler batch-capacity gate, which closes the small-trivial-work Rayon overhead gap for the audited helper rows. `ZeroCopyParallelIter` now borrows slices and closures directly for map execution instead of allocating `Arc` wrappers, moves reduce partials through owned pair compaction instead of cloned intermediate chunks, accepts non-`Clone` reducer closures, and keeps scoped OS-thread fanout behind one scheduler batch of cache chunks; the borrowed cache helper rows stay below the equivalent Rayon borrowed-slice rows for the audited boundaries. Borrowed `Vec<T>::par_iter` now maps non-`Clone` values without `T: Clone + 'static`, and the non-Clone borrowed source row stays below the equivalent Rayon borrowed row for the audited adapter boundary. Direct execution contexts now move owned chunks instead of cloning chunk slices, and the owned execution-context row stays below the equivalent Rayon owned-map row for the audited single-chunk boundary. NUMA iterator helpers now move owned batches for map and reduce instead of requiring clone-bound chunk materialization, and the owned NUMA context row stays below the equivalent Rayon owned-map row for the audited small-work boundary. Distributed iterator helpers now move owned partitions, produce value-semantic map results instead of placeholder empty outputs, and the owned distributed context row stays below the equivalent Rayon owned-map row for the audited small-work boundary. Multi-system iterator helpers now move owned partitions through the unified scheduler, distribute real partition iterators, and return value-semantic heterogeneous map results without `Clone`-bound direct item paths; the owned multi-system context row stays below the equivalent Rayon owned-map row for the audited small-work boundary. Tokio reactor-native drop-in I/O, WASM browser event-loop integration, and full Rayon ecosystem parity remain documented compatibility boundaries rather than failures in the native scheduler benchmark gate.

## 2026-05-27 Registry-Local Task ID and Token Lifecycle Split

Commands:
```bash
cargo bench -p moirai-benchmarks --bench performance_benchmarks -- task_scheduling_overhead --quiet
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- "public_result_handle_ready/(moirai_spawn_join_ready|tokio_spawn_join_ready|moirai_scope_single_ready|rayon_scope_single_ready)" --quiet
cargo bench -p moirai-benchmarks --features registry-diagnostics --bench result_handle_diagnostics -- "result_handle_diagnostics/(direct_registry_lifecycle|direct_registry_token_lifecycle|direct_registry_external_token_lifecycle|direct_external_id_registry_register|mutex_registry_register|direct_task_id_allocate)" --quiet
cargo bench -p moirai-benchmarks --features registry-diagnostics --bench result_handle_diagnostics -- "result_handle_diagnostics/(direct_scheduled_public_(registry_)?token_wrapper_(components|without_metrics)|direct_registry_token_lifecycle)" --quiet
```

Workload: `HybridExecutor` allocates public task IDs through the existing registry registration critical section, removing the executor-local `AtomicU64`. The comparison rows verify the scheduler gate, Tokio/Rayon public references, and registry-token attribution.

| Benchmark | Result |
| --- | ---: |
| `task_scheduling_overhead` | 494.78-502.83 ns |
| `public_result_handle_ready/moirai_spawn_join_ready` | 488.87-498.00 ns |
| `public_result_handle_ready/tokio_spawn_join_ready` | 1.5249-2.1615 us |
| `public_result_handle_ready/moirai_scope_single_ready` | 514.72-525.41 ns |
| `public_result_handle_ready/rayon_scope_single_ready` | 630.54-641.75 ns |
| `direct_task_id_allocate` | 6.0550-6.1497 ns |
| `direct_registry_lifecycle` | 85.619-86.454 ns |
| `direct_registry_token_lifecycle` | 85.856-91.273 ns |
| `direct_registry_external_token_lifecycle` | 91.660-94.574 ns |
| `direct_external_id_registry_register` | 38.484-38.812 ns |
| `mutex_registry_register` | 44.033-44.423 ns |
| `direct_scheduled_public_token_wrapper_components` | 515.14-651.83 ns |
| `direct_scheduled_public_registry_token_wrapper_components` | 503.58-516.52 ns |
| `direct_scheduled_public_registry_token_wrapper_after_send_quiescent` | 545.92-566.64 ns |
| `direct_scheduled_public_token_wrapper_without_metrics` | 440.49-447.41 ns |
| `direct_scheduled_public_registry_token_wrapper_without_metrics` | 422.85-435.93 ns |

Interpretation: registry-local ID allocation passed the scheduler gate and same-run public Tokio/Rayon references. The production token lifecycle is within the public lookup lifecycle range, and scheduled registry-token wrapper rows are faster than the externally supplied ID rows in the same post-split run.

## 2026-05-27 Generic Utility SIMD Addition Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench simd_benchmarks -- vector_addition --quiet
```

Workload: generic `moirai_utils::simd::add<T>` over `f32` slices compared against a scalar loop. The public API is generic and dispatches native vector backends through sealed scalar traits.

| Elements | Generic | Scalar | Native-checked |
| ---: | ---: | ---: | ---: |
| 64 | 12.326-12.437 ns | 48.944-53.782 ns | 15.026-16.293 ns |
| 256 | 15.864-18.333 ns | 225.63-230.11 ns | 20.162-21.402 ns |
| 1,024 | 45.265-51.112 ns | 768.24-771.60 ns | 31.294-31.425 ns |
| 4,096 | 222.05-223.32 ns | 3.1164-3.1295 µs | 223.05-225.52 ns |
| 16,384 | 1.0422-1.0571 µs | 15.311-16.535 µs | 1.2639-1.3559 µs |

Interpretation: the generic API overlaps the native-checked row and remains below the scalar reference. This row verifies the utility SIMD monomorphization path; it is not a Rayon/Tokio scheduler comparison.

## 2026-05-27 Async Iterator Enumerate/Zip Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench async_iterator_comparison -- async_iterator_enumerate_zip_pipeline --quiet
```

Workload: the Moirai row maps two async iterator inputs, zips them, enumerates the paired stream, and computes an ordered checksum. The Tokio row fans both inputs through `JoinSet`, sorts by source index, zips the ordered streams, enumerates, and computes the same checksum. Both paths assert equal checksums before timing.

| Benchmark | Result |
| --- | ---: |
| `async_iterator_enumerate_zip_pipeline/moirai/32768` | 672.68-734.62 µs |
| `async_iterator_enumerate_zip_pipeline/tokio_joinset/32768` | 48.260-49.144 ms |

Interpretation: this closes the async logical-position and pair-stream adapter slice for `AsyncIterator::enumerate` and `AsyncIterator::zip`. It does not claim Tokio stream ecosystem parity.

## 2026-05-25 Async TCP Facade Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench async_tcp_comparison -- --quiet
```

Workload: both rows accept one loopback TCP client, read the same 24-byte request, write the same 24-byte echo, and assert request and echo byte equality before timing.

| Benchmark | Result |
| --- | ---: |
| `async_tcp_loopback_echo/moirai/24` | 294.02-354.85 µs |
| `async_tcp_loopback_echo/tokio/24` | 323.75-365.72 µs |

Interpretation: this is a covered Moirai-owned TCP facade comparison against `tokio::net::TcpListener` and `tokio::net::TcpStream`. The Moirai row uses `Moirai::block_on`, not an external futures executor. It is not a claim of Tokio reactor-native network drop-in compatibility.

## 2026-05-26 Async TCP Persistent Stream Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench async_tcp_comparison -- async_tcp_stream_echo --quiet
```

Workload: both rows reuse an established loopback TCP stream, write the same 24-byte request, read the same 24-byte echo from a standard-library echo peer, set TCP_NODELAY on both client and echo sockets, and assert echo byte equality before timing. The Moirai row uses production `AsyncWriteExt::write_all` and `AsyncReadExt::read_exact` futures.

| Benchmark | Result |
| --- | ---: |
| `async_tcp_stream_echo/moirai/24` | 23.946-26.092 µs |
| `async_tcp_stream_echo/tokio/24` | 42.768-45.817 µs |

Interpretation: this row isolates established-stream read/write behavior from accept and thread-spawn setup. It remains a Moirai-owned TCP facade comparison, not a Tokio reactor-native drop-in compatibility claim.

## 2026-05-26 Async TCP Write Shutdown Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench async_tcp_comparison -- async_tcp_write_shutdown --quiet
```

Workload: both rows accept one loopback TCP client, write the same 19-byte payload, invoke write-side shutdown, and assert the standard-library peer receives the payload followed by EOF.

| Benchmark | Result |
| --- | ---: |
| `async_tcp_write_shutdown/moirai/19` | 26.185-34.695 ms |
| `async_tcp_write_shutdown/tokio/19` | 21.158-27.122 ms |

Interpretation: this row closes the prior no-op shutdown gap for the Moirai-owned TCP facade. It includes per-iteration accept, client thread, read-to-EOF, and FIN observation cost, so it is a correctness and compatibility row rather than a pure shutdown syscall latency row.

## 2026-05-26 Async TCP Write Backpressure Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench async_tcp_backpressure_comparison -- async_tcp_write_backpressure --quiet
```

Workload: both rows create a loopback TCP pair with bounded socket buffers, write 16 KiB chunks until the nonblocking stream reports write backpressure, assert positive progress before `Pending`, release the server, and assert the peer drained bytes.

| Benchmark | Result |
| --- | ---: |
| `async_tcp_write_backpressure/moirai/16384` | 20.171-61.392 ms |
| `async_tcp_write_backpressure/tokio/16384` | 16.257-43.003 ms |

Interpretation: this is a bounded readiness/backpressure correctness row. It verifies that the Moirai-owned TCP facade returns `Poll::Pending` under OS send-buffer pressure instead of spinning or masking backpressure. It is not a pure throughput row and does not claim full Tokio reactor-native I/O compatibility.

## 2026-05-26 Async TCP Read Readiness Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench async_tcp_readiness_comparison -- --quiet
```

Workload: both rows create a loopback TCP pair, directly poll the nonblocking stream before the peer writes and assert `Poll::Pending`, release the peer, read the same 5-byte payload, and assert exact byte equality.

| Benchmark | Result |
| --- | ---: |
| `async_tcp_read_readiness/moirai/5` | 564.43-903.33 µs |
| `async_tcp_read_readiness/tokio/5` | 474.64-739.83 µs |

Interpretation: this is a read-readiness correctness row. It verifies that the Moirai-owned TCP facade reports pending read readiness before peer data and delivers the exact payload after readiness. It does not claim full Tokio reactor-native I/O compatibility.

## 2026-05-26 Async TCP Pending Read Cancellation Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench async_tcp_cancel_safety_comparison -- --quiet
```

Workload: both rows create a loopback TCP pair, create a borrowed exact-read future over a caller-owned buffer, poll it to `Pending` before peer data, drop the pending future, assert the cancelled buffer remains unchanged, release the peer, read the same 5-byte payload, and assert exact byte equality.

| Benchmark | Result |
| --- | ---: |
| `async_tcp_pending_read_cancel_safety/moirai/5` | 299.08-340.01 µs |
| `async_tcp_pending_read_cancel_safety/tokio/5` | 339.36-368.55 µs |

Interpretation: this is a facade cancellation-safety row. It verifies that dropping a pending Moirai TCP read future does not consume bytes or corrupt the stream before later payload delivery. It does not claim OS-level cancellation of an in-flight kernel operation.

## 2026-05-26 Async I/O Tokio Compatibility Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench async_io_compat_comparison -- --quiet
```

Workload: each read row fills the same 4 KiB output buffer from the same chunked in-memory Moirai reader and asserts byte equality. Each write row writes the same 4 KiB payload into the same fixed-capacity Moirai writer, invokes shutdown, and asserts payload bytes plus one shutdown delegation. The Tokio rows use `TokioCompat<T>` and Tokio extension traits over the same native Moirai reader/writer.

| Benchmark | Result |
| --- | ---: |
| `async_io_compat_read_exact/moirai_native` | 2.5060-2.6553 µs |
| `async_io_compat_read_exact/tokio_compat` | 2.4962-2.6191 µs |
| `async_io_compat_write_shutdown/moirai_native` | 179.85-191.55 ns |
| `async_io_compat_write_shutdown/tokio_compat` | 186.41-195.91 ns |

Interpretation: this is a trait-compatibility overhead row, not a reactor I/O throughput row. Transparent compatibility wrappers preserve native byte semantics and keep Tokio as a feature-gated optional dependency outside the default runtime build.

## 2026-05-25 Iterator Try-For-Each Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench iterator_adapter_comparison -- iterator_adapter_try_for_each --quiet
```

Workload: both rows apply a fallible side-effect terminal over 32,768 values and assert equal atomic checksums before timing.

| Benchmark | Result |
| --- | ---: |
| `iterator_adapter_try_for_each/moirai/32768` | 142.55-149.28 µs |
| `iterator_adapter_try_for_each/rayon/32768` | 932.60 µs-1.1186 ms |

Interpretation: the `try_for_each` Rayon-style terminal has value-checked benchmark evidence and no longer has a pending comparison row.

## 2026-05-25 Async File Facade Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench async_fs_comparison -- --quiet
```

Workload: both rows read the same generated 64 KiB file. The benchmark asserts the Moirai and Tokio outputs equal the generated source bytes before timing.

| Benchmark | Result |
| --- | ---: |
| `async_fs_read_to_end/moirai/65536` | 39.127-45.710 µs |
| `async_fs_read_to_end/tokio/65536` | 96.964-100.34 µs |

Interpretation: this is a covered Moirai-owned file facade comparison against `tokio::fs::read`. The Moirai row uses `Moirai::block_on`, not an external futures executor. It is not a claim of Tokio reactor-native file or network drop-in compatibility.

## 2026-05-26 Async File Copy Facade Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench async_fs_comparison -- async_fs_copy_file --quiet
```

Workload: both rows copy the same generated 64 KiB file to a prepared destination path. The benchmark asserts the copied byte count equals 64 KiB, and setup asserts the copied destination bytes equal the generated source bytes for both Moirai and Tokio before timing.

| Benchmark | Result |
| --- | ---: |
| `async_fs_copy_file/moirai/65536` | 536.26-604.18 µs |
| `async_fs_copy_file/tokio/65536` | 541.41-716.30 µs |

Interpretation: this is a covered Moirai-owned file copy facade comparison against `tokio::fs::copy`. The Moirai path delegates to the PAL platform copy operation instead of allocating a user-space transfer buffer. It is not a claim of reactor-native file readiness.

## 2026-05-26 Async File Write Facade Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench async_fs_comparison -- async_fs_write_file --quiet
```

Workload: both rows write the same generated 64 KiB byte payload to prepared destination paths. Setup asserts the written destination bytes equal the generated source bytes for both Moirai and Tokio before timing.

| Benchmark | Result |
| --- | ---: |
| `async_fs_write_file/moirai/65536` | 2.8650-3.4698 ms |
| `async_fs_write_file/tokio/65536` | 2.5939-3.2074 ms |

Interpretation: this is a covered Moirai-owned file write facade comparison against `tokio::fs::write`. The Moirai path delegates to the PAL platform write operation over the caller-provided byte slice and avoids constructing the async file handle, stats state, write loop, and unconditional sync path used by the previous convenience implementation. Tokio is faster in this same-run write row; the row closes coverage, not a performance lead.

## 2026-05-26 Async File Append Facade Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench async_fs_comparison -- async_fs_append_file --quiet
```

Workload: both rows reset a destination file to the same prefix through Criterion batched setup, append the same generated 64 KiB byte payload, and setup asserts prefix plus appended byte equality for both Moirai and Tokio before timing.

| Benchmark | Result |
| --- | ---: |
| `async_fs_append_file/moirai/65536` | 272.59-291.93 µs |
| `async_fs_append_file/tokio/65536` | 190.29-320.18 µs |

Interpretation: this is a covered Moirai-owned file append facade comparison against Tokio append-open/write behavior. The Moirai path delegates to the PAL platform append operation over the caller-provided byte slice and avoids constructing the async file handle, stats state, write loop, and unconditional sync path used by the previous convenience implementation. The same-run confidence intervals overlap.

## 2026-05-26 Async File Metadata Facade Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench async_fs_comparison -- async_fs_metadata_file --quiet
```

Workload: both rows read metadata from the same generated 64 KiB file. Setup asserts `is_file()` and exact 64 KiB length for both Moirai and Tokio before timing.

| Benchmark | Result |
| --- | ---: |
| `async_fs_metadata_file/moirai/65536` | 25.187-28.833 µs |
| `async_fs_metadata_file/tokio/65536` | 85.097-87.725 µs |

Interpretation: this is a covered Moirai-owned file metadata facade comparison against `tokio::fs::metadata`. The Moirai path delegates to the PAL platform metadata operation and avoids constructing the async file handle or stats state. It is not a claim of reactor-native file readiness.

## 2026-05-26 Async File Rename Facade Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench async_fs_comparison -- async_fs_rename_file --quiet
```

Workload: per-iteration setup prepares a 64 KiB source file and an empty destination path for each row. The measured operation renames the source to the destination and asserts source removal plus exact destination length.

| Benchmark | Result |
| --- | ---: |
| `async_fs_rename_file/moirai/65536` | 603.37 µs-2.0949 ms |
| `async_fs_rename_file/tokio/65536` | 3.5253-7.3040 ms |

Interpretation: this is a covered Moirai-owned file rename facade comparison against `tokio::fs::rename`. The Moirai path delegates to the PAL platform rename operation and avoids reading or copying file contents through user-space buffers. It is not a claim of reactor-native file readiness or OS-level cancellation.

## 2026-05-27 Async File Remove Facade Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench async_fs_comparison -- async_fs_remove_file --quiet
```

Workload: per-iteration setup prepares a 64 KiB source file for each row. The measured operation removes that path and asserts the file no longer exists.

| Benchmark | Result |
| --- | ---: |
| `async_fs_remove_file/moirai/65536` | 168.50-193.31 µs |
| `async_fs_remove_file/tokio/65536` | 189.80-211.05 µs |

Interpretation: this is a covered Moirai-owned file remove facade comparison against `tokio::fs::remove_file`. The Moirai path delegates to the PAL platform remove operation and avoids constructing the async file handle or stats state. It is not a claim of reactor-native file readiness or OS-level cancellation.

## 2026-05-27 Async Directory Facade Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench async_fs_dir_comparison -- --quiet
```

Workload: the single-directory row creates one directory, asserts it is a directory, removes it, and asserts absence. The recursive row creates a three-level directory tree, writes and reads a marker file to verify the leaf path, removes the root recursively, and asserts root absence.

| Benchmark | Result |
| --- | ---: |
| `async_fs_create_remove_dir/moirai/1` | 228.49-251.78 µs |
| `async_fs_create_remove_dir/tokio/1` | 275.03-287.74 µs |
| `async_fs_create_remove_dir_all/moirai/1` | 2.8710-3.1976 ms |
| `async_fs_create_remove_dir_all/tokio/1` | 3.8355-4.2147 ms |

Interpretation: this is a covered Moirai-owned directory facade comparison against Tokio directory creation and removal APIs. The Moirai path delegates to PAL platform directory operations and removes direct async-layer platform ownership. It is not a claim of reactor-native file readiness or OS-level cancellation.

## 2026-05-25 Async UDP Facade Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench async_udp_comparison -- --quiet
```

Workload: both rows receive the same 27-byte UDP datagram from a standard-library loopback sender. The benchmark asserts received byte count and payload equality before timing.

| Benchmark | Result |
| --- | ---: |
| `async_udp_loopback_recv_from/moirai/27` | 6.1554-6.4334 µs |
| `async_udp_loopback_recv_from/tokio/27` | 6.2846-6.4721 µs |

Interpretation: this is a covered Moirai-owned UDP facade comparison against `tokio::net::UdpSocket::recv_from`. The Moirai row uses `Moirai::block_on`, not an external futures executor. It is not a claim of Tokio reactor-native network drop-in compatibility.

## 2026-05-25 Bounded Channel Matrix Contract Refresh

Command:
```bash
cargo bench -p moirai-benchmarks --bench channel_matrix -- p1_c1 --quiet
```

Workload: 8,192 integer items, one producer, capacity 1. Both the Tokio MPSC and Moirai MPMC rows assert the same closed-form transferred-item sum before timing.

| Benchmark | Result |
| --- | ---: |
| `bounded_channel_matrix/tokio_mpsc/p1_c1` | 2.5089-2.6101 ms |
| `bounded_channel_matrix/moirai_mpmc/p1_c1` | 1.4157-1.4504 ms |

Interpretation: the bounded channel comparison remains a covered Tokio row in the gap audit and comparison report. It now has explicit benchmark-contract coverage requiring bounded Criterion timing windows across all current comparison targets.

## 2026-05-25 Mixed Unified Scheduler Refresh

Command:
```bash
cargo bench -p moirai-benchmarks --bench thread_schedule_comparison -- mixed_unified_schedule --quiet
```

Workload: the Moirai row combines sync scoped completion, async result handles, and indexed reduction through one runtime and one scheduler. The reference row uses Tokio for async handles plus Rayon for scoped and indexed work. Both rows assert the same closed-form mixed sum before timing.

| Benchmark | Result |
| --- | ---: |
| `mixed_unified_schedule/moirai_unified_mixed` | 40.510-41.370 µs |
| `mixed_unified_schedule/tokio_rayon_mixed` | 50.147-56.014 µs |

Interpretation: the single Moirai runtime remains ahead of the Tokio plus Rayon two-engine reference for the covered mixed workload.

## 2026-05-26 Real Application Mixed Workload Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench thread_schedule_comparison -- real_application_mixed_workload --quiet
```

Workload: the Moirai row combines async task fan-out, scoped request processing, indexed analytics over 1,048,576 records, and bounded SPSC control-message transfer. The reference row uses Tokio async fan-out, Rayon scoped work, Rayon indexed analytics, and Tokio bounded MPSC transfer. Both rows assert the same closed-form checksum before timing.

| Benchmark | Result |
| --- | ---: |
| `real_application_mixed_workload/moirai_real_app_pipeline` | 90.956-92.283 µs |
| `real_application_mixed_workload/tokio_rayon_real_app_pipeline` | 108.39-115.36 µs |

Interpretation: the application-shaped mixed workload keeps the Moirai unified path ahead while preserving same-run Tokio/Rayon reference semantics and value equality.

## 2026-05-26 Public Result Handle And Wake Attribution Refresh

Command:
```bash
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- public_result_handle_ready --quiet
```

Workload: each row returns or publishes the exact value `42` before timing. Wake-once rows use a future that returns `Pending`, calls `wake_by_ref`, and then returns `42`.

| Benchmark | Result |
| --- | ---: |
| `public_result_handle_ready/moirai_spawn_join_ready` | 358.73-375.28 ns |
| `public_result_handle_ready/tokio_spawn_join_ready` | 1.0286-1.2098 µs |
| `public_result_handle_ready/moirai_spawn_join_captured_ready` | 367.96-390.52 ns |
| `public_result_handle_ready/tokio_spawn_join_captured_ready` | 1.1594-1.2275 µs |
| `public_result_handle_ready/moirai_spawn_join_oversized_captured_ready` | 466.72-488.69 ns |
| `public_result_handle_ready/tokio_spawn_join_oversized_captured_ready` | 1.1370-1.4497 µs |
| `public_result_handle_ready/moirai_spawn_async_ready` | 472.79-510.97 ns |
| `public_result_handle_ready/moirai_spawn_async_wake_once` | 513.57-544.64 ns |
| `public_result_handle_ready/tokio_spawn_async_wake_once` | 1.2109-1.2496 µs |
| `public_result_handle_ready/moirai_scope_single_ready` | 294.01-313.60 ns |
| `public_result_handle_ready/rayon_scope_single_ready` | 575.96-624.94 ns |

Attribution commands:
```bash
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- "result_handle_diagnostics/(moirai_spawn_async_ready|moirai_spawn_async_wake_once|direct_async_|direct_scheduler_submission_queue_publication|direct_spawn_metrics_(before|after)_scheduler_submission)" --quiet
cargo bench -p moirai-benchmarks --features scheduler-diagnostics --bench result_handle_diagnostics -- "result_handle_diagnostics/(direct_scheduler_submission_queue_publication|direct_scheduler_empty_wake_decision|direct_scheduler_contended_wake_decision|direct_scheduler_saturated_wake_decision|direct_scheduler_join_fast_spin_)" --quiet
```

Attribution: async state claims measured around 5.41-7.23 ns, lifecycle completion at 55.894-56.889 ns, sender-cell send/join at 46.804-47.024 ns, ready completion components at 165.99-166.88 ns, spawn metrics before/after scheduler submission at 306.57-332.83 ns and 278.42-327.44 ns, scheduler queue publication at 65.273-65.407 ns, quiescent join spin at 860.54-869.32 ps, pending join spin at 4.2080-4.2403 µs, empty wake decision at 29.494-30.093 ns, contended wake decision at 112.33-119.62 ns, and saturated wake decision at 427.11-428.45 ps.

Interpretation: equivalent public result-handle, async wake-once, and scoped completion rows remain ahead of Tokio/Rayon. Remaining variance is bounded to scheduler/lifecycle composition and diagnostic metrics publication rather than the value paths or async state primitives.

## 2026-05-25 Standalone Deque Reclaim Policy Diagnostic

Command:
```bash
cargo bench -p moirai-benchmarks --bench thread_schedule_comparison -- standalone_deque_reclaim_policy --quiet
```

Workload: both rows push 256 values into a standalone `ChaseLevDeque` with initial capacity 2, forcing resize, reclaiming retired backing arrays through the selected policy, draining the queue, and asserting the closed-form sum before timing.

| Benchmark | Result |
| --- | ---: |
| `standalone_deque_reclaim_policy/moirai_quiescent_reclaim` | 2.5038-2.5309 µs |
| `standalone_deque_reclaim_policy/moirai_shared_epoch_reclaim` | 6.8529-6.8897 µs |

Interpretation: default production queues should retain `QuiescentReclaim` because its state and guard are zero-sized. `SharedEpochReclaim` remains opt-in for shared cleanup points that need concurrent retired-array reclamation and can pay two atomic operations per queue access.

## 2026-05-22 Quick Scheduler Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench thread_schedule_comparison
```

Workload: 4 worker threads. Ready-task rows use 256 completion-only work units. Indexed reduction rows use the same closed-form sum and compare Moirai's typed indexed reduction against Rayon's `into_par_iter().map(...).sum()` pattern. Every measured path asserts the computed sum equals `n * (n + 1) / 2` before black-boxing the result.

| Benchmark | Result |
| --- | ---: |
| `ready_task_schedule/moirai_scope` | 26.816-27.033 µs, estimate 26.900 µs |
| `ready_task_schedule/tokio_spawn_ready` | 85.535-87.446 µs, estimate 86.633 µs |
| `ready_task_schedule/rayon_scope` | 63.130-77.987 µs, estimate 69.125 µs |
| `indexed_reduce_schedule/moirai_indexed_reduce` | 1.5913-1.6066 µs, estimate 1.5983 µs |
| `indexed_reduce_schedule/rayon_indexed` | 6.8983-7.3793 µs, estimate 7.1611 µs |

### Scoped Ready Scaling

| Work units | Moirai scope | Rayon scope | Tokio spawn ready |
| ---: | ---: | ---: | ---: |
| 64 | 10.913-11.370 µs | 26.830-27.158 µs | 21.275-21.724 µs |
| 256 | 26.380-26.692 µs | 78.297-79.040 µs | 80.327-80.826 µs |
| 1024 | 63.513-64.073 µs | 278.97-286.54 µs | 337.23-340.91 µs |

### Indexed Reduce Scaling

| Work units | Moirai indexed reduce | Rayon indexed |
| ---: | ---: | ---: |
| 64 | 7.9260-7.9882 ns | 3.9325-4.3511 µs |
| 256 | 1.4758-1.4970 µs | 7.1488-7.7043 µs |
| 1024 | 3.1718-3.1917 µs | 9.8248-9.9044 µs |

Interpretation: `moirai_scope` exceeds the Tokio and Rayon ready-work baselines by coalescing scoped logical jobs into worker-sized unified-scheduler batches. `moirai_indexed_reduce` uses typed worker chunks, caller-side participation, and caller-side final reduction. The 256-item indexed row now schedules one worker chunk and computes one chunk on the caller, which removes avoidable scheduler wakeup while preserving the same value assertion.

## 2026-05-22 Industry Ready-Task Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench industry_comparison
```

Workload: 4 worker threads for Moirai, Tokio, and Rayon. Ready-task rows assert the computed sum equals `n * (n + 1) / 2`.

| Ready work units | Moirai scope | Tokio spawn | Rayon scope |
| ---: | ---: | ---: | ---: |
| 100 | 13.641-14.512 µs | 31.051-31.666 µs | 35.892-36.675 µs |
| 1,000 | 62.649-63.627 µs | 326.96-331.45 µs | 272.30-275.11 µs |
| 10,000 | 487.03-519.19 µs | 3.9128-4.0256 ms | 2.0663-2.1245 ms |

### Official Rayon Map/Reduce Pattern

Workload: 4 worker threads, 8 black-boxed arithmetic steps per item. Rayon row uses the documented `into_par_iter().map(...).sum()` pattern; Moirai row uses `Moirai::map_reduce_indexed`. Both rows assert the same closed-form sum.

| Work items | Moirai indexed reduce | Rayon `into_par_iter` |
| ---: | ---: | ---: |
| 4,096 | 3.9433-4.0053 µs | 12.184-12.673 µs |
| 32,768 | 12.244-12.461 µs | 22.034-23.853 µs |
| 65,536 | 20.315-20.855 µs | 29.046-29.625 µs |

## 2026-05-22 Public Result-Handle Diagnostic

Command:
```bash
cargo bench -p moirai-benchmarks --bench performance_benchmarks task_scheduling_overhead
```

Workload: one public `Moirai::spawn_fn` task followed by `TaskHandle::join`. The benchmark asserts the returned value is `42` before black-boxing it.

| Benchmark | Result |
| --- | ---: |
| `task_scheduling_overhead` | 544.55-552.14 ns, estimate 548.05 ns |

### Public Result-Handle Comparison

Command:
```bash
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison
```

Workload: one task that returns `42`, or a captured array sum with the expected value asserted. Ready rows compare Moirai `Moirai::spawn_fn` plus `TaskHandle::join` against Tokio `tokio::spawn` plus `JoinHandle::await`. Moirai async-ready uses the same Tokio ready `JoinHandle` baseline because Tokio's equivalent ready task is already async-native. Wake-once rows use a real future that returns `Pending`, calls `wake_by_ref`, then returns `42` on the requeued poll. Scoped rows compare Moirai `Moirai::scope` against Rayon `scope` because Rayon does not expose an equivalent per-task result handle for scoped work. This benchmark uses 20 Criterion samples, 500 ms warm-up, and 2 second measurement windows.

| Benchmark | Result |
| --- | ---: |
| `public_result_handle_ready/moirai_spawn_join_ready` | 515.51-525.52 ns, estimate 519.71 ns |
| `public_result_handle_ready/tokio_spawn_join_ready` | 1.9694-2.2197 µs, estimate 2.0872 µs |
| `public_result_handle_ready/moirai_spawn_join_captured_ready` | 552.23-562.69 ns, estimate 557.70 ns |
| `public_result_handle_ready/tokio_spawn_join_captured_ready` | 1.8724-2.0308 µs, estimate 1.9544 µs |
| `public_result_handle_ready/moirai_spawn_join_oversized_captured_ready` | 740.32-756.19 ns, estimate 748.67 ns |
| `public_result_handle_ready/tokio_spawn_join_oversized_captured_ready` | 2.0403-2.1709 µs, estimate 2.1092 µs |
| `public_result_handle_ready/moirai_spawn_async_ready` | 761.89-779.07 ns, estimate 770.27 ns |
| `public_result_handle_ready/moirai_spawn_async_wake_once` | 782.06-792.38 ns, estimate 786.57 ns |
| `public_result_handle_ready/tokio_spawn_async_wake_once` | 2.9087-3.1672 µs, estimate 3.0526 µs |
| `public_result_handle_ready/moirai_scope_single_ready` | 506.22-515.42 ns, estimate 510.13 ns |
| `public_result_handle_ready/rayon_scope_single_ready` | 679.76-697.40 ns, estimate 688.99 ns |

Correctness evidence: `cargo test -p moirai-core --all-features task_handle` covers ready, delayed, cross-thread, and cancellation result states. `cargo test -p moirai-executor --all-features` covers inline and heap scheduled-job storage plus self-wake and external-wake async progress. `cargo test -p moirai --lib test_repeated_public_spawn_join_completes -- --nocapture` and `cargo test -p moirai --lib --release test_repeated_public_spawn_join_completes -- --nocapture` completed 1,048,576 public `spawn_fn`/`join` iterations with value assertions. `cargo test -p moirai-benchmarks --test benchmark_contracts` verifies bounded Criterion settings, disabled plot generation for this diagnostic, comparison-path value correctness, zero-sized work-class routing, and monomorphized async-executor future erasure.

Interpretation: public result-bearing `spawn_fn` and `spawn_async` remain a separate diagnostic path because they preserve one task handle and result slot per task. Competitive batch comparison targets keep these rows separate because Rayon `scope` is completion-only and not result-handle equivalent. Lifecycle state now uses registry-owned blocks rather than per-task lifecycle `Arc` allocation, and running lifecycle tokens return execution duration so result-handle metrics reuse lifecycle timing instead of sampling duplicate task-local clocks. The result-slot wait path uses an explicit `WAITING` state with an inline waiter cell, narrowed claim-only atomic orderings, an initial direct READY-to-TAKEN claim for already-complete handles, and a monomorphized zero-sized wait policy for load-gated pending spins. Async public tasks store futures inline inside the heap-stable async state, use inline lifecycle state, consume one coalesced in-poll wake before rescheduling, build the task waker directly from the future-state `Arc`, and use an inlined by-reference scheduler path so in-poll `wake_by_ref` can mark `ASYNC_NOTIFIED` without cloning the task `Arc`. The standalone async executor queues concrete futures behind monomorphized poll/drop function pointers instead of `dyn Future`. Scheduler workers now use per-worker `Thread::unpark` notifications, quiescent work-class routing, and local-queue idle spin so serial spawn/join does not rotate through sleeping workers. Serial handoff routing uses `WorkClass::SERIAL_AFFINITY_OFFSET`, so the route is still selected through a monomorphized associated constant and stores no runtime policy object. Scoped single-job completion avoids the chunk vector, boxed wrapper closure, and per-scope `Arc` state. Small scheduled jobs use 14 machine words of inline erased storage while keeping `InlineJob` at a two-cache-line footprint; oversized jobs allocate one typed `Box<F>` behind the same inline job trampoline instead of using `Box<dyn FnOnce>` or a separate raw-pointer heap job variant. The latest filtered async public result-handle comparison measured Moirai async-ready at 761.89-779.07 ns and wake-once at 782.06-792.38 ns, versus Tokio wake-once at 2.9087-3.1672 µs.

Regression guard: a raw-pointer two-endpoint result-slot variant was rejected after `task_scheduling_overhead` regressed to 633.01-640.02 ns, estimate 636.61 ns. Relaxed lifecycle metadata atomics were rejected after isolated lifecycle rows improved but `task_scheduling_overhead` regressed to 608.31-641.98 ns. Removing the duplicate scheduler worker identity field was rejected after the public scheduling gate failed to retain an improvement and reran at 584.46-590.88 ns. Metrics-before-result publication regressed `result_handle_diagnostics/moirai_spawn_join_ready` to 581.34-586.56 ns. Registry-owned task ID allocation regressed the same row to 628.34-641.23 ns, and a fresh-slot registry insertion variant regressed it to 683.31-768.95 ns. All three variants were reverted. An unconditional load-before-CAS result take path regressed direct already-ready result slots and is not retained; the retained path keeps the first claim as a direct CAS and uses relaxed-load gating only while pending. A per-worker running-bit wake suppression variant improved some oversized diagnostics but added atomic traffic to every scheduled job and regressed public result-handle rows, so it is rejected. A direct CAS-only `wake_by_ref` fast path improved wake-once but regressed async-ready; the retained async wake path uses the inlined by-reference scheduler state machine instead. After retaining the verified `Arc` ownership model, monomorphized pending-spin policy, `WorkClass::SERIAL_AFFINITY_OFFSET` serial handoff route, 14-word inline job storage, and narrowed scheduler execution counter orderings, `cargo bench -p moirai-benchmarks --bench performance_benchmarks task_scheduling_overhead --verbose` measured 553.42-568.04 ns, estimate 559.15 ns, with Criterion reporting a statistically significant improvement.

### 2026-05-23 Scheduler Counter Ordering Diagnostic

Command:
```bash
cargo bench -p moirai-benchmarks --bench performance_benchmarks task_scheduling_overhead --verbose
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- "public_result_handle_ready/(moirai_spawn_join_ready|tokio_spawn_join_ready|moirai_scope_single_ready|rayon_scope_single_ready)"
```

| Benchmark | Result |
| --- | ---: |
| `task_scheduling_overhead` | 553.42-568.04 ns, estimate 559.15 ns |
| `public_result_handle_ready/moirai_spawn_join_ready` | 597.04-610.30 ns, estimate 602.83 ns |
| `public_result_handle_ready/tokio_spawn_join_ready` | 1.5296-1.7032 us, estimate 1.6036 us |
| `public_result_handle_ready/moirai_scope_single_ready` | 495.80-521.76 ns, estimate 509.00 ns |
| `public_result_handle_ready/rayon_scope_single_ready` | 665.46-690.43 ns, estimate 675.62 ns |

Interpretation: scheduler execution uses `Release` for active/pending count publication where the returned value is unused, keeps `AcqRel` on the final active-worker decrement that gates quiescence notification, and uses `Relaxed` for completed/failed diagnostic counters. The public ready and scope rows remain ahead of their Tokio and Rayon same-run references. The worker-identity field removal candidate is rejected because it did not retain a stable public scheduling improvement.

### 2026-05-23 Production Lifecycle Clock Rejection Refresh

Commands:
```bash
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- "result_handle_diagnostics/direct_registry_lifecycle"
cargo bench -p moirai-benchmarks --bench performance_benchmarks task_scheduling_overhead --verbose
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- "public_result_handle_ready/(moirai_spawn_join_ready|tokio_spawn_join_ready|moirai_scope_single_ready|rayon_scope_single_ready)"
```

| Benchmark | Result |
| --- | ---: |
| `result_handle_diagnostics/direct_registry_lifecycle` | 87.811-90.472 ns, estimate 88.844 ns |
| `task_scheduling_overhead` | 544.55-552.14 ns, estimate 548.05 ns |
| `public_result_handle_ready/moirai_spawn_join_ready` | 574.67-584.36 ns, estimate 579.59 ns |
| `public_result_handle_ready/tokio_spawn_join_ready` | 1.6123-2.0449 us, estimate 1.8298 us |
| `public_result_handle_ready/moirai_scope_single_ready` | 501.70-549.27 ns, estimate 520.45 ns |
| `public_result_handle_ready/rayon_scope_single_ready` | 764.87-900.62 ns, estimate 839.74 ns |

Interpretation: production registry lifecycle timing remains on the `Instant` policy. The Windows QPC path stays in `result_handle_diagnostics` only: it is precise and lock-free, but production promotion and cached-frequency variants did not satisfy the scheduling gate. The retained source contract rejects `QueryPerformanceCounter`, `QueryPerformanceFrequency`, `qpc_created_ticks`, and `AtomicI64` in `moirai-executor/src/registry/mod.rs`.

### 2026-05-23 Public Wrapper Attribution Refresh

Commands:
```bash
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- "result_handle_diagnostics/(direct_task_id_allocate|direct_metrics_record_task_spawned|direct_metrics_record_task_completed|direct_public_wrapper_without_metrics|direct_public_wrapper_components|direct_registry_lifecycle|mutex_registry_register)"
cargo bench -p moirai-benchmarks --bench performance_benchmarks -- task_scheduling_overhead
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- "public_result_handle_ready/(moirai_spawn_join_ready|tokio_spawn_join_ready|moirai_scope_single_ready|rayon_scope_single_ready)"
```

| Benchmark | Result |
| --- | ---: |
| `result_handle_diagnostics/direct_task_id_allocate` | 6.1355-6.2125 ns, estimate 6.1796 ns |
| `result_handle_diagnostics/direct_metrics_record_task_spawned` | 28.634-29.053 ns, estimate 28.819 ns |
| `result_handle_diagnostics/direct_metrics_record_task_completed` | 32.521-32.850 ns, estimate 32.695 ns |
| `result_handle_diagnostics/direct_public_wrapper_without_metrics` | 133.18-135.09 ns, estimate 134.10 ns |
| `result_handle_diagnostics/direct_public_wrapper_components` | 196.58-198.85 ns, estimate 197.68 ns |
| `result_handle_diagnostics/direct_registry_lifecycle` | 86.249-87.135 ns, estimate 86.766 ns |
| `result_handle_diagnostics/mutex_registry_register` | 44.510-45.247 ns, estimate 44.861 ns |
| `task_scheduling_overhead` | 533.08-540.29 ns, estimate 535.84 ns |
| `public_result_handle_ready/moirai_spawn_join_ready` | 529.27-556.48 ns, estimate 545.33 ns |
| `public_result_handle_ready/tokio_spawn_join_ready` | 1.9803-2.1555 us, estimate 2.0723 us |
| `public_result_handle_ready/moirai_scope_single_ready` | 525.82-538.29 ns, estimate 531.06 ns |
| `public_result_handle_ready/rayon_scope_single_ready` | 697.25-714.03 ns, estimate 705.37 ns |

Interpretation: the next production target is hot-path registry cost attribution and metrics tail cost, not result-slot publication, relaxed scheduler counter ordering, or an unmeasured allocator rewrite. A result-slot write-then-swap prototype improved direct slot rows but regressed public spawn/join and quiescent-barrier rows, so the retained `WRITING` state-machine publication remains authoritative. A relaxed submit-side scheduler counter prototype regressed `task_scheduling_overhead` to 565.06-585.15 ns and was reverted. The retained public comparison still keeps Moirai ahead of same-run Tokio and Rayon reference rows.

### 2026-05-23 Lock-Free Registry Allocator A/B Rejection

Commands:
```bash
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- "result_handle_diagnostics/(moirai_spawn_join_ready|direct_public_wrapper_without_metrics|direct_public_wrapper_components|direct_registry_lifecycle|mutex_registry_register)"
cargo bench -p moirai-benchmarks --bench performance_benchmarks -- task_scheduling_overhead
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- "public_result_handle_ready/(moirai_spawn_join_ready|tokio_spawn_join_ready|moirai_scope_single_ready|rayon_scope_single_ready)"
```

Workload: the candidate replaced the dense `TaskRegistry` plus executor `Arc<Mutex<TaskRegistry>>` access with a concurrent block allocator and atomic lifecycle slots. The candidate was rejected after the scheduling gate regressed, even though one focused public ready diagnostic improved.

| Candidate / retained check | Result |
| --- | ---: |
| Candidate `result_handle_diagnostics/moirai_spawn_join_ready` | 459.61-487.90 ns, estimate 472.26 ns |
| Candidate `result_handle_diagnostics/direct_public_wrapper_without_metrics` | 154.49-159.21 ns |
| Candidate `result_handle_diagnostics/direct_public_wrapper_components` | 217.26-228.65 ns |
| Candidate `result_handle_diagnostics/direct_registry_lifecycle` | 106.94-110.11 ns |
| Candidate `result_handle_diagnostics/mutex_registry_register` | 60.959-62.140 ns |
| Candidate `task_scheduling_overhead` | 558.97-595.53 ns, estimate 572.89 ns; Criterion reported regression |
| Retained dense-registry `task_scheduling_overhead` rerun | 598.40-629.85 ns, estimate 618.73 ns |
| Retained `public_result_handle_ready/moirai_spawn_join_ready` | 655.42-726.90 ns, estimate 684.75 ns |
| Retained `public_result_handle_ready/tokio_spawn_join_ready` | 2.0296-2.4495 us, estimate 2.2156 us |
| Retained `public_result_handle_ready/moirai_scope_single_ready` | 662.79-761.31 ns, estimate 709.02 ns |
| Retained `public_result_handle_ready/rayon_scope_single_ready` | 1.0464-2.6938 us, estimate 1.8604 us |

Interpretation: the lock-free lifecycle-slot allocator is not retained. Its atomic slot publication and concurrent block path improve one focused ready row but do not satisfy the public scheduling gate, and it regresses isolated registry component rows relative to the accepted dense-block policy. The production contract now rejects `ConcurrentTaskRegistry`, `register_unique_task_with_id`, and `register_unique(&self, id: u64)` while requiring dense `Vec<TaskStateBlock>` storage, boxed optional slots, and executor `Arc<Mutex<TaskRegistry>>` access. The retained same-run public comparison keeps Moirai ahead of Tokio `JoinHandle` and Rayon `scope` reference rows, but the next step is a finer registry cost split rather than another allocator rewrite.

### 2026-05-23 Registry Hot-Path Attribution

Command:
```bash
cargo bench -p moirai-benchmarks --features registry-diagnostics --bench result_handle_diagnostics -- "result_handle_diagnostics/(registry_mutex_lock_only|registry_task_state_construct|registry_block_lookup|registry_slot_initialize|registry_mark_started_existing_slot|registry_mark_completed_existing_slot|registry_lifecycle_timestamp_publication|mutex_registry_register|direct_registry_lifecycle)"
cargo bench -p moirai-benchmarks --bench performance_benchmarks -- task_scheduling_overhead
```

| Benchmark | Result |
| --- | ---: |
| `result_handle_diagnostics/registry_mutex_lock_only` | 7.8811-8.1597 ns |
| `result_handle_diagnostics/registry_task_state_construct` | 24.804-25.402 ns |
| `result_handle_diagnostics/registry_block_lookup` | 15.045-16.116 ns |
| `result_handle_diagnostics/registry_slot_initialize` | 38.079-41.812 ns |
| `result_handle_diagnostics/registry_mark_started_existing_slot` | 28.504-29.063 ns |
| `result_handle_diagnostics/registry_mark_completed_existing_slot` | 29.622-31.151 ns |
| `result_handle_diagnostics/registry_lifecycle_timestamp_publication` | 80.514-82.547 ns |
| `result_handle_diagnostics/mutex_registry_register` | 46.689-48.079 ns |
| `result_handle_diagnostics/direct_registry_lifecycle` | 85.659-90.931 ns |
| `task_scheduling_overhead` | 612.29-627.91 ns |

Interpretation: the registry mutex alone is not the dominant retained registry cost. `Option::insert` keeps slot initialization below the earlier aggregate row, cleanup releases empty trailing lifecycle blocks after completed slots are removed, and dense task-state slots no longer store a redundant task id. Metadata ids are derived from direct lookup, preserving observability while reducing per-slot state. The next registry candidate targets duration-preserving timestamp publication.

### 2026-05-24 Registry Completion Duration Invariant

Command:
```bash
cargo bench -p moirai-benchmarks --features registry-diagnostics --bench result_handle_diagnostics -- "result_handle_diagnostics/(registry_duration_offset_math|registry_mark_completed_existing_slot|registry_lifecycle_timestamp_publication|direct_registry_lifecycle|mutex_registry_register)"
cargo bench -p moirai-benchmarks --bench performance_benchmarks -- task_scheduling_overhead
```

| Benchmark | Result |
| --- | ---: |
| `result_handle_diagnostics/registry_duration_offset_math` | 448.09-449.99 ps |
| `result_handle_diagnostics/registry_mark_completed_existing_slot` | 27.520-27.636 ns |
| `result_handle_diagnostics/registry_lifecycle_timestamp_publication` | 73.194-73.648 ns |
| `result_handle_diagnostics/direct_registry_lifecycle` | 85.400-85.811 ns |
| `result_handle_diagnostics/mutex_registry_register` | 44.120-44.602 ns |
| `task_scheduling_overhead` | 533.17-546.20 ns |

Interpretation: lifecycle completion now asserts the monotonic timestamp invariant and computes execution duration with plain subtraction instead of saturating arithmetic. The diagnostic fixture also avoids saturating setup arithmetic, and Criterion reported `registry_duration_offset_math` improved by 19.856-20.303%. Criterion reported no scheduling-gate regression. `benchmark_contracts` rejects reintroducing saturating completion-duration arithmetic in both the production lifecycle completion path and the registry duration-math diagnostic row.

### 2026-05-24 Running Lifecycle Completion Fast Path

Commands:
```bash
cargo bench -p moirai-benchmarks --bench performance_benchmarks -- task_scheduling_overhead
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- "public_result_handle_ready/(moirai_spawn_join_ready|tokio_spawn_join_ready|moirai_scope_single_ready|rayon_scope_single_ready)"
```

| Benchmark | Result |
| --- | ---: |
| `task_scheduling_overhead` | 534.64-549.65 ns |
| `public_result_handle_ready/moirai_spawn_join_ready` | 502.43-514.85 ns |
| `public_result_handle_ready/tokio_spawn_join_ready` | 1.5021-1.5354 μs |
| `public_result_handle_ready/moirai_scope_single_ready` | 479.32-493.46 ns |
| `public_result_handle_ready/rayon_scope_single_ready` | 661.60-671.01 ns |

Interpretation: explicit running-token completion now bypasses the drop-path `Option` branch while preserving drop-based implicit completion. The standalone scheduling gate reported no regression, and the warm same-run public comparison reported Moirai ready handles improved by 11.801-14.296% while remaining ahead of Tokio. Moirai scoped completion improved by 3.8519-9.1959% while remaining ahead of Rayon.

### 2026-05-24 Scheduler Queue Advisory Counter

Commands:
```bash
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- "result_handle_diagnostics/(direct_scheduler_result_slot|direct_scheduler_result_slot_with_quiescent_barrier|moirai_spawn_join_ready)"
cargo bench -p moirai-benchmarks --bench performance_benchmarks -- task_scheduling_overhead
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- "public_result_handle_ready/(moirai_spawn_join_ready|tokio_spawn_join_ready|moirai_scope_single_ready|rayon_scope_single_ready)"
```

| Benchmark | Result |
| --- | ---: |
| `result_handle_diagnostics/direct_scheduler_result_slot` | 328.03-335.01 ns |
| `result_handle_diagnostics/direct_scheduler_result_slot_with_quiescent_barrier` | 364.37-376.51 ns |
| `result_handle_diagnostics/moirai_spawn_join_ready` | 515.40-519.76 ns |
| `task_scheduling_overhead` | 538.01-545.54 ns |
| `public_result_handle_ready/moirai_spawn_join_ready` | 598.80-605.81 ns |
| `public_result_handle_ready/tokio_spawn_join_ready` | 1.2040-1.3170 μs |
| `public_result_handle_ready/moirai_scope_single_ready` | 422.82-457.87 ns |
| `public_result_handle_ready/rayon_scope_single_ready` | 611.52-637.08 ns |

Interpretation: per-worker queue length is now an advisory relaxed counter because the queue mutex synchronizes `VecDeque<ScheduledJob>` contents and the scheduler's global pending/active counters synchronize quiescence. The diagnostic scheduler result-slot row improved by 29.512-32.047%, the default scheduling gate improved by 9.1269-14.220%, and the isolated public comparison kept Moirai ahead of Tokio and Rayon.

### 2026-05-24 Registry Timestamp Primitive Attribution

Commands:
```bash
cargo bench -p moirai-benchmarks --features registry-diagnostics --bench result_handle_diagnostics -- "result_handle_diagnostics/(registry_elapsed_nanos_since_origin|registry_start_release_publication|registry_completion_release_publication|registry_duration_offset_math|registry_mark_started_existing_slot|registry_mark_completed_existing_slot|registry_lifecycle_timestamp_publication)"
cargo bench -p moirai-benchmarks --bench performance_benchmarks -- task_scheduling_overhead --verbose
```

| Benchmark | Result |
| --- | ---: |
| `result_handle_diagnostics/registry_elapsed_nanos_since_origin` | 24.645-24.783 ns |
| `result_handle_diagnostics/registry_start_release_publication` | 940.34-945.05 ps |
| `result_handle_diagnostics/registry_completion_release_publication` | 563.93-566.76 ps |
| `result_handle_diagnostics/registry_duration_offset_math` | 449.67-453.51 ps |
| `result_handle_diagnostics/registry_mark_started_existing_slot` | 25.159-25.406 ns |
| `result_handle_diagnostics/registry_mark_completed_existing_slot` | 27.402-27.507 ns |
| `result_handle_diagnostics/registry_lifecycle_timestamp_publication` | 73.004-73.573 ns |
| `task_scheduling_overhead` | 531.85-540.70 ns after a noisy preceding same-command run at 635.02-654.40 ns |

Interpretation: precise elapsed-offset sampling dominates duration-preserving lifecycle publication. Release-store publication and duration subtraction are sub-nanosecond in isolation, so replacing atomics or duration math is not the next production target. The next timing candidate must reduce precise monotonic clock sampling without weakening start/completion timestamp precision.

### 2026-05-24 Rayon/Tokio Gap Refresh After Timestamp Split

Commands:
```bash
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- public_result_handle_ready
cargo bench -p moirai-benchmarks --bench thread_schedule_comparison -- "(ready_task_schedule|indexed_reduce_schedule)"
```

| Benchmark | Result |
| --- | ---: |
| `public_result_handle_ready/moirai_spawn_join_ready` | 506.20-516.98 ns |
| `public_result_handle_ready/tokio_spawn_join_ready` | 1.6938-1.8250 us |
| `public_result_handle_ready/moirai_spawn_join_captured_ready` | 516.68-523.19 ns |
| `public_result_handle_ready/tokio_spawn_join_captured_ready` | 1.6755-1.7911 us |
| `public_result_handle_ready/moirai_spawn_join_oversized_captured_ready` | 700.12-723.74 ns |
| `public_result_handle_ready/tokio_spawn_join_oversized_captured_ready` | 1.6593-1.6871 us |
| `public_result_handle_ready/moirai_spawn_async_ready` | 736.18-762.21 ns |
| `public_result_handle_ready/moirai_spawn_async_wake_once` | 756.79-761.38 ns |
| `public_result_handle_ready/tokio_spawn_async_wake_once` | 1.7899-1.9801 us |
| `public_result_handle_ready/moirai_scope_single_ready` | 495.48-506.85 ns |
| `public_result_handle_ready/rayon_scope_single_ready` | 656.84-668.62 ns |
| `ready_task_schedule/moirai_scope` | 19.044-19.341 us |
| `ready_task_schedule/tokio_spawn_ready` | 89.273-90.520 us |
| `ready_task_schedule/rayon_scope` | 80.283-81.728 us |
| `indexed_reduce_schedule/moirai_indexed_reduce` | 714.22-729.27 ns |
| `indexed_reduce_schedule/rayon_indexed` | 7.7215-8.1235 us |

Interpretation: active same-run Rayon/Tokio comparison rows remain closed for public result handles, scoped completion, and indexed reduction. Criterion reported local baseline regressions on several Moirai rows, so the next performance target is scheduler handoff and async wake variance, not expanding this audit to non-equivalent Tokio I/O or full Rayon adapter parity.

### 2026-05-24 Async Sender Cell And State Primitive Attribution

Commands:
```bash
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- "result_handle_diagnostics/(direct_async_|moirai_spawn_async_(ready|wake_once))"
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- public_result_handle_ready
cargo bench -p moirai-benchmarks --bench performance_benchmarks -- task_scheduling_overhead --verbose
```

| Benchmark | Result |
| --- | ---: |
| `result_handle_diagnostics/direct_async_idle_to_queued_state_claim` | 5.8180-6.0374 ns |
| `result_handle_diagnostics/direct_async_polling_to_notified_state_claim` | 5.6951-5.9612 ns |
| `result_handle_diagnostics/direct_async_notified_to_polling_state_claim` | 5.9365-6.2878 ns |
| `result_handle_diagnostics/direct_async_polling_to_idle_state_release` | 5.9494-6.3783 ns |
| `result_handle_diagnostics/direct_async_waker_from_arc` | 7.4358-8.3286 ns |
| `result_handle_diagnostics/direct_async_wake_by_ref_polling_notification` | 5.5297-5.8557 ns |
| `result_handle_diagnostics/moirai_spawn_async_ready` | 732.28-765.81 ns |
| `result_handle_diagnostics/moirai_spawn_async_wake_once` | 717.68-772.95 ns |
| `public_result_handle_ready/moirai_spawn_join_ready` | 539.08-551.09 ns |
| `public_result_handle_ready/tokio_spawn_join_ready` | 1.1703-1.2998 us |
| `public_result_handle_ready/moirai_spawn_join_captured_ready` | 385.42-425.65 ns |
| `public_result_handle_ready/tokio_spawn_join_captured_ready` | 1.6329-2.1362 us |
| `public_result_handle_ready/moirai_spawn_join_oversized_captured_ready` | 641.40-677.11 ns |
| `public_result_handle_ready/tokio_spawn_join_oversized_captured_ready` | 1.4031-1.4753 us |
| `public_result_handle_ready/moirai_spawn_async_ready` | 656.81-720.42 ns |
| `public_result_handle_ready/moirai_spawn_async_wake_once` | 666.99-755.75 ns |
| `public_result_handle_ready/tokio_spawn_async_wake_once` | 1.3831-1.4600 us |
| `public_result_handle_ready/moirai_scope_single_ready` | 377.13-445.53 ns |
| `public_result_handle_ready/rayon_scope_single_ready` | 618.68-635.16 ns |
| `task_scheduling_overhead` | warm rerun 535.74-541.45 ns after rebuild run 597.62-614.95 ns |

Interpretation: async state transitions and `wake_by_ref` are single-digit nanosecond primitives and are not the dominant async public-handle cost. The result-sender mutex is removed from production async completion and replaced with a state-machine-guarded inline sender cell. The active Rayon/Tokio comparison remains closed, while async-ready still shows local baseline variance and remains the next composition target.

### 2026-05-24 Async Future-Present Flag And Completion Split

Commands:
```bash
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- "result_handle_diagnostics/(direct_async_future_present_drop_flag|direct_async_ready_completion_components|moirai_spawn_async_(ready|wake_once))"
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- public_result_handle_ready
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- "public_result_handle_ready/(moirai_scope_single_ready|rayon_scope_single_ready)"
cargo bench -p moirai-benchmarks --bench thread_schedule_comparison -- "(ready_task_schedule|indexed_reduce_schedule)"
cargo bench -p moirai-benchmarks --bench performance_benchmarks -- task_scheduling_overhead --verbose
```

| Benchmark | Result |
| --- | ---: |
| `result_handle_diagnostics/direct_async_future_present_drop_flag` | 191.60-194.35 ps |
| `result_handle_diagnostics/direct_async_ready_completion_components` | 150.12-151.23 ns |
| `result_handle_diagnostics/moirai_spawn_async_ready` | 711.65-739.10 ns |
| `result_handle_diagnostics/moirai_spawn_async_wake_once` | 540.30-577.27 ns |
| `public_result_handle_ready/moirai_spawn_join_ready` | 427.66-476.57 ns |
| `public_result_handle_ready/tokio_spawn_join_ready` | 1.2135-1.3928 us |
| `public_result_handle_ready/moirai_spawn_join_captured_ready` | 386.76-414.42 ns |
| `public_result_handle_ready/tokio_spawn_join_captured_ready` | 1.2970-1.3807 us |
| `public_result_handle_ready/moirai_spawn_join_oversized_captured_ready` | 515.32-556.14 ns |
| `public_result_handle_ready/tokio_spawn_join_oversized_captured_ready` | 1.5046-1.6921 us |
| `public_result_handle_ready/moirai_spawn_async_ready` | 496.95-545.67 ns |
| `public_result_handle_ready/moirai_spawn_async_wake_once` | 531.01-623.14 ns |
| `public_result_handle_ready/tokio_spawn_async_wake_once` | 1.3826-1.6928 us |
| `public_result_handle_ready/moirai_scope_single_ready` | 816.49-942.87 ns |
| `public_result_handle_ready/rayon_scope_single_ready` | 49.100-111.89 us |
| `ready_task_schedule/moirai_scope` | 18.385-19.064 us |
| `ready_task_schedule/tokio_spawn_ready` | 5.1317-8.8370 ms |
| `ready_task_schedule/rayon_scope` | 38.031-91.723 us |
| `indexed_reduce_schedule/moirai_indexed_reduce` | 1.0172-1.1264 us |
| `indexed_reduce_schedule/rayon_indexed` | 23.985-69.895 us |
| `task_scheduling_overhead` | rerun 658.10-744.73 ns with no statistically significant change after an initial noisy regression sample |

Interpretation: `AsyncFutureState::future_present` now uses a poll-owner `UnsafeCell<bool>` flag instead of atomic synchronization. The flag itself is sub-nanosecond in the corrected diagnostic, while full ready completion is dominated by lifecycle/result publication and metrics work. Same-run Tokio/Rayon references remain slower in the accepted comparison scope, but the run shows high local variance, so the next target remains scheduler handoff and async completion variance attribution.

### 2026-05-24 Async Poll Guard Removal

Commands:
```bash
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- direct_async_future_present_drop_flag
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- direct_async_sender_cell_take_send_join
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- direct_async_lifecycle_complete
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- direct_async_completed_state_store
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- direct_metrics_record_task_completed
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- direct_async_ready_completion_components
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- moirai_spawn_async_ready
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- moirai_spawn_async
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- public_result_handle_ready
cargo bench -p moirai-benchmarks --bench performance_benchmarks -- task_scheduling_overhead
```

| Benchmark | Result |
| --- | ---: |
| `result_handle_diagnostics/direct_async_future_present_drop_flag` | 192.21-195.11 ps |
| `result_handle_diagnostics/direct_async_sender_cell_take_send_join` | 44.938-45.421 ns |
| `result_handle_diagnostics/direct_async_lifecycle_complete` | 48.098-48.898 ns |
| `result_handle_diagnostics/direct_async_completed_state_store` | 649.70-655.48 ps |
| `result_handle_diagnostics/direct_metrics_record_task_completed` | 32.354-32.644 ns |
| `result_handle_diagnostics/direct_async_ready_completion_components` | 148.04-148.58 ns |
| `result_handle_diagnostics/moirai_spawn_async_ready` | 652.71-665.92 ns |
| `result_handle_diagnostics/moirai_spawn_async_wake_once` | 551.11-579.84 ns |
| `public_result_handle_ready/moirai_spawn_join_ready` | 522.74-534.52 ns |
| `public_result_handle_ready/tokio_spawn_join_ready` | 1.2838-1.4109 us |
| `public_result_handle_ready/moirai_spawn_join_captured_ready` | 386.97-430.70 ns |
| `public_result_handle_ready/tokio_spawn_join_captured_ready` | 1.1734-1.2807 us |
| `public_result_handle_ready/moirai_spawn_join_oversized_captured_ready` | 492.81-536.62 ns |
| `public_result_handle_ready/tokio_spawn_join_oversized_captured_ready` | 1.1633-1.3113 us |
| `public_result_handle_ready/moirai_spawn_async_ready` | 509.28-541.09 ns |
| `public_result_handle_ready/moirai_spawn_async_wake_once` | 533.86-569.50 ns |
| `public_result_handle_ready/tokio_spawn_async_wake_once` | 1.4111-1.4953 us |
| `public_result_handle_ready/moirai_scope_single_ready` | 365.82-382.57 ns |
| `public_result_handle_ready/rayon_scope_single_ready` | 599.59-628.47 ns |
| `task_scheduling_overhead` | 540.37-550.84 ns |

Interpretation: `AsyncFutureState::poll` no longer reads the future-present flag before polling. The async state machine is the authoritative polling permission, so the flag remains only a drop guard for initialized future storage. The direct async ready-completion component row improved on rerun, async-ready diagnostics improved, and same-run Tokio/Rayon references remain slower on the accepted equivalent rows. The first full direct completion sample and the first async-ready diagnostic sample showed transient local variance; the retained evidence uses reruns with Criterion improvement or no-change classifications.

### 2026-05-24 Scheduler Submit And Async Arc Candidate Rejections

Commands:
```bash
cargo bench -p moirai-benchmarks --bench performance_benchmarks -- task_scheduling_overhead
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- "result_handle_diagnostics/(direct_scheduler_submit_join|direct_scheduler_result_slot|moirai_spawn_join_ready|moirai_spawn_async_ready|moirai_spawn_async_wake_once)"
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- public_result_handle_ready
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- "result_handle_diagnostics/(moirai_spawn_async_ready|moirai_spawn_async_wake_once)"
```

| Candidate | Benchmark | Result |
| --- | --- | ---: |
| Fetch-first pending count | `task_scheduling_overhead` | first run 578.84-596.03 ns regression; rerun 540.84-549.30 ns improvement |
| Fetch-first pending count | `direct_scheduler_submit_join` | 300.52-309.82 ns regression |
| Fetch-first pending count | `direct_scheduler_result_slot` | 309.57-317.04 ns, no statistically significant change |
| Fetch-first pending count | `public_result_handle_ready/moirai_spawn_join_oversized_captured_ready` | 768.63-831.30 ns regression |
| Fetch-first pending count | `public_result_handle_ready/moirai_spawn_async_ready` | 755.83-766.82 ns regression |
| Fetch-first pending count | `public_result_handle_ready/moirai_spawn_async_wake_once` | 762.84-769.81 ns regression |
| Fetch-first pending count | `public_result_handle_ready/moirai_scope_single_ready` | 523.35-540.57 ns regression |
| Move initial async state Arc into schedule | `moirai_spawn_async_ready` | 736.57-751.46 ns, no statistically significant change |
| Move initial async state Arc into schedule | `moirai_spawn_async_wake_once` | rerun 902.11-920.01 ns regression |

Interpretation: both candidates are rejected and reverted. Reusing `pending_tasks.fetch_add` as the worker-selection input removed one atomic load but widened the interval where global pending work was visible before queue publication and regressed public rows. Moving the freshly created async state `Arc` into initial scheduling removed one refcount pair but regressed wake-once. The retained path keeps selection before pending publication and keeps the extra async state owner through spawn metrics recording. A benchmark contract now rejects reintroducing the removed `scheduler-inline-handoff` feature and its `InlineHandoffSlot` source shape.

### 2026-05-24 Scheduler Submission Diagnostics And Public Gate

Commands:
```bash
cargo bench -p moirai-benchmarks --features scheduler-diagnostics --bench result_handle_diagnostics -- "result_handle_diagnostics/(direct_scheduler_(select_worker_serial|pending_counter_pair|worker_unpark|priority_queue_push_pop|submission_queue_publication)|direct_spawn_metrics_(before|after)_scheduler_submission)"
cargo bench -p moirai-benchmarks --bench performance_benchmarks -- task_scheduling_overhead
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- public_result_handle_ready
```

Workload: diagnostic rows use the production scheduler primitives with value assertions. The public rows use real `Moirai::spawn_fn`, `Moirai::spawn_async`, Tokio `JoinHandle`, Moirai `scope`, and Rayon `scope` paths.

| Benchmark | Result |
| --- | ---: |
| `direct_scheduler_select_worker_serial` | 1.1736-1.1792 ns |
| `direct_scheduler_pending_counter_pair` | 9.6017-9.9314 ns |
| `direct_scheduler_worker_unpark` | 27.731-28.763 ns |
| `direct_scheduler_priority_queue_push_pop` | 59.064-59.332 ns |
| `direct_scheduler_submission_queue_publication` | 67.131-67.829 ns |
| `direct_spawn_metrics_before_scheduler_submission` | 241.22-255.10 ns |
| `direct_spawn_metrics_after_scheduler_submission` | 225.53-254.91 ns |
| `task_scheduling_overhead` | 387.46-416.14 ns, Criterion improvement |
| `public_result_handle_ready/moirai_spawn_join_ready` | 477.68-493.23 ns |
| `public_result_handle_ready/tokio_spawn_join_ready` | 1.1178-1.2865 us |
| `public_result_handle_ready/moirai_spawn_join_captured_ready` | 344.89-357.08 ns |
| `public_result_handle_ready/tokio_spawn_join_captured_ready` | 986.24 ns-1.0404 us |
| `public_result_handle_ready/moirai_spawn_join_oversized_captured_ready` | 525.24-583.31 ns |
| `public_result_handle_ready/tokio_spawn_join_oversized_captured_ready` | 1.1105-1.1795 us |
| `public_result_handle_ready/moirai_spawn_async_ready` | 463.95-474.29 ns |
| `public_result_handle_ready/moirai_spawn_async_wake_once` | 480.29-490.62 ns |
| `public_result_handle_ready/tokio_spawn_async_wake_once` | 1.1903-1.3200 us |
| `public_result_handle_ready/moirai_scope_single_ready` | 275.67-285.21 ns |
| `public_result_handle_ready/rayon_scope_single_ready` | 591.62-614.02 ns |

Interpretation: queue publication is measurable but smaller than full result-slot scheduling. Recording spawn metrics before scheduler submission is not supported by this run; the retained after-submission ordering remains at least as fast in the diagnostic row and avoids overcounting failed submissions. Same-run public rows remain ahead of Tokio and Rayon references with value assertions.

### 2026-05-24 Scheduler Wake Decision Diagnostics

Commands:
```bash
cargo bench -p moirai-benchmarks --features scheduler-diagnostics --bench result_handle_diagnostics -- "result_handle_diagnostics/(direct_scheduler_(empty_wake_decision|contended_wake_decision|saturated_wake_decision|worker_unpark|submission_queue_publication))"
cargo bench -p moirai-benchmarks --bench performance_benchmarks -- task_scheduling_overhead
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- public_result_handle_ready
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- "public_result_handle_ready/(moirai_scope_single_ready|rayon_scope_single_ready)"
```

Workload: wake diagnostics use sealed zero-sized marker types for empty, contended, and saturated wake decisions. The diagnostic helper is feature-gated; production scheduling retains the direct hot-path wake branch after the shared helper candidate failed the scheduling gate.

| Benchmark | Result |
| --- | ---: |
| `direct_scheduler_worker_unpark` | 23.614-25.729 ns |
| `direct_scheduler_submission_queue_publication` | 66.705-67.185 ns |
| `direct_scheduler_empty_wake_decision` | 23.393-25.197 ns |
| `direct_scheduler_contended_wake_decision` | 404.11-409.07 ns |
| `direct_scheduler_saturated_wake_decision` | 374.20-376.44 ps |
| Shared production wake helper candidate | `task_scheduling_overhead` first gate 540.36-584.30 ns, Criterion regression |
| Retained direct production wake branch | `task_scheduling_overhead` 547.63-564.18 ns, no statistically significant change |
| `public_result_handle_ready/moirai_spawn_join_ready` | 557.71-631.50 ns |
| `public_result_handle_ready/tokio_spawn_join_ready` | 1.3841-1.6394 us |
| `public_result_handle_ready/moirai_spawn_join_captured_ready` | 390.96-441.82 ns |
| `public_result_handle_ready/tokio_spawn_join_captured_ready` | 1.4080-2.0717 us |
| `public_result_handle_ready/moirai_spawn_join_oversized_captured_ready` | 540.83-617.32 ns |
| `public_result_handle_ready/tokio_spawn_join_oversized_captured_ready` | 1.2054-1.3513 us |
| `public_result_handle_ready/moirai_spawn_async_ready` | 481.56-527.12 ns |
| `public_result_handle_ready/moirai_spawn_async_wake_once` | 542.25-598.55 ns |
| `public_result_handle_ready/tokio_spawn_async_wake_once` | 1.6997-2.4255 us |
| `public_result_handle_ready/moirai_scope_single_ready` filtered rerun | 565.99-576.65 ns |
| `public_result_handle_ready/rayon_scope_single_ready` filtered rerun | 687.81-702.03 ns |

Interpretation: the empty serial wake path costs the same order as a selected-worker unpark, while the contended wake-all path is the expensive branch at roughly 0.4 us. The default public result-handle rows remain ahead of same-run Tokio references, and the filtered scope rerun remains ahead of Rayon. The next production candidate should reduce wake-all frequency or replace the wake-all branch with a bounded static wake strategy, but only after preserving queued-work visibility and avoiding extra atomics on the serial path.

### 2026-05-24 Bounded Contended Wake Strategy

Commands:
```bash
cargo bench -p moirai-benchmarks --features scheduler-diagnostics --bench result_handle_diagnostics -- "result_handle_diagnostics/(direct_scheduler_(contended_wake_decision|empty_wake_decision|saturated_wake_decision))"
cargo bench -p moirai-benchmarks --bench performance_benchmarks -- task_scheduling_overhead
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- "public_result_handle_ready/(moirai_spawn_join_ready|tokio_spawn_join_ready|moirai_spawn_join_captured_ready|tokio_spawn_join_captured_ready|moirai_spawn_async_wake_once|tokio_spawn_async_wake_once)"
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- "public_result_handle_ready/(moirai_spawn_join_oversized_captured_ready|tokio_spawn_join_oversized_captured_ready|moirai_scope_single_ready|rayon_scope_single_ready)"
```

Workload: production contended submissions now use a sealed static `BoundedContendedWake` policy. The path wakes the selected queue owner plus one deterministic peer derived from `previous_pending`, stores no runtime policy object, allocates no memory, and adds no atomics to submission. The helper is `#[inline(never)]` so the serial `previous_pending == 0` branch remains compact while the contended path remains monomorphized.

| Benchmark | Result |
| --- | ---: |
| `direct_scheduler_empty_wake_decision` | 24.890-28.034 ns, no statistically significant change |
| `direct_scheduler_contended_wake_decision` | 162.41-180.11 ns, Criterion improvement |
| Previous contended wake-all diagnostic | 404.11-409.07 ns |
| `direct_scheduler_saturated_wake_decision` | 436.12-475.26 ps; diagnostic no-wake branch, not a production bottleneck |
| Inline bounded helper candidate | `task_scheduling_overhead` 526.36-548.89 ns, but public captured and async rows regressed on rerun |
| Retained no-inline bounded helper | `task_scheduling_overhead` 546.64-561.03 ns, change within noise threshold |
| `public_result_handle_ready/moirai_spawn_join_ready` | 563.74-579.31 ns |
| `public_result_handle_ready/tokio_spawn_join_ready` | 1.2717-1.3821 us |
| `public_result_handle_ready/moirai_spawn_join_captured_ready` | 473.92-493.81 ns |
| `public_result_handle_ready/tokio_spawn_join_captured_ready` | 1.2943-1.5040 us |
| `public_result_handle_ready/moirai_spawn_async_wake_once` | 553.83-578.44 ns |
| `public_result_handle_ready/tokio_spawn_async_wake_once` | 1.4885-1.5539 us |
| `public_result_handle_ready/moirai_spawn_join_oversized_captured_ready` | 706.14-759.37 ns |
| `public_result_handle_ready/tokio_spawn_join_oversized_captured_ready` | 1.3046-1.3845 us |
| `public_result_handle_ready/moirai_scope_single_ready` | 403.98-502.30 ns |
| `public_result_handle_ready/rayon_scope_single_ready` | 637.15-664.14 ns |

Interpretation: the retained bounded wake path cuts contended wake cost by more than half versus the measured wake-all diagnostic while preserving queued-work visibility. Public result-handle and scoped rows remain faster than same-run Tokio/Rayon references. The next target is reducing serial-path result-handle variance without increasing code size in `schedule_job`.

### 2026-05-24 Result Wait Spin Budget Reduction

Commands:
```bash
cargo bench -p moirai-benchmarks --features scheduler-diagnostics,result-diagnostics --bench result_handle_diagnostics -- "result_handle_diagnostics/(direct_result_slot_(ready_take|spin_miss|register_waiter|complete_waiting)|direct_scheduler_join_fast_spin_(quiescent|pending)|moirai_spawn_join_ready|direct_scheduler_result_slot|direct_scheduler_submit_join)"
cargo bench -p moirai-benchmarks --bench performance_benchmarks -- task_scheduling_overhead
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- "public_result_handle_ready/(moirai_spawn_join_ready|tokio_spawn_join_ready|moirai_spawn_join_captured_ready|tokio_spawn_join_captured_ready|moirai_spawn_join_oversized_captured_ready|tokio_spawn_join_oversized_captured_ready|moirai_spawn_async_wake_once|tokio_spawn_async_wake_once|moirai_scope_single_ready|rayon_scope_single_ready)"
```

Workload: blocking task-handle joins keep the sealed zero-sized `BlockingResultWait` policy and direct first READY-to-TAKEN CAS. The const spin budget is reduced from 100 to 64, so pending joins execute fewer relaxed load-gated spin probes before entering the existing single-waiter `thread::park` fallback. No runtime policy value, allocation, dynamic dispatch, or result-slot layout change is introduced.

| Benchmark | Result |
| --- | ---: |
| `direct_result_slot_ready_take` | 12.582-12.680 ns |
| `direct_result_slot_spin_miss` | 626.15-640.32 ns with 64 observed misses |
| Prior documented 100-spin miss diagnostic | 1.1886-1.4520 us |
| `direct_result_slot_register_waiter` | 12.234-12.306 ns |
| `direct_result_slot_complete_waiting` | 32.926-33.286 ns |
| `direct_scheduler_join_fast_spin_quiescent` | 378.74-383.25 ps |
| `direct_scheduler_join_fast_spin_pending` | 2.4682-2.5115 us |
| `direct_scheduler_submit_join` | 351.33-364.38 ns |
| `direct_scheduler_result_slot` | 475.24-503.71 ns |
| `task_scheduling_overhead` | 533.78-555.30 ns, no statistically significant change |
| `public_result_handle_ready/moirai_spawn_join_ready` | 521.02-531.69 ns |
| `public_result_handle_ready/tokio_spawn_join_ready` | 1.6124-1.6591 us |
| `public_result_handle_ready/moirai_spawn_join_captured_ready` | 544.29-560.10 ns |
| `public_result_handle_ready/tokio_spawn_join_captured_ready` | 1.6114-1.6486 us |
| `public_result_handle_ready/moirai_spawn_async_wake_once` | 706.01-728.66 ns |
| `public_result_handle_ready/tokio_spawn_async_wake_once` | 1.7862-2.0278 us |
| `public_result_handle_ready/moirai_spawn_join_oversized_captured_ready` | 763.44-774.27 ns |
| `public_result_handle_ready/tokio_spawn_join_oversized_captured_ready` | 1.6500-1.6994 us |
| `public_result_handle_ready/moirai_scope_single_ready` | 504.37-513.64 ns |
| `public_result_handle_ready/rayon_scope_single_ready` | 644.33-660.58 ns |

Interpretation: reducing the static blocking result-wait budget lowers the documented pending-spin miss cost while preserving the zero-sized wait policy and same-run Tokio/Rayon comparison wins. Criterion still reports local baseline regressions on captured, wake-once, oversized, and scope rows, so the next increment should isolate scheduler/result-publication variance instead of lowering the budget further.

### 2026-05-24 Mixed Unified Scheduler Comparison

Command:
```bash
cargo bench -p moirai-benchmarks --bench thread_schedule_comparison -- mixed_unified_schedule
```

Workload: 64 units per class. Each iteration combines completion-only sync fan-out, async result handles, and indexed reduction. Moirai executes all three through one runtime and one scheduler. The reference uses Tokio for async result handles and Rayon for scoped plus indexed work. Both rows assert `3 * n * (n + 1) / 2` before black-boxing the result.

| Benchmark | Result |
| --- | ---: |
| `mixed_unified_schedule/moirai_unified_mixed` | 42.000-42.856 us |
| `mixed_unified_schedule/tokio_rayon_mixed` | 53.337-55.645 us |

Interpretation: the active mixed-engine comparison gap is closed for the covered semantics. The single Moirai runtime is faster than coordinating Tokio plus Rayon for this value-checked mix of sync completion, async results, and indexed reduction.

Follow-up rerun after iterator base and streaming cleanup:

| Benchmark | Result |
| --- | ---: |
| `mixed_unified_schedule/moirai_unified_mixed` | 42.198-46.090 us |
| `mixed_unified_schedule/tokio_rayon_mixed` | 51.068-60.959 us |

Criterion reported no performance change for either row.

### 2026-05-24 Bounded Channel Matrix Spot Check

Command:
```bash
cargo bench -p moirai-benchmarks --bench channel_matrix -- tokio_mpsc/p1_c1
cargo bench -p moirai-benchmarks --bench channel_matrix -- moirai_mpmc/p1_c1
```

Workload: 8,192 integer items, one producer, capacity 1. Each row asserts the same closed-form sum before black-boxing.

| Benchmark | Result |
| --- | ---: |
| `bounded_channel_matrix/tokio_mpsc/p1_c1` | 2.4743-2.5095 ms |
| `bounded_channel_matrix/moirai_mpmc/p1_c1` | 1.4080-1.4638 ms |

Interpretation: Moirai remains ahead of Tokio on this bounded-channel spot check, but Criterion reports a local Moirai baseline regression. The next channel increment should isolate transport variance before changing core MPMC internals.

### Async Wake/Requeue and Inline Job Storage Diagnostic

Command:
```bash
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- "public_result_handle_ready/(moirai_spawn_async_ready|moirai_spawn_async_wake_once|tokio_spawn_async_wake_once)"
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- "result_handle_diagnostics/(moirai_spawn_async_(ready|wake_once)|direct_scheduler_max_inline|direct_scheduler_oversized_(captured|capture_read_one)_result_slot)"
```

| Benchmark | Result |
| --- | ---: |
| `public_result_handle_ready/moirai_spawn_async_ready` | 761.89-779.07 ns, estimate 770.27 ns |
| `public_result_handle_ready/moirai_spawn_async_wake_once` | 782.06-792.38 ns, estimate 786.57 ns |
| `public_result_handle_ready/tokio_spawn_async_wake_once` | 2.9087-3.1672 µs, estimate 3.0526 µs |
| `result_handle_diagnostics/moirai_spawn_async_ready` | 731.44-755.33 ns, estimate 742.87 ns |
| `result_handle_diagnostics/moirai_spawn_async_wake_once` | 772.48-796.90 ns, estimate 785.33 ns |
| `result_handle_diagnostics/direct_scheduler_max_inline_captured_result_slot` | 498.22-520.61 ns, estimate 508.69 ns |
| `result_handle_diagnostics/direct_scheduler_oversized_captured_result_slot` | 608.32-649.76 ns, estimate 632.32 ns |
| `result_handle_diagnostics/direct_scheduler_oversized_capture_read_one_result_slot` | 503.79-516.69 ns, estimate 510.07 ns |

Transport archive evidence: `cargo test -p moirai-transport --all-features safe_channel -- --nocapture` verifies that archived `String` receives borrow `&str` from the message buffer, reject malformed archives, and preserve `i32`/`String` value semantics. `cargo clippy -p moirai-transport --all-features -- -D warnings` passes.

### Result-Handle Diagnostic Breakdown

Command:
```bash
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics --verbose
```

Workload: direct `TaskHandle` construction and completion rows isolate result-slot cost from scheduler submission. The Moirai row runs the full public `Moirai::spawn_fn` plus `TaskHandle::join` path. Registry rows isolate lifecycle bookkeeping. Every result-bearing row asserts that the returned value is `42`.

| Benchmark | Result |
| --- | ---: |
| `result_handle_diagnostics/direct_ready_result_slot` | 38.548-39.209 ns, estimate 38.864 ns |
| `result_handle_diagnostics/direct_send_then_join_result_slot` | 48.293-49.115 ns, estimate 48.628 ns |
| `result_handle_diagnostics/direct_cross_thread_result_slot` | 67.868-75.571 µs, estimate 71.118 µs |
| `result_handle_diagnostics/moirai_spawn_join_ready` | 552.31-560.74 ns, estimate 556.80 ns |
| `result_handle_diagnostics/moirai_spawn_join_captured_ready` | 558.03-565.66 ns, estimate 561.92 ns |
| `result_handle_diagnostics/moirai_spawn_join_oversized_captured_ready` | 782.53-814.68 ns, estimate 797.54 ns |
| `result_handle_diagnostics/direct_scheduler_submit_join` | 336.87-348.66 ns, estimate 342.05 ns |
| `result_handle_diagnostics/direct_scheduler_result_slot` | 380.06-402.10 ns, estimate 390.89 ns |
| `result_handle_diagnostics/direct_scheduler_captured_result_slot` | 361.82-374.24 ns, estimate 367.84 ns |
| `result_handle_diagnostics/direct_scheduler_oversized_captured_result_slot` | 1.1686-1.1831 µs, estimate 1.1753 µs |
| `result_handle_diagnostics/direct_scheduler_pinned_oversized_captured_result_slot` | 408.70-437.83 ns, estimate 425.38 ns |
| `result_handle_diagnostics/direct_public_wrapper_components` | 201.56-205.32 ns, estimate 203.30 ns |
| `result_handle_diagnostics/direct_registry_lifecycle` | 83.821-86.814 ns, estimate 85.111 ns |
| `result_handle_diagnostics/mutex_registry_register` | 43.140-49.856 ns, estimate 47.078 ns |

Interpretation: direct same-thread result-slot ownership and completion remain below 50 ns after completion endpoints consume their satisfied drop guards. The pending wait path now avoids repeated failed READY-to-TAKEN RMW operations by using a relaxed state load during the monomorphized spin phase, while the already-ready path keeps a single direct claim CAS. Public result-handle rows remain dominated by scheduler/result handoff, public wrapper work, and capture storage shape; the executable public comparison is the authoritative Tokio/Rayon comparison surface for these paths.

### 2026-05-23 Quiescent-Barrier Diagnostic

Command:
```bash
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- result_handle_diagnostics
```

Workload: the same public and direct scheduler result-handle rows, with added variants that call the non-destructive scheduler quiescence barrier after the result handle has joined.

| Benchmark | Result |
| --- | ---: |
| `result_handle_diagnostics/moirai_spawn_join_ready` | 552.31-560.74 ns, estimate 556.80 ns |
| `result_handle_diagnostics/moirai_spawn_join_ready_with_quiescent_barrier` | 667.67-681.32 ns, estimate 674.17 ns |
| `result_handle_diagnostics/direct_scheduler_result_slot` | 380.06-402.10 ns, estimate 390.89 ns |
| `result_handle_diagnostics/direct_scheduler_result_slot_with_quiescent_barrier` | 272.61-286.91 ns, estimate 279.57 ns |

Interpretation: per-result-handle quiescence is not a public result-handle performance path. It adds scheduler barrier work after public `spawn_fn`/`join`, while the direct scheduler barrier row benefits from the fast quiescent spin. `Moirai::join` remains the correct batch-level process-fusion barrier when producers have finished submitting work; individual result handles should not force scheduler-wide quiescence.

### 2026-05-23 Public Wrapper Component Diagnostic

Command:
```bash
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- result_handle_diagnostics
```

Workload: direct public wrapper components execute without scheduler submission: task registry lifecycle, result handle creation, panic boundary, result publication, handle join, and executor metrics recording.

| Benchmark | Result |
| --- | ---: |
| `result_handle_diagnostics/direct_public_wrapper_components` | 201.56-205.32 ns, estimate 203.30 ns |
| `result_handle_diagnostics/direct_scheduler_result_slot` | 380.06-402.10 ns, estimate 390.89 ns |
| `result_handle_diagnostics/moirai_spawn_join_ready` | 552.31-560.74 ns, estimate 556.80 ns |
| `result_handle_diagnostics/moirai_spawn_join_captured_ready` | 558.03-565.66 ns, estimate 561.92 ns |
| `result_handle_diagnostics/moirai_spawn_join_oversized_captured_ready` | 782.53-814.68 ns, estimate 797.54 ns |
| `result_handle_diagnostics/direct_scheduler_captured_result_slot` | 361.82-374.24 ns, estimate 367.84 ns |
| `result_handle_diagnostics/direct_scheduler_oversized_captured_result_slot` | 1.1686-1.1831 µs, estimate 1.1753 µs |

Interpretation: public wrapper work is measurable but not the full public spawn/join delta. The next optimization target is scheduler result-handoff variance and capture storage shape. Result-slot ownership and per-handle process joining remain rejected paths; the earlier registry-owned ID allocation rejection is superseded by the verified `register_next_task` retention recorded on 2026-05-27.

### 2026-05-23 Oversized Capture Fallback Diagnostic

Command:
```bash
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- "(captured|hybrid_spawn_blocking|direct_public_wrapper)"
```

Workload: captured and oversized captured result-handle rows across `Moirai`, direct `HybridExecutor`, and direct scheduler/result-slot layers. The run follows the replacement of the separate raw-pointer heap job variant with a typed boxed closure stored behind the inline job trampoline.

| Benchmark | Result |
| --- | ---: |
| `result_handle_diagnostics/moirai_spawn_join_captured_ready` | 464.21-507.87 ns, estimate 483.11 ns |
| `result_handle_diagnostics/moirai_spawn_join_oversized_captured_ready` | 494.10-548.80 ns, estimate 519.19 ns |
| `result_handle_diagnostics/hybrid_spawn_blocking_ready` | 420.32-447.90 ns, estimate 433.25 ns |
| `result_handle_diagnostics/hybrid_spawn_blocking_captured_ready` | 405.45-428.14 ns, estimate 417.61 ns |
| `result_handle_diagnostics/hybrid_spawn_blocking_oversized_captured_ready` | 543.14-579.37 ns, estimate 561.50 ns |
| `result_handle_diagnostics/direct_scheduler_captured_result_slot` | 277.59-290.58 ns, estimate 283.71 ns |
| `result_handle_diagnostics/direct_scheduler_oversized_captured_result_slot` | 383.99-452.70 ns, estimate 417.15 ns |
| `result_handle_diagnostics/direct_public_wrapper_components` | 197.57-206.66 ns, estimate 201.43 ns |

Interpretation: the boxed inline trampoline keeps the common inline job footprint at two cache lines while reducing oversized capture fallback overhead. The next target is public scheduler handoff variance rather than the oversized closure storage representation.

### 2026-05-23 Lifecycle Timestamp Source Diagnostic

Command:
```bash
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- lifecycle_
```

Workload: scheduled result-bearing jobs with value assertions. Full lifecycle rows perform timestamp reads and atomic lifecycle stores. Elapsed-only rows keep timestamp reads without lifecycle stores. Atomic-only rows keep lifecycle stores without timestamp reads.

| Benchmark | Result |
| --- | ---: |
| `result_handle_diagnostics/direct_scheduler_lifecycle_before_send_result_slot` | 713.71-752.56 ns |
| `result_handle_diagnostics/direct_scheduler_lifecycle_elapsed_only_result_slot` | 789.51-826.72 ns |
| `result_handle_diagnostics/direct_scheduler_lifecycle_atomic_only_result_slot` | 678.37-722.82 ns |
| `result_handle_diagnostics/direct_scheduler_lifecycle_after_send_result_slot` | 780.94-851.15 ns |
| `result_handle_diagnostics/direct_scheduler_oversized_lifecycle_before_send_result_slot` | 733.17-816.18 ns |
| `result_handle_diagnostics/direct_scheduler_oversized_lifecycle_elapsed_only_result_slot` | 609.69-663.20 ns |
| `result_handle_diagnostics/direct_scheduler_oversized_lifecycle_atomic_only_result_slot` | 578.22-620.01 ns |
| `result_handle_diagnostics/direct_scheduler_oversized_lifecycle_after_send_result_slot` | 634.66-698.83 ns |

Interpretation: lifecycle timing cost is not isolated to one atomic store or one elapsed-time read. The latest same-run rows show atomic-only lifecycle below full lifecycle, while elapsed-only remains material and noisy. Production lifecycle completion remains before result publication; changing that boundary has mixed evidence and would alter task-status observability. The next valid production optimization must preserve duration metrics through an explicit timing policy or cheaper clock source.

### 2026-05-23 Token-Carried Start-Instant Lifecycle Diagnostic

Command:
```bash
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- lifecycle_
```

Workload: scheduled result-bearing jobs with value assertions. Start-instant rows model a production candidate where the running lifecycle token carries the start `Instant`, computes execution duration from that token, and reconstructs completion offset from start offset plus duration. Duration-only rows preserve only execution duration and are retained as a rejected control.

| Benchmark | Result |
| --- | ---: |
| `result_handle_diagnostics/direct_scheduler_lifecycle_before_send_result_slot` | 622.71-633.90 ns |
| `result_handle_diagnostics/direct_scheduler_lifecycle_elapsed_only_result_slot` | 614.85-626.18 ns |
| `result_handle_diagnostics/direct_scheduler_lifecycle_atomic_only_result_slot` | 567.10-587.23 ns |
| `result_handle_diagnostics/direct_scheduler_lifecycle_start_instant_result_slot` | 663.08-674.06 ns |
| `result_handle_diagnostics/direct_scheduler_lifecycle_duration_only_result_slot` | 651.87-663.34 ns |
| `result_handle_diagnostics/direct_scheduler_lifecycle_after_send_result_slot` | 606.62-619.70 ns |
| `result_handle_diagnostics/direct_scheduler_oversized_lifecycle_before_send_result_slot` | 768.33-789.98 ns |
| `result_handle_diagnostics/direct_scheduler_oversized_lifecycle_elapsed_only_result_slot` | 960.49-979.68 ns |
| `result_handle_diagnostics/direct_scheduler_oversized_lifecycle_atomic_only_result_slot` | 674.32-690.50 ns |
| `result_handle_diagnostics/direct_scheduler_oversized_lifecycle_start_instant_result_slot` | 755.60-770.63 ns |
| `result_handle_diagnostics/direct_scheduler_oversized_lifecycle_duration_only_result_slot` | 774.76-794.44 ns |
| `result_handle_diagnostics/direct_scheduler_oversized_lifecycle_after_send_result_slot` | 759.28-771.34 ns |

Interpretation: token-carried start-instant timing is not a valid production replacement. It regresses ready result availability relative to the retained full lifecycle row and only matches the oversized row within benchmark variance. Atomic-only timing remains faster but removes duration observability, so it remains a diagnostic control rather than a production policy.

### 2026-05-23 Cached Clock Lifecycle Diagnostic

Command:
```bash
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- lifecycle_
```

Workload: scheduled result-bearing jobs with value assertions. Cached-clock rows use a benchmark-local clock driver and lifecycle samples read cached atomic offsets only. This models the lower-bound overhead of scheduler-local clock reads while making timestamp precision explicitly coarser than the retained `Instant` policy.

| Benchmark | Result |
| --- | ---: |
| `result_handle_diagnostics/direct_scheduler_lifecycle_before_send_result_slot` | 615.74-625.31 ns |
| `result_handle_diagnostics/direct_scheduler_lifecycle_elapsed_only_result_slot` | 541.16-593.83 ns |
| `result_handle_diagnostics/direct_scheduler_lifecycle_atomic_only_result_slot` | 494.90-524.77 ns |
| `result_handle_diagnostics/direct_scheduler_lifecycle_start_instant_result_slot` | 585.40-625.03 ns |
| `result_handle_diagnostics/direct_scheduler_lifecycle_cached_clock_result_slot` | 440.76-459.14 ns |
| `result_handle_diagnostics/direct_scheduler_lifecycle_duration_only_result_slot` | 592.40-628.63 ns |
| `result_handle_diagnostics/direct_scheduler_lifecycle_after_send_result_slot` | 573.88-597.97 ns |
| `result_handle_diagnostics/direct_scheduler_oversized_lifecycle_before_send_result_slot` | 749.88-841.37 ns |
| `result_handle_diagnostics/direct_scheduler_oversized_lifecycle_elapsed_only_result_slot` | 715.55-762.45 ns |
| `result_handle_diagnostics/direct_scheduler_oversized_lifecycle_atomic_only_result_slot` | 596.34-622.49 ns |
| `result_handle_diagnostics/direct_scheduler_oversized_lifecycle_start_instant_result_slot` | 671.03-704.26 ns |
| `result_handle_diagnostics/direct_scheduler_oversized_lifecycle_cached_clock_result_slot` | 625.52-682.42 ns |
| `result_handle_diagnostics/direct_scheduler_oversized_lifecycle_duration_only_result_slot` | 667.51-769.79 ns |
| `result_handle_diagnostics/direct_scheduler_oversized_lifecycle_after_send_result_slot` | 724.31-779.63 ns |

Interpretation: cached-clock timing proves timestamp reads remain a material overhead source, but the benchmark policy is not production-valid because start and completion offsets are bounded by the clock driver's update cadence rather than exact task transition instants. The next production candidate must keep exact or explicitly bounded timing semantics through a precise low-overhead monotonic clock source.

### 2026-05-23 Production QPC Lifecycle A/B Rejection

Commands:
```bash
cargo bench -p moirai-benchmarks --bench result_handle_diagnostics -- "moirai_spawn_join_(ready|oversized_captured_ready|oversized_capture_read_one)|hybrid_spawn_blocking_(ready|oversized_captured_ready|oversized_capture_read_one)|direct_scheduler_(lifecycle_before_send_result_slot|lifecycle_qpc_result_slot|oversized_lifecycle_before_send_result_slot|oversized_lifecycle_qpc_result_slot)"
cargo bench -p moirai-benchmarks --bench performance_benchmarks task_scheduling_overhead --verbose
```

Workload: production registry temporarily used Windows QPC lifecycle timing, then the change was reverted after the public-path A/B failed the retention criterion.

| Candidate / retained check | Result |
| --- | ---: |
| QPC candidate `moirai_spawn_join_ready` | 593.06-600.59 ns |
| QPC candidate `moirai_spawn_join_oversized_captured_ready` | 880.62-947.27 ns |
| QPC candidate `moirai_spawn_join_oversized_capture_read_one` | 566.55-616.57 ns |
| QPC candidate `hybrid_spawn_blocking_ready` | 506.39-532.28 ns |
| QPC candidate `hybrid_spawn_blocking_oversized_captured_ready` | 681.48-745.94 ns |
| QPC candidate `hybrid_spawn_blocking_oversized_capture_read_one` | 556.82-581.09 ns |
| Retained post-revert `task_scheduling_overhead` | 528.88-535.17 ns |

Interpretation: QPC remains diagnostic-only. It improves some ready and read-one paths, but it regresses the public oversized captured path. The production registry retains `Instant` lifecycle timing, and `benchmark_contracts` now verifies QPC stays out of the production registry path.

### 2026-05-23 Quick Rayon/Tokio Gap Rerun

Commands:
```bash
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- public_result_handle_ready
cargo bench -p moirai-benchmarks --bench thread_schedule_comparison -- ready_task_schedule
cargo bench -p moirai-benchmarks --bench thread_schedule_comparison -- indexed_reduce_schedule
```

Workload: executable Moirai/Tokio/Rayon comparison rows with value assertions before timing.

| Capability | Moirai | Reference |
| --- | ---: | ---: |
| Ready result handle | 609.17-671.88 ns | Tokio `JoinHandle`: 1.4148-1.5034 us |
| Captured ready result handle | 472.08-492.82 ns | Tokio `JoinHandle`: 1.3693-1.4251 us |
| Oversized captured result handle | 553.10-602.91 ns | Tokio `JoinHandle`: 1.3400-1.4143 us |
| Async-ready result handle | 553.20-592.73 ns | Tokio ready `JoinHandle`: 1.4148-1.5034 us |
| Wake-once async result handle | 560.44-622.27 ns | Tokio wake-once `JoinHandle`: 1.3573-1.5108 us |
| Single scoped completion | 327.38-357.03 ns | Rayon `scope`: 608.41-627.05 ns |
| 256 ready scoped tasks | 11.883-14.014 us | Tokio spawn: 78.105-80.982 us; Rayon scope: 75.660-103.65 us |
| 256 indexed reduction | 567.08-939.54 ns | Rayon indexed: 4.0030-6.2718 us |

Interpretation: the active public result-handle, scoped completion, and indexed-reduction comparison scope has no Rayon/Tokio performance gap in this post-revert rerun. Moirai remains below the equivalent Tokio and Rayon rows in each accepted comparison.

### 2026-05-23 Scoped Dynamic Dispatch Removal

Commands:
```bash
cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- scope_single
cargo bench -p moirai-benchmarks --bench thread_schedule_comparison -- scoped_ready_scaling
```

Workload: borrowed scoped ready jobs after `SchedulerScope` moved from boxed `dyn FnOnce` buffering to inline `ScheduledJob` buffering. The single-scope row compares the public completion-only Moirai scope path against Rayon scope. The scaling rows compare Moirai scope, Rayon scope, and Tokio ready spawn with value assertions.

| Benchmark | Result |
| --- | ---: |
| `public_result_handle_ready/moirai_scope_single_ready` | 596.74-607.93 ns |
| `public_result_handle_ready/rayon_scope_single_ready` | 687.89-697.33 ns |
| `scoped_ready_scaling/moirai_scope/64` | 5.3109-6.7267 µs |
| `scoped_ready_scaling/rayon_scope/64` | 18.661-26.285 µs |
| `scoped_ready_scaling/tokio_spawn_ready/64` | 48.915-77.235 µs |
| `scoped_ready_scaling/moirai_scope/256` | 14.624-15.144 µs |
| `scoped_ready_scaling/rayon_scope/256` | 62.561-69.633 µs |
| `scoped_ready_scaling/tokio_spawn_ready/256` | 94.252-149.61 µs |
| `scoped_ready_scaling/moirai_scope/1024` | 51.506-52.870 µs |
| `scoped_ready_scaling/rayon_scope/1024` | 284.56-290.76 µs |
| `scoped_ready_scaling/tokio_spawn_ready/1024` | 349.72-368.84 µs |

Interpretation: scoped dynamic-dispatch removal keeps the single-scope row ahead of Rayon and materially improves the multi-job scoped rows. The retained design registers scoped completion at spawn time, stores one inline scheduled job per scoped logical job, and schedules single scoped jobs directly without an extra wrapper job.

### 2026-05-23 Ready Task Schedule After Timeout Inline Storage

Command:
```bash
cargo bench -p moirai-benchmarks --bench thread_schedule_comparison -- ready_task_schedule
```

Workload: 256 ready work units with value assertions. This run followed the `Timeout<F>` storage change from `Pin<Box<F>>` to inline `F`; the benchmark keeps the active Rayon/Tokio scheduler comparison current.

| Benchmark | Result |
| --- | ---: |
| `ready_task_schedule/moirai_scope` | 13.962-14.075 µs |
| `ready_task_schedule/tokio_spawn_ready` | 83.851-85.366 µs |
| `ready_task_schedule/rayon_scope` | 79.360-82.596 µs |

Interpretation: the timeout and async-executor future-storage changes do not weaken the active scheduler comparison surface. Criterion reported history regressions for all three rows, including Tokio and Rayon, so the same-run comparison is the relevant signal. Moirai scope remains ahead of the Tokio and Rayon ready-work rows in the same run.

### 2026-05-23 Iterator ThreadPool And Caller-Lane Dispatch Audit

Commands:
```bash
cargo bench -p moirai-benchmarks --bench example_pattern_comparison -- example_rayon_patterns
cargo bench -p moirai-benchmarks --bench industry_comparison -- official_rayon_map_reduce
```

Workload: the example row measures the public example checksum pattern at 65,536 items through fixed-pool Rayon and Moirai indexed reduction. The industry row measures the official Rayon `into_par_iter().map(...).sum()` reference against Moirai indexed reduction at three sizes.

| Benchmark | Result |
| --- | ---: |
| `example_rayon_patterns/rayon_parallel_iterator` | 380.51-403.21 µs |
| `example_rayon_patterns/moirai_indexed_reduce` | 330.64-351.94 µs |
| `official_rayon_map_reduce/moirai_4096` | 2.6761-2.7742 µs |
| `official_rayon_map_reduce/rayon_4096` | 14.837-16.423 µs |
| `official_rayon_map_reduce/moirai_32768` | 13.258-14.134 µs |
| `official_rayon_map_reduce/rayon_32768` | 27.562-31.202 µs |
| `official_rayon_map_reduce/moirai_65536` | 22.735-23.425 µs |
| `official_rayon_map_reduce/rayon_65536` | 37.199-40.844 µs |

Interpretation: replacing the iterator thread-pool boxed job queue with `ErasedThreadJob` closes a dynamic-dispatch infrastructure gap. Correcting indexed chunk caps to include the caller execution lane closes the example-pattern Rayon variance while preserving the official Rayon-pattern comparison lead for Moirai at all measured sizes.

## 2026-05-23 Transport Archive Benchmark

Command:
```bash
cargo bench -p moirai-benchmarks --bench transport_archive_comparison --verbose
```

Workload: the same UTF-8 archive payload is validated through borrowed archive views and an owned-decode reference. The round-trip rows use `TransportManager` for both paths, then validate that the decoded or borrowed string equals the expected payload.

| Benchmark | Result |
| --- | ---: |
| `transport_archive_view/borrowed_archive_view` | 15.913-16.095 ns, estimate 15.996 ns |
| `transport_archive_view/owned_decode_reference` | 32.097-32.415 ns, estimate 32.237 ns |
| `transport_archive_roundtrip/archived_transport_borrowed_view` | 233.63-237.09 ns, estimate 235.17 ns |
| `transport_archive_roundtrip/raw_transport_owned_decode_reference` | 259.54-261.53 ns, estimate 260.44 ns |

Interpretation: the borrowed archive view is roughly half the direct receive-view cost of owned decode for the same bytes. Through `TransportManager`, the borrowed archive path remains ahead while preserving value assertions and avoiding receive-side `String` allocation.

## 2026-05-24 Timer Wheel Cancellation And Mixed Scheduler Rerun

Commands:
```bash
cargo bench -p moirai-benchmarks --bench example_pattern_comparison -- example_tokio_fanout
cargo bench -p moirai-benchmarks --bench thread_schedule_comparison -- mixed_unified_schedule
```

Workload: `example_tokio_fanout` compares 256 async sleep tasks with checksum validation through Tokio and Moirai. `mixed_unified_schedule` combines sync scoped completion, async result handles, and indexed reduction through one Moirai scheduler versus Tokio plus Rayon with the same closed-form checksum.

| Benchmark | Result |
| --- | ---: |
| `example_tokio_fanout/tokio_spawn_sleep` | 15.356-15.597 ms |
| `example_tokio_fanout/moirai_spawn_async_sleep` | 15.518-15.636 ms |
| `mixed_unified_schedule/moirai_unified_mixed` | 44.023-44.699 µs |
| `mixed_unified_schedule/tokio_rayon_mixed` | 57.095-58.571 µs |

Interpretation: timer fanout intervals overlap after replacing the placeholder `TimerWheel::cancel` path with real lazy cancellation state. The mixed unified-scheduler row remains below the two-engine Tokio plus Rayon reference after the async timer edit.

## 2026-05-24 Rayon Adapter Transform Expansion

Commands:
```bash
cargo bench -p moirai-benchmarks --bench example_pattern_comparison -- example_rayon_patterns
target/release/deps/thread_schedule_comparison-3109b32b962825f1.exe mixed_unified_schedule --bench
```

Workload: `example_rayon_patterns` compares fixed-pool Rayon `into_par_iter().map(...).sum()` with Moirai indexed reduction over the same checksum workload. `mixed_unified_schedule` refreshes the one-scheduler Moirai row against the Tokio plus Rayon reference after adding value-tested `filter_map` and `flat_map` adapters to the non-indexed Rayon-style adapter subset.

| Benchmark | Result |
| --- | ---: |
| `example_rayon_patterns/rayon_parallel_iterator` | 1.6761-2.8179 ms |
| `example_rayon_patterns/moirai_indexed_reduce` | 532.31-551.27 µs |
| `mixed_unified_schedule/moirai_unified_mixed` | 42.266-42.711 µs |
| `mixed_unified_schedule/tokio_rayon_mixed` | 55.762-60.278 µs |

Interpretation: both example-pattern rows regressed against their local Criterion histories under this run, but the same-run Moirai indexed path remains faster than the Rayon reference. The direct Criterion executable was used for the mixed scheduler rerun after the Cargo wrapper hit the 300s timeout waiting behind unrelated Cargo jobs; the direct benchmark completed and kept Moirai below the Tokio plus Rayon two-engine reference.

## 2026-05-25 Rayon Adapter Utility Expansion

Commands:
```bash
cargo bench -p moirai-benchmarks --bench iterator_adapter_comparison -- --quiet
cargo bench -p moirai-benchmarks --bench thread_schedule_comparison -- mixed_unified_schedule
```

Workload: `iterator_adapter_comparison` validates each Moirai adapter pipeline against the equivalent Rayon pipeline before timing. The new rows cover `inspect`, `panic_fuse`, `chunks`, and `partition`. The mixed scheduler row refreshes the one-runtime Moirai comparison against Tokio plus Rayon after the iterator adapter changes.

| Benchmark | Moirai | Reference |
| --- | ---: | ---: |
| `iterator_adapter_indexed_pipeline` | 35.664-35.796 µs | Rayon: 318.76-322.01 µs |
| `iterator_adapter_filter_flat_pipeline` | 22.001-22.292 µs | Rayon: 2.9053-3.0355 ms |
| `iterator_adapter_chain_rev_pipeline` | 17.993-18.389 µs | Rayon: 76.454-80.386 µs |
| `iterator_adapter_inspect_chunks_pipeline` | 31.061-31.810 µs | Rayon: 36.916-38.040 µs |
| `iterator_adapter_partition_pipeline` | 29.242-30.103 µs | Rayon: 658.16-693.21 µs |
| `mixed_unified_schedule` | 42.437-43.169 µs | Tokio plus Rayon: 48.574-52.167 µs |

Interpretation: the utility adapter expansion keeps every value-checked adapter comparison row ahead of the same-run Rayon reference after eliminating the partition collector recursion and removing avoidable inspect allocation before chunking. The mixed unified-scheduler comparison remains ahead of the two-engine Tokio plus Rayon reference.

## 2026-05-25 Sorting Slice Extension Boundary

Command:
```bash
cargo bench -p moirai-benchmarks --bench sorting_comparison -- --quiet
```

Workload: `sorting_comparison` validates stable and unstable Moirai slice sorting against Rayon `ParallelSliceMut` before timing. Each benchmark iteration clones the same 10,000-item input and sorts the clone.

| Benchmark | Moirai | Reference |
| --- | ---: | ---: |
| `parallel_sorting_stable` | 76.225-78.202 µs | Rayon: 143.38-146.10 µs |
| `parallel_sorting_unstable` | 48.838-51.041 µs | Rayon: 66.725-69.234 µs |

Interpretation: the dedicated `ParallelSliceMut` boundary keeps sorting out of the non-indexed `ParallelIterator` adapter surface while providing value-checked Rayon comparison rows for stable and unstable in-place sorting.

## 2026-05-27 Bounded Indexed Source Boundary

Command:
```bash
cargo bench -p moirai-benchmarks --bench iterator_adapter_comparison -- iterator_indexed_boundary --quiet
```

Workload: `iterator_indexed_boundary` preconstructs equivalent Moirai and Rayon owned, empty, and range sources, asserts the same `(owned_len, empty_flag, range_len)` tuple, and times only the exact-size metadata calls. Owned Moirai `Vec<T>` sources use the by-value `VecParIter<T>` path without `Arc<Vec<T>>`.

| Benchmark | Moirai | Reference |
| --- | ---: | ---: |
| `iterator_indexed_boundary` | 1.8682-1.8871 ns | Rayon: 1.8668-1.8727 ns |

Interpretation: the bounded `IndexedParallelIterator::{len, is_empty}` source-cardinality boundary is at Rayon metadata-call parity for the audited exact-size source subset. The benchmark does not claim Rayon's full indexed producer/consumer adapter model.

## Upstream Comparison Patterns

- Tokio comparison rows use the documented `tokio::spawn` plus `JoinHandle` pattern from the Tokio task-spawning guide.
- Rayon ready-work rows use Rayon `scope`, and indexed/map-reduce rows use `into_par_iter().map(...).sum()`.
- `cargo test -p moirai-benchmarks --test benchmark_contracts` verifies benchmark source integrity and value correctness for the comparison paths.
- `docs/rayon_tokio_gap_audit.md` maps the active Rayon/Tokio scheduler comparison scope to executable benchmark targets and zero-cost invariant checks.
- This file reports only executable Criterion benchmark results from this repository. Removed stale non-executable estimates are not valid benchmark evidence.
