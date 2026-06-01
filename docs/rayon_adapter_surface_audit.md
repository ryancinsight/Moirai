# Rayon Adapter Surface Audit

## Scope

This audit covers the `moirai-iter::parallel` public adapter surface against the Rayon `ParallelIterator` style API. It is separate from `docs/rayon_tokio_gap_audit.md`, which covers scheduler, result-handle, scoped work, indexed reduction, transport, and runtime dependency boundaries.

The audit classifies only repository-local evidence. It does not claim drop-in Rayon parity unless the active source exposes the adapter and tests or benchmarks verify value semantics.

## Verdict

Moirai does not currently provide full Rayon adapter parity. The supported surface is a focused subset:

- `IntoParallelIterator` for `Vec<T>` and `Range<usize>`.
- `IntoParallelRefIterator` for `Vec<T>` without requiring `T: Clone + 'static`.
- `IndexedParallelIterator` for exact-size source iterators: by-value `VecParIter<T>`, `VecRefParIter<'_, T>`, `RefVecParIter<'_, T>`, `RangeParIter<usize>`, exact-size sequential adapters, `len`, `is_empty`, `collect_into_vec`, `unzip_into_vecs`, `interleave`, `interleave_shortest`, `step_by`, `by_exponential_blocks`, and `by_uniform_blocks`.
- `ParallelIterator::map`, `map_with`, `map_init`, `update`, `filter`, `inspect`, `panic_fuse`, `filter_map`, `while_some`, `flat_map`, `flat_map_iter`, `flatten`, `flatten_iter`, `enumerate`, `zip`, `zip_eq`, `copied`, `cloned`, `take`, `skip`, `take_any`, `skip_any`, `take_any_while`, `skip_any_while`, `chain`, `intersperse`, `rev`, `chunks`, `partition`, `partition_map`, `unzip`, `collect`, `collect_vec_list`, `count`, `any`, `all`, `find_any`, `find_first`, `find_last`, `position_any`, `position_first`, `position_last`, `positions`, `find_map_any`, `find_map_first`, `find_map_last`, `for_each`, `for_each_with`, `for_each_init`, `try_for_each`, `try_for_each_with`, `try_for_each_init`, `reduce`, `reduce_with`, `try_reduce`, `try_reduce_with`, `sum`, `product`, `min`, `max`, `min_by`, `max_by`, `min_by_key`, `max_by_key`, and `fold`.
- `ParallelExtend<T>` for `Vec<T>`.
- `ParallelSliceMut` for the slice extension sorting boundary.

The active competitive Rayon comparison remains `Moirai::map_reduce_indexed` versus fixed-pool Rayon `into_par_iter().map(...).sum()`. The `moirai-iter::parallel` trait surface is not the active performance comparison path.

Indexed scheduler execution is exposed through `Moirai::for_each_indexed` and `Moirai::map_reduce_indexed`. `moirai-iter::parallel` now also exposes a bounded indexed source boundary for exact source cardinality, caller-provided collection, pair splitting, source interleaving, fixed-stride source selection, and value-preserving block adapter names. This is not the full Rayon indexed producer/consumer adapter model and must not be documented as full Rayon compatibility.

## Adapter Matrix

| Rayon-style capability | Moirai surface | Evidence | Status |
| --- | --- | --- | --- |
| Owned vector parallel iteration | `impl IntoParallelIterator for Vec<T>` | `moirai-iter/src/parallel/sources.rs`; unit tests for map/filter/reduce | Covered subset |
| Range parallel iteration | `impl IntoParallelIterator for Range<usize>` | `par_range`; `test_range_parallel` | Covered subset |
| Borrowed vector iteration | `impl IntoParallelRefIterator for Vec<T>` | `VecRefParIter<'data, T>`, `RefVecParIter<'a, T>`, `test_non_clone_parallel_ref_iterator_maps_borrowed_values`, and `iterator_adapter_non_clone_ref_map` | Covered subset |
| Indexed source cardinality | `IndexedParallelIterator::{len, is_empty}` for exact-size source iterators | `test_indexed_parallel_iterator_reports_source_lengths`; `iterator_indexed_boundary` against Rayon measured Moirai at 1.8682-1.8871 ns and Rayon at 1.8668-1.8727 ns | Bounded indexed source boundary |
| Indexed source collect-into-vec | `IndexedParallelIterator::collect_into_vec` for exact-size source iterators | `test_indexed_collect_into_vec_moves_non_clone_values`; `iterator_indexed_collect_into_vec` against Rayon measured Moirai at 54.745-75.638 us and Rayon at 95.255-102.59 us | Bounded indexed source boundary |
| Indexed source unzip-into-vecs | `IndexedParallelIterator::unzip_into_vecs` for exact-size pair source iterators | `test_indexed_unzip_into_vecs_moves_non_clone_pairs_into_existing_storage`; `iterator_indexed_unzip_into_vecs` against Rayon measured Moirai at 256.72-273.34 us and Rayon at 268.81-303.00 us | Bounded indexed source boundary |
| Indexed source interleave | `IndexedParallelIterator::{interleave, interleave_shortest}` for exact-size source iterators | `test_indexed_interleave_moves_non_clone_values_without_clone_bound`; `test_indexed_interleave_shortest_drops_truncated_tail_once`; `iterator_indexed_interleave` against Rayon measured Moirai at 401.13-439.28 us and Rayon at 433.44-453.31 us | Bounded indexed source boundary |
| Indexed source step-by | `IndexedParallelIterator::step_by` for exact-size source iterators | `test_indexed_step_by_moves_non_clone_values_without_clone_bound`; `test_indexed_step_by_reports_exact_length`; `test_indexed_step_by_rejects_zero_step`; `test_indexed_step_by_drops_skipped_values_once`; `iterator_indexed_step_by` against Rayon measured Moirai at 24.335-25.830 us and Rayon at 65.191-67.990 us | Bounded indexed source boundary |
| Indexed source block adapters | `IndexedParallelIterator::{by_exponential_blocks, by_uniform_blocks}` for exact-size source iterators | `test_indexed_block_adapters_preserve_values_without_clone_bound`; `test_indexed_by_uniform_blocks_rejects_zero_size`; zero-sized block policy marker test; `iterator_indexed_blocks` against Rayon measured Moirai at 30.128-32.300 us and Rayon at 4.4301-4.5698 ms | Bounded logical-output block boundary |
| Map adapters | `ParallelIterator::map`, `map_with`, `map_init`, `Map<I, F>`, `MapWith<I, T, F>`, and `MapInit<I, Init, F>` | `test_parallel_map`, `test_parallel_map_with_uses_cloned_state`, `test_parallel_map_init_uses_initialized_state`, and `iterator_adapter_map_state` benchmark rows | Covered subset |
| Mutation adapter | `ParallelIterator::update` and `Update<I, F>` | `test_parallel_update_mutates_items_before_yielding` and `iterator_adapter_update` benchmark rows | Covered subset |
| Filter adapter | `ParallelIterator::filter` and `Filter<I, F>` | `test_parallel_filter` | Covered subset |
| Inspect adapter | `ParallelIterator::inspect` and `Inspect<I, F>` | `test_parallel_inspect_observes_items_without_changing_output` validates observation without changing output | Covered subset |
| Panic-fuse adapter | `ParallelIterator::panic_fuse` and `PanicFuse<I>` | `test_parallel_panic_fuse_preserves_values`, `test_parallel_panic_fuse_propagates_panic`, and a zero-sized `PanicFusePolicy` test | Covered subset |
| Filter-map adapter | `ParallelIterator::filter_map` and `FilterMap<I, F>` | `test_parallel_filter_map_retains_present_values` validates optional retention semantics | Covered subset |
| While-some adapter | `ParallelIterator::while_some` and `WhileSome<I>` | `test_parallel_while_some_unwraps_present_prefix`, `test_parallel_while_some_empty_when_first_is_none`, and `iterator_adapter_while_some` benchmark rows | Covered subset |
| Flat-map adapters | `ParallelIterator::{flat_map, flat_map_iter}` and `FlatMap<I, F>` | `test_parallel_flat_map_preserves_flattened_order`, `test_parallel_flat_map_iter_accepts_serial_inner_iterators`, and `iterator_adapter_filter_flat_pipeline` against Rayon's `flat_map_iter` validate serial-inner flattened output order | Covered serial-inner subset |
| Flatten adapters | `ParallelIterator::{flatten, flatten_iter}` and `Flatten<I>` | `test_parallel_flatten_preserves_nested_order`, `test_parallel_flatten_iter_preserves_serial_inner_order`, and `iterator_adapter_flatten` against Rayon's `flatten_iter` validate serial-inner flattened output order | Covered serial-inner subset |
| Enumerate adapter | `ParallelIterator::enumerate` and `Enumerate<I>` | `test_parallel_enumerate_pairs_logical_indices` validates zero-based logical positions | Covered subset |
| Zip adapter | `ParallelIterator::zip` and `Zip<I, J>` | `test_parallel_zip_stops_at_shorter_input` validates shortest-input semantics | Covered subset |
| Equal-length zip adapter | `ParallelIterator::zip_eq` and `ZipEq<I, J>` | `test_parallel_zip_eq_preserves_equal_length_pairs`, `test_parallel_zip_eq_rejects_length_mismatch`, and `iterator_adapter_zip_eq` benchmark rows | Covered subset |
| Borrowed reference materialization adapters | `ParallelIterator::copied` and `ParallelIterator::cloned` | `test_parallel_copied_materializes_borrowed_copy_values`, `test_parallel_cloned_materializes_borrowed_clone_values`, and `iterator_adapter_ref_copy_clone` benchmark rows | Covered subset |
| Non-Clone borrowed source map | `Vec<T>::par_iter().map(...).sum()` over borrowed non-`Clone` values | `test_non_clone_parallel_ref_iterator_maps_borrowed_values` and `iterator_adapter_non_clone_ref_map` against Rayon | Covered bounded source boundary |
| Take adapter | `ParallelIterator::take` and `Take<I>` | `test_parallel_take_keeps_prefix` and `test_parallel_take_and_skip_saturate_at_bounds` validate prefix and over-bound behavior | Covered subset |
| Skip adapter | `ParallelIterator::skip` and `Skip<I>` | `test_parallel_skip_discards_prefix` and `test_parallel_take_and_skip_saturate_at_bounds` validate suffix and over-bound behavior | Covered subset |
| Bounded any-window adapters | `ParallelIterator::take_any` and `ParallelIterator::skip_any` | `test_parallel_take_any_and_skip_any_use_bounded_window_semantics` and `iterator_adapter_take_skip_any` benchmark rows | Deterministic bounded subset |
| Predicate any-window adapters | `ParallelIterator::take_any_while`, `ParallelIterator::skip_any_while`, `TakeAnyWhile<I, F>`, and `SkipAnyWhile<I, F>` | `test_parallel_take_any_while_and_skip_any_while_use_deterministic_prefix_semantics` and the full-pass `iterator_adapter_take_skip_any_while` benchmark row | Deterministic predicate-window subset |
| Chain adapter | `ParallelIterator::chain` and `Chain<I, J>` | `test_parallel_chain_preserves_left_then_right_order`; benchmarked in `iterator_adapter_comparison` | Covered subset |
| Intersperse adapter | `ParallelIterator::intersperse` and `Intersperse<I>` | `test_parallel_intersperse_inserts_separator_between_items`, `test_parallel_intersperse_preserves_empty_and_singleton_streams`, and `iterator_adapter_intersperse` benchmark rows | Covered subset |
| Reverse adapter | `ParallelIterator::rev` and `Rev<I>` | `test_parallel_rev_reverses_logical_order`; benchmarked in `iterator_adapter_comparison` | Covered subset |
| Chunk adapter | `ParallelIterator::chunks` and `Chunks<I>` | `test_parallel_chunks_groups_full_chunks_and_tail` and `test_parallel_chunks_rejects_zero_size`; benchmarked in `iterator_adapter_comparison` | Covered subset |
| Collect adapter | `ParallelIterator::collect` with `ParallelExtend<T> for Vec<T>` | collect tests and recursion-avoidance implementation | Covered subset |
| Collect-vec-list terminal | `ParallelIterator::collect_vec_list` returning `LinkedList<Vec<T>>` | `test_parallel_collect_vec_list_moves_non_clone_values`; `iterator_adapter_collect_vec_list` against Rayon measured Moirai at 18.349-18.558 us and Rayon at 315.88-327.29 us | Covered logical-output subset |
| Partition collector | `ParallelIterator::partition` | `test_parallel_partition_preserves_relative_order`; benchmarked in `iterator_adapter_comparison` | Covered subset |
| Partition-map collector | `ParallelIterator::partition_map` with public `Either<L, R>` | `test_parallel_partition_map_splits_either_streams` and `iterator_adapter_partition_map` benchmark rows | Covered subset |
| Pair split collector | `ParallelIterator::unzip` | `test_parallel_unzip_splits_pair_streams` and `iterator_adapter_unzip` benchmark rows | Covered subset |
| Count adapter | `ParallelIterator::count` | `test_parallel_count` | Covered subset |
| Predicate adapters | `any`, `all`, `find_any`, `find_first`, `find_last`, `position_any`, `position_first`, `position_last`, `positions`, `find_map_any`, `find_map_first`, `find_map_last` | `test_parallel_any`, `test_parallel_all`, `test_parallel_find_last_returns_last_matching_value`, `test_parallel_position_terminals_return_logical_indices`, `test_parallel_positions_yields_all_matching_logical_indices`, `test_parallel_find_map_first_maps_first_present_value`, `test_parallel_find_map_any_maps_present_value`, `test_parallel_find_map_last_maps_last_present_value`, `iterator_adapter_find_map`, `iterator_adapter_position`, and `iterator_adapter_positions` benchmark rows | Covered subset |
| Side-effect adapters | `for_each`, `for_each_with`, `for_each_init`, `try_for_each`, `try_for_each_with`, `try_for_each_init` | `for_each` is implemented via `map(op).drive(NullConsumer::new())`; `test_parallel_for_each_with_uses_cloned_state`, `test_parallel_for_each_init_uses_initialized_state`, `test_parallel_try_for_each_returns_ok_after_processing_all_items`, `test_parallel_try_for_each_returns_first_error`, `test_parallel_try_for_each_with_uses_cloned_state_and_propagates_error`, `test_parallel_try_for_each_init_uses_initialized_state_and_propagates_error`, `iterator_adapter_for_each_state`, `iterator_adapter_try_for_each_state`, and `iterator_adapter_try_for_each` cover stateful and fallible execution | Covered subset |
| Reduce adapters | `reduce`, `reduce_with`, `try_reduce`, `try_reduce_with` | `Reduction<T, F>` carries the associative operation through split-combine in the vertical `parallel/consumers.rs` leaf; `try_reduce_with` is bounded by the sealed `TryStreamItem` contract for `Option<T>` and `Result<T, E>`; tests cover empty, split, fallible identity, and fallible no-identity reduction values; `iterator_adapter_try_reduce` and `iterator_adapter_try_reduce_with` benchmark fallible checksum reducers | Covered subset |
| Terminal numeric/order reducers | `sum`, `product`, `min`, `max`, `min_by`, `max_by`, `min_by_key`, `max_by_key` | `test_parallel_sum_and_product_match_standard_values`, `test_parallel_min_and_max_match_standard_values`, `test_parallel_min_max_by_use_comparator`, `test_parallel_min_max_by_key_use_key_function`, `iterator_adapter_terminal_reducers`, and `iterator_adapter_ordered_reducers` benchmark rows | Covered subset |
| Fold adapter | `fold` | preserves sequential value semantics because this API has no separate operation for combining partial accumulators | Sequential by contract |
| Full indexed producer/consumer adapters | not exposed | `Moirai::for_each_indexed` and `Moirai::map_reduce_indexed` remain the scheduler indexed execution paths; `IndexedParallelIterator` covers exact source cardinality only | Boundary documented |
| Sorting slice extension boundary | `ParallelSliceMut::{par_sort, par_sort_by, par_sort_by_key, par_sort_unstable, par_sort_unstable_by, par_sort_unstable_by_key}` | sorting unit tests, panic-safety test, and `sorting_comparison` against Rayon `ParallelSliceMut` | Slice extension boundary |

## Formal Invariants

- A covered adapter must have a public Moirai method or trait implementation and value-semantic tests or benchmark-contract coverage.
- Unsupported Rayon adapters must not be implied by documentation as drop-in compatible.
- Competitive Rayon performance claims must continue using value-checked benchmark paths, including `sorting_comparison` for the `ParallelSliceMut` boundary.

## Required Follow-Up

### ISSUE-091 [patch]: Track Rayon adapter audit

Document the current adapter surface and guard this audit with `benchmark_contracts`.

### ISSUE-092 [minor]: Replace prototype reduction consumers

Completed: `reduce` and `reduce_with` now combine both halves through the supplied associative operation. `fold` remains sequential by contract because it lacks a separate partial-accumulator combine operation.

### ISSUE-093 [minor]: Define indexed iterator boundary

Completed: `Moirai::for_each_indexed` and `Moirai::map_reduce_indexed` are documented as the scheduler indexed execution paths. `moirai-iter::parallel::IndexedParallelIterator` now covers exact source cardinality for source iterators, while the audit still rejects any claim of the full Rayon indexed producer/consumer adapter model.

### ISSUE-166 [minor]: Add bounded indexed source trait

Completed: `IndexedParallelIterator::{len, is_empty}` is implemented for exact-size source iterators with value tests and a same-run Rayon `iterator_indexed_boundary` benchmark row. Owned vector sources now use one by-value `VecParIter<T>` backed by `Vec<T>` and `split_off`, removing the duplicate non-clone vector source and the prior `Arc<Vec<T>>` allocation path.

### ISSUE-175 [minor]: Add indexed source collect-into-vec boundary

Completed: `IndexedParallelIterator::collect_into_vec` is implemented for exact-size source iterators. `VecParIter<T>` bulk-moves owned values into caller-provided spare capacity without cloning, range and borrowed-reference sources extend directly without intermediate vectors, and the non-`Clone` unit test verifies moved-value semantics. `iterator_adapter_comparison` includes `iterator_indexed_collect_into_vec` against Rayon after asserting equal output vectors and checksums.

### ISSUE-181 [minor]: Add indexed source unzip-into-vecs boundary

Completed: `IndexedParallelIterator::unzip_into_vecs` is implemented for exact-size pair source iterators. The method clears caller-provided left/right vectors, reserves exact indexed capacity, and moves each pair side into existing storage without requiring `Clone`. Unit tests cover non-`Clone` pair movement and retained allocation capacity, and `iterator_adapter_comparison` includes `iterator_indexed_unzip_into_vecs` against Rayon after asserting equal side vectors and checksums.

### ISSUE-182 [minor]: Add indexed source interleave boundaries

Completed: `IndexedParallelIterator::{interleave, interleave_shortest}` are implemented for exact-size source iterators. Both adapters are concrete generic types, move values from both inputs without requiring `Clone`, and retain Rayon's documented shortest-interleave left-tail rule. Unit tests cover non-`Clone` value movement for full interleave, left-longest shortest interleave, right-longest shortest interleave, and exact tail drops for both shortest-input shapes, and `iterator_adapter_comparison` includes `iterator_indexed_interleave` against Rayon after asserting equal vectors.

### ISSUE-183 [minor]: Add indexed source step-by boundary

Completed: `IndexedParallelIterator::step_by` is implemented for exact-size source iterators. The adapter is a concrete generic type, rejects zero steps at construction, moves retained values without requiring `Clone`, drops skipped values exactly once, reports exact stepped cardinality, and `iterator_adapter_comparison` includes `iterator_indexed_step_by` against Rayon after asserting equal vectors.

### ISSUE-185 [minor]: Add indexed source block adapter boundary

Completed: `IndexedParallelIterator::{by_exponential_blocks, by_uniform_blocks}` are implemented for exact-size source iterators as value-preserving logical-output adapters. `ExponentialBlocks<I>` and `UniformBlocks<I>` are concrete generic adapters with zero-sized policy markers; `UniformBlocks<I>` validates non-zero block sizes. Unit tests cover non-`Clone` value movement, zero-sized policy markers, and zero-size rejection, and `iterator_adapter_comparison` includes `iterator_indexed_blocks` against Rayon after asserting equal `(first, collected)` outputs. This is not a claim of Rayon's full indexed producer block-scheduling model.

### ISSUE-186 [minor]: Add collect-vec-list terminal boundary

Completed: `ParallelIterator::collect_vec_list` returns Rayon's public `LinkedList<Vec<T>>` terminal shape while preserving Moirai's logical item stream as one moved vector segment. Unit tests cover non-`Clone` value movement and empty-list behavior, and `iterator_adapter_comparison` includes `iterator_adapter_collect_vec_list` against Rayon after asserting equal flattened summaries. Segment count is not claimed as equivalent because Rayon may expose internal split segments.

### ISSUE-176 [minor]: Add equal-length zip adapter

Completed: `zip_eq` is implemented as a statically dispatched `ZipEq<I, J>` adapter that materializes both logical streams, asserts equal lengths, and then pairs values without dynamic dispatch or boxed strategy state. Unit tests cover equal-length mapped output and mismatch panic behavior, and `iterator_adapter_comparison` includes `iterator_adapter_zip_eq` against Rayon after asserting equal output vectors.

### ISSUE-177 [minor]: Add partition-map collector

Completed: `partition_map` is implemented as a terminal collector over the public `Either<L, R>` sum type. The implementation routes mapped values into caller-selected collections in one pass over the logical item stream, preserves side-local order, and avoids a Rayon runtime dependency or boxed dispatch. `iterator_adapter_comparison` includes `iterator_adapter_partition_map` against Rayon after asserting equal left and right output vectors.

### ISSUE-178 [minor]: Add fallible no-identity reduction terminal

Completed: `try_reduce_with` is implemented over a sealed `TryStreamItem` contract for `Option<T>` and `Result<T, E>`. The terminal returns `None` for empty streams, `Some(residual)` for first `None` or `Err(_)`, and `Some(success)` for fully reduced streams. `Map<I, F>` has an inherent fast path that streams mapped fallible values into the reducer without an intermediate mapped vector. `iterator_adapter_comparison` includes `iterator_adapter_try_reduce_with` against Rayon after asserting equal outputs.

### ISSUE-179 [minor]: Add positions index-stream adapter

Completed: `positions` is implemented as a logical-index stream adapter. The default `Positions<I, F>` adapter yields all matching indices, while mapped streams use a fused `MapPositions<I, MapFn, Predicate>` adapter so mapped values are consumed directly without materializing an intermediate mapped vector. Unit tests cover owned, borrowed, and mapped streams, and `iterator_adapter_comparison` includes `iterator_adapter_positions` against Rayon after asserting equal index vectors.

### ISSUE-180 [minor]: Add predicate any-window adapters

Completed: `take_any_while` and `skip_any_while` are implemented as concrete deterministic predicate-window adapters. Unit tests cover prefix and suffix early-stop value semantics, and `iterator_adapter_comparison` includes a full-pass `iterator_adapter_take_skip_any_while` row against Rayon after asserting equal output vectors. Exact threshold early-stop equality is not claimed because Rayon permits unordered early-stop behavior for these APIs.

### ISSUE-094 [minor]: Expand adapter surface by priority

Completed: added the first priority adapter group, `enumerate` and `zip`, with value-semantic tests for logical index pairing and shortest-input zip behavior. No competitive performance claim is attached to this adapter layer.

### ISSUE-101 [minor]: Add filter-map and flat-map Rayon-style adapters

Completed: `filter_map` and `flat_map` are implemented with value-semantic tests for optional retention and flattened-order semantics. ISSUE-184 adds the Rayon-named serial-inner `flat_map_iter` comparison row for the same concrete `FlatMap<I, F>` adapter.

### ISSUE-102 [minor]: Expand utility and slicing adapter groups

Completed: `take` and `skip` are implemented with value-semantic tests for prefix retention, prefix discard, and over-bound saturation.

### ISSUE-103 [minor]: Expand remaining utility adapters

Completed: `chain` and `rev` are implemented with value-semantic tests and included in `iterator_adapter_comparison`.

### ISSUE-105 [minor]: Expand remaining utility adapters

Completed: `inspect`, `panic_fuse`, `chunks`, and `partition` are implemented with value-semantic tests and included in `iterator_adapter_comparison`. `PanicFuse` carries a zero-sized policy marker so the adapter stores no runtime strategy state beyond the base iterator.

### ISSUE-116 [minor]: Add terminal reducer adapters

Completed: `sum`, `product`, `min`, and `max` are implemented as terminal `ParallelIterator` methods with value-semantic tests for non-empty and empty streams. `iterator_adapter_comparison` now includes `iterator_adapter_terminal_reducers` against Rayon after asserting equal `(sum, min, max)` results.

### ISSUE-118 [minor]: Add borrowed reference materialization adapters

Completed: `copied` and `cloned` are implemented as terminal-preserving adapters for borrowed parallel streams. Tests cover `Copy` numeric materialization and cloned `String` values, and `iterator_adapter_comparison` includes `iterator_adapter_ref_copy_clone` against Rayon after asserting equal copied/cloned collections.

### ISSUE-174 [patch]: Remove borrowed vector source Clone/static bound

Completed: `IntoParallelRefIterator<'data> for Vec<T>` now requires `T: Send + Sync + 'data` instead of `T: Clone + 'static`, keeping borrowed source iteration zero-copy for non-`Clone` values. `test_non_clone_parallel_ref_iterator_maps_borrowed_values` verifies borrowed mapping without cloning, and `iterator_adapter_non_clone_ref_map` asserts equal Moirai/Rayon checksums before timing.

### ISSUE-119 [minor]: Add pair stream unzip collector

Completed: `unzip` is implemented as a terminal pair-stream collector with value-semantic tests. `iterator_adapter_comparison` now includes `iterator_adapter_unzip` against Rayon after asserting equal left and right collections.

### ISSUE-120 [minor]: Add ordered terminal reducers

Completed: `min_by`, `max_by`, `min_by_key`, and `max_by_key` are implemented as terminal reducers with value-semantic comparator and key tests. `iterator_adapter_comparison` now includes `iterator_adapter_ordered_reducers` against Rayon after asserting equal ordered reducer outputs.

### ISSUE-121 [minor]: Add find-map predicate terminals

Completed: `find_map_first` and `find_map_any` are implemented as terminal predicate/mapping reducers with value-semantic present and missing tests. `iterator_adapter_comparison` now includes `iterator_adapter_find_map` against Rayon after asserting equal mapped results.

### ISSUE-122 [minor]: Add reverse-order predicate terminals

Completed: `find_last` and `find_map_last` are implemented as terminal reverse-order predicate reducers with value-semantic last-match and missing tests. `iterator_adapter_find_map` now includes these reverse-order terminals against Rayon after asserting equal results.

### ISSUE-123 [minor]: Add while-some optional stream adapter

Completed: `while_some` is implemented as an adapter that unwraps the present prefix of an `Option<T>` stream. Tests cover prefix truncation and first-item `None`, and `iterator_adapter_comparison` now includes the shared all-present optional unwrapping case against Rayon after asserting equal collections.

### ISSUE-124 [minor]: Add fallible side-effect terminal

Completed: `try_for_each` is implemented as a fallible terminal that applies a `Result`-returning operation and stops on the first error. Tests cover complete success with a checksum side effect and first-error propagation, and `iterator_adapter_comparison` now includes `iterator_adapter_try_for_each` against Rayon after asserting equal checksums.

### ISSUE-125 [minor]: Add fallible reduction terminal

Completed: `try_reduce` is implemented as a fallible terminal over `Result<T, E>` item streams that reduces with an identity and a `Result`-returning associative operation. Tests cover successful reduction and item-error propagation, and `iterator_adapter_comparison` now includes `iterator_adapter_try_reduce` against Rayon after asserting equal reduced checksums.

### ISSUE-126 [minor]: Add position predicate terminals

Completed: `position_first`, `position_any`, and `position_last` are implemented as logical-index predicate terminals. Tests cover first, any, last, and missing matches, and `iterator_adapter_comparison` now includes `iterator_adapter_position` against Rayon after asserting equal index tuples.

### ISSUE-127 [minor]: Add stateful side-effect terminals

Completed: `for_each_with` and `for_each_init` are implemented as stateful side-effect terminals. Tests cover cloned shared state and lazily initialized shared state, and `iterator_adapter_comparison` now includes `iterator_adapter_for_each_state` against Rayon after asserting equal checksum tuples.

### ISSUE-128 [minor]: Add fallible stateful side-effect terminals

Completed: `try_for_each_with` and `try_for_each_init` are implemented as fallible stateful side-effect terminals. Tests cover cloned shared state, initialized shared state, and error propagation, and `iterator_adapter_comparison` now includes `iterator_adapter_try_for_each_state` against Rayon after asserting equal checksum tuples.

### ISSUE-129 [minor]: Add stateful map adapters

Completed: `map_with` and `map_init` are implemented as stateful mapping adapters. Tests cover cloned state and initialized state effects on mapped values, and `iterator_adapter_comparison` now includes `iterator_adapter_map_state` against Rayon after asserting equal mapped collections and checksum tuples.

### ISSUE-140 [minor]: Add update mutation adapter

Completed: `update` is implemented as a mutating adapter that applies `Fn(&mut Item)` before yielding each item. Tests cover mutated value output, and `iterator_adapter_comparison` now includes `iterator_adapter_update` against Rayon after asserting equal updated collections.

### ISSUE-142 [minor]: Add intersperse separator adapter

Completed: `intersperse` is implemented as a separator adapter that inserts a cloned separator between adjacent logical items while preserving empty and singleton streams. Tests cover separator insertion and boundary streams, and `iterator_adapter_comparison` now includes `iterator_adapter_intersperse` against Rayon after asserting equal interspersed collections.

### ISSUE-184 [minor]: Add serial-inner Rayon flatten names

Completed: `flat_map_iter` and `flatten_iter` are implemented as Rayon-named serial-inner adapter methods over the existing concrete `FlatMap<I, F>` and `Flatten<I>` types. Tests cover a non-`Sync` serial inner iterator and serial range flattening, and `iterator_adapter_comparison` now compares Moirai against Rayon's matching `flat_map_iter` and `flatten_iter` APIs after asserting equal output vectors.

### ISSUE-143 [minor]: Add flatten nested-stream adapter

Completed: `flatten` is implemented as a nested-stream adapter over `Item: IntoIterator` with left-to-right value semantics. Tests cover nested vectors with an empty inner stream, and `iterator_adapter_comparison` now includes `iterator_adapter_flatten` against Rayon after asserting equal flattened collections.

### ISSUE-144 [minor]: Add take-any and skip-any bounded adapters

Completed: `take_any` and `skip_any` are implemented through the existing `Take<I>` and `Skip<I>` bounded-window adapters in the deterministic non-indexed boundary. Tests cover bounded window semantics, and `iterator_adapter_comparison` now includes `iterator_adapter_take_skip_any` against Rayon after asserting equal constant-output retained collections.

## Benchmark Evidence

`cargo bench -p moirai-benchmarks --bench iterator_adapter_comparison -- --quiet` produced same-run evidence for the adapter set. Focused rows for `zip_eq`, `partition_map`, `try_reduce_with`, `positions`, `take_any_while`/`skip_any_while`, and `unzip_into_vecs` were refreshed on 2026-05-29 after their implementations landed. The indexed interleave, step-by, block, collect-vec-list, plus corrected serial-inner `flat_map_iter` and `flatten_iter` rows, were added on 2026-06-01:

| Group | Moirai | Rayon | Status |
| --- | --- | --- | --- |
| `iterator_adapter_indexed_pipeline` | 35.664-35.796 us | 318.76-322.01 us | Moirai ahead |
| `iterator_adapter_filter_flat_pipeline` | 79.134-123.35 us | 393.06-405.26 us | Moirai ahead |
| `iterator_adapter_flatten` | 73.234-74.541 us | 150.08-155.19 us | Moirai ahead |
| `iterator_adapter_take_skip_any` | 26.930-27.464 us | 792.01-855.45 us | Moirai ahead |
| `iterator_adapter_take_skip_any_while` | 91.813-102.11 us | 729.10-756.49 us | Moirai ahead |
| `iterator_adapter_map_state` | 1.2630-1.3841 ms | 4.4604-21.486 ms | Moirai ahead |
| `iterator_adapter_update` | 35.583-37.854 us | 373.83-393.54 us | Moirai ahead |
| `iterator_adapter_intersperse` | 91.120-94.203 us | 418.76-433.66 us | Moirai ahead |
| `iterator_adapter_while_some` | 118.97-188.35 us | 363.93-379.84 us | Moirai ahead |
| `iterator_adapter_try_for_each` | 142.55-149.28 us | 932.60 us-1.1186 ms | Moirai ahead |
| `iterator_adapter_for_each_state` | 453.72-518.46 us | 7.0571-11.419 ms | Moirai ahead |
| `iterator_adapter_try_for_each_state` | 720.44 us-1.0202 ms | 5.6971-39.419 ms | Moirai ahead |
| `iterator_adapter_try_reduce` | 20.183-21.585 us | 75.866-79.962 us | Moirai ahead |
| `iterator_adapter_try_reduce_with` | 8.5426-8.7513 us | 64.753-66.248 us | Moirai ahead |
| `iterator_adapter_chain_rev_pipeline` | 17.993-18.389 us | 76.454-80.386 us | Moirai ahead |
| `iterator_adapter_zip_eq` | 107.34-142.67 us | 364.99-373.05 us | Moirai ahead |
| `iterator_indexed_collect_into_vec` | 54.745-75.638 us | 95.255-102.59 us | Moirai ahead |
| `iterator_indexed_unzip_into_vecs` | 256.72-273.34 us | 268.81-303.00 us | Moirai ahead |
| `iterator_indexed_interleave` | 401.13-439.28 us | 433.44-453.31 us | Moirai ahead |
| `iterator_indexed_step_by` | 24.335-25.830 us | 65.191-67.990 us | Moirai ahead |
| `iterator_indexed_blocks` | 30.128-32.300 us | 4.4301-4.5698 ms | Moirai ahead |
| `iterator_adapter_collect_vec_list` | 18.349-18.558 us | 315.88-327.29 us | Moirai ahead |
| `iterator_adapter_inspect_chunks_pipeline` | 31.061-31.810 us | 36.916-38.040 us | Moirai ahead |
| `iterator_adapter_partition_pipeline` | 29.242-30.103 us | 658.16-693.21 us | Moirai ahead |
| `iterator_adapter_partition_map` | 32.468-32.719 us | 587.36-620.15 us | Moirai ahead |
| `iterator_adapter_terminal_reducers` | 64.686-65.272 us | 218.10-226.27 us | Moirai ahead |
| `iterator_adapter_ordered_reducers` | 179.38-190.67 us | 3.3072-5.9357 ms | Moirai ahead |
| `iterator_adapter_find_map` | 77.948-85.530 us | 238.34-242.20 us | Moirai ahead |
| `iterator_adapter_position` | 33.601-43.300 us | 13.150-41.006 ms | Moirai ahead |
| `iterator_adapter_positions` | 11.248-11.339 us | 234.78-239.80 us | Moirai ahead |
| `iterator_adapter_ref_copy_clone` | 1.9997-2.0162 ms | 3.0533-3.1264 ms | Moirai ahead |
| `iterator_adapter_non_clone_ref_map` | 16.446-17.165 us | 57.328-79.918 us | Moirai ahead |
| `iterator_adapter_unzip` | 63.013-63.838 us | 648.79-671.82 us | Moirai ahead |

### ISSUE-104 [minor]: Optimize indexed and chain/rev adapter benchmarks

Completed: pure adapter source structs no longer allocate `ParallelContext` or a thread pool on construction. Window/reverse collection hooks avoid full intermediate materialization for bounded and reversed pipelines. All current `iterator_adapter_comparison` rows beat same-run Rayon references without weakening value semantics or benchmark scenarios.

### ISSUE-106 [minor]: Define sorting extension boundary

Completed: sorting is implemented through the separate `ParallelSliceMut` slice-extension boundary rather than as a `ParallelIterator` adapter. The `ParallelSliceMut` benchmark target compares stable and unstable Moirai sorting against Rayon `ParallelSliceMut` after asserting equal sorted values.

`cargo bench -p moirai-benchmarks --bench sorting_comparison -- --quiet` produced same-run evidence on 2026-05-25:

| Group | Moirai | Rayon | Status |
| --- | --- | --- | --- |
| `parallel_sorting_stable` | 76.225-78.202 us | 143.38-146.10 us | Moirai ahead |
| `parallel_sorting_unstable` | 48.838-51.041 us | 66.725-69.234 us | Moirai ahead |
