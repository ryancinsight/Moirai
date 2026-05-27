# Rayon Adapter Surface Audit

## Scope

This audit covers the `moirai-iter::parallel` public adapter surface against the Rayon `ParallelIterator` style API. It is separate from `docs/rayon_tokio_gap_audit.md`, which covers scheduler, result-handle, scoped work, indexed reduction, transport, and runtime dependency boundaries.

The audit classifies only repository-local evidence. It does not claim drop-in Rayon parity unless the active source exposes the adapter and tests or benchmarks verify value semantics.

## Verdict

Moirai does not currently provide full Rayon adapter parity. The supported surface is a focused subset:

- `IntoParallelIterator` for `Vec<T>` and `Range<usize>`.
- `IntoParallelRefIterator` for `Vec<T>`.
- `ParallelIterator::map`, `map_with`, `map_init`, `update`, `filter`, `inspect`, `panic_fuse`, `filter_map`, `while_some`, `flat_map`, `flatten`, `enumerate`, `zip`, `copied`, `cloned`, `take`, `skip`, `take_any`, `skip_any`, `chain`, `intersperse`, `rev`, `chunks`, `partition`, `unzip`, `collect`, `count`, `any`, `all`, `find_any`, `find_first`, `find_last`, `position_any`, `position_first`, `position_last`, `find_map_any`, `find_map_first`, `find_map_last`, `for_each`, `for_each_with`, `for_each_init`, `try_for_each`, `try_for_each_with`, `try_for_each_init`, `reduce`, `reduce_with`, `try_reduce`, `sum`, `product`, `min`, `max`, `min_by`, `max_by`, `min_by_key`, `max_by_key`, and `fold`.
- `ParallelExtend<T>` for `Vec<T>`.
- `ParallelSliceMut` for the slice extension sorting boundary.

The active competitive Rayon comparison remains `Moirai::map_reduce_indexed` versus fixed-pool Rayon `into_par_iter().map(...).sum()`. The `moirai-iter::parallel` trait surface is not the active performance comparison path.

Indexed scheduler execution is exposed only through `Moirai::for_each_indexed` and `Moirai::map_reduce_indexed`. `moirai-iter::parallel` remains a Rayon-style non-indexed adapter subset and must not claim full Rayon compatibility or an `IndexedParallelIterator` boundary.

## Adapter Matrix

| Rayon-style capability | Moirai surface | Evidence | Status |
| --- | --- | --- | --- |
| Owned vector parallel iteration | `impl IntoParallelIterator for Vec<T>` | `moirai-iter/src/parallel/sources.rs`; unit tests for map/filter/reduce | Covered subset |
| Range parallel iteration | `impl IntoParallelIterator for Range<usize>` | `par_range`; `test_range_parallel` | Covered subset |
| Borrowed vector iteration | `impl IntoParallelRefIterator for Vec<T>` | `VecRefParIter<'data, T>` and `RefVecParIter<'a, T>` | Covered subset |
| Map adapters | `ParallelIterator::map`, `map_with`, `map_init`, `Map<I, F>`, `MapWith<I, T, F>`, and `MapInit<I, Init, F>` | `test_parallel_map`, `test_parallel_map_with_uses_cloned_state`, `test_parallel_map_init_uses_initialized_state`, and `iterator_adapter_map_state` benchmark rows | Covered subset |
| Mutation adapter | `ParallelIterator::update` and `Update<I, F>` | `test_parallel_update_mutates_items_before_yielding` and `iterator_adapter_update` benchmark rows | Covered subset |
| Filter adapter | `ParallelIterator::filter` and `Filter<I, F>` | `test_parallel_filter` | Covered subset |
| Inspect adapter | `ParallelIterator::inspect` and `Inspect<I, F>` | `test_parallel_inspect_observes_items_without_changing_output` validates observation without changing output | Covered subset |
| Panic-fuse adapter | `ParallelIterator::panic_fuse` and `PanicFuse<I>` | `test_parallel_panic_fuse_preserves_values`, `test_parallel_panic_fuse_propagates_panic`, and a zero-sized `PanicFusePolicy` test | Covered subset |
| Filter-map adapter | `ParallelIterator::filter_map` and `FilterMap<I, F>` | `test_parallel_filter_map_retains_present_values` validates optional retention semantics | Covered subset |
| While-some adapter | `ParallelIterator::while_some` and `WhileSome<I>` | `test_parallel_while_some_unwraps_present_prefix`, `test_parallel_while_some_empty_when_first_is_none`, and `iterator_adapter_while_some` benchmark rows | Covered subset |
| Flat-map adapter | `ParallelIterator::flat_map` and `FlatMap<I, F>` | `test_parallel_flat_map_preserves_flattened_order` validates flattened output order | Covered subset |
| Flatten adapter | `ParallelIterator::flatten` and `Flatten<I>` | `test_parallel_flatten_preserves_nested_order` and `iterator_adapter_flatten` benchmark rows | Covered subset |
| Enumerate adapter | `ParallelIterator::enumerate` and `Enumerate<I>` | `test_parallel_enumerate_pairs_logical_indices` validates zero-based logical positions | Covered subset |
| Zip adapter | `ParallelIterator::zip` and `Zip<I, J>` | `test_parallel_zip_stops_at_shorter_input` validates shortest-input semantics | Covered subset |
| Borrowed reference materialization adapters | `ParallelIterator::copied` and `ParallelIterator::cloned` | `test_parallel_copied_materializes_borrowed_copy_values`, `test_parallel_cloned_materializes_borrowed_clone_values`, and `iterator_adapter_ref_copy_clone` benchmark rows | Covered subset |
| Take adapter | `ParallelIterator::take` and `Take<I>` | `test_parallel_take_keeps_prefix` and `test_parallel_take_and_skip_saturate_at_bounds` validate prefix and over-bound behavior | Covered subset |
| Skip adapter | `ParallelIterator::skip` and `Skip<I>` | `test_parallel_skip_discards_prefix` and `test_parallel_take_and_skip_saturate_at_bounds` validate suffix and over-bound behavior | Covered subset |
| Bounded any-window adapters | `ParallelIterator::take_any` and `ParallelIterator::skip_any` | `test_parallel_take_any_and_skip_any_use_bounded_window_semantics` and `iterator_adapter_take_skip_any` benchmark rows | Deterministic bounded subset |
| Chain adapter | `ParallelIterator::chain` and `Chain<I, J>` | `test_parallel_chain_preserves_left_then_right_order`; benchmarked in `iterator_adapter_comparison` | Covered subset |
| Intersperse adapter | `ParallelIterator::intersperse` and `Intersperse<I>` | `test_parallel_intersperse_inserts_separator_between_items`, `test_parallel_intersperse_preserves_empty_and_singleton_streams`, and `iterator_adapter_intersperse` benchmark rows | Covered subset |
| Reverse adapter | `ParallelIterator::rev` and `Rev<I>` | `test_parallel_rev_reverses_logical_order`; benchmarked in `iterator_adapter_comparison` | Covered subset |
| Chunk adapter | `ParallelIterator::chunks` and `Chunks<I>` | `test_parallel_chunks_groups_full_chunks_and_tail` and `test_parallel_chunks_rejects_zero_size`; benchmarked in `iterator_adapter_comparison` | Covered subset |
| Collect adapter | `ParallelIterator::collect` with `ParallelExtend<T> for Vec<T>` | collect tests and recursion-avoidance implementation | Covered subset |
| Partition collector | `ParallelIterator::partition` | `test_parallel_partition_preserves_relative_order`; benchmarked in `iterator_adapter_comparison` | Covered subset |
| Pair split collector | `ParallelIterator::unzip` | `test_parallel_unzip_splits_pair_streams` and `iterator_adapter_unzip` benchmark rows | Covered subset |
| Count adapter | `ParallelIterator::count` | `test_parallel_count` | Covered subset |
| Predicate adapters | `any`, `all`, `find_any`, `find_first`, `find_last`, `position_any`, `position_first`, `position_last`, `find_map_any`, `find_map_first`, `find_map_last` | `test_parallel_any`, `test_parallel_all`, `test_parallel_find_last_returns_last_matching_value`, `test_parallel_position_terminals_return_logical_indices`, `test_parallel_find_map_first_maps_first_present_value`, `test_parallel_find_map_any_maps_present_value`, `test_parallel_find_map_last_maps_last_present_value`, `iterator_adapter_find_map`, and `iterator_adapter_position` benchmark rows | Covered subset |
| Side-effect adapters | `for_each`, `for_each_with`, `for_each_init`, `try_for_each`, `try_for_each_with`, `try_for_each_init` | `for_each` is implemented via `map(op).drive(NullConsumer::new())`; `test_parallel_for_each_with_uses_cloned_state`, `test_parallel_for_each_init_uses_initialized_state`, `test_parallel_try_for_each_returns_ok_after_processing_all_items`, `test_parallel_try_for_each_returns_first_error`, `test_parallel_try_for_each_with_uses_cloned_state_and_propagates_error`, `test_parallel_try_for_each_init_uses_initialized_state_and_propagates_error`, `iterator_adapter_for_each_state`, `iterator_adapter_try_for_each_state`, and `iterator_adapter_try_for_each` cover stateful and fallible execution | Covered subset |
| Reduce adapters | `reduce`, `reduce_with`, `try_reduce` | `Reduction<T, F>` carries the associative operation through split-combine in the vertical `parallel/consumers.rs` leaf; tests cover empty, split, and fallible reduction values; `iterator_adapter_try_reduce` benchmarks the fallible checksum reducer | Covered subset |
| Terminal numeric/order reducers | `sum`, `product`, `min`, `max`, `min_by`, `max_by`, `min_by_key`, `max_by_key` | `test_parallel_sum_and_product_match_standard_values`, `test_parallel_min_and_max_match_standard_values`, `test_parallel_min_max_by_use_comparator`, `test_parallel_min_max_by_key_use_key_function`, `iterator_adapter_terminal_reducers`, and `iterator_adapter_ordered_reducers` benchmark rows | Covered subset |
| Fold adapter | `fold` | preserves sequential value semantics because this API has no separate operation for combining partial accumulators | Sequential by contract |
| Indexed parallel iterator trait | runtime facade only | `Moirai::for_each_indexed` and `Moirai::map_reduce_indexed` are the sole indexed public scheduler paths; no `IndexedParallelIterator` trait or indexed producer surface exists in `moirai-iter::parallel` | Boundary documented |
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

Completed: `Moirai::for_each_indexed` and `Moirai::map_reduce_indexed` are documented as the sole indexed public scheduler paths. `moirai-iter::parallel` is documented as a Rayon-style non-indexed adapter subset and no longer claims broad Rayon compatibility.

### ISSUE-094 [minor]: Expand adapter surface by priority

Completed: added the first priority adapter group, `enumerate` and `zip`, with value-semantic tests for logical index pairing and shortest-input zip behavior. No competitive performance claim is attached to this adapter layer.

### ISSUE-101 [minor]: Add filter-map and flat-map Rayon-style adapters

Completed: `filter_map` and `flat_map` are implemented with value-semantic tests for optional retention and flattened-order semantics. No competitive performance claim is attached to this adapter layer.

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

### ISSUE-143 [minor]: Add flatten nested-stream adapter

Completed: `flatten` is implemented as a nested-stream adapter over `Item: IntoIterator` with left-to-right value semantics. Tests cover nested vectors with an empty inner stream, and `iterator_adapter_comparison` now includes `iterator_adapter_flatten` against Rayon after asserting equal flattened collections.

### ISSUE-144 [minor]: Add take-any and skip-any bounded adapters

Completed: `take_any` and `skip_any` are implemented through the existing `Take<I>` and `Skip<I>` bounded-window adapters in the deterministic non-indexed boundary. Tests cover bounded window semantics, and `iterator_adapter_comparison` now includes `iterator_adapter_take_skip_any` against Rayon after asserting equal constant-output retained collections.

## Benchmark Evidence

`cargo bench -p moirai-benchmarks --bench iterator_adapter_comparison -- --quiet` produced same-run evidence on 2026-05-25 after adding the utility adapter group and removing avoidable partition and inspect allocation overhead:

| Group | Moirai | Rayon | Status |
| --- | --- | --- | --- |
| `iterator_adapter_indexed_pipeline` | 35.664-35.796 us | 318.76-322.01 us | Moirai ahead |
| `iterator_adapter_filter_flat_pipeline` | 22.001-22.292 us | 2.9053-3.0355 ms | Moirai ahead |
| `iterator_adapter_flatten` | 108.93-137.47 us | 1.2705-1.3079 ms | Moirai ahead |
| `iterator_adapter_take_skip_any` | 26.930-27.464 us | 792.01-855.45 us | Moirai ahead |
| `iterator_adapter_map_state` | 1.2630-1.3841 ms | 4.4604-21.486 ms | Moirai ahead |
| `iterator_adapter_update` | 35.583-37.854 us | 373.83-393.54 us | Moirai ahead |
| `iterator_adapter_intersperse` | 91.120-94.203 us | 418.76-433.66 us | Moirai ahead |
| `iterator_adapter_while_some` | 118.97-188.35 us | 363.93-379.84 us | Moirai ahead |
| `iterator_adapter_try_for_each` | 142.55-149.28 us | 932.60 us-1.1186 ms | Moirai ahead |
| `iterator_adapter_for_each_state` | 453.72-518.46 us | 7.0571-11.419 ms | Moirai ahead |
| `iterator_adapter_try_for_each_state` | 720.44 us-1.0202 ms | 5.6971-39.419 ms | Moirai ahead |
| `iterator_adapter_try_reduce` | 20.183-21.585 us | 75.866-79.962 us | Moirai ahead |
| `iterator_adapter_chain_rev_pipeline` | 17.993-18.389 us | 76.454-80.386 us | Moirai ahead |
| `iterator_adapter_inspect_chunks_pipeline` | 31.061-31.810 us | 36.916-38.040 us | Moirai ahead |
| `iterator_adapter_partition_pipeline` | 29.242-30.103 us | 658.16-693.21 us | Moirai ahead |
| `iterator_adapter_terminal_reducers` | 64.686-65.272 us | 218.10-226.27 us | Moirai ahead |
| `iterator_adapter_ordered_reducers` | 179.38-190.67 us | 3.3072-5.9357 ms | Moirai ahead |
| `iterator_adapter_find_map` | 77.948-85.530 us | 238.34-242.20 us | Moirai ahead |
| `iterator_adapter_position` | 33.601-43.300 us | 13.150-41.006 ms | Moirai ahead |
| `iterator_adapter_ref_copy_clone` | 1.9997-2.0162 ms | 3.0533-3.1264 ms | Moirai ahead |
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
