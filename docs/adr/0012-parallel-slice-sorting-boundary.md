# ADR-012: Parallel Slice Sorting Boundary

Status: Accepted

**Date**: 2026-05-25
**Context**: Rayon exposes sorting through `ParallelSliceMut`, not through `ParallelIterator`. The adapter audit therefore needed a separate slice-extension boundary instead of another non-indexed iterator adapter.

### Decision

Add `moirai_iter::parallel::ParallelSliceMut` for `[T]` with stable and unstable sort entry points: `par_sort`, `par_sort_by`, `par_sort_by_key`, `par_sort_unstable`, `par_sort_unstable_by`, and `par_sort_unstable_by_key`.

### Rationale

- Keeps sorting in the slice domain where mutation and in-place ordering are explicit.
- Preserves static dispatch through a generic extension trait instead of dynamic sorting strategies.
- Keeps stable and unstable algorithms behind one trait surface rather than type-specific API names.
- Uses repository-local value tests, stability tests, panic-safety coverage, and direct Rayon `ParallelSliceMut` benchmark rows.

### Verification

- `cargo test -p moirai-iter sorting -- --nocapture`
- `cargo test -p moirai-benchmarks --test benchmark_contracts sorting_slice_extension_is_value_semantic_and_benchmarked -- --nocapture`
- `cargo clippy -p moirai-iter -- -D warnings`
- `cargo clippy -p moirai-benchmarks --test benchmark_contracts -- -D warnings`
- `cargo bench -p moirai-benchmarks --bench sorting_comparison -- --quiet`

### Residual Risk

Stable sorting uses a temporary left-half buffer during merge. This preserves stable ordering and keeps the public API in-place, but the implementation must keep panic-safety tests active because comparator panics can occur during merge.
