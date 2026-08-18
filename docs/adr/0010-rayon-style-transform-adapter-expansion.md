# ADR-010: Rayon-Style Transform Adapter Expansion

Status: Accepted

**Date**: 2026-05-24
**Context**: The Rayon adapter audit listed `filter_map` and `flat_map` as unsupported after `enumerate` and `zip` were already present with value tests.

### Decision

Add `ParallelIterator::filter_map` through `FilterMap<I, F>` and `ParallelIterator::flat_map` through `FlatMap<I, F>`. The adapters store concrete closure types and monomorphize through the existing `ParallelIterator` trait, preserving the non-indexed adapter boundary.

### Rationale

- Closes the next Rayon-style transform adapter gap without claiming full Rayon parity.
- Keeps adapter variation in generic types instead of dynamic callbacks.
- Uses value-semantic tests for optional retention and flattened output order.
- Leaves indexed execution on `Moirai::for_each_indexed` and `Moirai::map_reduce_indexed`.

### Verification

- `cargo test -p moirai-iter parallel -- --nocapture`
- `cargo test -p moirai-benchmarks --test benchmark_contracts -- --nocapture`
- `cargo clippy -p moirai-iter -- -D warnings`
