# ADR 0009: Parallel Iterator Vertical Split

Status: Accepted

**Date**: 2026-05-24
**Context**: `moirai-iter::parallel` mixed trait surfaces, sources, adapters, consumers, and tests in one file while reduction consumers had inconsistent result carriers.

### Decision

Split the parallel iterator implementation into `traits`, `sources`, `adapters`, `consumers`, and `tests` leaves under `moirai-iter/src/parallel/`. Keep `moirai-iter/src/parallel.rs` as the public module root and re-export the same public items.

### Rationale

- Keeps each touched leaf below the 500-line structural target.
- Separates Rayon-style public traits from source iterators and consumer machinery.
- Keeps reduction state in `Reduction<T, F>` so split halves combine through the caller-provided associative function.
- Adds an empty-vector base case before chunk splitting so empty reductions terminate.

### Verification

- `cargo test -p moirai-iter parallel -- --nocapture`
- `cargo test -p moirai-iter -- --nocapture`
- `cargo test -p moirai-benchmarks --test benchmark_contracts -- --nocapture`
- `cargo clippy -p moirai-iter -- -D warnings`
