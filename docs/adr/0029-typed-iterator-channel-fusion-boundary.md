# ADR-029: Typed Iterator Channel Fusion Boundary

Status: Accepted

**Date**: 2026-05-24
**Context**: `moirai-iter::channel_fusion` had boxed `FusableChannel` split/merge endpoints, a placeholder hash strategy, and a pipeline builder that returned success without executing stages.

### Decision

`ChannelSplitter<T, I, C>` and `ChannelMerger<T, C>` store concrete channel values in `Vec<C>` and dispatch through `C: FusableChannel<T>`. The incomplete `SplitStrategy::Hash` and `Pipeline` surface are removed instead of preserved through compatibility wrappers.

### Rationale

- Preserves one monomorphized channel type per split/merge instance and removes vtable dispatch from iterator channel routing.
- Keeps heterogeneous channel graphs explicit through caller-defined enum channel types rather than implicit boxed trait objects.
- Removes a placeholder hash branch that violated value-sensitive distribution semantics.
- Removes a non-executing pipeline API that reported success without performing work.

### Verification

- `cargo test -p moirai-iter -- --nocapture`
- `cargo clippy -p moirai-iter -- -D warnings`
- `cargo test -p moirai-benchmarks --test benchmark_contracts -- --nocapture`
- `cargo clippy -p moirai-benchmarks --test benchmark_contracts -- -D warnings`
- `cargo bench -p moirai-benchmarks --bench channel_matrix -- tokio_mpsc/p1_c1`
- `cargo bench -p moirai-benchmarks --bench channel_matrix -- moirai_mpmc/p1_c1`

### Residual Risk

`channel_matrix` keeps Moirai ahead of the same-run Tokio p1/c1 channel row, but Criterion reports a local baseline regression on the Moirai row. The next channel increment should isolate bounded-channel transport variance before changing the core MPMC implementation.
