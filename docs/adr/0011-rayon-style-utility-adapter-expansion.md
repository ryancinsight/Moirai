# ADR-011: Rayon-Style Utility Adapter Expansion

Status: Accepted

**Date**: 2026-05-25
**Context**: The adapter audit still classified `inspect`, `panic_fuse`, `chunks`, and `partition` as unsupported in the non-indexed `moirai-iter::parallel` surface.

### Decision

Add typed `Inspect<I, F>`, `PanicFuse<I>`, and `Chunks<I>` adapters plus a `ParallelIterator::partition` collector. Keep sorting out of this module because Rayon sorting is a slice-extension boundary, not a `ParallelIterator` adapter.

### Rationale

- Stores closures and policy state in concrete generic types with no `dyn Trait` dispatch.
- Uses a zero-sized `PanicFusePolicy` marker so panic-fuse routing stores no runtime strategy state.
- Uses a transparent `ChunkSize` newtype so zero chunk size is rejected at construction before iteration.
- Keeps side-effect and chunk implementations in adapter leaves to preserve the vertical file hierarchy and line-count target.
- Adds direct Rayon comparison rows only where Rayon exposes equivalent public paths.

### Verification

- `cargo test -p moirai-iter parallel -- --nocapture`
- `cargo test -p moirai-iter -- --nocapture`
- `cargo test -p moirai-benchmarks --test benchmark_contracts -- --nocapture`
- `cargo clippy -p moirai-iter -- -D warnings`
- `cargo clippy -p moirai-benchmarks --test benchmark_contracts -- -D warnings`
- `cargo bench -p moirai-benchmarks --bench iterator_adapter_comparison -- --quiet`

### Residual Risk

`PanicFuse` preserves value and panic propagation in the current non-indexed adapter layer. If this adapter layer later executes sibling branches concurrently, panic-fuse must gain a shared cancellation flag in the consumer path before claiming Rayon-equivalent early-stop behavior.
