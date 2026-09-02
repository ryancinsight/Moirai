# ADR 0030: Iterator Base And Streaming Monomorphization

Status: Accepted

**Date**: 2026-05-24
**Context**: `moirai-iter::base` exposed an unused boxed-future execution trait while `StreamingIter` boxed its producer and shifted buffered items.

### Decision

Remove the unused `base::ExecutionBase` trait and keep `execution::ExecutionBase` as the active context trait. Change `StreamingIter<T, F>` to store a concrete `F: FnMut() -> Option<T>` producer and a `VecDeque<T>` FIFO buffer. Split the touched iterator operations tree into streaming, stateful, and test leaves.

### Rationale

- Removes `Pin<Box<dyn Future<...>>>` from the iterator base surface.
- Preserves static dispatch for streaming producer calls through monomorphization.
- Replaces O(n) front removal with O(1) FIFO buffer operations.
- Keeps `iter_ops.rs` below the 500-line structural target without changing public adapter names.
- Avoids compatibility wrappers because this is a pre-1.0 breaking cleanup.

### Verification

- `cargo test -p moirai-iter -- --nocapture`
- `cargo clippy -p moirai-iter -- -D warnings`
- `cargo test -p moirai-benchmarks --test benchmark_contracts -- --nocapture`
- `cargo clippy -p moirai-benchmarks --test benchmark_contracts -- -D warnings`
- `cargo bench -p moirai-benchmarks --bench thread_schedule_comparison -- mixed_unified_schedule`
