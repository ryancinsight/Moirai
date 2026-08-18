# ADR-031: Timer Wheel Lazy Cancellation

Status: Accepted

**Date**: 2026-05-24
**Context**: `TimerWheel::cancel` ignored the timer id and always returned `false`, so the timer-wheel cancellation API did not perform the requested state transition.

### Decision

Move `TimerWheel` into `moirai-async/src/timer/wheel.rs` and implement cancellation with a lazy `HashSet<u64>` of canceled timer ids. Keep scheduled timer entries in a `BinaryHeap`, skip canceled entries during expiration polling, and expose `TimerWheel` through the existing `timer` module boundary.

### Rationale

- Preserves heap-based deadline ordering without arbitrary heap removal on the scheduling path.
- Makes cancellation value-sensitive: first cancel succeeds, duplicate or absent cancels fail.
- Prevents canceled timer wakers from firing when their heap entry expires.
- Keeps `timer.rs` below the 500-line structural target through a cohesive timer-wheel leaf module.

### Verification

- `cargo test -p moirai-async timer_wheel -- --nocapture`
- `cargo test -p moirai-benchmarks --test benchmark_contracts -- --nocapture`
- `cargo bench -p moirai-benchmarks --bench example_pattern_comparison -- example_tokio_fanout`
- `cargo bench -p moirai-benchmarks --bench thread_schedule_comparison -- mixed_unified_schedule`

### Residual Risk

Lazy cancellation retains canceled entries until their deadline reaches the heap root. This preserves low scheduling-path cost but means long-deadline canceled timers can occupy heap storage until expiration; a future compaction policy should be benchmarked before adoption.
