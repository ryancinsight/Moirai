# ADR-032: Blocking Result Wait Spin Budget

Status: Accepted

**Date**: 2026-05-24
**Context**: `TaskHandle::join` uses the sealed zero-sized `BlockingResultWait` policy to probe a pending result slot before entering the single-waiter park fallback. Caller-side attribution measured the previous 100-spin pending miss at 1.1886-1.4520 us.

### Decision

Set `MAX_SPIN_ATTEMPTS` to 64 for the blocking result-wait policy. Preserve the direct first READY-to-TAKEN CAS for already-ready handles, the relaxed-load gated pending probes, and the existing `WAITING` plus `thread::park` fallback.

### Rationale

- Keeps wait-policy dispatch static through `TaskResultSlot::wait::<BlockingResultWait>`.
- Preserves a zero-sized policy type and associated-const budget with no runtime storage.
- Reduces pending CPU spin work before the blocking fallback.
- Avoids result-slot layout changes, allocation, dynamic dispatch, result-slot pooling, and scheduler-side atomics.

### Verification

- `cargo test -p moirai-core --features result-diagnostics task:: -- --nocapture`
- `cargo test -p moirai-benchmarks --features scheduler-diagnostics,result-diagnostics --test benchmark_contracts -- --nocapture`
- `cargo clippy -p moirai-core --features result-diagnostics -- -D warnings`
- `cargo bench -p moirai-benchmarks --features scheduler-diagnostics,result-diagnostics --bench result_handle_diagnostics -- "result_handle_diagnostics/(direct_result_slot_(ready_take|spin_miss|register_waiter|complete_waiting)|direct_scheduler_join_fast_spin_(quiescent|pending)|moirai_spawn_join_ready|direct_scheduler_result_slot|direct_scheduler_submit_join)"`
- `cargo bench -p moirai-benchmarks --bench performance_benchmarks -- task_scheduling_overhead`
- `cargo bench -p moirai-benchmarks --bench public_result_handle_comparison -- "public_result_handle_ready/(moirai_spawn_join_ready|tokio_spawn_join_ready|moirai_spawn_join_captured_ready|tokio_spawn_join_captured_ready|moirai_spawn_join_oversized_captured_ready|tokio_spawn_join_oversized_captured_ready|moirai_spawn_async_wake_once|tokio_spawn_async_wake_once|moirai_scope_single_ready|rayon_scope_single_ready)"`

### Residual Risk

The 64-spin budget keeps Moirai ahead of same-run Tokio/Rayon public rows and leaves `task_scheduling_overhead` statistically unchanged, but captured, wake-once, oversized, and scope rows still show local Criterion baseline regressions. Further work should split scheduler result-publication variance before changing the wait budget again.
