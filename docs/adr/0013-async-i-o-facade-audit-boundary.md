# ADR-013: Async I/O Facade Audit Boundary

Status: Accepted

**Date**: 2026-05-25
**Context**: The Tokio gap audit needed to separate covered Moirai-owned file/network facade behavior from unsupported Tokio reactor-native I/O drop-in compatibility.

### Decision

Treat `moirai-async::fs` as a Moirai-owned file facade with value-semantic tests and a `tokio::fs::read` benchmark row. Treat `moirai-async::net` as a Moirai-owned socket facade with TCP and UDP loopback value tests. PAL TCP types may register wakers with an active `IoReactor`; without an active reactor they must self-wake before returning `Pending` so local executors do not deadlock on delayed readiness. PAL reactor-spawned tasks must publish completion to their `TaskHandle` through per-task state. PAL reactor platform dispatch must use the compile-target `PlatformReactor`, and queued reactor futures must use bounded inline storage plus monomorphized poll/drop dispatch instead of `dyn Future`. Moirai comparison benchmark rows must use `Moirai::block_on`, not an external futures executor. Do not claim Tokio reactor-native file or network drop-in compatibility until PAL file readiness, Tokio trait compatibility, cancellation, and backpressure contracts are specified.

### Rationale

- Keeps the comparison evidence tied to implemented APIs instead of broad ecosystem claims.
- Removes obsolete async-file placeholder future machinery that provided no authoritative execution path.
- Adds network payload and counter assertions without measuring std-socket immediate-poll facades against Tokio's reactor as if they were equivalent.
- Adds a PAL TCP/UDP no-active-reactor wake contract so cooperative socket facades remain progress-safe under `block_on`.
- Adds PAL reactor task-handle completion state so spawned ready tasks cannot leave awaiting handles pending forever.
- Removes PAL reactor dynamic dispatch at the platform and queued-future boundaries where the concrete implementation is known by compile target or monomorphized at spawn, and stores fitting futures inline under a static size/alignment contract.
- Keeps Moirai benchmark rows on the Moirai runtime surface instead of measuring through `futures::executor::block_on`.
- Replaces Linux epoll's no-op wake placeholder with an internal `eventfd` wake path.
- Preserves the production dependency boundary: Tokio remains a benchmark/reference dependency only.

### Verification

- `cargo test -p moirai-async net -- --nocapture`
- `cargo test -p moirai-async fs -- --nocapture`
- `cargo test -p moirai-pal -- --nocapture`
- `cargo test -p moirai-benchmarks --test benchmark_contracts -- --nocapture`
- `cargo bench -p moirai-benchmarks --bench async_fs_comparison -- --quiet`
- `cargo bench -p moirai-benchmarks --bench async_udp_comparison -- --quiet`

### Residual Risk

Tokio reactor-native I/O drop-in compatibility remains deferred. The next I/O increment must define PAL file readiness ownership, Tokio trait compatibility, cancellation semantics, and bounded resource behavior before adding Tokio network/file compatibility benchmarks.
