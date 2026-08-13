# moirai-executor

[![crates.io](https://img.shields.io/crates/v/moirai-executor.svg)](https://crates.io/crates/moirai-executor)
[![docs.rs](https://docs.rs/moirai-executor/badge.svg)](https://docs.rs/moirai-executor)

The hybrid executor for the [Moirai](https://github.com/ryancinsight/Moirai)
concurrency library. Synchronous, asynchronous, and blocking work run on one
unified scheduler facade: sync and async-ready jobs use the compute
work-stealing pool, while potentially blocking work uses a lazily initialized,
bounded lane isolated from that pool.

- Static work-class routing through zero-sized markers.
- Per-worker Chase-Lev deques indexed by `moirai_core::Priority::index`.
- `SchedulerScope` for completion-only borrowing of non-`'static` data.

```toml
[dependencies]
moirai-executor = "0.5"
```

```rust
// Moirai-owned synchronous wait primitive: bridges a future into a sync
// boundary without constructing the process-wide scheduler.
let value = moirai_executor::block_on(async { 21 * 2 });
assert_eq!(value, 42);
```

Most users depend on the [`moirai-runtime`](https://crates.io/crates/moirai-runtime)
facade instead.

Full documentation: <https://docs.rs/moirai-executor>

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or
[MIT](../LICENSE-MIT) at your option.
