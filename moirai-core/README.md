# moirai-core

[![crates.io](https://img.shields.io/crates/v/moirai-core.svg)](https://crates.io/crates/moirai-core)
[![docs.rs](https://docs.rs/moirai-core/badge.svg)](https://docs.rs/moirai-core)

Core abstractions for the [Moirai](https://github.com/ryancinsight/Moirai)
concurrency library: the task, executor, and scheduler traits every other Moirai
crate builds on, plus the channel, communication, memory-pool, and coroutine
primitives they share.

Modules: `task`, `executor`, `scheduler`, `error`, `channel`, `communication`,
`memory`, `pool`, `coroutine`, `ipc`, `platform`.

```toml
[dependencies]
moirai-core = "0.12"
```

```rust
use moirai_core::channel::spsc;

// Bounded lock-free single-producer/single-consumer ring.
let (tx, rx) = spsc::<u32>(1024);
tx.send(42).unwrap();
assert_eq!(rx.recv().unwrap(), 42);
```

Most users depend on the [`moirai-runtime`](https://crates.io/crates/moirai-runtime)
facade instead, which re-exports the parts of this crate that make up the public
runtime surface.

## `no_std`

`moirai-core` supports targets with `core` and `alloc` when default features are
disabled:

```toml
[dependencies]
moirai-core = { version = "0.12", default-features = false }
```

This configuration retains task values and combinators, scheduler identifiers,
executor configuration vocabulary, errors, allocation and object-pool
algorithms, and other APIs whose inputs contain all required state. The final
binary supplies the allocator used by allocation-backed APIs.

Capabilities supplied by an operating system are available only with `std`:
channels and communication, task result handles and executor spawning or
lifecycle traits, automatic processor-topology selection, monotonic-clock task
statistics, and the clock-backed `TaskWrapper`. `ExecutorConfig::default` is
also `std`-only because it derives worker counts from host topology; a `no_std`
consumer constructs the public fields from its platform's explicit values.
Moirai does not fabricate clocks, processor counts, threads, locks, or result
completion when those capabilities are absent.

Full documentation: <https://docs.rs/moirai-core>

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or
[MIT](../LICENSE-MIT) at your option.
