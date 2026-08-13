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
moirai-core = "0.5"
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

Full documentation: <https://docs.rs/moirai-core>

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or
[MIT](../LICENSE-MIT) at your option.
