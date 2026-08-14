# moirai-runtime

Facade crate for **Moirai**, a hybrid concurrency runtime for Rust that blends
asynchronous and parallel execution behind one API. The package is published as
`moirai-runtime`; the library is imported as `moirai`.

```toml
[dependencies]
moirai-runtime = "0.1"
```

```rust
use moirai::Moirai;

let runtime = Moirai::new().expect("default runtime configuration is valid");
let handle = runtime.spawn_fn(|| (1..=10).sum::<u32>());
let total = handle
    .join()
    .expect("task completed")
    .expect("task did not error");
assert_eq!(total, 55);
```

`join()` returns a nested result: the outer reports whether the task completed,
the inner whether it errored. The runnable form of this is
[`examples/basic_usage.rs`](../examples/basic_usage.rs).

## What this crate is

`moirai-runtime` re-exports the workspace's execution surface — the executor,
schedulers, channels, synchronization primitives, and iterator combinators —
so consumers depend on one crate rather than assembling `moirai-core`,
`moirai-executor`, `moirai-scheduler`, and the rest by hand. Selection between
sequential, parallel, and async execution is a compile-time choice through the
`ExecutionPolicy` seam, so the abstraction monomorphizes away rather than
dispatching at runtime.

Optional capability — IPC, metrics, and distributed transport — is behind
cargo features so a consumer only compiles what it uses.

## Documentation

Full API documentation, including the execution-policy seam and the
work-stealing scheduler's contracts, is on
[docs.rs](https://docs.rs/moirai-runtime). The workspace
[README](https://github.com/ryancinsight/moirai) covers the crate layout and
the design principles the runtime is built on.

## License

MIT OR Apache-2.0, at your option.
