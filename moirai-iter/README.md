# moirai-iter

[![crates.io](https://img.shields.io/crates/v/moirai-iter.svg)](https://crates.io/crates/moirai-iter)
[![docs.rs](https://docs.rs/moirai-iter/badge.svg)](https://docs.rs/moirai-iter)

Iterator combinators for the [Moirai](https://github.com/ryancinsight/Moirai)
concurrency library. One iterator type abstracts over three execution contexts:

- **Parallel** — CPU-bound work across worker threads with work stealing.
- **Async** — I/O-bound work through async/await.
- **Hybrid** — mixed workloads, the default for `moirai_iter`.

`map` and `filter` are synchronous adapters; `collect`, `reduce`, and `for_each`
are `async` and must be awaited. Async-closure variants (`map_async`,
`filter_async`, `for_each_async`) take a future-returning closure.

```toml
[dependencies]
moirai-iter = "0.5"
```

```rust
use moirai_iter::{moirai_iter, moirai_iter_parallel};

async fn example(data: Vec<u64>) {
    // Hybrid context.
    moirai_iter(data.clone())
        .map(|x| x * x)
        .filter(|&x| x > 10)
        .for_each(|x| println!("{x}"))
        .await;

    // Parallel context.
    let squares: Vec<u64> = moirai_iter_parallel(data).map(|x| x * x).collect().await;
    println!("{squares:?}");
}
```

Also exports the Rayon-style `ParallelIterator` / `IndexedParallelIterator`
surface, `par_range` / `async_range`, and the `ExecutionContext` /
`ExecutionStrategy` types.

Full documentation: <https://docs.rs/moirai-iter>

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or
[MIT](../LICENSE-MIT) at your option.
