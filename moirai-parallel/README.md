# moirai-parallel

[![crates.io](https://img.shields.io/crates/v/moirai-parallel.svg)](https://crates.io/crates/moirai-parallel)
[![docs.rs](https://docs.rs/moirai-parallel/badge.svg)](https://docs.rs/moirai-parallel)

Synchronous data-parallel primitives for the
[Moirai](https://github.com/ryancinsight/Moirai) runtime — the rayon-style
surface. This is the **parallel** domain (throughput over data), distinct from
the concurrent domain in `moirai-async`. Every operation is fully synchronous
(no `async`, no `.await`), so it is safe inside pure compute kernels without
async contagion, and operates on borrowed slices with in-place mutation.

Strategy is a zero-sized `ExecutionPolicy` type — `Sequential`, `Parallel`, or
`Adaptive` — chosen at compile time, so every call monomorphizes with no dynamic
dispatch. `.par()` / `.par_mut()` return `Adaptive` handles, which parallelize at
or above `ADAPTIVE_PARALLEL_THRESHOLD` and run sequentially below it.

```toml
[dependencies]
moirai-parallel = "0.5"
```

```rust
use moirai_parallel::{ParallelSlice, ParallelSliceMut};

let v: Vec<u64> = (0..1000).collect();
let sum = v.par().map_reduce(0, |&x| x, |a, b| a + b); // auto-routes
let mut m = v.clone();
m.par_mut().for_each(|x| *x += 1);
```

The `*_with::<P>` free functions pin the policy explicitly
(`for_each_with::<Sequential>(&data, f)`) for the rare case that needs forced
sequential or forced parallel execution.

These operations run on the same unified hybrid scheduler as async work
(`moirai_executor::global`), not a separate pool.

Full documentation: <https://docs.rs/moirai-parallel>

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or
[MIT](../LICENSE-MIT) at your option.
