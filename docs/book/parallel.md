# Parallel Execution

Moirai's parallel layer routes CPU-bound closures through a work-stealing
thread pool, dispatching them without async overhead.

## Spawning Parallel Tasks

```rust,ignore
let handle = runtime.spawn_fn(move || {
    // CPU-bound work: no await points
    (0..1_000_000_u64).sum::<u64>()
});

let sum = handle.join()??;
```

`spawn_fn` accepts `FnOnce() -> T` where `T: Send + 'static`. The returned
`TaskHandle<T>` can be joined synchronously with `.join()`.

## `TaskBuilder` and `Priority`

Tasks can be assigned a `Priority` level that influences scheduling order:

| Variant | Value | Use Case |
|---------|-------|---------|
| `Low` | 0 | Background preprocessing |
| `Normal` | 1 | Typical computation (default) |
| `High` | 2 | Interactive/latency-sensitive |
| `Critical` | 3 | System-level or time-critical |

```rust,ignore
use moirai_core::{TaskBuilder, Priority};

let task = TaskBuilder::new()
    .priority(Priority::High)
    .name("dose_kernel")
    .build(|| compute_dose_kernel());
```

## Parallel Iterators

`moirai-iter` provides parallel iterator combinators that distribute work
across the worker pool. The API mirrors Rayon's parallel iterators for
interoperability in benchmarks, with explicit value-semantic coverage and
benchmark-contract rows for every adapter.

```rust,ignore
use moirai::ParallelIteratorExt;

let total: f64 = data.par_iter()
    .map(|x| x * x)
    .sum();
```

## SIMD Optimization

The `moirai-iter` SIMD path delivers 4–8× throughput improvement for
vectorizable workloads by detecting the host's SIMD capability at runtime
and dispatching to the widest available instruction set.
