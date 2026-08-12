# Blocking Tasks

Blocking tasks are operations that must not block an async worker thread.
Moirai routes them through a separate blocking thread pool.

## `spawn_blocking`

```rust,ignore
let handle = runtime.spawn_blocking(|| {
    std::fs::read_to_string("large_file.dat")?
});
let content = handle.join()??;
```

## When to Use Blocking vs. Async

| Situation | Recommended API |
|-----------|----------------|
| CPU-bound computation | `spawn_fn` (parallel pool) |
| Async I/O, network, timers | `spawn` (async pool) |
| Synchronous I/O, FFI | `spawn_blocking` |

The blocking pool grows on demand up to `max_blocking_threads`. Threads park
after idle timeout and are recycled on the next call.
