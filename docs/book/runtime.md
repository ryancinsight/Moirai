# Runtime

Moirai is Atlas's hybrid concurrency runtime. It weaves together asynchronous
task scheduling and parallel work-stealing into a single unified executor.

## Building the Runtime

```rust,ignore
use moirai::Moirai;

let runtime = Moirai::builder()
    .worker_threads(4)
    .build()?;
```

`Moirai::builder()` returns a `MoiraiBuilder` that accepts:

- `worker_threads(n)` — number of CPU-bound worker threads (defaults to logical core count)
- `max_blocking_threads(n)` — pool size for blocking tasks
- `thread_name_prefix(s)` — prefix for thread names in diagnostics

The built `Moirai` handle owns the runtime and shuts it down cleanly when dropped.

## Execution Contexts

Moirai supports three execution contexts:

| Context | API | Use Case |
|---------|-----|---------|
| CPU-bound parallel | `spawn_fn` | Physics kernels, matrix ops, image processing |
| Async tasks | `spawn` (async block) | I/O, network, file, coordinated workflows |
| Blocking | `spawn_blocking` | Synchronous I/O or FFI that must not block async workers |

## Graceful Shutdown

```rust,ignore
runtime.shutdown();  // wait for all tasks to drain
```

The runtime drops all worker threads and their associated resources.

## Performance Characteristics

- Task scheduling overhead: < 1 µs per task
- Work-stealing load balancing across all worker threads
- Cache-aligned task metadata (< 64 bytes per task)
- Zero-copy task passing where possible
