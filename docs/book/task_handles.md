# Task Handles

A `TaskHandle<T>` is the ownership token for a spawned Moirai task.

## Joining a Handle

```rust,ignore
// Synchronous join (blocks the calling thread)
let result: Option<Result<T, TaskError>> = handle.join();

// Async await (suspends the calling async task)
let result: Result<T, TaskError> = handle.await;
```

`join()` returns `None` when the task was cancelled.

## `TaskStatus`

| Variant | Description |
|---------|-------------|
| `Pending` | Queued, not yet started |
| `Running` | Currently executing on a worker |
| `Completed` | Finished with a result |
| `Cancelled` | Explicitly cancelled |
| `Failed` | Completed with a `TaskError` |

## Task Cancellation

```rust,ignore
handle.cancel();  // request cancellation; non-blocking
```

Cancellation is best-effort: a running task finishes before stopping.

## `TaskExt` Combinators

| Combinator | Description |
|-----------|-------------|
| `map(f)` | Transform the output type |
| `then(f)` | Chain a dependent closure |
| `inspect(f)` | Side-effect on the result |
