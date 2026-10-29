# Async Tasks

Moirai's async layer lets you spawn `async` blocks and `Future`s alongside
CPU-bound parallel work, sharing the same worker pool.

## Spawning Async Tasks

```rust,ignore
use moirai::{Moirai, TaskError};

let runtime = Moirai::new()?;
let result = runtime.block_on(async {
    let handle = runtime.spawn_async(async { 42usize });
    handle.await
});
assert_eq!(result, Some(Ok::<usize, TaskError>(42)));
```

`spawn_async` accepts any `Future<Output = T>` where `T: Send + 'static`. It
returns a `TaskHandle<T>` that can be awaited.

## `TaskHandle<T>`

`TaskHandle<T>` is the future returned by `spawn_async`. It resolves to
`Option<Result<T, TaskError>>`, the same value returned by blocking `join`.
`None` means the handle was detached or its result was already consumed by
polling. Dropping the handle alone does not cancel its task. A `TaskError` can
report that:

- The spawned task panicked
- Explicit cancellation or shutdown cancelled the task at a poll boundary

```rust,ignore
match handle.await {
    Some(Ok(value)) => println!("result: {value}"),
    Some(Err(TaskError::Panicked)) => eprintln!("task panicked"),
    Some(Err(TaskError::Cancelled)) => eprintln!("task cancelled"),
    Some(Err(error)) => eprintln!("task failed: {error}"),
    None => eprintln!("task result was detached or already consumed"),
}
```

## `TaskId` and `TaskContext`

Each spawned task receives a `TaskId(u64)` that uniquely identifies it within
the runtime lifetime. `TaskContext` carries the id and the task's `Priority`.

`TaskExt` transforms task definitions before spawning. Its current methods are
`map` for output transformation and `with_context` for assigning a
`TaskContext`; it does not operate on a spawned `TaskHandle`.
