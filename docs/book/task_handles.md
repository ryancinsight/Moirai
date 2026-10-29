# Task Handles

A `TaskHandle<T>` is the ownership token for a spawned Moirai task.

## Collecting a Result

```rust,ignore
use moirai::{Moirai, TaskError};

let runtime = Moirai::new()?;

let blocking = runtime.spawn_fn(|| 21usize);
let joined: Option<Result<usize, TaskError>> = blocking.join();

let asynchronous = runtime.spawn_async(async { 42usize });
let awaited: Option<Result<usize, TaskError>> = runtime.block_on(asynchronous);
```

`join()` blocks the calling thread. Awaiting suspends the calling async task and
lets the worker run other work. Both paths return `Some(Ok(value))` on success
and `Some(Err(error))` on task failure. `None` means the handle was detached or
its result was already consumed by polling. Cancellation is reported as
`Some(Err(TaskError::Cancelled))`, not `None`.

Dropping a handle discards only the consumer's access to the result. It does not
request task cancellation.

## `TaskStatus`

| Variant | Description |
|---------|-------------|
| `Queued` | Queued, not yet started |
| `Running` | Currently executing on a worker |
| `Completed` | Finished with a result |
| `Cancelled` | Cancelled on request, or discarded before it finished (for example at shutdown) |
| `Failed` | Completed with a `TaskError` |

## Task Cancellation

```rust,ignore
use moirai::{Moirai, TaskError};
use moirai::prelude::TaskManager;

let runtime = Moirai::new()?;
let handle = runtime.spawn_async(core::future::pending::<usize>());
runtime.cancel_task(handle.id())?;
assert_eq!(runtime.block_on(handle), Some(Err(TaskError::Cancelled)));
```

Cancellation is explicit and cooperative. Queued work skips its body. An async
task observes cancellation at a poll boundary, although a poll already
returning a value may complete first. Running synchronous and blocking closures
are not preempted. Cancelling an unknown task ID returns an error.

## Completion Deadlines

`TaskManager::wait_for_task(id, Some(duration))` waits for success, failure, or
cancellation without consuming the result handle. Completion visible at a poll
wins a simultaneous timeout. A timeout ends only that wait and leaves the task
running; call `cancel_task` separately to request cancellation. Passing `None`
uses no timer.

## `TaskExt` Combinators

| Combinator | Description |
|-----------|-------------|
| `map(f)` | Transform the output type |
| `with_context(context)` | Replace the task's execution context |
