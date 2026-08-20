# Example: Task Spawning

**Crate**: `moirai`
**Source**: `moirai/examples/book_task_spawning.rs`

Create a `Moirai` runtime, spawn three independent compute tasks, join their
results, and demonstrate the `Priority` hint.

## Source

```rust
{{#include ../../../moirai/examples/book_task_spawning.rs}}
```

## Output

```text
sum(0..1000)    = 499500
sumsq(0..1000)  = 332833500
10!             = 3628800
high=1, low=0
all task-spawning assertions passed
```

## What to notice

- `Moirai::new()` creates the runtime with one worker thread per logical CPU.
  No pool size is passed; the scheduler detects hardware parallelism.

- `spawn_fn(|| ...)` schedules the closure as a task and immediately returns a
  `TaskHandle<R>`.  The task may begin executing before `join` is called.

- `handle.join()` blocks the calling thread until the task completes.  It
  returns `Some(Ok(value))` on success, `Some(Err(...))` on task panic, and
  `None` if the handle was already consumed.

- `spawn_fn_with_priority(|| ..., Priority::High)` hints to the scheduler that
  this task should run before lower-priority tasks when the ready queue is
  non-empty.  Priority is a hint, not a guarantee.
