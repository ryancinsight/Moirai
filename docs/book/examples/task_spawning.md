# Example: Task Spawning

**Crate**: `moirai`
**Source**: `moirai/examples/book_task_spawning.rs`

Create a `Moirai` runtime, collect synchronous and awaited asynchronous results,
cancel pending work, observe a non-cancelling deadline, and shut down with a
retained pending-task handle.

## Source

```rust,edition2024
{{#include ../../../moirai/examples/book_task_spawning.rs}}
```

## Output

```text
sum(0..1000)    = 499500
sumsq(0..1000)  = 332833500
10!             = 3628800
high=1, low=0
awaited async value = 42
all task-spawning assertions passed
```

## What to notice

- `Moirai::new()` creates the runtime with one worker thread per logical CPU.
  No pool size is passed; the scheduler detects hardware parallelism.

- `TaskSpawner::spawn` and `TaskSpawner::spawn_async` are the fallible facade
  operations. They return admission errors instead of panicking when shutdown
  has begun.

- A `TaskHandle<R>` can be joined or awaited. Both paths return
  `Some(Ok(value))` on success, a typed task error on failure, and `None` only
  after the result has already been consumed or the handle was detached.

- `Priority::High` is a scheduling hint, not a preemption guarantee. Cancellation
  and shutdown settle pending asynchronous work at poll boundaries; they do not
  forcibly interrupt a synchronous or blocking closure that is already running.

- A zero-duration `wait_for_task` deadline returns
  `ExecutorError::WaitTimedOut` with the task identifier while the task remains
  alive. The example then cancels and collects that task explicitly. Unknown
  identifiers and timer-service failures use distinct executor variants.

- Explicit shutdown cancels pending asynchronous work and joins the workers.
  Dropping the runtime also joins its workers, so Drop can wait for synchronous
  or blocking closures that are still running.
