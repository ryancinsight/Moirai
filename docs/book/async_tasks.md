# Async Tasks

Moirai's async layer lets you spawn `async` blocks and `Future`s alongside
CPU-bound parallel work, sharing the same worker pool.

## Spawning Async Tasks

```rust,ignore
let handle = runtime.spawn(async {
    let data = fetch_data().await;
    process(data).await
});

let result = handle.await?;
```

`spawn` accepts any `Future<Output = T>` where `T: Send + 'static`. It returns
a `TaskHandle<T>` that can be `.await`ed.

## `TaskHandle<T>`

`TaskHandle<T>` is the future returned by `spawn`. It resolves to
`Result<T, TaskError>`. A `TaskError` is returned when:

- The spawned task panicked
- The runtime was shut down before the task completed

```rust,ignore
match handle.await {
    Ok(value) => println!("result: {value}"),
    Err(TaskError::Panicked) => eprintln!("task panicked"),
    Err(TaskError::RuntimeShutdown) => eprintln!("runtime shut down"),
    _ => {}
}
```

## `TaskId` and `TaskContext`

Each spawned task receives a `TaskId(u64)` that uniquely identifies it within
the runtime lifetime. `TaskContext` carries the id and the task's `Priority`.

## Task Chaining

`TaskExt` provides combinators for chaining task computations:

```rust,ignore
use moirai_core::TaskExt;

let base = TaskBuilder::new().build(|| 21);
let doubled = base.then(|x| x * 2);  // chains onto the result
let mapped  = base.map(|x| x * 2);   // maps the output type
```
