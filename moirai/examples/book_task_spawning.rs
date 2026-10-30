//! Spawn, await, cancel, and shut down tasks through the public facade.
//!
//! [`Moirai`] schedules synchronous and asynchronous work on its work-stealing
//! runtime. This example retains every result-bearing handle, observes its
//! value or typed terminal error, and shuts the runtime down explicitly.

extern crate moirai;

use std::fmt;
use std::future::pending;
use std::time::Duration;

use moirai::prelude::TaskManager;
use moirai::{ExecutorError, Moirai, Priority, TaskBuilder, TaskError, TaskSpawner};

#[derive(Debug)]
enum ExampleError {
    Executor(ExecutorError),
    Task(TaskError),
    MissingResult,
}

impl fmt::Display for ExampleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Executor(error) => error.fmt(formatter),
            Self::Task(error) => error.fmt(formatter),
            Self::MissingResult => formatter.write_str("task handle has no result"),
        }
    }
}

impl std::error::Error for ExampleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Executor(error) => Some(error),
            Self::Task(error) => Some(error),
            Self::MissingResult => None,
        }
    }
}

impl From<ExecutorError> for ExampleError {
    fn from(error: ExecutorError) -> Self {
        Self::Executor(error)
    }
}

impl From<TaskError> for ExampleError {
    fn from(error: TaskError) -> Self {
        Self::Task(error)
    }
}

fn run_task_spawning() -> Result<(), ExampleError> {
    let runtime = Moirai::new()?;

    let h1 = TaskSpawner::spawn(
        &runtime,
        TaskBuilder::new().build(|| (0_u64..1_000).sum::<u64>()),
    )?;
    let h2 = TaskSpawner::spawn(
        &runtime,
        TaskBuilder::new().build(|| (0_u64..1_000).map(|n| n * n).sum::<u64>()),
    )?;
    let h3 = TaskSpawner::spawn(
        &runtime,
        TaskBuilder::new().build(|| (1_u64..=10).product::<u64>()),
    )?;

    let sum = h1.join().ok_or(ExampleError::MissingResult)??;
    let sumsq = h2.join().ok_or(ExampleError::MissingResult)??;
    let fact = h3.join().ok_or(ExampleError::MissingResult)??;

    println!("sum(0..1000)    = {sum}");
    println!("sumsq(0..1000)  = {sumsq}");
    println!("10!             = {fact}");

    assert_eq!(sum, 499_500);
    assert_eq!(sumsq, 332_833_500);
    assert_eq!(fact, 3_628_800);

    let high = TaskSpawner::spawn_with_priority(
        &runtime,
        TaskBuilder::new().build(|| 1_u32),
        Priority::High,
        None,
    )?;
    let low = TaskSpawner::spawn_with_priority(
        &runtime,
        TaskBuilder::new().build(|| 0_u32),
        Priority::Low,
        None,
    )?;

    let h = high.join().ok_or(ExampleError::MissingResult)??;
    let l = low.join().ok_or(ExampleError::MissingResult)??;
    println!("high={h}, low={l}");
    assert_eq!(h, 1);
    assert_eq!(l, 0);

    let awaited_value = runtime.block_on(async {
        let awaited =
            TaskSpawner::spawn_async(&runtime, async { 42_usize }).map_err(ExampleError::from)?;
        awaited
            .await
            .ok_or(ExampleError::MissingResult)?
            .map_err(ExampleError::from)
    })?;
    println!("awaited async value = {awaited_value}");
    assert_eq!(awaited_value, 42);

    let cancelled = TaskSpawner::spawn_async(&runtime, pending::<usize>())?;
    runtime.cancel_task(cancelled.id())?;
    assert_eq!(runtime.block_on(cancelled), Some(Err(TaskError::Cancelled)));

    let deadline = TaskSpawner::spawn_async(&runtime, pending::<usize>())?;
    let deadline_id = deadline.id();
    assert!(matches!(
        runtime.block_on(runtime.wait_for_task(deadline_id, Some(Duration::ZERO))),
        Err(ExecutorError::WaitTimedOut { id }) if id == deadline_id
    ));
    assert!(!deadline.is_finished());
    runtime.cancel_task(deadline.id())?;
    assert_eq!(runtime.block_on(deadline), Some(Err(TaskError::Cancelled)));

    let shutdown = TaskSpawner::spawn_async(&runtime, pending::<usize>())?;
    runtime.shutdown();
    assert_eq!(shutdown.join(), Some(Err(TaskError::Cancelled)));
    assert!(matches!(
        TaskSpawner::spawn_async(&runtime, async { 1_usize }),
        Err(ExecutorError::ShuttingDown)
    ));

    println!("all task-spawning assertions passed");
    Ok(())
}

fn main() -> Result<(), ExampleError> {
    run_task_spawning()
}

#[test]
fn task_spawning_workflow() -> Result<(), ExampleError> {
    run_task_spawning()
}
