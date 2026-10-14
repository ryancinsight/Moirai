//! Fallible core spawning from the facade for protocol-driver integration.
//!
//! Moirai's inherent convenience methods panic on admission failure. The core
//! trait preserves admission errors, allowing QUIC to retain spawner rejection
//! across its upstream runtime ABI. Both call the same scheduler implementation;
//! this introduces neither another pool nor a compatibility API.

use super::Moirai;
use moirai_core::{Priority, Task, TaskHandle, error::ExecutorResult, executor::TaskSpawner};

impl TaskSpawner for Moirai {
    #[inline]
    fn spawn<T: Task + Send + 'static>(&self, task: T) -> ExecutorResult<TaskHandle<T::Output>> {
        self.executor.spawn(task)
    }

    #[inline]
    fn spawn_async<F>(&self, future: F) -> ExecutorResult<TaskHandle<F::Output>>
    where
        F: core::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.executor.spawn_async(future)
    }

    #[inline]
    fn spawn_blocking<F, R>(&self, function: F) -> ExecutorResult<TaskHandle<R>>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.executor.spawn_blocking(function)
    }

    #[inline]
    fn spawn_detached<F>(&self, function: F) -> ExecutorResult<()>
    where
        F: FnOnce() + Send + 'static,
    {
        self.executor.spawn_detached(function)
    }

    #[inline]
    fn spawn_with_priority<T>(
        &self,
        task: T,
        priority: Priority,
        locality: Option<usize>,
    ) -> ExecutorResult<TaskHandle<T::Output>>
    where
        T: Task + Send + 'static,
    {
        self.executor.spawn_with_priority(task, priority, locality)
    }
}
