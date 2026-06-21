use crate::Moirai;
use moirai_core::task::TaskHandle;
use std::future::Future;

/// Global runtime instance for convenience.
static GLOBAL_RUNTIME: std::sync::OnceLock<Moirai> = std::sync::OnceLock::new();

/// Get or initialize the global Moirai runtime.
///
/// This provides a convenient way to access a shared runtime instance
/// without having to pass it around explicitly.
///
/// # Panics
///
/// Panics if the global runtime fails to initialize, which should not happen
/// under normal circumstances unless there are severe system resource constraints.
pub fn global() -> &'static Moirai {
    // Wrap the SAME shared executor that `moirai-parallel` schedules data-parallel
    // work on, so async tasks (spawn_async/block_on) and parallel work run on one
    // unified hybrid scheduler — a parallel worker can drive async work in-process.
    GLOBAL_RUNTIME.get_or_init(|| Moirai {
        executor: moirai_executor::shared(),
    })
}

/// Spawn an async task on the global runtime.
pub fn spawn_async<F>(future: F) -> TaskHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    global().spawn_async(future)
}

/// Spawn a parallel task on the global runtime.
pub fn spawn_fn<F, R>(func: F) -> TaskHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    global().spawn_fn(func)
}

/// Block on a future using the global runtime.
pub fn block_on<F>(future: F) -> F::Output
where
    F: Future,
{
    global().block_on(future)
}
