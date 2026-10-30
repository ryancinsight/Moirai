//! Executor lifecycle control interface.

/// Provides control operations for executor lifecycle management.
///
/// This trait enables external systems to manage executor state transitions,
/// perform health checks, and coordinate shutdown procedures.
#[allow(clippy::module_name_repetitions)]
pub trait ExecutorControl: Send + Sync + 'static {
    /// Block the current thread until the future completes.
    ///
    /// # Behavior Guarantees
    /// - Blocks calling thread until future resolves
    /// - Supports nested async operations within the future
    /// - Handles panic propagation from the future
    /// - May deadlock if future depends on blocked thread
    ///
    /// # Performance Characteristics
    /// - Optimal for CPU-bound futures with minimal I/O
    /// - May block calling thread indefinitely
    /// - Memory: Future size + execution context
    /// - Suitable for main thread or dedicated blocking contexts
    fn block_on<F>(&self, future: F) -> F::Output
    where
        F: core::future::Future;

    /// Attempt to run tasks without blocking.
    ///
    /// # Behavior Guarantees
    /// - Non-blocking operation, returns immediately
    /// - Returns true if any work was performed
    /// - May perform multiple task executions in single call
    /// - Suitable for integration with external event loops
    ///
    /// # Performance Characteristics
    /// - O(1) operation, < 1μs typical latency
    /// - Work stealing: Attempts to balance load across threads
    /// - Suitable for event loops requiring non-blocking progress
    fn try_run(&self) -> bool;

    /// Stop admission and settle async work at poll boundaries.
    ///
    /// Running synchronous and blocking closures continue until they return.
    /// External callers wait for shared shutdown completion. Executor workers
    /// and inline callbacks request shutdown without waiting on owned work.
    fn shutdown(&self);

    /// Bound the caller's wait for shutdown.
    ///
    /// Async cancellation occurs at poll boundaries. When the caller's timeout
    /// expires, running closures and their threads continue draining in the
    /// background. They are not forcibly terminated; executor destruction only
    /// requests that drain. A non-yielding async poll cannot be interrupted.
    fn shutdown_timeout(&self, timeout: core::time::Duration);

    /// Check if the executor is shutting down.
    ///
    /// # Behavior Guarantees
    /// - Returns true once shutdown has been initiated
    /// - Eventually consistent across all threads
    /// - Remains true until executor is fully stopped
    ///
    /// # Performance Characteristics
    /// - O(1) operation, < 10ns latency
    /// - Non-blocking atomic read operation
    /// - Memory ordering: Acquire semantics
    fn is_shutting_down(&self) -> bool;

    /// Get the number of worker threads.
    ///
    /// # Behavior Guarantees
    /// - Returns configured number of worker threads
    /// - Does not include async or blocking thread pools
    /// - Constant value set during executor creation
    ///
    /// # Performance Characteristics
    /// - O(1) operation, immediate return
    /// - No synchronization overhead
    fn worker_count(&self) -> usize;

    /// Get the current load (number of pending tasks).
    ///
    /// # Behavior Guarantees
    /// - Returns approximate pending task count
    /// - Eventually consistent across distributed queues
    /// - May include tasks currently being executed
    /// - Does not include blocked or suspended tasks
    ///
    /// # Performance Characteristics
    /// - O(1) operation for local queues
    /// - May involve atomic reads across threads
    /// - Latency: < 100ns typical
    fn load(&self) -> usize;
}
