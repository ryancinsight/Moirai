//! # Moirai - Weaving the Threads of Fate
//!
//! Moirai is a high-performance hybrid concurrency library for Rust that seamlessly
//! blends asynchronous and parallel execution models. Named after the Greek Fates
//! who controlled the threads of life, Moirai weaves together the best principles
//! from async task scheduling and parallel work-stealing into a unified framework.
//!
//! ## Core Design Principles
//!
//! Moirai follows elite programming practices:
//! - **SOLID**: Single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion
//! - **CUPID**: Composable, Unix philosophy, predictable, idiomatic, domain-centric
//! - **GRASP**: Information expert, creator, controller, low coupling, high cohesion
//! - **ACID**: Atomicity, consistency, isolation, durability in task execution
//!
//! ## Features
//!
//! - **Zero-cost abstractions**: All abstractions compile away to optimal code
//! - **Hybrid execution**: Seamlessly mix async and parallel tasks
//! - **Work-stealing scheduler**: Intelligent load balancing across CPU cores
//! - **Memory safety**: Leverage Rust's ownership system for safe concurrency
//! - **High performance**: Sub-microsecond task scheduling overhead
//! - **NUMA awareness**: Optimize for modern multi-socket systems
//! - **Rich iterator combinators**: Parallel and async iterator processing
//! - **IPC**: Inter-process communication (optional)
//! - **Metrics**: Performance monitoring (optional)
//! - **Distributed transport feature gates**: Optional transport and iterator helpers without a
//!   facade-level remote-closure API
//!
//! ## Performance Characteristics
//!
//! - **Task scheduling overhead**: < 1μs per task
//! - **Memory efficiency**: Zero-copy task passing where possible
//! - **Scalability**: Linear scaling up to CPU core count
//! - **SIMD optimization**: 4-8x performance improvement for vectorizable workloads
//! - **NUMA awareness**: Reduced memory latency on multi-socket systems
//!
//! ## Safety Guarantees
//!
//! - **Memory safety**: All operations are memory-safe by construction
//! - **Data race freedom**: Rust's ownership system prevents data races
//! - **Deadlock prevention**: Lock-free data structures where possible
//! - **Resource cleanup**: Automatic resource cleanup on task completion
//! - **Error handling**: Comprehensive error types with recovery mechanisms
//!
//! ## Quick Start Example
//!
//! ```rust
//! use moirai::Moirai;
//! use std::sync::atomic::{AtomicU32, Ordering};
//! use std::sync::Arc;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a new runtime with optimal configuration
//! let runtime = Moirai::builder()
//!     .worker_threads(4)
//!     .build()?;
//!
//! // CPU-bound parallel computation
//! let counter = Arc::new(AtomicU32::new(0));
//! let counter_clone = counter.clone();
//! let parallel_handle = runtime.spawn_fn(move || {
//!     // Simulate intensive computation
//!     for i in 0..1000 {
//!         counter_clone.fetch_add(i % 100, Ordering::Relaxed);
//!     }
//!     counter_clone.load(Ordering::Relaxed)
//! });
//!
//! // Another parallel task
//! let critical_handle = runtime.spawn_fn(move || "critical task executed");
//!
//! // Tasks execute concurrently with optimal scheduling
//! let parallel_result = parallel_handle.join().unwrap().unwrap();
//! let critical_result = critical_handle.join().unwrap().unwrap();
//!
//! println!("Parallel result: {}", parallel_result);
//! println!("Critical result: {}", critical_result);
//!
//! // Graceful shutdown with resource cleanup
//! runtime.shutdown();
//! # Ok(())
//! # }
//! ```
//!
//! ## Advanced Usage Patterns
//!
//! ### Task Chaining and Composition
//!
//! ```rust
//! use moirai::Moirai;
//!
//! # fn chaining_example() -> Result<(), Box<dyn std::error::Error>> {
//! let runtime = Moirai::new()?;
//!
//! // Chain tasks with dependencies using regular closures
//! let handle1 = runtime.spawn_fn(|| 42);
//! let result1 = handle1.join().unwrap().unwrap();
//!
//! let handle2 = runtime.spawn_fn(move || result1 * 2);
//! let result2 = handle2.join().unwrap().unwrap();
//!
//! let handle3 = runtime.spawn_fn(move || result2 + 10);
//! let result = handle3.join().unwrap().unwrap();
//!
//! assert_eq!(result, 94); // (42 * 2) + 10
//! # Ok(())
//! # }
//! ```
//!
//! ### Distributed Boundary
//!
//! ```rust
//! use moirai::Moirai;
//!
//! # fn boundary_example() -> Result<(), Box<dyn std::error::Error>> {
//! let runtime = Moirai::builder()
//!     .worker_threads(2)
//!     .build()?;
//!
//! // Execute task locally through the verified scheduler facade.
//! let handle = runtime.spawn_fn(move || "computed locally");
//! let result = handle.join().unwrap().unwrap();
//! println!("Result: {}", result);
//!
//! // Cross-machine execution uses fixed-format capability tokens; arbitrary
//! // remote closure execution is intentionally outside the public facade.
//! # Ok(())
//! # }
//! ```
//!
//! ## Migration Guide
//!
//! ### From `std::thread`
//!
//! ```rust
//! # fn expensive_computation() -> i32 { 42 }
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Before: std::thread
//! let handle = std::thread::spawn(|| {
//!     expensive_computation()
//! });
//! let result = handle.join().unwrap();
//!
//! // After: Moirai
//! let runtime = moirai::Moirai::new()?;
//! let handle = runtime.spawn_fn(|| {
//!     expensive_computation()
//! });
//! let result = handle.join().unwrap().unwrap();
//! # Ok(())
//! # }
//! ```
//!
//! ### From Tokio
//!
//! ```rust
//! # fn async_operation() -> String { "result".to_string() }
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Before: std::thread (since tokio requires async context)
//! let handle = std::thread::spawn(|| {
//!     async_operation()
//! });
//! let result = handle.join().unwrap();
//!
//! // After: Moirai
//! let runtime = moirai::Moirai::new()?;
//! let handle = runtime.spawn_fn(|| {
//!     async_operation()
//! });
//! let result = handle.join().unwrap().unwrap();
//! # Ok(())
//! # }
//! ```
//!
//! ### From Rayon
//!
//! ```rust
//! # fn expensive_transform(x: &i32) -> i32 { x * 2 }
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let data = vec![1, 2, 3, 4, 5];
//!
//! // Before: Sequential processing
//! let result: Vec<_> = data.iter()
//!     .map(|x| expensive_transform(x))
//!     .collect();
//!
//! // After: Moirai parallel processing
//! let runtime = moirai::Moirai::new()?;
//! let handles: Vec<_> = data.iter()
//!     .map(|&x| runtime.spawn_fn(move || expensive_transform(&x)))
//!     .collect();
//! let result: Result<Vec<_>, _> = handles.into_iter()
//!     .map(|h| h.join().unwrap())
//!     .collect();
//! # Ok(())
//! # }
//! ```

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#[cfg(feature = "mnemosyne")]
#[cfg(not(feature = "no-global-alloc"))]
#[global_allocator]
static ALLOC: mnemosyne::Mnemosyne = mnemosyne::Mnemosyne;

// Re-export core functionality (avoiding ExecutorStats conflict)
pub use moirai_core::{
    error::*,
    executor::{Executor, ExecutorConfig, ExecutorControl, TaskSpawner},
    scheduler::*,
    task::*,
    Priority, Task, TaskContext, TaskHandle, TaskId,
};

// Re-export executor functionality
pub use moirai_executor::{BlockingTask, HybridExecutor, SchedulerScope};

/// Completion-only borrowing scope for jobs submitted to the unified scheduler.
pub type MoiraiScope<'scope> = SchedulerScope<'scope, BlockingTask>;

// Re-export scheduler functionality
pub use moirai_scheduler::WorkStealingScheduler;

// Re-export transport functionality
pub use moirai_transport::{
    Address, InMemoryTransport, RemoteAddress, TransportError, TransportManager, TransportResult,
    UniversalChannel, UniversalReceiver, UniversalSender,
};

#[cfg(feature = "distributed")]
mod routed;

#[cfg(feature = "distributed")]
pub use moirai_executor::schedule::{
    AsyncTask, HybridRoutePolicy, HybridRouter, RoutePolicy, RouteTopology, SchedulerRoute,
    ServerRoutePolicy, SyncTask, ThreadRoutePolicy, WorkClass,
};

#[cfg(feature = "distributed")]
pub use moirai_transport::{
    process::{ProcessDropPolicy, ProcessSpec, ProcessWaitPolicy},
    remote_task::{
        EchoBytesCapability, IntoRemoteOperation, RemoteCapability, RemoteCapabilityToken,
        RemoteTaskId, RemoteTaskOperationKind, RemoteTaskOutput, RemoteTaskResult,
        SumU64Capability,
    },
    route::{
        ProcessEndpoint, RouteAddressBook, RouteNamespace, RouteService, RoutedProcessTaskError,
        RoutedProcessTaskOutput, ServerEndpoint,
    },
};

#[cfg(feature = "distributed")]
pub use routed::{FixedRemoteTask, RoutedProcessTarget, RoutedServerTarget};

// Re-export channel functionality from core
pub use moirai_core::channel;

#[cfg(feature = "network")]
pub use moirai_transport::{TcpTransport, UdpTransport};

// Re-export synchronization primitives
pub use moirai_sync::{AtomicCounter, Barrier, Condvar, Mutex, Once, RwLock};

// Re-export metrics functionality
#[cfg(feature = "metrics")]
pub use moirai_metrics::MetricsCollector;

// Re-export async functionality (specific imports to avoid conflicts)
#[cfg(feature = "async")]
pub use moirai_async::{
    executor::{AsyncExecutor, AsyncHandle},
    io::{
        AsyncBufRead, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, MoiraiCompat, TokioCompat,
    },
    timer::{sleep, timeout},
    File, FileOpenOptions, TcpListener, TcpStream, Timeout,
};

// Re-export iterator functionality
#[cfg(feature = "iter")]
pub use moirai_iter::{
    async_range, moirai_iter, moirai_iter_async, moirai_iter_distributed, moirai_iter_hybrid,
    moirai_iter_multi_system, moirai_iter_parallel, par_range, AsyncContext, AsyncIterator,
    AsyncParallelIterator, DistributedContext, DistributedIterator, ExecutionBase,
    ExecutionContext, ExecutionStrategy, HybridConfig, HybridContext, IndexedParallelIterator,
    IntoAsyncIterator, IntoParallelIterator, IntoParallelRefIterator, MoiraiIterator,
    MultiSystemContext, MultiSystemIterator, NodeConfig, ParallelContext, ParallelExtend,
    ParallelIterator, PerformanceHistory, RangeParIter, SystemConfig, ThreadPool, VecParIter,
    VecRefParIter,
};

// Re-export GPU functionality
#[cfg(feature = "gpu")]
pub use moirai_gpu::prelude::*;

// Synchronous data-parallel primitives (rayon-replacement surface), provided by
// the `moirai-parallel` domain crate: monomorphized ExecutionPolicy + the
// adaptive `par_*` helpers.
#[cfg(feature = "parallel")]
pub use moirai_parallel::*;

#[cfg(all(feature = "parallel", feature = "melinoe"))]
pub use moirai_parallel::melinoe_ext::*;

// Submodules
mod builder;
mod global;
/// Convenience functions for common operations.
///
/// Common imports for Moirai users.
pub mod prelude;
mod runtime;

#[cfg(test)]
mod tests;

// Facade re-exports
pub use builder::MoiraiBuilder;
pub use global::{block_on, global, spawn_async, spawn_fn};
pub use runtime::Moirai;
