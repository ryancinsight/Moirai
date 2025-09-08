//! Async/await support for Moirai concurrency library.
//!
//! This module provides async runtime integration for Moirai, enabling seamless
//! interop between sync and async tasks while maintaining high performance.
//! 
//! Following SLAP principle, this module now serves as a facade that re-exports
//! functionality from focused, single-responsibility modules.

// Focused modules following SLAP principle
pub mod executor;
pub mod fs;
pub mod net;
pub mod sync;
pub mod timer;

// Re-export async executor functionality
pub use executor::{AsyncExecutor, AsyncHandle, ExecutorStats};

// Re-export timer functionality
pub use timer::{
    sleep, timeout, interval, interval_at, 
    Delay, Timeout, TimeoutError, Interval, RateLimiter, RatePermit,
    TimerWheel, TimerCommand
};

// Re-export networking functionality
pub use net::{
    TcpListener, TcpStream, UdpSocket,
    TcpServerConfig, TcpServerStats, UdpConfig, UdpSocketStats,
    ConnectionInfo, ConnectionPool, ConnectionStats
};

// Re-export file I/O functionality
pub use fs::{
    File, FileOpenOptions, FileStats,
    read, read_to_string, write, write_str, append, append_str,
    copy, remove_file, create_dir, create_dir_all, remove_dir, remove_dir_all
};

// Re-export sync primitives
pub use sync::{
    Broadcast, BroadcastError, BroadcastReceiver, BroadcastSender,
    Notify, RwLock, Semaphore, SemaphorePermit, 
    Watch, WatchError, WatchReceiver, WatchSender,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_integration_async_executor() {
        let executor = AsyncExecutor::new().expect("Failed to create executor");
        
        let _handle = executor.spawn(async {
            moirai_pal::timer::sleep(Duration::from_millis(10)).await.ok();
            "async task completed"
        });

        // Run one iteration manually for testing
        // In production, executor.run() would be called
        let stats_before = executor.stats();
        assert_eq!(stats_before.tasks_spawned, 1);
        assert_eq!(stats_before.tasks_pending, 1);
        
        // For now, we can't test the full execution without running the executor
        // This test validates the spawning mechanism works
    }

    #[test]
    fn test_integration_executor_stats() {
        let executor = AsyncExecutor::new().expect("Failed to create executor");
        
        let stats = executor.stats();
        assert_eq!(stats.tasks_spawned, 0);
        assert_eq!(stats.tasks_completed, 0);
        assert_eq!(stats.tasks_pending, 0);
        assert_eq!(stats.io_operations, 0);
    }

    #[test] 
    fn test_integration_reactor_access() {
        let executor = AsyncExecutor::new().expect("Failed to create executor");
        
        // Verify we can access the underlying reactor
        let _reactor = executor.reactor();
        
        // This confirms the integration with PAL reactor
    }
}
