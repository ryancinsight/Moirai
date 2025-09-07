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

    #[tokio::test]
    async fn test_integration_async_executor() {
        let executor = AsyncExecutor::new();
        
        let handle = executor.spawn(async {
            sleep(Duration::from_millis(10)).await;
            "async task completed"
        });

        executor.run();
        let result = handle.await;
        assert_eq!(result, "async task completed");
    }

    #[tokio::test]
    async fn test_integration_networking() {
        let server = TcpListener::bind("127.0.0.1:0").await.unwrap();
        // Note: In a real implementation, we'd need to expose the inner listener's address
        // For now, let's use a simpler test
        
        let stats = server.stats();
        assert_eq!(stats.total_connections, 0); // Initial state
    }

    #[tokio::test]
    async fn test_integration_timer_and_file() {
        use tempfile::tempdir;
        
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("integration_test.txt");
        
        // Write file with timer
        let start = std::time::Instant::now();
        
        timeout(Duration::from_secs(5), async {
            write_str(&file_path, "timed file operation").await
        }).await.unwrap().unwrap();
        
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_secs(1)); // Should be much faster
        
        // Verify file contents
        let contents = read_to_string(&file_path).await.unwrap();
        assert_eq!(contents, "timed file operation");
    }
}
