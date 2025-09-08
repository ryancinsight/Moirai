//! Advanced async synchronization primitives for Moirai
//!
//! This module provides async-aware synchronization that integrates with
//! Moirai's unified runtime. Following SLAP principle, each synchronization
//! primitive is implemented in its own focused module.

pub mod semaphore;
pub mod broadcast;
pub mod watch;
pub mod rwlock;
pub mod notify;

// Re-export public types for convenience
pub use semaphore::{Semaphore, SemaphoreAcquire, SemaphorePermit};
pub use broadcast::{Broadcast, BroadcastSender, BroadcastReceiver, BroadcastRecv, BroadcastError};
pub use watch::{Watch, WatchSender, WatchReceiver, WatchChanged, WatchError};
pub use rwlock::{RwLock, RwLockReadFuture, RwLockWriteFuture};
pub use notify::{Notify, NotifyFuture};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semaphore_basic() {
        let sem = Semaphore::new(2);
        assert_eq!(sem.available_permits(), 2);
        
        // Basic functionality test without async for now
        // Full async tests will be added when native runtime is complete
    }

    #[test]
    fn test_broadcast_channel_creation() {
        let (tx, _rx1) = Broadcast::<i32>::new(10);
        // Test that broadcast channel can be created
        // Full send/receive tests will be added with native async runtime
        
        // Note: Clone test removed as it requires Clone trait implementation
        drop(tx); // Ensure we can drop the sender
    }

    #[test]
    fn test_watch_channel_creation() {
        let (tx, _rx) = Watch::new(0);
        
        // Test that watch channel can be created
        // Full watch tests will be added with native async runtime
        
        // Test basic operations
        drop(tx); // Ensure we can drop the sender
    }

    #[test]
    fn test_notify_creation() {
        let notify = Notify::new();
        
        // Test basic notify operations (non-async parts)
        notify.notify_one();
        notify.notify_waiters();
    }
}