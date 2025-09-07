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
    use std::time::Duration;

    #[test]
    fn test_semaphore_basic() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let sem = Semaphore::new(2);
            
            let permit1 = sem.acquire().await;
            let permit2 = sem.acquire().await;
            
            assert_eq!(sem.available_permits(), 0);
            assert!(sem.try_acquire().is_none());
            
            drop(permit1);
            assert_eq!(sem.available_permits(), 1);
            
            drop(permit2);
            assert_eq!(sem.available_permits(), 2);
        });
    }

    #[test]
    fn test_broadcast_channel() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let (tx, mut rx1) = Broadcast::new(10);
            let mut rx2 = rx1.resubscribe();
            
            tx.send("hello").unwrap();
            tx.send("world").unwrap();
            
            assert_eq!(rx1.recv().await.unwrap(), "hello");
            assert_eq!(rx1.recv().await.unwrap(), "world");
            
            assert_eq!(rx2.recv().await.unwrap(), "hello");
            assert_eq!(rx2.recv().await.unwrap(), "world");
        });
    }

    #[test]
    fn test_watch_channel() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let (tx, mut rx) = Watch::new(0);
            
            assert_eq!(rx.borrow(), 0);
            
            tx.send(42).unwrap();
            rx.changed().await.unwrap();
            assert_eq!(rx.borrow(), 42);
            
            tx.send_modify(|x| *x += 1).unwrap();
            rx.changed().await.unwrap();
            assert_eq!(rx.borrow(), 43);
        });
    }

    #[test]
    fn test_notify() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let notify = Notify::new();
            
            let mut notified = false;
            let future = async {
                notify.notified().await;
                notified = true;
            };
            
            // Future should not complete immediately
            tokio::select! {
                _ = future => panic!("Should not complete immediately"),
                _ = tokio::time::sleep(Duration::from_millis(10)) => {}
            }
            
            notify.notify_one();
            
            // Now it should complete
            tokio::select! {
                _ = future => {},
                _ = tokio::time::sleep(Duration::from_millis(100)) => panic!("Should have completed"),
            }
            
            assert!(notified);
        });
    }
}