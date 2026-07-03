//! Advanced async synchronization primitives for Moirai
//!
//! This module provides async-aware synchronization that integrates with
//! Moirai's unified runtime. Following SLAP principle, each synchronization
//! primitive is implemented in its own focused module.

pub mod broadcast;
pub mod notify;
pub mod rwlock;
pub mod semaphore;
pub(crate) mod wait_queue;
pub mod watch;

// Re-export public types for convenience
pub use broadcast::{Broadcast, BroadcastError, BroadcastReceiver, BroadcastRecv, BroadcastSender};
pub use notify::{Notify, NotifyFuture};
pub use rwlock::{RwLock, RwLockReadFuture, RwLockWriteFuture};
pub use semaphore::{Semaphore, SemaphoreAcquire, SemaphorePermit};
pub use watch::{Watch, WatchChanged, WatchError, WatchReceiver, WatchSender};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::AsyncExecutor;
    use std::future::Future;
    use std::sync::Arc;
    use std::task::{Context, Poll};
    use std::time::Duration;

    fn run_executor_to_completion_with_limit(executor: &AsyncExecutor, limit: usize) {
        for _ in 0..limit {
            executor.process_pending_tasks();
            executor
                .reactor()
                .run_iteration(Some(Duration::from_millis(0)))
                .ok();
            if executor.stats().tasks_pending == 0 {
                break;
            }
        }
    }

    #[test]
    fn test_semaphore_basic() {
        let sem = Semaphore::new(2);
        assert_eq!(sem.available_permits(), 2);
    }

    #[test]
    fn test_semaphore_async() {
        let executor = AsyncExecutor::new().unwrap();
        let sem = Arc::new(Semaphore::new(2));
        let notify = Arc::new(Notify::new());

        let sem_clone1 = sem.clone();
        let notify_clone1 = notify.clone();
        let handle1 = executor.spawn(async move {
            let _permit = sem_clone1.acquire().await;
            notify_clone1.notified().await;
            1
        });

        let sem_clone2 = sem.clone();
        let notify_clone2 = notify.clone();
        let handle2 = executor.spawn(async move {
            let _permit = sem_clone2.acquire().await;
            notify_clone2.notified().await;
            2
        });

        let sem_clone3 = sem.clone();
        let notify_clone3 = notify.clone();
        let handle3 = executor.spawn(async move {
            let _permit = sem_clone3.acquire().await;
            notify_clone3.notified().await;
            3
        });

        run_executor_to_completion_with_limit(&executor, 100);

        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);

        let mut handle1 = Box::pin(handle1);
        let mut handle2 = Box::pin(handle2);
        let mut handle3 = Box::pin(handle3);

        assert!(handle1.as_mut().poll(&mut context).is_pending());
        assert!(handle2.as_mut().poll(&mut context).is_pending());
        assert!(handle3.as_mut().poll(&mut context).is_pending());

        // Notify one task to let it complete and release its permit
        notify.notify_one();

        run_executor_to_completion_with_limit(&executor, 100);

        let p1 = handle1.as_mut().poll(&mut context);
        let p2 = handle2.as_mut().poll(&mut context);
        let p3 = handle3.as_mut().poll(&mut context);

        let r1 = p1.is_ready();
        let r2 = p2.is_ready();
        assert_eq!(if r1 { 1 } else { 0 } + if r2 { 1 } else { 0 }, 1);
        assert!(p3.is_pending());

        // Notify all remaining tasks
        notify.notify_waiters();

        run_executor_to_completion_with_limit(&executor, 100);

        if !r1 {
            assert!(handle1.as_mut().poll(&mut context).is_ready());
        }
        if !r2 {
            assert!(handle2.as_mut().poll(&mut context).is_ready());
        }
        assert!(handle3.as_mut().poll(&mut context).is_ready());
    }

    #[test]
    fn test_notify_async() {
        let executor = AsyncExecutor::new().unwrap();
        let notify = Arc::new(Notify::new());

        let notify_clone = notify.clone();
        let handle = executor.spawn(async move {
            notify_clone.notified().await;
            42
        });

        let mut handle = Box::pin(handle);
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);

        run_executor_to_completion_with_limit(&executor, 5);
        assert!(handle.as_mut().poll(&mut context).is_pending());

        notify.notify_one();

        run_executor_to_completion_with_limit(&executor, 5);
        assert!(matches!(
            handle.as_mut().poll(&mut context),
            Poll::Ready(42)
        ));
    }

    #[test]
    fn test_rwlock_async() {
        let executor = AsyncExecutor::new().unwrap();
        let lock = Arc::new(RwLock::new(100));

        let lock_clone1 = lock.clone();
        let handle_read1 = executor.spawn(async move {
            let guard = lock_clone1.read().await;
            *guard
        });

        let lock_clone2 = lock.clone();
        let handle_read2 = executor.spawn(async move {
            let guard = lock_clone2.read().await;
            *guard
        });

        run_executor_to_completion_with_limit(&executor, 5);

        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        let mut h_r1 = Box::pin(handle_read1);
        let mut h_r2 = Box::pin(handle_read2);

        assert!(matches!(h_r1.as_mut().poll(&mut context), Poll::Ready(100)));
        assert!(matches!(h_r2.as_mut().poll(&mut context), Poll::Ready(100)));

        // Drops the read handles, unlocking RwLock
        drop(h_r1);
        drop(h_r2);

        let lock_clone3 = lock.clone();
        let handle_write = executor.spawn(async move {
            let mut guard = lock_clone3.write().await;
            *guard += 50;
            *guard
        });

        run_executor_to_completion_with_limit(&executor, 5);
        let mut h_w = Box::pin(handle_write);
        assert!(matches!(h_w.as_mut().poll(&mut context), Poll::Ready(150)));
    }

    #[test]
    fn test_watch_async() {
        let executor = AsyncExecutor::new().unwrap();
        let (tx, rx) = Watch::new(10);

        let tx = Arc::new(tx);
        let rx_clone = rx.clone();
        let handle = executor.spawn(async move {
            let mut rx = rx_clone;
            rx.changed().await.unwrap();
            rx.borrow()
        });

        run_executor_to_completion_with_limit(&executor, 5);
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        let mut handle = Box::pin(handle);
        assert!(handle.as_mut().poll(&mut context).is_pending());

        tx.send(20).unwrap();

        run_executor_to_completion_with_limit(&executor, 5);
        assert!(matches!(
            handle.as_mut().poll(&mut context),
            Poll::Ready(20)
        ));
    }

    #[test]
    fn test_broadcast_async() {
        let executor = AsyncExecutor::new().unwrap();
        let (tx, rx1) = Broadcast::new(2);
        let rx2 = rx1.clone();

        let handle1 = executor.spawn(async move {
            let mut rx = rx1;
            let m1 = rx.recv().await.unwrap();
            let m2 = rx.recv().await.unwrap();
            (m1, m2)
        });

        let handle2 = executor.spawn(async move {
            let mut rx = rx2;
            let m1 = rx.recv().await.unwrap();
            let m2 = rx.recv().await.unwrap();
            (m1, m2)
        });

        run_executor_to_completion_with_limit(&executor, 5);

        tx.send(10).unwrap();
        tx.send(20).unwrap();

        run_executor_to_completion_with_limit(&executor, 10);

        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        let mut h1 = Box::pin(handle1);
        let mut h2 = Box::pin(handle2);

        assert!(matches!(
            h1.as_mut().poll(&mut context),
            Poll::Ready((10, 20))
        ));
        assert!(matches!(
            h2.as_mut().poll(&mut context),
            Poll::Ready((10, 20))
        ));
    }

    #[test]
    fn test_broadcast_waker_registration() {
        let (tx, mut rx) = Broadcast::new(2);
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);

        // Initially empty, should return Pending and register the waker
        assert!(rx.poll_recv(&mut context).is_pending());

        // Send a message
        tx.send(42).unwrap();

        // Now it should return Ready(Ok(42))
        assert!(matches!(rx.poll_recv(&mut context), Poll::Ready(Ok(42))));
    }

    #[test]
    fn test_notify_cancellation_safety() {
        use crate::sync::Notify;
        use futures::task::noop_waker;
        use std::future::Future;
        use std::task::Context;

        let notify = Notify::new();
        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);

        let mut f1 = Box::pin(notify.notified());
        let mut f2 = Box::pin(notify.notified());

        // Poll both to register them as waiters
        assert!(f1.as_mut().poll(&mut context).is_pending());
        assert!(f2.as_mut().poll(&mut context).is_pending());

        // Notify once, which grants permit to f1
        notify.notify_one();

        // Drop f1 before it is polled. This should transfer the permit to f2.
        drop(f1);

        // Now f2 should be ready
        assert!(f2.as_mut().poll(&mut context).is_ready());

        // Second case: no other pending waiters, permit is restored to state
        let notify = Notify::new();
        let mut f1 = Box::pin(notify.notified());

        // Poll to register f1 as a waiter
        assert!(f1.as_mut().poll(&mut context).is_pending());

        // Notify once, granting permit to f1
        notify.notify_one();

        // Drop f1. This should restore the permit back to notify's state.
        drop(f1);

        // Now a new future should be ready immediately
        let mut f3 = Box::pin(notify.notified());
        assert!(f3.as_mut().poll(&mut context).is_ready());
    }

    #[test]
    fn notify_waiters_preserves_stored_notify_one_permit() {
        // A `notify_one` issued with no waiters stores a single permit. An
        // unrelated `notify_waiters` (which only wakes currently-registered
        // waiters) must not destroy that stored permit, or the next
        // `notified()` would block forever.
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);

        let notify = Notify::new();
        notify.notify_one(); // store a permit (no waiters registered)
        notify.notify_waiters(); // must leave the stored permit intact

        let mut fut = Box::pin(notify.notified());
        assert!(
            fut.as_mut().poll(&mut context).is_ready(),
            "notify_one permit must survive notify_waiters when no waiters are registered"
        );
    }
}
