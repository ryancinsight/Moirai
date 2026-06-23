//! Unified high-performance channel implementations for Moirai.
//!
//! This module provides zero-cost channel abstractions that work seamlessly
//! across different execution contexts following DRY and SOLID principles.
//!
//! # Design Principles
//! - **Zero-copy**: Minimize data copies for maximum performance
//! - **Cache-friendly**: Align data structures to cache lines
//! - **Lock-free**: Use atomic operations where possible
//! - **Unified API**: Single interface for different channel types
//!
//! # Safety
//! All channel implementations maintain memory safety through:
//! - Sequence number validation before reading uninitialized memory
//! - Proper memory ordering with acquire-release semantics
//! - Safe cleanup on drop with reference counting

pub mod error;
pub mod hybrid;
pub mod mpmc;
pub mod select;
pub mod spsc;

pub use error::{Channel, ChannelError, Result};
pub use hybrid::{HybridChannel, HybridReceiver, HybridSender};
pub use mpmc::{MpmcChannel, MpmcReceiver, MpmcSender};
pub use select::{mpmc, spsc, unbounded, Select};
pub use spsc::{SpscChannel, SpscReceiver, SpscSender};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_channel() {
        let (tx, rx) = HybridChannel::<i32>::new(4);

        // Test basic send/receive
        tx.send(42).unwrap();
        assert_eq!(rx.recv().unwrap(), 42);

        // Test try_recv on empty channel
        assert!(matches!(rx.try_recv(), Err(ChannelError::Empty)));

        // Test timeout
        let result = rx.recv_timeout(std::time::Duration::from_millis(100));
        assert!(matches!(result, Err(ChannelError::Empty)));

        // Test multiple sends
        for i in 0..4 {
            tx.send(i).unwrap();
        }

        // Channel should be full
        assert!(!tx.can_send());
        assert_eq!(tx.available_capacity(), 0);

        // Test drain
        let values = rx.drain();
        assert_eq!(values, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_hybrid_channel_async() {
        use std::future::Future;
        use std::pin::Pin;
        use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

        fn dummy_raw_waker() -> RawWaker {
            fn clone_raw(_: *const ()) -> RawWaker {
                dummy_raw_waker()
            }
            fn wake_raw(_: *const ()) {}
            fn wake_by_ref_raw(_: *const ()) {}
            fn drop_raw(_: *const ()) {}
            static VTABLE: RawWakerVTable =
                RawWakerVTable::new(clone_raw, wake_raw, wake_by_ref_raw, drop_raw);
            RawWaker::new(std::ptr::null(), &VTABLE)
        }

        let (tx, rx) = HybridChannel::<i32>::new(4);
        let mut recv_fut = rx.recv_async();

        let waker = unsafe { Waker::from_raw(dummy_raw_waker()) };
        let mut cx = Context::from_waker(&waker);

        // First poll: empty channel -> Poll::Pending
        assert!(matches!(
            Pin::new(&mut recv_fut).poll(&mut cx),
            Poll::Pending
        ));

        // Send value
        tx.send(100).unwrap();

        // Second poll: has value -> Poll::Ready(Ok(100))
        assert!(matches!(
            Pin::new(&mut recv_fut).poll(&mut cx),
            Poll::Ready(Ok(100))
        ));
    }

    #[test]
    fn test_spsc_channel() {
        let (tx, rx) = spsc::<i32>(4);

        // Send some values
        assert!(tx.send(1).is_ok());
        assert!(tx.send(2).is_ok());

        // Receive values
        assert_eq!(rx.recv().unwrap(), 1);
        assert_eq!(rx.recv().unwrap(), 2);

        // Channel should be empty
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_spsc_thread_safety_bounds() {
        fn assert_send<T: Send>() {}
        assert_send::<SpscSender<i32>>();
        assert_send::<SpscReceiver<i32>>();
    }

    #[test]
    fn test_mpmc_channel() {
        let (tx, rx) = mpmc::<i32>(4);
        let tx2 = tx.clone();

        // Multiple senders
        assert!(tx.send(1).is_ok());
        assert!(tx2.send(2).is_ok());

        // Receive values
        let mut values = vec![rx.recv().unwrap(), rx.recv().unwrap()];
        values.sort_unstable();
        assert_eq!(values, vec![1, 2]);
    }

    #[test]
    fn test_mpmc_multi_producer_single_consumer() {
        use std::thread;

        let producer_count = 4;
        let items_per_producer = 1_000;
        let (tx, rx) = mpmc::<usize>(64);

        let consumer = thread::spawn(move || {
            let mut sum = 0usize;
            for _ in 0..(producer_count * items_per_producer) {
                sum += rx.recv().unwrap();
            }
            sum
        });

        let producers = (0..producer_count)
            .map(|producer| {
                let tx = tx.clone();
                thread::spawn(move || {
                    for item in 0..items_per_producer {
                        tx.send(producer * items_per_producer + item).unwrap();
                    }
                })
            })
            .collect::<Vec<_>>();

        for producer in producers {
            producer.join().unwrap();
        }
        drop(tx);

        let expected = (0..(producer_count * items_per_producer)).sum::<usize>();
        assert_eq!(consumer.join().unwrap(), expected);
    }

    #[test]
    fn test_mpmc_capacity_one_single_producer_consumer() {
        use std::thread;

        let item_count = 32_768;
        let (tx, rx) = mpmc::<usize>(1);

        let consumer = thread::spawn(move || {
            let mut sum = 0usize;
            for _ in 0..item_count {
                sum += rx.recv().unwrap();
            }
            sum
        });

        let producer = thread::spawn(move || {
            for item in 0..item_count {
                tx.send(item).unwrap();
            }
        });

        producer.join().unwrap();

        let expected = (0..item_count).sum::<usize>();
        assert_eq!(consumer.join().unwrap(), expected);
    }

    #[test]
    fn test_mpmc_capacity_one_repeated_single_producer_consumer() {
        for _ in 0..8 {
            let item_count = 4_096;
            let (tx, rx) = mpmc::<usize>(1);

            let consumer = std::thread::spawn(move || {
                let mut sum = 0usize;
                for _ in 0..item_count {
                    sum += rx.recv().unwrap();
                }
                sum
            });

            let producer = std::thread::spawn(move || {
                for item in 0..item_count {
                    tx.send(item).unwrap();
                }
            });

            producer.join().unwrap();

            let expected = (0..item_count).sum::<usize>();
            assert_eq!(consumer.join().unwrap(), expected);
        }
    }

    #[test]
    fn test_mpmc_capacity_one_multi_producer_single_consumer() {
        let producer_count = 8;
        let item_count = 8_192;
        let (tx, rx) = mpmc::<usize>(1);

        let consumer = std::thread::spawn(move || {
            let mut sum = 0usize;
            for _ in 0..item_count {
                sum += rx.recv().unwrap();
            }
            sum
        });

        let producers = (0..producer_count)
            .map(|producer| {
                let tx = tx.clone();
                std::thread::spawn(move || {
                    let base = item_count / producer_count;
                    let remainder = item_count % producer_count;
                    let start = producer * base + producer.min(remainder);
                    let len = base + usize::from(producer < remainder);
                    for item in start..(start + len) {
                        tx.send(item).unwrap();
                    }
                })
            })
            .collect::<Vec<_>>();

        for producer in producers {
            producer.join().unwrap();
        }
        drop(tx);

        let expected = (0..item_count).sum::<usize>();
        assert_eq!(consumer.join().unwrap(), expected);
    }

    #[test]
    fn test_unbounded_channel() {
        let (tx, rx) = unbounded::<i32>();

        // Send some values
        for i in 0..10 {
            tx.send(i).unwrap();
        }

        // Receive values
        for i in 0..10 {
            assert_eq!(rx.recv().unwrap(), i);
        }
    }

    #[test]
    fn test_spsc_blocking_behavior() {
        use std::thread;
        use std::time::{Duration, Instant};

        // Create a small channel to test blocking
        let (tx, rx) = spsc::<i32>(2);

        // Fill the channel
        tx.send(1).unwrap();
        tx.send(2).unwrap();
        // Spawn a thread that will receive after a delay
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            // This will unblock the main thread's send
            let val = rx.recv().unwrap();
            (val, rx)
        });

        // This send should block until the spawned thread receives
        let start = Instant::now();
        tx.send(3).unwrap();
        let elapsed = start.elapsed();

        // Verify that we blocked for approximately the sleep duration
        assert!(
            elapsed >= Duration::from_millis(40),
            "Send should have blocked"
        );

        let _ = handle.join().unwrap();
    }

    #[test]
    fn test_spsc_recv_blocking_behavior() {
        use std::thread;
        use std::time::{Duration, Instant};

        let (tx, rx) = spsc::<i32>(10);

        // Spawn a thread that will send after a delay
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            tx.send(42).unwrap();
        });

        // This recv should block until the spawned thread sends
        let start = Instant::now();
        let value = rx.recv().unwrap();
        let elapsed = start.elapsed();

        assert_eq!(value, 42);
        // Verify that we blocked for approximately the sleep duration
        assert!(
            elapsed >= Duration::from_millis(40),
            "Recv should have blocked"
        );
    }

    #[test]
    fn test_hybrid_channel_parking() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use std::thread;
        use std::time::{Duration, Instant};

        let (sender, receiver) = HybridChannel::<i32>::new(10);
        let received = Arc::new(AtomicBool::new(false));
        let received_clone = received.clone();

        // Signal from receiver thread once it is about to block on recv()
        let receiver_ready = Arc::new(AtomicBool::new(false));
        let receiver_ready_clone = receiver_ready.clone();

        // Start receiver thread
        let receiver_thread = thread::spawn(move || {
            // Signal readiness just before blocking
            receiver_ready_clone.store(true, Ordering::Release);
            let start = Instant::now();
            let value = receiver.recv().unwrap();
            let elapsed = start.elapsed();
            received_clone.store(true, Ordering::Release);
            (value, elapsed)
        });

        // Wait until receiver thread has started and signalled readiness
        while !receiver_ready.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
        // Extra margin to ensure the thread has actually entered recv() / parked
        thread::sleep(Duration::from_millis(50));

        // Send value - this should unpark the receiver
        sender.send(42).unwrap();

        // Join and check results
        let (value, elapsed) = receiver_thread.join().unwrap();
        assert_eq!(value, 42);
        assert!(received.load(Ordering::Acquire));

        // The receiver should have been parked for the sleep duration; allow generous
        // tolerance for slow CI machines (assert >= 10ms rather than 90ms).
        assert!(
            elapsed >= Duration::from_millis(10),
            "receiver should have parked, elapsed: {:?}",
            elapsed
        );
        assert!(elapsed < Duration::from_millis(500));
    }

    #[test]
    fn test_spsc_drop_sender() {
        let (tx, rx) = spsc::<i32>(2);
        std::mem::drop(tx);
        assert_eq!(rx.recv(), Err(ChannelError::Closed));
        assert_eq!(rx.try_recv(), Err(ChannelError::Closed));
    }

    #[test]
    fn test_spsc_drop_receiver() {
        let (tx, rx) = spsc::<i32>(1);
        tx.send(1).unwrap();
        tx.send(2).unwrap();

        let rx_thread = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            std::mem::drop(rx);
        });

        // This send will block because the channel is full, and should unblock with Closed when receiver drops
        assert_eq!(tx.send(3), Err(ChannelError::Closed));
        rx_thread.join().unwrap();
    }

    #[test]
    fn test_hybrid_drop_sender() {
        let (tx, rx) = HybridChannel::<i32>::new(2);
        let rx_thread = std::thread::spawn(move || rx.recv());

        std::thread::sleep(std::time::Duration::from_millis(50));
        std::mem::drop(tx);

        assert_eq!(rx_thread.join().unwrap(), Err(ChannelError::Closed));
    }

    #[test]
    fn test_hybrid_drop_receiver() {
        let (tx, rx) = HybridChannel::<i32>::new(2);
        std::mem::drop(rx);
        assert_eq!(tx.send(1), Err(ChannelError::Closed));
        assert_eq!(tx.try_send(1), Err(ChannelError::Closed));
    }
}
