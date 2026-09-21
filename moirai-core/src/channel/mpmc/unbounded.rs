//! Unbounded path of the MPMC channel.
//!
//! A bounded channel keeps its items in the lock-free ring; a channel with no
//! capacity keeps them in the mutex-guarded `VecDeque` instead. Those two
//! disciplines share nothing but the state they read, so each keeps its own
//! module: the ring path and the notifier live in `channel.rs`, and the mutex
//! path lives here.

#![expect(
    clippy::unwrap_used,
    reason = "ratchet MOIRAI-UNWRAP-1: pre-existing debt"
)]

use super::MPMC_BLOCK_SPINS;
use super::channel::MpmcChannel;
use crate::channel::error::{ChannelError, Result};
use std::sync::atomic::Ordering;

impl<T: Send> MpmcChannel<T> {
    /// Send on the unbounded path, waiting on the not-full condvar while the
    /// channel is at capacity.
    pub(super) fn send_unbounded(&self, value: T) -> Result<()> {
        let (mutex, not_full, not_empty) = &self.state;
        let mut guard = mutex.lock().unwrap();
        let mut spin_count = 0;

        while !guard.closed && guard.capacity.is_some_and(|cap| guard.queue.len() >= cap) {
            if spin_count < MPMC_BLOCK_SPINS {
                drop(guard);
                for _ in 0..(1 << spin_count) {
                    std::hint::spin_loop();
                }
                spin_count += 1;
                guard = mutex.lock().unwrap();
            } else {
                self.sender_waiter_count.fetch_add(1, Ordering::AcqRel);
                guard = not_full.wait(guard).unwrap();
                self.sender_waiter_count.fetch_sub(1, Ordering::AcqRel);
            }
        }

        if guard.closed {
            return Err(ChannelError::Closed);
        }

        guard.queue.push_back(value);
        drop(guard);

        if self.receiver_waiter_count.load(Ordering::Acquire) > 0 {
            not_empty.notify_one();
        }
        Ok(())
    }

    /// Send without waiting; `Err(Full)` and `Err(Closed)` as for the ring path.
    pub(super) fn try_send_unbounded(&self, value: T) -> Result<()> {
        let (mutex, _, not_empty) = &self.state;
        let mut guard = mutex.lock().unwrap();

        if guard.closed {
            return Err(ChannelError::Closed);
        }

        if guard.capacity.is_some_and(|cap| guard.queue.len() >= cap) {
            return Err(ChannelError::Full);
        }

        guard.queue.push_back(value);
        drop(guard);

        if self.receiver_waiter_count.load(Ordering::Acquire) > 0 {
            not_empty.notify_one();
        }
        Ok(())
    }

    /// Receive on the unbounded path, waiting on the not-empty condvar while the
    /// queue is empty.
    pub(super) fn recv_unbounded(&self) -> Result<T> {
        let (mutex, not_full, not_empty) = &self.state;
        let mut guard = mutex.lock().unwrap();
        let mut spin_count = 0;

        while guard.queue.is_empty() && !guard.closed {
            if spin_count < MPMC_BLOCK_SPINS {
                drop(guard);
                for _ in 0..(1 << spin_count) {
                    std::hint::spin_loop();
                }
                spin_count += 1;
                guard = mutex.lock().unwrap();
            } else {
                self.receiver_waiter_count.fetch_add(1, Ordering::AcqRel);
                guard = not_empty.wait(guard).unwrap();
                self.receiver_waiter_count.fetch_sub(1, Ordering::AcqRel);
            }
        }

        match guard.queue.pop_front() {
            Some(value) => {
                drop(guard);

                if self.sender_waiter_count.load(Ordering::Acquire) > 0 {
                    not_full.notify_one();
                }
                Ok(value)
            }
            _ => Err(ChannelError::Closed),
        }
    }

    /// Receive without waiting; `Err(Empty)` and `Err(Closed)` as for the ring
    /// path.
    pub(super) fn try_recv_unbounded(&self) -> Result<T> {
        let (mutex, not_full, _) = &self.state;
        let mut guard = mutex.lock().unwrap();

        match guard.queue.pop_front() {
            Some(value) => {
                drop(guard);

                if self.sender_waiter_count.load(Ordering::Acquire) > 0 {
                    not_full.notify_one();
                }
                Ok(value)
            }
            _ => {
                if guard.closed {
                    Err(ChannelError::Closed)
                } else {
                    Err(ChannelError::Empty)
                }
            }
        }
    }

    /// Whether the unbounded queue holds nothing.
    pub(super) fn is_empty_unbounded(&self) -> bool {
        let (mutex, _, _) = &self.state;
        let guard = mutex.lock().unwrap();
        guard.queue.is_empty()
    }

    /// Whether the unbounded queue is at its configured capacity.
    pub(super) fn is_full_unbounded(&self) -> bool {
        let (mutex, _, _) = &self.state;
        let guard = mutex.lock().unwrap();
        guard.capacity.is_some_and(|cap| guard.queue.len() >= cap)
    }

    /// The unbounded channel's configured capacity.
    pub(super) fn capacity_unbounded(&self) -> Option<usize> {
        let (mutex, _, _) = &self.state;
        let guard = mutex.lock().unwrap();
        guard.capacity
    }
}
