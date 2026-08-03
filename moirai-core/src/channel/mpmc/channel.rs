use super::queue::BoundedMpmcQueue;
use super::recv::MpmcReceiver;
use super::send::MpmcSender;
use super::{MpmcState, MPMC_BLOCK_SPINS};
use crate::channel::error::{Channel, ChannelError, Result};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Multi-Producer Multi-Consumer channel with bounded capacity
/// Uses mutex-based implementation for simplicity and correctness
pub struct MpmcChannel<T> {
    pub(super) state: Arc<(Mutex<MpmcState<T>>, Condvar, Condvar)>,
    pub(super) bounded: Option<Arc<BoundedMpmcQueue<T>>>,
    pub(super) closed: Arc<AtomicBool>,
    pub(super) sender_waiter_count: Arc<AtomicUsize>,
    pub(super) receiver_waiter_count: Arc<AtomicUsize>,
}

impl<T> MpmcChannel<T> {
    /// Create a new MPMC channel with optional capacity
    pub fn new(capacity: Option<usize>) -> Self {
        let state = MpmcState {
            queue: if capacity.is_some() {
                VecDeque::new()
            } else {
                VecDeque::with_capacity(16)
            },
            capacity,
            closed: false,
            sender_count: 0,
            receiver_count: 0,
        };

        let bounded = capacity.map(BoundedMpmcQueue::new).map(Arc::new);

        Self {
            state: Arc::new((Mutex::new(state), Condvar::new(), Condvar::new())),
            bounded,
            closed: Arc::new(AtomicBool::new(false)),
            sender_waiter_count: Arc::new(AtomicUsize::new(0)),
            receiver_waiter_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Create an unbounded channel
    pub fn unbounded() -> Self {
        Self::new(None)
    }

    /// Create a bounded channel with given capacity
    pub fn bounded(capacity: usize) -> Self {
        Self::new(Some(capacity))
    }

    /// Create a channel pair for ergonomic usage
    pub fn channel(capacity: Option<usize>) -> (MpmcSender<T>, MpmcReceiver<T>) {
        let channel = Arc::new(Self::new(capacity));
        let (mutex, _, _) = &*channel.state;

        {
            let mut state = mutex.lock().unwrap();
            state.sender_count = 1;
            state.receiver_count = 1;
        }

        (
            MpmcSender {
                channel: channel.clone(),
            },
            MpmcReceiver { channel },
        )
    }

    fn send_bounded(&self, queue: &BoundedMpmcQueue<T>, mut value: T) -> Result<()>
    where
        T: Send,
    {
        let mut spin_count = 0;

        loop {
            if self.closed.load(Ordering::Acquire) {
                return Err(ChannelError::Closed);
            }

            match queue.try_push(value) {
                Ok(()) => {
                    if self.receiver_waiter_count.load(Ordering::SeqCst) > 0 {
                        let (mutex, _, not_empty) = &*self.state;
                        let _guard = mutex.lock().unwrap();
                        not_empty.notify_one();
                    }
                    return Ok(());
                }
                Err(returned) => {
                    value = returned;
                }
            }

            if spin_count < MPMC_BLOCK_SPINS {
                for _ in 0..(1 << spin_count) {
                    std::hint::spin_loop();
                }
                spin_count += 1;
                continue;
            }

            // Fallback to condvar wait to prevent CPU contention and busy-looping
            let (mutex, not_full, _) = &*self.state;
            let mut guard = mutex.lock().unwrap();

            if self.closed.load(Ordering::Acquire) || guard.closed {
                return Err(ChannelError::Closed);
            }

            self.sender_waiter_count.fetch_add(1, Ordering::SeqCst);

            match queue.try_push(value) {
                Ok(()) => {
                    self.sender_waiter_count.fetch_sub(1, Ordering::SeqCst);
                    drop(guard);
                    if self.receiver_waiter_count.load(Ordering::SeqCst) > 0 {
                        let (_, _, not_empty) = &*self.state;
                        not_empty.notify_one();
                    }
                    return Ok(());
                }
                Err(returned) => {
                    value = returned;
                }
            }

            guard = not_full.wait(guard).unwrap();
            self.sender_waiter_count.fetch_sub(1, Ordering::SeqCst);
        }
    }

    fn recv_bounded(&self, queue: &BoundedMpmcQueue<T>) -> Result<T>
    where
        T: Send,
    {
        let mut spin_count = 0;

        loop {
            if let Some(value) = queue.try_pop() {
                if self.sender_waiter_count.load(Ordering::SeqCst) > 0 {
                    let (mutex, not_full, _) = &*self.state;
                    let _guard = mutex.lock().unwrap();
                    not_full.notify_one();
                }
                return Ok(value);
            }

            if self.closed.load(Ordering::Acquire) {
                if queue.is_empty() {
                    return Err(ChannelError::Closed);
                }
                std::hint::spin_loop();
                continue;
            }

            if spin_count < MPMC_BLOCK_SPINS {
                for _ in 0..(1 << spin_count) {
                    std::hint::spin_loop();
                }
                spin_count += 1;
                continue;
            }

            // Fallback to condvar wait to prevent CPU contention and busy-looping
            let (mutex, _, not_empty) = &*self.state;
            let mut guard = mutex.lock().unwrap();

            if let Some(value) = queue.try_pop() {
                if self.sender_waiter_count.load(Ordering::SeqCst) > 0 {
                    let (_, not_full, _) = &*self.state;
                    not_full.notify_one();
                }
                drop(guard);
                return Ok(value);
            }

            if self.closed.load(Ordering::Acquire) || guard.closed {
                return Err(ChannelError::Closed);
            }

            self.receiver_waiter_count.fetch_add(1, Ordering::SeqCst);
            guard = not_empty.wait(guard).unwrap();
            self.receiver_waiter_count.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

impl<T: Send> crate::channel::roles::Producer<T> for MpmcChannel<T> {
    #[inline]
    fn send(&self, value: T) -> Result<()> {
        Channel::send(self, value)
    }

    #[inline]
    fn try_send(&self, value: T) -> Result<()> {
        Channel::try_send(self, value)
    }

    #[inline]
    fn is_full(&self) -> bool {
        Channel::is_full(self)
    }

    #[inline]
    fn capacity(&self) -> Option<usize> {
        Channel::capacity(self)
    }
}

impl<T: Send> crate::channel::roles::Consumer<T> for MpmcChannel<T> {
    #[inline]
    fn recv(&self) -> Result<T> {
        Channel::recv(self)
    }

    #[inline]
    fn try_recv(&self) -> Result<T> {
        Channel::try_recv(self)
    }

    #[inline]
    fn is_empty(&self) -> bool {
        Channel::is_empty(self)
    }
}

impl<T: Send> Channel<T> for MpmcChannel<T> {
    fn send(&self, value: T) -> Result<()> {
        if let Some(queue) = &self.bounded {
            return self.send_bounded(queue, value);
        }

        let (mutex, not_full, not_empty) = &*self.state;
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

    fn try_send(&self, value: T) -> Result<()> {
        if let Some(queue) = &self.bounded {
            if self.closed.load(Ordering::Acquire) {
                return Err(ChannelError::Closed);
            }
            queue.try_push(value).map_err(|_| ChannelError::Full)?;
            if self.receiver_waiter_count.load(Ordering::SeqCst) > 0 {
                let (mutex, _, not_empty) = &*self.state;
                let _guard = mutex.lock().unwrap();
                not_empty.notify_one();
            }
            return Ok(());
        }

        let (mutex, _, not_empty) = &*self.state;
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

    fn recv(&self) -> Result<T> {
        if let Some(queue) = &self.bounded {
            return self.recv_bounded(queue);
        }

        let (mutex, not_full, not_empty) = &*self.state;
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

        if let Some(value) = guard.queue.pop_front() {
            drop(guard);

            if self.sender_waiter_count.load(Ordering::Acquire) > 0 {
                not_full.notify_one();
            }
            Ok(value)
        } else {
            Err(ChannelError::Closed)
        }
    }

    fn try_recv(&self) -> Result<T> {
        if let Some(queue) = &self.bounded {
            if let Some(value) = queue.try_pop() {
                if self.sender_waiter_count.load(Ordering::SeqCst) > 0 {
                    let (mutex, not_full, _) = &*self.state;
                    let _guard = mutex.lock().unwrap();
                    not_full.notify_one();
                }
                return Ok(value);
            }
            if self.closed.load(Ordering::Acquire) {
                return Err(ChannelError::Closed);
            }
            return Err(ChannelError::Empty);
        }

        let (mutex, not_full, _) = &*self.state;
        let mut guard = mutex.lock().unwrap();

        if let Some(value) = guard.queue.pop_front() {
            drop(guard);

            if self.sender_waiter_count.load(Ordering::Acquire) > 0 {
                not_full.notify_one();
            }
            Ok(value)
        } else if guard.closed {
            Err(ChannelError::Closed)
        } else {
            Err(ChannelError::Empty)
        }
    }

    fn is_empty(&self) -> bool {
        if let Some(queue) = &self.bounded {
            return queue.is_empty();
        }

        let (mutex, _, _) = &*self.state;
        let guard = mutex.lock().unwrap();
        guard.queue.is_empty()
    }

    fn is_full(&self) -> bool {
        if let Some(queue) = &self.bounded {
            return queue.is_full();
        }

        let (mutex, _, _) = &*self.state;
        let guard = mutex.lock().unwrap();
        guard.capacity.is_some_and(|cap| guard.queue.len() >= cap)
    }

    fn capacity(&self) -> Option<usize> {
        if let Some(queue) = &self.bounded {
            return Some(queue.logical_capacity());
        }

        let (mutex, _, _) = &*self.state;
        let guard = mutex.lock().unwrap();
        guard.capacity
    }
}
