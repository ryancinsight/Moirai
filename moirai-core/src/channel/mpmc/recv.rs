use crate::channel::error::{Channel, Result};
use crate::channel::mpmc::MpmcChannel;
use std::sync::Arc;

/// Receiver half of MPMC channel
pub struct MpmcReceiver<T> {
    pub(super) channel: Arc<MpmcChannel<T>>,
}

impl<T: Send> MpmcReceiver<T> {
    /// Receive a value from the channel, blocking if necessary
    pub fn recv(&self) -> Result<T> {
        self.channel.recv()
    }

    /// Try to receive a value without blocking
    pub fn try_recv(&self) -> Result<T> {
        self.channel.try_recv()
    }
}

impl<T> Clone for MpmcReceiver<T> {
    fn clone(&self) -> Self {
        let (mutex, _, _) = &*self.channel.state;
        let mut guard = mutex.lock().unwrap();
        guard.receiver_count += 1;
        Self {
            channel: self.channel.clone(),
        }
    }
}

impl<T> Drop for MpmcReceiver<T> {
    fn drop(&mut self) {
        let (mutex, not_full, _) = &*self.channel.state;
        let mut guard = mutex.lock().unwrap();
        guard.receiver_count -= 1;
        if guard.receiver_count == 0 {
            guard.closed = true;
            self.channel
                .closed
                .store(true, std::sync::atomic::Ordering::Release);
            not_full.notify_all();
        }
    }
}
