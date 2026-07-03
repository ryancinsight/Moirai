use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Latest-value watch cell for one-to-many communication.
///
/// Stores only the most recent broadcast value: every receiver observes the
/// latest value it has not yet seen (via [`BroadcastReceiver::try_recv`]),
/// but values broadcast while a receiver is not polling are overwritten, not
/// queued — this is watch-channel semantics, not a per-message broadcast
/// queue. A receiver that misses intermediate broadcasts sees only the newest
/// value.
pub struct BroadcastChannel<T: Clone> {
    /// Current value (protected by `RwLock` for concurrent access)
    value: Arc<RwLock<Option<T>>>,
    /// Version number for detecting updates
    version: Arc<AtomicUsize>,
    /// Number of active subscribers
    subscribers: Arc<AtomicUsize>,
}

impl<T: Clone> BroadcastChannel<T> {
    /// Create a new broadcast channel
    pub fn new() -> Self {
        Self {
            value: Arc::new(RwLock::new(None)),
            version: Arc::new(AtomicUsize::new(0)),
            subscribers: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Publish a value, replacing the previously stored one.
    pub fn broadcast(&self, value: T) {
        {
            let mut guard = self.value.write().unwrap();
            *guard = Some(value);
        }
        self.version.fetch_add(1, Ordering::Release);
    }

    /// Subscribe to broadcasts
    pub fn subscribe(&self) -> BroadcastReceiver<T> {
        self.subscribers.fetch_add(1, Ordering::Relaxed);
        BroadcastReceiver {
            channel: self.clone(),
            last_version: 0,
        }
    }

    /// Get the current number of subscribers
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.load(Ordering::Relaxed)
    }
}

impl<T: Clone> Default for BroadcastChannel<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> Clone for BroadcastChannel<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            version: self.version.clone(),
            subscribers: self.subscribers.clone(),
        }
    }
}

/// Receiver for broadcast channel
pub struct BroadcastReceiver<T: Clone> {
    channel: BroadcastChannel<T>,
    last_version: usize,
}

impl<T: Clone> BroadcastReceiver<T> {
    /// Return a clone of the latest value if it is newer than the last one
    /// this receiver observed; `None` when nothing new has been broadcast.
    pub fn try_recv(&mut self) -> Option<T> {
        let current_version = self.channel.version.load(Ordering::Acquire);

        if current_version > self.last_version {
            self.last_version = current_version;
            let guard = self.channel.value.read().unwrap();
            guard.clone()
        } else {
            None
        }
    }
}

impl<T: Clone> Drop for BroadcastReceiver<T> {
    fn drop(&mut self) {
        self.channel.subscribers.fetch_sub(1, Ordering::Relaxed);
    }
}
