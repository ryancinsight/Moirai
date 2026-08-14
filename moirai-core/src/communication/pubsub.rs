#![expect(
    clippy::unwrap_used,
    reason = "ratchet MOIRAI-UNWRAP-1: pre-existing debt"
)]

use crate::channel::{mpmc, ChannelError, MpmcReceiver, MpmcSender};
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

/// Bounded per-subscriber queue depth. Messages published while a subscriber's
/// queue is full are counted as not delivered to that subscriber (`publish`
/// returns the delivered count). Local to this module: the value bounds
/// pub/sub fan-out memory, not general channel capacity.
const SUBSCRIBER_QUEUE_CAPACITY: usize = 100;

/// Topic-based publish/subscribe system built on channels.
///
/// Dropped subscribers are pruned lazily: a publish that observes a closed
/// subscriber channel removes that sender from the topic, so the subscriber
/// list cannot grow unboundedly under subscribe/drop churn.
pub struct PubSub<K: Hash + Eq + Clone, V: Clone + Send + 'static> {
    /// Subscribers mapped by topic
    subscribers: Arc<RwLock<HashMap<K, Vec<MpmcSender<V>>>>>,
}

impl<K: Hash + Eq + Clone, V: Clone + Send + 'static> PubSub<K, V> {
    /// Create a new pub/sub system
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Subscribe to a topic
    pub fn subscribe(&self, topic: K) -> MpmcReceiver<V> {
        let (tx, rx) = mpmc(SUBSCRIBER_QUEUE_CAPACITY);

        let mut subs = self.subscribers.write().unwrap();
        subs.entry(topic).or_default().push(tx);

        rx
    }

    /// Publish a message to a topic, returning the number of subscribers the
    /// message was delivered to.
    ///
    /// Senders whose receiver has been dropped are pruned from the topic
    /// during this call.
    pub fn publish(&self, topic: &K, message: V) -> Result<usize, ChannelError> {
        let mut sent = 0;
        let mut saw_closed = false;
        {
            let subs = self.subscribers.read().unwrap();
            if let Some(subscribers) = subs.get(topic) {
                for sub in subscribers {
                    match sub.try_send(message.clone()) {
                        Ok(()) => sent += 1,
                        Err(ChannelError::Closed) => saw_closed = true,
                        Err(_) => {} // Full: subscriber alive but backlogged
                    }
                }
            } else {
                return Ok(0);
            }
        }

        if saw_closed {
            let mut subs = self.subscribers.write().unwrap();
            if let Some(subscribers) = subs.get_mut(topic) {
                subscribers.retain(|sub| !sub.is_closed());
                if subscribers.is_empty() {
                    subs.remove(topic);
                }
            }
        }

        Ok(sent)
    }

    /// Get the number of subscribers for a topic
    pub fn subscriber_count(&self, topic: &K) -> usize {
        let subs = self.subscribers.read().unwrap();
        subs.get(topic).map_or(0, Vec::len)
    }
}

impl<K: Hash + Eq + Clone, V: Clone + Send + 'static> Default for PubSub<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq + Clone, V: Clone + Send + 'static> Clone for PubSub<K, V> {
    fn clone(&self) -> Self {
        Self {
            subscribers: self.subscribers.clone(),
        }
    }
}
