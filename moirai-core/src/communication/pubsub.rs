use crate::channel::{mpmc, ChannelError, MpmcReceiver, MpmcSender};
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

/// Topic-based publish/subscribe system built on channels
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
        let (tx, rx) = mpmc(100);

        let mut subs = self.subscribers.write().unwrap();
        subs.entry(topic).or_insert_with(Vec::new).push(tx);

        rx
    }

    /// Publish a message to a topic
    pub fn publish(&self, topic: &K, message: V) -> Result<usize, ChannelError> {
        let subs = self.subscribers.read().unwrap();

        if let Some(subscribers) = subs.get(topic) {
            let mut sent = 0;
            for sub in subscribers {
                if sub.try_send(message.clone()).is_ok() {
                    sent += 1;
                }
            }
            Ok(sent)
        } else {
            Ok(0)
        }
    }

    /// Get the number of subscribers for a topic
    pub fn subscriber_count(&self, topic: &K) -> usize {
        let subs = self.subscribers.read().unwrap();
        subs.get(topic).map_or(0, |v| v.len())
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
