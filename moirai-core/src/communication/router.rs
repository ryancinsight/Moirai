use crate::channel::{ChannelError, MpmcSender};
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

/// Router for message-based communication patterns
pub struct MessageRouter<K: Hash + Eq + Clone, V: Send + 'static> {
    /// Routes mapped by key
    routes: Arc<RwLock<HashMap<K, MpmcSender<V>>>>,
}

impl<K: Hash + Eq + Clone, V: Send + 'static> MessageRouter<K, V> {
    /// Create a new message router
    pub fn new() -> Self {
        Self {
            routes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a route
    pub fn register(&self, key: K, sender: MpmcSender<V>) {
        let mut routes = self.routes.write().unwrap();
        routes.insert(key, sender);
    }

    /// Route a message to the appropriate channel
    pub fn route(&self, key: &K, message: V) -> Result<(), ChannelError> {
        let routes = self.routes.read().unwrap();

        if let Some(sender) = routes.get(key) {
            sender.try_send(message)
        } else {
            Err(ChannelError::Closed)
        }
    }

    /// Remove a route
    pub fn unregister(&self, key: &K) -> bool {
        let mut routes = self.routes.write().unwrap();
        routes.remove(key).is_some()
    }
}

impl<K: Hash + Eq + Clone, V: Send + 'static> Default for MessageRouter<K, V> {
    fn default() -> Self {
        Self::new()
    }
}
