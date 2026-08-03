use crate::{lock_mutex, transport::Address, Transport, TransportResult};
use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, Mutex},
};

/// Remote address for cross-machine communication
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RemoteAddress {
    /// Remote host name or IP address.
    pub host: String,
    /// Remote TCP port.
    pub port: u16,
    /// Service label carried in the address display form.
    pub service: String,
}

impl fmt::Display for RemoteAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}://{}:{}", self.service, self.host, self.port)
    }
}

/// Topic-based pub/sub router that delivers published messages to every
/// subscribed [`Address`] over a shared transport.
///
/// The router is generic over the backing [`Transport`] so delivery is
/// monomorphized and zero-cost; the transport must be the *same instance* the
/// subscribers receive from (e.g. one `Arc<crate::InMemoryTransport>`), since
/// in-memory channels are keyed by address within a single transport instance.
/// The prior implementation constructed a throwaway `InMemoryTransport` per
/// send and so silently discarded every message.
pub struct MessageRouter<T: Transport> {
    transport: Arc<T>,
    subscriptions: Mutex<HashMap<String, Vec<Address>>>,
}

impl<T: Transport> MessageRouter<T> {
    /// Create a router that delivers over `transport`.
    pub fn new(transport: Arc<T>) -> Self {
        Self {
            transport,
            subscriptions: Mutex::new(HashMap::new()),
        }
    }

    /// Subscribe `address` to `topic`. Duplicate (topic, address) pairs are
    /// ignored so a message is delivered to each subscriber exactly once.
    pub fn subscribe(&self, topic: &str, address: Address) {
        let mut subs = lock_mutex(&self.subscriptions);
        let entry = subs.entry(topic.to_string()).or_default();
        if !entry.contains(&address) {
            entry.push(address);
        }
    }

    /// Remove `address` from `topic`. Returns `true` if a subscription was
    /// removed.
    pub fn unsubscribe(&self, topic: &str, address: &Address) -> bool {
        let mut subs = lock_mutex(&self.subscriptions);
        if let Some(entry) = subs.get_mut(topic) {
            let before = entry.len();
            entry.retain(|a| a != address);
            let removed = entry.len() != before;
            if entry.is_empty() {
                subs.remove(topic);
            }
            return removed;
        }
        false
    }

    /// Publish `data` to every subscriber of `topic` via the shared transport.
    ///
    /// Returns the number of subscribers the message was delivered to. Delivery
    /// is fail-fast: the first transport error is propagated (after the
    /// subscribers ahead of it have already received the message).
    ///
    /// # Errors
    /// Propagates the first per-subscriber transport send error.
    pub fn publish(&self, topic: &str, data: Vec<u8>) -> TransportResult<usize> {
        // Snapshot the subscriber list so the transport sends happen without the
        // subscriptions lock held (a subscriber's send must not block resubscribe).
        let targets: Vec<Address> = {
            let subs = lock_mutex(&self.subscriptions);
            match subs.get(topic) {
                Some(addresses) => addresses.clone(),
                None => return Ok(0),
            }
        };

        // `Transport::send` takes ownership (each in-memory subscriber channel
        // stores its own `Vec<u8>`), so N subscribers need N owned buffers —
        // but only N-1 copies: the caller's original buffer is moved to the
        // final subscriber instead of being cloned and dropped.
        let mut delivered = 0;
        let Some(last) = targets.len().checked_sub(1) else {
            return Ok(0);
        };
        let mut data = Some(data);
        for (index, addr) in targets.iter().enumerate() {
            let payload = if index == last {
                data.take()
                    .expect("invariant: original buffer moved exactly once, at the last subscriber")
            } else {
                data.as_ref()
                    .expect("invariant: original buffer present until the last subscriber")
                    .clone()
            };
            self.transport.send(addr, payload)?;
            delivered += 1;
        }
        Ok(delivered)
    }

    /// Number of distinct addresses subscribed to `topic`.
    pub fn subscriber_count(&self, topic: &str) -> usize {
        lock_mutex(&self.subscriptions)
            .get(topic)
            .map_or(0, Vec::len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InMemoryTransport, Transport};

    #[test]
    fn message_router_delivers_to_each_subscriber_once() {
        let transport = Arc::new(InMemoryTransport::new());
        let router = MessageRouter::new(Arc::clone(&transport));
        let sub_a = Address::Local("sub_a".to_string());
        let sub_b = Address::Local("sub_b".to_string());

        router.subscribe("topic", sub_a.clone());
        router.subscribe("topic", sub_b.clone());
        router.subscribe("topic", sub_a.clone()); // duplicate ignored
        assert_eq!(router.subscriber_count("topic"), 2);

        let delivered = router.publish("topic", vec![1, 2, 3]).unwrap();
        assert_eq!(delivered, 2);

        assert_eq!(transport.recv(&sub_a).unwrap(), vec![1, 2, 3]);
        assert_eq!(transport.recv(&sub_b).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn message_router_single_subscriber_receives_moved_buffer() {
        let transport = Arc::new(InMemoryTransport::new());
        let router = MessageRouter::new(Arc::clone(&transport));
        let sub = Address::Local("solo".to_string());
        router.subscribe("t", sub.clone());

        let delivered = router.publish("t", vec![7, 8, 9]).unwrap();
        assert_eq!(delivered, 1);
        assert_eq!(transport.recv(&sub).unwrap(), vec![7, 8, 9]);
    }

    #[test]
    fn message_router_unknown_topic_delivers_nothing() {
        let transport = Arc::new(InMemoryTransport::new());
        let router = MessageRouter::new(transport);
        assert_eq!(router.publish("absent", vec![0]).unwrap(), 0);
    }

    #[test]
    fn message_router_unsubscribe_stops_delivery() {
        let transport = Arc::new(InMemoryTransport::new());
        let router = MessageRouter::new(Arc::clone(&transport));
        let sub = Address::Local("s".to_string());

        router.subscribe("t", sub.clone());
        assert!(router.unsubscribe("t", &sub));
        assert!(
            !router.unsubscribe("t", &sub),
            "second unsubscribe is a no-op"
        );
        assert_eq!(router.subscriber_count("t"), 0);
        assert_eq!(router.publish("t", vec![9]).unwrap(), 0);
    }
}
