//! Domain-based zero-copy message router.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use super::channel::{ZeroCopyChannel, ZeroCopyReceiver, ZeroCopySender};
use super::error::{ZeroCopyError, ZeroCopyResult};

/// Zero-copy message router with domain-based routing.
pub struct ZeroCopyRouter<T> {
    routes: Arc<RwLock<HashMap<DomainId, Arc<ZeroCopySender<T>>>>>,
    default_route: Option<Arc<ZeroCopySender<T>>>,
    stats: RouterStats,
}

/// Domain identifier for message routing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DomainId(u64);

impl DomainId {
    /// Synchronous execution domain
    pub const SYNC: Self = DomainId(0);
    /// Asynchronous execution domain
    pub const ASYNC: Self = DomainId(1);
    /// Parallel execution domain
    pub const PARALLEL: Self = DomainId(2);
    /// Distributed execution domain
    pub const DISTRIBUTED: Self = DomainId(3);

    /// Create a new domain identifier
    pub fn new(id: u64) -> Self {
        DomainId(id)
    }
}

#[derive(Debug, Default)]
struct RouterStats {
    messages_routed: AtomicUsize,
    routing_failures: AtomicUsize,
    zero_copy_sends: AtomicUsize,
}

impl<T: Send + 'static> ZeroCopyRouter<T> {
    /// Create a new zero-copy router
    pub fn new() -> Self {
        Self {
            routes: Arc::new(RwLock::new(HashMap::new())),
            default_route: None,
            stats: RouterStats::default(),
        }
    }

    /// Add a route for a specific domain
    pub fn add_route(
        &self,
        domain: DomainId,
        capacity: usize,
    ) -> ZeroCopyResult<ZeroCopyReceiver<T>> {
        let (s, r) = ZeroCopyChannel::new(capacity)?;
        self.routes.write().unwrap().insert(domain, Arc::new(s));
        Ok(r)
    }

    /// Set the default route for unspecified domains
    pub fn set_default_route(&mut self, capacity: usize) -> ZeroCopyResult<ZeroCopyReceiver<T>> {
        let (s, r) = ZeroCopyChannel::new(capacity)?;
        self.default_route = Some(Arc::new(s));
        Ok(r)
    }

    /// Route a message to the specified domain
    pub fn route(&self, domain: DomainId, message: T) -> Result<(), (T, ZeroCopyError)> {
        self.stats.messages_routed.fetch_add(1, Ordering::Relaxed);
        if let Some(ch) = self.routes.read().unwrap().get(&domain) {
            match ch.send(message) {
                Ok(()) => {
                    self.stats.zero_copy_sends.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
                Err((msg, e)) => {
                    self.stats.routing_failures.fetch_add(1, Ordering::Relaxed);
                    Err((msg, e))
                }
            }
        } else if let Some(def) = &self.default_route {
            def.send(message)
        } else {
            self.stats.routing_failures.fetch_add(1, Ordering::Relaxed);
            Err((message, ZeroCopyError::NoRoute))
        }
    }

    /// Get routing statistics (`messages_routed`, `routing_failures`, `zero_copy_sends`)
    pub fn stats(&self) -> (usize, usize, usize) {
        (
            self.stats.messages_routed.load(Ordering::Relaxed),
            self.stats.routing_failures.load(Ordering::Relaxed),
            self.stats.zero_copy_sends.load(Ordering::Relaxed),
        )
    }
}

impl<T: Send + 'static> Default for ZeroCopyRouter<T> {
    fn default() -> Self {
        Self::new()
    }
}
