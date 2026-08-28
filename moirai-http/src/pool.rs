//! Bounded idle-resource storage for HTTP connection reuse.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

struct Pooled<T> {
    value: T,
    idle_since: Instant,
}

/// Per-origin LIFO pool with bounded cardinality and access-triggered expiry.
pub(super) struct IdlePool<K, T> {
    entries: Mutex<HashMap<K, Vec<Pooled<T>>>>,
}

impl<K, T> Default for IdlePool<K, T> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }
}

impl<K, T> IdlePool<K, T>
where
    K: Eq + Hash + Clone,
{
    /// Return the newest resource whose idle age remains below `max_idle`.
    pub(super) fn take(&self, key: &K, max_idle: Duration) -> Option<T> {
        let now = Instant::now();
        let mut entries = self.lock_entries();
        let (selected, remove_bucket) = match entries.get_mut(key) {
            Some(bucket) => {
                let selected = loop {
                    let Some(pooled) = bucket.pop() else {
                        break None;
                    };
                    if now.saturating_duration_since(pooled.idle_since) < max_idle {
                        break Some(pooled.value);
                    }
                };
                (selected, bucket.is_empty())
            }
            None => (None, false),
        };
        if remove_bucket {
            entries.remove(key);
        }
        selected
    }

    /// Retain `value` only when the origin bucket is below `max_per_key`.
    pub(super) fn put(&self, key: &K, value: T, max_per_key: usize) {
        if max_per_key == 0 {
            return;
        }
        let mut entries = self.lock_entries();
        let bucket = entries.entry(key.clone()).or_default();
        if bucket.len() < max_per_key {
            bucket.push(Pooled {
                value,
                idle_since: Instant::now(),
            });
        }
    }

    fn lock_entries(&self) -> MutexGuard<'_, HashMap<K, Vec<Pooled<T>>>> {
        self.entries
            .lock()
            // A panic cannot invalidate HashMap or Vec structural invariants;
            // retain the bounded pool instead of making every later request panic.
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    #[test]
    fn capacity_is_per_key_and_zero_capacity_stores_nothing() {
        let pool = IdlePool::default();
        pool.put(&"a", 1, 2);
        pool.put(&"a", 2, 2);
        pool.put(&"a", 3, 2);
        pool.put(&"b", 4, 0);

        assert_eq!(pool.take(&"a", Duration::MAX), Some(2));
        assert_eq!(pool.take(&"a", Duration::MAX), Some(1));
        assert_eq!(pool.take(&"a", Duration::MAX), None);
        assert_eq!(pool.take(&"b", Duration::MAX), None);
    }

    #[test]
    fn zero_idle_duration_expires_without_waiting() {
        let pool = IdlePool::default();
        pool.put(&"origin", 7, 1);
        assert_eq!(pool.take(&"origin", Duration::ZERO), None);
    }

    #[test]
    fn poisoned_lock_recovers_structurally_valid_state() {
        let pool = IdlePool::<&str, usize>::default();
        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _guard = pool.entries.lock().expect("initial lock");
            panic!("poison pool for regression coverage");
        }));
        assert!(panic.is_err());

        pool.put(&"origin", 11, 1);
        assert_eq!(pool.take(&"origin", Duration::MAX), Some(11));
    }
}
