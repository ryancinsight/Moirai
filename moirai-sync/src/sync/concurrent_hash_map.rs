use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::Mutex;

// Import centralized constants (SSOT compliance)
use moirai_core::constants::DEFAULT_CONCURRENT_MAP_SEGMENTS;

/// Concurrent hash map with segment-based locking for scalability.
/// This provides better scalability than a single mutex-protected HashMap.
pub struct ConcurrentHashMap<K, V, S = RandomState> {
    pub(crate) segments: Vec<Mutex<HashMap<K, V, S>>>,
    pub(crate) hasher: S,
}

impl<K: Hash + Eq, V> ConcurrentHashMap<K, V> {
    /// Create a new concurrent hash map with default hasher.
    pub fn new() -> Self {
        Self::with_segments(DEFAULT_CONCURRENT_MAP_SEGMENTS)
    }

    /// Create with a specific number of segments (must be power of 2).
    pub fn with_segments(num_segments: usize) -> Self {
        let num_segments = num_segments.next_power_of_two();

        let segments = (0..num_segments)
            .map(|_| Mutex::new(HashMap::new()))
            .collect();

        Self {
            segments,
            hasher: RandomState::new(),
        }
    }
}

impl<K: Hash + Eq, V> Default for ConcurrentHashMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq, V, S: BuildHasher> ConcurrentHashMap<K, V, S> {
    /// Get the segment index for a key.
    pub(crate) fn segment_index(&self, key: &K) -> usize {
        let mut hasher = self.hasher.build_hasher();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        // Use bitmask for even distribution across power-of-2 segments
        (hash as usize) & (self.segments.len() - 1)
    }

    /// Insert a key-value pair.
    ///
    /// Returns the previous value if the key existed, or None if it was a new key.
    /// Uses Result to handle potential poisoned mutex errors.
    pub fn insert(&self, key: K, value: V) -> Result<Option<V>, String> {
        let idx = self.segment_index(&key);
        Ok(self.segments[idx]
            .lock()
            .map_err(|_| "Mutex poisoned".to_string())?
            .insert(key, value))
    }

    /// Get a value by key.
    ///
    /// Returns the cloned value if found, or None if not found.
    /// Uses Result to handle potential poisoned mutex errors.
    pub fn get(&self, key: &K) -> Result<Option<V>, String>
    where
        V: Clone,
    {
        let idx = self.segment_index(key);
        Ok(self.segments[idx]
            .lock()
            .map_err(|_| "Mutex poisoned".to_string())?
            .get(key)
            .cloned())
    }

    /// Remove a key-value pair.
    ///
    /// Returns the removed value if the key existed, or None if it didn't exist.
    /// Uses Result to handle potential poisoned mutex errors.
    pub fn remove(&self, key: &K) -> Result<Option<V>, String> {
        let idx = self.segment_index(key);
        Ok(self.segments[idx]
            .lock()
            .map_err(|_| "Mutex poisoned".to_string())?
            .remove(key))
    }

    /// Check if a key exists.
    ///
    /// Returns true if the key exists, false otherwise.
    /// Uses Result to handle potential poisoned mutex errors.
    pub fn contains_key(&self, key: &K) -> Result<bool, String> {
        let idx = self.segment_index(key);
        Ok(self.segments[idx]
            .lock()
            .map_err(|_| "Mutex poisoned".to_string())?
            .contains_key(key))
    }
}
