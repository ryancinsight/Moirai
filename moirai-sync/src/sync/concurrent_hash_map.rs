use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::RwLock;

// Import centralized constants (SSOT compliance)
use moirai_core::constants::DEFAULT_CONCURRENT_MAP_SEGMENTS;

/// Error returned when a segment's `RwLock` was poisoned by a panicked writer.
///
/// Carries the index of the poisoned segment so diagnostics can name the
/// offending shard. This is a genuine contract failure (a writer panicked
/// while holding the segment lock), so it is surfaced as a typed error
/// rather than recovered silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentPoisoned {
    /// Index of the poisoned segment.
    pub segment: usize,
}

impl fmt::Display for SegmentPoisoned {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "concurrent map segment {} poisoned by a panicked writer",
            self.segment
        )
    }
}

impl std::error::Error for SegmentPoisoned {}

/// Concurrent hash map with segment-based locking for scalability.
/// This provides better scalability than a single mutex-protected HashMap.
pub struct ConcurrentHashMap<K, V, S = RandomState> {
    pub(crate) segments: Vec<RwLock<HashMap<K, V, S>>>,
    pub(crate) hasher: S,
}

impl<K, V, S> fmt::Debug for ConcurrentHashMap<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConcurrentHashMap")
            .field("segments_count", &self.segments.len())
            .finish_non_exhaustive()
    }
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
            .map(|_| RwLock::new(HashMap::new()))
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
    ///
    /// # Errors
    ///
    /// Returns [`SegmentPoisoned`] if the segment lock was poisoned by a
    /// panicked writer.
    pub fn insert(&self, key: K, value: V) -> Result<Option<V>, SegmentPoisoned> {
        let idx = self.segment_index(&key);
        Ok(self.segments[idx]
            .write()
            .map_err(|_| SegmentPoisoned { segment: idx })?
            .insert(key, value))
    }

    /// Get a value by key, or insert it if it is not present.
    ///
    /// Executes atomically under the segment's write lock, ensuring that the
    /// `default` closure runs exactly once on a cache miss and no concurrent insert
    /// can overwrite it.
    ///
    /// # Errors
    ///
    /// Returns [`SegmentPoisoned`] if the segment lock was poisoned by a
    /// panicked writer.
    pub fn get_or_insert_with<F>(&self, key: K, default: F) -> Result<V, SegmentPoisoned>
    where
        F: FnOnce() -> V,
        V: Clone,
    {
        let idx = self.segment_index(&key);
        // Phase 1: Fast-path with read lock
        {
            let shard = self.segments[idx]
                .read()
                .map_err(|_| SegmentPoisoned { segment: idx })?;
            if let Some(value) = shard.get(&key) {
                return Ok(value.clone());
            }
        }

        // Phase 2: Slow-path with write lock
        let mut shard = self.segments[idx]
            .write()
            .map_err(|_| SegmentPoisoned { segment: idx })?;
        if let Some(value) = shard.get(&key) {
            Ok(value.clone())
        } else {
            let value = default();
            shard.insert(key, value.clone());
            Ok(value)
        }
    }

    /// Get a value by key.
    ///
    /// Returns the cloned value if found, or None if not found.
    ///
    /// # Errors
    ///
    /// Returns [`SegmentPoisoned`] if the segment lock was poisoned by a
    /// panicked writer.
    pub fn get(&self, key: &K) -> Result<Option<V>, SegmentPoisoned>
    where
        V: Clone,
    {
        let idx = self.segment_index(key);
        Ok(self.segments[idx]
            .read()
            .map_err(|_| SegmentPoisoned { segment: idx })?
            .get(key)
            .cloned())
    }

    /// Remove a key-value pair.
    ///
    /// Returns the removed value if the key existed, or None if it didn't exist.
    ///
    /// # Errors
    ///
    /// Returns [`SegmentPoisoned`] if the segment lock was poisoned by a
    /// panicked writer.
    pub fn remove(&self, key: &K) -> Result<Option<V>, SegmentPoisoned> {
        let idx = self.segment_index(key);
        Ok(self.segments[idx]
            .write()
            .map_err(|_| SegmentPoisoned { segment: idx })?
            .remove(key))
    }

    /// Check if a key exists.
    ///
    /// Returns true if the key exists, false otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`SegmentPoisoned`] if the segment lock was poisoned by a
    /// panicked writer.
    pub fn contains_key(&self, key: &K) -> Result<bool, SegmentPoisoned> {
        let idx = self.segment_index(key);
        Ok(self.segments[idx]
            .read()
            .map_err(|_| SegmentPoisoned { segment: idx })?
            .contains_key(key))
    }
}
