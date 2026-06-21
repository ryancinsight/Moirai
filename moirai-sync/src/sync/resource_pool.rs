use std::collections::{hash_map::DefaultHasher, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::sync::spin_lock::SpinLock;

/// A resource with a queryable byte size or capacity.
pub trait SizeBounded {
    /// The size or capacity of the resource in bytes.
    fn size(&self) -> u64;
}

/// Helper to map an arbitrary size to its power-of-two bin index.
#[inline]
fn bin_index(size: u64) -> usize {
    if size <= 1 {
        0
    } else {
        // Subtract 1 so that exact powers of two fall into the exact bin,
        // e.g. 1024 -> 10, 1025 -> 11.
        64 - (size - 1).leading_zeros() as usize
    }
}

struct Shard<T> {
    // 64 bins, representing power-of-two size classes (2^0 to 2^63).
    // SpinLock is cache-line aligned, preventing false sharing.
    bins: [SpinLock<VecDeque<T>>; 64],
    retained_bytes: AtomicU64,
    retained_count: AtomicUsize,
}

impl<T> Shard<T> {
    fn new() -> Self {
        let mut bins_vec = Vec::with_capacity(64);
        for _ in 0..64 {
            bins_vec.push(SpinLock::new(VecDeque::new()));
        }
        let bins: [SpinLock<VecDeque<T>>; 64] = bins_vec
            .try_into()
            .unwrap_or_else(|_| panic!("invariant: failed to convert vector of 64 bins"));

        Self {
            bins,
            retained_bytes: AtomicU64::new(0),
            retained_count: AtomicUsize::new(0),
        }
    }
}

/// A sharded, binned resource pool designed for high-concurrency reuse of transient allocations.
///
/// Resources are partitioned by thread affinity across 4 shards to minimize lock contention,
/// and internally binned into 64 power-of-two size classes. Pop operations use a non-blocking
/// stealing fallback across shards.
pub struct ShardedResourcePool<T> {
    shards: [Shard<T>; 4],
    shard_max_buffers: usize,
    shard_max_bytes: u64,
}

impl<T> std::fmt::Debug for ShardedResourcePool<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardedResourcePool")
            .field("shard_max_buffers", &self.shard_max_buffers)
            .field("shard_max_bytes", &self.shard_max_bytes)
            .finish_non_exhaustive()
    }
}

impl<T: SizeBounded> ShardedResourcePool<T> {
    /// Construct a new pool with the given capacity limits.
    #[must_use]
    pub fn new(max_buffers: usize, max_bytes: u64) -> Self {
        Self {
            shards: [
                Shard::new(),
                Shard::new(),
                Shard::new(),
                Shard::new(),
            ],
            shard_max_buffers: (max_buffers / 4).max(1),
            shard_max_bytes: max_bytes / 4,
        }
    }

    /// Retrieve the thread-local shard index.
    #[inline]
    fn get_shard_index() -> usize {
        thread_local! {
            static THREAD_SHARD_INDEX: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
        }
        THREAD_SHARD_INDEX.with(|cell| {
            if let Some(idx) = cell.get() {
                idx
            } else {
                let thread_id = std::thread::current().id();
                let mut hasher = DefaultHasher::new();
                thread_id.hash(&mut hasher);
                let idx = (hasher.finish() as usize) % 4;
                cell.set(Some(idx));
                idx
            }
        })
    }

    /// Retrieve a resource of size >= `size` from the pool, or return `None`.
    pub fn take_at_least(&self, size: u64) -> Option<T> {
        let local_idx = Self::get_shard_index();
        let start_bin = bin_index(size);

        // Try local shard first
        let local_shard = &self.shards[local_idx];
        
        // 1. Search the start_bin for a buffer >= size (since start_bin contains elements of varying sizes)
        {
            let mut guard = local_shard.bins[start_bin].lock();
            if let Some(pos) = guard.iter().rposition(|item| item.size() >= size) {
                let item = guard.remove(pos).expect("element exists at pos");
                let item_size = item.size();
                local_shard.retained_bytes.fetch_sub(item_size, Ordering::Release);
                local_shard.retained_count.fetch_sub(1, Ordering::Release);
                return Some(item);
            }
        }

        // 2. Search larger bins (all elements in larger bins are guaranteed to be >= size)
        for b in (start_bin + 1)..64 {
            let mut guard = local_shard.bins[b].lock();
            if let Some(item) = guard.pop_back() {
                let item_size = item.size();
                local_shard.retained_bytes.fetch_sub(item_size, Ordering::Release);
                local_shard.retained_count.fetch_sub(1, Ordering::Release);
                return Some(item);
            }
        }

        // Steal from other shards using non-blocking try_lock
        for i in 1..4 {
            let other_idx = (local_idx + i) % 4;
            let other_shard = &self.shards[other_idx];

            // 1. Search start_bin of other shard
            if let Some(mut guard) = other_shard.bins[start_bin].try_lock() {
                if let Some(pos) = guard.iter().rposition(|item| item.size() >= size) {
                    let item = guard.remove(pos).expect("element exists at pos");
                    let item_size = item.size();
                    other_shard.retained_bytes.fetch_sub(item_size, Ordering::Release);
                    other_shard.retained_count.fetch_sub(1, Ordering::Release);
                    return Some(item);
                }
            }

            // 2. Search larger bins of other shard
            for b in (start_bin + 1)..64 {
                if let Some(mut guard) = other_shard.bins[b].try_lock() {
                    if let Some(item) = guard.pop_back() {
                        let item_size = item.size();
                        other_shard.retained_bytes.fetch_sub(item_size, Ordering::Release);
                        other_shard.retained_count.fetch_sub(1, Ordering::Release);
                        return Some(item);
                    }
                }
            }
        }

        None
    }

    /// Recycle a resource back into the pool.
    pub fn recycle(&self, item: T) {
        let size = item.size();
        if size > self.shard_max_bytes || self.shard_max_buffers == 0 {
            return;
        }

        let local_idx = Self::get_shard_index();
        let local_shard = &self.shards[local_idx];
        let bin_idx = bin_index(size);

        // Evict oldest items if recycling would exceed limits
        let mut evicted = Vec::new();
        let mut current_bytes = local_shard.retained_bytes.load(Ordering::Acquire);
        let mut current_count = local_shard.retained_count.load(Ordering::Acquire);

        while current_count >= self.shard_max_buffers || current_bytes + size > self.shard_max_bytes {
            let mut progress = false;
            for b in 0..64 {
                if let Some(mut guard) = local_shard.bins[b].try_lock() {
                    if let Some(removed) = guard.pop_front() { // Evict oldest (FIFO)
                        let removed_size = removed.size();
                        local_shard.retained_bytes.fetch_sub(removed_size, Ordering::Release);
                        local_shard.retained_count.fetch_sub(1, Ordering::Release);
                        evicted.push(removed);
                        progress = true;
                        break;
                    }
                }
            }
            if !progress {
                break;
            }
            current_bytes = local_shard.retained_bytes.load(Ordering::Acquire);
            current_count = local_shard.retained_count.load(Ordering::Acquire);
        }

        drop(evicted);

        let mut guard = local_shard.bins[bin_idx].lock();
        local_shard.retained_bytes.fetch_add(size, Ordering::Release);
        local_shard.retained_count.fetch_add(1, Ordering::Release);
        guard.push_back(item);
    }

    /// Clear all pooled resources.
    pub fn clear(&self) {
        for shard in &self.shards {
            let mut evicted = Vec::new();
            for b in 0..64 {
                let mut guard = shard.bins[b].lock();
                evicted.extend(guard.drain(..));
            }
            shard.retained_bytes.store(0, Ordering::Release);
            shard.retained_count.store(0, Ordering::Release);
            drop(evicted);
        }
    }
}
