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
    #[cfg(test)]
    test_hook: test_support::Hook,
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
            shards: [Shard::new(), Shard::new(), Shard::new(), Shard::new()],
            shard_max_buffers: (max_buffers / 4).max(1),
            shard_max_bytes: max_bytes / 4,
            #[cfg(test)]
            test_hook: test_support::Hook::new(),
        }
    }

    /// Retrieve the thread-local shard index.
    #[inline]
    fn get_shard_index() -> usize {
        thread_local! {
            // clippy 1.97.0 false positive: initialiser is already
            // `const { Cell::new(None) }`. Retire when toolchain advances
            // past the regression (ATLAS-MNEMOSYNE-CI-1).
            #[allow(clippy::missing_const_for_thread_local)]
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

        if local_shard.retained_count.load(Ordering::Acquire) > 0
            && local_shard.retained_bytes.load(Ordering::Acquire) >= size
        {
            // 1. Search the start_bin for a buffer >= size (since start_bin contains elements of varying sizes)
            {
                let mut guard = local_shard.bins[start_bin].lock();
                if let Some(pos) = guard.iter().rposition(|item| item.size() >= size) {
                    let item = guard.remove(pos).expect("element exists at pos");
                    let item_size = item.size();
                    local_shard
                        .retained_bytes
                        .fetch_sub(item_size, Ordering::Release);
                    local_shard.retained_count.fetch_sub(1, Ordering::Release);
                    return Some(item);
                }
            }

            // 2. Search larger bins (all elements in larger bins are guaranteed to be >= size)
            for b in (start_bin + 1)..64 {
                let mut guard = local_shard.bins[b].lock();
                if let Some(item) = guard.pop_back() {
                    let item_size = item.size();
                    local_shard
                        .retained_bytes
                        .fetch_sub(item_size, Ordering::Release);
                    local_shard.retained_count.fetch_sub(1, Ordering::Release);
                    return Some(item);
                }
            }
        }

        // Steal from other shards using non-blocking try_lock
        for i in 1..4 {
            let other_idx = (local_idx + i) % 4;
            let other_shard = &self.shards[other_idx];

            // Fast path check: if the other shard does not have any items or does not have enough bytes, skip it.
            if other_shard.retained_count.load(Ordering::Acquire) == 0
                || other_shard.retained_bytes.load(Ordering::Acquire) < size
            {
                continue;
            }

            // 1. Search start_bin of other shard
            if let Some(mut guard) = other_shard.bins[start_bin].try_lock() {
                if let Some(pos) = guard.iter().rposition(|item| item.size() >= size) {
                    let item = guard.remove(pos).expect("element exists at pos");
                    let item_size = item.size();
                    other_shard
                        .retained_bytes
                        .fetch_sub(item_size, Ordering::Release);
                    other_shard.retained_count.fetch_sub(1, Ordering::Release);
                    return Some(item);
                }
            }

            // 2. Search larger bins of other shard
            for b in (start_bin + 1)..64 {
                if let Some(mut guard) = other_shard.bins[b].try_lock() {
                    if let Some(item) = guard.pop_back() {
                        let item_size = item.size();
                        other_shard
                            .retained_bytes
                            .fetch_sub(item_size, Ordering::Release);
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

        // The target-bin guard covers reservation through publication. `clear`
        // acquires every bin guard before draining or resetting counters, so it
        // cannot publish a zero-counter state between these two mutations.
        let mut target_guard = local_shard.bins[bin_idx].lock();

        // Reserve this item's count and bytes up front, before inserting, so the
        // eviction decision below sees a total that already includes this item
        // *and* every other concurrent recycler's in-flight contribution. The
        // prior load-decide-insert sequence read the counters, decided no
        // eviction was needed, then inserted — allowing N concurrent recyclers to
        // each skip eviction and overshoot the shard cap by up to N-1 buffers
        // (and exceed the byte budget). `fetch_add` returns the pre-add value, so
        // `+ 1` / `+ size` is this shard's total with our reservation applied.
        let mut current_count = local_shard.retained_count.fetch_add(1, Ordering::AcqRel) + 1;
        let mut current_bytes = local_shard.retained_bytes.fetch_add(size, Ordering::AcqRel) + size;

        // Evict oldest items (FIFO) until the shard — counting our reserved item —
        // is within both limits, or no further eviction is possible. The local
        // `current_*` counters are decremented per eviction (rather than
        // re-loaded) so the loop terminates under sustained concurrent recycling
        // instead of chasing a moving atomic snapshot; a single item always fits
        // because `size <= shard_max_bytes` and `shard_max_buffers >= 1`.
        let mut evicted = Vec::new();
        while current_count > self.shard_max_buffers || current_bytes > self.shard_max_bytes {
            let mut progress = false;
            for b in 0..64 {
                if b == bin_idx {
                    if let Some(removed) = target_guard.pop_front() {
                        let removed_size = removed.size();
                        // Decrements remove already-inserted items, never our
                        // reservation, so the net total keeps counting our item.
                        local_shard.retained_count.fetch_sub(1, Ordering::Release);
                        local_shard
                            .retained_bytes
                            .fetch_sub(removed_size, Ordering::Release);
                        current_count -= 1;
                        current_bytes = current_bytes.saturating_sub(removed_size);
                        evicted.push(removed);
                        progress = true;
                        break;
                    }
                } else if let Some(mut guard) = local_shard.bins[b].try_lock() {
                    if let Some(removed) = guard.pop_front() {
                        let removed_size = removed.size();
                        // Decrements remove already-inserted items, never our
                        // reservation, so the net total keeps counting our item.
                        local_shard.retained_count.fetch_sub(1, Ordering::Release);
                        local_shard
                            .retained_bytes
                            .fetch_sub(removed_size, Ordering::Release);
                        current_count -= 1;
                        current_bytes = current_bytes.saturating_sub(removed_size);
                        evicted.push(removed);
                        progress = true;
                        break;
                    }
                }
            }
            if !progress {
                break;
            }
        }

        #[cfg(test)]
        self.test_hook.pause_after_reservation(local_idx, bin_idx);

        // The counters already account for this item (reserved above); inserting
        // it makes the bin contents consistent with the published totals.
        target_guard.push_back(item);
        drop(target_guard);
        drop(evicted);
    }

    /// Clear all pooled resources.
    ///
    /// All bin guards remain held until the bins are drained and the counters
    /// are reset. This makes the reset a linearization point: a concurrent
    /// `recycle` or `take_at_least` either completes before the reset or starts
    /// after it, and cannot publish a resource behind zero counters.
    pub fn clear(&self) {
        for (shard_idx, shard) in self.shards.iter().enumerate() {
            #[cfg(not(test))]
            let _ = shard_idx;
            let mut guards: [Option<_>; 64] = std::array::from_fn(|_| None);
            for (bin_idx, bin) in shard.bins.iter().enumerate() {
                #[cfg(test)]
                self.test_hook.announce_clear(shard_idx, bin_idx);
                guards[bin_idx] = Some(bin.lock());
            }

            let mut evicted = Vec::new();
            for guard in guards.iter_mut().flatten() {
                evicted.extend(guard.drain(..));
            }
            shard.retained_bytes.store(0, Ordering::Release);
            shard.retained_count.store(0, Ordering::Release);

            drop(guards);
            drop(evicted);
        }
    }

    #[cfg(test)]
    pub(crate) fn install_test_hook(
        &self,
        recycle_entered: std::sync::mpsc::SyncSender<()>,
        clear_started: std::sync::mpsc::SyncSender<()>,
        release: std::sync::Arc<std::sync::Barrier>,
    ) -> test_support::HookGuard {
        self.test_hook
            .install(recycle_entered, clear_started, release)
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::{mpsc::SyncSender, Arc, Barrier, Mutex};

    struct InterleavingHook {
        recycle_entered: SyncSender<()>,
        clear_started: SyncSender<()>,
        release: Arc<Barrier>,
        target: Option<(usize, usize)>,
        clear_announced: bool,
    }

    pub(crate) struct Hook {
        state: Arc<Mutex<Option<InterleavingHook>>>,
    }

    impl Hook {
        pub(crate) fn new() -> Self {
            Self {
                state: Arc::new(Mutex::new(None)),
            }
        }

        pub(crate) fn install(
            &self,
            recycle_entered: SyncSender<()>,
            clear_started: SyncSender<()>,
            release: Arc<Barrier>,
        ) -> HookGuard {
            let mut hook = self
                .state
                .lock()
                .expect("invariant: test hook mutex poisoned");
            assert!(
                hook.is_none(),
                "invariant: only one interleaving hook is active"
            );
            *hook = Some(InterleavingHook {
                recycle_entered,
                clear_started,
                release,
                target: None,
                clear_announced: false,
            });
            HookGuard {
                state: Arc::clone(&self.state),
            }
        }

        pub(crate) fn pause_after_reservation(&self, shard_idx: usize, bin_idx: usize) {
            let (entered, release) = {
                let mut hook = self
                    .state
                    .lock()
                    .expect("invariant: test hook mutex poisoned");
                let Some(hook) = hook.as_mut() else {
                    return;
                };
                assert!(
                    hook.target.replace((shard_idx, bin_idx)).is_none(),
                    "invariant: only one recycle interleaving is active"
                );
                (hook.recycle_entered.clone(), Arc::clone(&hook.release))
            };

            entered
                .send(())
                .expect("invariant: interleaving test receiver remains active");
            release.wait();
        }

        pub(crate) fn announce_clear(&self, shard_idx: usize, bin_idx: usize) {
            let started = {
                let mut hook = self
                    .state
                    .lock()
                    .expect("invariant: test hook mutex poisoned");
                let Some(hook) = hook.as_mut() else {
                    return;
                };
                if hook.target == Some((shard_idx, bin_idx)) && !hook.clear_announced {
                    hook.clear_announced = true;
                    Some(hook.clear_started.clone())
                } else {
                    None
                }
            };

            if let Some(started) = started {
                started
                    .send(())
                    .expect("invariant: interleaving test receiver remains active");
            }
        }
    }

    pub(crate) struct HookGuard {
        state: Arc<Mutex<Option<InterleavingHook>>>,
    }

    impl Drop for HookGuard {
        fn drop(&mut self) {
            self.state
                .lock()
                .expect("invariant: test hook mutex poisoned")
                .take();
        }
    }
}
