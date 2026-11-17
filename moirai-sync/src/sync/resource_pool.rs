use std::collections::{VecDeque, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::sync::spin_lock::SpinLock;

const SHARD_COUNT: usize = 4;
const BIN_COUNT: usize = 64;

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
        // e.g. 1024 -> 10, 1025 -> 11. The final bin also contains sizes
        // above 2^63; exact-size filtering preserves retrieval semantics.
        usize::try_from(64 - (size - 1).leading_zeros())
            .expect("invariant: bin index fits in usize")
            .min(BIN_COUNT - 1)
    }
}

#[inline]
fn active_shard_count(max_buffers: usize, max_bytes: u64) -> usize {
    let byte_limited_shards = usize::try_from(max_bytes)
        .unwrap_or(SHARD_COUNT)
        .clamp(1, SHARD_COUNT);
    max_buffers.clamp(1, SHARD_COUNT).min(byte_limited_shards)
}

struct Shard<T> {
    // 64 bins representing power-of-two size classes. The final bin widens to
    // every size above 2^62 so the fixed layout covers the full u64 range.
    // SpinLock is cache-line aligned, preventing false sharing.
    bins: [SpinLock<VecDeque<T>>; BIN_COUNT],
    retained_bytes: AtomicU64,
    retained_count: AtomicUsize,
}

impl<T> Shard<T> {
    fn new() -> Self {
        Self {
            bins: std::array::from_fn(|_| SpinLock::new(VecDeque::new())),
            retained_bytes: AtomicU64::new(0),
            retained_count: AtomicUsize::new(0),
        }
    }

    fn try_reserve(&self, size: u64, max_buffers: usize, max_bytes: u64) -> bool {
        // Each update rejects before crossing its cap. If the byte reservation
        // fails, the count rollback removes the only partial reservation.
        if self
            .retained_count
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                count.checked_add(1).filter(|next| *next <= max_buffers)
            })
            .is_err()
        {
            return false;
        }

        if self
            .retained_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |bytes| {
                bytes.checked_add(size).filter(|next| *next <= max_bytes)
            })
            .is_err()
        {
            self.retained_count.fetch_sub(1, Ordering::Release);
            return false;
        }

        true
    }

    fn release(&self, size: u64) {
        self.retained_bytes.fetch_sub(size, Ordering::Release);
        self.retained_count.fetch_sub(1, Ordering::Release);
    }
}

/// A sharded, binned resource pool designed for high-concurrency reuse of transient allocations.
///
/// Resources are partitioned by thread affinity across up to four active shards to minimize lock
/// contention, and internally binned into 64 power-of-two size classes. The configured limits are
/// divided exactly across the active shards. Pop operations use a non-blocking stealing fallback
/// across shards. The final size class covers the full range through [`u64::MAX`].
pub struct ShardedResourcePool<T> {
    shards: [Shard<T>; SHARD_COUNT],
    max_buffers: usize,
    max_bytes: u64,
    #[cfg(test)]
    test_hook: test_support::Hook,
}

impl<T> std::fmt::Debug for ShardedResourcePool<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardedResourcePool")
            .field("max_buffers", &self.max_buffers)
            .field("max_bytes", &self.max_bytes)
            .field(
                "active_shards",
                &active_shard_count(self.max_buffers, self.max_bytes),
            )
            .finish_non_exhaustive()
    }
}

impl<T: SizeBounded> ShardedResourcePool<T> {
    /// Construct a new pool with the given capacity limits.
    #[must_use]
    pub fn new(max_buffers: usize, max_bytes: u64) -> Self {
        Self {
            shards: std::array::from_fn(|_| Shard::new()),
            max_buffers,
            max_bytes,
            #[cfg(test)]
            test_hook: test_support::Hook::new(),
        }
    }

    /// Retrieve the thread-local shard index.
    #[inline]
    fn get_shard_index(&self) -> usize {
        thread_local! {
            // clippy 1.97.0 false positive: initialiser is already
            // `const { Cell::new(None) }`. Retire when toolchain advances
            // past the regression (ATLAS-MNEMOSYNE-CI-1).
            #[allow(clippy::missing_const_for_thread_local)]
            static THREAD_HASH: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
        }
        THREAD_HASH.with(|cell| {
            let hash = if let Some(hash) = cell.get() {
                hash
            } else {
                let thread_id = std::thread::current().id();
                let mut hasher = DefaultHasher::new();
                thread_id.hash(&mut hasher);
                let hash = hasher.finish();
                cell.set(Some(hash));
                hash
            };
            let active_shards = u64::try_from(active_shard_count(self.max_buffers, self.max_bytes))
                .expect("invariant: active shard count fits in u64");
            usize::try_from(hash % active_shards)
                .expect("invariant: reduced shard index fits in usize")
        })
    }

    /// Retrieve a resource of size >= `size` from the pool, or return `None`.
    pub fn take_at_least(&self, size: u64) -> Option<T> {
        let local_idx = self.get_shard_index();
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
                    let item = guard
                        .remove(pos)
                        .expect("invariant: position came from the same guarded bin");
                    let item_size = item.size();
                    local_shard.release(item_size);
                    return Some(item);
                }
            }

            // 2. Search larger bins (all elements in larger bins are guaranteed to be >= size)
            for b in (start_bin + 1)..BIN_COUNT {
                let mut guard = local_shard.bins[b].lock();
                if let Some(item) = guard.pop_back() {
                    let item_size = item.size();
                    local_shard.release(item_size);
                    return Some(item);
                }
            }
        }

        // Steal from other shards using non-blocking try_lock
        let active_shards = active_shard_count(self.max_buffers, self.max_bytes);
        for i in 1..active_shards {
            let other_idx = (local_idx + i) % active_shards;
            let other_shard = &self.shards[other_idx];

            // Fast path check: if the other shard does not have any items or does not have enough bytes, skip it.
            if other_shard.retained_count.load(Ordering::Acquire) == 0
                || other_shard.retained_bytes.load(Ordering::Acquire) < size
            {
                continue;
            }

            // 1. Search start_bin of other shard
            if let Some(mut guard) = other_shard.bins[start_bin].try_lock()
                && let Some(pos) = guard.iter().rposition(|item| item.size() >= size)
            {
                let item = guard
                    .remove(pos)
                    .expect("invariant: position came from the same guarded bin");
                let item_size = item.size();
                other_shard.release(item_size);
                return Some(item);
            }

            // 2. Search larger bins of other shard
            for b in (start_bin + 1)..BIN_COUNT {
                if let Some(mut guard) = other_shard.bins[b].try_lock()
                    && let Some(item) = guard.pop_back()
                {
                    let item_size = item.size();
                    other_shard.release(item_size);
                    return Some(item);
                }
            }
        }

        None
    }

    /// Recycle a resource back into the pool.
    pub fn recycle(&self, item: T) {
        self.recycle_in_shard(item, self.get_shard_index());
    }

    fn recycle_in_shard(&self, item: T, local_idx: usize) {
        let active_shards = active_shard_count(self.max_buffers, self.max_bytes);
        let active_shards_u64 =
            u64::try_from(active_shards).expect("invariant: active shard count fits in u64");
        let local_idx_u64 = u64::try_from(local_idx).expect("invariant: shard index fits in u64");
        let shard_max_buffers = self.max_buffers / active_shards
            + usize::from(local_idx < self.max_buffers % active_shards);
        let shard_max_bytes = self.max_bytes / active_shards_u64
            + u64::from(local_idx_u64 < self.max_bytes % active_shards_u64);
        let size = item.size();
        if size > shard_max_bytes || shard_max_buffers == 0 {
            return;
        }

        let local_shard = &self.shards[local_idx];
        let bin_idx = bin_index(size);

        // The target-bin guard covers reservation through publication. `clear`
        // acquires every bin guard before draining or resetting counters, so it
        // cannot publish a zero-counter state between these two mutations.
        let mut target_guard = local_shard.bins[bin_idx].lock();

        let mut evicted = Vec::new();
        loop {
            if local_shard.try_reserve(size, shard_max_buffers, shard_max_bytes) {
                #[cfg(test)]
                self.test_hook.pause_after_reservation(local_idx, bin_idx);

                target_guard.push_back(item);
                drop(target_guard);
                drop(evicted);
                return;
            }

            let mut removed = None;
            // Locked bins may contain in-flight reservations. Skipping them
            // avoids a lock cycle; failure to find an evictable item rejects the
            // new item rather than exceeding either retention limit.
            for b in 0..BIN_COUNT {
                if b == bin_idx {
                    removed = target_guard.pop_front();
                } else if let Some(mut guard) = local_shard.bins[b].try_lock() {
                    removed = guard.pop_front();
                }
                if removed.is_some() {
                    break;
                }
            }

            let Some(removed) = removed else {
                drop(target_guard);
                drop(evicted);
                drop(item);
                return;
            };
            local_shard.release(removed.size());
            evicted.push(removed);
        }
    }

    /// Clear all pooled resources.
    ///
    /// All bin guards remain held until the bins are drained and the counters
    /// are reset. This makes the reset a linearization point: a concurrent
    /// `recycle` or `take_at_least` either completes before the reset or starts
    /// after it, and cannot publish a resource behind zero counters.
    pub fn clear(&self) {
        let active_shards = active_shard_count(self.max_buffers, self.max_bytes);
        for (shard_idx, shard) in self.shards.iter().take(active_shards).enumerate() {
            #[cfg(not(test))]
            let _ = shard_idx;
            let mut guards: [Option<_>; BIN_COUNT] = std::array::from_fn(|_| None);
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

    #[cfg(test)]
    pub(crate) fn recycle_on_shard(&self, item: T, shard_idx: usize) {
        assert!(
            shard_idx < active_shard_count(self.max_buffers, self.max_bytes),
            "invariant: test shard is active"
        );
        self.recycle_in_shard(item, shard_idx);
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::{Arc, Barrier, Mutex, mpsc::SyncSender};

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
                if hook.target.is_some() {
                    return;
                }
                hook.target = Some((shard_idx, bin_idx));
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
