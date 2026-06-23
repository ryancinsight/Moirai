//! Multi-word idle-worker bitset backing the lock-free wake lottery.
//!
//! The scheduler tracks which workers are parked so a producer can wake exactly
//! one of them. A single `AtomicU64` can only address 64 workers; a pool with
//! more workers than that would leave every worker with id >= 64 invisible to
//! the wake lottery, so high-index workers could only be reached via the
//! coarse fallback and would stay parked under load (a throughput stall on
//! large machines). This bitset spans `ceil(worker_count / 64)` cache-aligned
//! words so every worker is individually addressable regardless of pool size.
//!
//! All bit operations are `SeqCst` because each worker's `set` is one half of a
//! store-buffer (Dekker) handshake with the producer: a parking worker does
//! `set(id)` then loads `pending_tasks` (SeqCst), while a producer increments
//! `pending_tasks` (SeqCst) then scans this bitset (SeqCst). Sharing one SeqCst
//! total order across the per-word access and `pending_tasks` is what forbids
//! the lost-wakeup outcome where the worker observes no work and the producer
//! observes no idle worker.

use moirai_utils::cache::CacheAligned;
use std::sync::atomic::{AtomicU64, Ordering};

const BITS_PER_WORD: usize = 64;

/// A fixed-size set of parked-worker bits, one bit per worker id.
pub(super) struct IdleBitset {
    /// One cache-aligned word per 64 workers, preventing false sharing between
    /// the wake-lottery scan and a worker flipping its own bit.
    words: Box<[CacheAligned<AtomicU64>]>,
}

impl IdleBitset {
    /// Allocate a bitset large enough to address `worker_count` workers
    /// (always at least one word).
    pub(super) fn new(worker_count: usize) -> Self {
        let word_count = worker_count.div_ceil(BITS_PER_WORD).max(1);
        let words = (0..word_count)
            .map(|_| CacheAligned::new(AtomicU64::new(0)))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self { words }
    }

    /// Mark `id` as parked. SeqCst: producer half of the wake handshake.
    #[inline]
    pub(super) fn set(&self, id: usize) {
        let (word, bit) = (id / BITS_PER_WORD, 1u64 << (id % BITS_PER_WORD));
        self.words[word].fetch_or(bit, Ordering::SeqCst);
    }

    /// Mark `id` as no longer parked.
    #[inline]
    pub(super) fn clear(&self, id: usize) {
        let (word, bit) = (id / BITS_PER_WORD, 1u64 << (id % BITS_PER_WORD));
        self.words[word].fetch_and(!bit, Ordering::SeqCst);
    }

    /// Atomically claim one parked worker, clearing its bit, and return its id.
    /// Returns `None` if no worker is currently parked. `worker_count` bounds
    /// the search so a partially-filled trailing word never yields a phantom id.
    pub(super) fn claim_one(&self, worker_count: usize) -> Option<usize> {
        for (word_index, word) in self.words.iter().enumerate() {
            let mut idle = word.load(Ordering::SeqCst);
            while idle != 0 {
                let bit_index = idle.trailing_zeros() as usize;
                let id = word_index * BITS_PER_WORD + bit_index;
                if id >= worker_count {
                    break;
                }
                let mask = 1u64 << bit_index;
                match word.compare_exchange_weak(
                    idle,
                    idle & !mask,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => return Some(id),
                    Err(actual) => idle = actual,
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_bitset_claims_nothing() {
        let set = IdleBitset::new(128);
        assert_eq!(set.claim_one(128), None);
    }

    #[test]
    fn set_then_claim_returns_id_once() {
        let set = IdleBitset::new(128);
        set.set(7);
        assert_eq!(set.claim_one(128), Some(7));
        // The bit was cleared by the claim, so a second claim finds nothing.
        assert_eq!(set.claim_one(128), None);
    }

    #[test]
    fn clear_removes_bit() {
        let set = IdleBitset::new(64);
        set.set(3);
        set.clear(3);
        assert_eq!(set.claim_one(64), None);
    }

    #[test]
    fn addresses_workers_beyond_first_word() {
        // The single-AtomicU64 design could not represent these ids; the
        // multi-word bitset must round-trip them across word boundaries.
        let set = IdleBitset::new(200);
        for &id in &[0usize, 63, 64, 65, 127, 128, 199] {
            set.set(id);
        }
        let mut claimed = Vec::new();
        while let Some(id) = set.claim_one(200) {
            claimed.push(id);
        }
        claimed.sort_unstable();
        assert_eq!(claimed, vec![0, 63, 64, 65, 127, 128, 199]);
    }

    #[test]
    fn partial_trailing_word_yields_no_phantom_ids() {
        // worker_count = 70 -> two words, but ids 70..128 must never surface.
        let set = IdleBitset::new(70);
        set.set(69);
        // Manually set an out-of-range bit in the trailing word and confirm the
        // bounded scan never returns it.
        set.words[1].fetch_or(1u64 << 10, Ordering::SeqCst); // would be id 74
        assert_eq!(set.claim_one(70), Some(69));
        assert_eq!(set.claim_one(70), None);
    }

    #[test]
    fn single_worker_uses_one_word() {
        let set = IdleBitset::new(1);
        assert_eq!(set.words.len(), 1);
        set.set(0);
        assert_eq!(set.claim_one(1), Some(0));
    }
}
