//! Concurrency safety tests for the lock-free work-stealing deques.
//!
//! These validate the core work-stealing safety property under real thread
//! contention, for [`ChaseLevDeque`] and [`SplitDeque`]:
//! with a single owner (push/pop from the bottom) and many thieves (steal from the top),
//! **every pushed item is consumed exactly once** — no loss, no duplication, no
//! torn value. This is the empirical complement to the weak-memory fence
//! discipline (Lê, Pop, Cohen & Nardelli, PPoPP 2013); the matching `loom`
//! exhaustive model check of the steal/pop ordering protocol lives in
//! `tests/loom_chase_lev.rs` (run with `RUSTFLAGS="--cfg loom"`).
//!
//! The deque contract is single-owner / multi-thief: only the owning thread may
//! `push`/`pop`; any thread may `steal`. The tests honor that — the owner lives
//! on one thread and shares the deque with thieves through `Arc`.

use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;

use moirai_scheduler::{ChaseLevDeque, ChaseLevStealer, SplitDeque, StealResult, StolenBatch};

struct DropTracked {
    id: usize,
    drops: Arc<Vec<AtomicUsize>>,
}

impl Drop for DropTracked {
    fn drop(&mut self) {
        self.drops
            .get(self.id)
            .expect("test item id must be within the drop marks")
            .fetch_add(1, Ordering::Relaxed);
    }
}

/// One generic harness over both deque implementations. The method names match
/// the inherent ones; in the generic test body only the trait methods are in
/// scope, so dispatch is unambiguous.
trait WorkStealer<T>: Clone + Send + Sync {
    fn w_steal(&self) -> StealResult<T>;
    fn w_steal_batch(&self) -> StealResult<StolenBatch<T>>;
}

trait WorkOwner<T> {
    type Stealer: WorkStealer<T> + 'static;

    fn w_push(&mut self, item: T);
    fn w_pop(&mut self) -> Option<T>;
    fn w_stealer(&self) -> Self::Stealer;
}

impl<T: Send> WorkStealer<T> for ChaseLevStealer<T> {
    fn w_steal(&self) -> StealResult<T> {
        self.steal()
    }
    fn w_steal_batch(&self) -> StealResult<StolenBatch<T>> {
        self.steal_batch()
    }
}

impl<T: Send + 'static> WorkOwner<T> for ChaseLevDeque<T> {
    type Stealer = ChaseLevStealer<T>;

    fn w_push(&mut self, item: T) {
        self.push(item);
    }
    fn w_pop(&mut self) -> Option<T> {
        self.pop()
    }
    fn w_stealer(&self) -> Self::Stealer {
        self.stealer()
    }
}

impl<T: Send> WorkStealer<T> for Arc<SplitDeque<T>> {
    fn w_steal(&self) -> StealResult<T> {
        self.steal()
    }
    fn w_steal_batch(&self) -> StealResult<StolenBatch<T>> {
        self.steal_batch()
    }
}

impl<T: Send + 'static> WorkOwner<T> for Arc<SplitDeque<T>> {
    type Stealer = Arc<SplitDeque<T>>;

    fn w_push(&mut self, item: T) {
        self.push(item);
    }
    fn w_pop(&mut self) -> Option<T> {
        self.pop()
    }
    fn w_stealer(&self) -> Self::Stealer {
        Arc::clone(self)
    }
}

/// Run one owner + `thieves` stealing threads over items `0..n` against `deque`
/// and assert each item is consumed exactly once. When `batch` is set the
/// thieves use the batched steal path.
fn exactly_once_inner<D>(mut deque: D, n: usize, thieves: usize, batch: bool)
where
    D: WorkOwner<usize>,
{
    let marks: Arc<Vec<AtomicU8>> = Arc::new((0..n).map(|_| AtomicU8::new(0)).collect());
    let consumed = Arc::new(AtomicUsize::new(0));
    let out_of_range = Arc::new(AtomicUsize::new(0));

    let mark = |marks: &[AtomicU8], out_of_range: &AtomicUsize, item: usize| {
        // fetch_add returns the prior value; a non-zero prior means a duplicate
        // consume, caught by the final assertion.
        if let Some(mark) = marks.get(item) {
            mark.fetch_add(1, Ordering::Relaxed);
        } else {
            out_of_range.fetch_add(1, Ordering::Relaxed);
        }
    };

    let thief_handles: Vec<_> = (0..thieves)
        .map(|_| {
            let deque = deque.w_stealer();
            let marks = Arc::clone(&marks);
            let consumed = Arc::clone(&consumed);
            let out_of_range = Arc::clone(&out_of_range);
            std::thread::spawn(move || {
                while consumed.load(Ordering::Acquire) < n {
                    let result = if batch {
                        match deque.w_steal_batch() {
                            StealResult::Success(mut items) => {
                                let first = items
                                    .next()
                                    .expect("invariant: successful batch is non-empty");
                                for item in items {
                                    mark(&marks, &out_of_range, item);
                                    consumed.fetch_add(1, Ordering::Release);
                                }
                                StealResult::Success(first)
                            }
                            StealResult::Empty => StealResult::Empty,
                            StealResult::Retry => StealResult::Retry,
                        }
                    } else {
                        deque.w_steal()
                    };
                    match result {
                        StealResult::Success(item) => {
                            mark(&marks, &out_of_range, item);
                            consumed.fetch_add(1, Ordering::Release);
                        }
                        // Retry: lost a CAS race; Empty: nothing visible right
                        // now. Neither is terminal while items remain.
                        StealResult::Retry | StealResult::Empty => std::thread::yield_now(),
                    }
                }
            })
        })
        .collect();

    // Owner: push every item, interleaving occasional pops so the bottom path
    // races the thieves' top path. Then drain whatever the thieves did not take.
    for i in 0..n {
        deque.w_push(i);
        if i % 7 == 0 {
            if let Some(item) = deque.w_pop() {
                mark(&marks, &out_of_range, item);
                consumed.fetch_add(1, Ordering::Release);
            }
        }
    }
    while consumed.load(Ordering::Acquire) < n {
        if let Some(item) = deque.w_pop() {
            mark(&marks, &out_of_range, item);
            consumed.fetch_add(1, Ordering::Release);
        } else {
            std::thread::yield_now();
        }
    }

    for h in thief_handles {
        h.join().expect("thief thread must not panic");
    }

    // Exactly-once: every item consumed exactly one time.
    let mut lost = Vec::new();
    let mut duplicated = Vec::new();
    for (i, m) in marks.iter().enumerate() {
        match m.load(Ordering::Relaxed) {
            1 => {}
            0 => lost.push(i),
            c => duplicated.push((i, c)),
        }
    }
    assert!(
        lost.is_empty() && duplicated.is_empty() && out_of_range.load(Ordering::Relaxed) == 0,
        "deque violated exactly-once: {} lost (first few: {:?}), {} duplicated (first few: {:?}), {} out-of-range",
        lost.len(),
        &lost[..lost.len().min(8)],
        duplicated.len(),
        &duplicated[..duplicated.len().min(8)],
        out_of_range.load(Ordering::Relaxed),
    );
    assert_eq!(
        consumed.load(Ordering::Relaxed),
        n,
        "consumed count mismatch"
    );
}

fn chase_lev(n: usize, thieves: usize, capacity: usize, batch: bool) {
    exactly_once_inner(ChaseLevDeque::<usize>::new(capacity), n, thieves, batch);
}

fn split(n: usize, thieves: usize, batch: bool) {
    exactly_once_inner(Arc::new(SplitDeque::<usize>::new()), n, thieves, batch);
}

// ── ChaseLevDeque ───────────────────────────────────────────────────────────

#[test]
fn chase_lev_exactly_once_small_capacity_forces_resize() {
    // Small initial capacity forces repeated growth while thieves steal — the
    // resize/reclaim path is where a stale array pointer would corrupt a steal.
    for _ in 0..16 {
        chase_lev(20_000, 4, 16, false);
    }
}

#[test]
fn chase_lev_exactly_once_high_thief_contention() {
    for _ in 0..8 {
        chase_lev(50_000, 8, 1024, false);
    }
}

#[test]
fn chase_lev_exactly_once_single_thief() {
    // One thief maximizes the last-element owner/thief contest frequency.
    for _ in 0..16 {
        chase_lev(20_000, 1, 64, false);
    }
}

#[test]
fn chase_lev_batch_exactly_once_high_thief_contention() {
    for _ in 0..8 {
        chase_lev(30_000, 8, 128, true);
    }
}

#[test]
fn chase_lev_batch_claims_non_copy_values_before_reading_them() {
    let n = 30_000;
    let drops: Arc<Vec<AtomicUsize>> = Arc::new((0..n).map(|_| AtomicUsize::new(0)).collect());
    let consumed = Arc::new(AtomicUsize::new(0));
    let mut owner = ChaseLevDeque::<DropTracked>::new(128);

    for id in 0..n {
        owner.push(DropTracked {
            id,
            drops: Arc::clone(&drops),
        });
    }

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let stealer = owner.stealer();
            let consumed = Arc::clone(&consumed);
            std::thread::spawn(move || {
                while consumed.load(Ordering::Acquire) < n {
                    match stealer.steal_batch() {
                        StealResult::Success(items) => {
                            for item in items {
                                std::hint::black_box(item);
                                consumed.fetch_add(1, Ordering::Release);
                            }
                        }
                        StealResult::Empty | StealResult::Retry => std::thread::yield_now(),
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thief thread must not panic");
    }
    drop(owner);

    assert_eq!(consumed.load(Ordering::Acquire), n);
    assert!(drops.iter().all(|count| count.load(Ordering::Relaxed) == 1));
}

// ── SplitDeque ──────────────────────────────────────────────────────────────

#[test]
fn split_exactly_once_high_thief_contention() {
    for _ in 0..8 {
        split(50_000, 8, false);
    }
}

#[test]
fn split_exactly_once_single_thief() {
    for _ in 0..16 {
        split(20_000, 1, false);
    }
}

#[test]
fn split_batch_exactly_once_high_thief_contention() {
    for _ in 0..8 {
        split(30_000, 8, true);
    }
}
