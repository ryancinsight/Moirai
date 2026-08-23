//! Parallel slice sorting implementation.

use moirai_core::error::ExecutorError;
use moirai_executor::{global, HybridExecutor, SyncTask};
use std::mem::MaybeUninit;

/// Extension trait for parallel slice sorting.
pub trait ParallelSliceMut<T: Send> {
    /// Sorts the slice in parallel (stable).
    fn par_sort(&mut self)
    where
        T: Ord;

    /// Sorts the slice in parallel with a comparator (stable).
    fn par_sort_by<F>(&mut self, compare: F)
    where
        F: Fn(&T, &T) -> std::cmp::Ordering + Sync + Send;

    /// Sorts the slice in parallel with a key extraction function (stable).
    fn par_sort_by_key<K, F>(&mut self, f: F)
    where
        F: Fn(&T) -> K + Sync + Send,
        K: Ord + Send;

    /// Sorts the slice in parallel (unstable).
    fn par_sort_unstable(&mut self)
    where
        T: Ord;

    /// Sorts the slice in parallel with a comparator (unstable).
    fn par_sort_unstable_by<F>(&mut self, compare: F)
    where
        F: Fn(&T, &T) -> std::cmp::Ordering + Sync + Send;

    /// Sorts the slice in parallel with a key extraction function (unstable).
    fn par_sort_unstable_by_key<K, F>(&mut self, f: F)
    where
        F: Fn(&T) -> K + Sync + Send,
        K: Ord + Send;
}

impl<T: Send> ParallelSliceMut<T> for [T] {
    fn par_sort(&mut self)
    where
        T: Ord,
    {
        self.par_sort_by(T::cmp);
    }

    fn par_sort_by<F>(&mut self, compare: F)
    where
        F: Fn(&T, &T) -> std::cmp::Ordering + Sync + Send,
    {
        let executor = global();
        let grain = fork_grain(executor, self.len(), STABLE_SEQUENTIAL_THRESHOLD);
        par_merge_sort_impl(executor, self, &compare, grain);
    }

    fn par_sort_by_key<K, F>(&mut self, f: F)
    where
        F: Fn(&T) -> K + Sync + Send,
        K: Ord + Send,
    {
        self.par_sort_by(move |a, b| f(a).cmp(&f(b)));
    }

    fn par_sort_unstable(&mut self)
    where
        T: Ord,
    {
        self.par_sort_unstable_by(T::cmp);
    }

    fn par_sort_unstable_by<F>(&mut self, compare: F)
    where
        F: Fn(&T, &T) -> std::cmp::Ordering + Sync + Send,
    {
        let executor = global();
        let grain = fork_grain(executor, self.len(), UNSTABLE_SEQUENTIAL_THRESHOLD);
        par_sort_unstable_by_impl(executor, self, &compare, grain);
    }

    fn par_sort_unstable_by_key<K, F>(&mut self, f: F)
    where
        F: Fn(&T) -> K + Sync + Send,
        K: Ord + Send,
    {
        self.par_sort_unstable_by(move |a, b| f(a).cmp(&f(b)));
    }
}

// Sequential thresholds keep worker dispatch on inputs large enough to amortize
// task scheduling and merge/partition overhead.
const STABLE_SEQUENTIAL_THRESHOLD: usize = 2048;
const UNSTABLE_SEQUENTIAL_THRESHOLD: usize = 16_384;

/// Segments per worker the recursion aims for before it stops forking.
///
/// The thresholds above are an absolute floor, not a granularity policy: on a
/// large input they leave a leaf count proportional to `len`, and every leaf
/// costs a scope — measurably more than the sort work it enables once the leaf
/// is small relative to the machine. Oversubscribing the workers by this factor
/// keeps enough independent segments for stealing to balance an uneven split
/// (a straggler delays the phase by at most one segment) while making the fork
/// count a function of machine width rather than input size.
const SEGMENTS_PER_WORKER: usize = 8;

/// Smallest sub-slice still worth handing to another lane.
fn fork_grain(executor: &HybridExecutor, len: usize, sequential_threshold: usize) -> usize {
    let workers = executor.config().worker_threads.max(1);
    sequential_threshold.max(len.div_ceil(workers.saturating_mul(SEGMENTS_PER_WORKER)))
}

fn partition<T, F>(v: &mut [T], compare: &F) -> usize
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    let len = v.len();
    if len <= 1 {
        return 0;
    }

    let pivot_idx = len / 2;
    v.swap(0, pivot_idx);

    let mut i = 1;
    let mut j = len - 1;

    loop {
        while i < len && compare(&v[i], &v[0]) == std::cmp::Ordering::Less {
            i += 1;
        }
        while j > 0 && compare(&v[j], &v[0]) == std::cmp::Ordering::Greater {
            j -= 1;
        }
        if i >= j {
            break;
        }
        v.swap(i, j);
        i += 1;
        j -= 1;
    }
    v.swap(0, j);
    j
}

/// Sort both halves concurrently: one on a scheduler lane, the other on the
/// caller's.
///
/// The scheduler's scope is the fork-join primitive here rather than a plain
/// thread pool because a worker that waits inside a scope *runs queued work*
/// instead of parking (ADR-019). A pool without that property starves the
/// moment recursion blocks every worker on a half that is still queued, which
/// is what the deleted fork budget existed to prevent — at the cost of capping
/// the whole work tree at the pool's width. Scoped jobs also borrow, so the
/// halves cross the lane boundary as ordinary `&mut [T]` rather than as raw
/// pointers laundered through a `'static` bound.
///
/// Each half is captured by unique borrow, never moved, so a job the scheduler
/// refuses can still be run here: on refusal neither half has been touched.
///
/// The executor is a parameter rather than `global()` so the refusal path can
/// be exercised against a shut-down executor in tests.
///
/// # Panics
///
/// Panics if the scheduled half panicked, propagating the failure on the
/// caller's thread as rayon does.
fn fork_join_halves<T, F, S>(
    executor: &HybridExecutor,
    left: &mut [T],
    right: &mut [T],
    compare: &F,
    grain: usize,
    sort: S,
) where
    T: Send,
    F: Fn(&T, &T) -> std::cmp::Ordering + Sync + Send,
    S: Fn(&HybridExecutor, &mut [T], &F, usize) + Copy + Send + Sync,
{
    // The job below *reborrows* its half rather than moving it: passing a
    // `&mut` where a `&mut` is expected is an implicit reborrow, so the borrow
    // ends when the scope joins and both bindings are usable again afterwards.
    // That is what lets the refusal arm run a half the scheduler never ran.
    let forked = executor.scope::<SyncTask, _>(|scope| {
        scope.spawn(|_| sort(executor, left, compare, grain))?;
        // Enter the scheduler before the caller takes its own half, so the two
        // halves overlap instead of running back to back.
        scope.flush()?;
        sort(executor, right, compare, grain);
        Ok(())
    });

    match forked {
        Ok(()) => {}
        // The scheduler refused the job and dropped it unexecuted, so `flush`
        // returned before the caller's half started: neither half has run and
        // both are still owned here. `ShuttingDown` and a full admission queue
        // are the two ways that happens; running the work on the caller is the
        // same answer `for_each_indexed` gives a rejected chunk.
        Err(ExecutorError::ShuttingDown | ExecutorError::ResourceExhausted(_)) => {
            sort(executor, left, compare, grain);
            sort(executor, right, compare, grain);
        }
        Err(error) => panic!("invariant: scheduled sort half failed ({error})"),
    }
}

fn par_sort_unstable_by_impl<T, F>(
    executor: &HybridExecutor,
    slice: &mut [T],
    compare: &F,
    grain: usize,
) where
    T: Send,
    F: Fn(&T, &T) -> std::cmp::Ordering + Sync + Send,
{
    let len = slice.len();
    if len <= grain {
        slice.sort_unstable_by(compare);
        return;
    }

    let pivot_idx = partition(slice, compare);
    let (left, right) = slice.split_at_mut(pivot_idx);
    let right = if right.is_empty() {
        right
    } else {
        &mut right[1..] // Skip the pivot itself
    };

    fork_join_halves(
        executor,
        left,
        right,
        compare,
        grain,
        par_sort_unstable_by_impl,
    );
}

fn par_merge_sort_impl<T, F>(executor: &HybridExecutor, slice: &mut [T], compare: &F, grain: usize)
where
    T: Send,
    F: Fn(&T, &T) -> std::cmp::Ordering + Sync + Send,
{
    let len = slice.len();
    if len <= grain {
        slice.sort_by(compare);
        return;
    }

    let mid = len / 2;
    {
        let (left, right) = slice.split_at_mut(mid);
        fork_join_halves(executor, left, right, compare, grain, par_merge_sort_impl);
    }

    merge(slice, mid, compare);
}

struct MergeGuard<'a, T> {
    slice: &'a mut [T],
    left_vec: Vec<MaybeUninit<T>>,
    i: usize,
    j: usize,
    k: usize,
    mid: usize,
}

impl<'a, T> Drop for MergeGuard<'a, T> {
    fn drop(&mut self) {
        let remaining = self.mid - self.i;
        if remaining > 0 {
            // SAFETY: drop runs only when merge bailed early with unconsumed
            // left elements; source range stays inside `left_vec`'s len and
            // destination slots k..mid were vacated by the consumed prefix,
            // so ranges are disjoint and uninitialized-valid as MaybeUninit.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.left_vec.as_ptr().add(self.i),
                    self.slice.as_mut_ptr().add(self.k).cast::<MaybeUninit<T>>(),
                    remaining,
                );
            }
        }
    }
}

fn merge<T, F>(slice: &mut [T], mid: usize, compare: &F)
where
    T: Send,
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    let len = slice.len();
    if len <= 1 || mid == 0 || mid >= len {
        return;
    }

    let mut left_vec: Vec<MaybeUninit<T>> = Vec::with_capacity(mid);
    // SAFETY: capacity mid was just reserved; copying mid initialized
    // elements from the slice's left half makes them initialized owners, so
    // set_len is honest and no value is duplicated (the slice side of these
    // slots is logically moved out and never dropped twice — merge writes
    // every slot before any later drop).
    unsafe {
        std::ptr::copy_nonoverlapping(
            slice.as_ptr().cast::<MaybeUninit<T>>(),
            left_vec.as_mut_ptr(),
            mid,
        );
        left_vec.set_len(mid);
    }

    let mut guard = MergeGuard {
        slice,
        left_vec,
        i: 0,
        j: mid,
        k: 0,
        mid,
    };

    while guard.i < guard.mid && guard.j < len {
        // SAFETY: i < mid bounds the index and the vector holds mid
        // initialized values per the copy above.
        let left_val = unsafe { &*guard.left_vec.as_ptr().add(guard.i).cast::<T>() };
        let right_val = &guard.slice[guard.j];

        if compare(left_val, right_val) == std::cmp::Ordering::Greater {
            // SAFETY: j < len and k <= j hold during the right-run advance,
            // so forward `copy` handles the overlap correctly and both
            // indices stay in bounds.
            unsafe {
                std::ptr::copy(
                    guard.slice.as_ptr().add(guard.j),
                    guard.slice.as_mut_ptr().add(guard.k),
                    1,
                );
            }
            guard.j += 1;
        } else {
            // SAFETY: i < mid bounds the source; k <= i + (j - mid) keeps the
            // destination at or behind consumed positions, disjoint from the
            // still-referenced left_vec range, and within slice bounds.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    guard.left_vec.as_ptr().add(guard.i),
                    guard
                        .slice
                        .as_mut_ptr()
                        .add(guard.k)
                        .cast::<MaybeUninit<T>>(),
                    1,
                );
            }
            guard.i += 1;
        }
        guard.k += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Recursion depth well past any worker count. Under a runtime whose waiters
    // park instead of helping, the forked halves have nobody left to run them
    // and the sort never returns, so a regression trips nextest's terminate
    // bound rather than failing an assertion. The deterministic single-worker
    // proof lives at the scheduler layer, where the scope contract is owned
    // (ADR-019); this input is what the sort itself can constrain, since it
    // runs on the process-wide executor.
    #[test]
    fn deep_recursion_completes() {
        let mut data: Vec<u64> = (0..1_048_576u64).rev().collect();
        let grain = fork_grain(global(), data.len(), STABLE_SEQUENTIAL_THRESHOLD);
        par_merge_sort_impl(global(), &mut data, &u64::cmp, grain);

        assert!(
            data.windows(2).all(|pair| pair[0] <= pair[1]),
            "the sort must both finish and order the slice"
        );
    }

    #[test]
    fn deep_unstable_recursion_completes() {
        let mut data: Vec<u64> = (0..1_048_576u64).rev().collect();
        let grain = fork_grain(global(), data.len(), UNSTABLE_SEQUENTIAL_THRESHOLD);
        par_sort_unstable_by_impl(global(), &mut data, &u64::cmp, grain);

        assert!(
            data.windows(2).all(|pair| pair[0] <= pair[1]),
            "the sort must both finish and order the slice"
        );
    }

    // A scheduler that refuses the forked half must not lose it. Sorting
    // against a shut-down executor makes every fork take the refusal arm, so
    // the whole recursion falls back to the caller's lane: slower, still
    // correct. A refusal arm that dropped the half instead would leave the
    // slice unsorted at every level.
    #[test]
    fn refused_forks_run_on_the_caller() {
        let mut executor =
            moirai_executor::HybridExecutor::new(moirai_core::executor::ExecutorConfig {
                worker_threads: 2,
                ..moirai_core::executor::ExecutorConfig::default()
            })
            .expect("build a local executor");
        executor.shutdown().expect("shut the local executor down");
        assert!(
            executor.scope::<SyncTask, _>(|_| Ok(())).is_err(),
            "precondition: the executor must refuse scopes, or the sorts below \
             never reach the refusal arm"
        );

        let mut data: Vec<u64> = (0..16_384u64).rev().collect();
        par_merge_sort_impl(&executor, &mut data, &u64::cmp, STABLE_SEQUENTIAL_THRESHOLD);
        assert!(
            data.windows(2).all(|pair| pair[0] <= pair[1]),
            "a refused fork must still sort its half on the caller"
        );

        let mut data: Vec<u64> = (0..65_536u64).rev().collect();
        par_sort_unstable_by_impl(
            &executor,
            &mut data,
            &u64::cmp,
            UNSTABLE_SEQUENTIAL_THRESHOLD,
        );
        assert!(
            data.windows(2).all(|pair| pair[0] <= pair[1]),
            "a refused fork must still sort its half on the caller"
        );
    }

    // Sorts entered from *inside* a scheduler worker: the nested case, where a
    // waiter that parks removes the last runner from the pool. Several of them
    // run at once so the workers are saturated before any of them forks.
    #[test]
    fn nested_sorts_complete_from_scheduler_workers() {
        const SORTS: usize = 8;

        let mut inputs: Vec<Vec<u64>> =
            (0..SORTS).map(|_| (0..65_536u64).rev().collect()).collect();

        let slots: Vec<crate::base::SendPtr<Vec<u64>>> = inputs
            .iter_mut()
            .map(|input| crate::base::SendPtr(input as *mut Vec<u64>))
            .collect();

        moirai_executor::global()
            .for_each_indexed::<SyncTask, _>(SORTS, |index| {
                // Safety: each index owns exactly one element of `inputs`, and
                // `for_each_indexed` joins every invocation before returning,
                // so the borrows are disjoint and end before `inputs` is read.
                let data = unsafe { &mut *slots[index].as_ptr() };
                par_merge_sort_impl(
                    global(),
                    data.as_mut_slice(),
                    &u64::cmp,
                    STABLE_SEQUENTIAL_THRESHOLD,
                );
            })
            .expect("nested sort fan-out must complete");

        for input in &inputs {
            assert!(
                input.windows(2).all(|pair| pair[0] <= pair[1]),
                "every nested sort must both finish and order its slice"
            );
        }
    }

    #[test]
    fn test_sorting_empty_and_single() {
        let mut v: Vec<i32> = vec![];
        v.par_sort();
        assert!(v.is_empty());

        let mut v = vec![42];
        v.par_sort();
        assert_eq!(v, vec![42]);

        let mut v: Vec<i32> = vec![];
        v.par_sort_unstable();
        assert!(v.is_empty());

        let mut v = vec![42];
        v.par_sort_unstable();
        assert_eq!(v, vec![42]);
    }

    #[test]
    fn test_sorting_already_sorted_and_reverse() {
        let mut v = vec![1, 2, 3, 4, 5, 6];
        v.par_sort();
        assert_eq!(v, vec![1, 2, 3, 4, 5, 6]);

        let mut v = vec![6, 5, 4, 3, 2, 1];
        v.par_sort();
        assert_eq!(v, vec![1, 2, 3, 4, 5, 6]);

        let mut v = vec![1, 2, 3, 4, 5, 6];
        v.par_sort_unstable();
        assert_eq!(v, vec![1, 2, 3, 4, 5, 6]);

        let mut v = vec![6, 5, 4, 3, 2, 1];
        v.par_sort_unstable();
        assert_eq!(v, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_sorting_duplicates() {
        let mut v = vec![2, 2, 1, 1, 3, 3, 2, 2];
        v.par_sort();
        assert_eq!(v, vec![1, 1, 2, 2, 2, 2, 3, 3]);

        let mut v = vec![2, 2, 1, 1, 3, 3, 2, 2];
        v.par_sort_unstable();
        assert_eq!(v, vec![1, 1, 2, 2, 2, 2, 3, 3]);
    }

    #[test]
    fn test_sorting_large_random() {
        // Simple deterministic LCG random generator for testing

        let mut seed: u64 = 12345;
        let mut random_u32 = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed as u32
        };

        let mut original = Vec::new();
        for _ in 0..5000 {
            original.push(random_u32() % 10000);
        }

        let mut v1 = original.clone();
        v1.par_sort();
        let mut expected = original.clone();
        expected.sort();
        assert_eq!(v1, expected);

        let mut v2 = original.clone();
        v2.par_sort_unstable();
        assert_eq!(v2, expected);
    }

    #[derive(Debug, Eq, PartialEq)]
    struct KeyVal {
        key: i32,
        val: usize,
    }

    #[test]
    fn test_sorting_stability() {
        let mut v = vec![
            KeyVal { key: 2, val: 0 },
            KeyVal { key: 1, val: 1 },
            KeyVal { key: 2, val: 2 },
            KeyVal { key: 1, val: 3 },
            KeyVal { key: 3, val: 4 },
            KeyVal { key: 2, val: 5 },
        ];

        // Stable sort by key
        v.par_sort_by(|a, b| a.key.cmp(&b.key));

        assert_eq!(
            v,
            vec![
                KeyVal { key: 1, val: 1 },
                KeyVal { key: 1, val: 3 },
                KeyVal { key: 2, val: 0 },
                KeyVal { key: 2, val: 2 },
                KeyVal { key: 2, val: 5 },
                KeyVal { key: 3, val: 4 },
            ]
        );
    }

    #[test]
    fn test_sorting_by_key() {
        let mut v = [KeyVal { key: 2, val: 0 }, KeyVal { key: 1, val: 1 }];
        v.par_sort_by_key(|item| item.key);
        assert_eq!(v[0].key, 1);
        assert_eq!(v[1].key, 2);

        let mut v = [KeyVal { key: 2, val: 0 }, KeyVal { key: 1, val: 1 }];
        v.par_sort_unstable_by_key(|item| item.key);
        assert_eq!(v[0].key, 1);
        assert_eq!(v[1].key, 2);
    }

    static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

    #[derive(Debug, Clone, Eq, PartialEq)]
    struct TrackedItem(i32);

    impl Drop for TrackedItem {
        fn drop(&mut self) {
            DROP_COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn test_panic_safety_no_double_drop() {
        DROP_COUNT.store(0, Ordering::SeqCst);

        let mut v = vec![
            TrackedItem(3),
            TrackedItem(1),
            TrackedItem(2),
            TrackedItem(4),
        ];

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            v.par_sort_by(|a, b| {
                if a.0 == 2 || b.0 == 2 {
                    panic!("simulated comparator panic");
                }
                a.0.cmp(&b.0)
            });
        }));

        assert!(result.is_err());
        // Verify drop count matches number of elements exactly once when vector is dropped
        drop(v);
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 4);
    }
}
