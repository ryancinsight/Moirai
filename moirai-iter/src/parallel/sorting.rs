//! Parallel slice sorting implementation.

use crate::base::{get_shared_thread_pool, SendPtr, ThreadPool};
use std::mem::MaybeUninit;
use std::sync::Arc;

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
        let pool = get_shared_thread_pool();
        par_merge_sort_impl(self, &compare, &pool);
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
        let pool = get_shared_thread_pool();
        par_sort_unstable_by_impl(self, &compare, &pool);
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

fn par_sort_unstable_by_impl<T, F>(slice: &mut [T], compare: &F, pool: &Arc<ThreadPool>)
where
    T: Send,
    F: Fn(&T, &T) -> std::cmp::Ordering + Sync + Send,
{
    let len = slice.len();
    if len <= UNSTABLE_SEQUENTIAL_THRESHOLD {
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

    // Erase types to () to satisfy the 'static requirements of ThreadPool::execute
    let left_ptr = SendPtr(left.as_mut_ptr() as *mut ());
    let left_len = left.len();
    let compare_ptr = SendPtr(compare as *const F as *mut F as *mut ());

    let (tx, rx) = std::sync::mpsc::channel();
    let pool_clone = Arc::clone(pool);

    pool.execute(move || {
        let left_ptr = left_ptr;
        let compare_ptr = compare_ptr;
        unsafe {
            let left_slice = std::slice::from_raw_parts_mut(left_ptr.as_ptr() as *mut T, left_len);
            let compare_ref = &*(compare_ptr.as_ptr() as *const F);
            par_sort_unstable_by_impl(left_slice, compare_ref, &pool_clone);
        }
        let _ = tx.send(());
    });

    par_sort_unstable_by_impl(right, compare, pool);
    let _ = rx.recv();
}

fn par_merge_sort_impl<T, F>(slice: &mut [T], compare: &F, pool: &Arc<ThreadPool>)
where
    T: Send,
    F: Fn(&T, &T) -> std::cmp::Ordering + Sync + Send,
{
    let len = slice.len();
    if len <= STABLE_SEQUENTIAL_THRESHOLD {
        slice.sort_by(compare);
        return;
    }

    let mid = len / 2;
    let (left, right) = slice.split_at_mut(mid);

    // Erase types to () to satisfy the 'static requirements of ThreadPool::execute
    let left_ptr = SendPtr(left.as_mut_ptr() as *mut ());
    let left_len = left.len();
    let compare_ptr = SendPtr(compare as *const F as *mut F as *mut ());

    let (tx, rx) = std::sync::mpsc::channel();
    let pool_clone = Arc::clone(pool);

    pool.execute(move || {
        let left_ptr = left_ptr;
        let compare_ptr = compare_ptr;
        unsafe {
            let left_slice = std::slice::from_raw_parts_mut(left_ptr.as_ptr() as *mut T, left_len);
            let compare_ref = &*(compare_ptr.as_ptr() as *const F);
            par_merge_sort_impl(left_slice, compare_ref, &pool_clone);
        }
        let _ = tx.send(());
    });

    par_merge_sort_impl(right, compare, pool);
    let _ = rx.recv();

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
        let left_val = unsafe { &*guard.left_vec.as_ptr().add(guard.i).cast::<T>() };
        let right_val = &guard.slice[guard.j];

        if compare(left_val, right_val) == std::cmp::Ordering::Greater {
            unsafe {
                std::ptr::copy(
                    guard.slice.as_ptr().add(guard.j),
                    guard.slice.as_mut_ptr().add(guard.k),
                    1,
                );
            }
            guard.j += 1;
        } else {
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
