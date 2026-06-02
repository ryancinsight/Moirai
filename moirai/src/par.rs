//! Synchronous data-parallel primitives — the rayon-replacement surface.
//!
//! These run on the global Moirai runtime's work-stealing scheduler and are
//! fully **synchronous** (no `async`, no `.await`), so they are safe to use
//! inside pure compute kernels without introducing async contagion. They mirror
//! the common rayon patterns:
//!
//! - [`par_for_each`] — `slice.par_iter().for_each(f)`
//! - [`par_for_each_mut`] — `slice.par_iter_mut().for_each(f)`
//! - [`par_map_reduce`] — `slice.par_iter().map(m).reduce(id, r)`
//!
//! Work is split into one chunk per worker thread; each chunk is processed by a
//! scheduler task and every task completes before the call returns, so borrows
//! of the input slice remain valid for the whole parallel region.

use crate::global;

/// Pointer wrapper used to hand disjoint `&mut` sub-slices to worker tasks.
///
/// The `Send`/`Sync` impls are sound only because [`par_for_each_mut`] assigns
/// each task a non-overlapping index range, so no two tasks ever materialize
/// aliasing references from the wrapped pointer.
struct DisjointMutPtr<T>(*mut T);

// SAFETY: callers (only `par_for_each_mut`) dereference pairwise-disjoint
// ranges, so the pointer is never used to form aliasing `&mut` references, and
// `T: Send` permits moving element access across worker threads.
unsafe impl<T: Send> Send for DisjointMutPtr<T> {}
unsafe impl<T: Send> Sync for DisjointMutPtr<T> {}

impl<T> DisjointMutPtr<T> {
    /// Return the wrapped base pointer. Taking `&self` forces a closure to
    /// capture the whole wrapper (which is `Send`/`Sync`) rather than the bare
    /// `*mut T` field under 2021 disjoint capture.
    #[inline]
    fn base(&self) -> *mut T {
        self.0
    }
}

/// Compute `(number_of_chunks, chunk_len)` for splitting `len` items across the
/// runtime's worker threads. Never returns zero chunks for non-empty input.
#[inline]
fn chunk_layout(len: usize) -> (usize, usize) {
    let workers = global().worker_count().max(1);
    let chunks = workers.min(len).max(1);
    (chunks, len.div_ceil(chunks))
}

/// Apply `f` to every element of `data` in parallel.
///
/// Synchronous equivalent of rayon's `data.par_iter().for_each(f)`.
pub fn par_for_each<T, F>(data: &[T], f: F)
where
    T: Sync,
    F: Fn(&T) + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return;
    }
    let (chunks, chunk) = chunk_layout(n);
    if chunks <= 1 {
        data.iter().for_each(f);
        return;
    }
    let f = &f;
    global()
        .for_each_indexed(chunks, move |ci| {
            let start = ci * chunk;
            if start >= n {
                return;
            }
            let end = (start + chunk).min(n);
            for item in &data[start..end] {
                f(item);
            }
        })
        .expect("moirai global runtime: par_for_each");
}

/// Apply `f` to every element of `data` in parallel, mutating each in place.
///
/// Synchronous equivalent of rayon's `data.par_iter_mut().for_each(f)`.
pub fn par_for_each_mut<T, F>(data: &mut [T], f: F)
where
    T: Send,
    F: Fn(&mut T) + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return;
    }
    let (chunks, chunk) = chunk_layout(n);
    if chunks <= 1 {
        data.iter_mut().for_each(f);
        return;
    }
    let base = DisjointMutPtr(data.as_mut_ptr());
    let f = &f;
    global()
        .for_each_indexed(chunks, move |ci| {
            let start = ci * chunk;
            if start >= n {
                return;
            }
            let end = (start + chunk).min(n);
            // SAFETY: ranges `[start, end)` are pairwise disjoint across `ci`,
            // so the `&mut` sub-slices never alias. `data` is mutably borrowed
            // for the whole call and `for_each_indexed` joins every task before
            // returning, so `base` remains valid for the duration.
            let slice = unsafe { core::slice::from_raw_parts_mut(base.base().add(start), end - start) };
            for item in slice {
                f(item);
            }
        })
        .expect("moirai global runtime: par_for_each_mut");
}

/// Parallel map-reduce over `data`.
///
/// Each element is mapped with `map`; results are folded within and across
/// chunks using `reduce`, seeded by `identity`. `reduce` must be associative
/// and `identity` must be its neutral element, since chunk boundaries and
/// combination order are unspecified.
///
/// Synchronous equivalent of rayon's
/// `data.par_iter().map(map).reduce(|| identity, reduce)`.
pub fn par_map_reduce<T, R, M, Rd>(data: &[T], identity: R, map: M, reduce: Rd) -> R
where
    T: Sync,
    R: Send + Sync + Clone,
    M: Fn(&T) -> R + Send + Sync,
    Rd: Fn(R, R) -> R + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return identity;
    }
    let (chunks, chunk) = chunk_layout(n);
    if chunks <= 1 {
        let mut acc = identity;
        for item in data {
            acc = reduce(acc, map(item));
        }
        return acc;
    }
    let map = &map;
    let reduce = &reduce;
    let identity_for_map = identity.clone();
    global()
        .map_reduce_indexed(
            chunks,
            identity,
            move |ci| {
                let start = ci * chunk;
                let mut acc = identity_for_map.clone();
                if start < n {
                    let end = (start + chunk).min(n);
                    for item in &data[start..end] {
                        acc = reduce(acc, map(item));
                    }
                }
                acc
            },
            move |a, b| reduce(a, b),
        )
        .expect("moirai global runtime: par_map_reduce")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn par_for_each_visits_every_element_once() {
        let data: Vec<usize> = (0..10_000).collect();
        let counter = AtomicUsize::new(0);
        par_for_each(&data, |&x| {
            counter.fetch_add(x, Ordering::Relaxed);
        });
        let expected: usize = data.iter().sum();
        assert_eq!(counter.load(Ordering::Relaxed), expected);
    }

    #[test]
    fn par_for_each_mut_mutates_in_place() {
        let mut data: Vec<u64> = (0..10_000).collect();
        par_for_each_mut(&mut data, |x| *x *= 2);
        for (i, &v) in data.iter().enumerate() {
            assert_eq!(v, (i as u64) * 2);
        }
    }

    #[test]
    fn par_map_reduce_sums_correctly() {
        let data: Vec<u64> = (0..100_000).collect();
        let sum = par_map_reduce(&data, 0u64, |&x| x, |a, b| a + b);
        assert_eq!(sum, data.iter().copied().sum::<u64>());
    }

    #[test]
    fn empty_inputs_are_handled() {
        let empty: Vec<i32> = Vec::new();
        par_for_each(&empty, |_| panic!("must not run"));
        let mut empty_mut: Vec<i32> = Vec::new();
        par_for_each_mut(&mut empty_mut, |_| panic!("must not run"));
        assert_eq!(par_map_reduce(&empty, 42i64, |&x| x as i64, |a, b| a + b), 42);
    }

    #[test]
    fn single_element_uses_sequential_path() {
        let data = vec![7u64];
        let sum = par_map_reduce(&data, 0, |&x| x, |a, b| a + b);
        assert_eq!(sum, 7);
        let mut m = vec![7u64];
        par_for_each_mut(&mut m, |x| *x += 1);
        assert_eq!(m, vec![8]);
    }
}
