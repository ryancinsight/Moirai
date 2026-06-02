//! Synchronous data-parallel primitives — Moirai's rayon-replacement surface.
//!
//! This crate is the **parallel** domain (throughput over data), distinct from
//! the **concurrent** domain (`moirai-async`, async tasks/IO). All operations
//! here are fully synchronous (no `async`, no `.await`), so they are safe to use
//! inside pure compute kernels without introducing async contagion. They mirror
//! the common rayon patterns:
//!
//! - [`par_for_each`] / [`par_for_each_mut`] — `slice.par_iter[_mut]().for_each`
//! - [`par_enumerate`] / [`par_enumerate_mut`] — the `.enumerate()` variants
//! - [`par_map_collect`] — `slice.par_iter().map(f).collect()`
//! - [`par_map_reduce`] — `slice.par_iter().map(m).reduce(id, r)`
//!
//! Work is split into one chunk per worker thread and dispatched on the shared
//! process-wide executor ([`moirai_executor::global`]); every task completes
//! before the call returns, so borrows of the input slice remain valid for the
//! whole parallel region.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

use moirai_executor::{global, BlockingTask};

/// Pointer wrapper used to hand disjoint `&mut` sub-slices to worker tasks.
///
/// The `Send`/`Sync` impls are sound only because the `*_mut` functions assign
/// each task a non-overlapping index range, so the pointer is never used to
/// form aliasing references.
struct DisjointMutPtr<T>(*mut T);

// SAFETY: callers dereference pairwise-disjoint ranges only, so the pointer
// never forms aliasing `&mut` references; `T: Send` permits moving element
// access across worker threads.
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
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
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
        .for_each_indexed::<BlockingTask, _>(chunks, move |ci| {
            let start = ci * chunk;
            if start >= n {
                return;
            }
            let end = (start + chunk).min(n);
            for item in &data[start..end] {
                f(item);
            }
        })
        .expect("moirai global executor: par_for_each");
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
        .for_each_indexed::<BlockingTask, _>(chunks, move |ci| {
            let start = ci * chunk;
            if start >= n {
                return;
            }
            let end = (start + chunk).min(n);
            // SAFETY: ranges `[start, end)` are pairwise disjoint across `ci`,
            // so the `&mut` sub-slices never alias. `data` is mutably borrowed
            // for the whole call and `for_each_indexed` joins every task before
            // returning, so `base` remains valid for the duration.
            let slice =
                unsafe { core::slice::from_raw_parts_mut(base.base().add(start), end - start) };
            for item in slice {
                f(item);
            }
        })
        .expect("moirai global executor: par_for_each_mut");
}

/// Apply `f(index, &element)` to every element of `data` in parallel.
///
/// Synchronous equivalent of rayon's `data.par_iter().enumerate().for_each(f)`.
pub fn par_enumerate<T, F>(data: &[T], f: F)
where
    T: Sync,
    F: Fn(usize, &T) + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return;
    }
    let (chunks, chunk) = chunk_layout(n);
    if chunks <= 1 {
        data.iter().enumerate().for_each(|(i, x)| f(i, x));
        return;
    }
    let f = &f;
    global()
        .for_each_indexed::<BlockingTask, _>(chunks, move |ci| {
            let start = ci * chunk;
            if start >= n {
                return;
            }
            let end = (start + chunk).min(n);
            for (offset, item) in data[start..end].iter().enumerate() {
                f(start + offset, item);
            }
        })
        .expect("moirai global executor: par_enumerate");
}

/// Apply `f(index, &mut element)` to every element of `data` in parallel,
/// mutating each in place.
///
/// Synchronous equivalent of rayon's
/// `data.par_iter_mut().enumerate().for_each(f)`.
pub fn par_enumerate_mut<T, F>(data: &mut [T], f: F)
where
    T: Send,
    F: Fn(usize, &mut T) + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return;
    }
    let (chunks, chunk) = chunk_layout(n);
    if chunks <= 1 {
        data.iter_mut().enumerate().for_each(|(i, x)| f(i, x));
        return;
    }
    let base = DisjointMutPtr(data.as_mut_ptr());
    let f = &f;
    global()
        .for_each_indexed::<BlockingTask, _>(chunks, move |ci| {
            let start = ci * chunk;
            if start >= n {
                return;
            }
            let end = (start + chunk).min(n);
            // SAFETY: ranges `[start, end)` are pairwise disjoint across `ci`,
            // so the `&mut` sub-slices never alias; `data` outlives the joined
            // parallel region. See `par_for_each_mut`.
            let slice =
                unsafe { core::slice::from_raw_parts_mut(base.base().add(start), end - start) };
            for (offset, item) in slice.iter_mut().enumerate() {
                f(start + offset, item);
            }
        })
        .expect("moirai global executor: par_enumerate_mut");
}

/// Map each element of `data` with `f` in parallel, collecting results into a
/// `Vec<R>` in the original order.
///
/// Synchronous equivalent of rayon's `data.par_iter().map(f).collect()`.
pub fn par_map_collect<T, R, F>(data: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Send + Sync,
{
    let n = data.len();
    let mut out: Vec<core::mem::MaybeUninit<R>> = Vec::with_capacity(n);
    // SAFETY: `out` has capacity `n`; every slot is written exactly once below
    // before being read, and the elements are `MaybeUninit` so `set_len` does
    // not assume initialization.
    unsafe {
        out.set_len(n);
    }
    par_enumerate_mut(&mut out, |i, slot| {
        slot.write(f(&data[i]));
    });
    // SAFETY: `par_enumerate_mut` initialized every slot, and `MaybeUninit<R>`
    // has the same layout as `R`. Rebuild the `Vec` as `Vec<R>` without
    // re-allocating or running uninitialized drops.
    let mut out = core::mem::ManuallyDrop::new(out);
    unsafe { Vec::from_raw_parts(out.as_mut_ptr().cast::<R>(), n, out.capacity()) }
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
        .map_reduce_indexed::<BlockingTask, _, _, _>(
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
        .expect("moirai global executor: par_map_reduce")
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
        assert_eq!(counter.load(Ordering::Relaxed), data.iter().sum());
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
    fn par_enumerate_mut_uses_index() {
        let mut data = vec![0usize; 5_000];
        par_enumerate_mut(&mut data, |i, x| *x = i * 3);
        for (i, &v) in data.iter().enumerate() {
            assert_eq!(v, i * 3);
        }
    }

    #[test]
    fn par_enumerate_reads_with_index() {
        let data: Vec<usize> = (0..8_000).collect();
        let acc = AtomicUsize::new(0);
        par_enumerate(&data, |i, &x| {
            assert_eq!(i, x);
            acc.fetch_add(x, Ordering::Relaxed);
        });
        assert_eq!(acc.load(Ordering::Relaxed), data.iter().sum());
    }

    #[test]
    fn par_map_collect_preserves_order() {
        let data: Vec<u64> = (0..20_000).collect();
        let squared = par_map_collect(&data, |&x| x * x);
        assert_eq!(squared.len(), data.len());
        for (i, &v) in squared.iter().enumerate() {
            assert_eq!(v, (i as u64) * (i as u64));
        }
    }

    #[test]
    fn par_map_reduce_sums_correctly() {
        let data: Vec<u64> = (0..100_000).collect();
        let sum = par_map_reduce(&data, 0u64, |&x| x, |a, b| a + b);
        assert_eq!(sum, data.iter().copied().sum::<u64>());
    }

    #[test]
    fn empty_and_single_inputs_are_handled() {
        let empty: Vec<i32> = Vec::new();
        par_for_each(&empty, |_| panic!("must not run"));
        assert_eq!(par_map_reduce(&empty, 42i64, |&x| x as i64, |a, b| a + b), 42);
        let mut one = vec![7u64];
        par_for_each_mut(&mut one, |x| *x += 1);
        assert_eq!(one, vec![8]);
    }
}
