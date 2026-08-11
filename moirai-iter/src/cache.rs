//! Cache-aware iterator utilities.
//!
//! Two kinds of thing live here: sequential views that size themselves to the
//! cache ([`WindowIterator`], [`CacheAlignedChunks`]), and
//! [`ZeroCopyParallelIter`], which fans a borrowed slice out across workers
//! without copying it. The second is where the unsafe code is, and it rests on
//! the invariants below.
//!
//! # Fan-out safety
//!
//! Each operation partitions `data` with `slice::chunks(chunk_size)` and gives
//! worker `i` only chunk `i`, so no two workers touch the same elements. The
//! borrowed data, the caller's closure, and — for `map` — the output buffer all
//! cross the thread boundary as raw pointers wrapped in `SendPtr`, because the
//! borrow checker cannot see the partition. Three things make that sound:
//!
//! - **Disjointness.** Chunk `i` starts at `i * chunk_size`, so a worker writing
//!   `chunk_start + offset` for `offset < chunk.len()` stays inside its own
//!   range, and the ranges are pairwise disjoint and within `data.len()`.
//! - **Lifetime.** The pointers refer to locals of the calling frame (`data`'s
//!   backing store, `func`, `results`). The fan-out joins every lane before it
//!   returns, so no lane outlives what it points at.
//! - **Completion.** `map` writes its output through `MaybeUninit` and calls
//!   `assume_init` on every element once the fan-out is done, which is sound
//!   only if every chunk actually wrote its slice. That is enforced rather than
//!   assumed: a failed fan-out reaches `sequential_fallback_permitted`, which
//!   re-runs the whole domain on the caller only for a clean `ShuttingDown` and
//!   panics on a partial run. A lane that panics therefore surfaces as a panic
//!   here instead of an `assume_init` over memory it never wrote.
//!
//! # Chunk sizing
//!
//! Sizes are derived from `CACHE_CHUNK_SIZE` and the element width, and are
//! clamped to at least one element. A zero chunk size is not merely slow: it
//! makes the sequential iterators spin without advancing.

use std::mem;

use crate::base::SendPtr;
/// Default ring buffer capacity (power of 2)
const DEFAULT_RING_BUFFER_CAPACITY: usize = 1024;

/// Cache line size for alignment optimizations
pub const CACHE_LINE_SIZE: usize = 64;

/// Chunk size for cache-friendly iteration (L1 cache half)
pub const CACHE_CHUNK_SIZE: usize = 16384; // 16KB

/// Window iterator that processes data in cache-friendly chunks
pub struct WindowIterator<'a, T> {
    data: &'a [T],
    window_size: usize,
    stride: usize,
    position: usize,
}

impl<'a, T> WindowIterator<'a, T> {
    /// Create a new window iterator with specified window size and stride
    pub fn new(data: &'a [T], window_size: usize, stride: usize) -> Self {
        assert!(window_size > 0, "Window size must be positive");
        assert!(stride > 0, "Stride must be positive");
        Self {
            data,
            window_size,
            stride,
            position: 0,
        }
    }

    /// Create a window iterator with cache-friendly parameters
    pub fn for_cache(data: &'a [T]) -> Self {
        let element_size = mem::size_of::<T>();
        let window = CACHE_CHUNK_SIZE / element_size.max(1);
        Self::new(data, window, window)
    }
}

impl<'a, T> Iterator for WindowIterator<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.data.len() {
            return None;
        }
        let end = (self.position + self.window_size).min(self.data.len());
        let window = &self.data[self.position..end];
        self.position += self.stride;
        Some(window)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.position >= self.data.len() {
            return (0, Some(0));
        }
        let remaining = self.data.len() - self.position;
        let windows = remaining.div_ceil(self.stride);
        (windows, Some(windows))
    }
}

/// Chunk iterator that aligns chunks to cache boundaries
pub struct CacheAlignedChunks<'a, T> {
    data: &'a [T],
    chunk_size: usize,
    position: usize,
}

impl<'a, T> CacheAlignedChunks<'a, T> {
    /// Create a cache-aligned chunk iterator over `data`, sizing each chunk to a cache-line multiple.
    pub fn new(data: &'a [T]) -> Self {
        // How many elements fill one `CACHE_CHUNK_SIZE` block, and never zero:
        // an element wider than a cache line used to make the old
        // `(CACHE_LINE_SIZE / element_size) * (CACHE_CHUNK_SIZE / CACHE_LINE_SIZE)`
        // truncate to 0, and `next` then advanced `position` by nothing and
        // yielded empty slices forever. This form agrees with the old one for
        // every element size that divides a cache line, and yields one element
        // per chunk for anything larger.
        let element_size = mem::size_of::<T>();
        let chunk_size = (CACHE_CHUNK_SIZE / element_size.max(1)).max(1);
        Self {
            data,
            chunk_size,
            position: 0,
        }
    }
}

impl<'a, T> Iterator for CacheAlignedChunks<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.data.len() {
            return None;
        }
        let end = (self.position + self.chunk_size).min(self.data.len());
        let chunk = &self.data[self.position..end];
        if end < self.data.len() {
            // SAFETY: `end < self.data.len()`, so `add(end)` lands on a live
            // element rather than one past the end, and prefetching only hints
            // the cache — it neither reads nor writes.
            unsafe {
                let next_ptr = self.data.as_ptr().add(end);
                prefetch_read_data(next_ptr as *const u8, 3);
            }
        }
        self.position = end;
        Some(chunk)
    }
}

/// Prefetch data for reading with specified cache level
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to readable memory.
/// The `level` parameter should be in the range 0-3 for x86_64 architectures.
/// On non-x86_64 architectures, this function is a no-op.
#[inline(always)]
pub unsafe fn prefetch_read_data(ptr: *const u8, level: i32) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        match level {
            0 => _mm_prefetch(ptr as *const i8, _MM_HINT_T0),
            1 => _mm_prefetch(ptr as *const i8, _MM_HINT_T1),
            2 => _mm_prefetch(ptr as *const i8, _MM_HINT_T2),
            _ => _mm_prefetch(ptr as *const i8, _MM_HINT_NTA),
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let _ = (ptr, level);
    }
}

/// Prefetch data for writing with specified cache level
///
/// # Safety
///
/// The caller must ensure that `ptr` is a valid pointer to writable memory.
/// The `level` parameter should be in the range 0-3 for x86_64 architectures.
/// On non-x86_64 architectures, this function is a no-op.
#[inline(always)]
pub unsafe fn prefetch_write_data(ptr: *mut u8, level: i32) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        match level {
            0 => _mm_prefetch(ptr as *const i8, _MM_HINT_T0),
            1 => _mm_prefetch(ptr as *const i8, _MM_HINT_T1),
            2 => _mm_prefetch(ptr as *const i8, _MM_HINT_T2),
            _ => _mm_prefetch(ptr as *const i8, _MM_HINT_NTA),
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let _ = (ptr, level);
    }
}

/// A zero-copy parallel iterator that processes data without allocation
pub struct ZeroCopyParallelIter<'a, T> {
    data: &'a [T],
    chunk_size: usize,
}

impl<'a, T: Sync> ZeroCopyParallelIter<'a, T> {
    /// Create a zero-copy parallel iterator over `data`, choosing a chunk size from the number of available threads.
    pub fn new(data: &'a [T]) -> Self {
        let num_threads = themis::CpuTopology::detect()
            .map(|topology| topology.logical_processors())
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(1)
            .max(1);
        let element_size = mem::size_of::<T>();
        let elems_per_cache = CACHE_CHUNK_SIZE / element_size.max(1);
        let chunk_size = (data.len() / num_threads).max(elems_per_cache);
        Self { data, chunk_size }
    }

    /// Apply `func` to every element, in parallel when the data is large enough.
    pub fn for_each<F>(&self, func: F)
    where
        F: Fn(&T) + Send + Sync,
    {
        if !should_execute_scoped_cache::<T>(self.data.len(), self.chunk_size) {
            self.data.iter().for_each(func);
            return;
        }

        let chunks: Vec<_> = self.data.chunks(self.chunk_size).collect();
        let num_chunks = chunks.len();
        let func_ptr = SendPtr(&func as *const F as *const () as *mut ());

        // Bound once so the executor borrows it and the fallback below re-runs
        // the same body, instead of a second copy of it drifting out of step.
        let visit_chunk = |idx: usize| unsafe {
            let chunk = *chunks.get_unchecked(idx);
            let chunk_ptr = chunk.as_ptr();
            let chunk_len = chunk.len();
            let chunk_slice = std::slice::from_raw_parts(chunk_ptr, chunk_len);
            // Optimized: we use a raw pointer cast instead of `let func_ref = &func` to avoid capture lifetime bounds.
            let func_ref = &*(func_ptr.as_ptr() as *const F);
            let cache_line_elements = CACHE_LINE_SIZE / mem::size_of::<T>().max(1);
            for (i, item) in chunk_slice.iter().enumerate() {
                if i % cache_line_elements == 0 && i + cache_line_elements < chunk_slice.len() {
                    let next_ptr = chunk_slice.as_ptr().add(i + cache_line_elements);
                    prefetch_read_data(next_ptr as *const u8, 0);
                }
                func_ref(item);
            }
        };

        let run_on_global = moirai_executor::global()
            .for_each_indexed::<moirai_executor::schedule::SyncTask, _>(num_chunks, &visit_chunk);

        if crate::base::sequential_fallback_permitted(&run_on_global) {
            (0..num_chunks).for_each(visit_chunk);
        }
    }

    /// Map every element through `func` into a new vector, in parallel when the data is large enough.
    pub fn map<F, R>(&self, func: F) -> Vec<R>
    where
        F: Fn(&T) -> R + Send + Sync,
        R: Send,
    {
        use std::mem::MaybeUninit;

        if !should_execute_scoped_cache::<T>(self.data.len(), self.chunk_size) {
            return self.data.iter().map(&func).collect();
        }

        let mut results: Vec<MaybeUninit<R>> = Vec::with_capacity(self.data.len());
        // SAFETY: the allocation holds `data.len()` elements and `MaybeUninit<R>`
        // has no validity requirement, so growing the length to cover them
        // exposes no uninitialized `R`. Every element is written by the fan-out
        // below before the `assume_init` at the end, which both join paths
        // enforce rather than assume (see the module docs).
        unsafe {
            results.set_len(self.data.len());
        }
        let results_ptr: *mut MaybeUninit<R> = results.as_mut_ptr();
        let results_send_ptr = SendPtr(results_ptr);

        // Optimized: we collect chunks first instead of using `.chunks(chunk_size).enumerate()` directly, to improve layout.
        let chunks: Vec<_> = self.data.chunks(self.chunk_size).collect();
        let num_chunks = chunks.len();
        let func_ptr = SendPtr(&func as *const F as *const () as *mut ());

        let map_chunk = |chunk_idx: usize| unsafe {
            let chunk = *chunks.get_unchecked(chunk_idx);
            let chunk_start = chunk_idx * self.chunk_size;
            let chunk_results_ptr = results_send_ptr.as_ptr().add(chunk_start);
            let chunk_ptr = chunk.as_ptr();
            let chunk_len = chunk.len();

            let chunk_slice = std::slice::from_raw_parts(chunk_ptr, chunk_len);
            let func_ref = &*(func_ptr.as_ptr() as *const F);
            for (offset, item) in chunk_slice.iter().enumerate() {
                let result = func_ref(item);
                let result_ptr = chunk_results_ptr.add(offset);
                result_ptr.write(MaybeUninit::new(result));
            }
        };

        let run_on_global = moirai_executor::global()
            .for_each_indexed::<moirai_executor::schedule::SyncTask, _>(num_chunks, &map_chunk);

        if crate::base::sequential_fallback_permitted(&run_on_global) {
            (0..num_chunks).for_each(map_chunk);
        }

        // SAFETY: control only reaches here once every chunk ran —
        // `sequential_fallback_permitted` panics on a partially executed fan-out
        // and re-runs the whole domain on the caller otherwise — and the chunks
        // partition `0..data.len()`, so each element was written exactly once.
        unsafe { results.into_iter().map(|item| item.assume_init()).collect() }
    }

    /// Reduce all elements with the associative `func`, returning `None` for empty data.
    pub fn reduce<F>(&self, func: F) -> Option<T>
    where
        F: Fn(&T, &T) -> T + Send + Sync,
        T: Clone + Send,
    {
        if self.data.is_empty() {
            return None;
        }
        if self.data.len() == 1 {
            return Some(self.data[0].clone());
        }
        if !should_execute_scoped_cache::<T>(self.data.len(), self.chunk_size) {
            return self.data.iter().cloned().reduce(|a, b| func(&a, &b));
        }

        let chunks: Vec<_> = self.data.chunks(self.chunk_size).collect();
        let num_chunks = chunks.len();

        let mut results = Vec::with_capacity(num_chunks);
        for _ in 0..num_chunks {
            results.push(None);
        }

        let results_ptr = SendPtr(results.as_mut_ptr() as *mut ());
        let func_ptr = SendPtr(&func as *const F as *const () as *mut ());

        let reduce_chunk = |idx: usize| unsafe {
            let chunk = *chunks.get_unchecked(idx);
            let chunk_ptr = chunk.as_ptr();
            let chunk_len = chunk.len();
            let chunk_slice = std::slice::from_raw_parts(chunk_ptr, chunk_len);
            let func_ref = &*(func_ptr.as_ptr() as *const F);
            let chunk_result = chunk_slice.iter().cloned().reduce(|a, b| func_ref(&a, &b));
            *(results_ptr.as_ptr() as *mut Option<T>).add(idx) = chunk_result;
        };

        let run_on_global = moirai_executor::global()
            .for_each_indexed::<moirai_executor::schedule::SyncTask, _>(num_chunks, &reduce_chunk);

        if crate::base::sequential_fallback_permitted(&run_on_global) {
            (0..num_chunks).for_each(reduce_chunk);
        }

        let mut current_results: Vec<T> = results.into_iter().flatten().collect();
        while current_results.len() > 1 {
            current_results = reduce_owned_pairs(current_results, &func);
        }
        current_results.into_iter().next()
    }
}

fn reduce_owned_pairs<T, F>(items: Vec<T>, func: &F) -> Vec<T>
where
    F: Fn(&T, &T) -> T,
{
    let capacity = items.len().div_ceil(2);
    let mut iter = items.into_iter();
    let mut reduced = Vec::with_capacity(capacity);

    while let Some(left) = iter.next() {
        match iter.next() {
            Some(right) => reduced.push(func(&left, &right)),
            None => reduced.push(left),
        }
    }

    reduced
}

#[inline]
fn should_execute_scoped_cache<T>(len: usize, chunk_size: usize) -> bool {
    let element_size = mem::size_of::<T>().max(1);
    let cache_chunk_items = (CACHE_CHUNK_SIZE / element_size).max(1);
    let scoped_item_floor = cache_chunk_items.saturating_mul(DEFAULT_RING_BUFFER_CAPACITY);

    len > chunk_size && len > scoped_item_floor
}

/// Extension trait for slices to provide cache-aware iteration
pub trait CacheIterExt<T> {
    /// Iterate overlapping windows of `window_size` elements.
    fn cache_windows(&self, window_size: usize) -> WindowIterator<'_, T>;
    /// Iterate cache-aligned chunks of this slice.
    fn cache_chunks(&self) -> CacheAlignedChunks<'_, T>;
    /// Create a zero-copy parallel iterator over this slice.
    fn zero_copy_par_iter(&self) -> ZeroCopyParallelIter<'_, T>;
}

impl<T: Send + Sync> CacheIterExt<T> for [T] {
    fn cache_windows(&self, window_size: usize) -> WindowIterator<'_, T> {
        WindowIterator::new(self, window_size, window_size)
    }

    fn cache_chunks(&self) -> CacheAlignedChunks<'_, T> {
        CacheAlignedChunks::new(self)
    }

    fn zero_copy_par_iter(&self) -> ZeroCopyParallelIter<'_, T> {
        ZeroCopyParallelIter::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NonClone(u64);

    #[test]
    fn cache_chunks_advance_for_elements_wider_than_a_cache_line() {
        // Regression: the chunk size came from
        // `(CACHE_LINE_SIZE / element_size) * (CACHE_CHUNK_SIZE / CACHE_LINE_SIZE)`,
        // whose first term truncates to 0 once an element exceeds a cache line.
        // `next` then clamped `end` to `position`, advanced by nothing, and
        // yielded empty slices forever — an infinite iterator over any `T`
        // bigger than 64 bytes.
        #[repr(align(8))]
        struct Wide([u64; 24]); // 192 bytes, three cache lines

        let data: Vec<Wide> = (0..8).map(|i| Wide([i; 24])).collect();

        // Bounded so the pre-fix behaviour fails the assertions instead of
        // hanging the suite.
        let chunks: Vec<&[Wide]> = data.cache_chunks().take(64).collect();

        assert!(
            chunks.iter().all(|chunk| !chunk.is_empty()),
            "no chunk may be empty; an empty chunk means `position` did not advance"
        );

        let visited: Vec<u64> = chunks
            .iter()
            .flat_map(|chunk| chunk.iter().map(|wide| wide.0[0]))
            .collect();
        assert_eq!(
            visited,
            (0..8).collect::<Vec<u64>>(),
            "chunks must cover every element exactly once, in order"
        );
    }

    #[test]
    fn test_window_iterator() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let windows: Vec<_> = WindowIterator::new(&data, 3, 2).collect();
        assert_eq!(windows.len(), 4);
        assert_eq!(windows[0], &[1, 2, 3]);
        assert_eq!(windows[1], &[3, 4, 5]);
        assert_eq!(windows[2], &[5, 6, 7]);
        assert_eq!(windows[3], &[7, 8]);
    }

    #[test]
    fn test_cache_aligned_chunks() {
        let data: Vec<i32> = (0..1000).collect();
        let chunks: Vec<_> = data.cache_chunks().collect();
        assert!(!chunks.is_empty());
        assert_eq!(chunks.iter().map(|c| c.len()).sum::<usize>(), 1000);
    }

    #[test]
    fn test_zero_copy_parallel() {
        let data: Vec<i32> = (0..10000).collect();
        let sum = std::sync::atomic::AtomicI64::new(0);
        data.zero_copy_par_iter().for_each(|&x| {
            sum.fetch_add(x as i64, std::sync::atomic::Ordering::Relaxed);
        });
        let expected_sum: i64 = (0..10000).sum();
        assert_eq!(sum.load(std::sync::atomic::Ordering::Relaxed), expected_sum);
    }

    #[test]
    fn zero_copy_parallel_map_borrows_data_and_closure() {
        let data: Vec<i32> = (0..1024).collect();
        let factor = 3_i32;

        let mapped = data.zero_copy_par_iter().map(|value| value * factor);

        assert_eq!(
            mapped,
            data.iter().map(|value| value * factor).collect::<Vec<_>>()
        );
    }

    #[test]
    fn zero_copy_parallel_map_matches_sequential_values() {
        let data: Vec<u64> = (0..10000).collect();

        let mapped = data
            .zero_copy_par_iter()
            .map(|value| value.wrapping_mul(5).wrapping_add(7));

        assert_eq!(
            mapped,
            data.iter()
                .map(|value| value.wrapping_mul(5).wrapping_add(7))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn reduce_owned_pairs_moves_non_clone_odd_value() {
        let reduced = reduce_owned_pairs(
            vec![NonClone(1), NonClone(2), NonClone(3)],
            &|left, right| NonClone(left.0 + right.0),
        )
        .into_iter()
        .map(|item| item.0)
        .collect::<Vec<_>>();

        assert_eq!(reduced, vec![3, 3]);
    }

    #[test]
    fn cache_scoped_execution_gate_uses_batch_capacity_floor() {
        let cache_chunk_items = (CACHE_CHUNK_SIZE / std::mem::size_of::<u64>()).max(1);
        let floor = cache_chunk_items * DEFAULT_RING_BUFFER_CAPACITY;

        assert!(!should_execute_scoped_cache::<u64>(
            floor,
            cache_chunk_items
        ));
        assert!(should_execute_scoped_cache::<u64>(
            floor + 1,
            cache_chunk_items
        ));
    }

    #[test]
    fn zero_copy_parallel_reduce_accepts_non_clone_reducer() {
        let data = [1_u64, 2, 3, 4];
        let token = NonClone(1);

        let reduced = data
            .zero_copy_par_iter()
            .reduce(move |left, right| left + right + token.0)
            .expect("reduction should produce a value");

        assert_eq!(reduced, 13);
    }
}
