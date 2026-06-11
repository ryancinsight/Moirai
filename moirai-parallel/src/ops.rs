use super::DisjointMutPtr;
use crate::policy::{ExecutionPolicy, Parallel};
use moirai_executor::{global, SyncTask};

/// Apply `f` to every element of `data`, scheduled by policy `P`.
pub fn for_each_with<P, T, F>(data: &[T], f: F)
where
    P: ExecutionPolicy,
    T: Sync,
    F: Fn(&T) + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return;
    }
    if !P::parallelize(n) {
        data.iter().for_each(f);
        return;
    }
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(n, move |i| f(&data[i]))
        .expect("moirai global executor: for_each_with");
}

/// Apply `f` to every element of `data` in place, scheduled by policy `P`.
pub fn for_each_mut_with<P, T, F>(data: &mut [T], f: F)
where
    P: ExecutionPolicy,
    T: Send,
    F: Fn(&mut T) + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return;
    }
    if !P::parallelize(n) {
        data.iter_mut().for_each(f);
        return;
    }
    let base = DisjointMutPtr(data.as_mut_ptr());
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(n, move |i| {
            // SAFETY: the scheduler visits each index in `0..n` exactly once
            // across disjoint chunks, so no two tasks alias element `i`; `data`
            // is borrowed mutably for the whole joined call.
            f(unsafe { base.get_mut(i) });
        })
        .expect("moirai global executor: for_each_mut_with");
}

/// Apply `f(index, &element)` to every element of `data`, scheduled by policy `P`.
pub fn enumerate_with<P, T, F>(data: &[T], f: F)
where
    P: ExecutionPolicy,
    T: Sync,
    F: Fn(usize, &T) + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return;
    }
    if !P::parallelize(n) {
        data.iter().enumerate().for_each(|(i, x)| f(i, x));
        return;
    }
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(n, move |i| f(i, &data[i]))
        .expect("moirai global executor: enumerate_with");
}

/// Apply `f(index, &mut element)` to every element of `data` in place,
/// scheduled by policy `P`.
pub fn enumerate_mut_with<P, T, F>(data: &mut [T], f: F)
where
    P: ExecutionPolicy,
    T: Send,
    F: Fn(usize, &mut T) + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return;
    }
    if !P::parallelize(n) {
        data.iter_mut().enumerate().for_each(|(i, x)| f(i, x));
        return;
    }
    let base = DisjointMutPtr(data.as_mut_ptr());
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(n, move |i| {
            // SAFETY: each index in `0..n` is visited exactly once; see
            // `for_each_mut_with`.
            f(i, unsafe { base.get_mut(i) });
        })
        .expect("moirai global executor: enumerate_mut_with");
}

/// Apply `f` to every index in `0..len` in parallel, scheduled by policy `P`.
///
/// Synchronous equivalent of rayon's `(0..len).into_par_iter().for_each(f)`. Use
/// when the work is keyed by index and writes through external disjoint state
/// (atomics, per-index channels) rather than returning a value.
pub fn for_each_index_with<P, F>(len: usize, f: F)
where
    P: ExecutionPolicy,
    F: Fn(usize) + Send + Sync,
{
    if len == 0 {
        return;
    }
    if !P::parallelize(len) {
        (0..len).for_each(f);
        return;
    }
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(len, f)
        .expect("moirai global executor: for_each_index_with");
}

/// Apply `f` to each consecutive `chunk_size`-element mutable chunk of `data` in
/// parallel, scheduled by policy `P`. The final chunk may be shorter.
///
/// Synchronous equivalent of rayon's `data.par_chunks_mut(chunk_size).for_each(f)`
/// — the natural shape for batched/lane-wise transforms.
pub fn for_each_chunk_mut_with<P, T, F>(data: &mut [T], chunk_size: usize, f: F)
where
    P: ExecutionPolicy,
    T: Send,
    F: Fn(&mut [T]) + Send + Sync,
{
    let n = data.len();
    if n == 0 || chunk_size == 0 {
        return;
    }
    let num_chunks = n.div_ceil(chunk_size);
    if !P::parallelize(n) || num_chunks <= 1 {
        data.chunks_mut(chunk_size).for_each(&f);
        return;
    }
    let base = DisjointMutPtr(data.as_mut_ptr());
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(num_chunks, move |c| {
            let start = c * chunk_size;
            if start >= n {
                return;
            }
            let end = (start + chunk_size).min(n);
            // SAFETY: the chunks `[start, end)` for distinct `c` are pairwise
            // disjoint and each is visited exactly once, so no two tasks alias.
            let chunk =
                unsafe { core::slice::from_raw_parts_mut(base.base().add(start), end - start) };
            f(chunk);
        })
        .expect("moirai global executor: for_each_chunk_mut_with");
}

/// Apply `f(index, a_chunk, b_chunk)` to paired `chunk_size`-element mutable
/// chunks of two **distinct** buffers in parallel, scheduled by policy `P`.
///
/// Synchronous equivalent of
/// `a.par_chunks_mut(n).zip(b.par_chunks_mut(n)).enumerate().for_each(f)`. The
/// number of chunks is derived from `a`; `b` is chunked identically, so callers
/// must ensure `b.len() >= a.len()` (typically equal). The two buffers must not
/// alias.
pub fn for_each_chunk_pair_mut_enumerated_with<P, A, B, F>(
    a: &mut [A],
    b: &mut [B],
    chunk_size: usize,
    f: F,
) where
    P: ExecutionPolicy,
    A: Send,
    B: Send,
    F: Fn(usize, &mut [A], &mut [B]) + Send + Sync,
{
    let na = a.len();
    let nb = b.len();
    if chunk_size == 0 || na == 0 {
        return;
    }
    let num_chunks = na.div_ceil(chunk_size);
    if !P::parallelize(na) || num_chunks <= 1 {
        a.chunks_mut(chunk_size)
            .zip(b.chunks_mut(chunk_size))
            .enumerate()
            .for_each(|(i, (ca, cb))| f(i, ca, cb));
        return;
    }
    let abase = DisjointMutPtr(a.as_mut_ptr());
    let bbase = DisjointMutPtr(b.as_mut_ptr());
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(num_chunks, move |c| {
            let start = c * chunk_size;
            if start >= na || start >= nb {
                return;
            }
            let ea = (start + chunk_size).min(na);
            let eb = (start + chunk_size).min(nb);
            // SAFETY: chunks `[start, e*)` for distinct `c` are pairwise disjoint
            // within each buffer and each is visited once; `a` and `b` are
            // distinct, non-aliasing buffers, so the two references never alias.
            let ca =
                unsafe { core::slice::from_raw_parts_mut(abase.base().add(start), ea - start) };
            let cb =
                unsafe { core::slice::from_raw_parts_mut(bbase.base().add(start), eb - start) };
            f(c, ca, cb);
        })
        .expect("moirai global executor: for_each_chunk_pair_mut_enumerated_with");
}

/// Like [`for_each_chunk_mut_with`] but also passes the zero-based chunk index to
/// `f` (synchronous equivalent of
/// `data.par_chunks_mut(chunk_size).enumerate().for_each(f)`).
pub fn for_each_chunk_mut_enumerated_with<P, T, F>(data: &mut [T], chunk_size: usize, f: F)
where
    P: ExecutionPolicy,
    T: Send,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    let n = data.len();
    if n == 0 || chunk_size == 0 {
        return;
    }
    let num_chunks = n.div_ceil(chunk_size);
    if !P::parallelize(n) || num_chunks <= 1 {
        data.chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(i, c)| f(i, c));
        return;
    }
    let base = DisjointMutPtr(data.as_mut_ptr());
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(num_chunks, move |c| {
            let start = c * chunk_size;
            if start >= n {
                return;
            }
            let end = (start + chunk_size).min(n);
            // SAFETY: chunks `[start, end)` for distinct `c` are pairwise disjoint
            // and each visited exactly once, so no two tasks alias.
            let chunk =
                unsafe { core::slice::from_raw_parts_mut(base.base().add(start), end - start) };
            f(c, chunk);
        })
        .expect("moirai global executor: for_each_chunk_mut_enumerated_with");
}

/// Map each element of `data` with `f`, collecting into a `Vec<R>` in order,
/// scheduled by policy `P`.
pub fn map_collect_with<P, T, R, F>(data: &[T], f: F) -> Vec<R>
where
    P: ExecutionPolicy,
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Send + Sync,
{
    let n = data.len();
    if !P::parallelize(n) {
        return data.iter().map(f).collect();
    }
    let mut out: Vec<core::mem::MaybeUninit<R>> = Vec::with_capacity(n);
    // SAFETY: capacity is `n`; every slot is written exactly once below before
    // being read, and `MaybeUninit` makes `set_len` sound without initialization.
    unsafe {
        out.set_len(n);
    }
    enumerate_mut_with::<Parallel, _, _>(&mut out, |i, slot| {
        slot.write(f(&data[i]));
    });
    // SAFETY: every slot initialized above; `MaybeUninit<R>` shares `R`'s layout.
    let mut out = core::mem::ManuallyDrop::new(out);
    unsafe { Vec::from_raw_parts(out.as_mut_ptr().cast::<R>(), n, out.capacity()) }
}

/// Map-reduce over `data`, scheduled by policy `P`.
///
/// `reduce` must be associative and `identity` its neutral element, since chunk
/// boundaries and combination order are unspecified.
pub fn map_reduce_with<P, T, R, M, Rd>(data: &[T], identity: R, map: M, reduce: Rd) -> R
where
    P: ExecutionPolicy,
    T: Sync,
    R: Send + Sync + Clone,
    M: Fn(&T) -> R + Send + Sync,
    Rd: Fn(R, R) -> R + Send + Sync,
{
    let n = data.len();
    if n == 0 || !P::parallelize(n) {
        let mut acc = identity;
        for item in data {
            acc = reduce(acc, map(item));
        }
        return acc;
    }
    let map = &map;
    let reduce = &reduce;
    // The executor folds each worker chunk locally (seeded by `identity`) then
    // combines chunk results, so `map` is per-element and `reduce` per-pair.
    global()
        .map_reduce_indexed::<SyncTask, _, _, _>(n, identity, move |i| map(&data[i]), reduce)
        .expect("moirai global executor: map_reduce_with")
}

/// Parallel fold-reduce over the index domain `0..len`, scheduled by policy `P`.
///
/// Each worker chunk creates one accumulator with `init()`, folds its indices
/// into it with `fold`, and the per-chunk accumulators are combined with
/// `reduce`. Unlike [`reduce_index_with`], `fold` mutates a single accumulator
/// per chunk (no per-element temporary), which is the efficient shape for
/// accumulating into a collection — e.g. grouping entries into a `HashMap`.
/// `reduce` must be associative; `init()` must yield its neutral element.
pub fn fold_reduce_with<P, A, Init, Fold, Red>(len: usize, init: Init, fold: Fold, reduce: Red) -> A
where
    P: ExecutionPolicy,
    A: Send,
    Init: Fn() -> A + Send + Sync,
    Fold: Fn(A, usize) -> A + Send + Sync,
    Red: Fn(A, A) -> A,
{
    if len == 0 {
        return init();
    }
    if !P::parallelize(len) {
        let mut acc = init();
        for i in 0..len {
            acc = fold(acc, i);
        }
        return acc;
    }
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let chunks = workers.min(len).max(1);
    let chunk = len.div_ceil(chunks);
    let mut slots: Vec<Option<A>> = (0..chunks).map(|_| None).collect();
    let base = DisjointMutPtr(slots.as_mut_ptr());
    let init_ref = &init;
    let fold_ref = &fold;
    global()
        .for_each_indexed::<SyncTask, _>(chunks, move |ci| {
            let start = ci * chunk;
            if start >= len {
                return;
            }
            let end = (start + chunk).min(len);
            let mut acc = init_ref();
            for i in start..end {
                acc = fold_ref(acc, i);
            }
            // SAFETY: each `ci` writes its own slot exactly once; slots are
            // disjoint and `slots` outlives the joined call.
            unsafe {
                *base.get_mut(ci) = Some(acc);
            }
        })
        .expect("moirai global executor: fold_reduce_with");
    slots
        .into_iter()
        .flatten()
        .reduce(reduce)
        .unwrap_or_else(init)
}

/// Parallel map over the index domain `0..len`, collecting into a `Vec<R>` in
/// order, scheduled by policy `P`.
///
/// `map(i)` produces the element at index `i`. Use this for index-aligned maps
/// over multiple slices that [`map_collect_with`] cannot express — e.g. an
/// elementwise product `map_collect_index_with::<Adaptive>(n, |i| a[i] * b[i])`.
pub fn map_collect_index_with<P, R, Map>(len: usize, map: Map) -> Vec<R>
where
    P: ExecutionPolicy,
    R: Send,
    Map: Fn(usize) -> R + Send + Sync,
{
    if !P::parallelize(len) {
        return (0..len).map(map).collect();
    }
    let mut out: Vec<core::mem::MaybeUninit<R>> = Vec::with_capacity(len);
    // SAFETY: capacity is `len`; every slot is written exactly once below.
    unsafe {
        out.set_len(len);
    }
    enumerate_mut_with::<Parallel, _, _>(&mut out, |i, slot| {
        slot.write(map(i));
    });
    // SAFETY: every slot initialized; `MaybeUninit<R>` shares `R`'s layout.
    let mut out = core::mem::ManuallyDrop::new(out);
    unsafe { Vec::from_raw_parts(out.as_mut_ptr().cast::<R>(), len, out.capacity()) }
}

/// Map each element of `data` in place with `f(index, &mut element)`, collecting
/// each returned value into a `Vec<R>` in order, scheduled by policy `P`.
///
/// The synchronous equivalent of rayon's
/// `data.par_iter_mut().enumerate().map(f).collect()`: each element is mutated
/// and produces a result. Use for parallel solve-in-place-and-collect loops.
pub fn map_collect_mut_with<P, T, R, F>(data: &mut [T], f: F) -> Vec<R>
where
    P: ExecutionPolicy,
    T: Send,
    R: Send,
    F: Fn(usize, &mut T) -> R + Send + Sync,
{
    let n = data.len();
    if !P::parallelize(n) {
        return data.iter_mut().enumerate().map(|(i, x)| f(i, x)).collect();
    }
    let mut out: Vec<core::mem::MaybeUninit<R>> = Vec::with_capacity(n);
    // SAFETY: capacity is `n`; every slot is written exactly once below.
    unsafe {
        out.set_len(n);
    }
    let data_ptr = DisjointMutPtr(data.as_mut_ptr());
    let out_ptr = DisjointMutPtr(out.as_mut_ptr());
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(n, move |i| {
            // SAFETY: each index in `0..n` is visited exactly once, so neither the
            // input element nor the output slot at `i` aliases another task's.
            let elem = unsafe { data_ptr.get_mut(i) };
            let result = f(i, elem);
            unsafe { out_ptr.get_mut(i).write(result) };
        })
        .expect("moirai global executor: map_collect_mut_with");
    // SAFETY: every slot initialized; `MaybeUninit<R>` shares `R`'s layout.
    let mut out = core::mem::ManuallyDrop::new(out);
    unsafe { Vec::from_raw_parts(out.as_mut_ptr().cast::<R>(), n, out.capacity()) }
}

/// Parallel reduction over the index domain `0..len`, scheduled by policy `P`.
///
/// `map(i)` produces a value for index `i`; results are folded within and across
/// chunks with `reduce`, seeded by `identity` (which must be `reduce`'s neutral
/// element). Use this for index-aligned reductions over multiple slices that
/// [`map_reduce_with`] cannot express — e.g. a dot product
/// `reduce_index_with::<Adaptive>(n, T::zero(), |i| a[i] * b[i], |x, y| x + y)`.
pub fn reduce_index_with<P, R, Map, Red>(len: usize, identity: R, map: Map, reduce: Red) -> R
where
    P: ExecutionPolicy,
    R: Send + Sync + Clone,
    Map: Fn(usize) -> R + Send + Sync,
    Red: Fn(R, R) -> R + Send + Sync,
{
    if len == 0 || !P::parallelize(len) {
        let mut acc = identity;
        for i in 0..len {
            acc = reduce(acc, map(i));
        }
        return acc;
    }
    global()
        .map_reduce_indexed::<SyncTask, _, _, _>(len, identity, map, reduce)
        .expect("moirai global executor: reduce_index_with")
}
