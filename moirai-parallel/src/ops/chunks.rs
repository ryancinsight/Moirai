//! Mutable chunk operators over one or more disjoint buffers.

use super::super::DisjointMutPtr;
use crate::policy::ExecutionPolicy;
use moirai_executor::{global, SyncTask};

#[cfg(test)]
mod tests;

/// Failure to partition a fixed set of mutable buffers into matching chunks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ChunkBuffersError {
    /// A buffer does not have the same element count as the first buffer.
    LengthMismatch {
        /// Zero-based position of the mismatched buffer.
        buffer_index: usize,
        /// Required element count, taken from the first buffer.
        expected: usize,
        /// Actual element count of the mismatched buffer.
        actual: usize,
    },
}

impl core::fmt::Display for ChunkBuffersError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch {
                buffer_index,
                expected,
                actual,
            } => write!(
                formatter,
                "chunk buffer {buffer_index} has length {actual}, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for ChunkBuffersError {}

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

/// Apply `f(state, chunk)` to each consecutive mutable chunk, creating one
/// reusable state value per scheduled worker shard.
///
/// This is the scratch-buffer form of [`for_each_chunk_mut_with`]. It matches
/// the allocation discipline of Rayon-style `for_each_init`/`for_each_with`
/// loops: a worker shard initializes `S` once, then reuses it for every logical
/// chunk assigned to that shard.
pub fn for_each_chunk_mut_with_state<P, T, S, Init, F>(
    data: &mut [T],
    chunk_size: usize,
    init: Init,
    f: F,
) where
    P: ExecutionPolicy,
    T: Send,
    S: Send,
    Init: Fn() -> S + Send + Sync,
    F: Fn(&mut S, &mut [T]) + Send + Sync,
{
    let n = data.len();
    if n == 0 || chunk_size == 0 {
        return;
    }
    let num_chunks = n.div_ceil(chunk_size);
    if !P::parallelize(n) || num_chunks <= 1 {
        let mut state = init();
        for chunk in data.chunks_mut(chunk_size) {
            f(&mut state, chunk);
        }
        return;
    }

    let workers = themis::CpuTopology::detect()
        .map(|topology| topology.logical_processors())
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(1)
        .min(num_chunks)
        .max(1);
    let chunks_per_worker = num_chunks.div_ceil(workers);
    let base = DisjointMutPtr(data.as_mut_ptr());
    let init = &init;
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(workers, move |worker| {
            let first_chunk = worker * chunks_per_worker;
            let last_chunk = ((worker + 1) * chunks_per_worker).min(num_chunks);
            if first_chunk >= last_chunk {
                return;
            }
            let mut state = init();
            for chunk_index in first_chunk..last_chunk {
                let start = chunk_index * chunk_size;
                let end = (start + chunk_size).min(n);
                // SAFETY: logical chunks are assigned to exactly one worker and
                // are pairwise disjoint, so each mutable slice is exclusive.
                let chunk =
                    unsafe { core::slice::from_raw_parts_mut(base.base().add(start), end - start) };
                f(&mut state, chunk);
            }
        })
        .expect("moirai global executor: for_each_chunk_mut_with_state");
}

/// Apply `f(index, chunks)` to matching chunks from a fixed set of distinct
/// mutable buffers, scheduled by policy `P`.
///
/// All buffers must have the same length. Validation completes before any
/// buffer is mutated. The final chunk may be shorter than `chunk_size`; zero
/// buffers, empty buffers, and a zero chunk size are no-ops.
///
/// The fixed-size array keeps the operation allocation-free and lets callers
/// fuse any homogeneous number of output-buffer passes without adding another
/// arity-specific operator.
///
/// # Examples
///
/// ```
/// use moirai_parallel::{
///     for_each_chunk_buffers_mut_enumerated_with, ChunkBuffersError, Sequential,
/// };
///
/// let mut left = [0_u32; 5];
/// let mut right = [0_u32; 5];
/// for_each_chunk_buffers_mut_enumerated_with::<Sequential, _, _, 2>(
///     [&mut left, &mut right],
///     2,
///     |chunk_index, [left, right]| {
///         left.fill(chunk_index as u32);
///         right.fill((chunk_index as u32) + 10);
///     },
/// )?;
///
/// assert_eq!(left, [0, 0, 1, 1, 2]);
/// assert_eq!(right, [10, 10, 11, 11, 12]);
/// # Ok::<(), ChunkBuffersError>(())
/// ```
///
/// # Errors
///
/// Returns [`ChunkBuffersError::LengthMismatch`] when a buffer length differs
/// from the first buffer's length.
pub fn for_each_chunk_buffers_mut_enumerated_with<P, T, F, const N: usize>(
    mut buffers: [&mut [T]; N],
    chunk_size: usize,
    f: F,
) -> Result<(), ChunkBuffersError>
where
    P: ExecutionPolicy,
    T: Send,
    F: for<'chunk> Fn(usize, [&'chunk mut [T]; N]) + Send + Sync,
{
    let Some(first) = buffers.first() else {
        return Ok(());
    };
    let length = first.len();
    if let Some((buffer_index, actual)) =
        buffers
            .iter()
            .enumerate()
            .skip(1)
            .find_map(|(buffer_index, buffer)| {
                (buffer.len() != length).then_some((buffer_index, buffer.len()))
            })
    {
        return Err(ChunkBuffersError::LengthMismatch {
            buffer_index,
            expected: length,
            actual,
        });
    }
    if length == 0 || chunk_size == 0 {
        return Ok(());
    }

    let num_chunks = length.div_ceil(chunk_size);
    if !P::parallelize(length) || num_chunks <= 1 {
        for chunk_index in 0..num_chunks {
            let start = chunk_index * chunk_size;
            let end = (start + chunk_size).min(length);
            let chunks = buffers.each_mut().map(|buffer| {
                buffer
                    .get_mut(start..end)
                    .expect("invariant: equal buffer lengths were validated before mutation")
            });
            f(chunk_index, chunks);
        }
        return Ok(());
    }

    let bases = buffers
        .each_mut()
        .map(|buffer| DisjointMutPtr(buffer.as_mut_ptr()));
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(num_chunks, move |chunk_index| {
            let start = chunk_index * chunk_size;
            let end = (start + chunk_size).min(length);
            let chunks = core::array::from_fn(|buffer_index| {
                let base = bases
                    .get(buffer_index)
                    .expect("invariant: array-generated buffer index is in bounds");
                // SAFETY: safe construction of `buffers` proves the N mutable
                // slices do not alias. Equal lengths were validated above, and
                // distinct tasks own pairwise-disjoint `[start, end)` ranges.
                unsafe { core::slice::from_raw_parts_mut(base.base().add(start), end - start) }
            });
            f(chunk_index, chunks);
        })
        .expect("moirai global executor: for_each_chunk_buffers_mut_enumerated_with");
    Ok(())
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
            // SAFETY: same disjointness argument as `ca`, within `b`'s own
            // non-aliasing buffer.
            let cb =
                unsafe { core::slice::from_raw_parts_mut(bbase.base().add(start), eb - start) };
            f(c, ca, cb);
        })
        .expect("moirai global executor: for_each_chunk_pair_mut_enumerated_with");
}

/// Apply `f(index, a_chunk, b_chunk, c_chunk, d_chunk)` to four **distinct**
/// mutable buffers chunked identically, scheduled by policy `P`.
///
/// This is the four-buffer counterpart to
/// [`for_each_chunk_pair_mut_enumerated_with`]. It is intended for fused
/// statistics and stencil bookkeeping kernels where one authoritative pass
/// updates several output arrays without allocating intermediate tuples.
pub fn for_each_chunk_quad_mut_enumerated_with<P, A, B, C, D, F>(
    a: &mut [A],
    b: &mut [B],
    c: &mut [C],
    d: &mut [D],
    chunk_size: usize,
    f: F,
) where
    P: ExecutionPolicy,
    A: Send,
    B: Send,
    C: Send,
    D: Send,
    F: Fn(usize, &mut [A], &mut [B], &mut [C], &mut [D]) + Send + Sync,
{
    let na = a.len();
    let nb = b.len();
    let nc = c.len();
    let nd = d.len();
    assert_eq!(na, nb, "quad chunk buffers must have equal lengths");
    assert_eq!(na, nc, "quad chunk buffers must have equal lengths");
    assert_eq!(na, nd, "quad chunk buffers must have equal lengths");
    if chunk_size == 0 || na == 0 {
        return;
    }
    let num_chunks = na.div_ceil(chunk_size);
    if !P::parallelize(na) || num_chunks <= 1 {
        a.chunks_mut(chunk_size)
            .zip(b.chunks_mut(chunk_size))
            .zip(c.chunks_mut(chunk_size))
            .zip(d.chunks_mut(chunk_size))
            .enumerate()
            .for_each(|(i, (((ca, cb), cc), cd))| f(i, ca, cb, cc, cd));
        return;
    }
    let abase = DisjointMutPtr(a.as_mut_ptr());
    let bbase = DisjointMutPtr(b.as_mut_ptr());
    let cbase = DisjointMutPtr(c.as_mut_ptr());
    let dbase = DisjointMutPtr(d.as_mut_ptr());
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(num_chunks, move |chunk_index| {
            let start = chunk_index * chunk_size;
            if start >= na || start >= nb || start >= nc || start >= nd {
                return;
            }
            let ea = (start + chunk_size).min(na);
            let eb = (start + chunk_size).min(nb);
            let ec = (start + chunk_size).min(nc);
            let ed = (start + chunk_size).min(nd);
            // SAFETY: chunks `[start, e*)` for distinct `chunk_index` values
            // are pairwise disjoint within each buffer and each is visited at
            // most once. The four input buffers are distinct non-aliasing
            // `&mut` slices, so the returned mutable chunk references cannot
            // alias each other.
            let ca =
                unsafe { core::slice::from_raw_parts_mut(abase.base().add(start), ea - start) };
            // SAFETY: same disjointness argument as `ca`, within `b`'s own
            // non-aliasing buffer.
            let cb =
                unsafe { core::slice::from_raw_parts_mut(bbase.base().add(start), eb - start) };
            // SAFETY: same disjointness argument as `ca`, within `c`'s own
            // non-aliasing buffer.
            let cc =
                unsafe { core::slice::from_raw_parts_mut(cbase.base().add(start), ec - start) };
            // SAFETY: same disjointness argument as `ca`, within `d`'s own
            // non-aliasing buffer.
            let cd =
                unsafe { core::slice::from_raw_parts_mut(dbase.base().add(start), ed - start) };
            f(chunk_index, ca, cb, cc, cd);
        })
        .expect("moirai global executor: for_each_chunk_quad_mut_enumerated_with");
}

/// Apply `f(index, a_chunk, b_chunk, c_chunk)` to three **distinct** mutable
/// buffers chunked identically, scheduled by policy `P`.
///
/// This is the three-buffer counterpart to
/// [`for_each_chunk_pair_mut_enumerated_with`].
pub fn for_each_chunk_triple_mut_enumerated_with<P, A, B, C, F>(
    a: &mut [A],
    b: &mut [B],
    c: &mut [C],
    chunk_size: usize,
    f: F,
) where
    P: ExecutionPolicy,
    A: Send,
    B: Send,
    C: Send,
    F: Fn(usize, &mut [A], &mut [B], &mut [C]) + Send + Sync,
{
    let na = a.len();
    let nb = b.len();
    let nc = c.len();
    assert_eq!(na, nb, "triple chunk buffers must have equal lengths");
    assert_eq!(na, nc, "triple chunk buffers must have equal lengths");
    if chunk_size == 0 || na == 0 {
        return;
    }
    let num_chunks = na.div_ceil(chunk_size);
    if !P::parallelize(na) || num_chunks <= 1 {
        a.chunks_mut(chunk_size)
            .zip(b.chunks_mut(chunk_size))
            .zip(c.chunks_mut(chunk_size))
            .enumerate()
            .for_each(|(i, ((ca, cb), cc))| f(i, ca, cb, cc));
        return;
    }
    let abase = DisjointMutPtr(a.as_mut_ptr());
    let bbase = DisjointMutPtr(b.as_mut_ptr());
    let cbase = DisjointMutPtr(c.as_mut_ptr());
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(num_chunks, move |chunk_index| {
            let start = chunk_index * chunk_size;
            if start >= na || start >= nb || start >= nc {
                return;
            }
            let ea = (start + chunk_size).min(na);
            let eb = (start + chunk_size).min(nb);
            let ec = (start + chunk_size).min(nc);
            // SAFETY: chunks `[start, e*)` for distinct `chunk_index` values
            // are pairwise disjoint within each buffer and each is visited at
            // most once. The three input buffers are distinct non-aliasing
            // `&mut` slices, so the returned mutable chunk references cannot
            // alias each other.
            let ca =
                unsafe { core::slice::from_raw_parts_mut(abase.base().add(start), ea - start) };
            // SAFETY: same disjointness argument as `ca`, within `b`'s own
            // non-aliasing buffer.
            let cb =
                unsafe { core::slice::from_raw_parts_mut(bbase.base().add(start), eb - start) };
            // SAFETY: same disjointness argument as `ca`, within `c`'s own
            // non-aliasing buffer.
            let cc =
                unsafe { core::slice::from_raw_parts_mut(cbase.base().add(start), ec - start) };
            f(chunk_index, ca, cb, cc);
        })
        .expect("moirai global executor: for_each_chunk_triple_mut_enumerated_with");
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
