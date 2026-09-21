//! Mutable chunk operators over one or more disjoint buffers.

use crate::policy::ExecutionPolicy;
use melinoe::MelinoeCell;
use melinoe::region::WriterShard;
use moirai_executor::{SyncTask, global};

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
    if !P::parallelize_chunks(n, num_chunks) || num_chunks <= 1 {
        data.chunks_mut(chunk_size).for_each(&f);
        return;
    }
    let partitions = WriterShard::new(MelinoeCell::from_mut_slice(data)).par_chunks(chunk_size);
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(num_chunks, move |c| {
            // SAFETY: `for_each_indexed(num_chunks, _)` visits each chunk index
            // exactly once — the contract `get_unchecked_chunk` documents — and
            // distinct chunk indices name disjoint element ranges, so no two
            // tasks alias.
            let chunk = unsafe { partitions.get_unchecked_chunk(c) }.into_mut_slice();
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
    if !P::parallelize_chunks(n, num_chunks) || num_chunks <= 1 {
        let mut state = init();
        for chunk in data.chunks_mut(chunk_size) {
            f(&mut state, chunk);
        }
        return;
    }

    let workers = moirai_core::executor::logical_parallelism()
        .min(num_chunks)
        .max(1);
    let chunks_per_worker = num_chunks.div_ceil(workers);
    let partitions = WriterShard::new(MelinoeCell::from_mut_slice(data)).par_chunks(chunk_size);
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
                // SAFETY: the worker ranges `first_chunk..last_chunk` partition
                // `0..num_chunks`, so each chunk index is visited by exactly one
                // worker and distinct indices name disjoint element ranges.
                let chunk = unsafe { partitions.get_unchecked_chunk(chunk_index) }.into_mut_slice();
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
/// The fixed-size array and chunk derivation add no heap allocation after the
/// global executor is initialized. A first parallel call can allocate while
/// constructing that process-wide executor and its worker pool. The operation
/// lets callers fuse any homogeneous number of output-buffer passes without
/// adding another arity-specific operator.
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
    if !P::parallelize_chunks(length, num_chunks) || num_chunks <= 1 {
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

    let partitions = buffers
        .map(|buffer| WriterShard::new(MelinoeCell::from_mut_slice(buffer)).par_chunks(chunk_size));
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(num_chunks, move |chunk_index| {
            let chunks = core::array::from_fn(|buffer_index| {
                // SAFETY: safe construction of `buffers` proves the N mutable
                // slices do not alias; equal lengths were validated above, so
                // every partition view holds `num_chunks` entries and
                // `chunk_index` is in bounds for each. Distinct chunk indices
                // own pairwise-disjoint element ranges, so no two tasks alias.
                unsafe { partitions[buffer_index].get_unchecked_chunk(chunk_index) }
                    .into_mut_slice()
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
    if chunk_size == 0 || na == 0 {
        return;
    }
    let num_chunks = na.div_ceil(chunk_size);
    if !P::parallelize_chunks(na, num_chunks) || num_chunks <= 1 {
        a.chunks_mut(chunk_size)
            .zip(b.chunks_mut(chunk_size))
            .enumerate()
            .for_each(|(i, (ca, cb))| f(i, ca, cb));
        return;
    }
    let a_partitions = WriterShard::new(MelinoeCell::from_mut_slice(a)).par_chunks(chunk_size);
    let b_partitions = WriterShard::new(MelinoeCell::from_mut_slice(b)).par_chunks(chunk_size);
    // The paired pass stops at the shorter buffer, matching the sequential
    // `zip` path, so a `b` shorter than `a` is processed up to `b`'s extent
    // rather than reading out of bounds.
    let tasks = num_chunks.min(b_partitions.len());
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(tasks, move |c| {
            // SAFETY: `c < tasks <= a_partitions.len()` and `c <
            // b_partitions.len()`, so both indices are in bounds; distinct chunk
            // indices name disjoint element ranges in each buffer, and `a`/`b`
            // are distinct non-aliasing slices, so no two tasks alias within or
            // across the buffers.
            let ca = unsafe { a_partitions.get_unchecked_chunk(c) }.into_mut_slice();
            let cb = unsafe { b_partitions.get_unchecked_chunk(c) }.into_mut_slice();
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
    if !P::parallelize_chunks(na, num_chunks) || num_chunks <= 1 {
        a.chunks_mut(chunk_size)
            .zip(b.chunks_mut(chunk_size))
            .zip(c.chunks_mut(chunk_size))
            .zip(d.chunks_mut(chunk_size))
            .enumerate()
            .for_each(|(i, (((ca, cb), cc), cd))| f(i, ca, cb, cc, cd));
        return;
    }
    let a_partitions = WriterShard::new(MelinoeCell::from_mut_slice(a)).par_chunks(chunk_size);
    let b_partitions = WriterShard::new(MelinoeCell::from_mut_slice(b)).par_chunks(chunk_size);
    let c_partitions = WriterShard::new(MelinoeCell::from_mut_slice(c)).par_chunks(chunk_size);
    let d_partitions = WriterShard::new(MelinoeCell::from_mut_slice(d)).par_chunks(chunk_size);
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(num_chunks, move |chunk_index| {
            // SAFETY: the four buffers are distinct non-aliasing slices of equal
            // length (asserted above), so every partition view holds
            // `num_chunks` entries; distinct chunk indices own pairwise-disjoint
            // element ranges, so no two tasks alias.
            let ca = unsafe { a_partitions.get_unchecked_chunk(chunk_index) }.into_mut_slice();
            let cb = unsafe { b_partitions.get_unchecked_chunk(chunk_index) }.into_mut_slice();
            let cc = unsafe { c_partitions.get_unchecked_chunk(chunk_index) }.into_mut_slice();
            let cd = unsafe { d_partitions.get_unchecked_chunk(chunk_index) }.into_mut_slice();
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
    if !P::parallelize_chunks(na, num_chunks) || num_chunks <= 1 {
        a.chunks_mut(chunk_size)
            .zip(b.chunks_mut(chunk_size))
            .zip(c.chunks_mut(chunk_size))
            .enumerate()
            .for_each(|(i, ((ca, cb), cc))| f(i, ca, cb, cc));
        return;
    }
    let a_partitions = WriterShard::new(MelinoeCell::from_mut_slice(a)).par_chunks(chunk_size);
    let b_partitions = WriterShard::new(MelinoeCell::from_mut_slice(b)).par_chunks(chunk_size);
    let c_partitions = WriterShard::new(MelinoeCell::from_mut_slice(c)).par_chunks(chunk_size);
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(num_chunks, move |chunk_index| {
            // SAFETY: the three buffers are distinct non-aliasing slices of equal
            // length (asserted above), so every partition view holds
            // `num_chunks` entries; distinct chunk indices own pairwise-disjoint
            // element ranges, so no two tasks alias.
            let ca = unsafe { a_partitions.get_unchecked_chunk(chunk_index) }.into_mut_slice();
            let cb = unsafe { b_partitions.get_unchecked_chunk(chunk_index) }.into_mut_slice();
            let cc = unsafe { c_partitions.get_unchecked_chunk(chunk_index) }.into_mut_slice();
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
    if !P::parallelize_chunks(n, num_chunks) || num_chunks <= 1 {
        data.chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(i, c)| f(i, c));
        return;
    }
    let partitions = WriterShard::new(MelinoeCell::from_mut_slice(data)).par_chunks(chunk_size);
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(num_chunks, move |c| {
            // SAFETY: each chunk index is visited exactly once and names a
            // disjoint element range, so no two tasks alias.
            let chunk = unsafe { partitions.get_unchecked_chunk(c) }.into_mut_slice();
            f(c, chunk);
        })
        .expect("moirai global executor: for_each_chunk_mut_enumerated_with");
}
