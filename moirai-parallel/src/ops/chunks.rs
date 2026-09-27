//! Mutable chunk operators over one or more disjoint buffers.
//!
//! Each operator plans the pass, consults the policy, runs a sequential
//! fallback, and otherwise dispatches the disjoint partitions through the global
//! executor. That skeleton — and its single `SAFETY` argument — lives once in
//! [`crate::ops::shards::drive_chunks`]; the operators below are thin wrappers
//! that name their buffer set and adapt their closure.

use crate::ops::shards::{BufferArray, chunk_shards, drive_chunks};
use crate::policy::ExecutionPolicy;
use melinoe::MelinoeCell;
use melinoe::region::WriterShard;
use moirai_executor::{SyncTask, global};

#[cfg(test)]
mod tests;

chunk_shards! {
    /// The single-buffer chunk operator's buffer set.
    struct SingleShards { data: T }
}

chunk_shards! {
    /// The paired chunk operator's buffer set.
    struct PairShards { a: A, b: B }
}

chunk_shards! {
    /// The triple chunk operator's buffer set.
    struct TripleShards { a: A, b: B, c: C }
}

chunk_shards! {
    /// The quad chunk operator's buffer set.
    struct QuadShards { a: A, b: B, c: C, d: D }
}

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
    drive_chunks::<P, _, _>(
        SingleShards { data },
        chunk_size,
        "moirai global executor: for_each_chunk_mut_with",
        |_chunk_index, (chunk,)| f(chunk),
    );
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
    buffers: [&mut [T]; N],
    chunk_size: usize,
    f: F,
) -> Result<(), ChunkBuffersError>
where
    P: ExecutionPolicy,
    T: Send,
    F: for<'chunk> Fn(usize, [&'chunk mut [T]; N]) + Send + Sync,
{
    let length = buffers.first().map_or(0, |first| first.len());
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
    drive_chunks::<P, _, _>(
        BufferArray { buffers },
        chunk_size,
        "moirai global executor: for_each_chunk_buffers_mut_enumerated_with",
        f,
    );
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
    // The paired pass stops at the shorter buffer, matching the sequential
    // `zip` path, so a `b` shorter than `a` is processed up to `b`'s extent
    // rather than reading out of bounds; `drive_chunks` takes the minimum
    // partition count for exactly that reason.
    drive_chunks::<P, _, _>(
        PairShards { a, b },
        chunk_size,
        "moirai global executor: for_each_chunk_pair_mut_enumerated_with",
        |chunk_index, (run_a, run_b)| f(chunk_index, run_a, run_b),
    );
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
    assert_eq!(
        a.len(),
        b.len(),
        "quad chunk buffers must have equal lengths"
    );
    assert_eq!(
        a.len(),
        c.len(),
        "quad chunk buffers must have equal lengths"
    );
    assert_eq!(
        a.len(),
        d.len(),
        "quad chunk buffers must have equal lengths"
    );
    drive_chunks::<P, _, _>(
        QuadShards { a, b, c, d },
        chunk_size,
        "moirai global executor: for_each_chunk_quad_mut_enumerated_with",
        |chunk_index, (run_a, run_b, run_c, run_d)| f(chunk_index, run_a, run_b, run_c, run_d),
    );
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
    assert_eq!(
        a.len(),
        b.len(),
        "triple chunk buffers must have equal lengths"
    );
    assert_eq!(
        a.len(),
        c.len(),
        "triple chunk buffers must have equal lengths"
    );
    drive_chunks::<P, _, _>(
        TripleShards { a, b, c },
        chunk_size,
        "moirai global executor: for_each_chunk_triple_mut_enumerated_with",
        |chunk_index, (run_a, run_b, run_c)| f(chunk_index, run_a, run_b, run_c),
    );
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
    drive_chunks::<P, _, _>(
        SingleShards { data },
        chunk_size,
        "moirai global executor: for_each_chunk_mut_enumerated_with",
        |chunk_index, (chunk,)| f(chunk_index, chunk),
    );
}
