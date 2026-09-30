//! The one partition skeleton every mutable multi-buffer operator shares.
//!
//! The chunk operators in [`super::chunks`] and the unit-task operators in
//! [`super::unit_tasks`] are all the same shape: size the pass, consult the
//! execution policy, run a sequential fallback, then brand the caller's buffers
//! in place and hand the global executor one Melinoe partition view per buffer,
//! reconstructing each task's disjoint run from its index. That shape lives here
//! once — [`drive_chunks`] for the fixed-width chunk operators, and
//! `unit_tasks::driver::drive_unit_tasks` for the byte-sized unit-task
//! operators, both over the [`ChunkShards`] buffer-set trait — so each public
//! entry point is a thin wrapper that names its buffer signature and adapts its
//! closure, rather than restating the skeleton per arity.
//!
//! One disjointness argument covers every site, because the skeleton is the only
//! place the unsafe access happens: `for_each_indexed(count, f)` invokes `f` with
//! each index in `0..count` exactly once — the contract
//! [`ParChunks::get_unchecked_chunk`] documents — and the buffer set is a fixed
//! tuple of distinct `&mut [_]` arguments, so distinct indices name disjoint
//! element ranges in each buffer and no two tasks form `&mut` to one element.

use melinoe::MelinoeCell;
use melinoe::region::{ParChunks, WriterShard};
use moirai_executor::{SyncTask, global};

use crate::policy::ExecutionPolicy;

/// A fixed set of one or more distinct, equally long mutable buffers traversed
/// in lockstep by a chunk or unit-task operator.
///
/// Implemented for arities 1 through 4 by the [`chunk_shards`] macro and, for
/// the homogeneous `const N` buffer array, by [`BufferArray`]. The associated
/// [`Views`](Self::Views) is the Melinoe partition view built once per pass;
/// [`Chunk`](Self::Chunk) is the run argument handed to the closure.
pub(super) trait ChunkShards {
    /// Element type of the first buffer — the pass's nominal element.
    type Element: Send;

    /// The `&mut [..]` runs the closure receives for one chunk index, in buffer
    /// order.
    type Chunk;

    /// One [`ParChunks`] partition view per buffer, holding the buffers' borrow.
    type Views: Send + Sync;

    /// Element count of the first buffer.
    fn len(&self) -> usize;

    /// Chunk indices the pass visits: the minimum over the buffers of
    /// `len.div_ceil(chunk_size)`. Equal-length buffers make this the first
    /// buffer's count; the paired form's shorter buffer makes `min` the `zip`
    /// extent the sequential path has always used.
    fn task_count(&self, chunk_size: usize) -> usize;

    /// Partition the buffers into `chunk_size`-cell views.
    fn split(self, chunk_size: usize) -> Self::Views;

    /// The chunks for `index`.
    ///
    /// # Safety
    ///
    /// `index` must be less than `self.task_count(chunk_size)` — in bounds for
    /// every partition view — and each index must be requested at most once
    /// while a returned chunk is live.
    unsafe fn chunk(views: &Self::Views, index: usize) -> Self::Chunk;
}

/// Implements [`ChunkShards`] for a distinct-typed buffer set of arity 1..=4.
///
/// Everything in the expansion is path-qualified, so invoking it needs nothing
/// in scope beyond the buffer element types it names.
macro_rules! chunk_shards {
    (
        $(#[$attr:meta])*
        struct $name:ident {
            $first_field:ident : $first_ty:ident
            $(, $field:ident : $ty:ident)* $(,)?
        }
    ) => {
        $(#[$attr])*
        pub(in crate::ops) struct $name<'buffer, $first_ty $(, $ty)*> {
            pub(in crate::ops) $first_field: &'buffer mut [$first_ty]
            $(, pub(in crate::ops) $field: &'buffer mut [$ty])*
        }

        impl<'buffer, $first_ty: Send $(, $ty: Send)*> $crate::ops::shards::ChunkShards
            for $name<'buffer, $first_ty $(, $ty)*>
        {
            type Element = $first_ty;
            type Chunk = (&'buffer mut [$first_ty] $(, &'buffer mut [$ty])* ,);
            type Views = (
                melinoe::region::ParChunks<'buffer, 'buffer, $first_ty>
                $(, melinoe::region::ParChunks<'buffer, 'buffer, $ty>)*
                ,
            );

            #[inline]
            fn len(&self) -> usize {
                self.$first_field.len()
            }

            #[inline]
            fn task_count(&self, chunk_size: usize) -> usize {
                [self.$first_field.len() $(, self.$field.len())*]
                    .into_iter()
                    .map(|len| len.div_ceil(chunk_size))
                    .min()
                    .unwrap_or(0)
            }

            #[inline]
            fn split(self, chunk_size: usize) -> Self::Views {
                let Self { $first_field $(, $field)* } = self;
                (
                    melinoe::region::WriterShard::new(
                        melinoe::MelinoeCell::from_mut_slice($first_field),
                    )
                    .par_chunks(chunk_size)
                    $(
                        , melinoe::region::WriterShard::new(
                            melinoe::MelinoeCell::from_mut_slice($field),
                        )
                        .par_chunks(chunk_size)
                    )*
                    ,
                )
            }

            #[inline]
            unsafe fn chunk(views: &Self::Views, index: usize) -> Self::Chunk {
                let ($first_field $(, $field)* ,) = views;
                (
                    // SAFETY: delegated to this method's contract; `index` is in
                    // bounds for every view because it is below `task_count`.
                    unsafe { $first_field.get_unchecked_chunk(index) }.into_mut_slice()
                    $(
                        , unsafe { $field.get_unchecked_chunk(index) }.into_mut_slice()
                    )*
                    ,
                )
            }
        }
    };
}

pub(super) use chunk_shards;

/// One buffer set behind the same [`ChunkShards`] contract.
pub(super) struct BufferArray<'buffer, T, const N: usize> {
    /// The buffers, in order.
    pub(super) buffers: [&'buffer mut [T]; N],
}

impl<'buffer, T: Send, const N: usize> ChunkShards for BufferArray<'buffer, T, N> {
    type Element = T;
    type Chunk = [&'buffer mut [T]; N];
    type Views = [ParChunks<'buffer, 'buffer, T>; N];

    #[inline]
    fn len(&self) -> usize {
        self.buffers.first().map_or(0, |buffer| buffer.len())
    }

    #[inline]
    fn task_count(&self, chunk_size: usize) -> usize {
        self.buffers
            .iter()
            .map(|buffer| buffer.len().div_ceil(chunk_size))
            .min()
            .unwrap_or(0)
    }

    #[inline]
    fn split(self, chunk_size: usize) -> Self::Views {
        self.buffers.map(|buffer| {
            WriterShard::new(MelinoeCell::from_mut_slice(buffer)).par_chunks(chunk_size)
        })
    }

    #[inline]
    unsafe fn chunk(views: &Self::Views, index: usize) -> Self::Chunk {
        core::array::from_fn(|buffer| {
            // SAFETY: delegated to this method's contract; `index` is in bounds
            // for every view because it is below `task_count`.
            unsafe { views[buffer].get_unchecked_chunk(index) }.into_mut_slice()
        })
    }
}

/// Drive the fixed-width chunk operators over one buffer set.
///
/// `context` is appended to the executor error when the parallel branch cannot
/// be scheduled; each operator keeps its own so a failure names the entry point
/// that raised it.
pub(super) fn drive_chunks<P, B, F>(buffers: B, chunk_size: usize, context: &'static str, f: F)
where
    P: ExecutionPolicy,
    B: ChunkShards,
    F: Fn(usize, B::Chunk) + Send + Sync,
{
    let len = buffers.len();
    if len == 0 || chunk_size == 0 {
        return;
    }
    let num_chunks = len.div_ceil(chunk_size);
    let tasks = buffers.task_count(chunk_size);
    let views = buffers.split(chunk_size);
    if !P::parallelize_chunks(len, num_chunks) || num_chunks <= 1 {
        for index in 0..tasks {
            // SAFETY: `index < tasks`, the minimum partition count over the
            // buffers, so it is in bounds for every view; each index is visited
            // once by this loop, so no two chunks alias.
            let chunk = unsafe { B::chunk(&views, index) };
            f(index, chunk);
        }
        return;
    }
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(tasks, move |index| {
            // SAFETY: `for_each_indexed(tasks, _)` visits each index in
            // `0..tasks` exactly once — the contract `get_unchecked_chunk`
            // documents — and `tasks` is at most every buffer's partition count,
            // so each index is in bounds for each view. The buffers are distinct
            // `&mut [_]` arguments that cannot alias, and distinct chunk indices
            // name disjoint element ranges in each buffer, so no two tasks form
            // `&mut` to one element.
            let chunk = unsafe { B::chunk(&views, index) };
            f(index, chunk);
        })
        .expect(context);
}
