use super::super::{Consumer, ParallelIterator, VecParIter};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ChunkSize(usize);

impl ChunkSize {
    fn new(value: usize) -> Self {
        assert!(value != 0, "chunk size must be non-zero");
        Self(value)
    }

    const fn get(self) -> usize {
        self.0
    }
}

/// Chunk adapter with Rayon-style non-empty chunk-size semantics.
pub struct Chunks<I> {
    base: I,
    chunk_size: ChunkSize,
}

impl<I> Chunks<I> {
    pub(in crate::parallel) fn new(base: I, chunk_size: usize) -> Self {
        Self {
            base,
            chunk_size: ChunkSize::new(chunk_size),
        }
    }

    pub(in crate::parallel) fn into_parts(self) -> (I, usize) {
        (self.base, self.chunk_size.get())
    }
}

impl<I> ParallelIterator for Chunks<I>
where
    I: ParallelIterator,
    I::Item: Sync + 'static,
{
    type Item = Vec<I::Item>;

    fn seq_items(self) -> Vec<Self::Item> {
        let chunk_size = self.chunk_size.get();
        let mut items = self.base.seq_items();
        let mut chunks = Vec::with_capacity(items.len().div_ceil(chunk_size));

        let tail_len = items.len() % chunk_size;
        let tail = if tail_len == 0 {
            None
        } else {
            Some(items.split_off(items.len() - tail_len))
        };

        while !items.is_empty() {
            chunks.push(items.split_off(items.len() - chunk_size));
        }
        chunks.reverse();

        if let Some(tail) = tail {
            chunks.push(tail);
        }

        chunks
    }

    /// # Why this stays sequential (chunk boundaries are logical positions)
    ///
    /// A chunk is defined by its position in the logical stream, and the source
    /// splits at its own midpoint, which is not in general a multiple of
    /// `chunk_size`. A shard chunking its own range alone would emit a short
    /// chunk at every internal shard boundary, so a stream whose only short
    /// chunk should be the tail would gain one per split. Aligning splits to
    /// chunk boundaries is a producer-side decision, not something this adapter
    /// can express by pushing into a consumer.
    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}
