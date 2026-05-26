use super::super::{Consumer, ParallelIterator, VecNonCloneParIter};

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

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecNonCloneParIter::new(self.seq_items()))
    }
}
