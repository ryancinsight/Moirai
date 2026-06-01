use super::{
    ExponentialBlocks, Interleave, InterleaveShortest, IntoParallelIterator, ParallelIterator,
    StepBy, UniformBlocks,
};

/// Exact-size boundary for Moirai's bounded Rayon-style indexed source subset.
///
/// This trait deliberately covers source iterators with known cardinality. It
/// does not claim Rayon's full indexed producer/consumer adapter model.
pub trait IndexedParallelIterator: ParallelIterator {
    /// Return the exact number of logical items in the indexed source.
    fn len(&self) -> usize;

    /// Return whether the indexed source has no logical items.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Move all items into caller-provided storage.
    ///
    /// The destination vector is cleared but keeps its allocation, matching the
    /// bounded source contract for exact-size streams without requiring item
    /// cloning or allocating a second output vector.
    fn collect_into_vec(self, target: &mut Vec<Self::Item>) {
        target.clear();
        target.extend(self.seq_items());
    }

    /// Split pair items into caller-provided left and right storage.
    ///
    /// Both destination vectors are cleared but keep their allocations. Pair
    /// values are moved into their target sides exactly once, preserving the
    /// bounded exact-size source contract without cloning either side.
    fn unzip_into_vecs<A, B>(self, left: &mut Vec<A>, right: &mut Vec<B>)
    where
        Self: ParallelIterator<Item = (A, B)>,
        A: Send,
        B: Send,
    {
        let expected_len = self.len();
        let items = self.seq_items();
        debug_assert_eq!(items.len(), expected_len);

        left.clear();
        right.clear();
        left.reserve_exact(expected_len);
        right.reserve_exact(expected_len);

        for (left_item, right_item) in items {
            left.push(left_item);
            right.push(right_item);
        }
    }

    /// Alternately yield items from this source and another exact-size source.
    ///
    /// Values are moved from both sources into one logical stream. When one
    /// side is exhausted, remaining values from the other side are yielded.
    fn interleave<J>(self, other: J) -> Interleave<Self, J::Iter>
    where
        J: IntoParallelIterator<Item = Self::Item>,
        J::Iter: IndexedParallelIterator<Item = Self::Item>,
        Self::Item: Sync + 'static,
    {
        Interleave::new(self, other.into_par_iter())
    }

    /// Alternately yield items until the shorter exact-size source is consumed.
    ///
    /// This matches Rayon's indexed boundary: if the left source is longer,
    /// one trailing left item is retained after the final right item.
    fn interleave_shortest<J>(self, other: J) -> InterleaveShortest<Self, J::Iter>
    where
        J: IntoParallelIterator<Item = Self::Item>,
        J::Iter: IndexedParallelIterator<Item = Self::Item>,
        Self::Item: Sync + 'static,
    {
        InterleaveShortest::new(self, other.into_par_iter())
    }

    /// Yield every `step`th item from an exact-size source.
    ///
    /// The step size must be non-zero. Skipped items remain owned by the
    /// consumed source iterator and are dropped exactly once.
    fn step_by(self, step: usize) -> StepBy<Self>
    where
        Self::Item: Sync + 'static,
    {
        StepBy::new(self, step)
    }

    /// Convert this exact-size source into value-preserving exponential blocks.
    ///
    /// This bounded adapter preserves logical item order and exposes Rayon's
    /// block-hint API surface. It does not claim Rayon's full indexed
    /// producer/consumer block-splitting scheduler model.
    fn by_exponential_blocks(self) -> ExponentialBlocks<Self>
    where
        Self::Item: Sync + 'static,
    {
        ExponentialBlocks::new(self)
    }

    /// Convert this exact-size source into value-preserving uniform blocks.
    ///
    /// The block size must be non-zero. This bounded adapter validates the
    /// block-size contract and preserves logical item order without claiming
    /// Rayon's full block-splitting producer model.
    fn by_uniform_blocks(self, block_size: usize) -> UniformBlocks<Self>
    where
        Self::Item: Sync + 'static,
    {
        UniformBlocks::new(self, block_size)
    }
}

impl<I, J> IndexedParallelIterator for Interleave<I, J>
where
    I: IndexedParallelIterator,
    J: IndexedParallelIterator<Item = I::Item>,
    I::Item: Sync + 'static,
{
    fn len(&self) -> usize {
        self.left_len()
            .checked_add(self.right_len())
            .expect("overflow")
    }
}

impl<I, J> Interleave<I, J>
where
    I: IndexedParallelIterator,
    J: IndexedParallelIterator<Item = I::Item>,
{
    fn left_len(&self) -> usize {
        self.left.len()
    }

    fn right_len(&self) -> usize {
        self.right.len()
    }
}

impl<I, J> IndexedParallelIterator for InterleaveShortest<I, J>
where
    I: IndexedParallelIterator,
    J: IndexedParallelIterator<Item = I::Item>,
    I::Item: Sync + 'static,
{
    fn len(&self) -> usize {
        if self.left.len() <= self.right.len() {
            self.left.len().checked_mul(2).expect("overflow")
        } else {
            self.right
                .len()
                .checked_mul(2)
                .and_then(|len| len.checked_add(1))
                .expect("overflow")
        }
    }
}

impl<I> IndexedParallelIterator for StepBy<I>
where
    I: IndexedParallelIterator,
    I::Item: Sync + 'static,
{
    fn len(&self) -> usize {
        let len = self.base.len();
        if len == 0 {
            0
        } else {
            ((len - 1) / self.step()) + 1
        }
    }
}
