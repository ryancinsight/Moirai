use super::ParallelIterator;

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
}
