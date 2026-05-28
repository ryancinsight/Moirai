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
}
