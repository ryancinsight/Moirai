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
}
