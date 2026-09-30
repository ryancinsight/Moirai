use super::iterator::ParallelIterator;

/// Extension trait for collections to create parallel iterators.
pub trait IntoParallelIterator {
    /// Element type yielded by the iterator.
    type Item: Send;
    /// Parallel iterator produced by conversion.
    type Iter: ParallelIterator<Item = Self::Item>;

    /// Convert `self` into a parallel iterator.
    fn into_par_iter(self) -> Self::Iter;
}

/// Extension trait for collection references to create parallel iterators.
pub trait IntoParallelRefIterator<'data> {
    /// Element type yielded by the iterator.
    type Item: Send + Sync + 'data;
    /// Parallel iterator produced by conversion.
    type Iter: ParallelIterator<Item = Self::Item>;

    /// Create a parallel iterator over references to `self`.
    fn par_iter(&'data self) -> Self::Iter;
}
