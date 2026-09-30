use super::iterator::ParallelIterator;

/// Consumer trait for parallel iterator operations.
pub trait Consumer<T>: Send + Sync {
    /// Result type produced by consuming an iterator.
    type Result: Send;

    /// Consume items from a parallel iterator.
    fn consume<I>(self, iter: I) -> Self::Result
    where
        I: ParallelIterator<Item = T>;

    /// Split the consumer for parallel processing.
    fn split_at(self, index: usize) -> (Self, Self)
    where
        Self: Sized;

    /// Combine results from split consumers.
    fn combine(left: Self::Result, right: Self::Result) -> Self::Result;
}

/// Trait for collections that can be extended in parallel.
pub trait ParallelExtend<T>: Send {
    /// Extend the collection with items from a parallel iterator.
    fn par_extend<I>(&mut self, par_iter: I)
    where
        I: ParallelIterator<Item = T>;
}
