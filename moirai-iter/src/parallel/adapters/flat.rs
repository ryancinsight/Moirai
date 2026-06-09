use super::super::{Consumer, ParallelIterator, VecParIter};

/// Flat-map adapter with standard left-to-right flattening semantics.
pub struct FlatMap<I, F> {
    pub(super) base: I,
    pub(super) flat_map_fn: F,
}

impl<I, F> FlatMap<I, F> {
    pub(crate) fn new(base: I, flat_map_fn: F) -> Self {
        Self { base, flat_map_fn }
    }
}

impl<I, F, U> ParallelIterator for FlatMap<I, F>
where
    I: ParallelIterator,
    F: Fn(I::Item) -> U + Send + Sync + Clone,
    U: IntoIterator,
    U::Item: Send + Sync + 'static,
{
    type Item = U::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        self.base
            .seq_items()
            .into_iter()
            .flat_map(self.flat_map_fn)
            .collect()
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

/// Flatten adapter with standard left-to-right nested stream semantics.
pub struct Flatten<I> {
    pub(super) base: I,
}

impl<I> Flatten<I> {
    pub(crate) fn new(base: I) -> Self {
        Self { base }
    }
}

impl<I> ParallelIterator for Flatten<I>
where
    I: ParallelIterator,
    I::Item: IntoIterator,
    <I::Item as IntoIterator>::Item: Send + Sync + 'static,
{
    type Item = <I::Item as IntoIterator>::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        self.base.seq_items().into_iter().flatten().collect()
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}
