use super::super::{Consumer, ParallelIterator, VecParIter};

/// Enumerate adapter for value-semantic index pairing.
pub struct Enumerate<I> {
    pub(super) base: I,
}

impl<I> Enumerate<I> {
    pub(crate) fn new(base: I) -> Self {
        Self { base }
    }
}

impl<I> ParallelIterator for Enumerate<I>
where
    I: ParallelIterator,
    I::Item: Sync + 'static,
{
    type Item = (usize, I::Item);

    fn seq_items(self) -> Vec<Self::Item> {
        self.base.seq_items().into_iter().enumerate().collect()
    }

    fn seq_items_window(self, skip: usize, take: Option<usize>) -> Vec<Self::Item> {
        self.base
            .seq_items_window(skip, take)
            .into_iter()
            .enumerate()
            .map(|(offset, item)| (skip + offset, item))
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

/// Copied adapter with standard reference-copy semantics.
pub struct Copied<I> {
    pub(super) base: I,
}

impl<I> Copied<I> {
    pub(crate) fn new(base: I) -> Self {
        Self { base }
    }
}

impl<'data, I, T> ParallelIterator for Copied<I>
where
    I: ParallelIterator<Item = &'data T>,
    T: Copy + Send + Sync + 'data + 'static,
{
    type Item = T;

    fn seq_items(self) -> Vec<Self::Item> {
        self.base.seq_items().into_iter().copied().collect()
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

/// Cloned adapter with standard reference-clone semantics.
pub struct Cloned<I> {
    pub(super) base: I,
}

impl<I> Cloned<I> {
    pub(crate) fn new(base: I) -> Self {
        Self { base }
    }
}

impl<'data, I, T> ParallelIterator for Cloned<I>
where
    I: ParallelIterator<Item = &'data T>,
    T: Clone + Send + Sync + 'data + 'static,
{
    type Item = T;

    fn seq_items(self) -> Vec<Self::Item> {
        self.base.seq_items().into_iter().cloned().collect()
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}
