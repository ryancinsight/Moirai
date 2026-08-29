use super::super::{Consumer, MapConsumer, ParallelIterator, VecParIter};
use std::ops::ControlFlow;

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

    fn seq_try_fold<Acc, B, FoldFn>(self, init: Acc, mut fold_fn: FoldFn) -> ControlFlow<B, Acc>
    where
        FoldFn: FnMut(Acc, Self::Item) -> ControlFlow<B, Acc>,
    {
        self.base
            .seq_try_fold(init, move |accumulator, item| fold_fn(accumulator, *item))
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        // Push the copy into the consumer and drive the base, the way `Map`
        // does. Materializing `seq_items()` first collected the whole stream
        // into one vector before any split, which discarded the borrowed
        // source's zero-copy split for every chain containing `copied()`.
        self.base
            .drive(MapConsumer::new(consumer, |item: &'data T| *item))
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

    fn seq_try_fold<Acc, B, FoldFn>(self, init: Acc, mut fold_fn: FoldFn) -> ControlFlow<B, Acc>
    where
        FoldFn: FnMut(Acc, Self::Item) -> ControlFlow<B, Acc>,
    {
        self.base.seq_try_fold(init, move |accumulator, item| {
            fold_fn(accumulator, item.clone())
        })
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        // Push the clone into the consumer, as `Copied` does above.
        self.base
            .drive(MapConsumer::new(consumer, |item: &'data T| item.clone()))
    }
}
