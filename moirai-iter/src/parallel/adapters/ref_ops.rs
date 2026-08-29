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

    /// # Why this stays sequential (the logical-index boundary)
    ///
    /// Pairing an item with its logical index needs the count of items that
    /// precede it in the whole stream, and the non-indexed consumer protocol
    /// does not carry one. `Consumer::split_at` receives the *source's* split
    /// point — `left.len()` at the source being divided — which equals the
    /// logical offset only when nothing between the source and this adapter
    /// changes the element count. A `filter` below invalidates it, and the
    /// consumer cannot tell the two cases apart, so a shard handed that number
    /// as a base index would silently emit wrong indices for exactly the chains
    /// where it matters. No consumer in the tree reads the index today, which
    /// is why the mismatch is currently latent rather than a live defect.
    ///
    /// Supplying a true logical offset means an indexed producer boundary that
    /// knows each shard's position in the logical stream — the change recorded
    /// as the indexed adapter model in the Rayon adapter surface audit, not a
    /// consumer this adapter can push itself into. `positions`,
    /// `Map::positions`, and the borrowed position stream stay sequential for
    /// this same reason, as do `take`, `skip`, and `step_by`, whose retained
    /// items are a function of that same absent offset.
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
