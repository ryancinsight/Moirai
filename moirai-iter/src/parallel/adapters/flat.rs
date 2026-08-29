use super::super::{Consumer, FlatMapConsumer, ParallelIterator};
use std::ops::ControlFlow;

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

    fn seq_try_fold<Acc, B, FoldFn>(self, init: Acc, mut fold_fn: FoldFn) -> ControlFlow<B, Acc>
    where
        FoldFn: FnMut(Acc, Self::Item) -> ControlFlow<B, Acc>,
    {
        let flat_map_fn = self.flat_map_fn;
        self.base.seq_try_fold(init, move |accumulator, item| {
            flat_map_fn(item)
                .into_iter()
                .try_fold(accumulator, &mut fold_fn)
        })
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        // Push the expansion into the consumer and drive the base, the way `Map`
        // does. Materializing `seq_items()` first collected the whole flattened
        // stream into one vector before any split, discarding the source's
        // shards for every chain containing `flat_map()`. One input expanding to
        // many outputs does not block the push: each expansion depends on its
        // own item alone, so a shard produces exactly the sub-sequence a
        // sequential pass over its range would, and shards combine in logical
        // order.
        self.base
            .drive(FlatMapConsumer::new(consumer, self.flat_map_fn))
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

    fn seq_try_fold<Acc, B, FoldFn>(self, init: Acc, mut fold_fn: FoldFn) -> ControlFlow<B, Acc>
    where
        FoldFn: FnMut(Acc, Self::Item) -> ControlFlow<B, Acc>,
    {
        self.base.seq_try_fold(init, move |accumulator, item| {
            item.into_iter().try_fold(accumulator, &mut fold_fn)
        })
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        // Flattening is `flat_map` with the identity expansion, so it reuses
        // that consumer rather than duplicating the split and combine
        // forwarding.
        self.base
            .drive(FlatMapConsumer::new(consumer, |item: I::Item| item))
    }
}
