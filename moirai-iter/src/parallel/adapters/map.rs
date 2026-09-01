use super::super::{fallible, Consumer, MapConsumer, ParallelIterator, TryStreamItem, VecParIter};
use std::ops::ControlFlow;

/// Map adapter for parallel iterators.
pub struct Map<I, F> {
    pub(super) base: I,
    pub(super) map_fn: F,
}

impl<I, F> Map<I, F> {
    pub(crate) fn new(base: I, map_fn: F) -> Self {
        Self { base, map_fn }
    }

    /// Reduce a mapped fallible stream without materializing mapped items first.
    pub fn try_reduce_with<ReduceFn, R>(self, reduce_fn: ReduceFn) -> Option<R>
    where
        I: ParallelIterator,
        F: Fn(I::Item) -> R + Send + Sync + Clone,
        R: TryStreamItem,
        ReduceFn: Fn(<R as TryStreamItem>::Output, <R as TryStreamItem>::Output) -> R
            + Send
            + Sync
            + Clone,
    {
        fallible::try_reduce_with_items(
            self.base.seq_items().into_iter().map(self.map_fn),
            reduce_fn,
        )
    }

    /// Return mapped logical indices without materializing the mapped stream.
    pub fn positions<Predicate, R>(
        self,
        predicate: Predicate,
    ) -> super::position::MapPositions<I, F, Predicate>
    where
        I: ParallelIterator,
        F: Fn(I::Item) -> R + Send + Sync + Clone,
        Predicate: Fn(R) -> bool + Send + Sync + Clone,
        R: Send,
    {
        super::position::MapPositions::new(self.base, self.map_fn, predicate)
    }
}

impl<I, F, R> ParallelIterator for Map<I, F>
where
    I: ParallelIterator,
    F: Fn(I::Item) -> R + Send + Sync + Clone,
    R: Send,
{
    type Item = R;

    fn seq_items(self) -> Vec<Self::Item> {
        self.base.seq_items().into_iter().map(self.map_fn).collect()
    }

    fn seq_iter(self) -> impl Iterator<Item = Self::Item> {
        self.base.seq_iter().map(self.map_fn)
    }

    fn seq_try_fold<Acc, B, FoldFn>(self, init: Acc, mut fold_fn: FoldFn) -> ControlFlow<B, Acc>
    where
        FoldFn: FnMut(Acc, Self::Item) -> ControlFlow<B, Acc>,
    {
        let map_fn = self.map_fn;
        self.base.seq_try_fold(init, move |accumulator, item| {
            fold_fn(accumulator, map_fn(item))
        })
    }

    fn seq_items_window(self, skip: usize, take: Option<usize>) -> Vec<Self::Item> {
        self.base
            .seq_items_window(skip, take)
            .into_iter()
            .map(self.map_fn)
            .collect()
    }

    fn seq_items_reversed(self) -> Vec<Self::Item> {
        self.base
            .seq_items_reversed()
            .into_iter()
            .map(self.map_fn)
            .collect()
    }

    fn seq_items_reversed_prefix(self, count: usize) -> Vec<Self::Item> {
        self.base
            .seq_items_reversed_prefix(count)
            .into_iter()
            .map(self.map_fn)
            .collect()
    }

    fn drive<C, R2>(self, consumer: C) -> R2
    where
        C: Consumer<Self::Item, Result = R2> + Send + Sync,
        R2: Send,
    {
        self.base.drive(MapConsumer::new(consumer, self.map_fn))
    }

    fn position_first<P>(self, predicate: P) -> Option<usize>
    where
        P: Fn(Self::Item) -> bool + Send + Sync + Clone,
    {
        self.base
            .seq_items()
            .into_iter()
            .map(self.map_fn)
            .position(predicate)
    }

    fn position_any<P>(self, predicate: P) -> Option<usize>
    where
        P: Fn(Self::Item) -> bool + Send + Sync + Clone,
    {
        self.position_first(predicate)
    }

    fn position_last<P>(self, predicate: P) -> Option<usize>
    where
        P: Fn(Self::Item) -> bool + Send + Sync + Clone,
    {
        self.base
            .seq_items()
            .into_iter()
            .map(self.map_fn)
            .rposition(predicate)
    }

    fn try_reduce<Identity, ReduceFn, T, E>(
        self,
        identity: Identity,
        reduce_fn: ReduceFn,
    ) -> Result<T, E>
    where
        Identity: Fn() -> T + Send + Sync + Clone,
        ReduceFn: Fn(T, T) -> Result<T, E> + Send + Sync + Clone,
        R: Into<Result<T, E>>,
        T: Send,
        E: Send,
    {
        let mut accumulator = identity();
        for item in self.base.seq_items().into_iter().map(self.map_fn) {
            accumulator = reduce_fn(accumulator, item.into()?)?;
        }
        Ok(accumulator)
    }
}

/// Map adapter with cloned per-operation state.
pub struct MapWith<I, T, F> {
    pub(super) base: I,
    pub(super) init: T,
    pub(super) map_fn: F,
}

impl<I, T, F> MapWith<I, T, F> {
    pub(crate) fn new(base: I, init: T, map_fn: F) -> Self {
        Self { base, init, map_fn }
    }
}

impl<I, T, F, R> ParallelIterator for MapWith<I, T, F>
where
    I: ParallelIterator,
    T: Send + Clone,
    F: Fn(&mut T, I::Item) -> R + Send + Sync + Clone,
    R: Send + Sync + 'static,
{
    type Item = R;

    fn seq_items(self) -> Vec<Self::Item> {
        let mut state = self.init;
        self.base
            .seq_items()
            .into_iter()
            .map(|item| (self.map_fn)(&mut state, item))
            .collect()
    }

    /// # Why this stays sequential
    ///
    /// One state value threads through the whole stream, so `map_fn` observes
    /// every prior item's effect on it. Giving each shard its own clone is a
    /// different contract, not a parallelisation of this one — the same reason
    /// `for_each_with` and `try_for_each_with` stay sequential.
    fn drive<C, R2>(self, consumer: C) -> R2
    where
        C: Consumer<Self::Item, Result = R2> + Send + Sync,
        R2: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

/// Map adapter with lazily initialized state.
pub struct MapInit<I, Init, F> {
    pub(super) base: I,
    pub(super) init: Init,
    pub(super) map_fn: F,
}

impl<I, Init, F> MapInit<I, Init, F> {
    pub(crate) fn new(base: I, init: Init, map_fn: F) -> Self {
        Self { base, init, map_fn }
    }
}

impl<I, Init, T, F, R> ParallelIterator for MapInit<I, Init, F>
where
    I: ParallelIterator,
    Init: Fn() -> T + Send + Sync + Clone,
    T: Send,
    F: Fn(&mut T, I::Item) -> R + Send + Sync + Clone,
    R: Send + Sync + 'static,
{
    type Item = R;

    fn seq_items(self) -> Vec<Self::Item> {
        let mut state = (self.init)();
        self.base
            .seq_items()
            .into_iter()
            .map(|item| (self.map_fn)(&mut state, item))
            .collect()
    }

    /// # Why this stays sequential
    ///
    /// Threaded state, for the reason given on [`MapWith`].
    fn drive<C, R2>(self, consumer: C) -> R2
    where
        C: Consumer<Self::Item, Result = R2> + Send + Sync,
        R2: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

/// Update adapter that mutates each item before yielding it.
pub struct Update<I, F> {
    pub(super) base: I,
    pub(super) update_fn: F,
}

impl<I, F> Update<I, F> {
    pub(crate) fn new(base: I, update_fn: F) -> Self {
        Self { base, update_fn }
    }
}

impl<I, F> ParallelIterator for Update<I, F>
where
    I: ParallelIterator,
    F: Fn(&mut I::Item) + Send + Sync + Clone,
    I::Item: Sync + 'static,
{
    type Item = I::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        self.base
            .seq_items()
            .into_iter()
            .map(|mut item| {
                (self.update_fn)(&mut item);
                item
            })
            .collect()
    }

    fn seq_try_fold<Acc, B, FoldFn>(self, init: Acc, mut fold_fn: FoldFn) -> ControlFlow<B, Acc>
    where
        FoldFn: FnMut(Acc, Self::Item) -> ControlFlow<B, Acc>,
    {
        let update_fn = self.update_fn;
        self.base.seq_try_fold(init, move |accumulator, mut item| {
            update_fn(&mut item);
            fold_fn(accumulator, item)
        })
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        // One input, one output, no state between items: the same push `Map`
        // makes. Materializing `seq_items()` first discarded the source's shards
        // Every chain containing `update()` lost its source shards.
        let update_fn = self.update_fn;
        self.base
            .drive(MapConsumer::new(consumer, move |mut item: I::Item| {
                update_fn(&mut item);
                item
            }))
    }
}
