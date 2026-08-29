use super::super::{Consumer, FilterConsumer, FilterMapConsumer, ParallelIterator, VecParIter};
use super::map::Map;
use super::pair::ZipEq;
use std::ops::ControlFlow;

/// Filter adapter for parallel iterators.
pub struct Filter<I, F> {
    pub(super) base: I,
    pub(super) filter_fn: F,
}

impl<I, F> Filter<I, F> {
    pub(crate) fn new(base: I, filter_fn: F) -> Self {
        Self { base, filter_fn }
    }
}

impl<I, J, MapFn, FilterFn, Mapped> Filter<Map<ZipEq<I, J>, MapFn>, FilterFn>
where
    I: ParallelIterator,
    J: ParallelIterator,
    I::Item: Sync + 'static,
    J::Item: Sync + 'static,
    MapFn: Fn((I::Item, J::Item)) -> Mapped + Send + Sync + Clone,
    FilterFn: Fn(&Mapped) -> bool + Send + Sync + Clone,
    Mapped: Send,
{
    /// Collect a zipped, mapped, filtered stream without intermediate pair vectors.
    pub fn collect<C>(self) -> C
    where
        C: FromIterator<Mapped> + Send,
    {
        let filter_fn = self.filter_fn;
        let map = self.base;
        let map_fn = map.map_fn;
        let zip = map.base;
        let left = zip.left.seq_items();
        let right = zip.right.seq_items();
        assert_eq!(
            left.len(),
            right.len(),
            "zip_eq requires equal input lengths"
        );

        left.into_iter()
            .zip(right)
            .map(map_fn)
            .filter(filter_fn)
            .collect()
    }
}

impl<I, F> ParallelIterator for Filter<I, F>
where
    I: ParallelIterator,
    F: Fn(&I::Item) -> bool + Send + Sync + Clone,
{
    type Item = I::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        self.base
            .seq_items()
            .into_iter()
            .filter(|x| (self.filter_fn)(x))
            .collect()
    }

    fn seq_try_fold<Acc, B, FoldFn>(self, init: Acc, mut fold_fn: FoldFn) -> ControlFlow<B, Acc>
    where
        FoldFn: FnMut(Acc, Self::Item) -> ControlFlow<B, Acc>,
    {
        let filter_fn = self.filter_fn;
        self.base.seq_try_fold(init, move |accumulator, item| {
            if filter_fn(&item) {
                fold_fn(accumulator, item)
            } else {
                ControlFlow::Continue(accumulator)
            }
        })
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        self.base
            .drive(FilterConsumer::new(consumer, self.filter_fn))
    }
}

/// Filter-map adapter with value-semantic optional output.
pub struct FilterMap<I, F> {
    pub(super) base: I,
    pub(super) filter_map_fn: F,
}

impl<I, F> FilterMap<I, F> {
    pub(crate) fn new(base: I, filter_map_fn: F) -> Self {
        Self {
            base,
            filter_map_fn,
        }
    }
}

impl<I, F, R> ParallelIterator for FilterMap<I, F>
where
    I: ParallelIterator,
    F: Fn(I::Item) -> Option<R> + Send + Sync + Clone,
    R: Send + Sync + 'static,
{
    type Item = R;

    fn seq_try_fold<Acc, B, FoldFn>(self, init: Acc, mut fold_fn: FoldFn) -> ControlFlow<B, Acc>
    where
        FoldFn: FnMut(Acc, Self::Item) -> ControlFlow<B, Acc>,
    {
        let filter_map_fn = self.filter_map_fn;
        self.base
            .seq_try_fold(init, move |accumulator, item| match filter_map_fn(item) {
                Some(mapped) => fold_fn(accumulator, mapped),
                None => ControlFlow::Continue(accumulator),
            })
    }

    fn seq_items(self) -> Vec<Self::Item> {
        self.base
            .seq_items()
            .into_iter()
            .filter_map(self.filter_map_fn)
            .collect()
    }

    fn drive<C, R2>(self, consumer: C) -> R2
    where
        C: Consumer<Self::Item, Result = R2> + Send + Sync,
        R2: Send,
    {
        // Push the filtering map into the consumer and drive the base, the way
        // `Filter` and `Map` do. Materializing `seq_items()` first collected the
        // whole logical stream into one vector before any split, discarding the
        // source's shards for every chain containing `filter_map()`.
        self.base
            .drive(FilterMapConsumer::new(consumer, self.filter_map_fn))
    }
}

/// While-some adapter with prefix-unwrapping semantics for optional streams.
pub struct WhileSome<I> {
    pub(super) base: I,
}

impl<I> WhileSome<I> {
    pub(crate) fn new(base: I) -> Self {
        Self { base }
    }
}

impl<I, T> ParallelIterator for WhileSome<I>
where
    I: ParallelIterator<Item = Option<T>>,
    T: Send + Sync + 'static,
{
    type Item = T;

    fn seq_items(self) -> Vec<Self::Item> {
        self.base
            .seq_items()
            .into_iter()
            .map_while(|item| item)
            .collect()
    }

    /// # Why this stays sequential
    ///
    /// `while_some` stops the whole logical stream at the first `None`, so
    /// whether a shard's items survive depends on whether an earlier shard
    /// already stopped — a cross-shard dependency the consumer protocol cannot
    /// express. A shard that ran `map_while` over its own range alone would
    /// keep items that follow an earlier shard's `None`, which is a different
    /// stream, not a reordering of the same one. `take_any_while` and
    /// `skip_any_while` stay sequential for this same prefix dependency.
    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}
