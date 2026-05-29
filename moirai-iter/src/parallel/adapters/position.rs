use super::super::{Consumer, ParallelIterator, VecParIter};

/// Position adapter that yields logical indices for matching items.
pub struct Positions<I, F> {
    base: I,
    predicate: F,
}

/// Fused map-position adapter that yields logical indices without materializing mapped items.
pub struct MapPositions<I, MapFn, Predicate> {
    base: I,
    map_fn: MapFn,
    predicate: Predicate,
}

impl<I, F> Positions<I, F> {
    pub(in crate::parallel) fn new(base: I, predicate: F) -> Self {
        Self { base, predicate }
    }
}

impl<I, MapFn, Predicate> MapPositions<I, MapFn, Predicate> {
    pub(in crate::parallel) fn new(base: I, map_fn: MapFn, predicate: Predicate) -> Self {
        Self {
            base,
            map_fn,
            predicate,
        }
    }
}

impl<I, F> ParallelIterator for Positions<I, F>
where
    I: ParallelIterator,
    F: Fn(I::Item) -> bool + Send + Sync + Clone,
{
    type Item = usize;

    fn seq_items(self) -> Vec<Self::Item> {
        self.base
            .seq_items()
            .into_iter()
            .enumerate()
            .filter_map(|(index, item)| (self.predicate)(item).then_some(index))
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

impl<I, MapFn, Predicate, Mapped> ParallelIterator for MapPositions<I, MapFn, Predicate>
where
    I: ParallelIterator,
    MapFn: Fn(I::Item) -> Mapped + Send + Sync + Clone,
    Predicate: Fn(Mapped) -> bool + Send + Sync + Clone,
    Mapped: Send,
{
    type Item = usize;

    fn seq_items(self) -> Vec<Self::Item> {
        let map_fn = self.map_fn;
        let predicate = self.predicate;

        self.base
            .seq_items()
            .into_iter()
            .enumerate()
            .filter_map(|(index, item)| predicate(map_fn(item)).then_some(index))
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
