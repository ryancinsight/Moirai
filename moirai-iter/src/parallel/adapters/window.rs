use super::super::{Consumer, ParallelIterator, VecParIter};

/// Deterministic predicate-window adapter for the audited `take_any_while` subset.
pub struct TakeAnyWhile<I, F> {
    base: I,
    predicate: F,
}

/// Deterministic predicate-window adapter for the audited `skip_any_while` subset.
pub struct SkipAnyWhile<I, F> {
    base: I,
    predicate: F,
}

impl<I, F> TakeAnyWhile<I, F> {
    pub(in crate::parallel) fn new(base: I, predicate: F) -> Self {
        Self { base, predicate }
    }
}

impl<I, F> SkipAnyWhile<I, F> {
    pub(in crate::parallel) fn new(base: I, predicate: F) -> Self {
        Self { base, predicate }
    }
}

impl<I, F> ParallelIterator for TakeAnyWhile<I, F>
where
    I: ParallelIterator,
    F: Fn(&I::Item) -> bool + Send + Sync + Clone,
    I::Item: Sync + 'static,
{
    type Item = I::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        let predicate = self.predicate;
        let mut retained = Vec::new();

        for item in self.base.seq_items() {
            if !predicate(&item) {
                break;
            }
            retained.push(item);
        }

        retained
    }

    /// # Why this stays sequential
    ///
    /// The retained prefix ends at the first item failing the predicate
    /// anywhere in the stream, so a shard cannot decide its own items without
    /// knowing whether an earlier shard already stopped — the prefix dependency
    /// documented on [`WhileSome`](super::filter::WhileSome).
    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

impl<I, F> ParallelIterator for SkipAnyWhile<I, F>
where
    I: ParallelIterator,
    F: Fn(&I::Item) -> bool + Send + Sync + Clone,
    I::Item: Sync + 'static,
{
    type Item = I::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        let predicate = self.predicate;
        let mut items = self.base.seq_items().into_iter();

        for item in items.by_ref() {
            if !predicate(&item) {
                let mut retained = vec![item];
                retained.extend(items);
                return retained;
            }
        }

        Vec::new()
    }

    /// # Why this stays sequential
    ///
    /// The discarded prefix ends at the first item failing the predicate
    /// anywhere in the stream, a prefix dependency per [`TakeAnyWhile`].
    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}
