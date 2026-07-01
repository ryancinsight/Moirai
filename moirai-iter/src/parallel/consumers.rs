use super::{Consumer, ParallelIterator};
use std::marker::PhantomData;

pub struct MapConsumer<C, F> {
    base: C,
    map_fn: F,
}

impl<C, F> MapConsumer<C, F> {
    pub(super) fn new(base: C, map_fn: F) -> Self {
        Self { base, map_fn }
    }
}

impl<C, F, T, R> Consumer<T> for MapConsumer<C, F>
where
    C: Consumer<R>,
    F: Fn(T) -> R + Send + Sync + Clone,
    T: Send,
    R: Send,
{
    type Result = C::Result;

    fn consume<I>(self, iter: I) -> Self::Result
    where
        I: ParallelIterator<Item = T>,
    {
        self.base.consume(iter.map(self.map_fn))
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.base.split_at(index);
        (
            MapConsumer::new(left, self.map_fn.clone()),
            MapConsumer::new(right, self.map_fn),
        )
    }

    fn combine(left: Self::Result, right: Self::Result) -> Self::Result {
        C::combine(left, right)
    }
}

pub struct FilterConsumer<C, F> {
    base: C,
    filter_fn: F,
}

impl<C, F> FilterConsumer<C, F> {
    pub(super) fn new(base: C, filter_fn: F) -> Self {
        Self { base, filter_fn }
    }
}

impl<C, F, T> Consumer<T> for FilterConsumer<C, F>
where
    C: Consumer<T>,
    F: Fn(&T) -> bool + Send + Sync + Clone,
    T: Send,
{
    type Result = C::Result;

    fn consume<I>(self, iter: I) -> Self::Result
    where
        I: ParallelIterator<Item = T>,
    {
        self.base.consume(iter.filter(self.filter_fn))
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.base.split_at(index);
        (
            FilterConsumer::new(left, self.filter_fn.clone()),
            FilterConsumer::new(right, self.filter_fn),
        )
    }

    fn combine(left: Self::Result, right: Self::Result) -> Self::Result {
        C::combine(left, right)
    }
}

pub struct InspectConsumer<C, F> {
    base: C,
    inspect_fn: F,
}

impl<C, F> InspectConsumer<C, F> {
    pub(super) fn new(base: C, inspect_fn: F) -> Self {
        Self { base, inspect_fn }
    }
}

impl<C, F, T> Consumer<T> for InspectConsumer<C, F>
where
    C: Consumer<T>,
    F: Fn(&T) + Send + Sync + Clone,
    T: Send + Sync + 'static,
{
    type Result = C::Result;

    fn consume<I>(self, iter: I) -> Self::Result
    where
        I: ParallelIterator<Item = T>,
    {
        self.base.consume(iter.inspect(self.inspect_fn))
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.base.split_at(index);
        (
            InspectConsumer::new(left, self.inspect_fn.clone()),
            InspectConsumer::new(right, self.inspect_fn),
        )
    }

    fn combine(left: Self::Result, right: Self::Result) -> Self::Result {
        C::combine(left, right)
    }
}

/// Parallel associative-reduce consumer: collects each shard's items, reduces
/// them left-to-right, and combines partial `Reduction`s in shard order. Single
/// SSOT backing both the `reduce` and `reduce_with` terminals — their contracts
/// (associative `Fn(Item, Item) -> Item`, `Option` result, `None` on empty) are
/// identical, so they share this one combine path rather than duplicating it.
pub struct ReduceConsumer<F> {
    reduce_fn: F,
}

impl<F> ReduceConsumer<F> {
    pub(super) fn new(reduce_fn: F) -> Self {
        Self { reduce_fn }
    }
}

pub struct Reduction<T, F> {
    value: Option<T>,
    reduce_fn: F,
}

impl<T, F> Reduction<T, F> {
    pub(super) fn new(value: Option<T>, reduce_fn: F) -> Self {
        Self { value, reduce_fn }
    }

    pub(super) fn into_value(self) -> Option<T> {
        self.value
    }
}

impl<F, T> Consumer<T> for ReduceConsumer<F>
where
    F: Fn(T, T) -> T + Send + Sync + Clone,
    T: Send + Sync + Clone,
{
    type Result = Reduction<T, F>;

    fn consume<I>(self, iter: I) -> Self::Result
    where
        I: ParallelIterator<Item = T>,
    {
        let data: Vec<T> = iter.drive(CollectConsumer::new());

        let reduce_fn = self.reduce_fn;
        let value = data.into_iter().reduce(&reduce_fn);
        Reduction::new(value, reduce_fn)
    }

    fn split_at(self, _index: usize) -> (Self, Self) {
        (
            ReduceConsumer::new(self.reduce_fn.clone()),
            ReduceConsumer::new(self.reduce_fn),
        )
    }

    fn combine(left: Self::Result, right: Self::Result) -> Self::Result {
        let reduce_fn = left.reduce_fn;
        let value = match (left.value, right.value) {
            (Some(left), Some(right)) => Some(reduce_fn(left, right)),
            (Some(value), None) | (None, Some(value)) => Some(value),
            (None, None) => None,
        };
        Reduction::new(value, reduce_fn)
    }
}

/// Collect consumer that gathers all items into a Vec.
pub struct CollectConsumer;

impl CollectConsumer {
    pub(super) fn new() -> Self {
        CollectConsumer
    }
}

impl<T: Send + Sync> Consumer<T> for CollectConsumer {
    type Result = Vec<T>;

    fn consume<I>(self, iter: I) -> Self::Result
    where
        I: ParallelIterator<Item = T>,
    {
        // seq_items() is the non-recursive base collection path. Calling drive here
        // would re-enter the consumer protocol and cause infinite recursion.
        iter.seq_items()
    }

    fn split_at(self, _index: usize) -> (Self, Self) {
        (CollectConsumer, CollectConsumer)
    }

    fn combine(mut left: Self::Result, mut right: Self::Result) -> Self::Result {
        left.append(&mut right);
        left
    }
}

pub struct FindConsumer<F> {
    predicate: F,
}

impl<F> FindConsumer<F> {
    pub(super) fn new(predicate: F) -> Self {
        Self { predicate }
    }
}

impl<F, T> Consumer<T> for FindConsumer<F>
where
    F: Fn(&T) -> bool + Send + Sync + Clone,
    T: Send + Sync,
{
    type Result = Option<T>;

    fn consume<I>(self, iter: I) -> Self::Result
    where
        I: ParallelIterator<Item = T>,
    {
        let data: Vec<T> = iter.drive(CollectConsumer::new());

        data.into_iter().find(|item| (self.predicate)(item))
    }

    fn split_at(self, _index: usize) -> (Self, Self) {
        (
            FindConsumer::new(self.predicate.clone()),
            FindConsumer::new(self.predicate),
        )
    }

    fn combine(left: Self::Result, right: Self::Result) -> Self::Result {
        left.or(right)
    }
}

pub struct NullConsumer<T> {
    _phantom: PhantomData<T>,
}

impl<T> NullConsumer<T> {
    pub(super) fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Send + Sync> Consumer<T> for NullConsumer<T> {
    type Result = ();

    fn consume<I>(self, iter: I) -> Self::Result
    where
        I: ParallelIterator<Item = T>,
    {
        // Drive side effects (e.g. from for_each) by collecting items and dropping.
        // seq_items applies any upstream map/filter transforms for their side effects.
        let _ = iter.seq_items();
    }

    fn split_at(self, _index: usize) -> (Self, Self) {
        (NullConsumer::new(), NullConsumer::new())
    }

    fn combine(_left: Self::Result, _right: Self::Result) -> Self::Result {}
}
