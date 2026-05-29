mod chunks;
mod pair;
mod position;
mod side_effect;

pub use chunks::Chunks;
pub use pair::{Zip, ZipEq};
pub use position::{MapPositions, Positions};
pub use side_effect::{Inspect, PanicFuse};

use super::{
    fallible, Consumer, FilterConsumer, MapConsumer, ParallelIterator, TryStreamItem, VecParIter,
};

/// Map adapter for parallel iterators.
pub struct Map<I, F> {
    base: I,
    map_fn: F,
}

impl<I, F> Map<I, F> {
    pub(super) fn new(base: I, map_fn: F) -> Self {
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
    ) -> position::MapPositions<I, F, Predicate>
    where
        I: ParallelIterator,
        F: Fn(I::Item) -> R + Send + Sync + Clone,
        Predicate: Fn(R) -> bool + Send + Sync + Clone,
        R: Send,
    {
        position::MapPositions::new(self.base, self.map_fn, predicate)
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
    base: I,
    init: T,
    map_fn: F,
}

impl<I, T, F> MapWith<I, T, F> {
    pub(super) fn new(base: I, init: T, map_fn: F) -> Self {
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
    base: I,
    init: Init,
    map_fn: F,
}

impl<I, Init, F> MapInit<I, Init, F> {
    pub(super) fn new(base: I, init: Init, map_fn: F) -> Self {
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
    base: I,
    update_fn: F,
}

impl<I, F> Update<I, F> {
    pub(super) fn new(base: I, update_fn: F) -> Self {
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

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

/// Filter adapter for parallel iterators.
pub struct Filter<I, F> {
    base: I,
    filter_fn: F,
}

impl<I, F> Filter<I, F> {
    pub(super) fn new(base: I, filter_fn: F) -> Self {
        Self { base, filter_fn }
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
    base: I,
    filter_map_fn: F,
}

impl<I, F> FilterMap<I, F> {
    pub(super) fn new(base: I, filter_map_fn: F) -> Self {
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
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

/// While-some adapter with prefix-unwrapping semantics for optional streams.
pub struct WhileSome<I> {
    base: I,
}

impl<I> WhileSome<I> {
    pub(super) fn new(base: I) -> Self {
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

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

/// Flat-map adapter with standard left-to-right flattening semantics.
pub struct FlatMap<I, F> {
    base: I,
    flat_map_fn: F,
}

impl<I, F> FlatMap<I, F> {
    pub(super) fn new(base: I, flat_map_fn: F) -> Self {
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
    base: I,
}

impl<I> Flatten<I> {
    pub(super) fn new(base: I) -> Self {
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

/// Enumerate adapter for value-semantic index pairing.
pub struct Enumerate<I> {
    base: I,
}

impl<I> Enumerate<I> {
    pub(super) fn new(base: I) -> Self {
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
    base: I,
}

impl<I> Copied<I> {
    pub(super) fn new(base: I) -> Self {
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
    base: I,
}

impl<I> Cloned<I> {
    pub(super) fn new(base: I) -> Self {
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

/// Take adapter with prefix-bounded value semantics.
pub struct Take<I> {
    base: I,
    count: usize,
}

impl<I> Take<I> {
    pub(super) fn new(base: I, count: usize) -> Self {
        Self { base, count }
    }
}

impl<I> ParallelIterator for Take<I>
where
    I: ParallelIterator,
    I::Item: Sync + 'static,
{
    type Item = I::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        self.base.seq_items_window(0, Some(self.count))
    }

    fn seq_items_window(self, skip: usize, take: Option<usize>) -> Vec<Self::Item> {
        if skip >= self.count {
            return Vec::new();
        }

        let remaining = self.count - skip;
        let count = take.map_or(remaining, |count| count.min(remaining));
        self.base.seq_items_window(skip, Some(count))
    }

    fn seq_items_reversed(self) -> Vec<Self::Item> {
        self.base
            .seq_items_window(0, Some(self.count))
            .into_iter()
            .rev()
            .collect()
    }

    fn seq_items_reversed_prefix(self, count: usize) -> Vec<Self::Item> {
        let mut items = self.base.seq_items_window(0, Some(self.count));
        let keep = count.min(items.len());
        items.drain(..items.len().saturating_sub(keep));
        items.reverse();
        items
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

/// Skip adapter with prefix-discarding value semantics.
pub struct Skip<I> {
    base: I,
    count: usize,
}

impl<I> Skip<I> {
    pub(super) fn new(base: I, count: usize) -> Self {
        Self { base, count }
    }
}

impl<I> ParallelIterator for Skip<I>
where
    I: ParallelIterator,
    I::Item: Sync + 'static,
{
    type Item = I::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        self.base.seq_items_window(self.count, None)
    }

    fn seq_items_window(self, skip: usize, take: Option<usize>) -> Vec<Self::Item> {
        self.base
            .seq_items_window(self.count.saturating_add(skip), take)
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

/// Chain adapter with left-to-right concatenation semantics.
pub struct Chain<I, J> {
    left: I,
    right: J,
}

impl<I, J> Chain<I, J> {
    pub(super) fn new(left: I, right: J) -> Self {
        Self { left, right }
    }
}

impl<I, J> ParallelIterator for Chain<I, J>
where
    I: ParallelIterator,
    J: ParallelIterator<Item = I::Item>,
    I::Item: Sync + 'static,
{
    type Item = I::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        let mut left = self.left.seq_items();
        left.extend(self.right.seq_items());
        left
    }

    fn seq_items_reversed(self) -> Vec<Self::Item> {
        let mut items = self.right.seq_items_reversed();
        items.extend(self.left.seq_items_reversed());
        items
    }

    fn seq_items_reversed_prefix(self, count: usize) -> Vec<Self::Item> {
        let mut items = self.right.seq_items_reversed_prefix(count);
        if items.len() < count {
            items.extend(self.left.seq_items_reversed_prefix(count - items.len()));
        }
        items
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

/// Intersperse adapter with separator insertion between adjacent items.
pub struct Intersperse<I>
where
    I: ParallelIterator,
{
    base: I,
    separator: I::Item,
}

impl<I> Intersperse<I>
where
    I: ParallelIterator,
{
    pub(super) fn new(base: I, separator: I::Item) -> Self {
        Self { base, separator }
    }
}

impl<I> ParallelIterator for Intersperse<I>
where
    I: ParallelIterator,
    I::Item: Clone + Sync + 'static,
{
    type Item = I::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        let items = self.base.seq_items();
        if items.len() <= 1 {
            return items;
        }

        let mut output = Vec::with_capacity(items.len().saturating_mul(2).saturating_sub(1));
        let mut iter = items.into_iter();
        if let Some(first) = iter.next() {
            output.push(first);
        }
        for item in iter {
            output.push(self.separator.clone());
            output.push(item);
        }
        output
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

/// Reverse adapter with logical-order reversal semantics.
pub struct Rev<I> {
    base: I,
}

impl<I> Rev<I> {
    pub(super) fn new(base: I) -> Self {
        Self { base }
    }
}

impl<I> ParallelIterator for Rev<I>
where
    I: ParallelIterator,
    I::Item: Sync + 'static,
{
    type Item = I::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        self.base.seq_items_reversed()
    }

    fn seq_items_window(self, skip: usize, take: Option<usize>) -> Vec<Self::Item> {
        let count = take.unwrap_or(usize::MAX);
        let prefix = skip.saturating_add(count);
        let mut items = self.base.seq_items_reversed_prefix(prefix);
        if skip >= items.len() {
            return Vec::new();
        }
        items.drain(..skip);
        if let Some(count) = take {
            items.truncate(count);
        }
        items
    }

    fn seq_items_reversed(self) -> Vec<Self::Item> {
        self.base.seq_items()
    }

    fn seq_items_reversed_prefix(self, count: usize) -> Vec<Self::Item> {
        self.base.seq_items_window(0, Some(count))
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}
