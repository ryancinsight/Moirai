use super::super::{
    fallible, Consumer, MapConsumer, ParallelIterator, TryStreamItem, VecParIter,
};
use super::chunks::Chunks;
use super::pair::Interleave;
use super::stride::StepBy;
use super::ref_ops::Enumerate;

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

impl<I, MapFn, Mapped> Map<Chunks<I>, MapFn>
where
    I: ParallelIterator,
    I::Item: Sync + 'static,
    MapFn: Fn(Vec<I::Item>) -> Mapped + Send + Sync + Clone,
    Mapped: Send,
{
    /// Sum mapped chunk outputs without materializing the chunk-output stream.
    pub fn sum<S>(self) -> S
    where
        S: std::iter::Sum<Mapped> + Send,
    {
        let map_fn = self.map_fn;
        let (base, chunk_size) = self.base.into_parts();
        let mut items = base.seq_items().into_iter();

        std::iter::from_fn(move || {
            let chunk: Vec<_> = items.by_ref().take(chunk_size).collect();
            (!chunk.is_empty()).then(|| map_fn(chunk))
        })
        .sum()
    }
}

impl<T, MapFn, Mapped> Map<Enumerate<Interleave<StepBy<VecParIter<T>>, VecParIter<T>>>, MapFn>
where
    T: Send + Sync + 'static,
    MapFn: Fn((usize, T)) -> Mapped + Send + Sync + Clone,
    Mapped: Send,
{
    /// Sum mapped vector-backed interleaved index/value pairs without building pair streams.
    pub fn sum<S>(self) -> S
    where
        S: std::iter::Sum<Mapped> + Send,
    {
        let map_fn = self.map_fn;
        let interleave = self.base.base;
        let step = interleave.left.step();
        let left = interleave.left.base.into_vec();
        let right = interleave.right.into_vec();
        let left_count = if left.is_empty() {
            0
        } else {
            ((left.len() - 1) / step) + 1
        };
        let right_count = right.len();
        let paired_count = left_count.min(right_count);
        let tail_start = paired_count
            .checked_mul(2)
            .expect("interleave index overflow");

        if left_count <= right_count {
            let mut left = left.into_iter().step_by(step);
            let mut right = right.into_iter();
            let mut index = 0usize;
            let mut pending_right = None;
            let mut paired_done = false;

            std::iter::from_fn(move || {
                if let Some(mapped) = pending_right.take() {
                    return Some(mapped);
                }

                if !paired_done {
                    if let Some(left_value) = left.next() {
                        let right_value = right
                            .next()
                            .expect("right side must cover paired interleave item");
                        let left_index = index;
                        let right_index = index.checked_add(1).expect("interleave index overflow");
                        index = index.checked_add(2).expect("interleave index overflow");
                        pending_right = Some(map_fn((right_index, right_value)));
                        return Some(map_fn((left_index, left_value)));
                    }
                    paired_done = true;
                    index = tail_start;
                }

                right.next().map(|value| {
                    let mapped = map_fn((index, value));
                    index = index.checked_add(1).expect("interleave index overflow");
                    mapped
                })
            })
            .sum()
        } else {
            let mut left = left.into_iter().step_by(step);
            let mut right = right.into_iter();
            let mut index = 0usize;
            let mut pending_right = None;
            let mut paired_done = false;

            std::iter::from_fn(move || {
                if let Some(mapped) = pending_right.take() {
                    return Some(mapped);
                }

                if !paired_done {
                    if let Some(right_value) = right.next() {
                        let left_value = left
                            .next()
                            .expect("left side must cover paired interleave item");
                        let left_index = index;
                        let right_index = index.checked_add(1).expect("interleave index overflow");
                        index = index.checked_add(2).expect("interleave index overflow");
                        pending_right = Some(map_fn((right_index, right_value)));
                        return Some(map_fn((left_index, left_value)));
                    }
                    paired_done = true;
                    index = tail_start;
                }

                left.next().map(|value| {
                    let mapped = map_fn((index, value));
                    index = index.checked_add(1).expect("interleave index overflow");
                    mapped
                })
            })
            .sum()
        }
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

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}
