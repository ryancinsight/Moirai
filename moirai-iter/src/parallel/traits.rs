use super::{fallible, split, TryStreamItem};
use super::{
    Chain, Chunks, Cloned, Copied, Enumerate, Filter, FilterMap, FlatMap, Flatten, FoldConsumer,
    Inspect, Intersperse, Map, MapInit, MapWith, NullConsumer, PanicFuse, Positions,
    ReduceConsumer, Reduction, Rev, SequentialAdapter, ShortCircuitConsumer, Skip, SkipAnyWhile,
    Take, TakeAnyWhile, Update, WhileSome, Zip, ZipEq,
};
use std::ops::ControlFlow;

/// Core parallel iterator trait for Moirai's Rayon-style non-indexed subset.
pub trait ParallelIterator: Sized + Send {
    /// The type of items yielded by this parallel iterator.
    type Item: Send;

    /// Drive the `Consumer` protocol over this iterator's items.
    ///
    /// # Concurrency contract
    ///
    /// Large owned and borrowed vector sources split their consumer recursively
    /// and run one branch through Moirai's nesting-safe `SyncTask` scope. Small
    /// shards remain inline so scheduler overhead does not dominate the work.
    /// Scope admission refusal runs the branch on the caller, preserving the
    /// every-item contract under shutdown or bounded-queue pressure. The
    /// The resulting consumer combination preserves logical source order. The
    /// infallible iterator contract recovers an unclaimed branch on the caller
    /// if the scheduler cannot admit the scoped job; bounded admission refusal
    /// is handled by the scheduler's caller-lane fallback before this method
    /// returns. A scheduler shutdown therefore degrades this drive to ordered
    /// caller-side execution rather than dropping work.
    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send;

    /// Collect all items sequentially without routing through the consumer protocol.
    fn seq_items(self) -> Vec<Self::Item>;

    /// Fold this iterator's logical item stream left to right, stopping at the
    /// first `ControlFlow::Break`.
    ///
    /// This is the streaming counterpart to [`seq_items`](Self::seq_items) and
    /// the base every folding [`Consumer`] runs on: a shard's items reach the
    /// accumulator one at a time instead of being gathered into an intermediate
    /// `Vec`. The default implementation goes through `seq_items` so every
    /// source and adapter keeps working unchanged; sources and adapters on the
    /// terminal hot path override it to stream.
    ///
    /// The break value is the accumulator as it stood when the fold stopped, so
    /// a caller that needs the partial result on early exit reads it from the
    /// `Break` arm.
    fn seq_try_fold<T, B, F>(self, init: T, fold_fn: F) -> std::ops::ControlFlow<B, T>
    where
        F: FnMut(T, Self::Item) -> std::ops::ControlFlow<B, T>,
    {
        self.seq_items().into_iter().try_fold(init, fold_fn)
    }

    /// Fold this iterator's logical item stream left to right.
    ///
    /// The non-short-circuiting form of [`seq_try_fold`](Self::seq_try_fold);
    /// it inherits that method's streaming behaviour, so overriding
    /// `seq_try_fold` is enough to make both allocation-free.
    fn seq_fold<T, F>(self, init: T, mut fold_fn: F) -> T
    where
        F: FnMut(T, Self::Item) -> T,
    {
        let folded = self.seq_try_fold(init, move |accumulator, item| {
            std::ops::ControlFlow::<std::convert::Infallible, T>::Continue(fold_fn(
                accumulator,
                item,
            ))
        });

        match folded {
            std::ops::ControlFlow::Continue(accumulator) => accumulator,
            std::ops::ControlFlow::Break(never) => match never {},
        }
    }

    /// Collect a logical window from the sequential item stream.
    fn seq_items_window(self, skip: usize, take: Option<usize>) -> Vec<Self::Item> {
        let iter = self.seq_items().into_iter().skip(skip);
        match take {
            Some(count) => iter.take(count).collect(),
            None => iter.collect(),
        }
    }

    /// Collect items in reverse logical order.
    fn seq_items_reversed(self) -> Vec<Self::Item> {
        let mut items = self.seq_items();
        items.reverse();
        items
    }

    /// Collect a prefix from the reversed logical item stream.
    fn seq_items_reversed_prefix(self, count: usize) -> Vec<Self::Item> {
        self.seq_items_reversed().into_iter().take(count).collect()
    }

    /// Map operation that transforms each element in parallel.
    fn map<F, R>(self, map_fn: F) -> Map<Self, F>
    where
        F: Fn(Self::Item) -> R + Send + Sync + Clone,
        R: Send,
    {
        Map::new(self, map_fn)
    }

    /// Map operation with cloned per-operation state.
    fn map_with<T, F, R>(self, init: T, map_fn: F) -> MapWith<Self, T, F>
    where
        T: Send + Clone,
        F: Fn(&mut T, Self::Item) -> R + Send + Sync + Clone,
        R: Send + Sync + 'static,
    {
        MapWith::new(self, init, map_fn)
    }

    /// Map operation with lazily initialized state.
    fn map_init<Init, T, F, R>(self, init: Init, map_fn: F) -> MapInit<Self, Init, F>
    where
        Init: Fn() -> T + Send + Sync + Clone,
        T: Send,
        F: Fn(&mut T, Self::Item) -> R + Send + Sync + Clone,
        R: Send + Sync + 'static,
    {
        MapInit::new(self, init, map_fn)
    }

    /// Mutate each item by reference and yield the mutated item.
    fn update<F>(self, update_fn: F) -> Update<Self, F>
    where
        F: Fn(&mut Self::Item) + Send + Sync + Clone,
        Self::Item: Sync + 'static,
    {
        Update::new(self, update_fn)
    }

    /// Filter operation that retains elements matching a predicate.
    fn filter<F>(self, filter_fn: F) -> Filter<Self, F>
    where
        F: Fn(&Self::Item) -> bool + Send + Sync + Clone,
    {
        Filter::new(self, filter_fn)
    }

    /// Inspect each element by shared reference without changing the stream.
    fn inspect<F>(self, inspect_fn: F) -> Inspect<Self, F>
    where
        F: Fn(&Self::Item) + Send + Sync + Clone,
        Self::Item: Sync,
    {
        Inspect::new(self, inspect_fn)
    }

    /// Preserve value semantics while stopping sibling work after panic where applicable.
    fn panic_fuse(self) -> PanicFuse<Self>
    where
        Self::Item: Sync,
    {
        PanicFuse::new(self)
    }

    /// Map each element to an optional value and retain present values.
    fn filter_map<F, R>(self, filter_map_fn: F) -> FilterMap<Self, F>
    where
        F: Fn(Self::Item) -> Option<R> + Send + Sync + Clone,
        R: Send + Sync + 'static,
    {
        FilterMap::new(self, filter_map_fn)
    }

    /// Unwrap a prefix of present values from an optional stream.
    fn while_some<T>(self) -> WhileSome<Self>
    where
        Self: ParallelIterator<Item = Option<T>>,
        T: Send + Sync + 'static,
    {
        WhileSome::new(self)
    }

    /// Map each element to an iterator and flatten the resulting sequence.
    fn flat_map<F, U>(self, flat_map_fn: F) -> FlatMap<Self, F>
    where
        F: Fn(Self::Item) -> U + Send + Sync + Clone,
        U: IntoIterator,
        U::Item: Send + Sync + 'static,
    {
        FlatMap::new(self, flat_map_fn)
    }

    /// Map each element to a serial iterator and flatten the resulting sequence.
    fn flat_map_iter<F, U>(self, flat_map_fn: F) -> FlatMap<Self, F>
    where
        F: Fn(Self::Item) -> U + Send + Sync + Clone,
        U: IntoIterator,
        U::Item: Send + Sync + 'static,
    {
        FlatMap::new(self, flat_map_fn)
    }

    /// Flatten nested item streams with standard left-to-right semantics.
    fn flatten(self) -> Flatten<Self>
    where
        Self::Item: IntoIterator,
        <Self::Item as IntoIterator>::Item: Send + Sync + 'static,
    {
        Flatten::new(self)
    }

    /// Flatten nested serial iterators with standard left-to-right semantics.
    fn flatten_iter(self) -> Flatten<Self>
    where
        Self::Item: IntoIterator,
        <Self::Item as IntoIterator>::Item: Send + Sync + 'static,
    {
        Flatten::new(self)
    }

    /// Pair each element with its zero-based position in the logical sequence.
    fn enumerate(self) -> Enumerate<Self>
    where
        Self::Item: Sync + 'static,
    {
        Enumerate::new(self)
    }

    /// Pair elements with another parallel iterator, stopping at the shorter input.
    fn zip<J>(self, other: J) -> Zip<Self, J>
    where
        J: ParallelIterator,
        Self::Item: Sync + 'static,
        J::Item: Sync + 'static,
    {
        Zip::new(self, other)
    }

    /// Pair elements with another parallel iterator and require equal lengths.
    fn zip_eq<J>(self, other: J) -> ZipEq<Self, J>
    where
        J: ParallelIterator,
        Self::Item: Sync + 'static,
        J::Item: Sync + 'static,
    {
        ZipEq::new(self, other)
    }

    /// Retain at most `count` elements from the logical sequence prefix.
    fn take(self, count: usize) -> Take<Self>
    where
        Self::Item: Sync + 'static,
    {
        Take::new(self, count)
    }

    /// Retain at most `count` items from this non-indexed deterministic stream.
    fn take_any(self, count: usize) -> Take<Self>
    where
        Self::Item: Sync + 'static,
    {
        Take::new(self, count)
    }

    /// Discard `count` elements from the logical sequence prefix.
    fn skip(self, count: usize) -> Skip<Self>
    where
        Self::Item: Sync + 'static,
    {
        Skip::new(self, count)
    }

    /// Discard `count` items from this non-indexed deterministic stream.
    fn skip_any(self, count: usize) -> Skip<Self>
    where
        Self::Item: Sync + 'static,
    {
        Skip::new(self, count)
    }

    /// Retain this deterministic stream prefix while `predicate` returns `true`.
    fn take_any_while<F>(self, predicate: F) -> TakeAnyWhile<Self, F>
    where
        F: Fn(&Self::Item) -> bool + Send + Sync + Clone,
        Self::Item: Sync + 'static,
    {
        TakeAnyWhile::new(self, predicate)
    }

    /// Discard this deterministic stream prefix while `predicate` returns `true`.
    fn skip_any_while<F>(self, predicate: F) -> SkipAnyWhile<Self, F>
    where
        F: Fn(&Self::Item) -> bool + Send + Sync + Clone,
        Self::Item: Sync + 'static,
    {
        SkipAnyWhile::new(self, predicate)
    }

    /// Concatenate this iterator with another iterator of the same item type.
    fn chain<J>(self, other: J) -> Chain<Self, J>
    where
        J: ParallelIterator<Item = Self::Item>,
        Self::Item: Sync + 'static,
    {
        Chain::new(self, other)
    }

    /// Insert a cloned separator between adjacent logical items.
    fn intersperse(self, separator: Self::Item) -> Intersperse<Self>
    where
        Self::Item: Clone + Sync + 'static,
    {
        Intersperse::new(self, separator)
    }

    /// Reverse the logical sequence order.
    fn rev(self) -> Rev<Self>
    where
        Self::Item: Sync + 'static,
    {
        Rev::new(self)
    }

    /// Group the logical item stream into non-empty chunks.
    fn chunks(self, chunk_size: usize) -> Chunks<Self>
    where
        Self::Item: Sync + 'static,
    {
        Chunks::new(self, chunk_size)
    }

    /// Copy referenced items out of a borrowed parallel stream.
    fn copied<'data, T>(self) -> Copied<Self>
    where
        Self: ParallelIterator<Item = &'data T>,
        T: Copy + Send + Sync + 'data + 'static,
    {
        Copied::new(self)
    }

    /// Clone referenced items out of a borrowed parallel stream.
    fn cloned<'data, T>(self) -> Cloned<Self>
    where
        Self: ParallelIterator<Item = &'data T>,
        T: Clone + Send + Sync + 'data + 'static,
    {
        Cloned::new(self)
    }

    /// Reduce operation that combines all elements.
    fn reduce<F>(self, reduce_fn: F) -> Option<Self::Item>
    where
        F: Fn(Self::Item, Self::Item) -> Self::Item + Send + Sync + Clone,
        Self::Item: Clone + Sync,
    {
        let reduction: Reduction<Self::Item, F> = self.drive(ReduceConsumer::new(reduce_fn));
        reduction.into_value()
    }

    /// Fold operation with an initial value.
    fn fold<T, F>(self, init: T, fold_fn: F) -> T
    where
        T: Send + Sync + Clone,
        F: Fn(T, Self::Item) -> T + Send + Sync + Clone,
        Self::Item: Sync,
    {
        // A fold function maps `(accumulator, item) -> accumulator` and cannot
        // combine two partial accumulators without a separate associative
        // operation. Preserve sequential value semantics for this API.
        //
        // Sequential is the contract, but the intermediate `Vec` was not: the
        // stream folds item by item.
        self.seq_fold(init, fold_fn)
    }

    /// Collect into a collection.
    fn collect<C>(self) -> C
    where
        C: ParallelExtend<Self::Item> + Default + Send,
    {
        let mut collection = C::default();
        collection.par_extend(self);
        collection
    }

    /// Collect into a list of owned vector segments.
    ///
    /// This bounded terminal mirrors Rayon's public `collect_vec_list` return
    /// shape while preserving Moirai's logical item stream as one moved
    /// segment. Segment count is not part of the semantic contract; flattening
    /// the returned list yields the same logical item sequence as `collect`.
    fn collect_vec_list(self) -> std::collections::LinkedList<Vec<Self::Item>> {
        let items = self.seq_items();
        let mut list = std::collections::LinkedList::new();
        if !items.is_empty() {
            list.push_back(items);
        }
        list
    }

    /// Partition items into two collections while preserving relative order.
    fn partition<C, F>(self, predicate: F) -> (C, C)
    where
        C: FromIterator<Self::Item> + Send,
        F: Fn(&Self::Item) -> bool + Send + Sync + Clone,
        Self::Item: Sync + 'static,
    {
        // Measured sequential. Folding this in parallel was tried and was
        // slower at every input size, including sizes below the dispatch
        // threshold where no shard is created at all: the accumulator is a pair
        // of vectors moved through the fold closure per item, and the shard
        // outputs then have to be appended back together in order. Collecting
        // once and letting the standard partition size both outputs from the
        // known length beat both effects. Parallelising this terminal needs a
        // size-hinted output collection, not a fold.
        let (left_items, right_items): (Vec<Self::Item>, Vec<Self::Item>) = self
            .seq_items()
            .into_iter()
            .partition(|item| predicate(item));

        (
            left_items.into_iter().collect(),
            right_items.into_iter().collect(),
        )
    }

    /// Split mapped `Either` values into two collections while preserving side-local order.
    fn partition_map<A, B, P, L, R>(self, predicate: P) -> (A, B)
    where
        A: Default + Extend<L> + Send,
        B: Default + Extend<R> + Send,
        P: Fn(Self::Item) -> split::Either<L, R> + Send + Sync + Clone,
        L: Send,
        R: Send,
    {
        split::partition_map(self, predicate)
    }

    /// Split a stream of pairs into two collections while preserving order.
    fn unzip<A, B, FromA, FromB>(self) -> (FromA, FromB)
    where
        Self: ParallelIterator<Item = (A, B)>,
        FromA: Default + Extend<A> + Send,
        FromB: Default + Extend<B> + Send,
        A: Send,
        B: Send,
    {
        // Measured sequential for the reason given on
        // [`partition`](Self::partition).
        self.seq_items().into_iter().unzip()
    }

    /// Convert to a sequential iterator.
    fn sequential(self) -> SequentialAdapter<Self> {
        SequentialAdapter::new(self)
    }

    /// Count the number of elements.
    fn count(self) -> usize
    where
        Self::Item: Sync,
    {
        self.drive(FoldConsumer::new(
            || 0_usize,
            |count: usize, _item| count + 1,
            |left: usize, right: usize| left + right,
        ))
        .into_value()
    }

    /// Find the first element matching a predicate.
    ///
    /// Every shard runs: a shard that has not started may hold an earlier match
    /// than one already found, so this terminal cannot abandon shards the way
    /// [`find_any`](Self::find_any) does. Each shard still stops at its own
    /// first match.
    fn find_first<F>(self, predicate: F) -> Option<Self::Item>
    where
        F: Fn(&Self::Item) -> bool + Send + Sync + Clone,
        Self::Item: Sync,
    {
        self.drive(ShortCircuitConsumer::ordered(
            || None,
            move |_accumulator: Option<Self::Item>, item| {
                if predicate(&item) {
                    ControlFlow::Break(Some(item))
                } else {
                    ControlFlow::Continue(None)
                }
            },
            |left: Option<Self::Item>, right: Option<Self::Item>| left.or(right),
        ))
        .into_value()
    }

    /// Find the last element matching a predicate in the logical stream.
    fn find_last<F>(self, predicate: F) -> Option<Self::Item>
    where
        F: Fn(&Self::Item) -> bool + Send + Sync + Clone,
    {
        self.drive(FoldConsumer::new(
            || None,
            move |accumulator: Option<Self::Item>, item| {
                if predicate(&item) {
                    Some(item)
                } else {
                    accumulator
                }
            },
            |left: Option<Self::Item>, right: Option<Self::Item>| right.or(left),
        ))
        .into_value()
    }

    /// Find the first logical index matching a predicate.
    ///
    /// Sequential by contract: a logical index is a property of the whole
    /// stream, and the non-indexed consumer protocol cannot hand a shard its
    /// own base index. `Consumer::split_at` carries the source split point,
    /// which a length-changing adapter such as `filter` invalidates before it
    /// reaches the shard. The stream is folded rather than collected, so no
    /// intermediate vector is built.
    fn position_first<F>(self, predicate: F) -> Option<usize>
    where
        F: Fn(Self::Item) -> bool + Send + Sync + Clone,
    {
        let found = self.seq_try_fold(0_usize, |index, item| {
            if predicate(item) {
                ControlFlow::Break(index)
            } else {
                ControlFlow::Continue(index + 1)
            }
        });

        match found {
            ControlFlow::Break(index) => Some(index),
            ControlFlow::Continue(_) => None,
        }
    }

    /// Find any logical index matching a predicate.
    fn position_any<F>(self, predicate: F) -> Option<usize>
    where
        F: Fn(Self::Item) -> bool + Send + Sync + Clone,
    {
        self.position_first(predicate)
    }

    /// Find the last logical index matching a predicate.
    ///
    /// Sequential for the reason given on
    /// [`position_first`](Self::position_first), and folded rather than
    /// collected.
    fn position_last<F>(self, predicate: F) -> Option<usize>
    where
        F: Fn(Self::Item) -> bool + Send + Sync + Clone,
    {
        let (_, found) = self.seq_fold(
            (0_usize, None),
            |(index, found): (usize, Option<usize>), item| {
                if predicate(item) {
                    (index + 1, Some(index))
                } else {
                    (index + 1, found)
                }
            },
        );

        found
    }

    /// Return all logical indices whose items match a predicate.
    fn positions<F>(self, predicate: F) -> Positions<Self, F>
    where
        F: Fn(Self::Item) -> bool + Send + Sync + Clone,
    {
        Positions::new(self, predicate)
    }

    /// Find and map the first matching element in the logical stream.
    fn find_map_first<F, R>(self, map_fn: F) -> Option<R>
    where
        F: Fn(Self::Item) -> Option<R> + Send + Sync + Clone,
        R: Send,
    {
        self.drive(ShortCircuitConsumer::ordered(
            || None,
            move |_accumulator: Option<R>, item| match map_fn(item) {
                Some(mapped) => ControlFlow::Break(Some(mapped)),
                None => ControlFlow::Continue(None),
            },
            |left: Option<R>, right: Option<R>| left.or(right),
        ))
        .into_value()
    }

    /// Find and map any matching element in the logical stream.
    ///
    /// Shards that have not started are abandoned once any shard produces a
    /// mapped value, so the result is a mapped match rather than necessarily
    /// the logically first one. Use
    /// [`find_map_first`](Self::find_map_first) when order matters.
    fn find_map_any<F, R>(self, map_fn: F) -> Option<R>
    where
        F: Fn(Self::Item) -> Option<R> + Send + Sync + Clone,
        R: Send,
    {
        self.drive(ShortCircuitConsumer::abortable(
            || None,
            move |_accumulator: Option<R>, item| match map_fn(item) {
                Some(mapped) => ControlFlow::Break(Some(mapped)),
                None => ControlFlow::Continue(None),
            },
            |left: Option<R>, right: Option<R>| left.or(right),
        ))
        .into_value()
    }

    /// Find and map the last matching element in the logical stream.
    fn find_map_last<F, R>(self, map_fn: F) -> Option<R>
    where
        F: Fn(Self::Item) -> Option<R> + Send + Sync + Clone,
        R: Send,
    {
        self.drive(FoldConsumer::new(
            || None,
            move |accumulator: Option<R>, item| map_fn(item).or(accumulator),
            |left: Option<R>, right: Option<R>| right.or(left),
        ))
        .into_value()
    }

    /// Test if any element matches a predicate.
    fn any<F>(self, predicate: F) -> bool
    where
        F: Fn(&Self::Item) -> bool + Send + Sync + Clone,
        Self::Item: Sync,
    {
        self.find_any(predicate).is_some()
    }

    /// Test if all elements match a predicate.
    fn all<F>(self, predicate: F) -> bool
    where
        F: Fn(&Self::Item) -> bool + Send + Sync + Clone,
        Self::Item: Sync,
    {
        self.find_any(move |item| !predicate(item)).is_none()
    }

    /// Apply a function to each element.
    fn for_each<F>(self, op: F)
    where
        F: Fn(Self::Item) + Send + Sync + Clone,
    {
        self.map(op).drive(NullConsumer::new())
    }

    /// Apply a function to each element with cloned per-operation state.
    ///
    /// Sequential by contract: one state value threads through the whole
    /// stream, so `op` observes every prior item's effect. A parallel form
    /// would have to give each shard its own clone, which is a different
    /// contract. The stream is folded rather than collected.
    fn for_each_with<T, F>(self, init: T, op: F)
    where
        T: Send + Clone,
        F: Fn(&mut T, Self::Item) + Send + Sync + Clone,
    {
        self.seq_fold(init, |mut state, item| {
            op(&mut state, item);
            state
        });
    }

    /// Apply a function to each element with lazily initialized state.
    ///
    /// Sequential for the reason given on
    /// [`for_each_with`](Self::for_each_with).
    fn for_each_init<Init, T, F>(self, init: Init, op: F)
    where
        Init: Fn() -> T + Send + Sync + Clone,
        T: Send,
        F: Fn(&mut T, Self::Item) + Send + Sync + Clone,
    {
        self.seq_fold(init(), |mut state, item| {
            op(&mut state, item);
            state
        });
    }

    /// Apply a fallible function to each element and stop on the first error.
    ///
    /// The returned error is the first one in logical order. Each shard stops
    /// at its own first error, but no shard is abandoned: an earlier shard may
    /// still hold an earlier error than one already reported.
    fn try_for_each<F, E>(self, op: F) -> Result<(), E>
    where
        F: Fn(Self::Item) -> Result<(), E> + Send + Sync + Clone,
        E: Send,
    {
        self.drive(ShortCircuitConsumer::ordered(
            || Ok(()),
            move |_accumulator: Result<(), E>, item| match op(item) {
                Ok(()) => ControlFlow::Continue(Ok(())),
                Err(error) => ControlFlow::Break(Err(error)),
            },
            |left: Result<(), E>, right: Result<(), E>| {
                if left.is_err() {
                    left
                } else {
                    right
                }
            },
        ))
        .into_value()
    }

    /// Apply a fallible function to each element with cloned per-operation state.
    ///
    /// Sequential for the reason given on
    /// [`for_each_with`](Self::for_each_with).
    fn try_for_each_with<T, F, E>(self, init: T, op: F) -> Result<(), E>
    where
        T: Send + Clone,
        F: Fn(&mut T, Self::Item) -> Result<(), E> + Send + Sync + Clone,
        E: Send,
    {
        let folded = self.seq_try_fold((init, Ok(())), |(mut state, _), item| {
            match op(&mut state, item) {
                Ok(()) => ControlFlow::Continue((state, Ok(()))),
                Err(error) => ControlFlow::Break((state, Err(error))),
            }
        });
        let (_, outcome) = match folded {
            ControlFlow::Continue(state) | ControlFlow::Break(state) => state,
        };

        outcome
    }

    /// Apply a fallible function to each element with lazily initialized state.
    ///
    /// Sequential for the reason given on
    /// [`for_each_with`](Self::for_each_with).
    fn try_for_each_init<Init, T, F, E>(self, init: Init, op: F) -> Result<(), E>
    where
        Init: Fn() -> T + Send + Sync + Clone,
        T: Send,
        F: Fn(&mut T, Self::Item) -> Result<(), E> + Send + Sync + Clone,
        E: Send,
    {
        let folded = self.seq_try_fold((init(), Ok(())), |(mut state, _), item| {
            match op(&mut state, item) {
                Ok(()) => ControlFlow::Continue((state, Ok(()))),
                Err(error) => ControlFlow::Break((state, Err(error))),
            }
        });
        let (_, outcome) = match folded {
            ControlFlow::Continue(state) | ControlFlow::Break(state) => state,
        };

        outcome
    }

    /// Reduce with an associative operation.
    fn reduce_with<F>(self, reduce_fn: F) -> Option<Self::Item>
    where
        F: Fn(Self::Item, Self::Item) -> Self::Item + Send + Sync + Clone,
        Self::Item: Sync + Clone,
    {
        let reduction: Reduction<Self::Item, F> = self.drive(ReduceConsumer::new(reduce_fn));
        reduction.into_value()
    }

    /// Reduce a fallible item stream with an identity and associative operation.
    fn try_reduce<Identity, F, T, E>(self, identity: Identity, reduce_fn: F) -> Result<T, E>
    where
        Self::Item: Into<Result<T, E>>,
        Identity: Fn() -> T + Send + Sync + Clone,
        F: Fn(T, T) -> Result<T, E> + Send + Sync + Clone,
        T: Send,
        E: Send,
    {
        // Sequential by contract: `reduce_fn` threads one accumulator and may
        // fail, so partial accumulators have no order-independent merge. The
        // stream is folded rather than collected.
        let folded = self.seq_try_fold(Ok(identity()), |accumulator, item| {
            let accumulator = match accumulator {
                Ok(accumulator) => accumulator,
                Err(error) => return ControlFlow::Break(Err(error)),
            };
            match item.into().and_then(|value| reduce_fn(accumulator, value)) {
                Ok(accumulator) => ControlFlow::Continue(Ok(accumulator)),
                Err(error) => ControlFlow::Break(Err(error)),
            }
        });

        match folded {
            ControlFlow::Continue(accumulator) | ControlFlow::Break(accumulator) => accumulator,
        }
    }

    /// Reduce a fallible item stream without an identity value.
    fn try_reduce_with<F>(self, reduce_fn: F) -> Option<Self::Item>
    where
        Self::Item: TryStreamItem,
        F: Fn(
                <Self::Item as TryStreamItem>::Output,
                <Self::Item as TryStreamItem>::Output,
            ) -> Self::Item
            + Send
            + Sync
            + Clone,
    {
        fallible::try_reduce_with(self, reduce_fn)
    }

    /// Sum all items using the standard `Sum` contract for the item stream.
    ///
    /// # Ordering
    ///
    /// Shards are summed independently and merged in logical shard order. The
    /// merge tree is a function of the input length alone, so the result is
    /// reproducible across runs and worker counts; it is not necessarily
    /// bit-identical to a strictly left-to-right `Iterator::sum` when addition
    /// on `S` is non-associative, as it is for floating point. The
    /// `Sum<S>` bound is what makes merging two partial sums expressible.
    fn sum<S>(self) -> S
    where
        S: std::iter::Sum<Self::Item> + std::iter::Sum<S> + Send,
    {
        self.drive(FoldConsumer::new(
            || std::iter::empty::<Self::Item>().sum::<S>(),
            |accumulator: S, item: Self::Item| {
                [accumulator, std::iter::once(item).sum::<S>()]
                    .into_iter()
                    .sum::<S>()
            },
            |left: S, right: S| [left, right].into_iter().sum::<S>(),
        ))
        .into_value()
    }

    /// Multiply all items using the standard `Product` contract for the item stream.
    ///
    /// Shards are multiplied independently and merged in logical shard order;
    /// see [`sum`](Self::sum) for what that means when multiplication on `P` is
    /// non-associative.
    fn product<P>(self) -> P
    where
        P: std::iter::Product<Self::Item> + std::iter::Product<P> + Send,
    {
        self.drive(FoldConsumer::new(
            || std::iter::empty::<Self::Item>().product::<P>(),
            |accumulator: P, item: Self::Item| {
                [accumulator, std::iter::once(item).product::<P>()]
                    .into_iter()
                    .product::<P>()
            },
            |left: P, right: P| [left, right].into_iter().product::<P>(),
        ))
        .into_value()
    }

    /// Return the minimum item in the logical stream.
    fn min(self) -> Option<Self::Item>
    where
        Self::Item: Ord,
    {
        self.min_by(Self::Item::cmp)
    }

    /// Return the maximum item in the logical stream.
    fn max(self) -> Option<Self::Item>
    where
        Self::Item: Ord,
    {
        self.max_by(Self::Item::cmp)
    }

    /// Return the minimum item according to a comparator.
    ///
    /// Ties resolve to the earliest item in logical order, matching
    /// `Iterator::min_by`. Shards keep their own earliest minimum and merges
    /// keep the earlier shard's on equality, so the tie-break is the same at
    /// every level of the merge tree.
    fn min_by<F>(self, compare: F) -> Option<Self::Item>
    where
        F: Fn(&Self::Item, &Self::Item) -> std::cmp::Ordering + Send + Sync + Clone,
    {
        let fold_compare = compare.clone();
        self.drive(FoldConsumer::new(
            || None,
            move |accumulator: Option<Self::Item>, item| match accumulator {
                None => Some(item),
                Some(best) => {
                    if fold_compare(&item, &best) == std::cmp::Ordering::Less {
                        Some(item)
                    } else {
                        Some(best)
                    }
                }
            },
            move |left: Option<Self::Item>, right: Option<Self::Item>| match (left, right) {
                (None, other) | (other, None) => other,
                (Some(left), Some(right)) => {
                    if compare(&right, &left) == std::cmp::Ordering::Less {
                        Some(right)
                    } else {
                        Some(left)
                    }
                }
            },
        ))
        .into_value()
    }

    /// Return the maximum item according to a comparator.
    ///
    /// Ties resolve to the latest item in logical order, matching
    /// `Iterator::max_by`.
    fn max_by<F>(self, compare: F) -> Option<Self::Item>
    where
        F: Fn(&Self::Item, &Self::Item) -> std::cmp::Ordering + Send + Sync + Clone,
    {
        let fold_compare = compare.clone();
        self.drive(FoldConsumer::new(
            || None,
            move |accumulator: Option<Self::Item>, item| match accumulator {
                None => Some(item),
                Some(best) => {
                    if fold_compare(&item, &best) == std::cmp::Ordering::Less {
                        Some(best)
                    } else {
                        Some(item)
                    }
                }
            },
            move |left: Option<Self::Item>, right: Option<Self::Item>| match (left, right) {
                (None, other) | (other, None) => other,
                (Some(left), Some(right)) => {
                    if compare(&right, &left) == std::cmp::Ordering::Less {
                        Some(left)
                    } else {
                        Some(right)
                    }
                }
            },
        ))
        .into_value()
    }

    /// Return the minimum item according to an ordered key.
    ///
    /// Expressed through [`min_by`](Self::min_by), so tie-breaking matches
    /// `Iterator::min_by_key`. `key_fn` runs twice per comparison rather than
    /// being cached alongside the item, which keeps the key out of the value
    /// that crosses shard boundaries and so avoids a `K: Send` requirement.
    fn min_by_key<K, F>(self, key_fn: F) -> Option<Self::Item>
    where
        K: Ord,
        F: Fn(&Self::Item) -> K + Send + Sync + Clone,
    {
        self.min_by(move |left, right| key_fn(left).cmp(&key_fn(right)))
    }

    /// Return the maximum item according to an ordered key.
    ///
    /// Expressed through [`max_by`](Self::max_by); see
    /// [`min_by_key`](Self::min_by_key) for the key-evaluation note.
    fn max_by_key<K, F>(self, key_fn: F) -> Option<Self::Item>
    where
        K: Ord,
        F: Fn(&Self::Item) -> K + Send + Sync + Clone,
    {
        self.max_by(move |left, right| key_fn(left).cmp(&key_fn(right)))
    }

    /// Find any element matching a predicate.
    ///
    /// Shards that have not started are abandoned once any shard finds a match,
    /// so the returned item is a match rather than necessarily the logically
    /// first one. Use [`find_first`](Self::find_first) when order matters.
    fn find_any<F>(self, predicate: F) -> Option<Self::Item>
    where
        F: Fn(&Self::Item) -> bool + Send + Sync + Clone,
        Self::Item: Sync,
    {
        self.drive(ShortCircuitConsumer::abortable(
            || None,
            move |_accumulator: Option<Self::Item>, item| {
                if predicate(&item) {
                    ControlFlow::Break(Some(item))
                } else {
                    ControlFlow::Continue(None)
                }
            },
            |left: Option<Self::Item>, right: Option<Self::Item>| left.or(right),
        ))
        .into_value()
    }
}

/// Consumer trait for parallel iterator operations.
pub trait Consumer<T>: Send + Sync {
    /// Result type produced by consuming an iterator.
    type Result: Send;

    /// Consume items from a parallel iterator.
    fn consume<I>(self, iter: I) -> Self::Result
    where
        I: ParallelIterator<Item = T>;

    /// Split the consumer for parallel processing.
    fn split_at(self, index: usize) -> (Self, Self)
    where
        Self: Sized;

    /// Combine results from split consumers.
    fn combine(left: Self::Result, right: Self::Result) -> Self::Result;
}

/// Trait for collections that can be extended in parallel.
pub trait ParallelExtend<T>: Send {
    /// Extend the collection with items from a parallel iterator.
    fn par_extend<I>(&mut self, par_iter: I)
    where
        I: ParallelIterator<Item = T>;
}

/// Extension trait for collections to create parallel iterators.
pub trait IntoParallelIterator {
    /// Element type yielded by the iterator.
    type Item: Send;
    /// Parallel iterator produced by conversion.
    type Iter: ParallelIterator<Item = Self::Item>;

    /// Convert `self` into a parallel iterator.
    fn into_par_iter(self) -> Self::Iter;
}

/// Extension trait for collection references to create parallel iterators.
pub trait IntoParallelRefIterator<'data> {
    /// Element type yielded by the iterator.
    type Item: Send + Sync + 'data;
    /// Parallel iterator produced by conversion.
    type Iter: ParallelIterator<Item = Self::Item>;

    /// Create a parallel iterator over references to `self`.
    fn par_iter(&'data self) -> Self::Iter;
}
