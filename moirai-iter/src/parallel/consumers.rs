use super::{Consumer, ParallelIterator};
use std::marker::PhantomData;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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

/// Partial reduction result: the reduced value, if any, plus the function
/// needed to combine partials from split shards.
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
        let reduce_fn = self.reduce_fn;
        // Streaming fold rather than a shard-local `Vec`: the reduction is the
        // only value that has to survive the shard.
        let value = iter.seq_fold(None, |accumulator, item| match accumulator {
            None => Some(item),
            Some(accumulator) => Some(reduce_fn(accumulator, item)),
        });

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

/// Partial fold result: one shard's accumulator plus the operation needed to
/// merge it with a sibling shard's accumulator.
///
/// `Consumer::combine` is an associated function with no receiver, so the
/// merging operation has to travel inside the result. [`Reduction`] carries
/// `reduce` the same way; this is that shape generalized to an accumulator type
/// distinct from the item type.
pub struct Folded<Acc, CombineFn> {
    value: Acc,
    combine_fn: CombineFn,
}

impl<Acc, CombineFn> Folded<Acc, CombineFn> {
    pub(super) fn into_value(self) -> Acc {
        self.value
    }
}

/// Consumer that folds each shard's item stream into a partial accumulator and
/// merges partial accumulators in logical shard order.
///
/// # Ordering
///
/// `drive` splits a source at a midpoint determined solely by its length and
/// passes the logically earlier shard's result to `combine` as `left`. The
/// merge tree is therefore a function of the input length alone: it does not
/// depend on which shard finishes first, on worker count, or on scheduler
/// admission. A non-associative merge (floating-point addition) still yields
/// the same value on every run for a given input length, though not
/// necessarily the value a strictly left-to-right sequential fold would
/// produce.
///
/// # Allocation
///
/// Shards fold through [`ParallelIterator::seq_fold`], so a shard's items are
/// never gathered into an intermediate `Vec`.
pub struct FoldConsumer<Acc, InitFn, FoldFn, CombineFn> {
    init_fn: InitFn,
    fold_fn: FoldFn,
    combine_fn: CombineFn,
    _accumulator: PhantomData<fn() -> Acc>,
}

impl<Acc, InitFn, FoldFn, CombineFn> FoldConsumer<Acc, InitFn, FoldFn, CombineFn> {
    pub(super) fn new(init_fn: InitFn, fold_fn: FoldFn, combine_fn: CombineFn) -> Self {
        Self {
            init_fn,
            fold_fn,
            combine_fn,
            _accumulator: PhantomData,
        }
    }
}

impl<Item, Acc, InitFn, FoldFn, CombineFn> Consumer<Item>
    for FoldConsumer<Acc, InitFn, FoldFn, CombineFn>
where
    Item: Send,
    Acc: Send,
    InitFn: Fn() -> Acc + Send + Sync + Clone,
    FoldFn: Fn(Acc, Item) -> Acc + Send + Sync + Clone,
    CombineFn: Fn(Acc, Acc) -> Acc + Send + Sync + Clone,
{
    type Result = Folded<Acc, CombineFn>;

    fn consume<I>(self, iter: I) -> Self::Result
    where
        I: ParallelIterator<Item = Item>,
    {
        let fold_fn = self.fold_fn;
        let value = iter.seq_fold((self.init_fn)(), |accumulator, item| {
            fold_fn(accumulator, item)
        });

        Folded {
            value,
            combine_fn: self.combine_fn,
        }
    }

    fn split_at(self, _index: usize) -> (Self, Self) {
        (
            FoldConsumer::new(
                self.init_fn.clone(),
                self.fold_fn.clone(),
                self.combine_fn.clone(),
            ),
            FoldConsumer::new(self.init_fn, self.fold_fn, self.combine_fn),
        )
    }

    fn combine(left: Self::Result, right: Self::Result) -> Self::Result {
        let combine_fn = left.combine_fn;
        let value = combine_fn(left.value, right.value);

        Folded { value, combine_fn }
    }
}

/// Consumer that folds each shard with early exit, optionally sharing one stop
/// flag across shards.
///
/// # Stop scope
///
/// `stop` selects between two contracts a single fold shape serves:
///
/// - `None` — a shard stops at its own first `Break`, but every shard still
///   runs. Terminals whose result depends on logical order (`find_first`,
///   `try_for_each`) need this: skipping a shard that has not started could
///   discard an earlier answer than the one that triggered the stop.
/// - `Some(flag)` — the first shard to `Break` sets the flag and shards that
///   have not started return their initial accumulator. Terminals that accept
///   any matching answer (`find_any`, `any`, `all`) use this.
///
/// The flag is read once on entry and then once every [`STOP_POLL_STRIDE`]
/// items, so a shard already running abandons its range rather than finishing
/// it. Reading per item would put a shared cache line on the fold's inner loop;
/// reading only on entry left a running shard scanning its whole range after
/// the answer was already known, which measured as no short-circuit at all.
/// `Relaxed` ordering suffices because the flag is a hint: a shard that misses
/// the store does at most one more stride of work, whose result `combine`
/// discards.
/// Items a shard folds between reads of the shared stop flag.
///
/// A power of two so the countdown compiles to a mask-free decrement and test,
/// and small enough that abandoning a running shard costs at most this many
/// items of wasted work against a shard length bounded by the dispatch
/// threshold.
const STOP_POLL_STRIDE: usize = 128;

pub struct ShortCircuitConsumer<Acc, InitFn, FoldFn, CombineFn> {
    init_fn: InitFn,
    fold_fn: FoldFn,
    combine_fn: CombineFn,
    stop: Option<Arc<AtomicBool>>,
    _accumulator: PhantomData<fn() -> Acc>,
}

impl<Acc, InitFn, FoldFn, CombineFn> ShortCircuitConsumer<Acc, InitFn, FoldFn, CombineFn> {
    /// Fold with per-shard early exit only; every shard runs.
    pub(super) fn ordered(init_fn: InitFn, fold_fn: FoldFn, combine_fn: CombineFn) -> Self {
        Self {
            init_fn,
            fold_fn,
            combine_fn,
            stop: None,
            _accumulator: PhantomData,
        }
    }

    /// Fold with a stop flag shared across shards.
    pub(super) fn abortable(init_fn: InitFn, fold_fn: FoldFn, combine_fn: CombineFn) -> Self {
        Self {
            init_fn,
            fold_fn,
            combine_fn,
            stop: Some(Arc::new(AtomicBool::new(false))),
            _accumulator: PhantomData,
        }
    }

    fn split_shard(&self) -> Self
    where
        InitFn: Clone,
        FoldFn: Clone,
        CombineFn: Clone,
    {
        Self {
            init_fn: self.init_fn.clone(),
            fold_fn: self.fold_fn.clone(),
            combine_fn: self.combine_fn.clone(),
            stop: self.stop.clone(),
            _accumulator: PhantomData,
        }
    }
}

impl<Item, Acc, InitFn, FoldFn, CombineFn> Consumer<Item>
    for ShortCircuitConsumer<Acc, InitFn, FoldFn, CombineFn>
where
    Item: Send,
    Acc: Send,
    InitFn: Fn() -> Acc + Send + Sync + Clone,
    FoldFn: Fn(Acc, Item) -> ControlFlow<Acc, Acc> + Send + Sync + Clone,
    CombineFn: Fn(Acc, Acc) -> Acc + Send + Sync + Clone,
{
    type Result = Folded<Acc, CombineFn>;

    fn consume<I>(self, iter: I) -> Self::Result
    where
        I: ParallelIterator<Item = Item>,
    {
        let Self {
            init_fn,
            fold_fn,
            combine_fn,
            stop,
            _accumulator,
        } = self;

        if stop
            .as_ref()
            .is_some_and(|stop| stop.load(Ordering::Relaxed))
        {
            return Folded {
                value: init_fn(),
                combine_fn,
            };
        }

        let mut countdown = STOP_POLL_STRIDE;
        let folded = iter.seq_try_fold(init_fn(), |accumulator, item| {
            if let Some(stop) = &stop {
                countdown -= 1;
                if countdown == 0 {
                    countdown = STOP_POLL_STRIDE;
                    if stop.load(Ordering::Relaxed) {
                        // Abandoning returns the accumulator as it stands. Only
                        // the shared-stop terminals reach here, and their result
                        // is an `Option` that is still `None` at this point, so
                        // `combine` discards it.
                        return ControlFlow::Break(accumulator);
                    }
                }
            }

            fold_fn(accumulator, item)
        });

        let value = match folded {
            ControlFlow::Continue(value) => value,
            ControlFlow::Break(value) => {
                if let Some(stop) = &stop {
                    stop.store(true, Ordering::Relaxed);
                }
                value
            }
        };

        Folded { value, combine_fn }
    }

    fn split_at(self, _index: usize) -> (Self, Self) {
        (self.split_shard(), self)
    }

    fn combine(left: Self::Result, right: Self::Result) -> Self::Result {
        let combine_fn = left.combine_fn;
        let value = combine_fn(left.value, right.value);

        Folded { value, combine_fn }
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
        // Drive side effects (e.g. from for_each) by streaming the item stream
        // and dropping each item as it is produced. Folding rather than
        // collecting applies the same upstream map/filter transforms without
        // holding the whole shard in memory.
        iter.seq_fold((), |(), _item| ());
    }

    fn split_at(self, _index: usize) -> (Self, Self) {
        (NullConsumer::new(), NullConsumer::new())
    }

    fn combine(_left: Self::Result, _right: Self::Result) -> Self::Result {}
}
