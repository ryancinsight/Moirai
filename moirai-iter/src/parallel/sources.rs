use super::{
    CollectConsumer, Consumer, IndexedParallelIterator, IntoParallelIterator,
    IntoParallelRefIterator, ParallelExtend, ParallelIterator,
};
use moirai_executor::{global, SyncTask};
use std::ops::ControlFlow;
use std::sync::Mutex;

/// Minimum source size for scheduler-backed non-indexed driving.
///
/// Smaller sources stay on the existing recursive consumer path so dispatch
/// overhead does not dominate the work. Larger vector-backed sources split at
/// each drive level and run one branch through the nesting-safe scheduler
/// scope; child drives stop at the same threshold.
pub(super) const PARALLEL_DRIVE_THRESHOLD: usize = 1024;

fn drive_split<I, C, R>(left: I, right: I, left_consumer: C, right_consumer: C) -> R
where
    I: ParallelIterator,
    C: Consumer<I::Item, Result = R> + Send + Sync,
    R: Send,
{
    let left_result = Mutex::new(None);
    let left_branch = Mutex::new(Some((left, left_consumer)));
    let right_branch = Mutex::new(Some((right, right_consumer)));
    let mut right_result = None;

    let scope_result = global().scope::<SyncTask, _>(|scope| {
        scope.spawn(|_| {
            let (left, left_consumer) = left_branch
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
                .expect("parallel iterator left branch must be claimed once");
            let result = left_consumer.consume(left);
            *left_result
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(result);
        })?;
        // Flush before consuming the caller branch so the two branches overlap
        // whenever scheduler admission succeeds. A refused job is run inline
        // by the scope, preserving the every-job-runs contract under pressure.
        scope.flush()?;
        let (right, right_consumer) = right_branch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .expect("parallel iterator right branch must be claimed once");
        right_result = Some(right_consumer.consume(right));
        Ok(())
    });

    // `drive` is an infallible terminal API. If shutdown rejects the scoped
    // branch, recover the still-unclaimed branch and finish both halves on the
    // caller rather than dropping work or panicking after a partial drive.
    if let Err(error) = scope_result {
        match error {
            moirai_core::ExecutorError::ShuttingDown
            | moirai_core::ExecutorError::ResourceExhausted(_) => {
                let fallback = left_branch
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take();
                if let Some((left, left_consumer)) = fallback {
                    let result = left_consumer.consume(left);
                    *left_result
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(result);
                }
                if right_result.is_none() {
                    let fallback = right_branch
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .take();
                    if let Some((right, right_consumer)) = fallback {
                        right_result = Some(right_consumer.consume(right));
                    }
                }
            }
            error => panic!("moirai global executor: parallel iterator drive: {error}"),
        }
    }

    let left_result = left_result
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .expect("parallel iterator left branch must complete");
    let right_result = right_result.expect("parallel iterator right branch must complete");
    C::combine(left_result, right_result)
}

fn move_vec_items_into<T>(source: Vec<T>, target: &mut Vec<T>) {
    target.clear();
    let len = source.len();
    if target.capacity() < len {
        *target = source;
        return;
    }

    // Consuming `source` moves every element without a `Clone` bound and
    // releases its backing allocation while retaining `target`'s capacity.
    // The prior `ManuallyDrop` copy leaked the source buffer.
    target.extend(source);
}

/// Parallel iterator over a vector.
pub struct VecParIter<T> {
    data: Vec<T>,
}

impl<T> VecParIter<T> {
    /// Create a parallel iterator over the given vector.
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    pub(in crate::parallel) fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T: Send + Sync + 'static> ParallelIterator for VecParIter<T> {
    type Item = T;

    fn seq_items(self) -> Vec<Self::Item> {
        self.data
    }

    fn seq_try_fold<Acc, B, FoldFn>(self, init: Acc, fold_fn: FoldFn) -> ControlFlow<B, Acc>
    where
        FoldFn: FnMut(Acc, Self::Item) -> ControlFlow<B, Acc>,
    {
        self.data.into_iter().try_fold(init, fold_fn)
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        // At or below the dispatch threshold a shard is consumed in one
        // sequential pass. The superseded shape kept splitting to
        // single-element shards, which bought no parallelism below the
        // threshold — the scheduler is only engaged above it — and cost one
        // consumer split and one combine per element.
        if self.data.len() <= PARALLEL_DRIVE_THRESHOLD {
            return consumer.consume(self);
        }

        // Owned elements have no safe zero-copy split: handing a shard its own
        // range of a `Vec<T>` without moving the elements needs either raw
        // pointer reads or an `Option` slot per element, and `Option<T>` is only
        // niche-packed when `T` has a spare value — for a plain scalar it
        // doubles the buffer and adds a write per element read back out, which
        // measured worse than the copy it replaced. Splitting therefore still
        // copies, but only down to the threshold, so copy traffic is
        // proportional to `log(len / threshold)` levels rather than `log(len)`.
        let mut data = self.data;
        let mid = data.len() / 2;
        let right_data = data.split_off(mid);
        let left_data = std::mem::take(&mut data);

        let (left_consumer, right_consumer) = consumer.split_at(left_data.len());

        drive_split(
            VecParIter::new(left_data),
            VecParIter::new(right_data),
            left_consumer,
            right_consumer,
        )
    }
}

impl<T: Send + Sync + 'static> IndexedParallelIterator for VecParIter<T> {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn collect_into_vec(self, target: &mut Vec<Self::Item>) {
        move_vec_items_into(self.data, target);
    }
}

/// Range parallel iterator.
pub struct RangeParIter<T> {
    start: T,
    end: T,
}

impl<T> RangeParIter<T>
where
    T: Send + Sync + Clone + 'static + PartialOrd + std::ops::Add<Output = T> + From<u8>,
{
    /// Create a parallel iterator over the half-open range `start..end`.
    pub fn new(start: T, end: T) -> Self {
        Self { start, end }
    }
}

impl<T> ParallelIterator for RangeParIter<T>
where
    T: Send + Sync + Clone + 'static + PartialOrd + std::ops::Add<Output = T> + From<u8>,
{
    type Item = T;

    fn seq_items(self) -> Vec<Self::Item> {
        let mut items = Vec::new();
        let mut current = self.start;
        while current < self.end {
            items.push(current.clone());
            current = current + T::from(1u8);
        }
        items
    }

    fn seq_try_fold<Acc, B, FoldFn>(self, init: Acc, mut fold_fn: FoldFn) -> ControlFlow<B, Acc>
    where
        FoldFn: FnMut(Acc, Self::Item) -> ControlFlow<B, Acc>,
    {
        let mut accumulator = init;
        let mut current = self.start;
        while current < self.end {
            accumulator = fold_fn(accumulator, current.clone())?;
            current = current + T::from(1u8);
        }
        ControlFlow::Continue(accumulator)
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        let mut items = Vec::new();
        let mut current = self.start;
        while current < self.end {
            items.push(current.clone());
            current = current + T::from(1u8);
        }

        VecParIter::new(items).drive(consumer)
    }
}

impl IndexedParallelIterator for RangeParIter<usize> {
    fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    fn collect_into_vec(self, target: &mut Vec<Self::Item>) {
        target.clear();
        target.extend(self.start..self.end);
    }
}

/// Sequential iterator adapter for compatibility.
pub struct SequentialAdapter<I> {
    iter: I,
}

impl<I> SequentialAdapter<I> {
    pub(super) fn new(iter: I) -> Self {
        Self { iter }
    }
}

/// Adapter that drives a sequential iterator through the parallel-consumer
/// machinery as a single shard.
pub struct SequentialIterAdapter<I> {
    iter: I,
}

impl<I> SequentialIterAdapter<I> {
    pub(super) fn new(iter: I) -> Self {
        Self { iter }
    }
}

impl<I> ParallelIterator for SequentialIterAdapter<I>
where
    I: Iterator + Send,
    I::Item: Send + Sync + 'static,
{
    type Item = I::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        self.iter.collect()
    }

    fn seq_try_fold<Acc, B, FoldFn>(self, mut init: Acc, mut fold_fn: FoldFn) -> ControlFlow<B, Acc>
    where
        FoldFn: FnMut(Acc, Self::Item) -> ControlFlow<B, Acc>,
    {
        let mut iter = self.iter;
        for item in iter.by_ref() {
            init = fold_fn(init, item)?;
        }
        ControlFlow::Continue(init)
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        let items: Vec<Self::Item> = self.iter.collect();
        consumer.consume(VecParIter::new(items))
    }
}

impl<I> IndexedParallelIterator for SequentialIterAdapter<I>
where
    I: ExactSizeIterator + Send,
    I::Item: Send + Sync + 'static,
{
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn collect_into_vec(self, target: &mut Vec<Self::Item>) {
        target.clear();
        target.extend(self.iter);
    }
}

impl<T: Send + Sync + 'static> IntoParallelIterator for Vec<T> {
    type Item = T;
    type Iter = VecParIter<T>;

    fn into_par_iter(self) -> Self::Iter {
        VecParIter::new(self)
    }
}

impl<'data, T: Send + Sync + 'data> IntoParallelRefIterator<'data> for Vec<T> {
    type Item = &'data T;
    type Iter = VecRefParIter<'data, T>;

    fn par_iter(&'data self) -> Self::Iter {
        VecRefParIter::new(self)
    }
}

/// Parallel iterator over vector references.
pub struct VecRefParIter<'data, T> {
    data: &'data Vec<T>,
}

impl<'data, T> VecRefParIter<'data, T> {
    fn new(data: &'data Vec<T>) -> Self {
        Self { data }
    }

    pub(in crate::parallel) fn into_slice(self) -> &'data [T] {
        self.data.as_slice()
    }

    /// Return matching logical positions without materializing borrowed items.
    pub fn positions<F>(self, predicate: F) -> VecRefPositions<'data, T, F>
    where
        F: Fn(&'data T) -> bool + Send + Sync + Clone,
    {
        VecRefPositions {
            data: self.data,
            predicate,
        }
    }
}

impl<'data, T: Send + Sync + 'data> ParallelIterator for VecRefParIter<'data, T> {
    type Item = &'data T;

    fn seq_items(self) -> Vec<Self::Item> {
        self.data.iter().collect()
    }

    fn seq_try_fold<Acc, B, FoldFn>(self, init: Acc, fold_fn: FoldFn) -> ControlFlow<B, Acc>
    where
        FoldFn: FnMut(Acc, Self::Item) -> ControlFlow<B, Acc>,
    {
        self.data.iter().try_fold(init, fold_fn)
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        // Drive the backing storage directly. Collecting `Vec<&T>` first cost
        // one pointer per element before any work started, and every split
        // below then copied halves of that pointer vector.
        SliceParIter::new(self.data.as_slice()).drive(consumer)
    }
}

/// Borrowed shard addressed as a subslice of one shared slice.
///
/// Splitting is `slice::split_at`, so neither an element nor a reference to one
/// is copied at any depth of the drive recursion.
struct SliceParIter<'data, T> {
    data: &'data [T],
}

impl<'data, T> SliceParIter<'data, T> {
    fn new(data: &'data [T]) -> Self {
        Self { data }
    }
}

impl<'data, T: Send + Sync + 'data> ParallelIterator for SliceParIter<'data, T> {
    type Item = &'data T;

    fn seq_items(self) -> Vec<Self::Item> {
        self.data.iter().collect()
    }

    fn seq_try_fold<Acc, B, FoldFn>(self, init: Acc, fold_fn: FoldFn) -> ControlFlow<B, Acc>
    where
        FoldFn: FnMut(Acc, Self::Item) -> ControlFlow<B, Acc>,
    {
        self.data.iter().try_fold(init, fold_fn)
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        if self.data.len() <= PARALLEL_DRIVE_THRESHOLD {
            return consumer.consume(self);
        }

        let mid = self.data.len() / 2;
        let (left_data, right_data) = self.data.split_at(mid);
        let (left_consumer, right_consumer) = consumer.split_at(left_data.len());

        drive_split(
            SliceParIter::new(left_data),
            SliceParIter::new(right_data),
            left_consumer,
            right_consumer,
        )
    }
}

impl<'data, T: Send + Sync + 'data> IndexedParallelIterator for VecRefParIter<'data, T> {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn collect_into_vec(self, target: &mut Vec<Self::Item>) {
        target.clear();
        target.extend(self.data.iter());
    }
}

/// Position stream over borrowed vector storage.
pub struct VecRefPositions<'data, T, F> {
    data: &'data Vec<T>,
    predicate: F,
}

impl<'data, T, F> ParallelIterator for VecRefPositions<'data, T, F>
where
    T: Send + Sync + 'data,
    F: Fn(&'data T) -> bool + Send + Sync + Clone,
{
    type Item = usize;

    fn seq_items(self) -> Vec<Self::Item> {
        self.data
            .iter()
            .enumerate()
            .filter_map(|(index, item)| (self.predicate)(item).then_some(index))
            .collect()
    }

    fn seq_try_fold<Acc, B, FoldFn>(self, init: Acc, fold_fn: FoldFn) -> ControlFlow<B, Acc>
    where
        FoldFn: FnMut(Acc, Self::Item) -> ControlFlow<B, Acc>,
    {
        self.data
            .iter()
            .enumerate()
            .filter_map(|(index, item)| (self.predicate)(item).then_some(index))
            .try_fold(init, fold_fn)
    }

    /// # Why this stays sequential
    ///
    /// The yielded item is a logical index, so this stream needs the offset
    /// documented as absent on the `Enumerate` adapter.
    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

/// A parallel iterator specifically for reference vectors.
pub struct RefVecParIter<'a, T> {
    data: Vec<&'a T>,
}

impl<'a, T> RefVecParIter<'a, T> {
    fn new(data: Vec<&'a T>) -> Self {
        Self { data }
    }
}

impl<'a, T: Send + Sync> ParallelIterator for RefVecParIter<'a, T> {
    type Item = &'a T;

    fn seq_items(self) -> Vec<Self::Item> {
        self.data
    }

    fn seq_try_fold<Acc, B, FoldFn>(self, init: Acc, fold_fn: FoldFn) -> ControlFlow<B, Acc>
    where
        FoldFn: FnMut(Acc, Self::Item) -> ControlFlow<B, Acc>,
    {
        self.data.into_iter().try_fold(init, fold_fn)
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        // Sequential base case: consumer.consume(RefVecParIter) is safe because
        // the base consumers terminate on the item stream rather than driving
        // it again.
        if self.data.len() <= PARALLEL_DRIVE_THRESHOLD {
            return consumer.consume(self);
        }

        let mut data = self.data;
        let mid = data.len() / 2;
        let right_data = data.split_off(mid);
        let left_data = std::mem::take(&mut data);

        let (left_consumer, right_consumer) = consumer.split_at(left_data.len());

        drive_split(
            RefVecParIter::new(left_data),
            RefVecParIter::new(right_data),
            left_consumer,
            right_consumer,
        )
    }
}

impl<'a, T: Send + Sync> IndexedParallelIterator for RefVecParIter<'a, T> {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn collect_into_vec(self, target: &mut Vec<Self::Item>) {
        move_vec_items_into(self.data, target);
    }
}

impl IntoParallelIterator for std::ops::Range<usize> {
    type Item = usize;
    type Iter = RangeParIter<usize>;

    fn into_par_iter(self) -> Self::Iter {
        RangeParIter::new(self.start, self.end)
    }
}

impl<T: Send + Sync> ParallelExtend<T> for Vec<T> {
    fn par_extend<I>(&mut self, par_iter: I)
    where
        I: ParallelIterator<Item = T>,
    {
        // Drive through CollectConsumer rather than calling collect::<Vec<_>>(), which
        // would call par_extend again and create infinite mutual recursion.
        self.extend(par_iter.drive(CollectConsumer::new()));
    }
}
