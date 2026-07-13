use super::{
    CollectConsumer, Consumer, IndexedParallelIterator, IntoParallelIterator,
    IntoParallelRefIterator, ParallelExtend, ParallelIterator,
};

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

    fn drive<C, R>(mut self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        if self.data.len() <= 1 {
            return consumer.consume(SequentialIterAdapter::new(self.data.into_iter()));
        }

        let mid = self.data.len() / 2;
        let right_data = self.data.split_off(mid);
        let left_data = std::mem::take(&mut self.data);

        let (left_consumer, right_consumer) = consumer.split_at(left_data.len());

        let left_result = left_consumer.consume(VecParIter::new(left_data));
        let right_result = right_consumer.consume(VecParIter::new(right_data));

        C::combine(left_result, right_result)
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

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        let refs: Vec<&'data T> = self.data.iter().collect();
        consumer.consume(RefVecParIter::new(refs))
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

    fn drive<C, R>(mut self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        // Sequential base case: consumer.consume(RefVecParIter) is safe because
        // CollectConsumer::consume now calls seq_items(), terminating the chain.
        if self.data.len() <= 1 {
            return consumer.consume(RefVecParIter::new(self.data));
        }

        let mid = self.data.len() / 2;
        let right_data = self.data.split_off(mid);
        let left_data = std::mem::take(&mut self.data);

        let (left_consumer, right_consumer) = consumer.split_at(left_data.len());

        let left_result = left_consumer.consume(RefVecParIter::new(left_data));
        let right_result = right_consumer.consume(RefVecParIter::new(right_data));

        C::combine(left_result, right_result)
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
