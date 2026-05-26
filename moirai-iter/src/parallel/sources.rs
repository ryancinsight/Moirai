use super::{
    CollectConsumer, Consumer, IntoParallelIterator, IntoParallelRefIterator, ParallelExtend,
    ParallelIterator,
};
use std::sync::Arc;

/// Parallel iterator over a vector.
pub struct VecParIter<T> {
    data: Arc<Vec<T>>,
}

impl<T: Send + Clone + 'static> VecParIter<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self {
            data: Arc::new(data),
        }
    }
}

impl<T: Send + Sync + Clone + 'static> ParallelIterator for VecParIter<T> {
    type Item = T;

    fn seq_items(self) -> Vec<Self::Item> {
        Arc::try_unwrap(self.data).unwrap_or_else(|arc| (*arc).clone())
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        let chunk_size = (self.data.len() / num_cpus::get()).max(1);
        let chunks: Vec<_> = self.data.chunks(chunk_size).collect();

        if self.data.is_empty() {
            return consumer.consume(SequentialIterAdapter::new(std::iter::empty::<T>()));
        }

        if chunks.len() == 1 {
            return consumer.consume(SequentialIterAdapter::new(chunks[0].iter().cloned()));
        }

        let (left_chunks, right_chunks) = chunks.split_at(chunks.len() / 2);

        let left_data: Vec<T> = left_chunks
            .iter()
            .flat_map(|chunk| chunk.iter().cloned())
            .collect();
        let right_data: Vec<T> = right_chunks
            .iter()
            .flat_map(|chunk| chunk.iter().cloned())
            .collect();

        let (left_consumer, right_consumer) = consumer.split_at(left_data.len());

        let left_iter = VecParIter {
            data: Arc::new(left_data),
        };
        let right_iter = VecParIter {
            data: Arc::new(right_data),
        };

        let left_result = left_iter.drive(left_consumer);
        let right_result = right_iter.drive(right_consumer);

        C::combine(left_result, right_result)
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
        consumer.consume(VecNonCloneParIter::new(items))
    }
}

/// Parallel iterator over a vector that doesn't require Clone.
pub struct VecNonCloneParIter<T> {
    data: Vec<T>,
}

impl<T: Send + Sync + 'static> VecNonCloneParIter<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }
}

impl<T: Send + Sync + 'static> ParallelIterator for VecNonCloneParIter<T> {
    type Item = T;

    fn seq_items(self) -> Vec<Self::Item> {
        self.data
    }

    fn drive<C, R>(mut self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        // Sequential base case: hand items directly to the consumer.
        // Using seq_items avoids the infinite recursion that arose from
        // consumer.consume(VecNonCloneParIter::new(self.data)) when the consumer
        // dispatched back through drive.
        if self.data.len() <= 1 {
            return consumer.consume(SequentialIterAdapter::new(self.data.into_iter()));
        }

        let mid = self.data.len() / 2;
        let right_data = self.data.split_off(mid);
        let left_data = std::mem::take(&mut self.data);

        let (left_consumer, right_consumer) = consumer.split_at(left_data.len());

        let left_result = left_consumer.consume(VecNonCloneParIter::new(left_data));
        let right_result = right_consumer.consume(VecNonCloneParIter::new(right_data));

        C::combine(left_result, right_result)
    }
}

impl<T: Send + Sync + Clone + 'static> IntoParallelIterator for Vec<T> {
    type Item = T;
    type Iter = VecParIter<T>;

    fn into_par_iter(self) -> Self::Iter {
        VecParIter::new(self)
    }
}

impl<'data, T: Send + Sync + Clone + 'static> IntoParallelRefIterator<'data> for Vec<T> {
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
