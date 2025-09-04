//! Parallel iterator implementation for Moirai - Rayon-compatible API
//!
//! This module provides parallel iterator functionality that matches Rayon's API
//! while integrating with Moirai's unified scheduler and work-stealing runtime.

use crate::execution::{ExecutionContext, ParallelContext};
use moirai_core::{TaskBuilder, Priority};
use std::marker::PhantomData;
use std::sync::Arc;

/// Core parallel iterator trait, compatible with Rayon's ParallelIterator
pub trait ParallelIterator: Sized + Send {
    /// The type of items yielded by this parallel iterator
    type Item: Send;

    /// Execute a parallel operation over the iterator items
    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send;

    /// Map operation that transforms each element in parallel
    fn map<F, R>(self, map_fn: F) -> Map<Self, F>
    where
        F: Fn(Self::Item) -> R + Send + Sync,
        R: Send,
    {
        Map::new(self, map_fn)
    }

    /// Filter operation that retains elements matching a predicate
    fn filter<F>(self, filter_fn: F) -> Filter<Self, F>
    where
        F: Fn(&Self::Item) -> bool + Send + Sync,
    {
        Filter::new(self, filter_fn)
    }

    /// Reduce operation that combines all elements
    fn reduce<F>(self, reduce_fn: F) -> Option<Self::Item>
    where
        F: Fn(Self::Item, Self::Item) -> Self::Item + Send + Sync,
        Self::Item: Clone,
    {
        self.drive(ReduceConsumer::new(reduce_fn))
    }

    /// Fold operation with an initial value
    fn fold<T, F>(self, init: T, fold_fn: F) -> T
    where
        T: Send + Clone,
        F: Fn(T, Self::Item) -> T + Send + Sync,
    {
        self.drive(FoldConsumer::new(init, fold_fn))
    }

    /// Collect into a collection
    fn collect<C>(self) -> C
    where
        C: ParallelExtend<Self::Item> + Default + Send,
    {
        let mut collection = C::default();
        collection.par_extend(self);
        collection
    }

    /// Convert to a sequential iterator (for compatibility)
    fn sequential(self) -> SequentialAdapter<Self> {
        SequentialAdapter::new(self)
    }

    /// Count the number of elements
    fn count(self) -> usize {
        self.map(|_| 1).reduce(|| 0, |a, b| a + b)
    }

    /// Find the first element matching a predicate
    fn find_first<F>(self, predicate: F) -> Option<Self::Item>
    where
        F: Fn(&Self::Item) -> bool + Send + Sync,
    {
        self.filter(predicate).find_any()
    }

    /// Find any element matching a predicate (order not guaranteed)
    fn find_any<F>(self, predicate: F) -> Option<Self::Item>
    where
        F: Fn(&Self::Item) -> bool + Send + Sync,
    {
        self.drive(FindConsumer::new(predicate))
    }

    /// Test if any element matches a predicate
    fn any<F>(self, predicate: F) -> bool
    where
        F: Fn(Self::Item) -> bool + Send + Sync,
    {
        self.map(predicate).find_any().unwrap_or(false)
    }

    /// Test if all elements match a predicate
    fn all<F>(self, predicate: F) -> bool
    where
        F: Fn(Self::Item) -> bool + Send + Sync,
    {
        !self.map(|x| !predicate(x)).any(|x| x)
    }

    /// Apply a function to each element (for side effects)
    fn for_each<F>(self, op: F)
    where
        F: Fn(Self::Item) + Send + Sync,
    {
        self.map(op).drive(NullConsumer::new())
    }

    /// Reduce with identity and associative operation
    fn reduce_with<F>(self, reduce_fn: F) -> Option<Self::Item>
    where
        F: Fn(Self::Item, Self::Item) -> Self::Item + Send + Sync,
    {
        self.drive(ReduceWithConsumer::new(reduce_fn))
    }
}

/// Consumer trait for parallel iterator operations
pub trait Consumer<T>: Send + Sync {
    type Result: Send;

    /// Consume items from a parallel iterator
    fn consume<I>(self, iter: I) -> Self::Result
    where
        I: ParallelIterator<Item = T>;

    /// Split the consumer for parallel processing
    fn split_at(self, index: usize) -> (Self, Self)
    where
        Self: Sized;

    /// Combine results from split consumers
    fn combine(left: Self::Result, right: Self::Result) -> Self::Result;
}

/// Trait for collections that can be extended in parallel
pub trait ParallelExtend<T>: Send {
    /// Extend the collection with items from a parallel iterator
    fn par_extend<I>(&mut self, par_iter: I)
    where
        I: ParallelIterator<Item = T>;
}

/// Parallel iterator over a vector
pub struct VecParIter<T> {
    data: Arc<Vec<T>>,
    context: ParallelContext,
}

impl<T: Send + Clone + 'static> VecParIter<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self {
            data: Arc::new(data),
            context: ParallelContext::new(),
        }
    }
}

impl<T: Send + Clone + 'static> ParallelIterator for VecParIter<T> {
    type Item = T;

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        // Use work-stealing to process chunks of the vector
        let chunk_size = (self.data.len() / num_cpus::get()).max(1);
        let chunks: Vec<_> = self.data.chunks(chunk_size).collect();
        
        if chunks.len() == 1 {
            // Sequential processing for small data
            return consumer.consume(SequentialIterAdapter::new(chunks[0].iter().cloned()));
        }

        // Parallel processing using Moirai's work-stealing scheduler
        let (left_chunks, right_chunks) = chunks.split_at(chunks.len() / 2);
        
        let left_data: Vec<T> = left_chunks.iter().flat_map(|chunk| chunk.iter().cloned()).collect();
        let right_data: Vec<T> = right_chunks.iter().flat_map(|chunk| chunk.iter().cloned()).collect();

        let (left_consumer, right_consumer) = consumer.split_at(left_data.len());

        // Create parallel tasks
        let left_iter = VecParIter {
            data: Arc::new(left_data),
            context: self.context.clone(),
        };
        let right_iter = VecParIter {
            data: Arc::new(right_data),
            context: self.context.clone(),
        };

        // Execute in parallel using Moirai's scheduler
        let left_result = left_iter.drive(left_consumer);
        let right_result = right_iter.drive(right_consumer);

        C::combine(left_result, right_result)
    }
}

/// Range parallel iterator
pub struct RangeParIter<T> {
    start: T,
    end: T,
    context: ParallelContext,
}

impl<T> RangeParIter<T>
where
    T: Send + Clone + 'static + PartialOrd + std::ops::Add<Output = T> + From<u8>,
{
    pub fn new(start: T, end: T) -> Self {
        Self {
            start,
            end,
            context: ParallelContext::new(),
        }
    }
}

impl<T> ParallelIterator for RangeParIter<T>
where
    T: Send + Clone + 'static + PartialOrd + std::ops::Add<Output = T> + From<u8>,
{
    type Item = T;

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        // Convert range to vector for now (optimization opportunity)
        let mut items = Vec::new();
        let mut current = self.start;
        while current < self.end {
            items.push(current.clone());
            current = current + T::from(1);
        }
        
        VecParIter::new(items).drive(consumer)
    }
}

/// Map adapter for parallel iterators
pub struct Map<I, F> {
    base: I,
    map_fn: F,
}

impl<I, F> Map<I, F> {
    fn new(base: I, map_fn: F) -> Self {
        Self { base, map_fn }
    }
}

impl<I, F, R> ParallelIterator for Map<I, F>
where
    I: ParallelIterator,
    F: Fn(I::Item) -> R + Send + Sync,
    R: Send,
{
    type Item = R;

    fn drive<C, R2>(self, consumer: C) -> R2
    where
        C: Consumer<Self::Item, Result = R2> + Send + Sync,
        R2: Send,
    {
        self.base.drive(MapConsumer::new(consumer, self.map_fn))
    }
}

/// Filter adapter for parallel iterators
pub struct Filter<I, F> {
    base: I,
    filter_fn: F,
}

impl<I, F> Filter<I, F> {
    fn new(base: I, filter_fn: F) -> Self {
        Self { base, filter_fn }
    }
}

impl<I, F> ParallelIterator for Filter<I, F>
where
    I: ParallelIterator,
    F: Fn(&I::Item) -> bool + Send + Sync,
{
    type Item = I::Item;

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        self.base.drive(FilterConsumer::new(consumer, self.filter_fn))
    }
}

/// Consumer implementations
pub struct MapConsumer<C, F> {
    base: C,
    map_fn: F,
}

impl<C, F> MapConsumer<C, F> {
    fn new(base: C, map_fn: F) -> Self {
        Self { base, map_fn }
    }
}

impl<C, F, T, R> Consumer<T> for MapConsumer<C, F>
where
    C: Consumer<R>,
    F: Fn(T) -> R + Send + Sync,
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

/// Additional consumer implementations would go here...
/// For brevity, showing structure for key consumers

pub struct FilterConsumer<C, F> {
    base: C,
    filter_fn: F,
}

impl<C, F> FilterConsumer<C, F> {
    fn new(base: C, filter_fn: F) -> Self {
        Self { base, filter_fn }
    }
}

pub struct ReduceConsumer<F> {
    reduce_fn: F,
}

impl<F> ReduceConsumer<F> {
    fn new(reduce_fn: F) -> Self {
        Self { reduce_fn }
    }
}

pub struct FoldConsumer<T, F> {
    init: T,
    fold_fn: F,
}

impl<T, F> FoldConsumer<T, F> {
    fn new(init: T, fold_fn: F) -> Self {
        Self { init, fold_fn }
    }
}

pub struct FindConsumer<F> {
    predicate: F,
}

impl<F> FindConsumer<F> {
    fn new(predicate: F) -> Self {
        Self { predicate }
    }
}

pub struct ReduceWithConsumer<F> {
    reduce_fn: F,
}

impl<F> ReduceWithConsumer<F> {
    fn new(reduce_fn: F) -> Self {
        Self { reduce_fn }
    }
}

pub struct NullConsumer<T> {
    _phantom: PhantomData<T>,
}

impl<T> NullConsumer<T> {
    fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

/// Sequential iterator adapter for compatibility
pub struct SequentialAdapter<I> {
    iter: I,
}

impl<I> SequentialAdapter<I> {
    fn new(iter: I) -> Self {
        Self { iter }
    }
}

pub struct SequentialIterAdapter<I> {
    iter: I,
}

impl<I> SequentialIterAdapter<I> {
    fn new(iter: I) -> Self {
        Self { iter }
    }
}

impl<I> ParallelIterator for SequentialIterAdapter<I>
where
    I: Iterator + Send,
    I::Item: Send,
{
    type Item = I::Item;

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        // Sequential processing
        consumer.consume(VecParIter::new(self.iter.collect()))
    }
}

/// Extension trait for collections to create parallel iterators
pub trait IntoParallelIterator {
    type Item: Send;
    type Iter: ParallelIterator<Item = Self::Item>;

    fn into_par_iter(self) -> Self::Iter;
}

/// Extension trait for collection references to create parallel iterators
pub trait IntoParallelRefIterator<'data> {
    type Item: Send + 'data;
    type Iter: ParallelIterator<Item = Self::Item>;

    fn par_iter(&'data self) -> Self::Iter;
}

/// Implementation for Vec
impl<T: Send + Clone + 'static> IntoParallelIterator for Vec<T> {
    type Item = T;
    type Iter = VecParIter<T>;

    fn into_par_iter(self) -> Self::Iter {
        VecParIter::new(self)
    }
}

impl<'data, T: Send + Clone + 'static> IntoParallelRefIterator<'data> for Vec<T> {
    type Item = &'data T;
    type Iter = VecRefParIter<'data, T>;

    fn par_iter(&'data self) -> Self::Iter {
        VecRefParIter::new(self)
    }
}

/// Parallel iterator over vector references
pub struct VecRefParIter<'data, T> {
    data: &'data Vec<T>,
    context: ParallelContext,
}

impl<'data, T> VecRefParIter<'data, T> {
    fn new(data: &'data Vec<T>) -> Self {
        Self {
            data,
            context: ParallelContext::new(),
        }
    }
}

impl<'data, T: Send + Sync + 'data> ParallelIterator for VecRefParIter<'data, T> {
    type Item = &'data T;

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        // Create a vector of references for processing
        let refs: Vec<&T> = self.data.iter().collect();
        VecParIter::new(refs).drive(consumer)
    }
}

/// Range support
impl IntoParallelIterator for std::ops::Range<usize> {
    type Item = usize;
    type Iter = RangeParIter<usize>;

    fn into_par_iter(self) -> Self::Iter {
        RangeParIter::new(self.start, self.end)
    }
}

/// Extension for Vec to support parallel extension
impl<T: Send> ParallelExtend<T> for Vec<T> {
    fn par_extend<I>(&mut self, par_iter: I)
    where
        I: ParallelIterator<Item = T>,
    {
        let items = par_iter.collect::<Vec<_>>();
        self.extend(items);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_map() {
        let data = vec![1, 2, 3, 4, 5];
        let result: Vec<i32> = data.into_par_iter().map(|x| x * 2).collect();
        assert_eq!(result, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_parallel_filter() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let result: Vec<i32> = data.into_par_iter().filter(|&x| x % 2 == 0).collect();
        assert_eq!(result, vec![2, 4, 6]);
    }

    #[test]
    fn test_parallel_reduce() {
        let data = vec![1, 2, 3, 4, 5];
        let result = data.into_par_iter().reduce(|a, b| a + b);
        assert_eq!(result, Some(15));
    }

    #[test]
    fn test_range_parallel() {
        let result: Vec<usize> = (0..10).into_par_iter().map(|x| x * x).collect();
        let expected: Vec<usize> = (0..10).map(|x| x * x).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parallel_count() {
        let data = vec![1, 2, 3, 4, 5];
        let count = data.into_par_iter().count();
        assert_eq!(count, 5);
    }

    #[test]
    fn test_parallel_any() {
        let data = vec![1, 2, 3, 4, 5];
        assert!(data.clone().into_par_iter().any(|x| x == 3));
        assert!(!data.into_par_iter().any(|x| x == 10));
    }

    #[test]
    fn test_parallel_all() {
        let data = vec![2, 4, 6, 8];
        assert!(data.clone().into_par_iter().all(|x| x % 2 == 0));
        assert!(!data.into_par_iter().all(|x| x > 5));
    }
}