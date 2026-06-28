//! Core async iterator traits.

use std::future::Future;

use super::adapters::{
    AsyncEnumerate, AsyncFilter, AsyncMap, AsyncParallelAdapter, AsyncSkip, AsyncTake, AsyncZip,
};
use super::consumers::{AsyncCollect, AsyncFold, AsyncForEach, AsyncReduce};
use super::parallel::{self, ParAsyncFilter, ParAsyncMap};

/// Core async iterator trait for async/await compatible iteration
pub trait AsyncIterator: Send {
    /// The type of items yielded by this async iterator
    type Item: Send;

    /// Materialize the iterator into its logical item sequence.
    fn into_vec(self) -> Vec<Self::Item>
    where
        Self: Sized;

    /// Async map operation that transforms each element
    fn map<F, Fut, R>(self, map_fn: F) -> AsyncMap<Self, F>
    where
        Self: Sized,
        F: Fn(Self::Item) -> Fut + Send + Sync,
        Fut: Future<Output = R> + Send,
        R: Send,
    {
        AsyncMap::new(self, map_fn)
    }

    /// Async filter operation
    fn filter<F, Fut>(self, filter_fn: F) -> AsyncFilter<Self, F>
    where
        Self: Sized,
        F: Fn(&Self::Item) -> Fut + Send + Sync,
        Fut: Future<Output = bool> + Send,
    {
        AsyncFilter::new(self, filter_fn)
    }

    /// Retain at most `count` items from the logical async stream prefix.
    fn take(self, count: usize) -> AsyncTake<Self>
    where
        Self: Sized,
    {
        AsyncTake::new(self, count)
    }

    /// Discard `count` items from the logical async stream prefix.
    fn skip(self, count: usize) -> AsyncSkip<Self>
    where
        Self: Sized,
    {
        AsyncSkip::new(self, count)
    }

    /// Pair each item with its zero-based logical stream position.
    fn enumerate(self) -> AsyncEnumerate<Self>
    where
        Self: Sized,
    {
        AsyncEnumerate::new(self)
    }

    /// Pair items with another async iterator, stopping at the shorter input.
    fn zip<J>(self, other: J) -> AsyncZip<Self, J>
    where
        Self: Sized,
        J: AsyncIterator,
    {
        AsyncZip::new(self, other)
    }

    /// Async for_each operation with side effects
    fn for_each<F, Fut>(self, func: F) -> AsyncForEach<Self, F>
    where
        Self: Sized,
        F: Fn(Self::Item) -> Fut + Send + Sync,
        Fut: Future<Output = ()> + Send,
    {
        AsyncForEach::new(self, func)
    }

    /// Collect into a vector asynchronously
    fn collect<C>(self) -> AsyncCollect<Self, C>
    where
        Self: Sized,
        C: Default + Extend<Self::Item> + Send,
    {
        AsyncCollect::new(self)
    }

    /// Fold operation with async function
    fn fold<T, F, Fut>(self, init: T, fold_fn: F) -> AsyncFold<Self, T, F>
    where
        Self: Sized,
        F: Fn(T, Self::Item) -> Fut + Send + Sync,
        Fut: Future<Output = T> + Send,
        T: Send,
    {
        AsyncFold::new(self, init, fold_fn)
    }

    /// Reduce operation for async iterators
    fn reduce<F, Fut>(self, reduce_fn: F) -> AsyncReduce<Self, F>
    where
        Self: Sized,
        F: Fn(Self::Item, Self::Item) -> Fut + Send + Sync,
        Fut: Future<Output = Self::Item> + Send,
    {
        AsyncReduce::new(self, reduce_fn)
    }

    /// Convert to parallel iterator for hybrid processing
    fn into_parallel(self) -> AsyncParallelAdapter<Self>
    where
        Self: Sized,
    {
        AsyncParallelAdapter::new(self)
    }
}

/// Parallel async iterator for CPU+async hybrid workloads
pub trait AsyncParallelIterator: AsyncIterator {
    /// Execute async operations in parallel with controlled concurrency
    fn par_map<F, Fut, R>(self, concurrency: usize, map_fn: F) -> ParAsyncMap<Self, F>
    where
        Self: Sized,
        F: Fn(Self::Item) -> Fut + Send + Sync,
        Fut: Future<Output = R> + Send,
        R: Send,
    {
        ParAsyncMap::new(self, concurrency, map_fn)
    }

    /// Parallel async filter with concurrency control
    fn par_filter<F, Fut>(self, concurrency: usize, filter_fn: F) -> ParAsyncFilter<Self, F>
    where
        Self: Sized,
        F: Fn(&Self::Item) -> Fut + Send + Sync,
        Fut: Future<Output = bool> + Send,
    {
        ParAsyncFilter::new(self, concurrency, filter_fn)
    }

    /// Execute side effects in parallel with async operations.
    ///
    /// Returns a real `Future` driven by the caller's runtime; unlike a blocking
    /// consumer it never blocks the async executor.
    fn par_for_each<F, Fut>(self, concurrency: usize, func: F) -> impl Future<Output = ()> + Send
    where
        Self: Sized,
        F: Fn(Self::Item) -> Fut + Send + Sync,
        Fut: Future<Output = ()> + Send,
    {
        parallel::for_each(self, concurrency, func)
    }
}

/// Trait for converting types into async iterators
pub trait IntoAsyncIterator {
    /// The type of items yielded by this async iterator
    type Item: Send;
    /// The resulting async iterator type
    type IntoAsyncIter: AsyncIterator<Item = Self::Item>;

    /// Convert the type into an async iterator
    fn into_async_iter(self) -> Self::IntoAsyncIter;
}
