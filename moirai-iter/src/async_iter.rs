//! Async iterator implementations for I/O-bound workloads
//!
//! This module provides async-native iterator functionality that integrates
//! with Moirai's unified async runtime for efficient I/O processing.

#![allow(dead_code)] // Development structures per ADR requirements - will be used in future iterations

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::stream::{self, StreamExt};

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

    /// Execute side effects in parallel with async operations
    fn par_for_each<F, Fut>(self, concurrency: usize, func: F) -> ParAsyncForEach<Self, F>
    where
        Self: Sized,
        F: Fn(Self::Item) -> Fut + Send + Sync,
        Fut: Future<Output = ()> + Send,
    {
        ParAsyncForEach::new(self, concurrency, func)
    }
}

/// Trait for converting types into async iterators
pub trait IntoAsyncIterator {
    type Item: Send;
    type IntoAsyncIter: AsyncIterator<Item = Self::Item>;

    fn into_async_iter(self) -> Self::IntoAsyncIter;
}

impl<T: Send + 'static> IntoAsyncIterator for Vec<T> {
    type Item = T;
    type IntoAsyncIter = AsyncVecIter<T>;

    fn into_async_iter(self) -> Self::IntoAsyncIter {
        AsyncVecIter::new(self)
    }
}

/// Async iterator over a vector
pub struct AsyncVecIter<T> {
    items: Vec<T>,
    index: usize,
}

impl<T: Send + 'static> AsyncVecIter<T> {
    pub fn new(items: Vec<T>) -> Self {
        Self { items, index: 0 }
    }
}

impl<T: Send + 'static> AsyncIterator for AsyncVecIter<T> {
    type Item = T;

    fn into_vec(self) -> Vec<Self::Item> {
        self.items
    }
}

/// Async range iterator
pub struct AsyncRangeIter {
    start: usize,
    end: usize,
    current: usize,
}

impl AsyncRangeIter {
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            current: start,
        }
    }
}

impl AsyncIterator for AsyncRangeIter {
    type Item = usize;

    fn into_vec(self) -> Vec<Self::Item> {
        (self.start..self.end).collect()
    }
}

/// Async map operation
pub struct AsyncMap<I, F> {
    iter: I,
    map_fn: F,
}

impl<I, F> AsyncMap<I, F> {
    fn new(iter: I, map_fn: F) -> Self {
        Self { iter, map_fn }
    }
}

impl<I, F, Fut, R> AsyncIterator for AsyncMap<I, F>
where
    I: AsyncIterator,
    F: Fn(I::Item) -> Fut + Send + Sync,
    Fut: Future<Output = R> + Send,
    R: Send,
{
    type Item = R;

    fn into_vec(self) -> Vec<Self::Item> {
        self.iter
            .into_vec()
            .into_iter()
            .map(|item| futures::executor::block_on((self.map_fn)(item)))
            .collect()
    }
}

/// Async filter operation
pub struct AsyncFilter<I, F> {
    iter: I,
    filter_fn: F,
}

impl<I, F> AsyncFilter<I, F> {
    fn new(iter: I, filter_fn: F) -> Self {
        Self { iter, filter_fn }
    }
}

impl<I, F, Fut> AsyncIterator for AsyncFilter<I, F>
where
    I: AsyncIterator,
    F: Fn(&I::Item) -> Fut + Send + Sync,
    Fut: Future<Output = bool> + Send,
{
    type Item = I::Item;

    fn into_vec(self) -> Vec<Self::Item> {
        self.iter
            .into_vec()
            .into_iter()
            .filter(|item| futures::executor::block_on((self.filter_fn)(item)))
            .collect()
    }
}

/// Async take operation with prefix-bounded value semantics.
pub struct AsyncTake<I> {
    iter: I,
    count: usize,
}

impl<I> AsyncTake<I> {
    fn new(iter: I, count: usize) -> Self {
        Self { iter, count }
    }
}

impl<I> AsyncIterator for AsyncTake<I>
where
    I: AsyncIterator,
{
    type Item = I::Item;

    fn into_vec(self) -> Vec<Self::Item> {
        let mut items = self.iter.into_vec();
        items.truncate(self.count);
        items
    }
}

/// Async skip operation with prefix-discarding value semantics.
pub struct AsyncSkip<I> {
    iter: I,
    count: usize,
}

impl<I> AsyncSkip<I> {
    fn new(iter: I, count: usize) -> Self {
        Self { iter, count }
    }
}

impl<I> AsyncIterator for AsyncSkip<I>
where
    I: AsyncIterator,
{
    type Item = I::Item;

    fn into_vec(self) -> Vec<Self::Item> {
        let mut items = self.iter.into_vec();
        if self.count >= items.len() {
            Vec::new()
        } else {
            items.drain(..self.count);
            items
        }
    }
}

/// Async enumerate operation with zero-based logical positions.
pub struct AsyncEnumerate<I> {
    iter: I,
}

impl<I> AsyncEnumerate<I> {
    fn new(iter: I) -> Self {
        Self { iter }
    }
}

impl<I> AsyncIterator for AsyncEnumerate<I>
where
    I: AsyncIterator,
{
    type Item = (usize, I::Item);

    fn into_vec(self) -> Vec<Self::Item> {
        self.iter.into_vec().into_iter().enumerate().collect()
    }
}

/// Async zip operation with shortest-input semantics.
pub struct AsyncZip<I, J> {
    left: I,
    right: J,
}

impl<I, J> AsyncZip<I, J> {
    fn new(left: I, right: J) -> Self {
        Self { left, right }
    }
}

impl<I, J> AsyncIterator for AsyncZip<I, J>
where
    I: AsyncIterator,
    J: AsyncIterator,
{
    type Item = (I::Item, J::Item);

    fn into_vec(self) -> Vec<Self::Item> {
        self.left
            .into_vec()
            .into_iter()
            .zip(self.right.into_vec())
            .collect()
    }
}

/// Async for_each operation
pub struct AsyncForEach<I, F> {
    iter: Option<I>,
    func: F,
}

impl<I, F> AsyncForEach<I, F> {
    fn new(iter: I, func: F) -> Self {
        Self {
            iter: Some(iter),
            func,
        }
    }
}

impl<I, F, Fut> Future for AsyncForEach<I, F>
where
    I: AsyncIterator,
    F: Fn(I::Item) -> Fut + Send + Sync,
    Fut: Future<Output = ()> + Send,
    I: Unpin,
    F: Unpin,
{
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        if let Some(iter) = this.iter.take() {
            for item in iter.into_vec() {
                futures::executor::block_on((this.func)(item));
            }
        }
        Poll::Ready(())
    }
}

/// Async collect operation
pub struct AsyncCollect<I, C> {
    iter: Option<I>,
    _phantom: PhantomData<C>,
}

impl<I, C> AsyncCollect<I, C> {
    fn new(iter: I) -> Self {
        Self {
            iter: Some(iter),
            _phantom: PhantomData,
        }
    }
}

impl<I, C> Future for AsyncCollect<I, C>
where
    I: AsyncIterator,
    C: Default + Extend<I::Item> + Send,
    I: Unpin,
    C: Unpin,
{
    type Output = C;

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        let mut collection = C::default();
        if let Some(iter) = this.iter.take() {
            collection.extend(iter.into_vec());
        }
        Poll::Ready(collection)
    }
}

/// Async fold operation
pub struct AsyncFold<I, T, F> {
    iter: Option<I>,
    accumulator: Option<T>,
    fold_fn: F,
}

impl<I, T, F> AsyncFold<I, T, F> {
    fn new(iter: I, init: T, fold_fn: F) -> Self {
        Self {
            iter: Some(iter),
            accumulator: Some(init),
            fold_fn,
        }
    }
}

impl<I, T, F, Fut> Future for AsyncFold<I, T, F>
where
    I: AsyncIterator,
    F: Fn(T, I::Item) -> Fut + Send + Sync,
    Fut: Future<Output = T> + Send,
    T: Send + Unpin,
    I: Unpin,
    F: Unpin,
{
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        let mut accumulator = this
            .accumulator
            .take()
            .expect("async fold polled after completion");
        if let Some(iter) = this.iter.take() {
            for item in iter.into_vec() {
                accumulator = futures::executor::block_on((this.fold_fn)(accumulator, item));
            }
        }
        Poll::Ready(accumulator)
    }
}

/// Async reduce operation
pub struct AsyncReduce<I, F> {
    iter: Option<I>,
    reduce_fn: F,
}

impl<I, F> AsyncReduce<I, F> {
    fn new(iter: I, reduce_fn: F) -> Self {
        Self {
            iter: Some(iter),
            reduce_fn,
        }
    }
}

impl<I, F, Fut> Future for AsyncReduce<I, F>
where
    I: AsyncIterator,
    F: Fn(I::Item, I::Item) -> Fut + Send + Sync,
    Fut: Future<Output = I::Item> + Send,
    I: Unpin,
    F: Unpin,
{
    type Output = Option<I::Item>;

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        let Some(iter) = this.iter.take() else {
            return Poll::Ready(None);
        };
        let mut items = iter.into_vec().into_iter();
        let Some(mut accumulator) = items.next() else {
            return Poll::Ready(None);
        };
        for item in items {
            accumulator = futures::executor::block_on((this.reduce_fn)(accumulator, item));
        }
        Poll::Ready(Some(accumulator))
    }
}

/// Adapter to make async iterators work with parallel processing
pub struct AsyncParallelAdapter<I> {
    iter: I,
}

impl<I> AsyncParallelAdapter<I> {
    fn new(iter: I) -> Self {
        Self { iter }
    }
}

impl<I: AsyncIterator> AsyncIterator for AsyncParallelAdapter<I> {
    type Item = I::Item;

    fn into_vec(self) -> Vec<Self::Item> {
        self.iter.into_vec()
    }
}

impl<I: AsyncIterator> AsyncParallelIterator for AsyncParallelAdapter<I> {}

/// Parallel async map with concurrency control
pub struct ParAsyncMap<I, F> {
    iter: I,
    concurrency: usize,
    map_fn: F,
}

impl<I, F> ParAsyncMap<I, F> {
    fn new(iter: I, concurrency: usize, map_fn: F) -> Self {
        Self {
            iter,
            concurrency,
            map_fn,
        }
    }
}

impl<I, F, Fut, R> AsyncIterator for ParAsyncMap<I, F>
where
    I: AsyncIterator,
    F: Fn(I::Item) -> Fut + Send + Sync,
    Fut: Future<Output = R> + Send,
    R: Send,
{
    type Item = R;

    fn into_vec(self) -> Vec<Self::Item> {
        let concurrency = self.concurrency.max(1);
        let map_fn = self.map_fn;
        let items = self.iter.into_vec();
        futures::executor::block_on(async move {
            stream::iter(items)
                .map(|item| {
                    let map_fn = &map_fn;
                    async move { map_fn(item).await }
                })
                .buffered(concurrency)
                .collect()
                .await
        })
    }
}

/// Parallel async filter with concurrency control
pub struct ParAsyncFilter<I, F> {
    iter: I,
    concurrency: usize,
    filter_fn: F,
}

impl<I, F> ParAsyncFilter<I, F> {
    fn new(iter: I, concurrency: usize, filter_fn: F) -> Self {
        Self {
            iter,
            concurrency,
            filter_fn,
        }
    }
}

impl<I, F, Fut> AsyncIterator for ParAsyncFilter<I, F>
where
    I: AsyncIterator,
    F: Fn(&I::Item) -> Fut + Send + Sync,
    Fut: Future<Output = bool> + Send,
{
    type Item = I::Item;

    fn into_vec(self) -> Vec<Self::Item> {
        let concurrency = self.concurrency.max(1);
        let filter_fn = self.filter_fn;
        let items = self.iter.into_vec();
        futures::executor::block_on(async move {
            stream::iter(items)
                .map(|item| {
                    let filter_fn = &filter_fn;
                    async move {
                        let keep = filter_fn(&item).await;
                        (item, keep)
                    }
                })
                .buffered(concurrency)
                .filter_map(|(item, keep)| async move { keep.then_some(item) })
                .collect()
                .await
        })
    }
}

/// Parallel async for_each with concurrency control
pub struct ParAsyncForEach<I, F> {
    iter: Option<I>,
    concurrency: usize,
    func: F,
}

impl<I, F> ParAsyncForEach<I, F> {
    fn new(iter: I, concurrency: usize, func: F) -> Self {
        Self {
            iter: Some(iter),
            concurrency,
            func,
        }
    }
}

impl<I, F, Fut> Future for ParAsyncForEach<I, F>
where
    I: AsyncIterator,
    F: Fn(I::Item) -> Fut + Send + Sync,
    Fut: Future<Output = ()> + Send,
    I: Unpin,
    F: Unpin,
{
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        let concurrency = this.concurrency.max(1);
        if let Some(iter) = this.iter.take() {
            let func = &this.func;
            let items = iter.into_vec();
            futures::executor::block_on(async {
                stream::iter(items)
                    .map(|item| async move { func(item).await })
                    .buffered(concurrency)
                    .for_each(|_| async {})
                    .await;
            });
        }
        Poll::Ready(())
    }
}

#[cfg(test)]
#[path = "async_iter_tests.rs"]
mod async_iter_tests;
