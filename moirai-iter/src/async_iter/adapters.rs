//! Adapters transforming async iterators.

use super::traits::{AsyncIterator, AsyncParallelIterator};
use std::future::Future;

/// Async map operation
pub struct AsyncMap<I, F> {
    iter: I,
    map_fn: F,
}

impl<I, F> AsyncMap<I, F> {
    pub(super) fn new(iter: I, map_fn: F) -> Self {
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
    pub(super) fn new(iter: I, filter_fn: F) -> Self {
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
    pub(super) fn new(iter: I, count: usize) -> Self {
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
    pub(super) fn new(iter: I, count: usize) -> Self {
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
    pub(super) fn new(iter: I) -> Self {
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
    pub(super) fn new(left: I, right: J) -> Self {
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

/// Adapter to make async iterators work with parallel processing
pub struct AsyncParallelAdapter<I> {
    iter: I,
}

impl<I> AsyncParallelAdapter<I> {
    pub(super) fn new(iter: I) -> Self {
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
