//! Consumer futures driving async iterators to completion.

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll};

use super::traits::AsyncIterator;

/// Async for_each operation
pub struct AsyncForEach<I, F> {
    iter: Option<I>,
    func: F,
}

impl<I, F> AsyncForEach<I, F> {
    pub(super) fn new(iter: I, func: F) -> Self {
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
    pub(super) fn new(iter: I) -> Self {
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
    pub(super) fn new(iter: I, init: T, fold_fn: F) -> Self {
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
    pub(super) fn new(iter: I, reduce_fn: F) -> Self {
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
