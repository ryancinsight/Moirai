//! Parallel async operations with controlled concurrency.

use futures::stream::{self, StreamExt};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use super::traits::AsyncIterator;

/// Parallel async map with concurrency control
pub struct ParAsyncMap<I, F> {
    iter: I,
    concurrency: usize,
    map_fn: F,
}

impl<I, F> ParAsyncMap<I, F> {
    pub(super) fn new(iter: I, concurrency: usize, map_fn: F) -> Self {
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
    pub(super) fn new(iter: I, concurrency: usize, filter_fn: F) -> Self {
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
    pub(super) fn new(iter: I, concurrency: usize, func: F) -> Self {
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
