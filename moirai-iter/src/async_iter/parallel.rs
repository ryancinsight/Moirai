//! Parallel async operations with controlled concurrency.

use futures::stream::{self, StreamExt};
use std::future::Future;

use super::traits::AsyncIterator;
use crate::stream::{retained_buffered, retained_unordered};

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
            let futures = stream::iter(items).map(|item| {
                let map_fn = &map_fn;
                async move { map_fn(item).await }
            });
            retained_buffered(futures, concurrency).collect().await
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
            let futures = stream::iter(items).map(|item| {
                let filter_fn = &filter_fn;
                async move {
                    let keep = filter_fn(&item).await;
                    (item, keep)
                }
            });
            retained_buffered(futures, concurrency)
                .filter_map(|(item, keep)| async move { keep.then_some(item) })
                .collect()
                .await
        })
    }
}

/// Parallel async for_each with concurrency control.
///
/// Returns a real `Future` driven by the caller's runtime — it never blocks the
/// executor. `for_each` is order-independent, so it reuses completion-order
/// retained slots without head-of-line blocking while keeping at most
/// `concurrency` item futures in flight.
pub(super) async fn for_each<I, F, Fut>(iter: I, concurrency: usize, func: F)
where
    I: AsyncIterator,
    F: Fn(I::Item) -> Fut + Send + Sync,
    Fut: Future<Output = ()> + Send,
{
    let items = iter.into_vec();
    retained_unordered(stream::iter(items).map(func), concurrency)
        .for_each(|()| async {})
        .await;
}
