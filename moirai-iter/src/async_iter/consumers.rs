//! Consumer futures driving async iterators to completion.
//!
//! Each terminal drives the per-item user futures *cooperatively*: it holds the
//! materialized item list plus the currently in-flight user future as state, and
//! on every [`Future::poll`] it polls that in-flight future. On `Ready` it folds
//! the result and advances to the next item; on `Pending` it returns `Pending`
//! so the outer waker propagates. No terminal blocks the executor.
//!
//! The user closures return unnameable `impl Future` types, so the in-flight
//! future is type-erased into a `Pin<Box<dyn Future>>` — one allocation per item.
//! The alternative (threading a generic future type through the trait surface)
//! would churn every caller; boxing keeps the public shape stable.

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll};

use super::traits::AsyncIterator;

/// Type-erased in-flight user future.
type ItemFuture<O> = Pin<Box<dyn Future<Output = O> + Send>>;

/// Async for_each operation.
pub struct AsyncForEach<I, F>
where
    I: AsyncIterator,
{
    /// Remaining items in forward order, consumed from the back for O(1) pops.
    items: Option<Vec<I::Item>>,
    func: F,
    in_flight: Option<ItemFuture<()>>,
}

impl<I, F> AsyncForEach<I, F>
where
    I: AsyncIterator,
{
    pub(super) fn new(iter: I, func: F) -> Self
    where
        I: Sized,
    {
        let mut items = iter.into_vec();
        // Reverse once so `pop` yields items in original order.
        items.reverse();
        Self {
            items: Some(items),
            func,
            in_flight: None,
        }
    }
}

impl<I, F, Fut> Future for AsyncForEach<I, F>
where
    I: AsyncIterator,
    F: Fn(I::Item) -> Fut + Send + Sync,
    Fut: Future<Output = ()> + Send + 'static,
    I: Unpin,
    F: Unpin,
    I::Item: Unpin,
{
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        loop {
            if let Some(fut) = this.in_flight.as_mut() {
                match fut.as_mut().poll(cx) {
                    Poll::Ready(()) => this.in_flight = None,
                    Poll::Pending => return Poll::Pending,
                }
            }
            let items = this
                .items
                .as_mut()
                .expect("async for_each polled after completion");
            match items.pop() {
                Some(item) => this.in_flight = Some(Box::pin((this.func)(item))),
                None => {
                    this.items = None;
                    return Poll::Ready(());
                }
            }
        }
    }
}

/// Async collect operation.
///
/// `collect` performs no per-item async work — items are materialized by the
/// source and extended into the target collection — so it completes in a single
/// poll without blocking.
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

/// Async fold operation.
pub struct AsyncFold<I, T, F>
where
    I: AsyncIterator,
{
    items: Option<Vec<I::Item>>,
    accumulator: Option<T>,
    fold_fn: F,
    in_flight: Option<ItemFuture<T>>,
}

impl<I, T, F> AsyncFold<I, T, F>
where
    I: AsyncIterator,
{
    pub(super) fn new(iter: I, init: T, fold_fn: F) -> Self
    where
        I: Sized,
    {
        let mut items = iter.into_vec();
        items.reverse();
        Self {
            items: Some(items),
            accumulator: Some(init),
            fold_fn,
            in_flight: None,
        }
    }
}

impl<I, T, F, Fut> Future for AsyncFold<I, T, F>
where
    I: AsyncIterator,
    F: Fn(T, I::Item) -> Fut + Send + Sync,
    Fut: Future<Output = T> + Send + 'static,
    T: Send + Unpin,
    I: Unpin,
    F: Unpin,
    I::Item: Unpin,
{
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        loop {
            if let Some(fut) = this.in_flight.as_mut() {
                match fut.as_mut().poll(cx) {
                    Poll::Ready(acc) => {
                        this.accumulator = Some(acc);
                        this.in_flight = None;
                    }
                    Poll::Pending => return Poll::Pending,
                }
            }
            let items = this
                .items
                .as_mut()
                .expect("async fold polled after completion");
            match items.pop() {
                Some(item) => {
                    let acc = this
                        .accumulator
                        .take()
                        .expect("fold accumulator present between items");
                    this.in_flight = Some(Box::pin((this.fold_fn)(acc, item)));
                }
                None => {
                    this.items = None;
                    return Poll::Ready(
                        this.accumulator
                            .take()
                            .expect("fold accumulator present at completion"),
                    );
                }
            }
        }
    }
}

/// Async reduce operation.
pub struct AsyncReduce<I, F>
where
    I: AsyncIterator,
{
    items: Option<Vec<I::Item>>,
    accumulator: Option<I::Item>,
    reduce_fn: F,
    in_flight: Option<ItemFuture<I::Item>>,
}

impl<I, F> AsyncReduce<I, F>
where
    I: AsyncIterator,
{
    pub(super) fn new(iter: I, reduce_fn: F) -> Self
    where
        I: Sized,
    {
        let mut items = iter.into_vec();
        items.reverse();
        // Seed the accumulator with the first logical item (last after reverse).
        let accumulator = items.pop();
        Self {
            items: Some(items),
            accumulator,
            reduce_fn,
            in_flight: None,
        }
    }
}

impl<I, F, Fut> Future for AsyncReduce<I, F>
where
    I: AsyncIterator,
    F: Fn(I::Item, I::Item) -> Fut + Send + Sync,
    Fut: Future<Output = I::Item> + Send + 'static,
    I: Unpin,
    F: Unpin,
    I::Item: Unpin,
{
    type Output = Option<I::Item>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        loop {
            if let Some(fut) = this.in_flight.as_mut() {
                match fut.as_mut().poll(cx) {
                    Poll::Ready(acc) => {
                        this.accumulator = Some(acc);
                        this.in_flight = None;
                    }
                    Poll::Pending => return Poll::Pending,
                }
            }
            let items = this
                .items
                .as_mut()
                .expect("async reduce polled after completion");
            match items.pop() {
                Some(item) => {
                    let acc = this
                        .accumulator
                        .take()
                        .expect("reduce accumulator present once seeded");
                    this.in_flight = Some(Box::pin((this.reduce_fn)(acc, item)));
                }
                None => {
                    this.items = None;
                    // Empty input yields `None`; otherwise the folded accumulator.
                    return Poll::Ready(this.accumulator.take());
                }
            }
        }
    }
}
