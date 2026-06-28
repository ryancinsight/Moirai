//! Concurrent stream combinators dispatched to the unified hybrid scheduler.
//!
//! The caller expresses **how much concurrency the work warrants** via `limit`,
//! and the combinator does no more than that:
//!
//! - `limit == 1` — the work does not warrant concurrency: each item future runs
//!   **inline and sequentially** on the consuming thread, with no spawn, no
//!   cross-thread hop, and no channel. This is the zero-overhead path.
//! - `limit > 1` — items are **distributed across the scheduler's worker
//!   threads**, up to `limit` in flight, so the hybrid `ThreadScheduler` runs
//!   them in parallel.
//!
//! The API says `concurrent`, not `parallel`, because at `limit > 1` an item
//! future may still be I/O-bound and never saturate a core; concurrency is the
//! contract, the execution mechanism is the scheduler's to optimize.
//!
//! Two ordering disciplines, same dispatch:
//!
//! - [`concurrent_map`](ConcurrentStreamExt::concurrent_map) yields in
//!   **completion order** ([`StreamExt::buffer_unordered`]) — no head-of-line
//!   blocking, maximum throughput.
//! - [`concurrent_map_ordered`](ConcurrentStreamExt::concurrent_map_ordered)
//!   yields in **input order** ([`StreamExt::buffered`]) — a slow early item
//!   delays later-completed items, but order is preserved.
//!
//! Only operations whose per-item work is heavy enough to outweigh a thread hop
//! belong here. **Cheap, sequential operations — filtering on a simple
//! predicate, light maps — should use the standard [`StreamExt`] combinators**
//! (`map`, `filter`, `filter_map`), which run inline; they compose directly with
//! `concurrent_map`, e.g. `stream.concurrent_map(n, heavy).filter_map(cheap)`.
//!
//! Design, building on the lessons of the [`parallel-stream`](https://docs.rs/parallel-stream)
//! crate but routed through moirai's own infrastructure:
//!
//! - **Unified scheduler.** At `limit > 1` each item future is spawned on
//!   [`moirai_executor::global()`] — the same work-stealing scheduler that backs
//!   `spawn_async` and the parallel iterators. The result is handed back through
//!   a one-shot channel ([`ScheduledItem`]) so the consuming stream awaits it
//!   *cooperatively*; it never blocks a worker the way `TaskHandle::join` would.
//! - **Bounded by construction.** `limit` caps in-flight item futures — the
//!   central lesson from `parallel-stream`: stream fan-out must be bounded,
//!   never unbounded.
//! - **Monomorphized, zero-cost.** Generic over the stream, the closure, and the
//!   item future; the spawned result is a named [`ScheduledItem`] future and the
//!   sequential/distributed split is a [`Either`], so there is no `Box<dyn>` on
//!   the data path.
//!
//! # Performance — sizing `limit`
//!
//! The distributed path (`limit > 1`) pays a per-item dispatch cost: the spawn,
//! the one-shot hand-back, and a cross-thread wake. Measured on a 24-core
//! x86-64 box with identity item work (`tests/stream_overhead.rs`): ~0.8 µs per
//! item distributed, versus ~70 ns per item on the `limit == 1` inline path
//! (~12× cheaper). The spawn and cross-thread wake dominate that 0.8 µs; the
//! one-shot is a minority of it.
//!
//! Consequence: distribution is a net win only when **per-item work exceeds
//! roughly a microsecond**. Below that, the dispatch overhead dominates — prefer
//! `limit == 1` or the inline [`StreamExt`] combinators. The bounded `limit`
//! already exposes this choice; the measurement just sizes the crossover.
//!
//! ```no_run
//! use futures::StreamExt;
//! use moirai_iter::stream::ConcurrentStreamExt;
//!
//! # async fn demo() {
//! let source = futures::stream::iter(0..1_000u64);
//! // Up to 16 item futures in flight; the scheduler distributes them.
//! let doubled: Vec<u64> = source.concurrent_map(16, |x| async move { x * 2 }).collect().await;
//! # let _ = doubled;
//! # }
//! ```

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::channel::oneshot;
use futures::future::Either;
use futures::stream::{Stream, StreamExt};
use moirai_core::executor::TaskSpawner;

/// A future resolving to the output of an item spawned on the unified scheduler.
///
/// Awaiting it is *cooperative*: it registers the task waker on a one-shot
/// channel and yields the worker, rather than blocking it the way
/// `TaskHandle::join` would. Zero-cost — a single channel receiver, no `Box`.
#[must_use = "a ScheduledItem does nothing unless polled to completion"]
pub struct ScheduledItem<R> {
    rx: oneshot::Receiver<R>,
}

impl<R> Future for ScheduledItem<R> {
    type Output = R;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<R> {
        // `oneshot::Receiver` is `Unpin`, so `ScheduledItem` is too — poll the
        // receiver directly.
        match Pin::new(&mut self.get_mut().rx).poll(cx) {
            Poll::Ready(Ok(value)) => Poll::Ready(value),
            // Cancellation means the spawned task's sender was dropped without
            // sending: the runtime shut down before running it, or the item
            // future panicked (the spawn layer caught it). Surface it.
            Poll::Ready(Err(oneshot::Canceled)) => panic!(
                "concurrent-stream item dropped before completing (runtime shut down or item panicked)"
            ),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Spawn `fut` on the global unified scheduler, yielding a [`ScheduledItem`] the
/// consuming stream can await cooperatively.
fn spawn_on_scheduler<Fut, R>(fut: Fut) -> ScheduledItem<R>
where
    Fut: Future<Output = R> + Send + 'static,
    R: Send + 'static,
{
    let (tx, rx) = oneshot::channel();
    // `spawn_async` runs `fut` on a scheduler worker. The only failure is the
    // runtime shutting down, in which case the un-spawned future — and `tx` with
    // it — is dropped, so the consumer observes a cancelled item (handled in
    // `ScheduledItem::poll`): the failure is surfaced, not masked.
    let _ = moirai_executor::global().spawn_async(async move {
        let output = fut.await;
        // Ignored only when the consumer has already dropped the stream (no
        // receiver wants the value).
        let _ = tx.send(output);
    });
    ScheduledItem { rx }
}

/// Shared dispatch: turn a stream of items into a stream of per-item futures,
/// each either run inline (`limit == 1`, no concurrency requested) or spawned on
/// the scheduler (`limit > 1`). The caller applies the bound via `buffered` /
/// `buffer_unordered`; this is the single place the inline/distributed choice is
/// made, so the ordered and unordered combinators share it.
fn dispatch_items<S, F, Fut, R>(
    stream: S,
    limit: usize,
    mut f: F,
) -> impl Stream<Item = Either<ScheduledItem<R>, Fut>> + Send
where
    S: Stream + Send + 'static,
    S::Item: Send + 'static,
    F: FnMut(S::Item) -> Fut + Send + 'static,
    Fut: Future<Output = R> + Send + 'static,
    R: Send + 'static,
{
    // `f` runs sequentially here to *produce* each item future; whether that
    // future then runs inline (Right) or on a worker (Left) is the only
    // difference between the sequential and distributed paths.
    stream.map(move |item| {
        let fut = f(item);
        if limit == 1 {
            Either::Right(fut)
        } else {
            Either::Left(spawn_on_scheduler(fut))
        }
    })
}

/// Concurrent [`Stream`] combinators dispatched through the unified hybrid
/// scheduler.
///
/// Implemented for every [`Stream`]; bring it into scope to call the
/// `concurrent_*` methods on any stream. See the [module docs](self) for when to
/// reach for these versus the inline [`StreamExt`] combinators, and for the
/// ordered/unordered distinction.
pub trait ConcurrentStreamExt: Stream + Sized {
    /// Map each item through the async `f`, keeping up to `limit` item futures
    /// in flight and yielding results in **completion order** (unordered — no
    /// head-of-line blocking).
    ///
    /// `limit == 1` runs each item inline and sequentially with no spawn or
    /// cross-thread hop; `limit > 1` distributes items across the scheduler's
    /// worker threads. `limit` is clamped to at least 1.
    fn concurrent_map<F, Fut, R>(self, limit: usize, f: F) -> impl Stream<Item = R> + Send
    where
        Self: Send + 'static,
        Self::Item: Send + 'static,
        F: FnMut(Self::Item) -> Fut + Send + 'static,
        Fut: Future<Output = R> + Send + 'static,
        R: Send + 'static,
    {
        let limit = limit.max(1);
        dispatch_items(self, limit, f).buffer_unordered(limit)
    }

    /// Map each item through the async `f`, keeping up to `limit` item futures
    /// in flight and yielding results in **input order**.
    ///
    /// Order-preserving: a slow early item delays yielding later-completed items
    /// (head-of-line blocking). Prefer [`concurrent_map`](Self::concurrent_map)
    /// when order does not matter. Same `limit` semantics — `1` is sequential
    /// and inline, `> 1` distributes across workers; clamped to at least 1.
    fn concurrent_map_ordered<F, Fut, R>(self, limit: usize, f: F) -> impl Stream<Item = R> + Send
    where
        Self: Send + 'static,
        Self::Item: Send + 'static,
        F: FnMut(Self::Item) -> Fut + Send + 'static,
        Fut: Future<Output = R> + Send + 'static,
        R: Send + 'static,
    {
        let limit = limit.max(1);
        dispatch_items(self, limit, f).buffered(limit)
    }

    /// Run the async `f` for every item with up to `limit` futures in flight,
    /// completing once every item is done. Order-independent, so dispatched
    /// through the unordered path.
    ///
    /// Same `limit` semantics as [`concurrent_map`](Self::concurrent_map): `1`
    /// is sequential and inline, `> 1` distributes across workers; clamped to at
    /// least 1.
    fn concurrent_for_each<F, Fut>(self, limit: usize, f: F) -> impl Future<Output = ()> + Send
    where
        Self: Send + 'static,
        Self::Item: Send + 'static,
        F: FnMut(Self::Item) -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.concurrent_map(limit, f).for_each(|()| async {})
    }
}

impl<S: Stream + Sized> ConcurrentStreamExt for S {}

#[cfg(test)]
mod tests;
