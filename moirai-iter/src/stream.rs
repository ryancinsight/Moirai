//! Concurrent stream combinators dispatched to the unified hybrid scheduler.
//!
//! The caller expresses **bounded concurrency** — how many item futures may be
//! in flight at once — and the hybrid `ThreadScheduler` decides *how* to run
//! them: cooperatively on its async lane, or in parallel across worker threads,
//! and (in future) across processes. The combinators never assume the work is
//! CPU-parallel; an item future may be I/O-bound and never leave one core. That
//! is why the API says `concurrent`, not `parallel`: concurrency is the
//! contract, the execution mechanism is the scheduler's to optimize.
//!
//! Design, building on the lessons of the [`parallel-stream`](https://docs.rs/parallel-stream)
//! crate but routed through moirai's own infrastructure:
//!
//! - **Unified scheduler.** Each item future is spawned on
//!   [`moirai_executor::global()`] — the same work-stealing scheduler that backs
//!   `spawn_async` and the parallel iterators — so the scheduler load-balances
//!   the items rather than a separate ambient runtime. The result is handed back
//!   through a one-shot channel so the consuming stream awaits it
//!   *cooperatively*; it never blocks a worker the way `TaskHandle::join` would.
//! - **Bounded by construction.** `limit` caps in-flight item futures via
//!   [`StreamExt::buffer_unordered`] — the central lesson from `parallel-stream`:
//!   stream fan-out must be bounded, never unbounded.
//! - **Monomorphized.** Generic over the stream, the closure, and the item
//!   future, returned through RPITIT (`-> impl Stream` / `impl Future`) — no
//!   `Box<dyn>` on the data path.
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

use futures::stream::{Stream, StreamExt};
use moirai_core::executor::TaskSpawner;

/// Spawn `fut` on the global unified scheduler and return a future that resolves
/// to its output. The result is handed back through a one-shot channel so the
/// consuming stream awaits it *cooperatively* — it never blocks a worker the way
/// `TaskHandle::join` would.
fn spawn_on_scheduler<Fut, R>(fut: Fut) -> impl Future<Output = R> + Send
where
    Fut: Future<Output = R> + Send + 'static,
    R: Send + 'static,
{
    let (tx, rx) = futures::channel::oneshot::channel();
    // `spawn_async` runs `fut` on a scheduler worker. The only failure is the
    // runtime shutting down, in which case the un-spawned future — and `tx` with
    // it — is dropped, so the consumer observes a cancelled item below: the
    // failure is surfaced, not masked.
    let _ = moirai_executor::global().spawn_async(async move {
        let output = fut.await;
        // Ignored only when the consumer has already dropped the stream (no
        // receiver wants the value).
        let _ = tx.send(output);
    });
    async move {
        rx.await.expect(
            "concurrent-stream item dropped before completing (runtime shut down or item panicked)",
        )
    }
}

/// Concurrent [`Stream`] combinators dispatched through the unified hybrid
/// scheduler.
///
/// Implemented for every [`Stream`]; bring it into scope to call the
/// `concurrent_*` methods on any stream. The `limit` argument bounds in-flight
/// item futures; the scheduler decides whether that concurrency is realized as
/// async multiplexing or thread/process parallelism.
pub trait ConcurrentStreamExt: Stream + Sized {
    /// Map each item through the async `f`, keeping up to `limit` item futures
    /// in flight on the scheduler and yielding results in completion order
    /// (unordered — no head-of-line blocking).
    ///
    /// `limit` is clamped to at least 1.
    fn concurrent_map<F, Fut, R>(self, limit: usize, mut f: F) -> impl Stream<Item = R> + Send
    where
        Self: Send + 'static,
        Self::Item: Send + 'static,
        F: FnMut(Self::Item) -> Fut + Send + 'static,
        Fut: Future<Output = R> + Send + 'static,
        R: Send + 'static,
    {
        // `f` runs sequentially here to *produce* each item future; the futures
        // themselves are what the scheduler runs concurrently.
        self.map(move |item| spawn_on_scheduler(f(item)))
            .buffer_unordered(limit.max(1))
    }

    /// Map and filter in one pass: each item is mapped through the async `f` to
    /// an `Option<R>` with up to `limit` futures in flight; `None` results are
    /// dropped. The fused form avoids yielding rejected items downstream.
    ///
    /// `limit` is clamped to at least 1.
    fn concurrent_filter_map<F, Fut, R>(self, limit: usize, f: F) -> impl Stream<Item = R> + Send
    where
        Self: Send + 'static,
        Self::Item: Send + 'static,
        F: FnMut(Self::Item) -> Fut + Send + 'static,
        Fut: Future<Output = Option<R>> + Send + 'static,
        R: Send + 'static,
    {
        self.concurrent_map(limit, f)
            .filter_map(|maybe| async move { maybe })
    }

    /// Retain items for which the async predicate `f` returns `true`, evaluating
    /// up to `limit` predicates in flight on the scheduler.
    ///
    /// The predicate is `Fn` + `Clone` because each in-flight item owns its own
    /// invocation; typical predicates are zero-sized fn items or `Copy`
    /// closures, so the clone is free. `limit` is clamped to at least 1.
    fn concurrent_filter<F, Fut>(self, limit: usize, f: F) -> impl Stream<Item = Self::Item> + Send
    where
        Self: Send + 'static,
        Self::Item: Send + 'static,
        F: Fn(&Self::Item) -> Fut + Clone + Send + 'static,
        Fut: Future<Output = bool> + Send + 'static,
    {
        self.concurrent_filter_map(limit, move |item| {
            // Clone the predicate into each item's future so it can borrow the
            // item it owns; the original `f` stays available for the next item.
            let f = f.clone();
            async move { f(&item).await.then_some(item) }
        })
    }

    /// Run the async `f` for every item with up to `limit` futures in flight on
    /// the scheduler, completing once every item is done.
    ///
    /// `limit` is clamped to at least 1.
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
