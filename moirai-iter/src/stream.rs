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
//!   a one-shot channel so the consuming stream awaits it *cooperatively*; it
//!   never blocks a worker the way `TaskHandle::join` would.
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

use futures::future::Either;
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
/// `concurrent_*` methods on any stream. See the [module docs](self) for when to
/// reach for these versus the inline [`StreamExt`] combinators.
pub trait ConcurrentStreamExt: Stream + Sized {
    /// Map each item through the async `f`, keeping up to `limit` item futures
    /// in flight and yielding results in completion order (unordered — no
    /// head-of-line blocking).
    ///
    /// `limit == 1` runs each item inline and sequentially with no spawn or
    /// cross-thread hop; `limit > 1` distributes items across the scheduler's
    /// worker threads. `limit` is clamped to at least 1.
    fn concurrent_map<F, Fut, R>(self, limit: usize, mut f: F) -> impl Stream<Item = R> + Send
    where
        Self: Send + 'static,
        Self::Item: Send + 'static,
        F: FnMut(Self::Item) -> Fut + Send + 'static,
        Fut: Future<Output = R> + Send + 'static,
        R: Send + 'static,
    {
        let limit = limit.max(1);
        // `f` runs sequentially here to *produce* each item future; whether that
        // future then runs inline (Right) or on a worker (Left) is the only
        // difference between the sequential and distributed paths.
        self.map(move |item| {
            let fut = f(item);
            if limit == 1 {
                // No concurrency requested: stay on this thread.
                // buffer_unordered(1) polls exactly one at a time, i.e. sequentially.
                Either::Right(fut)
            } else {
                Either::Left(spawn_on_scheduler(fut))
            }
        })
        .buffer_unordered(limit)
    }

    /// Run the async `f` for every item with up to `limit` futures in flight,
    /// completing once every item is done.
    ///
    /// Follows the same `limit` semantics as
    /// [`concurrent_map`](Self::concurrent_map): `1` is sequential and inline,
    /// `> 1` distributes across workers. `limit` is clamped to at least 1.
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
