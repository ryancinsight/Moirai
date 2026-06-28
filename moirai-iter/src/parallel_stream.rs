//! Parallel streams — bounded-concurrency parallel processing of a `Stream`,
//! with each item's work dispatched to the unified `moirai` scheduler.
//!
//! Inspired by the [`parallel-stream`](https://docs.rs/parallel-stream) crate
//! (conceptually "Rayon for streams"), with two deliberate differences that make
//! it a first-class part of `moirai`:
//!
//! - **True parallelism on the unified scheduler.** Each item's future is
//!   spawned on [`moirai_executor::global()`] — the work-stealing
//!   `ThreadScheduler` — so items run on worker *threads*, not merely as
//!   concurrent tasks on one thread. `parallel-stream` spawns on the ambient
//!   async runtime; here it is `moirai`'s own scheduler, the same one the
//!   parallel iterators and `spawn_async` use.
//! - **Explicit bounded concurrency.** `limit` caps the number of in-flight item
//!   futures (resource control / backpressure), implemented with
//!   [`StreamExt::buffer_unordered`]. This is the central lesson from
//!   `parallel-stream`: parallel stream processing must be *bounded*, never an
//!   unbounded fan-out.
//!
//! The combinators are monomorphized — generic over the stream, the closure, and
//! the item future, returned through RPITIT (`-> impl Stream`/`impl Future`), so
//! there is no `Box<dyn>` on the data path.
//!
//! ```no_run
//! use futures::StreamExt;
//! use moirai_iter::parallel_stream::ParallelStreamExt;
//!
//! # async fn demo() {
//! let source = futures::stream::iter(0..1_000u64);
//! // Up to 16 items processed in parallel across the scheduler's workers.
//! let doubled: Vec<u64> = source.par_map(16, |x| async move { x * 2 }).collect().await;
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
    // `spawn_async` runs `fut` on a `ThreadScheduler` worker (async lane). The
    // only failure is the runtime shutting down, in which case the un-spawned
    // future — and `tx` with it — is dropped, so the consumer observes a
    // cancelled item below: the failure is surfaced, not masked.
    let _ = moirai_executor::global().spawn_async(async move {
        let output = fut.await;
        // Ignored only when the consumer has already dropped the stream (no
        // receiver wants the value).
        let _ = tx.send(output);
    });
    async move {
        rx.await.expect(
            "parallel-stream item dropped before completing (runtime shut down or item panicked)",
        )
    }
}

/// Parallel `Stream` combinators dispatched through the unified scheduler.
///
/// Implemented for every [`Stream`]; bring it into scope to call `par_map` /
/// `par_for_each` on any stream.
pub trait ParallelStreamExt: Stream + Sized {
    /// Map each item through the async `f`, running up to `limit` item futures
    /// concurrently on the unified scheduler and yielding results in completion
    /// order (unordered — no head-of-line blocking).
    ///
    /// `limit` bounds the in-flight work; it is clamped to at least 1.
    fn par_map<F, Fut, R>(self, limit: usize, mut f: F) -> impl Stream<Item = R> + Send
    where
        Self: Send + 'static,
        Self::Item: Send + 'static,
        F: FnMut(Self::Item) -> Fut + Send + 'static,
        Fut: Future<Output = R> + Send + 'static,
        R: Send + 'static,
    {
        self.map(move |item| spawn_on_scheduler(f(item)))
            .buffer_unordered(limit.max(1))
    }

    /// Run the async `f` for every item with up to `limit` item futures in
    /// flight on the unified scheduler, completing once every item is done.
    ///
    /// `limit` is clamped to at least 1.
    fn par_for_each<F, Fut>(self, limit: usize, f: F) -> impl Future<Output = ()> + Send
    where
        Self: Send + 'static,
        Self::Item: Send + 'static,
        F: FnMut(Self::Item) -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.par_map(limit, f).for_each(|()| async {})
    }
}

impl<S: Stream + Sized> ParallelStreamExt for S {}

#[cfg(test)]
mod tests;
