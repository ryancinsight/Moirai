use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use crate::timer::delay::Delay;

/// Timeout wrapper for futures with comprehensive cancellation
pub struct Timeout<F> {
    future: F,
    delay: Delay,
}

impl<F> Timeout<F>
where
    F: Future,
{
    pub(super) fn new(future: F, duration: Duration) -> Self {
        Self {
            future,
            delay: Delay::new(duration),
        }
    }
}

impl<F> Future for Timeout<F>
where
    F: Future,
{
    type Output = Result<F::Output, TimeoutError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Safety: once `Timeout<F>` is pinned, its fields are not moved in
        // `poll`. Projecting the generic future in place preserves support for
        // `!Unpin` futures without allocating a `Pin<Box<F>>`.
        let this = unsafe { self.get_unchecked_mut() };

        // First check if the future is ready
        let future = unsafe { Pin::new_unchecked(&mut this.future) };
        if let Poll::Ready(output) = future.poll(cx) {
            return Poll::Ready(Ok(output));
        }

        // Then check if the timeout has elapsed
        if let Poll::Ready(()) = Pin::new(&mut this.delay).poll(cx) {
            return Poll::Ready(Err(TimeoutError));
        }

        Poll::Pending
    }
}

/// Error returned when a timeout elapses
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeoutError;

impl std::fmt::Display for TimeoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("operation timed out")
    }
}

impl std::error::Error for TimeoutError {}
