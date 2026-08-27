use crate::channel::error::Result;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{fence, Ordering};
use std::task::{Context, Poll};

/// Future implementation for zero-cost async receive
pub struct RecvFuture<'a, T> {
    pub(super) receiver: &'a super::recv::HybridReceiver<T>,
    pub(super) id: Option<u64>,
}

impl<T: Send> Future for RecvFuture<'_, T> {
    type Output = Result<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Fast path check
        if let Some(value) = self.receiver.ring.try_consume() {
            return Poll::Ready(Ok(value));
        }

        if self.receiver.closed.load(Ordering::Acquire) {
            return Poll::Ready(Err(crate::channel::error::ChannelError::Closed));
        }

        let this = self.get_mut();
        let mut wakers = this
            .receiver
            .async_wakers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        // Register FIRST, then re-check. The sender's counter gate runs
        // without this lock, so registering after the re-check leaves a plain
        // interleaving hole (no reordering required): re-check sees an empty
        // ring, the sender produces and reads a zero `waker_count`, then the
        // registration lands — nobody ever wakes the task. Registration
        // before the fenced re-check closes it: either the re-check observes
        // the message, or the registration is ordered before the sender's
        // gate load and the sender drains-and-wakes this waker.
        let waker = cx.waker();
        let id = if let Some(id) = this.id {
            if let Some(pos) = wakers.iter().position(|(w_id, _)| *w_id == id) {
                wakers[pos].1.clone_from(waker);
            } else {
                // A previous send drained this future's entry; re-register
                // under the same ID and re-count it.
                wakers.push((id, waker.clone()));
                this.receiver.waker_count.fetch_add(1, Ordering::SeqCst);
            }
            id
        } else {
            let id = this.receiver.next_id.fetch_add(1, Ordering::Relaxed);
            wakers.push((id, waker.clone()));
            this.id = Some(id);
            this.receiver.waker_count.fetch_add(1, Ordering::SeqCst);
            id
        };

        // Dekker fence between the registration above and the re-checks
        // below; pairs with the fence in `notify_consumers` between the
        // sender's publication and its counter gate loads (the same
        // store-buffer pair `moirai-sync`'s `FutexMutex` documents).
        fence(Ordering::SeqCst);

        let ready = if let Some(value) = this.receiver.ring.try_consume() {
            Some(Ok(value))
        } else if this.receiver.closed.load(Ordering::Acquire) {
            Some(Err(crate::channel::error::ChannelError::Closed))
        } else {
            None
        };

        if let Some(output) = ready {
            // Deregister while the lock is still held: a resolved future must
            // not leave a counted waker behind for senders to wake.
            if let Some(pos) = wakers.iter().position(|(w_id, _)| *w_id == id) {
                wakers.remove(pos);
                this.receiver.waker_count.fetch_sub(1, Ordering::SeqCst);
            }
            return Poll::Ready(output);
        }

        Poll::Pending
    }
}

impl<T> Drop for RecvFuture<'_, T> {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            let mut wakers = self
                .receiver
                .async_wakers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(pos) = wakers.iter().position(|(w_id, _)| *w_id == id) {
                wakers.remove(pos);
                self.receiver.waker_count.fetch_sub(1, Ordering::SeqCst);
            }
        }
    }
}
