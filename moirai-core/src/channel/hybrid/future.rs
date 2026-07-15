use crate::channel::error::Result;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::Ordering;
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
        // Lock wakers to register this task
        if let Ok(mut wakers) = this.receiver.async_wakers.lock() {
            // Check again after locking to prevent race condition
            if let Some(value) = this.receiver.ring.try_consume() {
                return Poll::Ready(Ok(value));
            }
            if this.receiver.closed.load(Ordering::Acquire) {
                return Poll::Ready(Err(crate::channel::error::ChannelError::Closed));
            }

            let waker = cx.waker();
            if let Some(id) = this.id {
                if let Some(pos) = wakers.iter().position(|(w_id, _)| *w_id == id) {
                    wakers[pos].1.clone_from(waker);
                } else {
                    wakers.push((id, waker.clone()));
                    this.receiver.waker_count.fetch_add(1, Ordering::Release);
                }
            } else {
                let id = this.receiver.next_id.fetch_add(1, Ordering::Relaxed);
                wakers.push((id, waker.clone()));
                this.id = Some(id);
                this.receiver.waker_count.fetch_add(1, Ordering::Release);
            }
        }

        Poll::Pending
    }
}

impl<T> Drop for RecvFuture<'_, T> {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            if let Ok(mut wakers) = self.receiver.async_wakers.lock() {
                if let Some(pos) = wakers.iter().position(|(w_id, _)| *w_id == id) {
                    wakers.remove(pos);
                    self.receiver.waker_count.fetch_sub(1, Ordering::Release);
                }
            }
        }
    }
}
