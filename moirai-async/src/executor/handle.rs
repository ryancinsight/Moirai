use crate::executor::result_slot::AsyncResultSlot;
use moirai_core::TaskId;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

/// A handle to an async task that can be awaited.
pub struct AsyncHandle<T> {
    pub(super) task_id: TaskId,
    pub(super) result_slot: Arc<AsyncResultSlot<T>>,
}

impl<T> AsyncHandle<T> {
    /// Return the executor-assigned task identifier.
    #[must_use]
    pub fn id(&self) -> TaskId {
        self.task_id
    }
}

impl<T> Future for AsyncHandle<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Some(value) = self.result_slot.try_take_ready() {
            return Poll::Ready(value);
        }

        self.result_slot.register_waker(cx.waker());

        self.result_slot
            .try_take_ready()
            .map_or(Poll::Pending, Poll::Ready)
    }
}
