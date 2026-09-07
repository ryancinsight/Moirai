//! Browser-local future cancellation state.

use std::cell::{Cell, RefCell};
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll, Waker};

struct LocalTaskState {
    cancelled: Cell<bool>,
    waker: RefCell<Option<Waker>>,
}

impl LocalTaskState {
    fn cancel(&self) {
        if self.cancelled.replace(true) {
            return;
        }
        if let Some(waker) = self.waker.borrow_mut().take() {
            waker.wake();
        }
    }
}

/// Owns cancellation for one future scheduled on the browser event loop.
///
/// The handle is single-owner. Calling [`Self::cancel`] or dropping the handle
/// wakes the task, which then drops its child future and any PAL resources it
/// owns. A task that has already completed is unaffected.
#[must_use = "retain the handle to cancel the browser task"]
pub struct LocalTaskHandle {
    state: Rc<LocalTaskState>,
}

impl LocalTaskHandle {
    /// Requests cancellation of the task.
    pub fn cancel(&self) {
        self.state.cancel();
    }

    /// Returns whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.state.cancelled.get()
    }
}

impl Drop for LocalTaskHandle {
    fn drop(&mut self) {
        self.state.cancel();
    }
}

pub(crate) struct CancellableFuture<F> {
    future: Option<Pin<Box<F>>>,
    state: Rc<LocalTaskState>,
}

pub(crate) fn cancellable<F>(future: F) -> (LocalTaskHandle, CancellableFuture<F>)
where
    F: Future<Output = ()> + 'static,
{
    let state = Rc::new(LocalTaskState {
        cancelled: Cell::new(false),
        waker: RefCell::new(None),
    });
    let handle = LocalTaskHandle {
        state: Rc::clone(&state),
    };
    let future = CancellableFuture {
        future: Some(Box::pin(future)),
        state,
    };
    (handle, future)
}

impl<F: Future<Output = ()>> Future for CancellableFuture<F> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: the wrapper is pinned by the executor, and this projection
        // never moves `future`; it only accesses the already pinned box.
        let this = unsafe { self.get_unchecked_mut() };
        if this.state.cancelled.get() {
            this.future.take();
            this.state.waker.borrow_mut().take();
            return Poll::Ready(());
        }

        this.state.waker.borrow_mut().replace(cx.waker().clone());
        let Some(future) = this.future.as_mut() else {
            this.state.waker.borrow_mut().take();
            return Poll::Ready(());
        };
        let result = future.as_mut().poll(cx);
        if result.is_ready() {
            this.future.take();
            this.state.waker.borrow_mut().take();
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::task::{Wake, Waker};

    struct PendingFuture {
        polls: Rc<Cell<usize>>,
        dropped: Rc<Cell<bool>>,
    }

    impl Future for PendingFuture {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            let this = self.get_mut();
            this.polls.set(this.polls.get() + 1);
            Poll::Pending
        }
    }

    impl Drop for PendingFuture {
        fn drop(&mut self) {
            self.dropped.set(true);
        }
    }

    #[derive(Default)]
    struct WakeCounter(AtomicUsize);

    impl Wake for WakeCounter {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn cancellation_before_poll_drops_the_child_without_polling_it() {
        let polls = Rc::new(Cell::new(0));
        let dropped = Rc::new(Cell::new(false));
        let (handle, mut future) = cancellable(PendingFuture {
            polls: Rc::clone(&polls),
            dropped: Rc::clone(&dropped),
        });

        handle.cancel();
        assert!(handle.is_cancelled());
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        assert!(Pin::new(&mut future).poll(&mut context).is_ready());
        assert_eq!(polls.get(), 0);
        assert!(dropped.get());
    }

    #[test]
    fn cancellation_wakes_a_pending_task_and_drops_the_child() {
        let dropped = Rc::new(Cell::new(false));
        let (handle, mut future) = cancellable(PendingFuture {
            polls: Rc::new(Cell::new(0)),
            dropped: Rc::clone(&dropped),
        });
        let signal = Arc::new(WakeCounter::default());
        let waker = Waker::from(Arc::clone(&signal));
        let mut context = Context::from_waker(&waker);

        assert!(Pin::new(&mut future).poll(&mut context).is_pending());
        handle.cancel();
        assert_eq!(signal.0.load(Ordering::Relaxed), 1);
        assert!(Pin::new(&mut future).poll(&mut context).is_ready());
        assert!(dropped.get());
    }

    #[test]
    fn dropping_the_handle_requests_cancellation() {
        let dropped = Rc::new(Cell::new(false));
        let (handle, mut future) = cancellable(PendingFuture {
            polls: Rc::new(Cell::new(0)),
            dropped: Rc::clone(&dropped),
        });
        drop(handle);

        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        assert!(Pin::new(&mut future).poll(&mut context).is_ready());
        assert!(dropped.get());
    }

    #[test]
    fn completed_task_can_be_cancelled_without_repolling() {
        let (handle, mut future) = cancellable(std::future::ready(()));
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        assert!(Pin::new(&mut future).poll(&mut context).is_ready());
        assert!(!handle.is_cancelled());
        handle.cancel();
        assert!(handle.is_cancelled());
    }
}
