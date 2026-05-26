//! Waker support for futures driven by scheduler workers.

use std::{
    future::Future,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
    thread::{self, Thread},
};

struct ThreadWaker {
    thread: Thread,
}

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.thread.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.thread.unpark();
    }
}

/// Run a future to completion on the current thread.
///
/// The waker unparks the current thread, so pending futures suspend the worker
/// without a polling sleep loop.
pub fn block_on_current_thread<F>(future: F) -> F::Output
where
    F: Future,
{
    let waker = Waker::from(Arc::new(ThreadWaker {
        thread: thread::current(),
    }));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);

    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => return output,
            Poll::Pending => thread::park(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::block_on_current_thread;

    #[test]
    fn block_on_resolves_ready_future() {
        let value = block_on_current_thread(async { 42 });
        assert_eq!(value, 42);
    }
}
