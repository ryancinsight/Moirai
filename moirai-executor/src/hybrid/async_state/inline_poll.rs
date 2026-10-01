//! Thread-local bound on nested inline polls.

use std::cell::Cell;

const ASYNC_INLINE_POLL_DEPTH_LIMIT: usize = 1;

thread_local! {
    static ASYNC_INLINE_POLL_DEPTH: Cell<usize> = const { Cell::new(0) };
}

pub(super) struct InlinePollDepthGuard {
    previous: usize,
}

impl InlinePollDepthGuard {
    pub(super) fn try_enter() -> Option<Self> {
        ASYNC_INLINE_POLL_DEPTH.with(|depth| {
            let previous = depth.get();
            (previous < ASYNC_INLINE_POLL_DEPTH_LIMIT).then(|| {
                depth.set(previous + 1);
                Self { previous }
            })
        })
    }
}

impl Drop for InlinePollDepthGuard {
    fn drop(&mut self) {
        ASYNC_INLINE_POLL_DEPTH.with(|depth| depth.set(self.previous));
    }
}
