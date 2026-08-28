//! Platform-agnostic async file-system operations.

mod file;
mod path;

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

pub use file::{AsyncFile, FileOpenOptions};
pub use path::{
    append, copy, create_dir, create_dir_all, metadata, remove_dir, remove_dir_all, remove_file,
    rename, write,
};

/// Future that yields to the executor exactly once, then resolves.
pub struct YieldFuture {
    yielded: bool,
}

impl Future for YieldFuture {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.yielded {
            Poll::Ready(())
        } else {
            self.yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

/// Yield to the executor once before resuming (cooperative scheduling point).
pub fn yield_now() -> YieldFuture {
    YieldFuture { yielded: false }
}

#[cfg(test)]
mod tests;
