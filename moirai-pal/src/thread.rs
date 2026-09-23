//! Errors from starting the runtime's own helper threads.

use std::{fmt, io};

/// A runtime helper thread (a resolver worker, the connect re-probe) failed to
/// start.
///
/// Converts into an [`io::Error`] of the spawn error's kind, with the spawn
/// error as its [`source`](std::error::Error::source), so it propagates
/// through the I/O paths that start these threads on demand. The next
/// operation that needs the thread retries the spawn.
#[derive(Debug)]
pub struct ThreadStartError {
    thread: &'static str,
    source: io::Error,
}

impl ThreadStartError {
    /// Record that spawning the `thread` helper failed with `source`.
    #[must_use]
    pub fn new(thread: &'static str, source: io::Error) -> Self {
        Self { thread, source }
    }

    /// The helper thread that failed to start.
    #[must_use]
    pub fn thread(&self) -> &'static str {
        self.thread
    }
}

impl fmt::Display for ThreadStartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "could not start the {} thread", self.thread)
    }
}

impl std::error::Error for ThreadStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

impl From<ThreadStartError> for io::Error {
    fn from(error: ThreadStartError) -> Self {
        let kind = error.source.kind();
        Self::new(kind, error)
    }
}
