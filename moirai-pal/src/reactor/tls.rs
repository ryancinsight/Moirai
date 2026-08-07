//! Thread-local reactor ownership and test-only reactor suppression.
//!
//! The shared Melinoe 0.9.0 macro owns the stable `thread_local!` initializer;
//! this module keeps the scoped allowance until Moirai advances that provider
//! pin to the const-initializer revision.
#![allow(
    clippy::missing_const_for_thread_local,
    reason = "Melinoe 0.9.0's pinned thread_cached! expansion owns this initializer"
)]

use super::core::IoReactor;

melinoe::thread_cached! {
    pub(crate) mod active_reactor: *const IoReactor;
}

pub(crate) static GLOBAL_REACTOR: std::sync::OnceLock<Option<std::sync::Arc<IoReactor>>> =
    std::sync::OnceLock::new();

#[cfg(test)]
thread_local! {
    /// Test-only switch that suppresses the lazily-started global reactor for
    /// the current thread, so `get_active` returns `None` and socket operations
    /// take the cooperative busy-poll self-wake fallback in `net.rs`.
    static FORCE_NO_REACTOR: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl IoReactor {
    /// Run a closure with this reactor set as the thread-local active reactor.
    pub fn with_active<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        // Restore the previous thread-local reactor on scope exit via RAII. If
        // `f` panics, a manual restore would be skipped, leaving a dangling
        // `self` pointer in the thread-local that a later `get_active()` would
        // dereference (use-after-free once `self` is dropped during unwinding).
        struct Restore(Option<*const IoReactor>);
        impl Drop for Restore {
            fn drop(&mut self) {
                match self.0 {
                    Some(old) => active_reactor::set(old),
                    None => active_reactor::clear(),
                }
            }
        }

        let _restore = Restore(active_reactor::get());
        active_reactor::set(self as *const IoReactor);
        f()
    }

    /// Retrieve the active reactor for the current thread, if any.
    ///
    /// Returns the thread-local reactor when one is installed via
    /// [`with_active`](Self::with_active); otherwise lazily starts a
    /// process-global readiness reactor (epoll/kqueue/`WSAPoll`) on its own
    /// thread. If that reactor cannot be created or its driver thread cannot be
    /// spawned, this caches and returns `None` so socket operations degrade to
    /// the cooperative busy-poll self-wake fallback in `net.rs` rather than
    /// panicking — readiness still makes progress, just without an event loop.
    pub fn get_active() -> Option<&'static IoReactor> {
        if let Some(ptr) = active_reactor::get() {
            return Some(unsafe { &*ptr });
        }

        #[cfg(test)]
        if FORCE_NO_REACTOR.with(std::cell::Cell::get) {
            return None;
        }

        GLOBAL_REACTOR
            .get_or_init(|| {
                let reactor = std::sync::Arc::new(IoReactor::new().ok()?);
                let driver = std::sync::Arc::clone(&reactor);
                std::thread::Builder::new()
                    .name("moirai-global-reactor".to_string())
                    .spawn(move || {
                        let _ = driver.run();
                    })
                    .ok()?;
                Some(reactor)
            })
            .as_deref()
    }

    /// Test-only: run `f` with the global reactor suppressed for this thread, so
    /// [`get_active`](Self::get_active) returns `None` and socket operations
    /// exercise the `net.rs` busy-poll self-wake fallback deterministically.
    #[cfg(test)]
    pub(crate) fn with_reactor_disabled<F, R>(f: F) -> R
    where
        F: FnOnce() -> R,
    {
        struct Restore(bool);
        impl Drop for Restore {
            fn drop(&mut self) {
                FORCE_NO_REACTOR.with(|cell| cell.set(self.0));
            }
        }

        let _restore = Restore(FORCE_NO_REACTOR.with(|cell| cell.replace(true)));
        f()
    }
}
