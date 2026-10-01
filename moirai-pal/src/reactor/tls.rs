//! Thread-local reactor ownership and test-only reactor suppression.
//!
//! The shared Melinoe `thread_cached!` macro owns the `thread_local!`
//! initializer, which it expands as a `const` block on stable.

use super::core::IoReactor;

melinoe::thread_cached! {
    pub(crate) mod active_reactor: *const IoReactor;
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) static GLOBAL_REACTOR: std::sync::OnceLock<Option<std::sync::Arc<IoReactor>>> =
    std::sync::OnceLock::new();

#[cfg(test)]
thread_local! {
    /// Test-only switch that suppresses the lazily-started global reactor for
    /// the current thread, so `with_current` passes `None` and socket operations
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
        // `self` pointer in the thread-local that a later `with_current` would
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

    /// Run `f` with the reactor active on the current thread, if any.
    ///
    /// The reactor is the thread-local one installed by
    /// [`with_active`](Self::with_active); otherwise the call lazily starts a
    /// process-global readiness reactor (epoll/kqueue/`WSAPoll`) on its own
    /// thread. If that reactor cannot be created or its driver thread cannot be
    /// spawned, this caches the failure and passes `None`, so socket operations
    /// degrade to the cooperative busy-poll self-wake fallback in `net.rs`
    /// rather than panicking — readiness still makes progress, just without an
    /// event loop.
    ///
    /// The reference is scoped to the call because a `with_active` reactor is
    /// only borrowed for its own scope; letting it escape would outlive that
    /// borrow:
    ///
    /// ```compile_fail
    /// use moirai_pal::reactor::IoReactor;
    ///
    /// let reactor = IoReactor::new().unwrap();
    /// let leaked = reactor.with_active(|| IoReactor::with_current(|active| active.unwrap()));
    /// drop(reactor);
    /// leaked.wake();
    /// ```
    pub fn with_current<R>(f: impl FnOnce(Option<&IoReactor>) -> R) -> R {
        if let Some(ptr) = active_reactor::get() {
            // SAFETY: the pointer was installed by `with_active` on this thread
            // from a reactor its `&self` borrow keeps alive for the whole scope,
            // and the RAII restore clears the slot before that scope ends. `f`
            // cannot return the reference (it is higher-ranked over the borrow),
            // so no use outlives this call.
            return f(Some(unsafe { &*ptr }));
        }
        f(Self::global())
    }

    /// The process-global reactor, started on first use, or `None` when it
    /// cannot run or this thread suppresses it.
    fn global() -> Option<&'static IoReactor> {
        #[cfg(test)]
        if FORCE_NO_REACTOR.with(std::cell::Cell::get) {
            return None;
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Browser callbacks already run on the event-loop thread. A
            // background Rust thread cannot own Web API handles, so callers
            // install a reactor with `with_active` when they need readiness
            // registration; otherwise the socket layer uses its cooperative
            // self-wake path.
            None
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            GLOBAL_REACTOR
                .get_or_init(|| {
                    let reactor = std::sync::Arc::new(IoReactor::new().ok()?);
                    let driver = std::sync::Arc::clone(&reactor);
                    std::thread::Builder::new()
                        .name("moirai-global-reactor".to_string())
                        .spawn(move || driver.run())
                        .ok()?;
                    Some(reactor)
                })
                .as_deref()
        }
    }

    /// Test-only: run `f` with the global reactor suppressed for this thread, so
    /// [`with_current`](Self::with_current) passes `None` and socket operations
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
