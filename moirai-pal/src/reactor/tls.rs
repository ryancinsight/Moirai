use super::core::IoReactor;

melinoe::thread_cached! {
    pub(crate) mod active_reactor: *const IoReactor;
}

pub(crate) static GLOBAL_REACTOR: std::sync::OnceLock<std::sync::Arc<IoReactor>> =
    std::sync::OnceLock::new();

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
    pub fn get_active() -> Option<&'static IoReactor> {
        let maybe_ptr = active_reactor::get();

        if let Some(ptr) = maybe_ptr {
            return Some(unsafe { &*ptr });
        }

        // Lazily start a process-global reactor on its own thread. The reactor is
        // readiness-based on every platform: epoll (Linux), kqueue (BSD/macOS), and
        // `WSAPoll` (Windows). Sockets registering a waker are driven by this
        // reactor instead of the cooperative busy-poll fallback in `net.rs`.
        let reactor = GLOBAL_REACTOR.get_or_init(|| {
            let r =
                std::sync::Arc::new(IoReactor::new().expect("failed to create global IoReactor"));
            let r_clone = std::sync::Arc::clone(&r);
            std::thread::Builder::new()
                .name("moirai-global-reactor".to_string())
                .spawn(move || {
                    let _ = r_clone.run();
                })
                .expect("failed to spawn global reactor thread");
            r
        });
        Some(&**reactor)
    }
}
