use super::core::IoReactor;

melinoe::thread_cached! {
    pub(crate) mod active_reactor: *const IoReactor;
}

#[cfg(not(target_os = "windows"))]
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

        #[cfg(not(target_os = "windows"))]
        {
            let reactor = GLOBAL_REACTOR.get_or_init(|| {
                let r = std::sync::Arc::new(
                    IoReactor::new().expect("failed to create global IoReactor"),
                );
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

        #[cfg(target_os = "windows")]
        {
            // No implicit global reactor on Windows. The IOCP backend completes
            // posted *overlapped operations*, not socket *readiness*, so it cannot
            // drive the readiness-based futures in `net.rs`. Returning `None` makes
            // those futures use the cooperative self-wake fallback
            // (`wake_without_active_reactor`), which is correct (used by the net
            // tests) though it busy-polls. A readiness-capable Windows reactor
            // (AFD/`\Device\Afd` polling) is a separate, larger feature.
            None
        }
    }
}
