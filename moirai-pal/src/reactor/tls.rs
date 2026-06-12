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
        let old = active_reactor::get();
        active_reactor::set(self as *const IoReactor);
        let result = f();
        if let Some(old_ptr) = old {
            active_reactor::set(old_ptr);
        } else {
            active_reactor::clear();
        }
        result
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
            None
        }
    }
}
