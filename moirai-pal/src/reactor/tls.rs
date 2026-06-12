use std::cell::RefCell;

use super::core::IoReactor;

#[cfg(nightly_tls_active)]
#[thread_local]
pub(crate) static mut ACTIVE_REACTOR_NIGHTLY: Option<*const IoReactor> = None;

#[cfg(not(nightly_tls_active))]
thread_local! {
    pub(crate) static ACTIVE_REACTOR: RefCell<Option<*const IoReactor>> = const { RefCell::new(None) };
}

#[cfg(not(target_os = "windows"))]
pub(crate) static GLOBAL_REACTOR: std::sync::OnceLock<std::sync::Arc<IoReactor>> = std::sync::OnceLock::new();


impl IoReactor {
    /// Run a closure with this reactor set as the thread-local active reactor.
    pub fn with_active<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        #[cfg(nightly_tls_active)]
        unsafe {
            let old = ACTIVE_REACTOR_NIGHTLY;
            ACTIVE_REACTOR_NIGHTLY = Some(self as *const IoReactor);
            let result = f();
            ACTIVE_REACTOR_NIGHTLY = old;
            result
        }
        #[cfg(not(nightly_tls_active))]
        {
            let old = ACTIVE_REACTOR.with(|cell| cell.replace(Some(self as *const IoReactor)));
            let result = f();
            ACTIVE_REACTOR.with(|cell| cell.replace(old));
            result
        }
    }

    /// Retrieve the active reactor for the current thread, if any.
    pub fn get_active() -> Option<&'static IoReactor> {
        #[cfg(nightly_tls_active)]
        let maybe_ptr = unsafe { ACTIVE_REACTOR_NIGHTLY };
        #[cfg(not(nightly_tls_active))]
        let maybe_ptr = ACTIVE_REACTOR.with(|cell| *cell.borrow());

        if let Some(ptr) = maybe_ptr {
            return Some(unsafe { &*ptr });
        }

        #[cfg(not(target_os = "windows"))]
        {
            let reactor = GLOBAL_REACTOR.get_or_init(|| {
                let r = std::sync::Arc::new(IoReactor::new().expect("failed to create global IoReactor"));
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
