#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use super::core::IoReactor;
use std::sync::atomic::Ordering;

#[test]
fn test_reactor_creation() {
    let reactor = IoReactor::new();
    assert!(reactor.is_ok());
}

#[test]
fn test_reactor_metrics() {
    let reactor = IoReactor::new().unwrap();
    let metrics = reactor.metrics();
    assert_eq!(metrics.events_processed.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.peak_fd_count.load(Ordering::Relaxed), 0);
}

#[test]
fn with_active_restores_thread_local_on_panic() {
    // Regression: if `f` panics, `with_active` must still restore the previous
    // thread-local reactor (via RAII), not leave a dangling pointer to the inner
    // reactor that a later `get_active()` would dereference (use-after-free).
    let outer = IoReactor::new().expect("outer reactor");
    let inner = IoReactor::new().expect("inner reactor");

    outer.with_active(|| {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            inner.with_active(|| panic!("boom"));
        }));
        assert!(result.is_err(), "inner closure must have panicked");

        // The active reactor must be restored to `outer`, never left as `inner`.
        let active = IoReactor::get_active().expect("outer is still active");
        assert!(
            std::ptr::eq(active, &outer),
            "thread-local must be restored to the outer reactor after panic"
        );
    });
}
