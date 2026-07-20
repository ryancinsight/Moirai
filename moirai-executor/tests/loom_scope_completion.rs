//! Bounded model of the scoped-completion lifetime handshake.
//!
//! A scheduler scope owns its completion counter, mutex, and condition
//! variable on the caller's stack. The final completion must therefore hold
//! the wait lock before publishing a zero count. A waiter that observes zero
//! acquires the same lock before destroying the state, which proves that the
//! completion thread has finished every access to that state.
//!
//! The model explores one final completion racing one waiter. This is the
//! minimal state that exposed the lifetime violation; non-final decrements do
//! not authorize destruction.

#![cfg(loom)]

use loom::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};
use loom::thread;

struct ScopeState {
    pending: AtomicUsize,
    wait_lock: Mutex<()>,
    completion_released_state: AtomicBool,
}

#[test]
fn zero_publication_happens_inside_wait_lock() {
    loom::model(|| {
        let state = Arc::new(ScopeState {
            pending: AtomicUsize::new(1),
            wait_lock: Mutex::new(()),
            completion_released_state: AtomicBool::new(false),
        });

        let completion_state = Arc::clone(&state);
        let completion = thread::spawn(move || {
            let _guard = completion_state.wait_lock.lock().unwrap();
            assert_eq!(
                completion_state.pending.compare_exchange(
                    1,
                    0,
                    Ordering::AcqRel,
                    Ordering::Acquire
                ),
                Ok(1)
            );
            completion_state
                .completion_released_state
                .store(true, Ordering::Release);
        });

        while state.pending.load(Ordering::Acquire) != 0 {
            thread::yield_now();
        }
        let _guard = state.wait_lock.lock().unwrap();
        assert!(
            state.completion_released_state.load(Ordering::Acquire),
            "zero must not authorize destruction before completion releases state"
        );
        drop(_guard);

        completion.join().unwrap();
    });
}
