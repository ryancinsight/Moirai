//! Loom model of compute admission racing scheduler shutdown.
//!
//! Run with:
//! `RUSTFLAGS="--cfg loom" cargo test -p moirai-executor --test loom_shutdown_admission --release`

#![cfg(loom)]

use loom::sync::atomic::{fence, AtomicBool, AtomicUsize, Ordering};
use loom::sync::Arc;
use loom::thread;

struct Admission {
    shutdown: AtomicBool,
    pending: AtomicUsize,
}

impl Admission {
    fn new() -> Self {
        Self {
            shutdown: AtomicBool::new(false),
            pending: AtomicUsize::new(0),
        }
    }

    fn admit_or_rollback(&self) -> bool {
        self.pending.fetch_add(1, Ordering::SeqCst);
        // Loom needs the StoreLoad barrier made explicit; production's SeqCst
        // operations provide this ordering directly.
        fence(Ordering::SeqCst);
        if self.shutdown.load(Ordering::SeqCst) {
            self.pending.fetch_sub(1, Ordering::SeqCst);
            false
        } else {
            true
        }
    }

    fn publish_shutdown_and_would_exit(&self) -> bool {
        self.shutdown.store(true, Ordering::SeqCst);
        // Mirrors the producer-side StoreLoad barrier above.
        fence(Ordering::SeqCst);
        self.pending.load(Ordering::SeqCst) == 0
    }
}

#[test]
fn shutdown_cannot_exit_while_compute_admission_succeeds() {
    loom::model(|| {
        let admission = Arc::new(Admission::new());
        let producer = {
            let admission = Arc::clone(&admission);
            thread::spawn(move || admission.admit_or_rollback())
        };

        let worker_would_exit = admission.publish_shutdown_and_would_exit();
        let admitted = producer.join().unwrap();
        assert!(
            !(worker_would_exit && admitted),
            "shutdown observed no work while a producer admitted work"
        );
    });
}
