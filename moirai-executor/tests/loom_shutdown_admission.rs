//! Loom models of scheduler shutdown admission and join completion.
//!
//! Run with:
//! `RUSTFLAGS="--cfg loom" cargo nextest run -p moirai-executor --test loom_shutdown_admission --release`

#![cfg(loom)]

use loom::sync::atomic::{fence, AtomicBool, AtomicU8, AtomicUsize, Ordering};
use loom::sync::{Arc, Condvar, Mutex};
use loom::thread;

const JOIN_IN_PROGRESS: u8 = 1;
const JOIN_COMPLETE: u8 = 2;

struct Admission {
    shutdown: AtomicBool,
    pending: AtomicUsize,
}

struct JoinCompletion {
    state: AtomicU8,
    wait_lock: Mutex<()>,
    wait_signal: Condvar,
}

impl JoinCompletion {
    fn new() -> Self {
        Self {
            state: AtomicU8::new(JOIN_IN_PROGRESS),
            wait_lock: Mutex::new(()),
            wait_signal: Condvar::new(),
        }
    }

    fn publish(&self) {
        {
            let _guard = self.wait_lock.lock().unwrap();
            self.state.store(JOIN_COMPLETE, Ordering::Release);
        }
        self.wait_signal.notify_all();
    }

    fn wait(&self) {
        let mut guard = self.wait_lock.lock().unwrap();
        while self.state.load(Ordering::Acquire) != JOIN_COMPLETE {
            guard = self.wait_signal.wait(guard).unwrap();
        }
    }
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

#[test]
fn join_completion_notification_cannot_pass_external_waiter() {
    loom::model(|| {
        let completion = Arc::new(JoinCompletion::new());
        let waiter = {
            let completion = Arc::clone(&completion);
            thread::spawn(move || completion.wait())
        };
        let publisher = thread::spawn(move || completion.publish());

        waiter.join().unwrap();
        publisher.join().unwrap();
    });
}
