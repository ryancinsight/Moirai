//! Loom models of scheduler shutdown admission and join completion.
//!
//! Run with:
//! `RUSTFLAGS="--cfg loom" cargo nextest run -p moirai-executor --test loom_shutdown_admission --release`

#![cfg(loom)]

use loom::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering, fence};
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

struct ExternalHandles {
    remaining: AtomicUsize,
    shutdown_elections: AtomicUsize,
}

impl ExternalHandles {
    fn new(count: usize) -> Self {
        Self {
            remaining: AtomicUsize::new(count),
            shutdown_elections: AtomicUsize::new(0),
        }
    }

    fn drop_one(&self) {
        let previous = self.remaining.fetch_sub(1, Ordering::AcqRel);
        assert!(previous > 0, "external handle count must not underflow");
        if previous == 1 {
            self.shutdown_elections.fetch_add(1, Ordering::Relaxed);
        }
    }
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
            let _guard = self
                .wait_lock
                .lock()
                .expect("loom join-completion mutex must not be poisoned");
            self.state.store(JOIN_COMPLETE, Ordering::Release);
        }
        self.wait_signal.notify_all();
    }

    fn wait(&self) {
        let mut guard = self
            .wait_lock
            .lock()
            .expect("loom external-waiter mutex must not be poisoned");
        while self.state.load(Ordering::Acquire) != JOIN_COMPLETE {
            guard = self
                .wait_signal
                .wait(guard)
                .expect("loom external-waiter mutex must not be poisoned");
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
        let admitted = producer
            .join()
            .expect("loom admission producer must not panic");
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

        waiter.join().expect("loom external waiter must not panic");
        publisher
            .join()
            .expect("loom completion publisher must not panic");
    });
}

#[test]
fn final_external_handle_elects_one_shutdown_owner() {
    loom::model(|| {
        let handles = Arc::new(ExternalHandles::new(2));
        let first = {
            let handles = Arc::clone(&handles);
            thread::spawn(move || handles.drop_one())
        };
        let second = {
            let handles = Arc::clone(&handles);
            thread::spawn(move || handles.drop_one())
        };

        first
            .join()
            .expect("first loom external owner must not panic");
        second
            .join()
            .expect("second loom external owner must not panic");
        assert_eq!(handles.remaining.load(Ordering::Acquire), 0);
        assert_eq!(handles.shutdown_elections.load(Ordering::Acquire), 1);
    });
}

#[test]
fn blocking_close_cannot_pass_the_last_empty_check() {
    loom::model(|| {
        let closed = Arc::new(AtomicBool::new(false));
        let worker_closed = Arc::clone(&closed);
        let worker = thread::spawn(move || {
            loop {
                if worker_closed.load(Ordering::Acquire) {
                    return;
                }
                thread::park();
            }
        });

        // Production obtains this handle from `JoinHandle` before publishing
        // the lane, so every possible closer owns a valid unpark target.
        let worker_thread = worker.thread().clone();
        let closer = thread::spawn(move || {
            closed.store(true, Ordering::Release);
            worker_thread.unpark();
        });

        closer.join().expect("loom blocking closer must not panic");
        worker.join().expect("loom blocking worker must not panic");
    });
}

#[test]
fn lazy_lane_publication_and_close_share_shutdown_rmw() {
    loom::model(|| {
        let shutdown = Arc::new(AtomicBool::new(false));
        let published = Arc::new(AtomicBool::new(false));
        let closed = Arc::new(AtomicBool::new(false));

        let initializer = {
            let shutdown = Arc::clone(&shutdown);
            let published = Arc::clone(&published);
            let closed = Arc::clone(&closed);
            thread::spawn(move || {
                // Models `OnceLock::set` followed by the production RMW.
                published.store(true, Ordering::Release);
                if shutdown.fetch_or(false, Ordering::SeqCst) {
                    closed.store(true, Ordering::Relaxed);
                }
            })
        };
        let closer_shutdown = Arc::clone(&shutdown);
        let closer_published = Arc::clone(&published);
        let closer_closed = Arc::clone(&closed);
        let closer = thread::spawn(move || {
            let previous = closer_shutdown.swap(true, Ordering::SeqCst);
            assert!(!previous, "the model has one shutdown publisher");
            if closer_published.load(Ordering::Acquire) {
                closer_closed.store(true, Ordering::Relaxed);
            }
        });
        initializer.join().expect("loom initializer must not panic");
        closer.join().expect("loom closer must not panic");
        assert!(
            closed.load(Ordering::Relaxed),
            "the published lane must close"
        );
    });
}
