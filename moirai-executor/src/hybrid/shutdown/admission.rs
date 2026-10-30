// The model-checking build substitutes Loom's atomic cell while retaining the
// production state machine and memory orderings below.
#[cfg(loom)]
use loom::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{sync::OnceLock, thread::Thread};

use moirai_core::error::{ExecutorError, ExecutorResult};

use super::Context;

const CLOSED: usize = 1usize << (usize::BITS - 1);
const ACTIVE_MASK: usize = CLOSED - 1;

/// Linearizes external executor calls against shutdown.
pub(in crate::hybrid) struct Admission {
    state: AtomicUsize,
    worker: OnceLock<Thread>,
}
impl Admission {
    pub(in crate::hybrid) fn new() -> Self {
        Self {
            state: AtomicUsize::new(0),
            worker: OnceLock::new(),
        }
    }

    /// Runs one facade call while its input and cleanup state remain counted.
    ///
    /// The closure owns the call arguments. Its captures therefore drop before
    /// the permit on return or unwind, including scheduler rejection cleanup.
    #[cfg(test)]
    pub(in crate::hybrid) fn run<T>(
        &self,
        operation: impl FnOnce() -> ExecutorResult<T>,
    ) -> ExecutorResult<T> {
        self.admit(operation)?.run(|operation| operation())
    }

    pub(in crate::hybrid) fn admit<T>(&self, input: T) -> ExecutorResult<Call<'_, T>> {
        let mut state = self.state.load(Ordering::Acquire);
        loop {
            if state & CLOSED != 0 {
                return Err(ExecutorError::ShuttingDown);
            }
            let active = state & ACTIVE_MASK;
            let Some(next) = active.checked_add(1).filter(|next| *next <= ACTIVE_MASK) else {
                return Err(ExecutorError::ResourceExhausted(
                    "executor facade call count reached its representation limit".into(),
                ));
            };
            match self
                .state
                .compare_exchange_weak(state, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    return Ok(Call {
                        input: Some(input),
                        permit: Permit { admission: self },
                    });
                }
                Err(observed) => state = observed,
            }
        }
    }

    pub(in crate::hybrid) fn close(&self) {
        self.state.fetch_or(CLOSED, Ordering::AcqRel);
        self.wake_worker();
    }

    pub(in crate::hybrid) fn is_closed(&self) -> bool {
        self.state.load(Ordering::Acquire) & CLOSED != 0
    }

    pub(in crate::hybrid) fn active_calls(&self) -> usize {
        self.state.load(Ordering::Acquire) & ACTIVE_MASK
    }

    pub(in crate::hybrid) fn install_worker(&self, worker: Thread) {
        assert!(
            self.worker.set(worker).is_ok(),
            "invariant: shutdown worker is installed once"
        );
    }

    pub(in crate::hybrid) fn wake_worker(&self) {
        if let Some(worker) = self.worker.get() {
            worker.unpark();
        }
    }

    pub(in crate::hybrid) fn is_shutdown_worker(&self) -> bool {
        self.worker
            .get()
            .is_some_and(|worker| worker.id() == std::thread::current().id())
    }

    #[cfg(test)]
    pub(in crate::hybrid) fn at_capacity() -> Self {
        Self {
            state: AtomicUsize::new(ACTIVE_MASK),
            worker: OnceLock::new(),
        }
    }
}
pub(in crate::hybrid) struct Call<'admission, T> {
    // Rust drops fields in declaration order and continues with later fields
    // while unwinding a preceding field destructor. Input cleanup therefore
    // completes, or unwinds through, before the permit releases the count.
    input: Option<T>,
    permit: Permit<'admission>,
}

impl<T> Call<'_, T> {
    pub(in crate::hybrid) fn run<R>(mut self, operation: impl FnOnce(T) -> R) -> R {
        debug_assert_ne!(
            self.permit.admission.state.load(Ordering::Relaxed) & ACTIVE_MASK,
            0,
            "invariant: an admitted call retains one active permit"
        );
        Context::with(self.permit.admission, || {
            let input = self
                .input
                .take()
                .expect("invariant: admitted input is taken once");
            operation(input)
        })
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        sync::atomic::{AtomicUsize, Ordering},
    };

    use super::{Admission, Context};

    struct ContextDrop<'a> {
        admission: &'a Admission,
        drops: &'a AtomicUsize,
    }

    impl Drop for ContextDrop<'_> {
        fn drop(&mut self) {
            assert!(Context::contains(self.admission));
            assert_eq!(self.drops.fetch_add(1, Ordering::AcqRel), 0);
        }
    }

    #[test]
    fn panic_drops_input_in_context_then_restores_and_releases() {
        let admission = Admission::new();
        let drops = AtomicUsize::new(0);
        let call = admission
            .admit(ContextDrop {
                admission: &admission,
                drops: &drops,
            })
            .expect("open admission accepts input");

        let unwind = catch_unwind(AssertUnwindSafe(|| {
            call.run(|_input| panic!("operation panic"));
        }));

        assert_eq!(
            unwind
                .expect_err("operation panic propagates")
                .downcast_ref::<&str>(),
            Some(&"operation panic")
        );
        assert_eq!(drops.load(Ordering::Acquire), 1);
        assert_eq!(admission.active_calls(), 0);
        assert!(!Context::contains(&admission));
    }
}

struct Permit<'admission> {
    admission: &'admission Admission,
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        let previous = self.admission.state.fetch_sub(1, Ordering::Release);
        if previous == CLOSED | 1 {
            self.admission.wake_worker();
        }
    }
}

#[cfg(all(test, loom))]
mod loom_tests {
    use super::Admission;
    use loom::{sync::Arc, thread};
    use moirai_core::error::ExecutorError;

    #[test]
    fn production_admission_linearizes_close_and_release() {
        loom::model(|| {
            let admission = Arc::new(Admission::new());
            let contender = Arc::clone(&admission);
            let worker = thread::spawn(move || match contender.admit(()) {
                Ok(call) => {
                    assert_eq!(contender.active_calls(), 1);
                    drop(call);
                }
                Err(error) => assert!(matches!(error, ExecutorError::ShuttingDown)),
            });
            admission.close();
            worker.join().unwrap();
            assert!(admission.is_closed());
            assert_eq!(admission.active_calls(), 0);
        });
    }
}
