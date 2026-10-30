//! Executor identity active on the current call stack.

use std::{cell::Cell, ptr};

use super::admission::Admission;

thread_local! {
    static ACTIVE: Cell<*const Frame> = const { Cell::new(ptr::null()) };
}

struct Frame {
    admission: *const Admission,
    previous: *const Frame,
}

/// Tracks facade and asynchronous work executing for one executor.
pub(in crate::hybrid) struct Context;

impl Context {
    /// Runs `operation` with `admission` present in the thread-local stack.
    ///
    /// Re-entering an executor already present anywhere in the stack does not
    /// add another frame. This keeps nested `A -> B -> A` execution bounded by
    /// the number of distinct active executors.
    #[inline]
    pub(in crate::hybrid) fn with<R>(admission: &Admission, operation: impl FnOnce() -> R) -> R {
        if Self::contains(admission) {
            return operation();
        }

        let frame = Frame {
            admission: ptr::from_ref(admission),
            previous: ACTIVE.with(Cell::get),
        };
        ACTIVE.with(|active| active.set(ptr::from_ref(&frame)));
        let _restore = Restore { frame: &frame };
        operation()
    }

    /// Reports whether `admission` occurs anywhere in the active stack.
    #[inline]
    pub(in crate::hybrid) fn contains(admission: &Admission) -> bool {
        let identity = ptr::from_ref(admission);
        let mut current = ACTIVE.with(Cell::get);
        while !current.is_null() {
            // SAFETY: `with` installs a pointer only after its immutable local
            // frame has a stable address. The operation cannot move or access
            // that frame, and the private restore guard resets the thread-local
            // head before frame destruction on return or unwind. Every link is
            // created and traversed on the same thread; neither the frame nor
            // the guard escapes to code that could forget it.
            let frame = unsafe { &*current };
            if ptr::eq(frame.admission, identity) {
                return true;
            }
            current = frame.previous;
        }
        false
    }

    #[cfg(test)]
    fn depth() -> usize {
        let mut depth = 0usize;
        let mut current = ACTIVE.with(Cell::get);
        while !current.is_null() {
            // SAFETY: the test traverses the same stack under the invariant
            // documented by `contains`.
            current = unsafe { &*current }.previous;
            depth += 1;
        }
        depth
    }
}

struct Restore<'frame> {
    frame: &'frame Frame,
}

impl Drop for Restore<'_> {
    fn drop(&mut self) {
        ACTIVE.with(|active| active.set(self.frame.previous));
    }
}

#[cfg(test)]
mod tests {
    use super::{Admission, Context};

    #[test]
    fn nested_stack_finds_outer_identity_and_deduplicates_reentry() {
        let first = Admission::new();
        let second = Admission::new();
        let unrelated = Admission::new();

        Context::with(&first, || {
            assert!(Context::contains(&first));
            assert!(!Context::contains(&unrelated));
            assert_eq!(Context::depth(), 1);
            Context::with(&second, || {
                assert!(Context::contains(&first));
                assert!(Context::contains(&second));
                assert!(!Context::contains(&unrelated));
                assert_eq!(Context::depth(), 2);
                Context::with(&first, || {
                    assert!(Context::contains(&first));
                    assert!(Context::contains(&second));
                    assert_eq!(Context::depth(), 2);
                });
            });
            assert!(Context::contains(&first));
            assert!(!Context::contains(&second));
        });

        assert!(!Context::contains(&first));
        assert!(!Context::contains(&second));
    }
}
