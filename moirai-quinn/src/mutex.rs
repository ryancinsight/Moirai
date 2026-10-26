use std::{
    fmt::Debug,
    ops::{Deref, DerefMut},
};

#[cfg(feature = "lock_tracking")]
mod tracking {
    use super::*;
    use crate::{Duration, Instant};
    use std::collections::VecDeque;
    use tracing::warn;

    #[derive(Debug)]
    struct Inner<T> {
        last_lock_owner: VecDeque<(&'static str, Duration)>,
        value: T,
    }

    /// A Mutex which optionally allows to track the time a lock was held and
    /// emit warnings in case of excessive lock times
    pub(crate) struct Mutex<T> {
        inner: std::sync::Mutex<Inner<T>>,
    }

    impl<T: Debug> std::fmt::Debug for Mutex<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            std::fmt::Debug::fmt(&self.inner, f)
        }
    }

    impl<T> Mutex<T> {
        pub(crate) fn new(value: T) -> Self {
            Self {
                inner: std::sync::Mutex::new(Inner {
                    last_lock_owner: VecDeque::new(),
                    value,
                }),
            }
        }

        /// Acquires the lock for a certain purpose
        ///
        /// The purpose will be recorded in the list of last lock owners
        fn acquire(&self, purpose: &'static str) -> (MutexGuard<'_, T>, bool) {
            // We don't bother dispatching through Runtime::now because they're pure performance
            // diagnostics.
            let now = Instant::now();
            let (guard, poisoned) = match self.inner.lock() {
                Ok(guard) => (guard, false),
                Err(poisoned) => (poisoned.into_inner(), true),
            };

            let lock_time = Instant::now();
            let elapsed = lock_time.duration_since(now);

            if elapsed > Duration::from_millis(1) {
                warn!(
                    "Locking the connection for {} took {:?}. Last owners: {:?}",
                    purpose, elapsed, guard.last_lock_owner
                );
            }

            (
                MutexGuard {
                    guard,
                    start_time: lock_time,
                    purpose,
                },
                poisoned,
            )
        }

        pub(crate) fn lock(&self, purpose: &'static str) -> MutexGuard<'_, T> {
            let (guard, poisoned) = self.acquire(purpose);
            if poisoned {
                drop(guard);
                panic!("QUIC connection state poisoned; protocol processing cannot resume");
            }
            guard
        }

        /// Recover only for notification and channel cleanup, never protocol processing.
        pub(crate) fn lock_for_teardown(&self, purpose: &'static str) -> (MutexGuard<'_, T>, bool) {
            self.acquire(purpose)
        }
    }

    pub(crate) struct MutexGuard<'a, T> {
        guard: std::sync::MutexGuard<'a, Inner<T>>,
        start_time: Instant,
        purpose: &'static str,
    }

    impl<T> Drop for MutexGuard<'_, T> {
        fn drop(&mut self) {
            if self.guard.last_lock_owner.len() == MAX_LOCK_OWNERS {
                self.guard.last_lock_owner.pop_back();
            }

            let duration = self.start_time.elapsed();

            if duration > Duration::from_millis(1) {
                warn!(
                    "Utilizing the connection for {} took {:?}",
                    self.purpose, duration
                );
            }

            self.guard
                .last_lock_owner
                .push_front((self.purpose, duration));
        }
    }

    impl<T> Deref for MutexGuard<'_, T> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            &self.guard.value
        }
    }

    impl<T> DerefMut for MutexGuard<'_, T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.guard.value
        }
    }

    const MAX_LOCK_OWNERS: usize = 20;
}

#[cfg(feature = "lock_tracking")]
pub(crate) use tracking::Mutex;

#[cfg(not(feature = "lock_tracking"))]
mod non_tracking {
    use super::*;

    /// A Mutex which optionally allows to track the time a lock was held and
    /// emit warnings in case of excessive lock times
    #[derive(Debug)]
    pub(crate) struct Mutex<T> {
        inner: std::sync::Mutex<T>,
    }

    impl<T> Mutex<T> {
        pub(crate) fn new(value: T) -> Self {
            Self {
                inner: std::sync::Mutex::new(value),
            }
        }

        /// Acquires the lock for a certain purpose
        ///
        /// The purpose will be recorded in the list of last lock owners
        pub(crate) fn lock(&self, _purpose: &'static str) -> MutexGuard<'_, T> {
            let (guard, poisoned) = self.lock_for_teardown("lock");
            if poisoned {
                drop(guard);
                panic!("QUIC connection state poisoned; protocol processing cannot resume");
            }
            guard
        }

        /// Recover only for notification and channel cleanup, never protocol processing.
        pub(crate) fn lock_for_teardown(
            &self,
            _purpose: &'static str,
        ) -> (MutexGuard<'_, T>, bool) {
            let (guard, poisoned) = match self.inner.lock() {
                Ok(guard) => (guard, false),
                Err(poisoned) => (poisoned.into_inner(), true),
            };
            (MutexGuard { guard }, poisoned)
        }
    }

    pub(crate) struct MutexGuard<'a, T> {
        guard: std::sync::MutexGuard<'a, T>,
    }

    impl<T> Deref for MutexGuard<'_, T> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            self.guard.deref()
        }
    }

    impl<T> DerefMut for MutexGuard<'_, T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            self.guard.deref_mut()
        }
    }
}

#[cfg(not(feature = "lock_tracking"))]
pub(crate) use non_tracking::Mutex;

#[cfg(test)]
mod tests {
    use super::Mutex;

    #[test]
    fn poisoned_connection_lock_allows_cleanup_but_not_protocol_resume() {
        let state = Mutex::new(41usize);
        let original = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut guard = state.lock("poison");
            *guard += 1;
            panic!("deliberate state panic");
        }))
        .expect_err("the original panic must propagate");
        assert_eq!(
            original.downcast_ref::<&str>(),
            Some(&"deliberate state panic")
        );

        let (guard, poisoned) = state.lock_for_teardown("cleanup");
        assert!(poisoned);
        assert_eq!(*guard, 42);
        drop(guard);
        let resumed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = state.lock("resume");
        }))
        .expect_err("poisoned state must reject protocol resumption");
        assert_eq!(
            resumed.downcast_ref::<&str>(),
            Some(&"QUIC connection state poisoned; protocol processing cannot resume")
        );
    }
}
