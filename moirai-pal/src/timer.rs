//! Platform-agnostic timer and timeout operations.

use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

/// Platform-agnostic one-shot timer future.
///
/// Dropping a pending timer cancels its sleeper and releases its registered
/// waker.
pub struct Timer {
    deadline: Instant,
    state: Arc<TimerState>,
}

impl Timer {
    /// Create a timer that completes after `duration`.
    #[must_use]
    pub fn new(duration: Duration) -> Self {
        // Clamp absurd durations and use `checked_add` so a near-`Duration::MAX`
        // input cannot panic the deadline computation (a library must not panic
        // on input-dependent paths). The clamp caps the wait at ~100 years —
        // effectively "never" — which `checked_add` then resolves without
        // overflowing `Instant`.
        const MAX_TIMER: Duration = Duration::from_secs(100 * 365 * 24 * 60 * 60);
        let now = Instant::now();
        // `unwrap_or(now)` is a safe (non-panicking) degenerate fallback; it is
        // unreachable on any real platform, where `Instant` has decades of
        // headroom above `now`.
        let deadline = now.checked_add(duration.min(MAX_TIMER)).unwrap_or(now);
        Self {
            deadline,
            state: Arc::new(TimerState::new()),
        }
    }

    /// Return the absolute completion deadline.
    #[must_use]
    pub fn deadline(&self) -> Instant {
        self.deadline
    }
}

struct TimerState {
    cancelled: AtomicBool,
    completed: AtomicBool,
    sleeper_started: AtomicBool,
    sleeper_wait: Condvar,
    sleeper_wait_lock: Mutex<()>,
    #[cfg(test)]
    sleeper_events: std::sync::OnceLock<std::sync::mpsc::SyncSender<SleeperEvent>>,
    waker: Mutex<Option<std::task::Waker>>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SleeperEvent {
    Waiting,
    Exited,
}

impl TimerState {
    fn new() -> Self {
        Self {
            cancelled: AtomicBool::new(false),
            completed: AtomicBool::new(false),
            sleeper_started: AtomicBool::new(false),
            sleeper_wait: Condvar::new(),
            sleeper_wait_lock: Mutex::new(()),
            #[cfg(test)]
            sleeper_events: std::sync::OnceLock::new(),
            waker: Mutex::new(None),
        }
    }

    #[cfg(test)]
    fn observe_sleeper_with(
        &self,
        observer: std::sync::mpsc::SyncSender<SleeperEvent>,
    ) -> Result<(), std::sync::mpsc::SyncSender<SleeperEvent>> {
        self.sleeper_events.set(observer)
    }

    #[cfg(test)]
    fn publish_sleeper_event(&self, event: SleeperEvent) {
        let Some(observer) = self.sleeper_events.get() else {
            return;
        };
        // The worker can publish while holding `sleeper_wait_lock`, so the
        // observer must never block it. Disconnect means test teardown; saturation
        // fails the test rather than silently losing the observed transition.
        if let Err(undelivered) = observer.try_send(event) {
            assert!(
                matches!(undelivered, std::sync::mpsc::TrySendError::Disconnected(_)),
                "sleeper observer must consume its bounded event stream"
            );
        }
    }

    fn register_waker(&self, waker: &std::task::Waker) {
        let replacement = waker.clone();
        let removed = {
            let mut stored = self.waker.lock().unwrap_or_else(|error| error.into_inner());
            if stored
                .as_ref()
                .is_none_or(|current| !current.will_wake(waker))
            {
                stored.replace(replacement)
            } else {
                Some(replacement)
            }
        };
        drop(removed);
    }

    fn complete(&self) {
        let wake = if !self.cancelled.load(Ordering::Acquire)
            && !self.completed.swap(true, Ordering::AcqRel)
        {
            self.waker
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .take()
        } else {
            None
        };
        if let Some(wake) = wake {
            wake.wake();
        }
    }

    fn cancel(&self) {
        let guard = self
            .sleeper_wait_lock
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        self.cancelled.store(true, Ordering::Release);
        self.sleeper_wait.notify_one();
        drop(guard);
        let removed = self
            .waker
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        drop(removed);
    }

    fn spawn_sleeper(self: &Arc<Self>, deadline: Instant) -> io::Result<()> {
        if self
            .sleeper_started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Ok(());
        }

        let state = Arc::clone(self);
        match std::thread::Builder::new()
            .name("moirai-pal-timer".to_owned())
            .spawn(move || {
                let mut guard = state
                    .sleeper_wait_lock
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                while !state.cancelled.load(Ordering::Acquire) {
                    let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                        break;
                    };
                    #[cfg(test)]
                    state.publish_sleeper_event(SleeperEvent::Waiting);
                    let (next, _) = state
                        .sleeper_wait
                        .wait_timeout(guard, remaining)
                        .unwrap_or_else(|error| error.into_inner());
                    guard = next;
                }
                drop(guard);
                state.complete();
                #[cfg(test)]
                state.publish_sleeper_event(SleeperEvent::Exited);
            }) {
            Ok(_) => Ok(()),
            Err(error) => {
                self.completed.store(true, Ordering::Release);
                Err(error)
            }
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        self.state.cancel();
    }
}

impl Future for Timer {
    type Output = io::Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let now = Instant::now();
        if now >= self.deadline || self.state.completed.load(Ordering::Acquire) {
            self.state.completed.store(true, Ordering::Release);
            self.state.cancel();
            Poll::Ready(Ok(()))
        } else {
            self.state.register_waker(cx.waker());
            if let Err(error) = self.state.spawn_sleeper(self.deadline) {
                self.state.cancel();
                return Poll::Ready(Err(error));
            }

            if self.state.completed.load(Ordering::Acquire) {
                self.state.cancel();
                Poll::Ready(Ok(()))
            } else {
                Poll::Pending
            }
        }
    }
}

/// Create a timer that completes after the specified duration.
pub fn sleep(duration: Duration) -> Timer {
    Timer::new(duration)
}

#[cfg(test)]
#[path = "timer/tests.rs"]
mod tests;
