//! Platform-agnostic timer and timeout operations.

use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::task::{Context, Poll};
use std::thread;
use std::time::{Duration, Instant};

/// Platform-agnostic one-shot timer future.
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
    completed: AtomicBool,
    sleeper_started: AtomicBool,
    waker: Mutex<Option<std::task::Waker>>,
}

impl TimerState {
    fn new() -> Self {
        Self {
            completed: AtomicBool::new(false),
            sleeper_started: AtomicBool::new(false),
            waker: Mutex::new(None),
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
        let wake = if !self.completed.swap(true, Ordering::AcqRel) {
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

    fn clear_waker(&self) {
        let removed = self
            .waker
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        drop(removed);
    }

    fn finish_poll(&self) {
        self.completed.store(true, Ordering::Release);
        self.clear_waker();
    }

    fn finish_spawn(&self, worker: io::Result<thread::JoinHandle<()>>) -> io::Result<()> {
        match worker {
            Ok(_) => Ok(()),
            Err(error) => {
                self.finish_poll();
                Err(error)
            }
        }
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
        let worker = thread::Builder::new()
            .name("moirai-pal-timer".to_owned())
            .spawn(move || {
                let now = Instant::now();
                if deadline > now {
                    thread::sleep(deadline.duration_since(now));
                }
                state.complete();
            });
        self.finish_spawn(worker)
    }
}

impl Future for Timer {
    type Output = io::Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let now = Instant::now();
        if now >= self.deadline || self.state.completed.load(Ordering::Acquire) {
            self.state.completed.store(true, Ordering::Release);
            Poll::Ready(Ok(()))
        } else {
            self.state.register_waker(cx.waker());
            if let Err(error) = self.state.spawn_sleeper(self.deadline) {
                return Poll::Ready(Err(error));
            }

            if self.state.completed.load(Ordering::Acquire) {
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
