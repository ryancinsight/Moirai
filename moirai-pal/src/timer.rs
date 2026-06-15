//! Platform-agnostic timer and timeout operations.

use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::task::{Context, Poll};
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
        Self {
            deadline: Instant::now() + duration,
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
        let mut stored = self.waker.lock().unwrap_or_else(|e| e.into_inner());
        let replace = match stored.as_ref() {
            Some(current) => !current.will_wake(waker),
            None => true,
        };
        if replace {
            *stored = Some(waker.clone());
        }
    }

    fn complete(&self) {
        if !self.completed.swap(true, Ordering::AcqRel) {
            if let Some(waker) = self.waker.lock().unwrap_or_else(|e| e.into_inner()).take() {
                waker.wake();
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
        match std::thread::Builder::new()
            .name("moirai-pal-timer".to_owned())
            .spawn(move || {
                let now = Instant::now();
                if deadline > now {
                    std::thread::sleep(deadline.duration_since(now));
                }
                state.complete();
            }) {
            Ok(_) => Ok(()),
            Err(error) => {
                self.completed.store(true, Ordering::Release);
                Err(error)
            }
        }
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
