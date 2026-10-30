//! Platform-agnostic timer and timeout operations.

use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

#[cfg(loom)]
use loom::{
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicU8, Ordering},
    },
    thread,
};
#[cfg(not(loom))]
use std::{
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicU8, Ordering},
    },
    thread,
};

const TIMER_PENDING: u8 = 0;
const TIMER_COMPLETED: u8 = 1;
const TIMER_CANCELLED: u8 = 2;
const TIMER_LIFECYCLE_MASK: u8 = 3;
const TIMER_SLEEPER_PUBLISHED: u8 = 4;

/// Platform-agnostic one-shot timer future.
///
/// Dropping a pending timer cancels its sleeper and releases its waker.
pub struct Timer {
    deadline: Instant,
    state: Arc<TimerState>,
}

impl Timer {
    /// Create a timer that completes after `duration`.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::InvalidInput`] when `duration` cannot be
    /// represented as an [`Instant`] after the current time.
    pub fn new(duration: Duration) -> io::Result<Self> {
        let deadline = deadline_after(Instant::now(), duration)?;
        Ok(Self {
            deadline,
            state: Arc::new(TimerState::new()),
        })
    }

    /// Return the absolute completion deadline.
    #[must_use]
    pub fn deadline(&self) -> Instant {
        self.deadline
    }
}

fn deadline_after(start: Instant, duration: Duration) -> io::Result<Instant> {
    start.checked_add(duration).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "timer duration exceeds the monotonic clock range",
        )
    })
}

struct TimerState {
    status: AtomicU8,
    sleeper_started: AtomicBool,
    sleeper_thread: OnceLock<thread::Thread>,
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
            status: AtomicU8::new(TIMER_PENDING),
            sleeper_started: AtomicBool::new(false),
            sleeper_thread: OnceLock::new(),
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
        if self.transition_from_pending(TIMER_COMPLETED).is_err() {
            self.clear_waker();
            return;
        }
        let wake = self
            .waker
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        if let Some(wake) = wake {
            wake.wake();
        }
    }

    fn is_completed(&self) -> bool {
        lifecycle(self.status.load(Ordering::Acquire)) == TIMER_COMPLETED
    }

    fn finish_poll(&self) {
        let observed = self
            .transition_from_pending(TIMER_COMPLETED)
            .unwrap_or_else(|current| current);
        self.clear_waker();
        self.unpark_sleeper(observed);
    }

    fn clear_waker(&self) {
        let removed = self
            .waker
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        drop(removed);
    }

    fn cancel(&self) {
        let observed = self
            .transition_from_pending(TIMER_CANCELLED)
            .unwrap_or_else(|current| current);
        self.unpark_sleeper(observed);
    }

    fn transition_from_pending(&self, next: u8) -> Result<u8, u8> {
        self.status
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (lifecycle(current) == TIMER_PENDING)
                    .then_some((current & !TIMER_LIFECYCLE_MASK) | next)
            })
    }

    fn unpark_sleeper(&self, observed: u8) {
        if observed & TIMER_SLEEPER_PUBLISHED != 0 {
            self.sleeper_thread
                .get()
                .expect("invariant: publication bit follows sleeper initialization")
                .unpark();
        }
    }

    fn run_sleeper(&self, deadline: Instant) {
        self.sleeper_thread
            .set(thread::current())
            .expect("invariant: one worker owns each timer state");
        self.status
            .fetch_or(TIMER_SLEEPER_PUBLISHED, Ordering::AcqRel);

        loop {
            if lifecycle(self.status.load(Ordering::Acquire)) != TIMER_PENDING {
                self.clear_waker();
                break;
            }
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                self.complete();
                break;
            };
            #[cfg(test)]
            self.publish_sleeper_event(SleeperEvent::Waiting);
            park_timeout(remaining);
        }

        #[cfg(test)]
        self.publish_sleeper_event(SleeperEvent::Exited);
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
            .spawn(move || state.run_sleeper(deadline));
        self.finish_spawn(worker)
    }
}

fn lifecycle(status: u8) -> u8 {
    status & TIMER_LIFECYCLE_MASK
}

fn park_timeout(_duration: Duration) {
    #[cfg(not(loom))]
    thread::park_timeout(_duration);
    #[cfg(loom)]
    thread::park();
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
        if now >= self.deadline || self.state.is_completed() {
            self.state.finish_poll();
            Poll::Ready(Ok(()))
        } else {
            self.state.register_waker(cx.waker());
            if let Err(error) = self.state.spawn_sleeper(self.deadline) {
                return Poll::Ready(Err(error));
            }

            if self.state.is_completed() {
                self.state.finish_poll();
                Poll::Ready(Ok(()))
            } else {
                Poll::Pending
            }
        }
    }
}

/// Create a timer that completes after the specified duration.
///
/// # Errors
///
/// Returns [`io::ErrorKind::InvalidInput`] when `duration` cannot be
/// represented as an [`Instant`] after the current time.
pub fn sleep(duration: Duration) -> io::Result<Timer> {
    Timer::new(duration)
}

#[cfg(test)]
#[path = "timer/tests.rs"]
mod tests;
