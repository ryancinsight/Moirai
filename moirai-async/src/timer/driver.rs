#![expect(
    clippy::unwrap_used,
    reason = "ratchet MOIRAI-UNWRAP-1: pre-existing debt"
)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::panic::{AssertUnwindSafe, catch_unwind};
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Instant;

use crate::timer::registration::TimerRegistration;

pub(super) struct ScheduledTimer {
    pub(super) deadline: Instant,
    pub(super) sequence: u64,
    pub(super) registration: Arc<TimerRegistration>,
}

impl PartialEq for ScheduledTimer {
    fn eq(&self, other: &Self) -> bool {
        self.deadline == other.deadline && self.sequence == other.sequence
    }
}

impl Eq for ScheduledTimer {}

impl PartialOrd for ScheduledTimer {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledTimer {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .deadline
            .cmp(&self.deadline)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

pub(super) struct TimerDriver {
    state: Mutex<TimerDriverState>,
    available: Condvar,
    #[cfg(test)]
    notifications: AtomicUsize,
}

struct TimerDriverState {
    timers: BinaryHeap<ScheduledTimer>,
    next_sequence: u64,
    /// Number of heap entries whose registration has been cancelled but which
    /// are still resident in `timers`. Invariant (all mutations happen under
    /// the state mutex): every cancelled-and-resident entry is counted exactly
    /// once — [`TimerDriver::cancel`] increments on the first cancellation of
    /// an in-heap registration; entry removal (head drain, compaction)
    /// decrements or resets.
    dead: usize,
}

impl TimerDriverState {
    /// Rebuild the heap retaining only live (non-cancelled) timers.
    ///
    /// Without compaction, the dominant `timeout(op)` pattern — cancellation
    /// after milliseconds against a 30-60 s deadline — retains every cancelled
    /// entry until its deadline, so memory grows as cancel-rate x
    /// timeout-horizon.
    fn compact(&mut self) {
        self.timers.retain(|timer| {
            let live = !timer.registration.is_cancelled();
            if !live {
                timer.registration.clear_in_heap();
            }
            live
        });
        self.dead = 0;
    }
}

impl TimerDriver {
    fn new() -> Self {
        Self {
            state: Mutex::new(TimerDriverState {
                timers: BinaryHeap::new(),
                next_sequence: 0,
                dead: 0,
            }),
            available: Condvar::new(),
            #[cfg(test)]
            notifications: AtomicUsize::new(0),
        }
    }

    fn start() -> Arc<Self> {
        let driver = Arc::new(Self::new());

        let worker = Arc::clone(&driver);
        std::thread::Builder::new()
            .name("moirai-timer-driver".to_string())
            .spawn(move || worker.run())
            .expect("failed to start Moirai timer driver");

        driver
    }

    pub(super) fn schedule(&self, deadline: Instant, registration: Arc<TimerRegistration>) {
        let mut state = self.state.lock().unwrap();
        let sequence = state.next_sequence;
        state.next_sequence = state.next_sequence.wrapping_add(1);
        registration.mark_in_heap();
        state.timers.push(ScheduledTimer {
            deadline,
            sequence,
            registration,
        });
        drop(state);
        self.notify_driver();
    }

    /// Cancel a registration, accounting for its now-dead heap entry and
    /// compacting the heap when dead entries dominate.
    ///
    /// Compaction threshold derivation: the heap is rebuilt (`retain` +
    /// re-heapify, O(n)) when dead entries outnumber live ones
    /// (`2 * dead > len`). `dead` resets to 0 on every rebuild, so at least
    /// `len / 2` cancellations must occur between two rebuilds; the O(n)
    /// rebuild therefore amortizes to O(1) per cancellation, while retained
    /// heap size stays bounded below 2x the live timer count instead of
    /// growing with the cancel-rate x timeout-horizon product.
    pub(super) fn cancel(&self, registration: &TimerRegistration) -> Option<std::task::Waker> {
        let mut state = self.state.lock().unwrap();
        let mut wake_driver = false;
        // First cancellation of a still-resident entry: count it dead. Both
        // checks happen under the state mutex, so they cannot race the driver
        // thread's pops.
        if registration.cancel() && registration.is_in_heap() {
            wake_driver = state
                .timers
                .peek()
                .is_some_and(|timer| std::ptr::eq(timer.registration.as_ref(), registration));
            state.dead += 1;
            if state.dead > state.timers.len() / 2 {
                state.compact();
                wake_driver = true;
            }
        }
        drop(state);

        // Moving the waker out guarantees cancellation releases its captured
        // task state promptly. The caller invokes or drops it after both the
        // driver-state and registration mutexes have been released.
        let waker = registration.take_waker();

        // Cancellation changes the driver's current wait only when it removes
        // the heap head or a compaction rebuilds the heap. The state mutex and
        // condition-variable wait prevent a notification from being lost.
        if wake_driver {
            self.notify_driver();
        }
        waker
    }

    fn notify_driver(&self) {
        #[cfg(test)]
        self.notifications.fetch_add(1, AtomicOrdering::Relaxed);
        self.available.notify_one();
    }

    /// Number of heap-resident entries (live + not-yet-reclaimed dead).
    /// Test-only observability for the compaction invariant.
    #[cfg(test)]
    pub(super) fn scheduled_len(&self) -> usize {
        self.state.lock().unwrap().timers.len()
    }

    #[cfg(test)]
    fn notification_count(&self) -> usize {
        self.notifications.load(AtomicOrdering::Relaxed)
    }

    fn run(&self) {
        let mut state = self.state.lock().unwrap();
        loop {
            while state
                .timers
                .peek()
                .is_some_and(|timer| timer.registration.is_cancelled())
            {
                let timer = state.timers.pop().expect("timer existed after peek");
                timer.registration.clear_in_heap();
                // The entry was dead-counted when cancelled in-heap; reclaim it.
                debug_assert!(state.dead > 0, "cancelled in-heap entry must be counted");
                state.dead = state.dead.saturating_sub(1);
            }

            let Some(next_deadline) = state.timers.peek().map(|timer| timer.deadline) else {
                state = self.available.wait(state).unwrap();
                continue;
            };

            let now = Instant::now();
            if next_deadline <= now {
                let timer = state.timers.pop().expect("timer existed after peek");
                timer.registration.clear_in_heap();
                drop(state);
                if !timer.registration.is_cancelled() {
                    // A disabled trace evaluates no clock/field arguments.
                    // Observe the shared driver's wake point, including QUIC,
                    // operation and probe timers; protocol execution follows.
                    tracing::trace!(
                        lateness_ns = Instant::now()
                            .saturating_duration_since(timer.deadline)
                            .as_nanos(),
                        "timer driver wakes registration"
                    );
                    wake_registration(&timer.registration);
                }
                state = self.state.lock().unwrap();
                continue;
            }

            let timeout = next_deadline - now;
            let (guard, _) = self.available.wait_timeout(state, timeout).unwrap();
            state = guard;
        }
    }
}

fn wake_registration(registration: &TimerRegistration) {
    let Err(payload) = catch_unwind(AssertUnwindSafe(|| registration.wake())) else {
        return;
    };

    // A panic payload may itself own arbitrary code in Drop. Keep that second
    // unwind off the process-wide driver thread too. A recursively panicking
    // replacement payload cannot be dropped safely, so retain that exceptional
    // payload rather than terminating the timer service.
    let payload_drop_panicked = match catch_unwind(AssertUnwindSafe(|| drop(payload))) {
        Ok(()) => false,
        Err(payload) => {
            std::mem::forget(payload);
            true
        }
    };
    tracing::error!(
        payload_drop_panicked,
        "timer waker panicked; process-wide timer driver continues"
    );
}

/// The process-wide timer driver, started on first use.
///
/// The driver thread is detached and runs for the rest of the process: the
/// handle lives in a `static`, which is never dropped, so no owner exists whose
/// drop could stop and join it, and it parks on `available` while no timer is
/// scheduled. Under Miri it fails a test run with "the main thread terminated
/// without waiting for all remaining threads"; `-Zmiri-ignore-leaks` disables
/// that check. Drivers built with `TimerDriver::new` in tests own no thread.
pub(super) fn timer_driver() -> &'static Arc<TimerDriver> {
    static DRIVER: OnceLock<Arc<TimerDriver>> = OnceLock::new();
    DRIVER.get_or_init(TimerDriver::start)
}

#[cfg(test)]
mod tests {
    use super::{TimerDriver, timer_driver};
    use crate::timer::Delay;
    use crate::timer::registration::TimerRegistration;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::{Arc, mpsc::SyncSender, mpsc::sync_channel};
    use std::task::{Context, Wake, Waker};
    use std::time::{Duration, Instant};

    const TIMER_WORKER_TEST_TIMEOUT: Duration = Duration::from_secs(5);

    struct OwnerWake {
        owner: Arc<()>,
    }

    impl Wake for OwnerWake {
        fn wake(self: Arc<Self>) {
            assert!(Arc::strong_count(&self.owner) > 0);
            panic!("pending owner waker must not run");
        }
    }

    struct PanicPayload {
        dropping: SyncSender<()>,
    }

    impl Drop for PanicPayload {
        fn drop(&mut self) {
            self.dropping
                .send(())
                .expect("invariant: payload-drop receiver remains active");
            panic!("panic payload destructor");
        }
    }

    struct PanickingWake {
        started: SyncSender<()>,
        payload_dropping: SyncSender<()>,
    }

    impl Wake for PanickingWake {
        fn wake(self: Arc<Self>) {
            self.started
                .send(())
                .expect("invariant: panic-start receiver remains active");
            std::panic::panic_any(PanicPayload {
                dropping: self.payload_dropping.clone(),
            });
        }
    }

    struct ChannelWake {
        woke: SyncSender<()>,
    }

    impl Wake for ChannelWake {
        fn wake(self: Arc<Self>) {
            self.woke
                .send(())
                .expect("invariant: later-timer receiver remains active");
        }
    }

    #[test]
    fn cancellation_notifies_only_for_effective_head_changes() {
        let driver = TimerDriver::new();
        let now = Instant::now();
        let head = TimerRegistration::new(Waker::noop().clone());
        let later = TimerRegistration::new(Waker::noop().clone());

        driver.schedule(now + Duration::from_secs(60), Arc::clone(&head));
        driver.schedule(now + Duration::from_secs(120), Arc::clone(&later));
        let scheduled_notifications = driver.notification_count();

        drop(driver.cancel(&later));
        assert_eq!(driver.notification_count(), scheduled_notifications);

        drop(driver.cancel(&head));
        assert_eq!(driver.notification_count(), scheduled_notifications + 1);
        assert_eq!(driver.scheduled_len(), 0);

        drop(driver.cancel(&head));
        assert_eq!(driver.notification_count(), scheduled_notifications + 1);
    }

    #[test]
    fn cancelled_delay_registration_detaches_waker_behind_live_head() {
        let driver = TimerDriver::new();
        let now = Instant::now();
        let head = TimerRegistration::new(Waker::noop().clone());
        let owner = Arc::new(());
        let later = TimerRegistration::new(Waker::from(Arc::new(OwnerWake {
            owner: Arc::clone(&owner),
        })));

        driver.schedule(now + Duration::from_secs(60), Arc::clone(&head));
        driver.schedule(now + Duration::from_secs(120), Arc::clone(&later));
        assert_eq!(Arc::strong_count(&owner), 2);

        drop(driver.cancel(&later));

        assert_eq!(driver.scheduled_len(), 2);
        assert!(later.is_cancelled());
        assert_eq!(Arc::strong_count(&owner), 1);
    }

    #[test]
    fn panicking_waker_does_not_stop_timer_driver() {
        let (panic_started_tx, panic_started_rx) = sync_channel(0);
        let (payload_dropping_tx, payload_dropping_rx) = sync_channel(0);
        let (later_woke_tx, later_woke_rx) = sync_channel(0);
        let now = Instant::now();
        let panicking = TimerRegistration::new(Waker::from(Arc::new(PanickingWake {
            started: panic_started_tx,
            payload_dropping: payload_dropping_tx,
        })));
        let later = TimerRegistration::new(Waker::from(Arc::new(ChannelWake {
            woke: later_woke_tx,
        })));

        timer_driver().schedule(now, panicking);
        timer_driver().schedule(now + Duration::from_millis(1), later);

        panic_started_rx
            .recv_timeout(TIMER_WORKER_TEST_TIMEOUT)
            .expect("panicking timer must begin within the test budget");
        payload_dropping_rx
            .recv_timeout(TIMER_WORKER_TEST_TIMEOUT)
            .expect("panic payload destructor must run within the test budget");
        later_woke_rx
            .recv_timeout(TIMER_WORKER_TEST_TIMEOUT)
            .expect("later timer must wake within the test budget");
    }

    #[test]
    fn cancelled_timers_are_compacted_before_their_deadline() {
        // Regression for unbounded retention: cancelled entries used to stay in
        // the heap until their deadline (removed only at the heap head), so the
        // dominant timeout(op) pattern — ms-scale cancels against 30-60 s
        // deadlines — grew memory as cancel-rate x timeout-horizon.
        //
        // Bound derivation: compaction fires whenever `2 * dead > len` and
        // resets `dead` to 0, so at rest `2 * dead <= len` always holds. With
        // `live = 1` timer remaining, `len = live + dead = 1 + dead` gives
        // `dead <= 1`, hence `len <= 2`.
        //
        // Pre-existing timer registrations are permitted, so the bound tracks
        // the measured initial heap size instead of assuming process-global
        // isolation from other timer coverage.
        let mut cx = Context::from_waker(Waker::noop());

        // Measure the driver's initial state; other tests may have left timers
        // in the process-global driver, so all assertions are relative.
        let initial_len = timer_driver().scheduled_len();
        let far = Duration::from_secs(3600);

        // Add enough timers that cancelling all but one triggers compaction
        // even when `initial_len` is non-zero.
        let to_add = initial_len + 100;
        let mut delays: Vec<Delay> = (0..to_add).map(|_| Delay::new(far)).collect();
        for delay in &mut delays {
            // First poll registers the delay with the driver.
            assert!(Pin::new(delay).poll(&mut cx).is_pending());
        }
        assert_eq!(timer_driver().scheduled_len(), initial_len + to_add);

        // Cancel all but one (Delay::drop routes through TimerDriver::cancel).
        delays.truncate(1);

        let retained = timer_driver().scheduled_len();
        assert!(
            retained <= 2 * initial_len + 2,
            "compaction must retain at most twice the live timer bound: retained {retained}, expected <= {}",
            2 * initial_len + 2
        );

        // The surviving delay is still scheduled and pending.
        assert!(Pin::new(&mut delays[0]).poll(&mut cx).is_pending());
        assert!(timer_driver().scheduled_len() >= 1);
    }
}
