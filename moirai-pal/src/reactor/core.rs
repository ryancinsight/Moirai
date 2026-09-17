use std::collections::HashMap;
use std::io;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::task::Waker;
use std::time::{Duration, Instant};

use super::driver_failure::DriverFailureState;
use super::metrics::ReactorMetrics;
use super::registration::PlatformUpdateFailure;
#[cfg(any(unix, windows))]
use super::registration::PolledEvent;
#[cfg(windows)]
use super::registration::RegistrationGeneration;
#[cfg(windows)]
use super::socket_owner::SocketLease;
#[cfg(windows)]
use super::waiter_cancellation::{WaiterCancellation, WaiterCancellationState};
use crate::{Event, Interest, PlatformReactor, RawFd, Reactor, create_reactor};

#[cfg(windows)]
type WakerRegistration = Option<WaiterCancellation>;
#[cfg(not(windows))]
type WakerRegistration = ();

/// Send/Sync-safe internal key for platform handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct FdKey(pub(crate) usize);

impl From<RawFd> for FdKey {
    fn from(fd: RawFd) -> Self {
        Self(fd as usize)
    }
}

/// Information about registered file descriptors
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used for future telemetry/debugging per ADR requirements
pub struct FdInfo {
    /// Registered readiness interest.
    pub interest: Interest,
    /// When the descriptor was registered.
    pub registered_at: Instant,
    /// Number of events observed for this descriptor.
    pub event_count: u64,
    /// Waker armed for read readiness.
    pub read_waker: Option<Waker>,
    /// Waker armed for write readiness.
    pub write_waker: Option<Waker>,
}

/// Central async I/O reactor managing all platform-specific operations.
pub struct IoReactor {
    /// Platform-specific reactor implementation
    pub(crate) platform_reactor: Arc<PlatformReactor>,
    /// Event loop control
    pub(crate) running: Arc<AtomicBool>,
    /// Registered file descriptor tracking
    pub(crate) registered_fds: Arc<Mutex<HashMap<FdKey, FdInfo>>>,
    /// First terminal failure from a driven event loop.
    pub(super) driver_failure: DriverFailureState,
    /// Windows platform generation paired with each central registration.
    #[cfg(windows)]
    pub(super) platform_generations: Arc<Mutex<HashMap<FdKey, RegistrationGeneration>>>,
    /// Reactor-bound identity for owned Windows waiter cancellation.
    #[cfg(windows)]
    pub(super) waiter_cancellations: Arc<WaiterCancellationState>,
    /// Performance metrics
    pub(crate) metrics: Arc<ReactorMetrics>,
}

impl IoReactor {
    /// Create a new I/O reactor with platform-optimal implementation.
    pub fn new() -> io::Result<Self> {
        let platform_reactor = create_reactor()?;

        let platform_reactor = Arc::new(platform_reactor);
        let running = Arc::new(AtomicBool::new(false));
        let registered_fds = Arc::new(Mutex::new(HashMap::new()));
        let driver_failure = DriverFailureState::default();
        #[cfg(windows)]
        let platform_generations = Arc::new(Mutex::new(HashMap::new()));
        #[cfg(windows)]
        let waiter_cancellations = WaiterCancellationState::new(
            Arc::clone(&platform_reactor),
            Arc::clone(&running),
            Arc::clone(&registered_fds),
            Arc::clone(&platform_generations),
            driver_failure.clone(),
        );

        Ok(Self {
            platform_reactor,
            running,
            registered_fds,
            driver_failure,
            #[cfg(windows)]
            platform_generations,
            #[cfg(windows)]
            waiter_cancellations,
            metrics: Arc::new(ReactorMetrics::default()),
        })
    }

    #[cfg(all(test, windows))]
    pub(super) fn has_platform_generation(&self, fd: RawFd) -> bool {
        self.platform_generations
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .contains_key(&FdKey::from(fd))
    }

    /// Register a file descriptor for async I/O operations.
    ///
    /// # Errors
    ///
    /// Returns a platform registration error or the retained terminal driver
    /// failure after a driven event loop has stopped on an error.
    ///
    /// On Windows this raw-descriptor API requires the caller to keep the
    /// socket open until it unregisters or consumes the readiness interest.
    /// PAL network sockets use the private owner-aware registration path that
    /// retains snapshot ownership through `WSAPoll`.
    pub fn register_fd(&self, fd: RawFd, interest: Interest) -> io::Result<()> {
        let mut fds = self
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(error) = self.driver_failure.registration_error() {
            return Err(error);
        }

        // Hold central state across platform publication so readiness cannot be
        // dispatched before its matching central registration exists.
        #[cfg(windows)]
        let platform_generation = self
            .platform_reactor
            .register_waiter(fd, interest)
            .map_err(PlatformUpdateFailure::into_error)?;
        #[cfg(not(windows))]
        self.platform_reactor.register_fd(fd, interest)?;

        let key = FdKey::from(fd);
        #[cfg(windows)]
        self.platform_generations
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(key, platform_generation);
        #[cfg(windows)]
        self.waiter_cancellations.clear_interest(key, true, true);
        let displaced = fds.insert(
            key,
            FdInfo {
                interest,
                registered_at: Instant::now(),
                event_count: 0,
                read_waker: None,
                write_waker: None,
            },
        );

        // Update peak FD count metric
        let current_count = fds.len() as u64;
        self.metrics
            .peak_fd_count
            .fetch_max(current_count, Ordering::Relaxed);

        drop(fds);
        drop(displaced);
        Ok(())
    }

    /// Unregister a file descriptor.
    pub fn unregister_fd(&self, fd: RawFd) -> io::Result<()> {
        let mut fds = self
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        self.platform_reactor.unregister_fd(fd)?;
        let key = FdKey::from(fd);
        #[cfg(windows)]
        self.platform_generations
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .remove(&key);
        #[cfg(windows)]
        self.waiter_cancellations.clear_interest(key, true, true);
        let displaced = fds.remove(&key);
        drop(fds);
        drop(displaced);
        Ok(())
    }

    /// Run the event loop until stopped.
    ///
    /// A platform iteration failure is terminal for this reactor. The first
    /// failure is retained before every registered waiter is woken; later
    /// registrations return an error whose source is that original failure.
    ///
    /// # Errors
    ///
    /// Returns an error when the reactor was already started or a platform
    /// iteration fails.
    pub fn run(&self) -> io::Result<()> {
        // Relaxed: `running` is a single-location loop-control flag. It does
        // not publish reactor state; `stop` separately wakes the platform poll
        // so this loop observes the flag at its next iteration boundary.
        self.metrics
            .start_time
            .set(Instant::now())
            .map_err(|_| io::Error::other("Reactor already started"))?;
        self.running.store(true, Ordering::Relaxed);

        while self.running.load(Ordering::Relaxed) {
            if let Err(error) = self.run_iteration(Some(Duration::from_millis(10))) {
                return Err(self.publish_driver_failure(error));
            }
        }

        Ok(())
    }

    /// Run a single iteration of the event loop.
    pub fn run_iteration(&self, timeout: Option<Duration>) -> io::Result<()> {
        #[cfg(test)]
        if let Some(error) = self.driver_failure.take_iteration_failure() {
            return Err(error);
        }
        let iteration_start = Instant::now();

        #[cfg(any(unix, windows))]
        {
            let events = self.platform_reactor.poll_registered_events(timeout)?;
            for event in events {
                self.handle_polled_event(event)?;
            }
        }

        #[cfg(not(any(unix, windows)))]
        {
            let events = self.platform_reactor.poll_events(timeout)?;
            for event in events {
                self.handle_event(event)?;
            }
        }

        // Update metrics
        let iteration_time = iteration_start.elapsed().as_nanos() as u64;
        self.metrics
            .avg_event_time_ns
            .store(iteration_time, Ordering::Relaxed);

        Ok(())
    }

    /// Stop the event loop.
    pub fn stop(&self) -> io::Result<()> {
        // Relaxed: this store only requests loop termination. The platform
        // wake is the independent progress edge that releases a blocked poll;
        // no data written before this store is consumed through `running`.
        self.running.store(false, Ordering::Relaxed);
        self.platform_reactor.wake()
    }

    /// Handle a single I/O event.
    #[cfg(not(any(unix, windows)))]
    fn handle_event(&self, event: Event) -> io::Result<()> {
        // Update metrics
        self.metrics
            .events_processed
            .fetch_add(1, Ordering::Relaxed);

        // Consume matching one-shot interests before waking their tasks. A
        // task that still observes WouldBlock re-arms its interest on re-poll.
        self.wake_fd_waiters(event)
    }

    /// Handle readiness paired with its platform registration generation.
    #[cfg(any(unix, windows))]
    pub(super) fn handle_polled_event(&self, event: PolledEvent) -> io::Result<()> {
        self.metrics
            .events_processed
            .fetch_add(1, Ordering::Relaxed);

        #[cfg(windows)]
        if event.was_invalidated() {
            return self.wake_invalidated_waiters(event);
        }
        let readiness = event.event().clone();
        self.wake_fd_waiters_if_current(readiness, |platform| {
            platform.is_current_polled_event(&event)
        })
    }

    #[cfg(windows)]
    fn wake_invalidated_waiters(&self, event: PolledEvent) -> io::Result<()> {
        let key = FdKey::from(event.event().fd);
        let mut fds = self
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if !fds.contains_key(&key) {
            return Ok(());
        }
        let mut platform_generations = self
            .platform_generations
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if platform_generations.get(&key).copied() != Some(event.generation()) {
            return Ok(());
        }
        platform_generations.remove(&key);
        self.waiter_cancellations.clear_interest(key, true, true);
        let Some(mut fd_info) = fds.remove(&key) else {
            return Ok(());
        };
        drop(platform_generations);
        drop(fds);

        if let Some(waker) = fd_info.read_waker.take() {
            waker.wake();
        }
        if let Some(waker) = fd_info.write_waker.take() {
            waker.wake();
        }
        Ok(())
    }

    /// Wake tasks waiting on a specific file descriptor event.
    #[cfg(any(not(any(unix, windows)), test))]
    pub(super) fn wake_fd_waiters(&self, event: Event) -> io::Result<()> {
        self.wake_fd_waiters_if_current(event, |_| true)
    }

    fn wake_fd_waiters_if_current(
        &self,
        event: Event,
        is_current: impl FnOnce(&PlatformReactor) -> bool,
    ) -> io::Result<()> {
        self.wake_fd_waiters_with_platform(event, is_current, |_, fd, interest| {
            self.update_platform_registration(fd, interest)
        })
    }

    fn update_platform_registration(
        &self,
        fd: RawFd,
        interest: Interest,
    ) -> Result<(), PlatformUpdateFailure> {
        #[cfg(any(unix, windows))]
        {
            self.platform_reactor.update_registration(fd, interest)
        }
        #[cfg(not(any(unix, windows)))]
        {
            self.platform_reactor
                .unregister_fd(fd)
                .map_err(|error| PlatformUpdateFailure::new(error, None))?;
            if interest.readable || interest.writable {
                self.platform_reactor
                    .register_fd(fd, interest)
                    .map_err(|error| PlatformUpdateFailure::new(error, None))?;
            }
            Ok(())
        }
    }

    pub(super) fn wake_fd_waiters_with_platform(
        &self,
        event: Event,
        is_current: impl FnOnce(&PlatformReactor) -> bool,
        update_platform: impl FnOnce(
            &PlatformReactor,
            RawFd,
            Interest,
        ) -> Result<(), PlatformUpdateFailure>,
    ) -> io::Result<()> {
        let key = FdKey::from(event.fd);
        let mut fds = self
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let Some(fd_info) = fds.get_mut(&key) else {
            return Ok(());
        };
        if !is_current(&self.platform_reactor) {
            return Ok(());
        }
        fd_info.event_count += 1;

        let consume_read =
            fd_info.interest.readable && (event.readable || event.error || event.hangup);
        let consume_write =
            fd_info.interest.writable && (event.writable || event.error || event.hangup);
        if !consume_read && !consume_write {
            return Ok(());
        }

        let remaining = Interest {
            readable: fd_info.interest.readable && !consume_read,
            writable: fd_info.interest.writable && !consume_write,
            error: fd_info.interest.error,
        };

        // Every registered readiness interest is one-shot at this layer. The
        // platform backends are level-triggered only to close the syscall-to-
        // registration race; retaining a delivered writable interest would
        // otherwise make the event loop spin indefinitely.
        let platform_result = update_platform(&self.platform_reactor, event.fd, remaining);

        let fd_info = fds
            .get_mut(&key)
            .expect("fd registration remained locked during readiness update");
        let read_waker = if consume_read {
            #[cfg(windows)]
            self.waiter_cancellations.clear_interest(key, true, false);
            fd_info.read_waker.take()
        } else {
            None
        };
        let write_waker = if consume_write {
            #[cfg(windows)]
            self.waiter_cancellations.clear_interest(key, false, true);
            fd_info.write_waker.take()
        } else {
            None
        };
        let mut stranded_read_waker = None;
        let mut stranded_write_waker = None;
        let (remove_registration, platform_error) = match platform_result {
            Ok(()) => {
                if remaining.readable || remaining.writable {
                    fd_info.interest = remaining;
                    (false, None)
                } else {
                    (true, None)
                }
            }
            Err(failure) => {
                // A delivered waiter must observe its wake even when the
                // backend transition fails. Wake every remaining waiter too:
                // each one must re-poll and republish its interest instead of
                // depending on a transition that did not complete.
                stranded_read_waker = fd_info.read_waker.take();
                stranded_write_waker = fd_info.write_waker.take();
                if let Some(armed_interest) = failure.armed_interest() {
                    fd_info.interest = armed_interest;
                    (false, Some(failure.into_error()))
                } else {
                    (true, Some(failure.into_error()))
                }
            }
        };
        if remove_registration {
            #[cfg(windows)]
            self.platform_generations
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .remove(&key);
            fds.remove(&key);
            #[cfg(windows)]
            self.waiter_cancellations.clear_interest(key, true, true);
        }
        drop(fds);

        if let Some(waker) = read_waker {
            waker.wake();
        }
        if let Some(waker) = write_waker {
            waker.wake();
        }
        if let Some(waker) = stranded_read_waker {
            waker.wake();
        }
        if let Some(waker) = stranded_write_waker {
            waker.wake();
        }
        platform_error.map_or(Ok(()), Err)
    }

    /// Register a task's waker for a file descriptor and interest.
    ///
    /// # Errors
    ///
    /// Returns a platform registration error or the retained terminal driver
    /// failure after a driven event loop has stopped on an error.
    pub fn register_waker(&self, fd: RawFd, interest: Interest, waker: Waker) -> io::Result<()> {
        #[cfg(windows)]
        {
            self.register_waker_with_owner(fd, interest, waker, None)
                .map(drop)
        }
        #[cfg(not(windows))]
        {
            self.register_waker_with_owner(fd, interest, waker)
        }
    }

    #[cfg(windows)]
    pub(crate) fn register_owned_waker(
        &self,
        fd: RawFd,
        interest: Interest,
        waker: Waker,
        owner: SocketLease,
    ) -> io::Result<WaiterCancellation> {
        self.register_waker_with_owner(fd, interest, waker, Some(owner.downgrade()))?
            .ok_or_else(|| io::Error::other("owned waiter cancellation was not published"))
    }

    fn register_waker_with_owner(
        &self,
        fd: RawFd,
        interest: Interest,
        waker: Waker,
        #[cfg(windows)] owner: Option<super::socket_owner::WeakSocketOwner>,
    ) -> io::Result<WakerRegistration> {
        #[cfg(windows)]
        let cancellation = owner
            .as_ref()
            .map(|_| self.waiter_cancellations.reserve(fd, interest))
            .transpose()?;
        let mut fds = self
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(error) = self.driver_failure.registration_error() {
            return Err(error);
        }
        if let Some(fd_info) = fds.get_mut(&FdKey::from(fd)) {
            let mut new_interest = fd_info.interest;
            if interest.readable {
                new_interest.readable = true;
            }
            if interest.writable {
                new_interest.writable = true;
            }
            #[cfg(windows)]
            let replacement = if let Some(owner) = owner.clone() {
                self.platform_reactor.replace_owned_waiter_registration(
                    fd,
                    new_interest,
                    interest,
                    owner,
                )
            } else {
                self.platform_reactor
                    .replace_waiter_registration(fd, new_interest, interest)
            };
            #[cfg(unix)]
            let replacement = self
                .platform_reactor
                .replace_registration(fd, new_interest)
                .map(|()| true);
            #[cfg(not(any(unix, windows)))]
            let replacement = self
                .update_platform_registration(fd, new_interest)
                .map(|()| true);
            let replaced_existing = match replacement {
                #[cfg(windows)]
                Ok(replacement) => {
                    self.platform_generations
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner())
                        .insert(FdKey::from(fd), replacement.generation);
                    replacement.replaced_existing
                }
                #[cfg(not(windows))]
                Ok(replaced_existing) => replaced_existing,
                Err(failure) => {
                    let read_waker = fd_info.read_waker.take();
                    let write_waker = fd_info.write_waker.take();
                    let remove_registration = if let Some(armed_interest) = failure.armed_interest()
                    {
                        fd_info.interest = armed_interest;
                        false
                    } else {
                        true
                    };
                    let error = failure.into_error();
                    if remove_registration {
                        #[cfg(windows)]
                        self.platform_generations
                            .lock()
                            .unwrap_or_else(|poison| poison.into_inner())
                            .remove(&FdKey::from(fd));
                        fds.remove(&FdKey::from(fd));
                    }
                    drop(fds);
                    if let Some(waker) = read_waker {
                        waker.wake();
                    }
                    if let Some(waker) = write_waker {
                        waker.wake();
                    }
                    return Err(error);
                }
            };
            let displaced_read_waker = (!replaced_existing || interest.readable)
                .then(|| fd_info.read_waker.take())
                .flatten();
            let displaced_write_waker = (!replaced_existing || interest.writable)
                .then(|| fd_info.write_waker.take())
                .flatten();
            fd_info.interest = if replaced_existing {
                new_interest
            } else {
                interest
            };

            if interest.readable {
                fd_info.read_waker = Some(waker.clone());
            }
            if interest.writable {
                fd_info.write_waker = Some(waker);
            }
            #[cfg(windows)]
            let registration = if let Some(cancellation) = cancellation {
                if !replaced_existing {
                    self.waiter_cancellations
                        .clear_interest(FdKey::from(fd), true, true);
                }
                self.waiter_cancellations.publish(&cancellation);
                Some(cancellation)
            } else {
                self.waiter_cancellations.clear_interest(
                    FdKey::from(fd),
                    !replaced_existing || interest.readable,
                    !replaced_existing || interest.writable,
                );
                None
            };
            drop(fds);
            if replaced_existing {
                drop(displaced_read_waker);
                drop(displaced_write_waker);
            } else {
                if let Some(waker) = displaced_read_waker {
                    waker.wake();
                }
                if let Some(waker) = displaced_write_waker {
                    waker.wake();
                }
            }
            #[cfg(windows)]
            {
                Ok(registration)
            }
            #[cfg(not(windows))]
            Ok(())
        } else {
            // Publish the waker in the same state-lock transaction as the
            // platform registration. The poll thread may observe readiness as
            // soon as registration wakes it, but it cannot consume a
            // temporarily wakerless entry before insertion completes.
            #[cfg(windows)]
            let platform_generation = if let Some(owner) = owner.clone() {
                self.platform_reactor
                    .register_owned_waiter(fd, interest, owner)
            } else {
                self.platform_reactor.register_waiter(fd, interest)
            }
            .map_err(PlatformUpdateFailure::into_error)?;
            #[cfg(not(windows))]
            self.platform_reactor.register_fd(fd, interest)?;
            let mut fd_info = FdInfo {
                interest,
                registered_at: Instant::now(),
                event_count: 0,
                read_waker: None,
                write_waker: None,
            };
            if interest.readable {
                fd_info.read_waker = Some(waker.clone());
            }
            if interest.writable {
                fd_info.write_waker = Some(waker);
            }

            let key = FdKey::from(fd);
            #[cfg(windows)]
            self.platform_generations
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .insert(key, platform_generation);
            fds.insert(key, fd_info);
            let current_count = fds.len() as u64;
            self.metrics
                .peak_fd_count
                .fetch_max(current_count, Ordering::Relaxed);
            #[cfg(windows)]
            {
                if let Some(cancellation) = cancellation {
                    self.waiter_cancellations.publish(&cancellation);
                    Ok(Some(cancellation))
                } else {
                    self.waiter_cancellations.clear_interest(
                        key,
                        interest.readable,
                        interest.writable,
                    );
                    Ok(None)
                }
            }
            #[cfg(not(windows))]
            Ok(())
        }
    }

    /// Remove wakers for a file descriptor.
    pub fn deregister_waker(&self, fd: RawFd, interest: Interest) {
        let mut read_waker = None;
        let mut write_waker = None;
        if let Ok(mut fds) = self.registered_fds.lock()
            && let Some(fd_info) = fds.get_mut(&FdKey::from(fd))
        {
            if interest.readable {
                read_waker = fd_info.read_waker.take();
            }
            if interest.writable {
                write_waker = fd_info.write_waker.take();
            }
            #[cfg(windows)]
            self.waiter_cancellations.clear_interest(
                FdKey::from(fd),
                interest.readable,
                interest.writable,
            );
        }
        drop(read_waker);
        drop(write_waker);
    }

    /// Wake up the reactor from blocking poll.
    pub fn wake(&self) -> io::Result<()> {
        self.platform_reactor.wake()
    }

    /// Get current reactor metrics.
    pub fn metrics(&self) -> ReactorMetrics {
        ReactorMetrics {
            events_processed: AtomicU64::new(self.metrics.events_processed.load(Ordering::Relaxed)),
            avg_event_time_ns: AtomicU64::new(
                self.metrics.avg_event_time_ns.load(Ordering::Relaxed),
            ),
            peak_fd_count: AtomicU64::new(self.metrics.peak_fd_count.load(Ordering::Relaxed)),
            start_time: std::sync::OnceLock::new(),
        }
    }
}
