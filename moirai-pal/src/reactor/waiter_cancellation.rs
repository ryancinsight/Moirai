//! Reactor-bound cancellation for Windows socket readiness waiters.

use std::collections::HashMap;
use std::io;
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::task::Waker;

use super::core::{FdInfo, FdKey};
use super::driver_failure::DriverFailureState;
use super::registration::RegistrationGeneration;
use crate::{Interest, PlatformReactor, RawFd, Reactor};

#[derive(Clone, Copy, Default)]
struct WaiterIds {
    read: Option<u64>,
    write: Option<u64>,
}

pub(super) struct WaiterCancellationState {
    platform: Arc<PlatformReactor>,
    running: Arc<AtomicBool>,
    fds: Arc<Mutex<HashMap<FdKey, FdInfo>>>,
    generations: Arc<Mutex<HashMap<FdKey, RegistrationGeneration>>>,
    driver_failure: DriverFailureState,
    ids: Mutex<HashMap<FdKey, WaiterIds>>,
    next_id: AtomicU64,
}

impl WaiterCancellationState {
    pub(super) fn new(
        platform: Arc<PlatformReactor>,
        running: Arc<AtomicBool>,
        fds: Arc<Mutex<HashMap<FdKey, FdInfo>>>,
        generations: Arc<Mutex<HashMap<FdKey, RegistrationGeneration>>>,
        driver_failure: DriverFailureState,
    ) -> Arc<Self> {
        Arc::new(Self {
            platform,
            running,
            fds,
            generations,
            driver_failure,
            ids: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(0),
        })
    }

    pub(super) fn reserve(
        self: &Arc<Self>,
        fd: RawFd,
        interest: Interest,
    ) -> io::Result<WaiterCancellation> {
        let id = self
            .next_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| io::Error::other("reactor waiter cancellation identity exhausted"))?
            + 1;
        Ok(WaiterCancellation {
            state: Arc::downgrade(self),
            key: FdKey::from(fd),
            id,
            interest,
        })
    }

    pub(super) fn publish(&self, cancellation: &WaiterCancellation) {
        let key = cancellation.key;
        let mut ids = self.ids.lock().unwrap_or_else(|poison| poison.into_inner());
        let entry = ids.entry(key).or_default();
        if cancellation.interest.readable {
            entry.read = Some(cancellation.id);
        }
        if cancellation.interest.writable {
            entry.write = Some(cancellation.id);
        }
    }

    pub(super) fn clear_interest(&self, key: FdKey, read: bool, write: bool) {
        let mut ids = self.ids.lock().unwrap_or_else(|poison| poison.into_inner());
        let Some(entry) = ids.get_mut(&key) else {
            return;
        };
        if read {
            entry.read = None;
        }
        if write {
            entry.write = None;
        }
        if entry.read.is_none() && entry.write.is_none() {
            ids.remove(&key);
        }
    }

    pub(super) fn clear_all(&self) {
        self.ids
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clear();
    }

    #[cfg(test)]
    pub(super) fn has_interest(&self, key: FdKey, interest: Interest) -> bool {
        let ids = self.ids.lock().unwrap_or_else(|poison| poison.into_inner());
        ids.get(&key).is_some_and(|entry| {
            (!interest.readable || entry.read.is_some())
                && (!interest.writable || entry.write.is_some())
        })
    }

    fn cancel(&self, cancellation: &WaiterCancellation) {
        let mut displaced = Vec::<Waker>::new();
        let mut fds = self.fds.lock().unwrap_or_else(|poison| poison.into_inner());
        let mut ids = self.ids.lock().unwrap_or_else(|poison| poison.into_inner());
        let Some(current_ids) = ids.get_mut(&cancellation.key) else {
            return;
        };
        let cancel_read =
            cancellation.interest.readable && current_ids.read == Some(cancellation.id);
        let cancel_write =
            cancellation.interest.writable && current_ids.write == Some(cancellation.id);
        if !cancel_read && !cancel_write {
            return;
        }
        if cancel_read {
            current_ids.read = None;
        }
        if cancel_write {
            current_ids.write = None;
        }
        if current_ids.read.is_none() && current_ids.write.is_none() {
            ids.remove(&cancellation.key);
        }
        drop(ids);

        let Some(fd_info) = fds.get_mut(&cancellation.key) else {
            return;
        };
        if cancel_read && let Some(waker) = fd_info.read_waker.take() {
            displaced.push(waker);
        }
        if cancel_write && let Some(waker) = fd_info.write_waker.take() {
            displaced.push(waker);
        }
        let remaining = Interest {
            readable: fd_info.interest.readable && !cancel_read,
            writable: fd_info.interest.writable && !cancel_write,
            error: fd_info.interest.error,
        };
        let platform_result = self
            .platform
            .update_registration(cancellation.key.0 as RawFd, remaining);
        let mut platform_error = None;
        match platform_result {
            Ok(()) => {
                if remaining.readable || remaining.writable {
                    fd_info.interest = remaining;
                } else {
                    fds.remove(&cancellation.key);
                    self.generations
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner())
                        .remove(&cancellation.key);
                }
            }
            Err(failure) => {
                platform_error = Some(failure.into_error());
            }
        }
        drop(fds);
        drop(displaced);

        if let Some(error) = platform_error {
            self.publish_failure(error);
        } else if let Err(error) = self.platform.wake() {
            self.publish_failure(error);
        }
    }

    fn publish_failure(&self, error: io::Error) {
        let _ = self
            .driver_failure
            .publish(error, &self.running, &self.fds, || {
                self.generations
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .clear();
                self.clear_all();
            });
    }
}

/// Exact per-interest registration cancellation bound to its originating reactor.
pub(crate) struct WaiterCancellation {
    state: Weak<WaiterCancellationState>,
    key: FdKey,
    id: u64,
    interest: Interest,
}

impl Drop for WaiterCancellation {
    fn drop(&mut self) {
        if let Some(state) = self.state.upgrade() {
            state.cancel(self);
        }
    }
}
