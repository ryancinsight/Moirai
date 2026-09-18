//! Shared waker instrumentation, injected-failure types, and descriptor
//! helpers used across reactor behavior tests.

use super::super::core::IoReactor;
#[cfg(any(unix, windows))]
use super::super::core::{FdInfo, FdKey};
#[cfg(windows)]
use super::super::waiter_cancellation::WaiterCancellation;
#[cfg(any(unix, windows))]
use crate::Interest;
#[cfg(any(unix, windows))]
use std::collections::HashMap;
#[cfg(any(unix, windows))]
use std::error::Error as _;
#[cfg(any(unix, windows))]
use std::fmt;
#[cfg(any(unix, windows))]
use std::net::UdpSocket;
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(windows)]
use std::os::windows::io::AsRawSocket;
use std::sync::atomic::Ordering;
#[cfg(any(unix, windows))]
use std::sync::atomic::{AtomicBool, AtomicUsize};
#[cfg(any(unix, windows))]
use std::sync::{Arc, Mutex, Weak};
#[cfg(any(unix, windows))]
use std::task::{Wake, Waker};

#[cfg(any(unix, windows))]
#[derive(Default)]
pub(super) struct WakeCount(pub(super) AtomicUsize);

#[cfg(any(unix, windows))]
impl Wake for WakeCount {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(any(unix, windows))]
pub(super) struct LockObservingWake {
    pub(super) count: AtomicUsize,
    pub(super) woke_while_locked: AtomicBool,
    pub(super) registrations: Weak<Mutex<HashMap<FdKey, FdInfo>>>,
}

#[cfg(windows)]
pub(super) struct DropObservation {
    pub(super) dropped: AtomicBool,
    pub(super) dropped_while_locked: AtomicBool,
}

#[cfg(windows)]
pub(super) struct DropObservingWake {
    pub(super) observation: Arc<DropObservation>,
    pub(super) registrations: Weak<Mutex<HashMap<FdKey, FdInfo>>>,
    pub(super) cancellation: Mutex<Option<WaiterCancellation>>,
}

#[cfg(windows)]
#[expect(
    clippy::manual_noop_waker,
    reason = "this test waker observes destruction rather than wake delivery"
)]
impl Wake for DropObservingWake {
    fn wake(self: Arc<Self>) {}
}

#[cfg(windows)]
impl Drop for DropObservingWake {
    fn drop(&mut self) {
        if self
            .registrations
            .upgrade()
            .is_some_and(|registrations| registrations.try_lock().is_err())
        {
            self.observation
                .dropped_while_locked
                .store(true, Ordering::Relaxed);
        }
        drop(
            self.cancellation
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .take(),
        );
        self.observation.dropped.store(true, Ordering::Relaxed);
    }
}

#[cfg(any(unix, windows))]
impl Wake for LockObservingWake {
    fn wake(self: Arc<Self>) {
        self.record();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.record();
    }
}

#[cfg(any(unix, windows))]
impl LockObservingWake {
    fn record(&self) {
        self.count.fetch_add(1, Ordering::Relaxed);
        if self
            .registrations
            .upgrade()
            .is_some_and(|registrations| registrations.try_lock().is_err())
        {
            self.woke_while_locked.store(true, Ordering::Relaxed);
        }
    }
}

#[cfg(any(unix, windows))]
#[derive(Debug)]
pub(super) struct InjectedDriverFailure(pub(super) u32);

#[cfg(any(unix, windows))]
impl fmt::Display for InjectedDriverFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "injected driver failure {}", self.0)
    }
}

#[cfg(any(unix, windows))]
impl std::error::Error for InjectedDriverFailure {}

#[cfg(any(unix, windows))]
pub(super) struct ReentrantWake {
    pub(super) count: AtomicUsize,
    pub(super) reactor: Weak<IoReactor>,
    pub(super) fd: usize,
    pub(super) registration_error: Mutex<Option<std::io::Error>>,
}

#[cfg(any(unix, windows))]
impl Wake for ReentrantWake {
    fn wake(self: Arc<Self>) {
        self.record();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.record();
    }
}

#[cfg(any(unix, windows))]
impl ReentrantWake {
    fn record(&self) {
        self.count.fetch_add(1, Ordering::Relaxed);
        let reactor = self.reactor.upgrade().expect("reactor remains live");
        let error = reactor
            .register_waker(
                key_to_raw(self.fd),
                Interest::READABLE,
                Waker::noop().clone(),
            )
            .expect_err("terminal driver rejects reentrant registration");
        *self
            .registration_error
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = Some(error);
    }
}

#[cfg(unix)]
pub(super) fn socket_to_raw(socket: &UdpSocket) -> crate::RawFd {
    socket.as_raw_fd()
}

#[cfg(unix)]
pub(super) fn key_to_raw(key: usize) -> crate::RawFd {
    i32::try_from(key).expect("Unix descriptor fits i32")
}

#[cfg(windows)]
pub(super) fn socket_to_raw(socket: &UdpSocket) -> crate::RawFd {
    socket.as_raw_socket() as crate::RawFd
}

#[cfg(windows)]
pub(super) fn key_to_raw(key: usize) -> crate::RawFd {
    key as crate::RawFd
}

#[cfg(any(unix, windows))]
pub(super) fn assert_injected_driver_source(error: &std::io::Error, expected: u32) {
    assert_eq!(error.kind(), std::io::ErrorKind::ConnectionAborted);
    let mut source = error.source();
    while let Some(cause) = source {
        if let Some(retained) = cause.downcast_ref::<std::io::Error>()
            && let Some(injected) = retained
                .get_ref()
                .and_then(|inner| inner.downcast_ref::<InjectedDriverFailure>())
        {
            assert_eq!(injected.0, expected);
            return;
        }
        if let Some(injected) = cause.downcast_ref::<InjectedDriverFailure>() {
            assert_eq!(injected.0, expected);
            return;
        }
        source = cause.source();
    }
    panic!("retained driver error must preserve its typed source");
}

#[cfg(any(unix, windows))]
pub(super) fn assert_direct_injected_error(error: &std::io::Error, expected: u32) {
    assert_eq!(error.kind(), std::io::ErrorKind::ConnectionAborted);
    let injected = error
        .get_ref()
        .and_then(|source| source.downcast_ref::<InjectedDriverFailure>())
        .expect("direct iteration error preserves its typed payload");
    assert_eq!(injected.0, expected);
}

#[cfg(windows)]
pub(super) fn bind_reusing_socket(fd: crate::RawFd) -> UdpSocket {
    const REUSE_ATTEMPTS: usize = 256;
    let mut held = Vec::with_capacity(REUSE_ATTEMPTS);
    for _ in 0..REUSE_ATTEMPTS {
        let socket = UdpSocket::bind("127.0.0.1:0").expect("replacement socket bind");
        if socket_to_raw(&socket) == fd {
            return socket;
        }
        held.push(socket);
    }
    panic!("Winsock did not reuse the retired socket value within {REUSE_ATTEMPTS} allocations");
}
