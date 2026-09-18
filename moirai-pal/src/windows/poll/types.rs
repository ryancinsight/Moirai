//! `WsaPollReactor` state: the registration table and the reused `WSAPoll`
//! snapshot buffers.

use std::net::UdpSocket;
use std::ops::{Deref, DerefMut};
use std::sync::{Mutex, MutexGuard};

use windows::Win32::Networking::WinSock::WSAPOLLFD;

use crate::reactor::registration::{RegistrationGeneration, RegistrationTable};
use crate::reactor::socket_owner::SocketLease;

/// `WSAPoll`-based readiness reactor.
pub struct WsaPollReactor {
    /// Registered sockets and the generation that distinguishes reused raw
    /// `SOCKET` values.
    pub(super) registrations: Mutex<RegistrationTable<usize>>,
    /// Loopback UDP socket used to interrupt a blocking `WSAPoll`: `wake()` sends
    /// a datagram to `wake_addr`, making this socket readable so the poll returns
    /// promptly (e.g. after a new registration or on shutdown).
    pub(super) wake: UdpSocket,
    pub(super) wake_addr: std::net::SocketAddr,
    /// Reused `WSAPoll` snapshot, so the hot poll loop does not allocate fd or
    /// generation arrays per iteration. Lock order: `poll_buffer` before
    /// `registrations`; every other path takes at most `registrations`.
    pub(super) poll_buffer: Mutex<PollBuffer>,
    /// Reused strong-owner storage, kept separate so final owner release never
    /// occurs while the poll snapshot lock is held.
    pub(super) lease_buffer: Mutex<Vec<SocketLease>>,
}

#[derive(Default)]
pub(super) struct PollBuffer {
    pub(super) fds: Vec<WSAPOLLFD>,
    pub(super) generations: Vec<RegistrationGeneration>,
}

pub(super) struct PollSnapshot<'a> {
    lease_source: &'a Mutex<Vec<SocketLease>>,
    buffer: Option<MutexGuard<'a, PollBuffer>>,
    pub(super) leases: Vec<SocketLease>,
}

impl<'a> PollSnapshot<'a> {
    pub(super) fn acquire(
        source: &'a Mutex<PollBuffer>,
        lease_source: &'a Mutex<Vec<SocketLease>>,
    ) -> Self {
        let buffer = lock_mutex(source);
        let leases = std::mem::take(&mut *lock_mutex(lease_source));
        Self {
            lease_source,
            buffer: Some(buffer),
            leases,
        }
    }

    pub(super) fn finish(mut self) {
        drop(self.buffer.take());
        self.leases.clear();
        std::mem::swap(&mut *lock_mutex(self.lease_source), &mut self.leases);
    }
}

impl Deref for PollSnapshot<'_> {
    type Target = PollBuffer;

    fn deref(&self) -> &Self::Target {
        self.buffer
            .as_deref()
            .expect("poll snapshot owns its buffer until release")
    }
}

impl DerefMut for PollSnapshot<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer
            .as_deref_mut()
            .expect("poll snapshot owns its buffer until release")
    }
}

impl Drop for PollSnapshot<'_> {
    fn drop(&mut self) {
        drop(self.buffer.take());
        self.leases.clear();
    }
}

// SAFETY: all shared state is behind the `registrations` and `poll_buffer` `Mutex`es;
// the `wake` `UdpSocket` supports concurrent `send_to` (any thread) and `recv`
// (the poll thread), which winsock permits for UDP.
unsafe impl Send for WsaPollReactor {}
unsafe impl Sync for WsaPollReactor {}

pub(super) fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
