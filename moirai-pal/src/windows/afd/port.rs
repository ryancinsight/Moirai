//! [`AfdPort`]: socket readiness delivered through one completion port.

use std::io;
use std::os::windows::io::RawSocket;
use std::sync::{Mutex, OnceLock, TryLockError};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Networking::WinSock::{
    SIO_BASE_HANDLE, SIO_BSP_HANDLE, SIO_BSP_HANDLE_POLL, SIO_BSP_HANDLE_SELECT, SOCKET,
    WSAGetLastError, WSAIoctl,
};
use windows::Win32::System::IO::OVERLAPPED_ENTRY;

use super::abi::AfdPollInfo;
use super::completion_port::CompletionPort;
use super::device::{AfdDevice, finished};
use super::slots::{Completion, SlotTable, Token};
use crate::{Event, Interest};

/// Completion key of every AFD handle bound to the port.
const DEVICE_KEY: usize = 0;
/// Slots served by one AFD handle, the grouping mio uses.
const GROUP_SIZE: usize = 32;
/// Packets dequeued per call.
const BATCH: usize = 256;
/// Liveness bound on draining cancelled polls when the port is dropped. The
/// kernel completes a cancelled poll without waiting on any peer, so the
/// bound is reached only if the driver misbehaves.
const DRAIN_DEADLINE: Duration = Duration::from_secs(5);

/// Readiness polling over one completion port.
///
/// [`arm`](Self::arm) starts a one-shot poll of a socket; [`poll`](Self::poll)
/// dequeues completions and reports each to a sink. At most one poll may be
/// armed per socket: the driver offers no way to modify a pending poll, so a
/// change of interest is [`cancel`](Self::cancel) followed by a new `arm`, and
/// the caller guarantees the old poll is gone or cancelled first.
pub struct AfdPort {
    port: CompletionPort,
    devices: Box<[OnceLock<AfdDevice>]>,
    table: SlotTable,
    entries: Mutex<Box<[OVERLAPPED_ENTRY]>>,
}

// SAFETY: the port and devices are kernel handles, the table is atomics plus
// records guarded by its state protocol, and `entries` (which holds raw
// pointers the kernel wrote) is reachable only through its mutex.
unsafe impl Send for AfdPort {}
// SAFETY: as above; every method takes `&self`.
unsafe impl Sync for AfdPort {}

impl AfdPort {
    /// Create a port able to hold `capacity` armed polls at once.
    ///
    /// # Errors
    ///
    /// Fails if `capacity` is zero or exceeds `u32::MAX`, the kernel refuses to
    /// create the port, or the AFD driver cannot be opened.
    pub fn new(capacity: usize) -> io::Result<Self> {
        if capacity == 0 || u32::try_from(capacity).is_err() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "AFD port capacity must be between 1 and u32::MAX",
            ));
        }
        let port = Self {
            port: CompletionPort::new()?,
            devices: (0..capacity.div_ceil(GROUP_SIZE))
                .map(|_| OnceLock::new())
                .collect(),
            table: SlotTable::new(capacity),
            entries: Mutex::new(vec![OVERLAPPED_ENTRY::default(); BATCH].into_boxed_slice()),
        };
        // Opening the first device now reports an unavailable driver at
        // construction instead of at the first arm.
        port.open_device(0)?;
        Ok(port)
    }

    /// Start a one-shot poll of `socket` for `interest`.
    ///
    /// # Errors
    ///
    /// `InvalidInput` when `interest` has neither direction, `QuotaExceeded`
    /// when every slot is armed, and the driver's error when the socket has no
    /// pollable base handle or the request is refused.
    pub fn arm(&self, socket: RawSocket, interest: Interest) -> io::Result<Token> {
        if !interest.readable && !interest.writable {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "an AFD poll needs readable or writable interest",
            ));
        }
        let base = base_socket(socket)?;
        let index = self.table.claim().ok_or_else(|| {
            io::Error::new(io::ErrorKind::QuotaExceeded, "every AFD poll slot is armed")
        })?;
        if self.device(index).is_none()
            && let Err(error) = self.open_device(index)
        {
            self.table.unclaim(index);
            return Err(error);
        }
        let device = self.device(index).expect("the device was opened above");
        let token = self.table.publish(
            index,
            socket as usize,
            AfdPollInfo::new(HANDLE(base as _), interest),
        );
        let request = self.table.request(index);
        // SAFETY: the record lives in the table, which neither moves nor frees
        // it before the packet naming `request.context` is dequeued (the
        // table outlives every poll; see `Drop`), and nothing else touches it
        // until then.
        let started = unsafe { device.poll(request.info, request.status_block, request.context) };
        match started {
            Ok(()) => Ok(token),
            Err(error) => {
                self.table.abandon(index);
                Err(error)
            }
        }
    }

    /// Cancel the poll named by `token`. A token whose poll already completed
    /// or was cancelled is ignored, and cannot affect a later poll of the
    /// reused slot. The cancelled poll's readiness is suppressed and its slot
    /// is reusable once [`poll`](Self::poll) has dequeued its packet.
    ///
    /// # Errors
    ///
    /// The driver's error when it refuses the cancellation.
    pub fn cancel(&self, token: Token) -> io::Result<()> {
        if !self.table.begin_cancel(token) {
            return Ok(());
        }
        let cancelled = match self.devices[token.index() / GROUP_SIZE].get() {
            // SAFETY: `begin_cancel` holds the slot, so its status block is the
            // live block of a request on this device whose packet is not yet
            // released.
            Some(device) => unsafe { device.cancel(self.table.status_block(token.index())) },
            None => Ok(()),
        };
        self.table.end_cancel(token.index());
        cancelled
    }

    /// Dequeue completions, waiting at most `timeout` (forever for `None`),
    /// and call `sink` for each poll that finished. Returns how many polls
    /// were dequeued, including cancelled ones whose readiness is suppressed;
    /// a wake or a timeout returns `0`.
    ///
    /// One thread polls at a time. A call made while another thread is inside
    /// `poll` returns `WouldBlock` at once instead of waiting behind it, so
    /// `timeout` bounds every call. The sink may [`arm`](Self::arm) and
    /// [`cancel`](Self::cancel) but must not call `poll`.
    ///
    /// # Errors
    ///
    /// `WouldBlock` when another thread is polling, or the driver's error when
    /// the wait fails. A poll the driver completed
    /// with a failure status is reported to the sink as `Err`.
    pub fn poll(
        &self,
        timeout: Option<Duration>,
        mut sink: impl FnMut(Token, io::Result<Event>),
    ) -> io::Result<usize> {
        let mut entries = match self.entries.try_lock() {
            Ok(entries) => entries,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => {
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "another thread is polling this AFD port",
                ));
            }
        };
        let dequeued = self.port.dequeue(&mut entries, timeout)?;
        let mut polls = 0;
        for entry in &entries[..dequeued] {
            if entry.lpOverlapped.is_null() {
                continue;
            }
            match self.table.complete(entry.lpOverlapped) {
                Completion::Foreign => {}
                Completion::Cancelled => polls += 1,
                Completion::Finished {
                    token,
                    status,
                    readiness,
                } => {
                    polls += 1;
                    sink(token, finished(status, readiness));
                }
            }
        }
        Ok(polls)
    }

    /// Number of polls whose completion packet has not been dequeued yet.
    #[must_use]
    pub fn armed(&self) -> usize {
        self.table.outstanding()
    }

    /// Make a blocked or future [`poll`](Self::poll) return promptly.
    ///
    /// # Errors
    ///
    /// The driver's error when the packet cannot be queued.
    pub fn wake(&self) -> io::Result<()> {
        self.port.post()
    }

    fn device(&self, index: usize) -> Option<&AfdDevice> {
        self.devices[index / GROUP_SIZE].get()
    }

    fn open_device(&self, index: usize) -> io::Result<()> {
        let device = AfdDevice::open(&self.port, DEVICE_KEY)?;
        // A concurrent opener that won the race keeps its device; ours is
        // closed when dropped.
        drop(self.devices[index / GROUP_SIZE].set(device));
        Ok(())
    }
}

impl Drop for AfdPort {
    fn drop(&mut self) {
        let mut refused = false;
        for index in 0..self.table.len() {
            if let Some(token) = self.table.armed_token(index) {
                refused |= self.cancel(token).is_err();
            }
        }
        if refused {
            // A poll the driver would not cancel may still be written, and may
            // never complete, so waiting for it is pointless and freeing its
            // record is unsound.
            self.table.leak();
            return;
        }
        let deadline = Instant::now() + DRAIN_DEADLINE;
        while self.table.outstanding() > 0 {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() || self.poll(Some(remaining), |_, _| {}).is_err() {
                // The kernel may still write to the records, so they are never
                // freed.
                self.table.leak();
                return;
            }
        }
    }
}

/// Base handle of `socket`, which AFD polls instead of a layered handle.
fn base_socket(socket: RawSocket) -> io::Result<usize> {
    let mut first_error = None;
    for (attempt, code) in [
        SIO_BASE_HANDLE,
        SIO_BSP_HANDLE_SELECT,
        SIO_BSP_HANDLE_POLL,
        SIO_BSP_HANDLE,
    ]
    .into_iter()
    .enumerate()
    {
        let mut base = SOCKET(0);
        let mut returned = 0_u32;
        // SAFETY: `base` and `returned` are live locals sized for the output
        // and byte count, and the ioctl takes no input buffer or completion.
        let result = unsafe {
            WSAIoctl(
                SOCKET(socket as usize),
                code,
                None,
                0,
                Some((&raw mut base).cast()),
                size_of::<SOCKET>() as u32,
                &raw mut returned,
                None,
                None,
            )
        };
        if result == 0 && (attempt == 0 || base.0 != socket as usize) {
            return Ok(base.0);
        }
        if attempt == 0 {
            // SAFETY: `WSAGetLastError` has no preconditions.
            first_error = Some(unsafe { WSAGetLastError() }.0);
        }
    }
    Err(io::Error::from_raw_os_error(
        first_error.unwrap_or_default(),
    ))
}
