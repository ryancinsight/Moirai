//! Owned I/O completion port.

use std::io;
use std::time::Duration;

use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_TIMEOUT};
use windows::Win32::System::IO::{
    CreateIoCompletionPort, GetQueuedCompletionStatusEx, OVERLAPPED_ENTRY,
    PostQueuedCompletionStatus,
};
use windows::Win32::System::Threading::INFINITE;
use windows::core::HRESULT;

/// Completion key of packets posted by [`CompletionPort::post`].
pub(super) const WAKE_KEY: usize = usize::MAX;

/// A completion port closed on drop.
pub(super) struct CompletionPort(HANDLE);

// SAFETY: a completion port handle is a kernel object usable from any thread;
// the wrapper exposes only calls the kernel serializes internally.
unsafe impl Send for CompletionPort {}
// SAFETY: as above, every method takes `&self` and calls thread-safe APIs.
unsafe impl Sync for CompletionPort {}

impl CompletionPort {
    /// Create a port with no associated handles.
    pub(super) fn new() -> io::Result<Self> {
        // SAFETY: an invalid file handle with no existing port asks the kernel
        // to create a new port; no caller memory is involved.
        let port = unsafe { CreateIoCompletionPort(HANDLE(-1isize as _), None, 0, 1) }
            .map_err(io::Error::from)?;
        Ok(Self(port))
    }

    /// Associate `handle` with this port; completions carry `key`.
    ///
    /// # Safety
    ///
    /// `handle` must be an open handle opened for overlapped I/O that is not
    /// yet associated with any completion port.
    pub(super) unsafe fn bind(&self, handle: HANDLE, key: usize) -> io::Result<()> {
        // SAFETY: the caller guarantees `handle` is open and unassociated, and
        // `self.0` is a live port.
        unsafe { CreateIoCompletionPort(handle, Some(self.0), key, 0) }
            .map(drop)
            .map_err(io::Error::from)
    }

    /// Queue a packet with no operation record, waking one blocked waiter.
    pub(super) fn post(&self) -> io::Result<()> {
        // SAFETY: `self.0` is a live port and a null overlapped pointer is
        // permitted for a posted packet.
        unsafe { PostQueuedCompletionStatus(self.0, 0, WAKE_KEY, None) }.map_err(io::Error::from)
    }

    /// Dequeue up to `entries.len()` packets, waiting at most `timeout`
    /// (forever for `None`, rounded up to whole milliseconds). Returns how many
    /// were written to the front of `entries`; `0` means the wait timed out.
    pub(super) fn dequeue(
        &self,
        entries: &mut [OVERLAPPED_ENTRY],
        timeout: Option<Duration>,
    ) -> io::Result<usize> {
        let millis = timeout.map_or(INFINITE, |wait| {
            let rounded = wait.as_nanos().div_ceil(1_000_000);
            u32::try_from(rounded).map_or(INFINITE - 1, |value| value.min(INFINITE - 1))
        });
        let mut removed = 0_u32;
        // SAFETY: `entries` is valid writable storage for its length, `removed`
        // outlives the call, and `self.0` is a live port.
        let result =
            unsafe { GetQueuedCompletionStatusEx(self.0, entries, &mut removed, millis, false) };
        match result {
            Ok(()) => Ok(removed as usize),
            Err(error) if error.code() == HRESULT::from_win32(WAIT_TIMEOUT.0) => Ok(0),
            Err(error) => Err(error.into()),
        }
    }
}

impl Drop for CompletionPort {
    fn drop(&mut self) {
        // SAFETY: the port handle is owned by this value and closed once.
        let closed = unsafe { CloseHandle(self.0) };
        debug_assert!(closed.is_ok(), "an owned completion port handle closes");
    }
}
