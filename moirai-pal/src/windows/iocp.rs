//! Windows I/O completion port reactor.

use std::{
    collections::HashMap,
    ffi::c_void,
    io, ptr,
    sync::{Mutex, MutexGuard},
    time::Duration,
};

use windows::Win32::{
    Foundation::{CloseHandle, GetLastError, HANDLE, INVALID_HANDLE_VALUE, WAIT_TIMEOUT},
    System::IO::{
        CreateIoCompletionPort, GetQueuedCompletionStatus, PostQueuedCompletionStatus, OVERLAPPED,
    },
};

use crate::{Event, Interest, RawFd, Reactor};

const WAKE_COMPLETION_KEY: usize = usize::MAX;
const INFINITE_TIMEOUT_MS: u32 = u32::MAX;

/// IOCP-backed reactor for Windows overlapped I/O handles.
pub struct IocpReactor {
    port: HANDLE,
    interests: Mutex<HashMap<usize, Interest>>,
}

// Safety: `HANDLE` refers to a kernel-managed completion port. All access to
// mutable reactor metadata is synchronized, and IOCP operations are thread-safe
// kernel calls for concurrent producers and consumers.
unsafe impl Send for IocpReactor {}

// Safety: Shared references only issue thread-safe IOCP syscalls or lock the
// interest map before mutation.
unsafe impl Sync for IocpReactor {}

impl IocpReactor {
    /// Create a new completion port.
    pub fn new() -> io::Result<Self> {
        // Safety: `INVALID_HANDLE_VALUE` with a null existing port creates a new
        // completion port per the Windows API contract. No aliasing is created.
        let port = unsafe { CreateIoCompletionPort(INVALID_HANDLE_VALUE, HANDLE::default(), 0, 0) }
            .map_err(windows_error_to_io)?;

        Ok(Self {
            port,
            interests: Mutex::new(HashMap::new()),
        })
    }
}

impl Reactor for IocpReactor {
    fn register_fd(&self, fd: RawFd, interest: Interest) -> io::Result<()> {
        let handle = HANDLE(fd);
        if handle.is_invalid() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid Windows handle for IOCP registration",
            ));
        }

        let completion_key = fd as usize;

        // Safety: The handle is supplied by the caller as a Windows raw handle.
        // Associating it with this completion port is the documented IOCP
        // registration operation; ownership of the file handle is not transferred.
        unsafe { CreateIoCompletionPort(handle, self.port, completion_key, 0) }
            .map_err(windows_error_to_io)?;

        lock_mutex(&self.interests).insert(completion_key, interest);
        Ok(())
    }

    fn unregister_fd(&self, fd: RawFd) -> io::Result<()> {
        lock_mutex(&self.interests).remove(&(fd as usize));
        Ok(())
    }

    fn poll_events(&self, timeout: Option<Duration>) -> io::Result<Vec<Event>> {
        let mut bytes_transferred = 0u32;
        let mut completion_key = 0usize;
        let mut overlapped: *mut OVERLAPPED = ptr::null_mut();
        let timeout_ms = timeout.map_or(INFINITE_TIMEOUT_MS, duration_to_millis);

        // Safety: All output pointers refer to stack locals that remain valid for
        // the duration of the call. `self.port` is a live completion port owned by
        // this reactor.
        let status = unsafe {
            GetQueuedCompletionStatus(
                self.port,
                &mut bytes_transferred,
                &mut completion_key,
                &mut overlapped,
                timeout_ms,
            )
        };

        if status.is_err() && overlapped.is_null() {
            // Safety: `GetLastError` reads the calling thread's last OS error.
            let error = unsafe { GetLastError() };
            if error.0 == WAIT_TIMEOUT.0 {
                return Ok(Vec::new());
            }

            return Err(io::Error::last_os_error());
        }

        if completion_key == WAKE_COMPLETION_KEY {
            return Ok(Vec::new());
        }

        let interest = lock_mutex(&self.interests)
            .get(&completion_key)
            .copied()
            .unwrap_or(Interest::READ_WRITE);

        Ok(vec![Event {
            fd: completion_key as *mut c_void,
            readable: interest.readable,
            writable: interest.writable,
            error: status.is_err(),
            hangup: false,
        }])
    }

    fn wake(&self) -> io::Result<()> {
        // Safety: Posting a completion with a reserved key is the documented way
        // to wake threads blocked in `GetQueuedCompletionStatus`.
        unsafe { PostQueuedCompletionStatus(self.port, 0, WAKE_COMPLETION_KEY, None) }
            .map_err(windows_error_to_io)
    }
}

impl Drop for IocpReactor {
    fn drop(&mut self) {
        if !self.port.is_invalid() {
            // Safety: `self.port` is owned by this reactor and closed exactly once
            // during drop after all shared references have ended.
            let _ = unsafe { CloseHandle(self.port) };
        }
    }
}

fn duration_to_millis(duration: Duration) -> u32 {
    duration.as_millis().min(u128::from(u32::MAX)) as u32
}

fn windows_error_to_io(error: windows::core::Error) -> io::Error {
    io::Error::other(error.to_string())
}

fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
