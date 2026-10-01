//! An AFD device handle bound to the completion port, and the two calls made
//! on it: start a poll request and cancel one.

use std::ffi::c_void;
use std::io;

use windows::Wdk::Foundation::OBJECT_ATTRIBUTES;
use windows::Wdk::Storage::FileSystem::{
    FILE_OPEN, NTCREATEFILE_CREATE_OPTIONS, NtCancelIoFileEx, NtCreateFile,
};
use windows::Wdk::System::IO::NtDeviceIoControlFile;
use windows::Win32::Foundation::{
    CloseHandle, HANDLE, NTSTATUS, RtlNtStatusToDosError, STATUS_NOT_FOUND, STATUS_PENDING,
    STATUS_SUCCESS, UNICODE_STRING,
};
use windows::Win32::Storage::FileSystem::{
    FILE_FLAGS_AND_ATTRIBUTES, FILE_SHARE_READ, FILE_SHARE_WRITE, SYNCHRONIZE,
    SetFileCompletionNotificationModes,
};
use windows::Win32::System::IO::IO_STATUS_BLOCK;
use windows::core::PWSTR;

use super::abi::{AfdPollInfo, IOCTL_AFD_POLL};
use super::completion_port::CompletionPort;

/// `FILE_SKIP_SET_EVENT_ON_HANDLE`: do not signal the file object on
/// completion, which a port-bound handle never waits on.
const SKIP_SET_EVENT_ON_HANDLE: u8 = 0x2;

/// NT path of the AFD endpoint used for polling.
const DEVICE_PATH: &str = r"\Device\Afd\Moirai";

/// Map an `NTSTATUS` to the `io::Error` of its Win32 equivalent.
pub(super) fn status_error(status: NTSTATUS) -> io::Error {
    // SAFETY: `RtlNtStatusToDosError` reads only its by-value argument.
    let code = unsafe { RtlNtStatusToDosError(status) };
    io::Error::from_raw_os_error(code.cast_signed())
}

/// A handle to the AFD driver, associated with one completion port.
pub(super) struct AfdDevice(HANDLE);

// SAFETY: the handle is a kernel object; `NtDeviceIoControlFile` and
// `NtCancelIoFileEx` on one handle are safe from any thread.
unsafe impl Send for AfdDevice {}
// SAFETY: as above, every method takes `&self`.
unsafe impl Sync for AfdDevice {}

impl AfdDevice {
    /// Open the AFD endpoint and bind it to `port` once; completions carry
    /// `key`.
    pub(super) fn open(port: &CompletionPort, key: usize) -> io::Result<Self> {
        let mut path: Vec<u16> = DEVICE_PATH.encode_utf16().collect();
        let bytes = u16::try_from(path.len() * 2).map_err(|_| io::ErrorKind::InvalidInput)?;
        let name = UNICODE_STRING {
            Length: bytes,
            MaximumLength: bytes,
            Buffer: PWSTR(path.as_mut_ptr()),
        };
        let attributes = OBJECT_ATTRIBUTES {
            Length: size_of::<OBJECT_ATTRIBUTES>() as u32,
            ObjectName: &raw const name,
            ..OBJECT_ATTRIBUTES::default()
        };
        let mut handle = HANDLE::default();
        let mut status_block = IO_STATUS_BLOCK::default();
        // SAFETY: `handle`, `status_block`, `attributes`, and the name buffer it
        // points to are live locals for the whole call; the kernel copies the
        // name and retains none of these pointers.
        let status = unsafe {
            NtCreateFile(
                &raw mut handle,
                SYNCHRONIZE,
                &raw const attributes,
                &raw mut status_block,
                None,
                FILE_FLAGS_AND_ATTRIBUTES(0),
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                FILE_OPEN,
                NTCREATEFILE_CREATE_OPTIONS(0),
                None,
                0,
            )
        };
        if status != STATUS_SUCCESS {
            return Err(status_error(status));
        }
        let device = Self(handle);
        // SAFETY: `handle` was just opened by NtCreateFile for the AFD device
        // (opened with no `FILE_FLAG_OVERLAPPED` restriction, as every AFD
        // handle is asynchronous) and belongs to no port.
        unsafe { port.bind(handle, key)? };
        // SAFETY: `handle` is open and owned by `device`.
        unsafe { SetFileCompletionNotificationModes(handle, SKIP_SET_EVENT_ON_HANDLE) }
            .map_err(io::Error::from)?;
        Ok(device)
    }

    /// Start a poll request whose completion packet carries `context`.
    ///
    /// # Safety
    ///
    /// `info` and `status_block` must stay at their addresses, untouched except
    /// by the kernel, until the completion packet naming `context` is dequeued.
    pub(super) unsafe fn poll(
        &self,
        info: *mut AfdPollInfo,
        status_block: *mut IO_STATUS_BLOCK,
        context: *const c_void,
    ) -> io::Result<()> {
        // SAFETY: the caller guarantees `info` and `status_block` are valid and
        // pinned until completion; the buffer serves as input and output.
        let status = unsafe {
            NtDeviceIoControlFile(
                self.0,
                None,
                None,
                Some(context),
                status_block,
                IOCTL_AFD_POLL,
                Some(info.cast_const().cast()),
                size_of::<AfdPollInfo>() as u32,
                Some(info.cast()),
                size_of::<AfdPollInfo>() as u32,
            )
        };
        // A synchronous success still queues a completion packet, so both
        // outcomes leave the request owned by the kernel until dequeued.
        if status == STATUS_SUCCESS || status == STATUS_PENDING {
            Ok(())
        } else {
            Err(status_error(status))
        }
    }

    /// Request cancellation of the poll issued with `status_block`. A request
    /// that already completed is not an error; its packet is still queued.
    ///
    /// # Safety
    ///
    /// `status_block` must be the block of a request on this device whose
    /// completion packet has not yet been dequeued.
    pub(super) unsafe fn cancel(&self, status_block: *const IO_STATUS_BLOCK) -> io::Result<()> {
        let mut cancel_block = IO_STATUS_BLOCK::default();
        // SAFETY: the caller guarantees `status_block` is live; `cancel_block`
        // is a local the kernel fills before returning.
        let status = unsafe { NtCancelIoFileEx(self.0, Some(status_block), &raw mut cancel_block) };
        if status == STATUS_SUCCESS || status == STATUS_NOT_FOUND {
            Ok(())
        } else {
            Err(status_error(status))
        }
    }
}

impl Drop for AfdDevice {
    fn drop(&mut self) {
        // SAFETY: the handle is owned by this value and closed once.
        let closed = unsafe { CloseHandle(self.0) };
        debug_assert!(closed.is_ok(), "an owned AFD handle closes");
    }
}
