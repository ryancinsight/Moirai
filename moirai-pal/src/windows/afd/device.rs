//! An AFD device handle bound to the completion port, and the two calls made
//! on it: start a poll request and cancel one.

use std::ffi::c_void;
use std::io;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};

use windows::Wdk::Foundation::OBJECT_ATTRIBUTES;
use windows::Wdk::Storage::FileSystem::{
    FILE_OPEN, NTCREATEFILE_CREATE_OPTIONS, NtCancelIoFileEx, NtCreateFile,
};
use windows::Wdk::System::IO::NtDeviceIoControlFile;
use windows::Win32::Foundation::{
    HANDLE, NTSTATUS, RtlNtStatusToDosError, STATUS_NOT_FOUND, STATUS_SUCCESS, UNICODE_STRING,
};
use windows::Win32::Storage::FileSystem::{
    FILE_FLAGS_AND_ATTRIBUTES, FILE_SHARE_READ, FILE_SHARE_WRITE, SYNCHRONIZE,
    SetFileCompletionNotificationModes,
};
use windows::Win32::System::IO::IO_STATUS_BLOCK;
use windows::core::PWSTR;

use super::abi::{AfdPollInfo, IOCTL_AFD_POLL};
use super::completion_port::CompletionPort;
use crate::Event;

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

/// Outcome of starting a poll, from the status `NtDeviceIoControlFile`
/// returned.
///
/// The handle is asynchronous and not set to skip the port on success, so the
/// kernel queues one completion packet for every request that returns
/// success, informational, or warning severity (`STATUS_PENDING`, a
/// synchronous `STATUS_SUCCESS`, `STATUS_BUFFER_OVERFLOW`, and the like): the
/// request stays owned by the kernel until that packet is dequeued. Only an
/// error-severity status (the top two bits set) guarantees no packet and no
/// further kernel access, so only it lets the caller release the slot at
/// once.
pub(super) fn started(status: NTSTATUS) -> io::Result<()> {
    if status.0.cast_unsigned() >> 30 == SEVERITY_ERROR {
        Err(status_error(status))
    } else {
        Ok(())
    }
}

/// Severity field (bits 30 and 31) of an error status.
const SEVERITY_ERROR: u32 = 3;

/// `Ok` for an `NT_SUCCESS` status (severity success or informational, the
/// non-negative values), otherwise the error of the Win32 equivalent.
pub(super) fn nt_result(status: NTSTATUS) -> io::Result<()> {
    if status.0 >= 0 {
        Ok(())
    } else {
        Err(status_error(status))
    }
}

/// A completed poll's outcome: its readiness, or the error of a failure status.
pub(super) fn finished(status: NTSTATUS, readiness: Event) -> io::Result<Event> {
    nt_result(status).map(|()| readiness)
}

/// A handle to the AFD driver, associated with one completion port. The owning
/// handle is `Send + Sync`: `NtDeviceIoControlFile` and `NtCancelIoFileEx` on
/// one handle are safe from any thread.
pub(super) struct AfdDevice(OwnedHandle);

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
        // SAFETY: `NtCreateFile` returned a new handle that nothing else owns
        // or closes.
        let device = Self(unsafe { OwnedHandle::from_raw_handle(handle.0 as _) });
        // SAFETY: `handle` was just opened by NtCreateFile for the AFD device
        // (opened with no `FILE_FLAG_OVERLAPPED` restriction, as every AFD
        // handle is asynchronous) and belongs to no port.
        unsafe { port.bind(handle, key)? };
        // SAFETY: `handle` is open and owned by `device`.
        unsafe { SetFileCompletionNotificationModes(handle, SKIP_SET_EVENT_ON_HANDLE) }
            .map_err(io::Error::from)?;
        Ok(device)
    }

    fn raw(&self) -> HANDLE {
        HANDLE(self.0.as_raw_handle() as _)
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
                self.raw(),
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
        started(status)
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
        let status =
            unsafe { NtCancelIoFileEx(self.raw(), Some(status_block), &raw mut cancel_block) };
        if status == STATUS_SUCCESS || status == STATUS_NOT_FOUND {
            Ok(())
        } else {
            Err(status_error(status))
        }
    }
}
