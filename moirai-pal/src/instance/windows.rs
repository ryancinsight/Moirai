//! Windows instance claims: the first instance of a session-scoped pipe.

use std::fs::File;
use std::io::{self, ErrorKind, Read};
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{
    ERROR_ACCESS_DENIED, ERROR_BROKEN_PIPE, ERROR_FILE_NOT_FOUND, ERROR_NO_DATA, ERROR_PIPE_BUSY,
    ERROR_PIPE_CONNECTED, ERROR_PIPE_LISTENING, GENERIC_WRITE, GetLastError, HANDLE,
};
use windows::Win32::Storage::FileSystem::{
    CreateFileW, FILE_FLAG_FIRST_PIPE_INSTANCE, FILE_FLAGS_AND_ATTRIBUTES, FILE_SHARE_MODE,
    OPEN_EXISTING, PIPE_ACCESS_INBOUND, ReadFile, SECURITY_IDENTIFICATION, SECURITY_SQOS_PRESENT,
};
use windows::Win32::System::Pipes::{
    ConnectNamedPipe, CreateNamedPipeW, DisconnectNamedPipe, PIPE_NOWAIT, PIPE_READMODE_BYTE,
    PIPE_REJECT_REMOTE_CLIENTS, PIPE_TYPE_BYTE, WaitNamedPipeW,
};
use windows::Win32::System::RemoteDesktop::ProcessIdToSessionId;
use windows::Win32::System::Threading::GetCurrentProcessId;
use windows::core::PCWSTR;

use super::{
    INSTANCE_IO_TIMEOUT, InstanceName, MAX_INSTANCE_MESSAGE_BYTES, read_frame, write_frame,
};

/// Pause between polls of a non-blocking pipe or a pipe being recreated.
const POLL_INTERVAL: Duration = Duration::from_millis(1);

pub(super) enum Role {
    Primary(Primary),
    Secondary(Secondary),
}

#[derive(Debug)]
pub(super) struct Primary {
    pipe: OwnedHandle,
}

#[derive(Debug)]
pub(super) struct Secondary {
    file: File,
}

pub(super) fn claim(name: &InstanceName) -> io::Result<Role> {
    let path = pipe_path(name)?;
    let deadline = Instant::now() + INSTANCE_IO_TIMEOUT;
    loop {
        if let Some(pipe) = create_first_instance(&path)? {
            return Ok(Role::Primary(Primary { pipe }));
        }
        match open_client(&path, deadline)? {
            Some(file) => return Ok(Role::Secondary(Secondary { file })),
            // The holder exited between the two calls; claim again.
            None if Instant::now() < deadline => std::thread::sleep(POLL_INTERVAL),
            None => {
                return Err(io::Error::new(
                    ErrorKind::TimedOut,
                    "instance name changed hands repeatedly",
                ));
            }
        }
    }
}

/// `\\.\pipe\moirai-instance-<session>-<name>`, NUL-terminated.
fn pipe_path(name: &InstanceName) -> io::Result<Vec<u16>> {
    let mut session = 0;
    // SAFETY: `session` is writable storage for the synchronous call.
    unsafe { ProcessIdToSessionId(GetCurrentProcessId(), &mut session) }.map_err(os_error)?;
    let path = format!(r"\\.\pipe\moirai-instance-{session}-{}", name.as_str());
    Ok(path.encode_utf16().chain(std::iter::once(0)).collect())
}

/// Creates the first pipe instance, or returns `None` when it already exists.
fn create_first_instance(path: &[u16]) -> io::Result<Option<OwnedHandle>> {
    let buffer = u32::try_from(MAX_INSTANCE_MESSAGE_BYTES * 4).unwrap_or(u32::MAX);
    // SAFETY: `path` is a NUL-terminated UTF-16 buffer that outlives the call.
    let handle = unsafe {
        CreateNamedPipeW(
            PCWSTR(path.as_ptr()),
            PIPE_ACCESS_INBOUND | FILE_FLAG_FIRST_PIPE_INSTANCE,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_NOWAIT | PIPE_REJECT_REMOTE_CLIENTS,
            1,
            0,
            buffer,
            0,
            None,
        )
    };
    if handle.is_invalid() {
        // SAFETY: reads the calling thread's last error.
        let error = unsafe { GetLastError() };
        if error == ERROR_ACCESS_DENIED || error == ERROR_PIPE_BUSY {
            return Ok(None);
        }
        return Err(io::Error::from_raw_os_error(error.0 as i32));
    }
    // SAFETY: the handle was just created and is owned by nothing else.
    Ok(Some(unsafe { OwnedHandle::from_raw_handle(handle.0) }))
}

/// Opens the pipe for writing, waiting while the holder serves another
/// client; `None` when no holder exists any more.
fn open_client(path: &[u16], deadline: Instant) -> io::Result<Option<File>> {
    loop {
        // SAFETY: `path` is NUL-terminated and outlives the call. The quality
        // of service flags stop the pipe server from impersonating this user.
        let opened = unsafe {
            CreateFileW(
                PCWSTR(path.as_ptr()),
                GENERIC_WRITE.0,
                FILE_SHARE_MODE(0),
                None,
                OPEN_EXISTING,
                FILE_FLAGS_AND_ATTRIBUTES(SECURITY_SQOS_PRESENT.0 | SECURITY_IDENTIFICATION.0),
                None,
            )
        };
        match opened {
            // SAFETY: the handle was just opened and is owned by nothing else.
            Ok(handle) => {
                return Ok(Some(File::from(unsafe {
                    OwnedHandle::from_raw_handle(handle.0)
                })));
            }
            Err(error) if error.code() == ERROR_FILE_NOT_FOUND.to_hresult() => return Ok(None),
            Err(error) if error.code() == ERROR_PIPE_BUSY.to_hresult() => {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return Err(io::Error::new(
                        ErrorKind::TimedOut,
                        "instance primary is busy",
                    ));
                }
                let wait = u32::try_from(remaining.as_millis())
                    .unwrap_or(u32::MAX)
                    .max(1);
                // SAFETY: `path` is NUL-terminated and outlives the call.
                let _ = unsafe { WaitNamedPipeW(PCWSTR(path.as_ptr()), wait) };
            }
            Err(error) => return Err(os_error(error)),
        }
    }
}

impl Primary {
    pub(super) fn try_receive(&mut self) -> io::Result<Option<Vec<u8>>> {
        let handle = HANDLE(self.pipe.as_raw_handle());
        // SAFETY: the handle is this primary's live, non-blocking pipe.
        match unsafe { ConnectNamedPipe(handle, None) } {
            Err(error) if error.code() == ERROR_PIPE_LISTENING.to_hresult() => return Ok(None),
            // A sender that already wrote and closed reports "no data" here,
            // but its message stays buffered until read.
            Ok(()) => {}
            Err(error)
                if error.code() == ERROR_PIPE_CONNECTED.to_hresult()
                    || error.code() == ERROR_NO_DATA.to_hresult() => {}
            Err(error) => return Err(os_error(error)),
        }
        let mut reader = PipeReader {
            handle,
            deadline: Instant::now() + INSTANCE_IO_TIMEOUT,
            received: 0,
        };
        let message = read_frame(&mut reader);
        // SAFETY: as above; the pipe returns to listening for the next client.
        unsafe { DisconnectNamedPipe(handle) }.map_err(os_error)?;
        match message {
            Ok(message) => Ok(Some(message)),
            // A client that connected and left without writing sent nothing.
            Err(error) if error.kind() == ErrorKind::UnexpectedEof && reader.received == 0 => {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }
}

impl Secondary {
    pub(super) fn send(mut self, message: &[u8]) -> io::Result<()> {
        write_frame(&mut self.file, message)
    }
}

/// Reads a non-blocking pipe, waiting for data until a deadline.
struct PipeReader {
    handle: HANDLE,
    deadline: Instant,
    received: usize,
}

impl Read for PipeReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        loop {
            let mut read = 0;
            // SAFETY: `buffer` and `read` are writable for the synchronous
            // call on the primary's live pipe handle.
            match unsafe { ReadFile(self.handle, Some(buffer), Some(&mut read), None) } {
                Ok(()) if read > 0 => {
                    self.received += read as usize;
                    return Ok(read as usize);
                }
                Ok(()) => {}
                Err(error) if error.code() == ERROR_BROKEN_PIPE.to_hresult() => return Ok(0),
                Err(error) if error.code() == ERROR_NO_DATA.to_hresult() => {}
                Err(error) => return Err(os_error(error)),
            }
            if Instant::now() >= self.deadline {
                return Err(io::Error::new(
                    ErrorKind::TimedOut,
                    "instance sender stalled",
                ));
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }
}

fn os_error(error: windows::core::Error) -> io::Error {
    io::Error::from(error)
}
