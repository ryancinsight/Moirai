use core::fmt;

/// IPC-specific error type to minimize dependencies
#[derive(Debug, Clone)]
pub enum IpcError {
    /// System error with error code
    SystemError(i32),
    /// Invalid argument
    InvalidArgument,
    /// Resource not found
    NotFound,
    /// Permission denied
    PermissionDenied,
}

impl fmt::Display for IpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IpcError::SystemError(code) => write!(f, "System error: {code}"),
            IpcError::InvalidArgument => write!(f, "Invalid argument"),
            IpcError::NotFound => write!(f, "Resource not found"),
            IpcError::PermissionDenied => write!(f, "Permission denied"),
        }
    }
}

impl core::error::Error for IpcError {}

/// Convert OS error to `IpcError`
#[cfg(unix)]
pub fn last_os_error() -> IpcError {
    unsafe { IpcError::SystemError(*libc::__errno_location()) }
}

/// Convert OS error to `IpcError`
#[cfg(windows)]
pub fn last_os_error() -> IpcError {
    extern "system" {
        fn GetLastError() -> u32;
    }
    // SAFETY: `GetLastError` takes no arguments and reads thread-local state.
    // justification: the Win32 error code is carried verbatim in an `i32` field;
    // the `u32 -> i32` reinterpretation preserves all bits (no value is lost).
    #[allow(clippy::cast_possible_wrap)]
    unsafe {
        IpcError::SystemError(GetLastError() as i32)
    }
}
