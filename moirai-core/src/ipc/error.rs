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

/// Convert the calling thread's last operating-system error to `IpcError`.
#[cfg(any(unix, windows))]
pub fn last_os_error() -> IpcError {
    let code = std::io::Error::last_os_error()
        .raw_os_error()
        .expect("invariant: last_os_error captures a raw operating-system error code");
    IpcError::SystemError(code)
}
