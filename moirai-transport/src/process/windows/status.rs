//! Win32 return-code checking and last-error capture.

use super::super::{ProcessError, ProcessResult};

pub(super) fn check(result: i32, operation: ProcessError) -> ProcessResult<()> {
    if result == 0 {
        Err(os_error(operation))
    } else {
        Ok(())
    }
}
pub(super) fn os_error(operation: ProcessError) -> ProcessError {
    ProcessError::OperatingSystem {
        operation: operation.operation(),
        code: std::io::Error::last_os_error().raw_os_error(),
    }
}
