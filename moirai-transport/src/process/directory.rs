//! Owned child-only directory admission at the native process boundary.

use std::path::Path;

use super::{ProcessError, ProcessOperation, ProcessResult};

pub(super) fn validate(path: &Path) -> ProcessResult<()> {
    if !path.is_absolute()
        || path.as_os_str().is_empty()
        || path.as_os_str().as_encoded_bytes().contains(&0)
    {
        return Err(ProcessError::InvalidSpecification);
    }
    validate_representation(path)?;
    let metadata = std::fs::metadata(path).map_err(|error| ProcessError::OperatingSystem {
        operation: ProcessOperation::Spawn,
        code: error.raw_os_error(),
    })?;
    if !metadata.is_dir() {
        return Err(ProcessError::InvalidSpecification);
    }
    Ok(())
}

#[cfg(unix)]
fn validate_representation(path: &Path) -> ProcessResult<()> {
    use std::os::unix::ffi::OsStrExt;
    // PATH_MAX includes the terminating byte for native pathname arguments.
    let capacity =
        usize::try_from(libc::PATH_MAX).map_err(|_| ProcessError::InvalidSpecification)?;
    if path.as_os_str().as_bytes().len() >= capacity {
        return Err(ProcessError::InvalidSpecification);
    }
    Ok(())
}

// SetCurrentDirectory/CreateProcessW require MAX_PATH including the trailing
// separator and terminator. Reference:
// https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setcurrentdirectory
#[cfg(windows)]
const DIRECTORY_CAPACITY: usize = 260;

#[cfg(windows)]
fn validate_representation(path: &Path) -> ProcessResult<()> {
    use std::os::windows::ffi::OsStrExt;
    let units = path.as_os_str().encode_wide().count();
    let separator = usize::from(
        !path
            .as_os_str()
            .encode_wide()
            .last()
            .is_some_and(|unit| matches!(unit, 47 | 92)),
    );
    if units + separator + 1 > DIRECTORY_CAPACITY {
        return Err(ProcessError::InvalidSpecification);
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn validate_representation(_path: &Path) -> ProcessResult<()> {
    // No process directory encoding exists on this target. Reject an explicit
    // request before invoking its process backend rather than ignore it.
    Err(ProcessError::InvalidSpecification)
}

#[cfg(windows)]
pub(super) fn encode(path: &Path) -> ProcessResult<Vec<u16>> {
    use std::os::windows::ffi::OsStrExt;
    validate_representation(path)?;
    let units = path.as_os_str().encode_wide().count();
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(units + 1)
        .map_err(|_| ProcessError::InvalidSpecification)?;
    encoded.extend(path.as_os_str().encode_wide());
    encoded.push(0);
    Ok(encoded)
}

#[cfg(test)]
mod tests;
