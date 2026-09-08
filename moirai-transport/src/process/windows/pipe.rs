//! Anonymous pipe creation and handle inheritance control.

use std::{
    ffi::c_void,
    os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle},
    ptr::null_mut,
};

use super::super::{ProcessError, ProcessResult};
use super::ffi::{self, SecurityAttributes};
use super::status::check;

pub(super) fn pipe() -> ProcessResult<(OwnedHandle, OwnedHandle)> {
    let attributes = SecurityAttributes {
        length: u32::try_from(size_of::<SecurityAttributes>())
            .map_err(|_| ProcessError::InvalidSpecification)?,
        descriptor: null_mut(),
        inherit: 1,
    };
    let (mut reader, mut writer) = (null_mut(), null_mut());
    // SAFETY: valid output pointers; default ACL, inheritable anonymous pipe.
    check(
        unsafe { ffi::CreatePipe(&mut reader, &mut writer, &attributes, 0) },
        ProcessError::SpawnFailed,
    )?;
    // SAFETY: success returns distinct owned valid pipe handles.
    Ok(unsafe {
        (
            OwnedHandle::from_raw_handle(reader),
            OwnedHandle::from_raw_handle(writer),
        )
    })
}
pub(super) fn remove_inheritance(handle: &OwnedHandle) -> ProcessResult<()> {
    // SAFETY: handle is live; HANDLE_FLAG_INHERIT is mask 1, cleared to zero.
    check(
        unsafe { ffi::SetHandleInformation(handle.as_raw_handle(), 1, 0) },
        ProcessError::SpawnFailed,
    )
}
pub(super) fn duplicate(source: *mut c_void) -> ProcessResult<OwnedHandle> {
    // SAFETY: pseudo handle identifies this process and is not closed.
    let current = unsafe { ffi::GetCurrentProcess() };
    let mut result = null_mut();
    // SAFETY: source is a live standard handle borrowed for the call; output
    // receives an independent owned inheritable duplicate, same access (2).
    check(
        unsafe { ffi::DuplicateHandle(current, source, current, &mut result, 0, 1, 2) },
        ProcessError::SpawnFailed,
    )?;
    // SAFETY: successful DuplicateHandle returns an owned valid handle.
    Ok(unsafe { OwnedHandle::from_raw_handle(result) })
}
