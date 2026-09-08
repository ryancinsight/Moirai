//! Job-object creation carrying the drop policy's kill-on-close limit.

use std::{
    os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle},
    ptr::null,
};

use super::super::{ProcessDropPolicy, ProcessError, ProcessResult};
use super::ffi::{self, BasicLimits, ExtendedLimits};
use super::status::{check, os_error};

pub(super) fn create_job(policy: ProcessDropPolicy) -> ProcessResult<OwnedHandle> {
    // SAFETY: null attributes/name request a new non-inherited unnamed job.
    let raw = unsafe { ffi::CreateJobObjectW(null(), null()) };
    if raw.is_null() {
        return Err(os_error(ProcessError::SpawnFailed));
    }
    // SAFETY: CreateJobObjectW returned a new non-null owned handle.
    let job = unsafe { OwnedHandle::from_raw_handle(raw) };
    let flags = if policy == ProcessDropPolicy::TerminateOnDrop {
        0x2000
    } else {
        0
    };
    let limits = ExtendedLimits {
        basic: BasicLimits {
            flags,
            ..BasicLimits::default()
        },
        ..ExtendedLimits::default()
    };
    // SAFETY: repr(C) layout matches JOBOBJECT_EXTENDED_LIMIT_INFORMATION;
    // information class 9 selects it, and length is its exact ABI size.
    check(
        unsafe {
            ffi::SetInformationJobObject(
                job.as_raw_handle(),
                9,
                (&raw const limits).cast(),
                u32::try_from(size_of::<ExtendedLimits>())
                    .map_err(|_| ProcessError::InvalidSpecification)?,
            )
        },
        ProcessError::SpawnFailed,
    )?;
    Ok(job)
}
