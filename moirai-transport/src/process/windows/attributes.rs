//! The process-thread attribute list carrying job and inherited handles.

use std::{ffi::c_void, marker::PhantomData, ptr::null_mut};

use super::super::{ProcessError, ProcessResult};
use super::ffi;
use super::status::{check, os_error};

pub(super) struct Attributes<'a> {
    storage: Vec<usize>,
    _values: PhantomData<&'a [*mut c_void]>,
}
impl<'a> Attributes<'a> {
    pub(super) fn new(
        jobs: &'a [*mut c_void],
        inherited: &'a [*mut c_void],
    ) -> ProcessResult<Self> {
        let mut size = 0;
        // SAFETY: documented sizing probe uses null list and a valid size output.
        let probe = unsafe { ffi::InitializeProcThreadAttributeList(null_mut(), 2, 0, &mut size) };
        if probe != 0 || std::io::Error::last_os_error().raw_os_error() != Some(122) {
            return Err(os_error(ProcessError::SpawnFailed));
        }
        let mut storage = vec![0; size.div_ceil(size_of::<usize>())];
        // SAFETY: pointer-aligned allocation covers the probed size and is
        // retained until DeleteProcThreadAttributeList.
        check(
            unsafe {
                ffi::InitializeProcThreadAttributeList(storage.as_mut_ptr().cast(), 2, 0, &mut size)
            },
            ProcessError::SpawnFailed,
        )?;
        let mut list = Self {
            storage,
            _values: PhantomData,
        };
        list.insert(0x0002_000d, jobs)?;
        list.insert(0x0002_0002, inherited)?;
        Ok(list)
    }
    pub(super) fn pointer(&mut self) -> *mut c_void {
        self.storage.as_mut_ptr().cast()
    }
    fn insert(&mut self, kind: usize, values: &'a [*mut c_void]) -> ProcessResult<()> {
        // SAFETY: initialized attribute list, recognized handle-array key, and
        // borrowed values remain alive until list destruction by its lifetime.
        check(
            unsafe {
                ffi::UpdateProcThreadAttribute(
                    self.pointer(),
                    0,
                    kind,
                    values.as_ptr().cast(),
                    size_of_val(values),
                    null_mut(),
                    null_mut(),
                )
            },
            ProcessError::SpawnFailed,
        )
    }
}
impl Drop for Attributes<'_> {
    fn drop(&mut self) {
        // SAFETY: this exclusively owned list initialized successfully; its
        // backing allocation and borrowed values still outlive this destructor.
        unsafe {
            ffi::DeleteProcThreadAttributeList(self.pointer());
        }
    }
}
