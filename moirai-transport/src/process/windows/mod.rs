//! Owned Windows handles for atomic job assignment and bounded cleanup.
//! Native declarations follow Windows SDK 10.0.26100.0.
mod command;
mod environment;
mod ffi;
use super::{ProcessDropPolicy, ProcessError, ProcessResult, ProcessSpec};
use ffi::{BasicLimits, ExtendedLimits, ProcessInformation, SecurityAttributes, StartupInfoEx};
use std::{
    ffi::c_void,
    fs::File,
    marker::PhantomData,
    os::windows::{
        io::{AsRawHandle, FromRawHandle, OwnedHandle},
        process::ExitStatusExt,
    },
    process::ExitStatus,
    ptr::{null, null_mut},
    time::Duration,
};

#[derive(Debug)]
pub(super) struct Process {
    process: OwnedHandle,
    // Closing this handle is the nonblocking last-resort tree cleanup.
    job: OwnedHandle,
    id: u32,
    pub stdin: Option<File>,
    pub stdout: Option<File>,
}
impl Process {
    pub fn spawn(spec: ProcessSpec, drop_policy: ProcessDropPolicy) -> ProcessResult<Self> {
        let (application, mut command_line) = command::encode(&spec.program, &spec.args)?;
        let environment = environment::encode(&spec)?;
        let job = create_job(drop_policy)?;
        let (child_input, stdin, child_output, stdout) = if spec.piped {
            let (read_input, write_input) = pipe()?;
            let (read_output, write_output) = pipe()?;
            remove_inheritance(&write_input)?;
            remove_inheritance(&read_output)?;
            (
                read_input,
                Some(File::from(write_input)),
                write_output,
                Some(File::from(read_output)),
            )
        } else {
            (
                duplicate(std::io::stdin().as_raw_handle())?,
                None,
                duplicate(std::io::stdout().as_raw_handle())?,
                None,
            )
        };
        let child_error = duplicate(std::io::stderr().as_raw_handle())?;
        let jobs = [job.as_raw_handle()];
        let inherited = [
            child_input.as_raw_handle(),
            child_output.as_raw_handle(),
            child_error.as_raw_handle(),
        ];
        let mut attributes = Attributes::new(&jobs, &inherited)?;
        let mut startup = StartupInfoEx::default();
        startup.base.size = u32::try_from(size_of::<StartupInfoEx>())
            .map_err(|_| ProcessError::InvalidSpecification)?;
        // STARTF_USESTDHANDLES; no STARTF_USESHOWWINDOW, preserving GUI startup.
        startup.base.flags = 0x100;
        startup.base.stdin = child_input.as_raw_handle();
        startup.base.stdout = child_output.as_raw_handle();
        startup.base.stderr = child_error.as_raw_handle();
        startup.attributes = attributes.pointer();
        let mut information = ProcessInformation::default();
        // SAFETY: buffers are live, mutable command line is zero-terminated;
        // inherited handles are owned, inheritable, and explicitly listed.
        // Attributes retain the borrowed arrays until CreateProcessW returns.
        // The unnamed non-inherited job is assigned atomically at creation.
        let created = unsafe {
            ffi::CreateProcessW(
                application.as_ptr(),
                command_line.as_mut_ptr(),
                null(),
                null(),
                1,
                0x0008_0000 | 0x0000_0400 | 0x0800_0000,
                environment
                    .as_ref()
                    .map_or(null(), |block| block.as_ptr().cast()),
                null(),
                &startup,
                &mut information,
            )
        };
        check(created, ProcessError::SpawnFailed)?;
        // SAFETY: successful CreateProcessW transfers distinct valid handles.
        let process = unsafe { OwnedHandle::from_raw_handle(information.process) };
        // SAFETY: the primary thread handle is separately owned and closes here.
        let primary_thread = unsafe { OwnedHandle::from_raw_handle(information.thread) };
        drop(primary_thread);
        Ok(Self {
            process,
            job,
            id: information.process_id,
            stdin,
            stdout,
        })
    }
    pub fn id(&self) -> u32 {
        self.id
    }
    pub fn try_wait(&mut self) -> ProcessResult<Option<ExitStatus>> {
        self.wait_timeout(Duration::ZERO)
    }
    pub fn wait_timeout(&mut self, timeout: Duration) -> ProcessResult<Option<ExitStatus>> {
        // MAX is INFINITE; reject it and longer durations rather than weakening
        // the requested finite bound. Round sub-millisecond budgets upward.
        let millis = timeout
            .as_millis()
            .checked_add(u128::from(
                !timeout.subsec_nanos().is_multiple_of(1_000_000),
            ))
            .ok_or(ProcessError::InvalidSpecification)?;
        let millis = u32::try_from(millis)
            .ok()
            .filter(|&value| value != u32::MAX)
            .ok_or(ProcessError::InvalidSpecification)?;
        // SAFETY: process handle remains owned throughout the bounded wait.
        match unsafe { ffi::WaitForSingleObject(self.process.as_raw_handle(), millis) } {
            0 => {
                let mut code = 0;
                // SAFETY: live signaled process handle and exclusive output.
                check(
                    unsafe { ffi::GetExitCodeProcess(self.process.as_raw_handle(), &mut code) },
                    ProcessError::WaitFailed,
                )?;
                Ok(Some(ExitStatus::from_raw(code)))
            }
            258 => Ok(None),
            _ => Err(os_error(ProcessError::WaitFailed)),
        }
    }
    pub fn terminate(&mut self) -> ProcessResult<()> {
        // SAFETY: job is owned, includes this child and its descendants, and
        // was created by this provider with termination rights.
        check(
            unsafe { ffi::TerminateJobObject(self.job.as_raw_handle(), 1) },
            ProcessError::TerminateFailed,
        )
    }

    pub fn terminate_timeout(&mut self, timeout: Duration) -> ProcessResult<Option<ExitStatus>> {
        let deadline = std::time::Instant::now()
            .checked_add(timeout)
            .ok_or(ProcessError::InvalidSpecification)?;
        self.terminate()?;
        let Some(status) = self.wait_timeout(timeout)? else {
            return Ok(None);
        };
        loop {
            let mut accounting = ffi::BasicAccounting::default();
            // SAFETY: class 1 selects the repr(C) basic accounting structure;
            // the owned job and exact-size exclusive output remain live.
            check(
                unsafe {
                    ffi::QueryInformationJobObject(
                        self.job.as_raw_handle(),
                        1,
                        (&raw mut accounting).cast(),
                        u32::try_from(size_of::<ffi::BasicAccounting>())
                            .map_err(|_| ProcessError::InvalidSpecification)?,
                        null_mut(),
                    )
                },
                ProcessError::WaitFailed,
            )?;
            if accounting.active_processes == 0 {
                return Ok(Some(status));
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            // Job handles have no all-descendants-exited wait operation.
            // Bounded polling confirms termination rather than root exit alone.
            std::thread::sleep(remaining.min(Duration::from_millis(1)));
        }
    }
}

fn create_job(policy: ProcessDropPolicy) -> ProcessResult<OwnedHandle> {
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
fn pipe() -> ProcessResult<(OwnedHandle, OwnedHandle)> {
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
fn remove_inheritance(handle: &OwnedHandle) -> ProcessResult<()> {
    // SAFETY: handle is live; HANDLE_FLAG_INHERIT is mask 1, cleared to zero.
    check(
        unsafe { ffi::SetHandleInformation(handle.as_raw_handle(), 1, 0) },
        ProcessError::SpawnFailed,
    )
}
fn duplicate(source: *mut c_void) -> ProcessResult<OwnedHandle> {
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

struct Attributes<'a> {
    storage: Vec<usize>,
    _values: PhantomData<&'a [*mut c_void]>,
}
impl<'a> Attributes<'a> {
    fn new(jobs: &'a [*mut c_void], inherited: &'a [*mut c_void]) -> ProcessResult<Self> {
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
    fn pointer(&mut self) -> *mut c_void {
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
fn check(result: i32, operation: ProcessError) -> ProcessResult<()> {
    if result == 0 {
        Err(os_error(operation))
    } else {
        Ok(())
    }
}
fn os_error(operation: ProcessError) -> ProcessError {
    ProcessError::OperatingSystem {
        operation: operation.operation(),
        code: std::io::Error::last_os_error().raw_os_error(),
    }
}
