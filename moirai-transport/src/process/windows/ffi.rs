//! Native process ABI, verified against Windows SDK 10.0.26100.0 headers.
//! Definitions mirror winnt.h, processthreadsapi.h, and WinBase.h.
use std::ffi::c_void;

#[derive(Default)]
#[repr(C)]
pub(super) struct SecurityAttributes {
    pub length: u32,
    pub descriptor: *mut c_void,
    pub inherit: i32,
}
#[derive(Default)]
#[repr(C)]
pub(super) struct StartupInfo {
    pub size: u32,
    pub reserved: *mut u16,
    pub desktop: *mut u16,
    pub title: *mut u16,
    pub x: u32,
    pub y: u32,
    pub x_size: u32,
    pub y_size: u32,
    pub x_chars: u32,
    pub y_chars: u32,
    pub fill: u32,
    pub flags: u32,
    pub show: u16,
    pub reserved_size: u16,
    pub reserved_bytes: *mut u8,
    pub stdin: *mut c_void,
    pub stdout: *mut c_void,
    pub stderr: *mut c_void,
}
#[derive(Default)]
#[repr(C)]
pub(super) struct StartupInfoEx {
    pub base: StartupInfo,
    pub attributes: *mut c_void,
}
#[derive(Default)]
#[repr(C)]
pub(super) struct ProcessInformation {
    pub process: *mut c_void,
    pub thread: *mut c_void,
    pub process_id: u32,
    pub thread_id: u32,
}
#[derive(Default)]
#[repr(C)]
pub(super) struct BasicLimits {
    pub process_time: i64,
    pub job_time: i64,
    pub flags: u32,
    pub minimum_working_set: usize,
    pub maximum_working_set: usize,
    pub active_processes: u32,
    pub affinity: usize,
    pub priority: u32,
    pub scheduling: u32,
}
#[derive(Default)]
#[repr(C)]
pub(super) struct BasicAccounting {
    pub user_time: i64,
    pub kernel_time: i64,
    pub period_user_time: i64,
    pub period_kernel_time: i64,
    pub page_faults: u32,
    pub total_processes: u32,
    pub active_processes: u32,
    pub terminated_processes: u32,
}
#[derive(Default)]
#[repr(C)]
pub(super) struct ExtendedLimits {
    pub basic: BasicLimits,
    pub io_counters: [u64; 6],
    pub process_memory: usize,
    pub job_memory: usize,
    pub peak_process_memory: usize,
    pub peak_job_memory: usize,
}
#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(size_of::<StartupInfo>() == 104);
    assert!(size_of::<StartupInfoEx>() == 112);
    assert!(size_of::<ExtendedLimits>() == 144);
};

#[link(name = "kernel32")]
unsafe extern "system" {
    pub(super) fn QueryInformationJobObject(
        job: *mut c_void,
        class: i32,
        information: *mut c_void,
        length: u32,
        returned: *mut u32,
    ) -> i32;
    pub(super) fn CompareStringOrdinal(
        left: *const u16,
        left_count: i32,
        right: *const u16,
        right_count: i32,
        ignore_case: i32,
    ) -> i32;
    pub(super) fn CreateJobObjectW(
        attributes: *const SecurityAttributes,
        name: *const u16,
    ) -> *mut c_void;
    pub(super) fn SetInformationJobObject(
        job: *mut c_void,
        class: i32,
        information: *const c_void,
        length: u32,
    ) -> i32;
    pub(super) fn TerminateJobObject(job: *mut c_void, exit_code: u32) -> i32;
    pub(super) fn InitializeProcThreadAttributeList(
        list: *mut c_void,
        count: u32,
        flags: u32,
        size: *mut usize,
    ) -> i32;
    pub(super) fn UpdateProcThreadAttribute(
        list: *mut c_void,
        flags: u32,
        attribute: usize,
        value: *const c_void,
        size: usize,
        previous: *mut c_void,
        returned: *mut usize,
    ) -> i32;
    pub(super) fn DeleteProcThreadAttributeList(list: *mut c_void);
    pub(super) fn CreatePipe(
        reader: *mut *mut c_void,
        writer: *mut *mut c_void,
        attributes: *const SecurityAttributes,
        size: u32,
    ) -> i32;
    pub(super) fn SetHandleInformation(handle: *mut c_void, mask: u32, flags: u32) -> i32;
    pub(super) fn GetCurrentProcess() -> *mut c_void;
    pub(super) fn DuplicateHandle(
        source_process: *mut c_void,
        source: *mut c_void,
        target_process: *mut c_void,
        target: *mut *mut c_void,
        access: u32,
        inherit: i32,
        options: u32,
    ) -> i32;
    pub(super) fn CreateProcessW(
        application: *const u16,
        command_line: *mut u16,
        process_attributes: *const SecurityAttributes,
        thread_attributes: *const SecurityAttributes,
        inherit: i32,
        flags: u32,
        environment: *const c_void,
        directory: *const u16,
        startup: *const StartupInfoEx,
        information: *mut ProcessInformation,
    ) -> i32;
    pub(super) fn WaitForSingleObject(handle: *mut c_void, milliseconds: u32) -> u32;
    pub(super) fn GetExitCodeProcess(process: *mut c_void, code: *mut u32) -> i32;
}
