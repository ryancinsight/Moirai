use super::error::{last_os_error, IpcError};
use core::slice;

#[cfg(unix)]
use std::os::unix::io::RawFd;

/// Raw Win32 file-mapping bindings
#[cfg(windows)]
mod win {
    pub const PAGE_READWRITE: u32 = 0x04;
    pub const FILE_MAP_ALL_ACCESS: u32 = 0x000F_001F;
    pub const INVALID_HANDLE_VALUE: usize = usize::MAX;

    extern "system" {
        pub fn CreateFileMappingW(
            file: usize,
            attributes: *mut core::ffi::c_void,
            protect: u32,
            size_high: u32,
            size_low: u32,
            name: *const u16,
        ) -> usize;
        pub fn OpenFileMappingW(desired_access: u32, inherit: i32, name: *const u16) -> usize;
        pub fn MapViewOfFile(
            mapping: usize,
            desired_access: u32,
            offset_high: u32,
            offset_low: u32,
            size: usize,
        ) -> *mut core::ffi::c_void;
        pub fn UnmapViewOfFile(address: *const core::ffi::c_void) -> i32;
        pub fn CloseHandle(handle: usize) -> i32;
    }

    pub fn wide_name(name: &str) -> Vec<u16> {
        name.trim_start_matches('/')
            .encode_utf16()
            .chain(core::iter::once(0))
            .collect()
    }
}

/// Shared memory segment for zero-copy IPC
pub struct SharedMemory {
    /// Memory-mapped region
    pub(crate) ptr: *mut u8,
    /// Size of the shared memory
    pub(crate) size: usize,
    /// File descriptor (Unix) or handle (Windows)
    #[cfg(unix)]
    fd: RawFd,
    #[cfg(windows)]
    handle: usize,
    /// Whether this instance owns the memory
    owner: bool,
}

unsafe impl Send for SharedMemory {}
unsafe impl Sync for SharedMemory {}

impl SharedMemory {
    /// Create a new shared memory segment
    #[cfg(unix)]
    pub fn create(name: &str, size: usize) -> Result<Self, IpcError> {
        use std::ffi::CString;

        let c_name = CString::new(name).map_err(|_| IpcError::InvalidArgument)?;

        unsafe {
            use std::ptr::null_mut;
            let fd = libc::shm_open(c_name.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o666);

            if fd < 0 {
                return Err(last_os_error());
            }

            if libc::ftruncate(fd, size as i64) < 0 {
                libc::close(fd);
                return Err(last_os_error());
            }

            let ptr = libc::mmap(
                null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );

            if ptr == libc::MAP_FAILED {
                libc::close(fd);
                return Err(last_os_error());
            }

            Ok(Self {
                ptr: ptr as *mut u8,
                size,
                fd,
                owner: true,
            })
        }
    }

    /// Open an existing shared memory segment
    #[cfg(unix)]
    pub fn open(name: &str, size: usize) -> Result<Self, IpcError> {
        use std::ffi::CString;

        let c_name = CString::new(name).map_err(|_| IpcError::InvalidArgument)?;

        unsafe {
            use std::ptr::null_mut;
            let fd = libc::shm_open(c_name.as_ptr(), libc::O_RDWR, 0);

            if fd < 0 {
                return Err(last_os_error());
            }

            let ptr = libc::mmap(
                null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );

            if ptr == libc::MAP_FAILED {
                libc::close(fd);
                return Err(last_os_error());
            }

            Ok(Self {
                ptr: ptr as *mut u8,
                size,
                fd,
                owner: false,
            })
        }
    }

    /// Create a new shared memory segment
    #[cfg(windows)]
    pub fn create(name: &str, size: usize) -> Result<Self, IpcError> {
        if size == 0 {
            return Err(IpcError::InvalidArgument);
        }
        let wide = win::wide_name(name);

        unsafe {
            let handle = win::CreateFileMappingW(
                win::INVALID_HANDLE_VALUE,
                core::ptr::null_mut(),
                win::PAGE_READWRITE,
                (size as u64 >> 32) as u32,
                size as u32,
                wide.as_ptr(),
            );
            if handle == 0 {
                return Err(last_os_error());
            }

            let ptr = win::MapViewOfFile(handle, win::FILE_MAP_ALL_ACCESS, 0, 0, size);
            if ptr.is_null() {
                let error = last_os_error();
                win::CloseHandle(handle);
                return Err(error);
            }

            Ok(Self {
                ptr: ptr as *mut u8,
                size,
                handle,
                owner: true,
            })
        }
    }

    /// Open an existing shared memory segment
    #[cfg(windows)]
    pub fn open(name: &str, size: usize) -> Result<Self, IpcError> {
        if size == 0 {
            return Err(IpcError::InvalidArgument);
        }
        let wide = win::wide_name(name);

        unsafe {
            let handle = win::OpenFileMappingW(win::FILE_MAP_ALL_ACCESS, 0, wide.as_ptr());
            if handle == 0 {
                return Err(last_os_error());
            }

            let ptr = win::MapViewOfFile(handle, win::FILE_MAP_ALL_ACCESS, 0, 0, size);
            if ptr.is_null() {
                let error = last_os_error();
                win::CloseHandle(handle);
                return Err(error);
            }

            Ok(Self {
                ptr: ptr as *mut u8,
                size,
                handle,
                owner: false,
            })
        }
    }

    /// Get a slice of the shared memory
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr, self.size) }
    }

    /// Get a mutable slice of the shared memory
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.size) }
    }
}

impl Drop for SharedMemory {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.size);
            libc::close(self.fd);
            let _ = self.owner;
        }
        #[cfg(windows)]
        unsafe {
            win::UnmapViewOfFile(self.ptr as *const core::ffi::c_void);
            win::CloseHandle(self.handle);
            let _ = self.owner;
        }
    }
}
