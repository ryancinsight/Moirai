//! Same-machine inter-process communication over shared memory.
//!
//! Scope: a named [`SharedMemory`] mapping (POSIX `shm_open`/`mmap`, Windows
//! `CreateFileMappingW`/`MapViewOfFile`) and a lock-free single-producer
//! single-consumer [`SharedQueue`] laid out inside it. Cross-machine and
//! cross-device transport is out of scope here: GPU interop is the
//! hephaestus substrate's domain, network transport lives in
//! moirai-transport.

use crate::platform::*;
use core::fmt;
use core::mem;
use core::slice;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[cfg(unix)]
use std::os::unix::io::RawFd;

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
            IpcError::SystemError(code) => write!(f, "System error: {}", code),
            IpcError::InvalidArgument => write!(f, "Invalid argument"),
            IpcError::NotFound => write!(f, "Resource not found"),
            IpcError::PermissionDenied => write!(f, "Permission denied"),
        }
    }
}

impl core::error::Error for IpcError {}

/// Convert OS error to IpcError
#[cfg(unix)]
fn last_os_error() -> IpcError {
    unsafe { IpcError::SystemError(*libc::__errno_location()) }
}

/// Convert OS error to IpcError
#[cfg(windows)]
fn last_os_error() -> IpcError {
    extern "system" {
        fn GetLastError() -> u32;
    }
    // SAFETY: `GetLastError` takes no arguments and reads thread-local state.
    unsafe { IpcError::SystemError(GetLastError() as i32) }
}

/// Raw Win32 file-mapping bindings (extern decls keep the crate free of a
/// windows-sys dependency, matching the platform-query style elsewhere).
#[cfg(windows)]
mod win {
    pub const PAGE_READWRITE: u32 = 0x04;
    pub const FILE_MAP_ALL_ACCESS: u32 = 0x000F_001F;
    /// Pseudo-handle selecting a pagefile-backed mapping.
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

    /// NUL-terminated UTF-16 mapping name; the unix-style leading `/` is
    /// dropped so one logical name addresses the same object on both
    /// platforms.
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
    ptr: *mut u8,
    /// Size of the shared memory
    size: usize,
    /// File descriptor (Unix) or handle (Windows)
    #[cfg(unix)]
    fd: RawFd,
    #[cfg(windows)]
    handle: usize, // Use usize instead of *mut c_void to avoid dependency
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
            // Create shared memory object
            let fd = libc::shm_open(c_name.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o666);

            if fd < 0 {
                return Err(last_os_error());
            }

            // Set size
            if libc::ftruncate(fd, size as i64) < 0 {
                libc::close(fd);
                return Err(last_os_error());
            }

            // Map into memory
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
            // Open shared memory object
            let fd = libc::shm_open(c_name.as_ptr(), libc::O_RDWR, 0);

            if fd < 0 {
                return Err(last_os_error());
            }

            // Map into memory
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

        // SAFETY: `CreateFileMappingW` receives a NUL-terminated UTF-16 name
        // and a pagefile-backed pseudo-handle; `MapViewOfFile` maps `size`
        // bytes of the returned mapping. Both pointers are valid for the
        // duration of the calls and failure paths release the handle.
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

        // SAFETY: as in `create`; `OpenFileMappingW` only reads the
        // NUL-terminated name, and failure paths release the handle.
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
        // SAFETY: `ptr`/`size`/`fd` come from the successful `mmap`/`shm_open`
        // in `create`/`open` and are released exactly once here.
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.size);
            libc::close(self.fd);
            // Unlinking the name requires the name, which is not stored;
            // owners unlink explicitly when the object must be destroyed.
            let _ = self.owner;
        }
        #[cfg(windows)]
        // SAFETY: `ptr`/`handle` come from the successful
        // `MapViewOfFile`/`CreateFileMappingW` in `create`/`open` and are
        // released exactly once here. Windows file mappings are kernel
        // refcounted, so no owner-side unlink exists.
        unsafe {
            win::UnmapViewOfFile(self.ptr as *const core::ffi::c_void);
            win::CloseHandle(self.handle);
            let _ = self.owner;
        }
    }
}

/// Lock-free shared memory queue for IPC
pub struct SharedQueue<T> {
    /// Shared memory backing. Never read directly: held so the mapping's
    /// RAII lifetime covers `meta`/`buffer`, which point into it.
    #[allow(dead_code)]
    memory: SharedMemory,
    /// Queue metadata (stored at beginning of shared memory)
    meta: *mut QueueMetadata,
    /// Data buffer
    buffer: *mut T,
    /// Capacity
    capacity: usize,
}

#[repr(C)]
struct QueueMetadata {
    /// Producer position
    head: AtomicUsize,
    /// Consumer position  
    tail: AtomicUsize,
    /// Queue closed flag
    closed: AtomicBool,
    /// Padding for cache line alignment
    _padding: [u8; 64 - 17],
}

impl<T: Copy> SharedQueue<T> {
    /// Create a new shared queue
    pub fn create(name: &str, capacity: usize) -> Result<Self, IpcError> {
        let meta_size = mem::size_of::<QueueMetadata>();
        let data_size = capacity * mem::size_of::<T>();
        let total_size = meta_size + data_size;

        let memory = SharedMemory::create(name, total_size)?;

        unsafe {
            let meta = memory.ptr as *mut QueueMetadata;
            (*meta).head = AtomicUsize::new(0);
            (*meta).tail = AtomicUsize::new(0);
            (*meta).closed = AtomicBool::new(false);

            let buffer = memory.ptr.add(meta_size) as *mut T;

            Ok(Self {
                memory,
                meta,
                buffer,
                capacity,
            })
        }
    }

    /// Open an existing shared queue
    pub fn open(name: &str, capacity: usize) -> Result<Self, IpcError> {
        let meta_size = mem::size_of::<QueueMetadata>();
        let data_size = capacity * mem::size_of::<T>();
        let total_size = meta_size + data_size;

        let memory = SharedMemory::open(name, total_size)?;

        unsafe {
            let meta = memory.ptr as *mut QueueMetadata;
            let buffer = memory.ptr.add(meta_size) as *mut T;

            Ok(Self {
                memory,
                meta,
                buffer,
                capacity,
            })
        }
    }

    /// Send a value
    pub fn send(&self, value: T) -> Result<(), T> {
        unsafe {
            if (*self.meta).closed.load(Ordering::Relaxed) {
                return Err(value);
            }

            let head = (*self.meta).head.load(Ordering::Relaxed);
            let tail = (*self.meta).tail.load(Ordering::Acquire);

            if head.wrapping_sub(tail) >= self.capacity {
                return Err(value);
            }

            core::ptr::write(self.buffer.add(head % self.capacity), value);
            (*self.meta)
                .head
                .store(head.wrapping_add(1), Ordering::Release);

            Ok(())
        }
    }

    /// Receive a value
    pub fn recv(&self) -> Option<T> {
        unsafe {
            let tail = (*self.meta).tail.load(Ordering::Relaxed);
            let head = (*self.meta).head.load(Ordering::Acquire);

            if tail == head {
                return None;
            }

            let value = core::ptr::read(self.buffer.add(tail % self.capacity));
            (*self.meta)
                .tail
                .store(tail.wrapping_add(1), Ordering::Release);

            Some(value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_memory() {
        let name = "/moirai_test_shm";
        let size = 1024;

        // Create shared memory
        let mut shm1 = SharedMemory::create(name, size).unwrap();

        // Write some data
        let data = b"Hello, shared memory!";
        shm1.as_mut_slice()[..data.len()].copy_from_slice(data);

        // Open from another "process"
        let shm2 = SharedMemory::open(name, size).unwrap();

        // Read the data
        assert_eq!(&shm2.as_slice()[..data.len()], data);
    }

    #[test]
    fn open_of_missing_segment_reports_system_error() {
        let result = SharedMemory::open("/moirai_test_no_such_segment", 64);
        assert!(matches!(result, Err(IpcError::SystemError(_))));
    }

    #[test]
    fn zero_size_segment_is_rejected_on_windows() {
        #[cfg(windows)]
        {
            let result = SharedMemory::create("/moirai_test_zero", 0);
            assert!(matches!(result, Err(IpcError::InvalidArgument)));
        }
    }

    #[test]
    fn test_shared_queue() {
        let name = "/moirai_test_queue";
        let capacity = 10;

        // Create queue
        let queue = SharedQueue::<u32>::create(name, capacity).unwrap();

        // Send some values
        queue.send(1).unwrap();
        queue.send(2).unwrap();
        queue.send(3).unwrap();

        // Receive values
        assert_eq!(queue.recv(), Some(1));
        assert_eq!(queue.recv(), Some(2));
        assert_eq!(queue.recv(), Some(3));
        assert_eq!(queue.recv(), None);
    }
}
