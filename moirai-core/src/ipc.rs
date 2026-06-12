//! Inter-process and inter-system communication infrastructure.
//!
//! This module provides efficient communication between:
//! - Different processes on the same machine
//! - Different machines over the network
//! - Different devices (GPU, FPGA, etc.)
//!
//! Inspired by:
//! - MPI for distributed computing
//! - RDMA for low-latency networking
//! - CUDA IPC for GPU communication

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
    /// Operation or feature is not supported on this platform/configuration
    Unsupported,
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
            IpcError::Unsupported => write!(f, "Unsupported operation"),
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

    /// Get a slice of the shared memory
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr, self.size) }
    }

    /// Get a mutable slice of the shared memory
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.size) }
    }
}

#[cfg(unix)]
impl Drop for SharedMemory {
    fn drop(&mut self) {
        unsafe {
            #[cfg(unix)]
            {
                // Unmap memory
                libc::munmap(self.ptr as *mut libc::c_void, self.size);

                // Close file descriptor
                libc::close(self.fd);

                // Unlink if owner
                if self.owner {
                    // Note: We don't have the name here, so unlinking
                    // should be done explicitly by the user
                }
            }
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
    #[cfg(unix)]
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
    #[cfg(unix)]
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
