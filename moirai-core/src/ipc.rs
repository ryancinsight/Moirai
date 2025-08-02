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
use core::slice;
use core::mem;
use core::fmt;

#[cfg(unix)]
use std::os::unix::io::RawFd;

/// IPC-specific error type to minimize dependencies
#[derive(Debug, Clone)]
pub enum IpcError {
    /// System error with error code
    SystemError(i32),
    /// Invalid argument
    InvalidArgument,
    /// Not implemented
    NotImplemented,
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
            IpcError::NotImplemented => write!(f, "Not implemented"),
            IpcError::NotFound => write!(f, "Resource not found"),
            IpcError::PermissionDenied => write!(f, "Permission denied"),
        }
    }
}

impl core::error::Error for IpcError {}

/// Convert OS error to IpcError
#[cfg(unix)]
fn last_os_error() -> IpcError {
    unsafe {
        IpcError::SystemError(*libc::__errno_location())
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
            // Create shared memory object
            let fd = libc::shm_open(
                c_name.as_ptr(),
                libc::O_CREAT | libc::O_RDWR,
                0o666
            );
            
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
                0
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
            // Open shared memory object
            let fd = libc::shm_open(
                c_name.as_ptr(),
                libc::O_RDWR,
                0
            );
            
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
                0
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
    /// Shared memory backing
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
            (*self.meta).head.store(head.wrapping_add(1), Ordering::Release);
            
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
            (*self.meta).tail.store(tail.wrapping_add(1), Ordering::Release);
            
            Some(value)
        }
    }
}

/// RDMA connection for high-performance networking
#[allow(dead_code)]
pub struct RdmaConnection {
    /// Connection handle (placeholder)
    handle: usize,
    /// Remote address as string (to avoid std::net dependency)
    remote_addr: String,
    /// Local address as string
    local_addr: String,
    /// Queue pair number
    qp_num: u32,
}

impl RdmaConnection {
    /// Connect to an RDMA endpoint
    pub fn connect(_addr: &str) -> Result<Self, IpcError> {
        Err(IpcError::NotImplemented)
    }
    
    /// Register memory region for RDMA
    pub fn register_memory(&self, _addr: *mut u8, _len: usize) -> Result<u32, IpcError> {
        Err(IpcError::NotImplemented)
    }
    
    /// Write data to remote memory
    pub fn write(&self, _local: *const u8, _remote_addr: u64, _len: usize, _rkey: u32) -> Result<(), IpcError> {
        Err(IpcError::NotImplemented)
    }
    
    /// Read data from remote memory
    pub fn read(&self, _local: *mut u8, _remote_addr: u64, _len: usize, _rkey: u32) -> Result<(), IpcError> {
        Err(IpcError::NotImplemented)
    }
}

/// GPU IPC for CUDA/ROCm interoperability
#[allow(dead_code)]
pub struct GpuIpc {
    /// Device ID
    device_id: u32,
    /// Memory handles
    handles: HashMap<u64, GpuMemHandle>,
}

#[allow(dead_code)]
struct GpuMemHandle {
    /// GPU memory pointer
    ptr: u64,
    /// Size of allocation
    size: usize,
    /// IPC handle for sharing
    handle: [u8; 64],
}

impl GpuIpc {
    /// Create a new GPU IPC context
    pub fn new(device_id: u32) -> Self {
        Self {
            device_id,
            handles: HashMap::new(),
        }
    }
    
    /// Create a shareable GPU memory handle
    pub fn create_handle(&mut self, gpu_ptr: u64, size: usize) -> Result<[u8; 64], IpcError> {
        // In production, this would use CUDA IPC API
        let handle = [0u8; 64]; // Placeholder
        
        self.handles.insert(gpu_ptr, GpuMemHandle {
            ptr: gpu_ptr,
            size,
            handle,
        });
        
        Ok(handle)
    }
    
    /// Open a GPU memory handle from another process
    pub fn open_handle(&self, _handle: [u8; 64]) -> Result<u64, IpcError> {
        // TODO: Implement handle opening
        Ok(0)
    }
}

/// Distributed communication coordinator
#[allow(dead_code)]
pub struct DistributedComm {
    /// Node ID in the cluster
    node_id: u32,
    /// Total number of nodes
    num_nodes: u32,
}

#[allow(dead_code)]
enum CommBackend {
    /// MPI backend
    Mpi,
    /// TCP sockets
    Tcp,
    /// RDMA
    Rdma,
}

/// Reduction operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum all values
    Sum,
    /// Multiply all values
    Product,
    /// Find minimum value
    Min,
    /// Find maximum value
    Max,
    /// Bitwise AND
    And,
    /// Bitwise OR
    Or,
    /// Bitwise XOR
    Xor,
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