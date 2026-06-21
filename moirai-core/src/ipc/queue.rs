use super::error::IpcError;
use super::memory::SharedMemory;
use core::mem;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Lock-free shared memory queue for IPC
pub struct SharedQueue<T> {
    #[allow(dead_code)]
    memory: SharedMemory,
    /// Queue metadata (stored at beginning of shared memory)
    meta: *mut QueueMetadata,
    /// Data buffer
    buffer: *mut T,
    /// Capacity
    capacity: usize,
}

#[repr(C, align(64))]
struct QueueMetadata {
    /// Producer position, cache-line aligned
    head: AtomicUsize,
    /// Padding to isolate head and tail (8 + 56 = 64 bytes)
    _pad1: [u8; 56],
    /// Consumer position, cache-line aligned
    tail: AtomicUsize,
    /// Padding to isolate tail and closed flag (8 + 56 = 64 bytes)
    _pad2: [u8; 56],
    /// Queue closed flag
    closed: AtomicBool,
    /// Padding to align the entire structure to 64 bytes (1 + 63 = 64 bytes)
    _pad3: [u8; 63],
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
