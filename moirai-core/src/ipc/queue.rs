#![deny(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

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

unsafe impl<T: Send> Send for SharedQueue<T> {}

/// Alignment of the metadata header. The data buffer begins at
/// `ptr + size_of::<QueueMetadata>()`, which — because the OS maps page-aligned
/// memory and the header is a multiple of 64 — is 64-byte aligned. Element types
/// whose alignment exceeds this would be placed at a misaligned address, so they
/// are rejected at construction.
const HEADER_ALIGN: usize = 64;

#[repr(C, align(64))]
struct QueueMetadata {
    /// Producer position, cache-line aligned
    head: AtomicUsize,
    /// Element capacity recorded by the creator, validated by every `open` so a
    /// peer cannot map a differently-sized view of the same segment.
    capacity: AtomicUsize,
    /// Padding to isolate the producer line (8 + 8 + 48 = 64 bytes)
    _pad1: [u8; 48],
    /// Consumer position, cache-line aligned
    tail: AtomicUsize,
    /// Padding to isolate tail and closed flag (8 + 56 = 64 bytes)
    _pad2: [u8; 56],
    /// Queue closed flag
    closed: AtomicBool,
    /// Padding to align the entire structure to 64 bytes (1 + 63 = 64 bytes)
    _pad3: [u8; 63],
}

/// Compute the total mapping size for `capacity` elements of `T`, rejecting a
/// zero capacity (`% capacity` would divide by zero) and any size-overflow
/// (which would otherwise produce an undersized mapping and out-of-bounds
/// element access). Also rejects over-aligned element types.
fn layout_for<T>(capacity: usize) -> Result<usize, IpcError> {
    if capacity == 0 || mem::align_of::<T>() > HEADER_ALIGN {
        return Err(IpcError::InvalidArgument);
    }
    let meta_size = mem::size_of::<QueueMetadata>();
    capacity
        .checked_mul(mem::size_of::<T>())
        .and_then(|data_size| meta_size.checked_add(data_size))
        .ok_or(IpcError::InvalidArgument)
}

impl<T: bytemuck::Pod> SharedQueue<T> {
    /// Create a new shared queue.
    ///
    /// `T` is bounded by [`bytemuck::Pod`]: shared-memory contents are written by
    /// one process and read as `T` by another, so the element type must be valid
    /// for every bit pattern (no `bool`/`char`/enum/reference discriminants a
    /// peer could corrupt into an invalid value).
    pub fn create(name: &str, capacity: usize) -> Result<Self, IpcError> {
        let meta_size = mem::size_of::<QueueMetadata>();
        let total_size = layout_for::<T>(capacity)?;

        let memory = SharedMemory::create(name, total_size)?;

        unsafe {
            // justification: `memory.ptr` is the base of an OS shared-memory
            // mapping (mmap / MapViewOfFile), which is always page-aligned and
            // therefore satisfies `QueueMetadata`'s 64-byte alignment.
            #[allow(clippy::cast_ptr_alignment)]
            let meta = memory.ptr as *mut QueueMetadata;
            (*meta).head = AtomicUsize::new(0);
            (*meta).tail = AtomicUsize::new(0);
            (*meta).capacity = AtomicUsize::new(capacity);
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

    /// Open an existing shared queue. Fails with [`IpcError::InvalidArgument`] if
    /// the segment was created with a different capacity, which would otherwise
    /// map a view inconsistent with the creator's and fault on access.
    pub fn open(name: &str, capacity: usize) -> Result<Self, IpcError> {
        let meta_size = mem::size_of::<QueueMetadata>();
        let total_size = layout_for::<T>(capacity)?;

        let memory = SharedMemory::open(name, total_size)?;

        unsafe {
            // justification: `memory.ptr` is a page-aligned OS mapping base, which
            // satisfies `QueueMetadata`'s 64-byte alignment requirement.
            #[allow(clippy::cast_ptr_alignment)]
            let meta = memory.ptr as *mut QueueMetadata;
            // The header lives in the first page, so it is always within the
            // mapping regardless of `capacity`; validate before touching data.
            if (*meta).capacity.load(Ordering::Acquire) != capacity {
                return Err(IpcError::InvalidArgument);
            }
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
    pub fn send(&mut self, value: T) -> Result<(), T> {
        unsafe {
            if (*self.meta).closed.load(Ordering::Relaxed) {
                return Err(value);
            }

            let head = (*self.meta).head.load(Ordering::Relaxed);
            let tail = (*self.meta).tail.load(Ordering::Acquire);

            if head.wrapping_sub(tail) >= self.capacity {
                return Err(value);
            }

            // SAFETY-adjacent lint note: `capacity` is >= 1 by construction
            // (`layout_for` rejects zero at create/open), so the modulo
            // cannot panic.
            #[expect(
                clippy::arithmetic_side_effects,
                reason = "capacity >= 1 is validated at create/open via layout_for"
            )]
            core::ptr::write(self.buffer.add(head % self.capacity), value);
            (*self.meta)
                .head
                .store(head.wrapping_add(1), Ordering::Release);

            Ok(())
        }
    }

    /// Receive a value
    pub fn recv(&mut self) -> Option<T> {
        unsafe {
            let tail = (*self.meta).tail.load(Ordering::Relaxed);
            let head = (*self.meta).head.load(Ordering::Acquire);

            if tail == head {
                return None;
            }

            #[expect(
                clippy::arithmetic_side_effects,
                reason = "capacity >= 1 is validated at create/open via layout_for"
            )]
            let value = core::ptr::read(self.buffer.add(tail % self.capacity));
            (*self.meta)
                .tail
                .store(tail.wrapping_add(1), Ordering::Release);

            Some(value)
        }
    }
}
