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

// SAFETY: queue contents move between threads and processes as plain `Pod`
// bits, so `T: Send` is required; no references into shared memory escape.
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

/// Header size in bytes; the capacity field sits right after the producer
/// position (`head`) at this offset.
pub(crate) const QUEUE_META_SIZE: usize = mem::size_of::<QueueMetadata>();

/// Pure layout arithmetic behind [`layout_for`]: total mapping size for
/// `meta_size` header bytes plus `elem_count * elem_size`, rejecting zero
/// count and overflow. Split out so the fuzz targets can exercise the exact
/// arithmetic `create`/`open` rely on without OS resources.
pub(crate) fn layout_total(
    meta_size: usize,
    elem_size: usize,
    elem_align: usize,
    elem_count: usize,
) -> Result<usize, IpcError> {
    if elem_count == 0 || elem_align > HEADER_ALIGN || meta_size == 0 || elem_size == 0 {
        return Err(IpcError::InvalidArgument);
    }
    elem_count
        .checked_mul(elem_size)
        .and_then(|data| data.checked_add(meta_size))
        .ok_or(IpcError::InvalidArgument)
}

/// Parse the recorded capacity out of raw header bytes. Pure so the fuzz
/// targets can throw peer-controlled bytes at the exact check `open`
/// performs; tolerant of unaligned input because it copies through
/// `from_le_bytes`.
pub(crate) fn parse_header_capacity(bytes: &[u8]) -> Result<usize, IpcError> {
    const WIDTH: usize = mem::size_of::<usize>();
    let off = mem::size_of::<AtomicUsize>();
    if bytes.len() < QUEUE_META_SIZE {
        return Err(IpcError::InvalidArgument);
    }
    let end = off.checked_add(WIDTH).ok_or(IpcError::InvalidArgument)?;
    let raw: [u8; WIDTH] = bytes
        .get(off..end)
        .and_then(|field| field.try_into().ok())
        .ok_or(IpcError::InvalidArgument)?;
    Ok(usize::from_le_bytes(raw))
}

/// Compute the total mapping size for `capacity` elements of `T`, rejecting a
/// zero capacity (`% capacity` would divide by zero) and any size-overflow
/// (which would otherwise produce an undersized mapping and out-of-bounds
/// element access). Also rejects over-aligned element types.
fn layout_for<T>(capacity: usize) -> Result<usize, IpcError> {
    if capacity == 0 || mem::align_of::<T>() > HEADER_ALIGN {
        return Err(IpcError::InvalidArgument);
    }
    layout_total(
        QUEUE_META_SIZE,
        mem::size_of::<T>(),
        mem::align_of::<T>(),
        capacity,
    )
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

        // SAFETY: `memory.ptr` is the base of an OS shared-memory mapping
        // (mmap / MapViewOfFile), always page-aligned, satisfying
        // `QueueMetadata`'s 64-byte alignment; header fields are written
        // before any peer maps the segment (created above).
        unsafe {
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

        // SAFETY: `memory.ptr` is a page-aligned OS mapping base, satisfying
        // `QueueMetadata`'s 64-byte alignment; the header lives in the first
        // page regardless of `capacity`.
        let mut header = [0u8; QUEUE_META_SIZE];
        unsafe {
            core::ptr::copy_nonoverlapping(
                memory.ptr.cast::<u8>(),
                header.as_mut_ptr(),
                QUEUE_META_SIZE,
            );
        }
        // Capacity is immutable after creation, so a plain copy carries no
        // ordering obligation; the recorded value must match ours or the
        // segment was created with a different geometry.
        let stored = parse_header_capacity(&header)?;
        if stored != capacity {
            return Err(IpcError::InvalidArgument);
        }

        unsafe {
            #[allow(clippy::cast_ptr_alignment)]
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
    pub fn send(&mut self, value: T) -> Result<(), T> {
        // SAFETY: `&mut self` makes this process the sole sender endpoint
        // (SPSC contract across processes); the fullness check keeps the
        // head slot outside the consumer window, and Pod writes need no
        // drop coordination.
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
        // SAFETY: `&mut self` makes this process the sole receiver endpoint;
        // the emptiness check guarantees the tail slot was published by the
        // sender and reading it as Pod bits moves it out exactly once.
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
