#![deny(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

use super::error::IpcError;
use super::memory::SharedMemory;
use core::mem;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Lock-free single-producer, single-consumer queue in shared memory.
///
/// Each end is exclusive: the first `send` on a handle claims the queue's sender
/// endpoint and the first `recv` its receiver endpoint, through flags in the
/// shared header, so two handles -- in one process or several -- can never both
/// send or both receive. A handle may hold both ends. Claims release when the
/// handle drops; a process killed while holding one leaves it held until the
/// creator drops the segment.
pub struct SharedQueue<T> {
    #[allow(dead_code)]
    memory: SharedMemory,
    /// Queue metadata (stored at beginning of shared memory)
    meta: *mut QueueMetadata,
    /// Data buffer
    buffer: *mut T,
    /// Capacity
    capacity: usize,
    /// Whether this handle holds the queue's sender endpoint
    holds_sender: bool,
    /// Whether this handle holds the queue's receiver endpoint
    holds_receiver: bool,
}

/// Why [`SharedQueue::send`] did not enqueue; each variant returns the value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendError<T> {
    /// The ring holds `capacity` unreceived values.
    Full(T),
    /// The queue was closed.
    Closed(T),
    /// Another handle, in this process or another, already sends on this queue.
    EndpointInUse(T),
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
    /// Set while one handle holds the sender endpoint
    sender_claimed: AtomicBool,
    /// Set while one handle holds the receiver endpoint
    receiver_claimed: AtomicBool,
    /// Padding to align the entire structure to 64 bytes (3 + 61 = 64 bytes)
    _pad3: [u8; 61],
}

/// Header size in bytes; the capacity field sits right after the producer
/// position (`head`) at this offset.
pub(crate) const QUEUE_META_SIZE: usize = mem::size_of::<QueueMetadata>();

const _: () = assert!(QUEUE_META_SIZE == 3 * HEADER_ALIGN);

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
    /// Create a new shared queue under a name no live segment holds.
    ///
    /// Fails with [`IpcError::AlreadyExists`] when the name is taken, so a
    /// second creator can never reinitialise the header under live handles.
    ///
    /// `T` is bounded by [`bytemuck::Pod`]: shared-memory contents are written by
    /// one process and read as `T` by another, so the element type must be valid
    /// for every bit pattern (no `bool`/`char`/enum/reference discriminants a
    /// peer could corrupt into an invalid value).
    ///
    /// # Errors
    /// Returns [`IpcError::AlreadyExists`] for a taken name,
    /// [`IpcError::InvalidArgument`] for a zero capacity, an over-aligned `T`,
    /// or a size overflow, and the OS error otherwise.
    pub fn create(name: &str, capacity: usize) -> Result<Self, IpcError> {
        let total_size = layout_for::<T>(capacity)?;
        let memory = SharedMemory::create(name, total_size)?;

        // SAFETY: `memory.ptr` is the base of a fresh OS mapping (mmap /
        // MapViewOfFile), always page-aligned and so satisfying
        // `QueueMetadata`'s 64-byte alignment, and `total_size` covers the
        // header. The header is only touched through atomics, which is sound
        // against a peer that opens the name while this runs; `capacity` is
        // stored last with `Release` so an opener that observes it sees the
        // rest.
        let meta = unsafe { &*header_of(&memory) };
        meta.head.store(0, Ordering::Relaxed);
        meta.tail.store(0, Ordering::Relaxed);
        meta.closed.store(false, Ordering::Relaxed);
        meta.sender_claimed.store(false, Ordering::Relaxed);
        meta.receiver_claimed.store(false, Ordering::Relaxed);
        meta.capacity.store(capacity, Ordering::Release);

        Ok(Self::attach(memory, capacity))
    }

    /// Open an existing shared queue. Fails with [`IpcError::InvalidArgument`] if
    /// the segment was created with a different capacity, which would otherwise
    /// map a view inconsistent with the creator's and fault on access.
    ///
    /// # Errors
    /// Returns [`IpcError::InvalidArgument`] for a capacity mismatch (including
    /// a creator that has not finished initialising), and the OS error when the
    /// segment is missing or too small.
    pub fn open(name: &str, capacity: usize) -> Result<Self, IpcError> {
        let total_size = layout_for::<T>(capacity)?;
        let memory = SharedMemory::open(name, total_size)?;

        // SAFETY: as in `create`; `SharedMemory::open` proved the object covers
        // `total_size >= QUEUE_META_SIZE` bytes, and the recorded capacity is
        // read atomically because the creator writes it while peers attach.
        let stored = unsafe { &*header_of(&memory) }
            .capacity
            .load(Ordering::Acquire);
        if stored != capacity {
            return Err(IpcError::InvalidArgument);
        }

        Ok(Self::attach(memory, capacity))
    }

    fn attach(memory: SharedMemory, capacity: usize) -> Self {
        let meta = header_of(&memory);
        // SAFETY: `layout_for` sized the mapping as the header followed by
        // `capacity` elements, so the offset stays inside it; the header is a
        // multiple of `HEADER_ALIGN`, and `layout_for` rejected any `T`
        // aligned more strictly, so the buffer is aligned for `T`.
        let buffer = unsafe { memory.ptr.add(QUEUE_META_SIZE) }.cast::<T>();
        Self {
            memory,
            meta,
            buffer,
            capacity,
            holds_sender: false,
            holds_receiver: false,
        }
    }

    /// Send a value.
    ///
    /// The first call claims the queue's sender endpoint for this handle; the
    /// claim is released when the handle drops.
    ///
    /// # Errors
    /// Returns the value inside [`SendError::Full`] when the ring is full,
    /// [`SendError::Closed`] when the queue is closed, and
    /// [`SendError::EndpointInUse`] when another handle already sends on this
    /// queue.
    pub fn send(&mut self, value: T) -> Result<(), SendError<T>> {
        if !self.holds_sender {
            // SAFETY: `meta` points at the live mapping's header for the
            // lifetime of `self`; only atomics are accessed through it.
            if !claim(unsafe { &(*self.meta).sender_claimed }) {
                return Err(SendError::EndpointInUse(value));
            }
            self.holds_sender = true;
        }

        // SAFETY: holding the sender claim makes this handle the queue's only
        // sender, in this process and every other (the flag lives in the shared
        // header and is won by one compare-exchange). The fullness check keeps
        // the head slot outside the consumer window, and Pod writes need no
        // drop coordination.
        unsafe {
            if (*self.meta).closed.load(Ordering::Relaxed) {
                return Err(SendError::Closed(value));
            }

            let head = (*self.meta).head.load(Ordering::Relaxed);
            let tail = (*self.meta).tail.load(Ordering::Acquire);

            if head.wrapping_sub(tail) >= self.capacity {
                return Err(SendError::Full(value));
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

    /// Receive a value, or `None` when the queue is empty.
    ///
    /// The first call claims the queue's receiver endpoint for this handle; the
    /// claim is released when the handle drops.
    ///
    /// # Errors
    /// Returns [`IpcError::EndpointInUse`] when another handle already receives
    /// on this queue.
    pub fn recv(&mut self) -> Result<Option<T>, IpcError> {
        if !self.holds_receiver {
            // SAFETY: as in `send`.
            if !claim(unsafe { &(*self.meta).receiver_claimed }) {
                return Err(IpcError::EndpointInUse);
            }
            self.holds_receiver = true;
        }

        // SAFETY: holding the receiver claim makes this handle the queue's only
        // receiver; the emptiness check guarantees the tail slot was published
        // by the sender, and reading it as Pod bits moves it out exactly once.
        unsafe {
            let tail = (*self.meta).tail.load(Ordering::Relaxed);
            let head = (*self.meta).head.load(Ordering::Acquire);

            if tail == head {
                return Ok(None);
            }

            #[expect(
                clippy::arithmetic_side_effects,
                reason = "capacity >= 1 is validated at create/open via layout_for"
            )]
            let value = core::ptr::read(self.buffer.add(tail % self.capacity));
            (*self.meta)
                .tail
                .store(tail.wrapping_add(1), Ordering::Release);

            Ok(Some(value))
        }
    }
}

impl<T> Drop for SharedQueue<T> {
    fn drop(&mut self) {
        // SAFETY: `meta` points into `self.memory`, which is unmapped only after
        // this body returns. Releasing publishes this handle's final `head` or
        // `tail` store to the next holder's acquiring claim.
        unsafe {
            if self.holds_sender {
                (*self.meta).sender_claimed.store(false, Ordering::Release);
            }
            if self.holds_receiver {
                (*self.meta)
                    .receiver_claimed
                    .store(false, Ordering::Release);
            }
        }
    }
}

/// The header at the base of a mapping.
///
/// The OS maps page-aligned memory, so the base satisfies `QueueMetadata`'s
/// 64-byte alignment even though `ptr` is typed as bytes.
#[expect(
    clippy::cast_ptr_alignment,
    reason = "mmap and MapViewOfFile return page-aligned bases"
)]
fn header_of(memory: &SharedMemory) -> *mut QueueMetadata {
    memory.ptr.cast::<QueueMetadata>()
}

/// Win an endpoint flag: exactly one caller across all handles and processes
/// sees `true` until the holder releases it.
fn claim(flag: &AtomicBool) -> bool {
    flag.compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
        .is_ok()
}
