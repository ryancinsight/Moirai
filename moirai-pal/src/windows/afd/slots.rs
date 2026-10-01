//! Fixed-address table of poll records and the protocol that decides who may
//! touch one.
//!
//! A slot is claimed from an atomic bitmap, armed (its record handed to the
//! kernel), and released only when its completion packet has been dequeued and
//! no canceller is still inside `NtCancelIoFileEx` with its status block. The
//! table never moves or grows, so a record address handed to the kernel stays
//! valid for the table's lifetime.
//!
//! Each slot has one state word: the generation in the high 32 bits and three
//! flags below. The generation advances at every arm, so a [`Token`] from an
//! earlier arm of a reused slot matches nothing (until the 32-bit generation
//! of one slot wraps).

use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use windows::Win32::Foundation::{NTSTATUS, STATUS_CANCELLED, STATUS_PENDING};
use windows::Win32::System::IO::{IO_STATUS_BLOCK, IO_STATUS_BLOCK_0, OVERLAPPED};

use super::abi::AfdPollInfo;
use crate::Event;

/// The record is owned by the kernel: armed and not yet dequeued.
const ARMED: u64 = 1;
/// A canceller is inside `NtCancelIoFileEx` with this slot's status block.
const CANCELLING: u64 = 2;
/// The completion packet was dequeued.
const COMPLETED: u64 = 4;
/// A cancellation was requested for this arm. Unlike `CANCELLING` it is never
/// cleared by the canceller, so a completion that raced the cancellation and
/// reached the queue first is still suppressed.
const REQUESTED: u64 = 8;
const FLAGS: u64 = 15;

/// Identity of one armed poll: slot index and the generation of that arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Token {
    index: u32,
    generation: u32,
}

impl Token {
    fn new(index: usize, generation: u32) -> Self {
        Self {
            index: u32::try_from(index).expect("table capacity fits a token index"),
            generation,
        }
    }

    pub(super) fn index(self) -> usize {
        self.index as usize
    }
}

/// Request state shared with the kernel. While a request is armed the kernel
/// writes `info` and `status_block` at any time, so no reference to a `Record`
/// or to any of its fields is ever formed; every access goes through raw
/// pointers projected from the cell.
struct Record {
    info: AfdPollInfo,
    status_block: IO_STATUS_BLOCK,
    socket: usize,
}

struct Slot {
    word: AtomicU64,
    record: UnsafeCell<Record>,
}

// SAFETY: the state word grants access to `record`. The claiming thread writes
// it before the Release store that arms the slot; from then until the packet
// is dequeued only the kernel writes it, and other threads touch only its
// address; the dequeuing thread then reads it, and a canceller reads only the
// status block address while it holds the slot. All access is through raw
// pointers, so no two threads ever hold a reference to the same bytes.
unsafe impl Sync for Slot {}

/// What dequeuing one packet produced.
pub(super) enum Completion {
    /// The packet named no slot of this table.
    Foreign,
    /// The poll was cancelled; its slot is released or about to be.
    Cancelled,
    /// The poll finished with `status` and `readiness`.
    Finished {
        token: Token,
        status: NTSTATUS,
        readiness: Event,
    },
}

/// Kernel-facing addresses of a published slot.
pub(super) struct Request {
    pub(super) info: *mut AfdPollInfo,
    pub(super) status_block: *mut IO_STATUS_BLOCK,
    /// Completion context that names the slot in its packet.
    pub(super) context: *const c_void,
}

pub(super) struct SlotTable {
    slots: Box<[Slot]>,
    claimed: Box<[AtomicU64]>,
    outstanding: AtomicUsize,
}

impl SlotTable {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            slots: (0..capacity)
                .map(|_| Slot {
                    word: AtomicU64::new(0),
                    record: UnsafeCell::new(Record {
                        info: AfdPollInfo::idle(),
                        status_block: IO_STATUS_BLOCK::default(),
                        socket: 0,
                    }),
                })
                .collect(),
            claimed: (0..capacity.div_ceil(64))
                .map(|_| AtomicU64::new(0))
                .collect(),
            outstanding: AtomicUsize::new(0),
        }
    }

    pub(super) fn len(&self) -> usize {
        self.slots.len()
    }

    /// Number of slots whose record the kernel may still own.
    pub(super) fn outstanding(&self) -> usize {
        self.outstanding.load(Ordering::Acquire)
    }

    /// Give up the table's storage without freeing it, for a kernel that has
    /// not returned its records.
    pub(super) fn leak(&mut self) {
        std::mem::forget(std::mem::take(&mut self.slots));
    }

    /// Claim a free slot, or `None` when all are in use.
    pub(super) fn claim(&self) -> Option<usize> {
        for (word_index, word) in self.claimed.iter().enumerate() {
            let mut current = word.load(Ordering::Relaxed);
            loop {
                let bit = (!current).trailing_zeros() as usize;
                let index = word_index * 64 + bit;
                if bit == 64 || index >= self.slots.len() {
                    break;
                }
                match word.compare_exchange_weak(
                    current,
                    current | (1 << bit),
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return Some(index),
                    Err(actual) => current = actual,
                }
            }
        }
        None
    }

    /// Store the request in a claimed slot and mark it armed.
    pub(super) fn publish(&self, index: usize, socket: usize, info: AfdPollInfo) -> Token {
        let slot = &self.slots[index];
        let generation = ((slot.word.load(Ordering::Relaxed) >> 32) as u32).wrapping_add(1);
        let record = slot.record.get();
        // SAFETY: the slot is claimed and unarmed, so this thread is the only
        // one that can reach its record and the kernel holds no request on it.
        // The writes go through field pointers without forming a reference.
        unsafe {
            (&raw mut (*record).info).write(info);
            (&raw mut (*record).status_block).write(IO_STATUS_BLOCK {
                Anonymous: IO_STATUS_BLOCK_0 {
                    Status: STATUS_PENDING,
                },
                Information: 0,
            });
            (&raw mut (*record).socket).write(socket);
        }
        self.outstanding.fetch_add(1, Ordering::AcqRel);
        slot.word
            .store(u64::from(generation) << 32 | ARMED, Ordering::Release);
        Token::new(index, generation)
    }

    /// Addresses the kernel is given for a slot published by this thread.
    pub(super) fn request(&self, index: usize) -> Request {
        let slot = &self.slots[index];
        let record = slot.record.get();
        // SAFETY: `record` points into the live table; only field addresses are
        // computed, nothing is read or written and no reference is formed.
        unsafe {
            Request {
                info: &raw mut (*record).info,
                status_block: &raw mut (*record).status_block,
                context: std::ptr::from_ref(slot).cast(),
            }
        }
    }

    /// Return a claimed slot that was never published.
    pub(super) fn unclaim(&self, index: usize) {
        self.claimed[index / 64].fetch_and(!(1 << (index % 64)), Ordering::Release);
    }

    /// Release a slot whose request the kernel never accepted.
    pub(super) fn abandon(&self, index: usize) {
        self.release(index);
    }

    /// Token of a slot that is armed and neither completed nor being
    /// cancelled, for shutdown.
    pub(super) fn armed_token(&self, index: usize) -> Option<Token> {
        let word = self.slots[index].word.load(Ordering::Acquire);
        (word & FLAGS == ARMED).then(|| Token::new(index, (word >> 32) as u32))
    }

    /// Mark `token`'s slot as being cancelled. `false` when the token is stale
    /// or the poll already completed, so there is nothing to cancel.
    pub(super) fn begin_cancel(&self, token: Token) -> bool {
        let armed = u64::from(token.generation) << 32 | ARMED;
        self.slots[token.index()]
            .word
            .compare_exchange(
                armed,
                armed | CANCELLING | REQUESTED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    /// Address of a slot's status block, for the canceller that marked it.
    pub(super) fn status_block(&self, index: usize) -> *const IO_STATUS_BLOCK {
        let record = self.slots[index].record.get();
        // SAFETY: `record` points into the live table; only a field address is
        // computed, nothing is read and no reference is formed.
        unsafe { &raw const (*record).status_block }
    }

    /// The canceller left `NtCancelIoFileEx`; release the slot if the packet
    /// was dequeued meanwhile.
    pub(super) fn end_cancel(&self, index: usize) {
        let prior = self.slots[index]
            .word
            .fetch_and(!CANCELLING, Ordering::AcqRel);
        if prior & COMPLETED != 0 {
            self.release(index);
        }
    }

    /// Process one dequeued packet.
    pub(super) fn complete(&self, context: *mut OVERLAPPED) -> Completion {
        let offset = context.addr().wrapping_sub(self.slots.as_ptr().addr());
        let size = size_of::<Slot>();
        if !offset.is_multiple_of(size) || offset / size >= self.slots.len() {
            return Completion::Foreign;
        }
        let index = offset / size;
        let slot = &self.slots[index];
        let prior = slot.word.fetch_or(COMPLETED, Ordering::AcqRel);
        debug_assert!(
            prior & ARMED != 0 && prior & COMPLETED == 0,
            "a packet names an armed, uncompleted slot"
        );
        if prior & REQUESTED != 0 {
            // A canceller still inside `NtCancelIoFileEx` releases the slot in
            // `end_cancel`; otherwise this thread does. Either way the caller
            // asked for the poll to stop, so no readiness is reported.
            if prior & CANCELLING == 0 {
                self.release(index);
            }
            return Completion::Cancelled;
        }
        let record = slot.record.get();
        // SAFETY: the kernel finished with the record (its packet is dequeued)
        // and no canceller holds the slot, so this thread is the only reader.
        // Fields are copied out through pointers without forming a reference.
        let (status, readiness) = unsafe {
            let block = (&raw const (*record).status_block).read();
            let info = (&raw const (*record).info).read();
            let socket = (&raw const (*record).socket).read();
            (block.Anonymous.Status, info.readiness(socket))
        };
        let token = Token::new(index, (prior >> 32) as u32);
        self.release(index);
        if status == STATUS_CANCELLED {
            Completion::Cancelled
        } else {
            Completion::Finished {
                token,
                status,
                readiness,
            }
        }
    }

    fn release(&self, index: usize) {
        let slot = &self.slots[index];
        let generation = slot.word.load(Ordering::Relaxed) >> 32;
        slot.word.store(generation << 32, Ordering::Release);
        self.outstanding.fetch_sub(1, Ordering::AcqRel);
        self.unclaim(index);
    }
}

#[cfg(test)]
mod tests;
