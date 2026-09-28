//! Multi-Producer Multi-Consumer channel with bounded capacity.
//!
//! Uses mutex-based implementation for simplicity and correctness,
//! with a lock-free `LockFreeQueue` fast-path for bounded cases.

use std::collections::VecDeque;

mod block;
mod channel;
mod recv;
mod send;
mod unbounded;

pub use self::channel::MpmcChannel;
pub use self::recv::MpmcReceiver;
pub use self::send::MpmcSender;

/// Exponential-backoff spin rounds (`1 << round` spin-loop hints per round,
/// ~1023 total hints) before a blocked send/recv falls back to a condvar wait.
/// Tuned for this channel's mutex+condvar slow path; intentionally local
/// rather than a crate-wide constant because SPSC uses a different budget
/// matched to its yield-based fallback.
const MPMC_BLOCK_SPINS: usize = 10;

/// [`MPMC_BLOCK_SPINS`] as the shared [`moirai_utils::backoff::Spins`] budget,
/// so the ring paths spend their rounds through the same schedule as every other
/// contended site while keeping this channel's condvar-matched number.
pub(super) struct MpmcBlockSpins;

impl moirai_utils::backoff::Spins for MpmcBlockSpins {
    const SPIN_ATTEMPTS: usize = MPMC_BLOCK_SPINS;
}

pub(super) struct MpmcState<T> {
    pub(super) queue: VecDeque<T>,
    pub(super) capacity: Option<usize>,
    pub(super) closed: bool,
    pub(super) sender_count: usize,
    pub(super) receiver_count: usize,
}
