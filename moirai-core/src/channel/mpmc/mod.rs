//! Multi-Producer Multi-Consumer channel with bounded capacity.
//!
//! Uses mutex-based implementation for simplicity and correctness,
//! with a lock-free `BoundedMpmcQueue` fast-path for bounded cases.

use std::collections::VecDeque;

mod channel;
mod queue;
mod recv;
mod send;

pub use self::channel::MpmcChannel;
pub use self::recv::MpmcReceiver;
pub use self::send::MpmcSender;

/// Exponential-backoff spin rounds (`1 << round` spin-loop hints per round,
/// ~1023 total hints) before a blocked send/recv falls back to a condvar wait.
/// Tuned for this channel's mutex+condvar slow path; intentionally local
/// rather than a crate-wide constant because SPSC uses a different budget
/// matched to its yield-based fallback.
const MPMC_BLOCK_SPINS: usize = 10;

pub(super) struct MpmcState<T> {
    pub(super) queue: VecDeque<T>,
    pub(super) capacity: Option<usize>,
    pub(super) closed: bool,
    pub(super) sender_count: usize,
    pub(super) receiver_count: usize,
}
