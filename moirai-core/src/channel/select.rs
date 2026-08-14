//! `Select` helper and top-level channel constructor functions.

use super::error::Result;
use super::mpmc::{MpmcChannel, MpmcReceiver, MpmcSender};
use super::spsc::{SpscChannel, SpscReceiver, SpscSender};

/// Non-blocking poll over multiple receive closures.
///
/// This is a single-pass poll, not a Go-style blocking `select`: each closure
/// is tried once in order and the first successful receive wins; when every
/// closure fails the call returns `None` immediately without waiting or
/// registering wakeups.
pub struct Select;

impl Select {
    /// Poll each receiver closure once in order, returning the first
    /// available value with its index, or `None` if none is ready.
    pub fn try_recv<T>(receivers: &mut [&mut dyn FnMut() -> Result<T>]) -> Option<(usize, T)> {
        for (idx, recv) in receivers.iter_mut().enumerate() {
            if let Ok(value) = recv() {
                return Some((idx, value));
            }
        }
        None
    }
}

/// Create a new SPSC channel pair with the given capacity.
pub fn spsc<T>(capacity: usize) -> (SpscSender<T>, SpscReceiver<T>) {
    SpscChannel::channel(capacity)
}

/// Create a new bounded MPMC channel with the given capacity.
///
/// This is the constructor to reach for. Callers with no rate information of
/// their own should pass [`DEFAULT_CHANNEL_CAPACITY`].
///
/// [`DEFAULT_CHANNEL_CAPACITY`]: super::config::DEFAULT_CHANNEL_CAPACITY
pub fn mpmc<T>(capacity: usize) -> (MpmcSender<T>, MpmcReceiver<T>) {
    MpmcChannel::channel(Some(capacity))
}

/// Create a new MPMC channel whose queue grows without limit.
///
/// # Memory
///
/// This channel applies **no backpressure**: a producer that outruns its
/// consumer grows the queue until allocation fails. It is deliberately absent
/// from [`crate::prelude`] and from the crate root so that reaching for it is
/// a decision rather than the path of least resistance — prefer [`mpmc`], and
/// use this only where the total number of sends is bounded by
/// construction (a fixed fan-in, a drained one-shot batch) and blocking a
/// producer would deadlock.
pub fn unbounded<T>() -> (MpmcSender<T>, MpmcReceiver<T>) {
    MpmcChannel::channel(None)
}
