//! `Select` helper and top-level channel constructor functions.

use super::error::Result;
use super::mpmc::{MpmcChannel, MpmcReceiver, MpmcSender};
use super::spsc::{SpscChannel, SpscReceiver, SpscSender};

/// Select over multiple channels following Go's design.
///
/// Allows waiting on multiple channels simultaneously.
pub struct Select;

impl Select {
    /// Try to receive from multiple receivers, returning the first available
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
pub fn mpmc<T>(capacity: usize) -> (MpmcSender<T>, MpmcReceiver<T>) {
    MpmcChannel::channel(Some(capacity))
}

/// Create a new unbounded MPMC channel.
pub fn unbounded<T>() -> (MpmcSender<T>, MpmcReceiver<T>) {
    MpmcChannel::channel(None)
}
