//! Producer and consumer halves as separate contracts.
//!
//! [`Channel`](super::Channel) bundles sending, receiving, and introspection
//! into one trait bound by `Send + Sync`. That suits a channel every thread may
//! touch, but it cannot describe a single-producer/single-consumer queue: an
//! SPSC half is deliberately not `Sync`, because sharing one would let two
//! threads claim the same slot. Requiring `Sync` of every channel-like type is
//! what once let the SPSC channel be shared and raced (ADR-024).
//!
//! Splitting the roles states the smaller thing each half actually promises, so
//! generic code can accept a producer without also demanding the ability to
//! receive from it, or to share it.
//!
//! The impls live beside the types they describe rather than as a blanket impl
//! over [`Channel`](super::Channel). A blanket impl compiles, but coherence then forbids every
//! per-type impl — a downstream crate could implement `Channel<TheirType>` for
//! `SpscSender<TheirType>`, so the compiler must assume the two overlap. Since
//! the SPSC halves are precisely the types that cannot be `Channel`, and they
//! are the reason these traits exist, the per-type impls are the ones worth
//! keeping. Shareable channels implement the roles explicitly alongside their
//! `Channel` impl.

use super::error::Result;

/// The sending half of a channel.
///
/// Implementors are not required to be `Sync`; a single-producer half is a
/// legitimate producer precisely because only one thread ever holds it.
pub trait Producer<T> {
    /// Send a value, blocking until there is room.
    fn send(&self, value: T) -> Result<()>;

    /// Send a value, or report [`ChannelError::Full`](super::ChannelError::Full)
    /// rather than waiting.
    fn try_send(&self, value: T) -> Result<()>;

    /// Whether a further send would have to wait.
    fn is_full(&self) -> bool;

    /// Bounded capacity, or `None` when the channel is unbounded.
    fn capacity(&self) -> Option<usize>;
}

/// The receiving half of a channel.
///
/// As with [`Producer`], `Sync` is not required, so a single-consumer half
/// qualifies.
pub trait Consumer<T> {
    /// Receive a value, blocking until one arrives.
    fn recv(&self) -> Result<T>;

    /// Receive a value, or report
    /// [`ChannelError::Empty`](super::ChannelError::Empty) rather than waiting.
    fn try_recv(&self) -> Result<T>;

    /// Whether a receive would have to wait.
    fn is_empty(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::{Consumer, Producer};
    use crate::channel::{mpmc, spsc, MpmcChannel};

    /// The point of the split: a function can demand only the sending role, and
    /// an SPSC half — which is deliberately not `Sync` — satisfies it. Before the
    /// roles existed this had to be written against `Channel<T>`, whose
    /// `Send + Sync` bound no single-producer type can honestly meet.
    fn drain_into<P: Producer<u64>>(producer: &P, values: &[u64]) -> usize {
        values
            .iter()
            .filter(|v| producer.try_send(**v).is_ok())
            .count()
    }

    fn take_all<C: Consumer<u64>>(consumer: &C, limit: usize) -> Vec<u64> {
        (0..limit)
            .filter_map(|_| consumer.try_recv().ok())
            .collect()
    }

    #[test]
    fn spsc_halves_satisfy_the_roles() {
        let (tx, rx) = spsc::<u64>(8);

        assert_eq!(drain_into(&tx, &[1, 2, 3]), 3);
        assert_eq!(take_all(&rx, 8), vec![1, 2, 3]);
    }

    #[test]
    fn shareable_channels_and_their_halves_satisfy_the_roles() {
        // A shareable channel is usable whole, wherever a role is asked for.
        let channel = MpmcChannel::<u64>::new(Some(8));
        assert_eq!(drain_into(&channel, &[7, 8]), 2);
        assert_eq!(take_all(&channel, 8), vec![7, 8]);

        // ...and so are its halves, which is the form callers actually hold.
        let (tx, rx) = mpmc::<u64>(8);
        assert_eq!(drain_into(&tx, &[9, 10]), 2);
        assert_eq!(take_all(&rx, 8), vec![9, 10]);
    }

    #[test]
    fn producer_reports_capacity_and_fullness() {
        let (tx, rx) = spsc::<u64>(2);

        assert_eq!(Producer::capacity(&tx), Some(2));
        assert!(!Producer::is_full(&tx));

        assert_eq!(drain_into(&tx, &[1, 2]), 2);
        assert!(Producer::is_full(&tx));
        assert!(!Consumer::is_empty(&rx));
    }
}
