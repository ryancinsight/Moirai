//! Zero-copy hybrid channel for async/sync interop.
//!
//! Uses a lock-free ring buffer with memory barriers to ensure safe
//! zero-copy communication between async and sync contexts.

use crate::communication::RingBuffer;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};
use std::sync::{Arc, Mutex};

mod future;
mod recv;
mod send;
#[cfg(test)]
mod tests;

pub use self::future::RecvFuture;
pub use self::recv::HybridReceiver;
pub use self::send::HybridSender;

/// Zero-sized factory for a zero-copy hybrid channel endpoint pair.
///
/// The sender and receiver own the shared ring and wake registries. The factory
/// carries only its scalar parameter, so constructing endpoints never creates
/// an unused channel-owner allocation or duplicate synchronization state.
pub struct HybridChannel<T>(PhantomData<fn() -> T>);

impl<T: Send> HybridChannel<T> {
    /// Create a new hybrid channel with specified capacity
    pub fn new(capacity: usize) -> (HybridSender<T>, HybridReceiver<T>) {
        let ring = Arc::new(RingBuffer::new(capacity));
        let parker = Arc::new(Mutex::new(Vec::new()));
        let async_wakers = Arc::new(Mutex::new(Vec::new()));
        let parked_count = Arc::new(AtomicUsize::new(0));
        let waker_count = Arc::new(AtomicUsize::new(0));
        let closed = Arc::new(AtomicBool::new(false));
        let next_id = Arc::new(AtomicU64::new(0));

        let sender = HybridSender {
            ring: ring.clone(),
            parker: parker.clone(),
            async_wakers: async_wakers.clone(),
            parked_count: parked_count.clone(),
            waker_count: waker_count.clone(),
            closed: closed.clone(),
            _marker: PhantomData,
        };

        let receiver = HybridReceiver {
            ring,
            parker,
            async_wakers,
            parked_count,
            waker_count,
            closed,
            next_id,
            _marker: PhantomData,
        };

        (sender, receiver)
    }
}

// SAFETY: `HybridSender`/`HybridReceiver` each hold an `Arc<RingBuffer<T>>`,
// which is not auto-`Send` because `RingBuffer` is `!Sync` (a single-producer /
// single-consumer structure). These manual impls assert the SPSC contract: the
// channel hands out exactly one sender and one receiver, each touching a
// disjoint end of the ring (producer_seq vs consumer_seq), so the single
// producer and single consumer may live on different threads without ever
// aliasing the same slot or sequence. Soundness therefore requires that neither
// half is `Clone` — a second sender or receiver would create two producers or
// two consumers racing the same end. The `spsc_send_invariant` guard below turns
// any future `Clone` impl on a half into a compile error so this contract cannot
// be silently broken.
unsafe impl<T: Send> Send for HybridSender<T> {}
unsafe impl<T: Send> Send for HybridReceiver<T> {}

/// Compile-time guard protecting the SPSC `Send` soundness contract above.
///
/// The manual `unsafe impl Send` on each half is sound only while each half is
/// the unique owner of its endpoint. Adding `Clone` to either would allow two
/// producers/consumers on the `!Sync` ring — latent UB with no other compile
/// error. This module statically asserts neither half is `Clone`, and includes a
/// positive/negative control proving the detector itself is wired correctly.
const _: () = {
    use core::marker::PhantomData;

    struct Probe<T>(PhantomData<T>);
    trait CloneProbe {
        const IS_CLONE: bool = false;
    }
    impl<T> CloneProbe for Probe<T> {}
    impl<T: Clone> Probe<T> {
        const IS_CLONE: bool = true;
    }

    struct NeverClone;

    assert!(
        Probe::<u32>::IS_CLONE,
        "Clone detector failed positive control"
    );
    assert!(
        !Probe::<NeverClone>::IS_CLONE,
        "Clone detector failed negative control"
    );

    assert!(
        !Probe::<HybridSender<u8>>::IS_CLONE,
        "HybridSender must not be Clone: a second producer would race the SPSC ring"
    );
    assert!(
        !Probe::<HybridReceiver<u8>>::IS_CLONE,
        "HybridReceiver must not be Clone: a second consumer would race the SPSC ring"
    );
};
