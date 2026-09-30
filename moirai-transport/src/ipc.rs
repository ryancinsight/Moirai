//! Same-machine inter-process [`Transport`] over shared memory.
//!
//! Messages are carried as fixed-size [`IpcFrame`]s through a
//! [`moirai_core::ipc::SharedQueue`] (a lock-free SPSC-style ring in a named
//! shared-memory segment), one segment per [`Address::Local`] name. Unlike
//! [`crate::InMemoryTransport`] — which routes within a single process via
//! channels — this crosses the process boundary, so two processes mapping the
//! same segment name exchange bytes directly.
//!
//! Scope and limits:
//! - A single message is at most [`IPC_FRAME_DATA`] bytes; larger payloads are
//!   rejected with [`TransportError::Full`] (fragmentation is intentionally out of
//!   scope — callers chunk).
//! - The first party to touch a segment creates it; others attach. Creation is
//!   exclusive, so concurrent first touches resolve to one creator and the rest
//!   attach.
//! - Each segment is single-producer, single-consumer: at most one handle sends
//!   and one receives (a handle may do both). A second sender or receiver on the
//!   same name gets [`TransportError::Closed`]; use distinct names per direction
//!   for two-way traffic.
//! - This is deliberately not registered in [`crate::TransportManager`]: it would
//!   collide with `InMemoryTransport` on `Address::Local`. Construct and use it
//!   directly when shared-memory IPC is wanted.

use std::collections::HashMap;
use std::sync::Mutex;

use moirai_core::ipc::{IpcError, SendError, SharedQueue};

use crate::{Address, Transport, TransportError, TransportResult};

/// Per-message payload capacity. The backing frame is `4 + IPC_FRAME_DATA`
/// bytes; keeping the total a 4 KiB multiple (and a multiple of the frame
/// alignment) makes [`IpcFrame`] padding-free POD.
pub const IPC_FRAME_DATA: usize = 4096 - core::mem::size_of::<u32>();

/// Number of in-flight frames a segment's ring can hold before `send` reports
/// [`TransportError::Full`].
const IPC_QUEUE_CAPACITY: usize = 64;

/// Yields an attach spends waiting for a concurrent creator to finish
/// initialising a segment before reporting it unreachable.
const ATTACH_ATTEMPTS: u32 = 1024;

/// A fixed-size shared-memory frame: a length-prefixed byte payload.
#[repr(C)]
#[derive(Clone, Copy)]
struct IpcFrame {
    /// Number of valid bytes in `data` (always `<= IPC_FRAME_DATA`).
    len: u32,
    data: [u8; IPC_FRAME_DATA],
}

// SAFETY: `IpcFrame` is `#[repr(C)]` over a `u32` followed by `[u8; IPC_FRAME_DATA]`
// where `IPC_FRAME_DATA` is a multiple of 4, so the struct has size `4 +
// IPC_FRAME_DATA` (a multiple of its 4-byte alignment) with no padding, and every
// bit pattern is a valid value — exactly the `Zeroable`/`Pod` contract. This is
// required because `SharedQueue<T>` writes/reads `T` across the process boundary.
unsafe impl bytemuck::Zeroable for IpcFrame {}
unsafe impl bytemuck::Pod for IpcFrame {}

/// Shared-memory IPC transport. Holds one `SharedQueue` handle per segment name,
/// created lazily on first use.
pub struct IpcTransport {
    // `SharedQueue::{send,recv}` need `&mut self`, and the `Transport` trait is
    // `&self`, so the per-segment handles live behind a `Mutex`. `Mutex<HashMap<
    // _, SharedQueue<IpcFrame>>>` is `Send + Sync` (SharedQueue is `Send`), so
    // `IpcTransport` satisfies the `Transport: Send + Sync` bound.
    segments: Mutex<HashMap<String, SharedQueue<IpcFrame>>>,
}

impl IpcTransport {
    /// Create an IPC transport with no open segments.
    #[must_use]
    pub fn new() -> Self {
        Self {
            segments: Mutex::new(HashMap::new()),
        }
    }

    /// Attach to the segment `name`, creating it when no live segment holds the
    /// name.
    ///
    /// `create` is exclusive, so two parties first-touching one name cannot both
    /// create it: the loser reports `AlreadyExists` and retries `open`, which
    /// succeeds once the winner finishes initialising the header. That window
    /// spans the winner's `ftruncate` and header stores, so the retry is bounded
    /// by `ATTACH_ATTEMPTS` yields rather than a wait.
    fn attach(name: &str) -> TransportResult<SharedQueue<IpcFrame>> {
        for _ in 0..ATTACH_ATTEMPTS {
            if let Ok(queue) = SharedQueue::open(name, IPC_QUEUE_CAPACITY) {
                return Ok(queue);
            }
            match SharedQueue::create(name, IPC_QUEUE_CAPACITY) {
                Ok(queue) => return Ok(queue),
                Err(IpcError::AlreadyExists) => std::thread::yield_now(),
                Err(_) => return Err(TransportError::Closed),
            }
        }
        Err(TransportError::Closed)
    }

    /// Borrow (attaching or creating on first use) the segment for `name`.
    fn segment<'a>(
        segments: &'a mut HashMap<String, SharedQueue<IpcFrame>>,
        name: &str,
    ) -> TransportResult<&'a mut SharedQueue<IpcFrame>> {
        if !segments.contains_key(name) {
            let queue = Self::attach(name)?;
            segments.insert(name.to_string(), queue);
        }
        // Just inserted or already present.
        Ok(segments
            .get_mut(name)
            .expect("invariant: segment present after insert"))
    }
}

impl Default for IpcTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl Transport for IpcTransport {
    fn send(&self, target: &Address, data: Vec<u8>) -> TransportResult<()> {
        let Address::Local(name) = target else {
            return Err(TransportError::Closed);
        };
        // Bound the payload to one frame; oversized messages are backpressure-class
        // failures the caller must chunk around.
        let len = u32::try_from(data.len())
            .ok()
            .filter(|&n| (n as usize) <= IPC_FRAME_DATA)
            .ok_or(TransportError::Full)?;

        let mut frame = IpcFrame {
            len,
            data: [0u8; IPC_FRAME_DATA],
        };
        frame.data[..data.len()].copy_from_slice(&data);

        let mut segments = crate::lock_mutex(&self.segments);
        let queue = Self::segment(&mut segments, name)?;
        // SharedQueue::send returns the value back on a full ring.
        queue.send(frame).map_err(|error| match error {
            SendError::Full(_) => TransportError::Full,
            SendError::Closed(_) | SendError::EndpointInUse(_) => TransportError::Closed,
        })
    }

    fn recv(&self, source: &Address) -> TransportResult<Vec<u8>> {
        let Address::Local(name) = source else {
            return Err(TransportError::Closed);
        };
        let mut segments = crate::lock_mutex(&self.segments);
        let queue = Self::segment(&mut segments, name)?;
        match queue.recv().map_err(|_| TransportError::Closed)? {
            Some(frame) => {
                let len = frame.len as usize;
                // Guard against a corrupt/hostile length from shared memory.
                if len > IPC_FRAME_DATA {
                    return Err(TransportError::Closed);
                }
                Ok(frame.data[..len].to_vec())
            }
            None => Err(TransportError::Empty),
        }
    }

    fn supports(&self, address: &Address) -> bool {
        matches!(address, Address::Local(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ipc_transport_round_trips_through_shared_memory() {
        let ipc = IpcTransport::new();
        let addr = Address::Local("/moirai_ipc_transport_roundtrip".to_string());

        ipc.send(&addr, b"hello ipc".to_vec()).expect("send");
        ipc.send(&addr, b"second".to_vec()).expect("send");

        // FIFO delivery across the shared-memory ring.
        assert_eq!(ipc.recv(&addr).expect("recv"), b"hello ipc");
        assert_eq!(ipc.recv(&addr).expect("recv"), b"second");
        assert_eq!(ipc.recv(&addr), Err(TransportError::Empty));
    }

    #[test]
    fn ipc_transport_attaches_across_separate_handles() {
        // Two independent transports (as two processes would) sharing one segment:
        // the first creates it, the second attaches.
        let sender = IpcTransport::new();
        let receiver = IpcTransport::new();
        let addr = Address::Local("/moirai_ipc_transport_attach".to_string());

        sender.send(&addr, b"cross-handle".to_vec()).expect("send");
        assert_eq!(receiver.recv(&addr).expect("recv"), b"cross-handle");
    }

    #[test]
    fn ipc_transport_rejects_oversized_and_non_local() {
        let ipc = IpcTransport::new();
        let addr = Address::Local("/moirai_ipc_transport_oversized".to_string());

        let too_big = vec![0u8; IPC_FRAME_DATA + 1];
        assert_eq!(ipc.send(&addr, too_big), Err(TransportError::Full));

        let remote = Address::Remote(crate::RemoteAddress {
            host: "127.0.0.1".to_string(),
            port: 1,
            service: "x".to_string(),
        });
        assert_eq!(ipc.send(&remote, vec![1]), Err(TransportError::Closed));
        assert!(!ipc.supports(&remote));
        assert!(ipc.supports(&addr));
    }
}
