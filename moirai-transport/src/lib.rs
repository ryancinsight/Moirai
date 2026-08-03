//! Unified transport layer for Moirai concurrency library.
//!
//! This module provides transport abstractions that work across different
//! communication boundaries: threads, processes, and machines. It builds on
//! top of the core channel primitives to provide location-transparent messaging.
//!
//! # Design Principles
//! - Location transparency: same API for local and remote communication
//! - Zero-copy optimization for local transport
//! - Pluggable transport backends (in-memory, IPC, network)
//! - Integration with Moirai scheduler for optimal performance

#![allow(clippy::new_without_default)]
#![allow(clippy::unwrap_or_default)]
#![deny(missing_docs)]

#[cfg(any(unix, windows))]
mod ipc;
mod network;
mod router;
mod transport;

pub mod payload;
pub mod process;
pub mod remote_task;
#[cfg(feature = "scheduler-routes")]
pub mod route;
pub mod safe_channel;

use std::sync::{Mutex, MutexGuard, PoisonError};

/// Crate-wide lock policy: recover from poisoning instead of propagating the
/// panic. Guarded state here (channel maps, subscription lists, connection
/// states) stays structurally valid under a poisoned lock — a writer that
/// panicked mid-critical-section cannot leave a torn invariant in these maps —
/// so continuing with the recovered guard is sound. Matches the pal reactor
/// backends' `lock_mutex` helpers.
pub(crate) fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

// Re-export core channel types for compatibility
/// Shared-memory same-machine IPC transport (Unix/Windows only).
#[cfg(any(unix, windows))]
pub use ipc::IpcTransport;
pub use moirai_core::channel::{
    ChannelError as TransportError, MpmcReceiver as Receiver, MpmcSender as Sender,
};
pub use network::NetworkTransport;
#[cfg(feature = "network")]
pub use network::TcpTransport;
pub(crate) use network::{read_network_frame_from_stream, NETWORK_IO_TIMEOUT};
pub use router::{MessageRouter, RemoteAddress};
// The canonical typed cross-boundary channel: rkyv-style archive serialization
// over a transport (zero-copy borrowed views on receive).
pub use safe_channel::{
    ArchiveSerialize, ArchiveView, ArchivedMessage, ArchivedUniversalReceiver,
    ArchivedUniversalSender,
};
pub use transport::{
    Address, ConnectionManager, ConnectionState, InMemoryTransport, TransportManager,
};

/// Result type for transport operations
pub type TransportResult<T> = Result<T, TransportError>;

/// Transport trait for different communication mechanisms
pub trait Transport: Send + Sync {
    /// Send a message to the specified address
    fn send(&self, target: &Address, data: Vec<u8>) -> TransportResult<()>;

    /// Receive a message from the specified address
    fn recv(&self, source: &Address) -> TransportResult<Vec<u8>>;

    /// Check if the transport supports the given address
    fn supports(&self, address: &Address) -> bool;
}

// A typed cross-boundary channel over a transport is provided by the rkyv-style
// archive channels in `safe_channel` (`ArchivedUniversalSender<T: ArchiveSerialize>`
// / `ArchivedUniversalReceiver<T: ArchiveView>`), re-exported below. The previous
// `UniversalChannel<T: Send>` / `UniversalSender` / `UniversalReceiver` were
// non-functional placeholders (their `send`/`recv` ignored their argument and
// returned `Closed`): a channel generic over an arbitrary `Send` `T` cannot
// serialize the value for transport without a serialization bound, which is
// exactly what the archive traits add. They were removed in favor of the working
// archive channels rather than left as mocks.
