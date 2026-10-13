//! Bounded QUIC multiplexing over Moirai scheduling, timers and PAL readiness.
//!
//! One bidirectional stream carries ordered, length-prefixed messages. Separate
//! streams share congestion control while retaining independent receive state.
//! All operations require finite deadlines; cancellation resets the affected
//! stream instead of retaining a partially consumed frame. No early data,
//! unreliable datagrams, application replay or implicit TCP downgrade is used.
//! See ADR 0075 for admission and memory bounds.

#![cfg(any(unix, windows))]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod admission;
mod config;
mod endpoint;
mod error;
mod runtime;
mod stream;

pub use config::{Identity, QuicConfig};
pub use endpoint::{QuicConnection, QuicEndpoint};
pub use error::QuicError;
/// TLS certificate, key and trust-root vocabulary for the QUIC boundary.
pub use quinn::rustls;
pub use stream::QuicStream;

#[cfg(test)]
mod tests;

fn lock_mutex<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
