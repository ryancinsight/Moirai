//! Resource and deadline policy for the one-shot HTTP server.

use std::io;
use std::time::Duration;

use crate::request::MAX_HEADER_SLOTS;

/// Default maximum number of accepted connections tracked by the listener.
pub const DEFAULT_MAX_CONNECTIONS: usize = 256;
/// Default maximum size of one request header block.
pub const DEFAULT_MAX_HEADER_BYTES: usize = 16 * 1024;
/// Default maximum number of request or response headers.
pub const DEFAULT_MAX_HEADER_COUNT: usize = 64;
/// Default maximum size of one request body.
pub const DEFAULT_MAX_BODY_BYTES: usize = 1024 * 1024;
/// Default maximum logical size of one response, including its body.
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
/// Default deadline for one request read or response write.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Resource and deadline limits for the one-shot HTTP server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServerConfig {
    /// Maximum number of concurrent accepted connections.
    pub max_connections: usize,
    /// Maximum bytes read while parsing one request head.
    pub max_header_bytes: usize,
    /// Maximum request and response header count.
    pub max_header_count: usize,
    /// Maximum request body size.
    pub max_body_bytes: usize,
    /// Maximum logical response size, including the response body.
    pub max_response_bytes: usize,
    /// Deadline for one complete request read or response write.
    pub request_timeout: Duration,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            max_connections: DEFAULT_MAX_CONNECTIONS,
            max_header_bytes: DEFAULT_MAX_HEADER_BYTES,
            max_header_count: DEFAULT_MAX_HEADER_COUNT,
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
        }
    }
}

impl ServerConfig {
    /// Construct and validate a server configuration.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when a resource limit is zero,
    /// the header count cannot carry the transport framing fields or exceeds
    /// the parser's fixed storage, or the deadline is zero.
    pub fn new(
        max_connections: usize,
        max_header_bytes: usize,
        max_header_count: usize,
        max_body_bytes: usize,
        max_response_bytes: usize,
        request_timeout: Duration,
    ) -> io::Result<Self> {
        let config = Self {
            max_connections,
            max_header_bytes,
            max_header_count,
            max_body_bytes,
            max_response_bytes,
            request_timeout,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validate this configuration before allocating or binding sockets.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] for a zero bound, unsupported
    /// header count, or zero deadline.
    pub fn validate(&self) -> io::Result<()> {
        if self.max_connections == 0
            || self.max_header_bytes == 0
            || self.max_header_count < 2
            || self.max_header_count > MAX_HEADER_SLOTS
            || self.max_body_bytes == 0
            || self.max_response_bytes == 0
            || self.request_timeout.is_zero()
        {
            return Err(invalid_input(
                "HTTP server limits and deadline must be non-zero and bounded",
            ));
        }
        Ok(())
    }
}

fn invalid_input(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_rejects_unbounded_values() {
        assert!(ServerConfig::new(0, 1024, 4, 1024, 1024, Duration::from_secs(1)).is_err());
        assert!(ServerConfig::new(1, 1024, 1, 1024, 1024, Duration::from_secs(1)).is_err());
        assert!(
            ServerConfig::new(
                1,
                1024,
                MAX_HEADER_SLOTS + 1,
                1024,
                1024,
                Duration::from_secs(1)
            )
            .is_err()
        );
        assert!(ServerConfig::new(1, 1024, 4, 1024, 1024, Duration::ZERO).is_err());
    }
}
