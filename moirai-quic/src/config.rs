use super::{CongestionPolicy, QuicError, rustls};
use moirai_pal::net::UdpBufferConfig;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use std::sync::Arc;
use std::time::Duration;

/// Caller-owned certificate chain and private key; no trust is installed.
pub struct Identity {
    pub(super) certificates: Vec<CertificateDer<'static>>,
    pub(super) key: PrivateKeyDer<'static>,
}

impl Identity {
    /// Supply a leaf-first chain and its signing key.
    ///
    /// TLS configuration validates the key and chain during endpoint creation.
    pub const fn new(
        certificates: Vec<CertificateDer<'static>>,
        key: PrivateKeyDer<'static>,
    ) -> Self {
        Self { certificates, key }
    }
}

/// Validated admission, memory and deadline limits for one QUIC endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuicConfig {
    pub(super) alpn: Arc<[u8]>,
    pub(super) max_frame_bytes: u32,
    pub(super) max_streams: u32,
    pub(super) max_connections: u32,
    pub(super) max_handshake_bytes: u32,
    pub(super) operation_timeout: Duration,
    pub(super) idle_timeout: Duration,
    pub(super) udp_buffers: UdpBufferConfig,
    pub(super) congestion_policy: CongestionPolicy,
}

impl QuicConfig {
    /// Validate an application protocol and positive resource limits.
    ///
    /// Receive and send budgets equal S(B+4) bytes per connection, for S
    /// streams and B message bytes plus the length prefix. Idle timeout must
    /// cover one operation deadline. TLS ALPN identifiers contain 1..=255 bytes.
    ///
    /// # Errors
    /// Rejects zero/overflowing limits, invalid ALPN and non-finite deadlines.
    pub fn new(
        alpn: &[u8],
        max_frame_bytes: u32,
        max_streams: u32,
        max_connections: u32,
        max_handshake_bytes: u32,
        operation_timeout: Duration,
        idle_timeout: Duration,
    ) -> Result<Self, QuicError> {
        if alpn.is_empty() || alpn.len() > usize::from(u8::MAX) {
            return Err(QuicError::InvalidConfig("ALPN must contain 1..=255 bytes"));
        }
        if max_frame_bytes == 0 || max_streams == 0 || max_connections == 0 {
            return Err(QuicError::InvalidConfig(
                "frame, stream and connection limits must be positive",
            ));
        }
        // RFC 9000 section 14.1: an Initial datagram contains at least 1200 bytes.
        if max_handshake_bytes < 1200 {
            return Err(QuicError::InvalidConfig(
                "handshake budget must admit a 1200-byte Initial",
            ));
        }
        if operation_timeout.is_zero()
            || operation_timeout > idle_timeout
            || idle_timeout > moirai_async::timer::MAX_TIMER_DURATION
            || std::time::Instant::now()
                .checked_add(idle_timeout)
                .is_none()
        {
            return Err(QuicError::InvalidConfig(
                "operation/idle deadlines must be finite and ordered",
            ));
        }
        let config = Self {
            alpn: Arc::from(alpn),
            max_frame_bytes,
            max_streams,
            max_connections,
            max_handshake_bytes,
            operation_timeout,
            idle_timeout,
            udp_buffers: UdpBufferConfig::default(),
            congestion_policy: CongestionPolicy::default(),
        };
        config.transport()?;
        Ok(config)
    }

    /// Apply explicit native receive/send queue requests when binding.
    ///
    /// Kernel defaults remain in effect for absent requests. Effective sizes
    /// may include kernel rounding, caps and bookkeeping; inspect the bound
    /// endpoint's `udp_buffer_sizes`. Protocol flow-control budgets and finite
    /// deadlines remain independent of these socket queues.
    ///
    /// # Errors
    /// Returns `InvalidInput` I/O context for a native integer overflow.
    pub fn with_udp_buffers(mut self, buffers: UdpBufferConfig) -> Result<Self, QuicError> {
        buffers.validate().map_err(QuicError::Io)?;
        self.udp_buffers = buffers;
        Ok(self)
    }

    /// Native receive/send requests applied when binding this endpoint.
    pub const fn udp_buffers(&self) -> UdpBufferConfig {
        self.udp_buffers
    }

    /// Select a stock congestion policy without altering transport contracts.
    ///
    /// CUBIC remains the default. Alternatives need matched workload, latency,
    /// fairness and queue-growth evidence; BBR remains upstream-experimental.
    pub const fn with_congestion_policy(mut self, policy: CongestionPolicy) -> Self {
        self.congestion_policy = policy;
        self
    }

    /// Congestion policy selected at endpoint construction.
    pub const fn congestion_policy(&self) -> CongestionPolicy {
        self.congestion_policy
    }

    /// Maximum bytes in one application message, excluding its prefix.
    pub const fn max_frame_bytes(&self) -> u32 {
        self.max_frame_bytes
    }

    /// Maximum concurrently admitted bidirectional streams per connection.
    pub const fn max_streams(&self) -> u32 {
        self.max_streams
    }

    /// Deadline for handshakes, stream admission and complete message I/O.
    pub const fn operation_timeout(&self) -> Duration {
        self.operation_timeout
    }

    /// Application protocol authenticated by TLS ALPN negotiation.
    pub fn alpn(&self) -> &[u8] {
        &self.alpn
    }

    pub(super) fn transport(&self) -> Result<Arc<moirai_quinn::TransportConfig>, QuicError> {
        let stream_window = u64::from(self.max_frame_bytes) + 4;
        let connection_window = stream_window
            .checked_mul(u64::from(self.max_streams))
            .ok_or(QuicError::InvalidConfig("connection window overflows"))?;
        let mut transport = moirai_quinn::TransportConfig::default();
        transport.congestion_controller_factory(self.congestion_policy.factory());
        transport.max_concurrent_bidi_streams(self.max_streams.into());
        transport.max_concurrent_uni_streams(0_u8.into());
        transport.stream_receive_window(
            moirai_quinn::VarInt::from_u64(stream_window)
                .map_err(|_| QuicError::InvalidConfig("stream window exceeds QUIC varint"))?,
        );
        transport.receive_window(
            moirai_quinn::VarInt::from_u64(connection_window)
                .map_err(|_| QuicError::InvalidConfig("connection window exceeds QUIC varint"))?,
        );
        transport.send_window(connection_window);
        transport.crypto_buffer_size(usize::try_from(self.max_handshake_bytes).map_err(|_| {
            QuicError::InvalidConfig("handshake budget exceeds native address space")
        })?);
        transport.datagram_receive_buffer_size(None);
        transport.max_idle_timeout(Some(
            self.idle_timeout
                .try_into()
                .map_err(|_| QuicError::InvalidConfig("idle timeout exceeds QUIC varint"))?,
        ));
        Ok(Arc::new(transport))
    }
}
