//! Passive connection observations; none changes admission or recovery state.

use super::QuicConnection;
use std::time::Duration;

/// One coherent protocol-engine snapshot, including retransmitted wire traffic.
///
/// UDP bytes include QUIC overhead. Lost packets are the recovery engine's
/// observations, not a count of application failures or kernel socket drops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConnectionMetrics {
    /// UDP datagrams transmitted, including retransmissions.
    pub sent_datagrams: u64,
    /// UDP datagrams received.
    pub received_datagrams: u64,
    /// Bytes transmitted inside UDP datagrams.
    pub sent_bytes: u64,
    /// Bytes received inside UDP datagrams.
    pub received_bytes: u64,
    /// Recovery engine's lost-packet count.
    pub lost_packets: u64,
    /// Congestion events observed by the recovery engine.
    pub congestion_events: u64,
    /// Current congestion window in bytes.
    pub congestion_window_bytes: u64,
    /// Current smoothed path round-trip estimate.
    pub round_trip_time: Duration,
    /// Largest supported UDP payload on the current path.
    pub path_mtu: u16,
}

impl QuicConnection {
    /// Observe recovery and wire counters without exposing the engine's types.
    ///
    /// This snapshot remains available after errors for failure diagnosis.
    /// Counters measure actual protocol traffic; they do not certify application
    /// delivery, which requires complete frame validation and acknowledgement.
    pub fn metrics(&self) -> ConnectionMetrics {
        let stats = self.connection.stats();
        ConnectionMetrics {
            sent_datagrams: stats.udp_tx.datagrams,
            received_datagrams: stats.udp_rx.datagrams,
            sent_bytes: stats.udp_tx.bytes,
            received_bytes: stats.udp_rx.bytes,
            lost_packets: stats.path.lost_packets,
            congestion_events: stats.path.congestion_events,
            congestion_window_bytes: stats.path.cwnd,
            round_trip_time: stats.path.rtt,
            path_mtu: stats.path.current_mtu,
        }
    }
}
