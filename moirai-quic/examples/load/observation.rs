//! Low-frequency recovery and value-verified progress observations.

use moirai_quic::QuicConnection;
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[derive(Default)]
pub(super) struct Progress(AtomicU64);

impl Progress {
    pub(super) fn completed(&self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) async fn observe(&self, connection: &QuicConnection) -> Infallible {
        loop {
            moirai_async::timer::sleep(Duration::from_secs(1)).await;
            let metrics = connection.metrics();
            tracing::info!(
                completed_messages = self.0.load(Ordering::Relaxed),
                sent_bytes = metrics.sent_bytes,
                received_bytes = metrics.received_bytes,
                lost_packets = metrics.lost_packets,
                congestion_events = metrics.congestion_events,
                congestion_window_bytes = metrics.congestion_window_bytes,
                round_trip_ns = metrics.round_trip_time.as_nanos(),
                path_mtu = metrics.path_mtu,
                "verified progress and recovery snapshot"
            );
        }
    }
}
